/* amp_solve_zc.c — zcache 口径乘性放大器逐层闭式解算(C, 2026-08-25 Python→C 迁移)。
 *
 * 【这是什么】zlever/amp_solve.py(v6.1, 129 行)的【逐式转录】。同一个 zcache 输入,
 * 同一套网格, 同一份记录字节。转录纪律 = 禁改进: 每一处精度阶梯(哪里 f32 哪里 f64)、
 * 每一次求和顺序、每一个随机种子都照抄 .py, 哪怕"顺手改好"更漂亮。
 *   本仓刚因 Python↔C 静默分歧付出一周代价(见 feedback_c_not_python_shared_impl):
 *   两套实现只要有一处不一致, 解算器算出来的 U 到引擎里就对不上, 而且不报错。
 *
 * 【和 calib/amp_solve.c 什么关系】两码事, 别混:
 *   - calib/amp_solve.c   = 换代工具(2026-08-23)。输入 capnpy 目录(ffn_in/obase_v3/routed
 *     三个 .npy), 四损失对齐口径, SVD 整支砍除(实测 PCA 从未当选), Cholesky 求解。
 *   - calib/amp_solve_zc.c(本文件) = 上一代 zcache 口径。输入 zlayer 产的 zcache_LXX.npz,
 *     比值域目标, PCA/rand × pca2/rnd2 四组合网格, np.linalg.solve(LU)。
 *   两者产的都是 zrec "zl.AMPD"(type9) 记录, 引擎回放语义相同, 但目标函数与选基口径不同,
 *   数值不可互相对拍。要复现旧战役(amp86/dspark86/zside)用本文件, 新战役用 amp_solve.c。
 *
 * 【引擎语义(两代一致)】ds4_zchain.c / ds4_cuda.cu 的 type9 zl.AMPD:
 *     routed'_t = routed_t ⊙ (1 + Σ_c U[:,c] · tanh(V_c·φ(x_t)/s) · tanh(A_c·φ(x_t)/s))
 *     φ(x) = [x, x⊙x/rms(x), relu(x)]   (din==3d 时引擎在线展开)
 *
 * 【用法】与 .py 完全一致(第 4 参可省):
 *     amp_solve_zc <anchor> <zcache_LXX.npz> <out_amprec.bin> [NFIT]
 *   NFIT 缺省或 ≤0 → S*8//10(协议 80/20)。层号 L 从 zcache 文件名 "_L" 后两位取。
 *   旧战役脚本传的是 1638(只在 S=2048 时才等于 80%; S=8192 时是显式短拟合段, 照抄不纠)。
 *   额外 C 侧入口: `amp_solve_zc --selftest` 自检 RNG/特征分解/f16 舍入, 不读任何输入。
 *
 * 【编译】默认路无外部依赖能跑, 但【产线请务必带 -DDQ_BLAS】:
 *     gcc -O3 -o amp_solve_zc amp_solve_zc.c -lm -lpthread                      (能跑, 但慢)
 *     gcc -O3 -march=native -ffp-contract=off -DDQ_BLAS -DACCELERATE_NEW_LAPACK \
 *         -o amp_solve_zc amp_solve_zc.c -framework Accelerate -lm -lpthread     (macOS)
 *     Linux 照 migrate/build_ctools.sh 里 zlayer 那条(scipy_openblas)写。
 *   实测真尺寸 SVD 一段(13106×12288 取前 2048): 无 BLAS 430.7s vs Accelerate 28.0s ——
 *   差 15 倍, 而两边精度一模一样(Ritz 残差都是 4.65e-8)。手写 GEMM 与二级前代循环撑不起
 *   这个规模, 别让它落进 build_ctools.sh 的默认 `*)` 分支。
 *   ⚠ -ffp-contract=off 不是可选项而是正确性要求: 开着 FMA 合并会把 `a*b+c` 融成一条指令,
 *     少一次舍入 —— f32 精度阶梯上这会真的改数。文件里放了 #pragma STDC FP_CONTRACT OFF
 *     (clang 认), gcc 只认命令行开关; 不带 -march 的基线 x86-64 无 FMA 所以默认构建安全。
 *
 * 【金标口径 —— 能对到什么程度, 不能对到什么程度】
 *  ① 可逐位: 三条随机基(dither seed 1 / V₀=rand seed 7 / 门=rnd2 seed 11)是 numpy legacy
 *     MT19937 + legacy gauss 的逐位复刻。赢家是 rand/rnd2 时(实测 215 条记录里 V₀=rand
 *     100% 胜、门=rnd2 98.6% 胜), 载荷里的 A 块与 V 块与 .py 逐字节相同。
 *  ② 不可逐位: U 块与 scale。链路上有三处 BLAS/LAPACK 依赖的求和顺序 —— 矩阵乘(numpy 走
 *     dgemm)、np.linalg.solve(LAPACK dgesv)、SVD(gesdd)。C 侧自带 GEMM/LU/特征分解, 数学
 *     同式但求和顺序不同, f64 尾位差 ~1e-13 相对。U 落 f16(11 位尾数)后绝大多数条目相同,
 *     但边界值会翻末位 ⇒ 整文件 md5 不作判据。
 *  ③ SVD 是唯一的【算法替换】: numpy 的 gesdd → 本文件用【随机子空间迭代 + Rayleigh-Ritz】
 *     (详见下面 "np.linalg.svd 位" 那一大段的成因与实测)。前导方向与 numpy 一致到 10 位
 *     有效数字, 截断边缘那 ~1% 方向的 σ 低估 ~1%; 符号与简并子空间内的旋转由算法决定
 *     ⇒ 与 numpy 不同。对结果无影响(tanh 是奇函数, 基列翻号被 U 对应行翻号抵消, held
 *     分数不变), 但若 PCA/pca2 当选, 载荷字节必然不同。
 *  ④ 因此判据 = 【数值容差 + 终判在合并/回放侧】: 同一 zcache 下,
 *       - stdout 那一行的 held% 与 .py 差应 ≤0.05pp, 选出的 (V₀名, 门名, λ, k) 应完全相同;
 *       - 载荷长度必须逐位相同(k 相同即定长);
 *       - rec_fidelity 复评 held/fit 双侧应落在同一档(≤0.1pp)。
 *     赢家是 rand+rnd2 时, A/V 两块可以直接 cmp 逐字节验(占载荷 2/3)。
 *
 * 【已验证(2026-08-25, Mac 合成夹具, 与 .py 同机对跑)】
 *   - 出载荷路(S=1300 D=700 DIN=2100 NFIT=1024, 赢家 PCA×pca2): 赢家/λ=30.0/k=768/
 *     held 1.17%/文件长度 7526532B 全同; scale 逐位同; V 块【每一列都只差一个全局符号】
 *     (371 同号 397 反号, 无一列"都不是"), A 块 765/768 同样只差符号、3 列不是 ——
 *     那 3 列落在 pca2 门的截断边缘(Vt 第 1024..2048 行), 正是子空间法精度最低的地方;
 *     独立回放复评 held: .py 1.167461% vs C 1.167442%(差 1.9e-5 pp)。
 *   - 层闸路(held 0.04% ≤0.5%): 116B 记录逐字节相同(md5 69d6ccff…)。该夹具同时走了
 *     无 xcap(读锚)+ 无 yqe(np.add.at 重建 yq)两条退路。
 *   - 随机基逐位: 生产尺寸 DIN=12288×KMAX=1024 的 seed 7 与 seed 11 两条基, 落 f16 后与
 *     numpy 12582912/12582912 元素【全部逐位相同】。
 *   - SVD 真尺寸(13106×12288 取前 2048, `--selftest 13106 12288 2048 12288 1.0`):
 *     28s(Accelerate) / 见文末成本, 正交性 4.7e-15, Ritz 残差 4.6e-8, 无重随机化;
 *     同矩阵对 numpy: σ[0..7] 十位有效数字全同, 截断边缘 σ 低估约 1%。
 *     秩 ≤ nv 的构造下投影残差 ‖Y−(YVᵀ)V‖/‖Y‖ = 4.3e-15(< 1e-10)。
 *
 * 【★macOS 上别拿本机 numpy 当 RNG 金标★】arm64 的 numpy 2.5.1 把 legacy_gauss 里
 *   `r2 = x1*x1 + x2*x2` 编译成了一条 FMA, r2 差 1 ULP。实测 200 万次抽样有 13.79% 的
 *   高斯值与非 FMA 写法差 1 ULP —— 所以在 Mac 上 randn 的 f64 前 8 个值会有一个对不上
 *   (第 3 个), 这【不是转录错】: spark(Linux x86-64)上 numpy 不合并, calib/zlayer.c 同款
 *   RNG 段已在那边过了逐位金标。而且这 1 ULP 到不了产物: 同样 200 万样本, ×1/√DIN 落 f16
 *   之后 0 个不同。要复核 RNG 请在 spark 上做, 或直接比 f16 基(见上)。
 *
 * 【小端前提】.py 全用 struct '<' 与 numpy 小端 dtype; 本文件同样假定小端主机, 大端不支持。
 *
 * 【成本提醒】.py 走 cupy GPU 路(真数据 L20 整层 258s)。本文件是纯 CPU, 按"spark 重计算
 *   必须 GPU 化"铁律, 若要上产线批量重解仍需补 CUDA 路(与 zlayer.c 一期同款欠账)。
 *   带 -DDQ_BLAS 时 SVD 一段 28s, 网格才是大头(28 组 × 9 个 k, 见 py:113 的 k 表)。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/mman.h>
#ifdef DQ_BLAS
#  ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

#pragma STDC FP_CONTRACT OFF

/* ---------------- 基础设施 ---------------- */

static void die(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "Error: ");
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    exit(1);
}
static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("malloc %zu 失败", n);
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) die("calloc %zu×%zu 失败", n, s);
    return p;
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static int n_threads(void) {
    static int n = 0;
    if (!n) { long c = sysconf(_SC_NPROCESSORS_ONLN); n = (int)(c > 0 ? c : 1); if (n > 32) n = 32; }
    return n;
}

/* 并行 for(与 calib/zlayer.c 同构): 把 [0,n) 切成 n_threads 段, 阻塞到全部完成。 */
typedef void (*pf_fn)(void *ctx, int i0, int i1);
typedef struct { pf_fn fn; void *ctx; int i0, i1; } pf_arg;
static void *pf_tramp(void *a) { pf_arg *p = (pf_arg *)a; p->fn(p->ctx, p->i0, p->i1); return NULL; }
static void parallel_for(int n, pf_fn fn, void *ctx) {
    int th = n_threads();
    if (th > n) th = n > 0 ? n : 1;
    pthread_t tid[32]; pf_arg pa[32];
    int per = (n + th - 1) / th, nt = 0;
    for (int t = 0; t < th; t++) {
        int i0 = t * per, i1 = i0 + per > n ? n : i0 + per;
        if (i0 >= i1) break;
        pa[nt] = (pf_arg){fn, ctx, i0, i1};
        if (pthread_create(&tid[nt], NULL, pf_tramp, &pa[nt])) { pa[nt].fn(ctx, i0, i1); continue; }
        nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* ---------------- 半精度(与 calib/zlayer.c 同源) ---------------- */

static float f16_to_f32(uint16_t h) {
    int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = (float)ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = (float)ldexp((double)(m + 1024), e - 25);
    return s ? -v : v;
}
/* f64 → f16, round-to-nearest-even(= numpy .astype(np.float16))。
 * 直接从 double 舍一次: .py 里 U/V/A 都是 f64, 走 f32 中转会双重舍入。 */
static uint16_t f64_to_f16(double d) {
    uint64_t x; memcpy(&x, &d, 8);
    uint32_t sign = (uint32_t)((x >> 48) & 0x8000u);
    int e64 = (int)((x >> 52) & 0x7FF);
    uint64_t man = x & 0xFFFFFFFFFFFFFULL;
    if (e64 == 0x7FF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (e64 == 0) return (uint16_t)sign;
    int e = e64 - 1023 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        int sh = 43 - e;
        uint64_t full = man | (1ULL << 52);
        uint64_t m = full >> sh, rem = full & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
        if (rem > half || (rem == half && (m & 1))) m++;
        return (uint16_t)(sign | (uint32_t)m);
    }
    uint32_t h = ((uint32_t)e << 10) | (uint32_t)(man >> 42);
    uint64_t rem = man & ((1ULL << 42) - 1), half = 1ULL << 41;
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

/* ---------------- numpy legacy MT19937 + legacy gauss(逐位复刻, 源: calib/zlayer.c) ----
 * np.random.RandomState(seed).randn(...) 的逐位复刻。三处必须一模一样:
 *   ① 播种 = numpy 的 mt19937_seed(等价标准 init_genrand), 不是 init_by_array;
 *   ② 双精度 = (next>>5)*2^26 + (next>>6) 再除 2^53(两次抽 32 位);
 *   ③ 高斯 = Marsaglia polar, 【先返回 f*x2, 把 f*x1 存进缓存】(numpy legacy_gauss 的顺序)。
 * 任何一处写反, 三条随机基就整体换了一套 —— 而且不报错, 只是解出来的 U 悄悄不一样。
 * 自检: ./amp_solve_zc --selftest 打印前 8 个值, 应等于 np.random.RandomState(1).randn(8)。 */
typedef struct { uint32_t key[624]; int pos; int has_gauss; double gauss; } mt_t;

static void mt_seed(mt_t *s, uint32_t seed) {
    for (int pos = 0; pos < 624; pos++) {
        s->key[pos] = seed;
        seed = (uint32_t)(1812433253u * (seed ^ (seed >> 30)) + (uint32_t)pos + 1u);
    }
    s->pos = 624; s->has_gauss = 0; s->gauss = 0.0;
}
static void mt_gen(mt_t *s) {
    const uint32_t UP = 0x80000000u, LO = 0x7fffffffu, MA = 0x9908b0dfu;
    uint32_t *mt = s->key, y;
    int i = 0;
    for (; i < 624 - 397; i++) {
        y = (mt[i] & UP) | (mt[i + 1] & LO);
        mt[i] = mt[i + 397] ^ (y >> 1) ^ ((y & 1) ? MA : 0u);
    }
    for (; i < 623; i++) {
        y = (mt[i] & UP) | (mt[i + 1] & LO);
        mt[i] = mt[i + (397 - 624)] ^ (y >> 1) ^ ((y & 1) ? MA : 0u);
    }
    y = (mt[623] & UP) | (mt[0] & LO);
    mt[623] = mt[396] ^ (y >> 1) ^ ((y & 1) ? MA : 0u);
    s->pos = 0;
}
static uint32_t mt_next(mt_t *s) {
    if (s->pos >= 624) mt_gen(s);
    uint32_t y = s->key[s->pos++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}
static double mt_double(mt_t *s) {
    uint32_t a = mt_next(s) >> 5, b = mt_next(s) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
static double mt_gauss(mt_t *s) {
    if (s->has_gauss) { s->has_gauss = 0; return s->gauss; }
    double x1, x2, r2, f;
    do {
        x1 = 2.0 * mt_double(s) - 1.0;
        x2 = 2.0 * mt_double(s) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    f = sqrt(-2.0 * log(r2) / r2);
    s->gauss = f * x1; s->has_gauss = 1;
    return f * x2;
}

/* ---------------- 最小 npz(ZIP_STORED / ZIP64)读取 ----------------
 * 源: calib/rec_fidelity.c 的解析器(那边一律升 f64), 这里保留原 dtype 零拷贝 —— 因为本工具
 * 的精度阶梯要求 dH/yq/xcap 留在 f32(升 f64 会把 .py 的 f32 中间舍入抹掉 = 静默改数),
 * 而且 pYQ 在 S=8192 时有 1GB 量级, 升 f64 白吃一倍内存。改动只在"转成什么类型", 扫描
 * 逻辑(局部文件头 PK\3\4 + ZIP64 extra id=0x0001 真尺寸 + npy 头)与 rec_fidelity 逐字同。 */
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

typedef struct { const uint8_t *data; char dt[8]; long long d0, d1, nd; } npy_t;

static int npz_find(const uint8_t *buf, long long sz, const char *name, npy_t *out) {
    char want[128]; snprintf(want, sizeof want, "%s.npy", name);
    long long off = 0;
    while (off + 30 <= sz) {
        if (rd32(buf + off) != 0x04034b50u) break;
        uint16_t meth = rd16(buf + off + 8), nlen = rd16(buf + off + 26), xlen = rd16(buf + off + 28);
        long long csz = rd32(buf + off + 18);
        const char *nm = (const char *)buf + off + 30;
        long long pay = off + 30 + nlen + xlen;
        if (csz == 0xFFFFFFFFLL) {   /* numpy 用 force_zip64 写, 真尺寸在 extra id=0x0001 */
            const uint8_t *x = buf + off + 30 + nlen, *xe = x + xlen;
            while (x + 4 <= xe) {
                uint16_t id = rd16(x), l = rd16(x + 2);
                if (id == 1 && l >= 16) { memcpy(&csz, x + 12, 8); break; }
                x += 4 + l;
            }
        }
        if ((long long)strlen(want) == nlen && !memcmp(nm, want, (size_t)nlen)) {
            if (meth != 0) die("npz %s: 压缩条目不支持(np.savez 应为 STORED)", name);
            const uint8_t *p = buf + pay;
            if (memcmp(p, "\x93NUMPY", 6)) die("npz %s: 非 npy 头", name);
            int maj = p[6];
            uint32_t hl = maj >= 2 ? rd32(p + 8) : rd16(p + 8);
            const char *hdr = (const char *)p + (maj >= 2 ? 12 : 10);
            memset(out, 0, sizeof *out);
            out->data = p + (maj >= 2 ? 12 : 10) + hl;
            { const char *q = strstr(hdr, "'descr':"); if (!q) die("npz %s: 无 descr", name);
              q = strchr(q + 8, '\''); const char *e = strchr(q + 1, '\'');
              size_t l = (size_t)(e - q - 1); if (l > 7) l = 7; memcpy(out->dt, q + 1, l); }
            { const char *q = strstr(hdr, "'shape':"); if (!q) die("npz %s: 无 shape", name);
              q = strchr(q, '(') + 1; out->d0 = 1; out->d1 = 1; out->nd = 0;
              while (*q && *q != ')') {
                  while (*q == ' ' || *q == ',') q++;
                  if (*q == ')') break;
                  long long v = strtoll(q, (char **)&q, 10);
                  if (out->nd == 0) out->d0 = v; else if (out->nd == 1) out->d1 = v;
                  out->nd++;
              } }
            if (strstr(hdr, "'fortran_order': True")) die("npz %s: fortran_order 不支持", name);
            return 0;
        }
        off = pay + csz;
    }
    return -1;
}
/* npz 里的数据可能非对齐, 一律 memcpy 逐元素搬。 */
static void npz_to_f32(const npy_t *a, float *out, long long n) {
    if (!strcmp(a->dt, "<f4")) { memcpy(out, a->data, (size_t)n * 4); return; }
    if (!strcmp(a->dt, "<f8")) { for (long long i = 0; i < n; i++) { double v; memcpy(&v, a->data + i * 8, 8); out[i] = (float)v; } return; }
    if (!strcmp(a->dt, "<f2")) { for (long long i = 0; i < n; i++) { uint16_t v; memcpy(&v, a->data + i * 2, 2); out[i] = f16_to_f32(v); } return; }
    die("npz: dtype %s 不能转 f32", a->dt);
}
static void npz_to_i64(const npy_t *a, long long *out, long long n) {
    if (!strcmp(a->dt, "<i8")) { for (long long i = 0; i < n; i++) { int64_t v; memcpy(&v, a->data + i * 8, 8); out[i] = v; } return; }
    if (!strcmp(a->dt, "<i4")) { for (long long i = 0; i < n; i++) { int32_t v; memcpy(&v, a->data + i * 4, 4); out[i] = v; } return; }
    die("npz: dtype %s 不能转 i64", a->dt);
}

/* ---------------- numpy 求和语义(照抄, 不是"更好的求和") ----------------
 * numpy 的归约分两套, 混了就对不上:
 *   ① 归约轴连续(整块 .sum() / .mean(axis=1)) → pairwise 求和, 块 128, 8 个累加器;
 *   ② 归约轴不连续(.mean(axis=0)) → 逐行朴素累加(out[j] += a[i][j])。
 * 下面 pw_* 是 numpy pairwise_sum_@TYPE@ 的逐式复刻; ax0_* 是第二套。
 * 累加类型也照抄: f32 数组的 mean 用 f32 累加器(numpy 默认), 不偷偷升 f64。 */
#define PW_BLOCK 128

/* Σ a[i]² —— .py 里 (M**2).sum()/.mean() 先物化平方数组再 pairwise 求和, 这里现算现加,
 * 值与求和顺序都一样, 只是省掉临时大数组。 */
static double pw_sq_f64(const double *a, long long n) {
    if (n < 8) { double r = 0.0; for (long long i = 0; i < n; i++) r += a[i] * a[i]; return r; }
    if (n <= PW_BLOCK) {
        double r[8]; for (int j = 0; j < 8; j++) r[j] = a[j] * a[j];
        long long i;
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j] * a[i + j];
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += a[i] * a[i];
        return s;
    }
    long long n2 = n / 2; n2 -= n2 % 8;
    return pw_sq_f64(a, n2) + pw_sq_f64(a + n2, n - n2);
}
static float pw_sq_f32(const float *a, long long n) {
    if (n < 8) { float r = 0.0f; for (long long i = 0; i < n; i++) r += a[i] * a[i]; return r; }
    if (n <= PW_BLOCK) {
        float r[8]; for (int j = 0; j < 8; j++) r[j] = a[j] * a[j];
        long long i;
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j] * a[i + j];
        float s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += a[i] * a[i];
        return s;
    }
    long long n2 = n / 2; n2 -= n2 % 8;
    return pw_sq_f32(a, n2) + pw_sq_f32(a + n2, n - n2);
}
/* e0 用: Σ ((f64)(yfp[i]-yq[i]))² 。注意减法在 f32(yfp 本身就是 f32 的 yq+dH),
 * 升 f64 在平方之前 —— 顺序反了会少一次 f32 舍入。 */
static double pw_sqdiff(const float *a, const float *b, long long n) {
#define SQD(i) ({ double _v = (double)(a[i] - b[i]); _v * _v; })
    if (n < 8) { double r = 0.0; for (long long i = 0; i < n; i++) r += SQD(i); return r; }
    if (n <= PW_BLOCK) {
        double r[8]; for (int j = 0; j < 8; j++) r[j] = SQD(j);
        long long i;
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += SQD(i + j);
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += SQD(i);
        return s;
    }
    long long n2 = n / 2; n2 -= n2 % 8;
    return pw_sqdiff(a, b, n2) + pw_sqdiff(a + n2, b + n2, n - n2);
#undef SQD
}
/* rec 用: Σ ((f64)(yfp[i] - yq[i]*(1+f32(g[i]))))² 。g 是 f64 矩阵乘结果, .py 先
 * .astype(float32) 再进 yh —— 这次 f32 截断必须留着。 */
static double pw_sqres(const float *yfp, const float *yq, const double *g, long long n) {
#define SQR(i) ({ float _t = 1.0f + (float)g[i]; float _d = yfp[i] - yq[i] * _t; double _v = (double)_d; _v * _v; })
    if (n < 8) { double r = 0.0; for (long long i = 0; i < n; i++) r += SQR(i); return r; }
    if (n <= PW_BLOCK) {
        double r[8]; for (int j = 0; j < 8; j++) r[j] = SQR(j);
        long long i;
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += SQR(i + j);
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += SQR(i);
        return s;
    }
    long long n2 = n / 2; n2 -= n2 % 8;
    return pw_sqres(yfp, yq, g, n2) + pw_sqres(yfp + n2, yq + n2, g + n2, n - n2);
#undef SQR
}

/* ---------------- 矩阵乘(行主序; BLAS 可选) ----------------
 * 无 BLAS 路: i 分块 × k 分块, k 升序累加。与朴素三重循环逐位同, 与 BLAS 的分块/向量化
 * 不同 —— 两种构建之间不可逐位对拍(与 calib/zlayer.c 歧义清单 #9 同款既判)。 */
typedef struct { const double *A, *B; double *C; int M, N, K, lda, ldb, ldc, MB, KB; } mm_ctx;

static void nn_worker(void *vc, int i0, int i1) {
    mm_ctx *c = (mm_ctx *)vc;
    for (int ib = i0; ib < i1; ib += c->MB) {
        int ie = ib + c->MB > i1 ? i1 : ib + c->MB;
        for (int kb = 0; kb < c->K; kb += c->KB) {
            int ke = kb + c->KB > c->K ? c->K : kb + c->KB;
            for (int i = ib; i < ie; i++) {
                double *cr = c->C + (size_t)i * c->ldc;
                for (int k = kb; k < ke; k++) {
                    double a = c->A[(size_t)i * c->lda + k];
                    const double *br = c->B + (size_t)k * c->ldb;
                    for (int j = 0; j < c->N; j++) cr[j] += a * br[j];
                }
            }
        }
    }
}
static void nt_worker(void *vc, int i0, int i1) {
    mm_ctx *c = (mm_ctx *)vc;
    for (int ib = i0; ib < i1; ib += c->MB) {
        int ie = ib + c->MB > i1 ? i1 : ib + c->MB;
        for (int kb = 0; kb < c->K; kb += c->KB) {
            int ke = kb + c->KB > c->K ? c->K : kb + c->KB;
            for (int i = ib; i < ie; i++) {
                const double *ar = c->A + (size_t)i * c->lda;
                double *cr = c->C + (size_t)i * c->ldc;
                for (int j = 0; j < c->N; j++) {
                    const double *br = c->B + (size_t)j * c->ldb;
                    double s = 0.0;
                    for (int k = kb; k < ke; k++) s += ar[k] * br[k];
                    cr[j] += s;
                }
            }
        }
    }
}
static void mm_zero(double *C, int M, int N, int ldc) {
    for (int i = 0; i < M; i++) memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(double));
}
/* C(M×N) = A(M×K)·B(K×N) */
static void mm_nn(int M, int N, int K, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
    if (M <= 0 || N <= 0) return;
    mm_zero(C, M, N, ldc);
    if (K <= 0) return;
#ifdef DQ_BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
#else
    int MB = 262144 / (N > 0 ? N : 1); if (MB < 8) MB = 8; if (MB > 256) MB = 256;
    mm_ctx c = {A, B, C, M, N, K, lda, ldb, ldc, MB, 256};
    parallel_for(M, nn_worker, &c);
#endif
}
/* C(M×N) = A(M×K)·B(N×K)ᵀ */
static void mm_nt(int M, int N, int K, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
    if (M <= 0 || N <= 0) return;
    mm_zero(C, M, N, ldc);
    if (K <= 0) return;
#ifdef DQ_BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
#else
    int MB = 262144 / (N > 0 ? N : 1); if (MB < 8) MB = 8; if (MB > 256) MB = 256;
    mm_ctx c = {A, B, C, M, N, K, lda, ldb, ldc, MB, 512};
    parallel_for(M, nt_worker, &c);
#endif
}

/* ---------------- np.linalg.solve 位: 部分主元 LU ----------------
 * numpy 走 LAPACK dgesv(部分主元 LU), 不是 Cholesky —— 虽然 G = ZᵀZ + 脊 是对称正定,
 * 换 Cholesky 数学上更稳但求和顺序不同, 属"顺手改好"。这里照 dgesv 的算法。
 * 右端项按【转置布局】(nrhs 行 × n 列)求解: 每个右端项一整行连续, 前代/回代全是连续访问。
 * 数学与按列解等价, 只是内存布局 —— 不影响任一次浮点运算的操作数。 */
static void lu_factor(double *A, int n, int *piv) {
    for (int k = 0; k < n; k++) {
        int p = k; double mx = fabs(A[(size_t)k * n + k]);
        for (int i = k + 1; i < n; i++) { double v = fabs(A[(size_t)i * n + k]); if (v > mx) { mx = v; p = i; } }
        piv[k] = p;
        if (p != k) for (int j = 0; j < n; j++) { double t = A[(size_t)k * n + j]; A[(size_t)k * n + j] = A[(size_t)p * n + j]; A[(size_t)p * n + j] = t; }
        double d = A[(size_t)k * n + k];
        if (d == 0.0) die("LU: 第 %d 个主元为 0(G 奇异)", k);
        for (int i = k + 1; i < n; i++) {
            double m = A[(size_t)i * n + k] / d;
            A[(size_t)i * n + k] = m;
            if (m == 0.0) continue;
            const double *ak = A + (size_t)k * n;
            double *ai = A + (size_t)i * n;
            for (int j = k + 1; j < n; j++) ai[j] -= m * ak[j];
        }
    }
}
typedef struct { const double *LU; const int *piv; double *BT; int n, ldbt; } lus_ctx;
static void lu_solve_worker(void *vc, int c0, int c1) {
    lus_ctx *s = (lus_ctx *)vc;
    const int n = s->n;
    for (int c = c0; c < c1; c++) {
        double *b = s->BT + (size_t)c * s->ldbt;
        for (int k = 0; k < n; k++) { int p = s->piv[k]; if (p != k) { double t = b[k]; b[k] = b[p]; b[p] = t; } }
        for (int i = 1; i < n; i++) {                      /* L y = Pb (L 单位下三角) */
            const double *li = s->LU + (size_t)i * n;
            double t = b[i];
            for (int k = 0; k < i; k++) t -= li[k] * b[k];
            b[i] = t;
        }
        for (int i = n - 1; i >= 0; i--) {                 /* U x = y */
            const double *ui = s->LU + (size_t)i * n;
            double t = b[i];
            for (int k = i + 1; k < n; k++) t -= ui[k] * b[k];
            b[i] = t / ui[i];
        }
    }
}
/* 解 A·X = B, A 为 n×n(原地毁), BT 为 nrhs×n 的【B 转置】, 原地出 Xᵀ。 */
static void lu_solve_T(double *A, int n, double *BT, int nrhs, int ldbt) {
    int *piv = (int *)xmalloc((size_t)n * sizeof(int));
    lu_factor(A, n, piv);
    lus_ctx s = {A, piv, BT, n, ldbt};
    parallel_for(nrhs, lu_solve_worker, &s);
    free(piv);
}

/* ---------------- np.linalg.svd 位: 随机子空间迭代 + Rayleigh-Ritz ----------------
 * 只需要 Vt 的前 2*KMAX=2048 行(右奇异向量)。真实规模是 Y = 13106×12288 ——
 * 拿"整块 Gram + 全特征分解"去取前 16% 的方向是错的工具, 有两个硬伤(2026-08-25 实测):
 *   ① 慢到不可用: 全特征分解是 O(n³) 且 QL 的特征向量累加天然串行。实测 n=2000 → 3.6s、
 *      n=4000 → 28s(≈n^3.2), 外推 n=13106 ≈ 16 分钟 —— 而这一段是【历史上从不当选】的候选。
 *   ② 会崩: 首版走【行】Gram(na×na), 而真数据 na=13106 > DIN=12288 —— 取到了大的那一边,
 *      且 Gram 秩亏 818。spark 真数据报 `tqli: 第 0 个特征值 60 轮不收敛`。
 *      归因: QL 的收缩判据 `|e[m]| + dd == dd` 是【纯相对】判据。特征值动态范围一旦超过
 *      1/ε(≈1e16, 秩亏 + 激活谱重尾时必然), 小特征值那端的 e[m] 里全是来自大特征值的
 *      ~ε·λ_max 舍入噪声, 相对判据【永远不可能满足】⇒ 迭代上限耗尽。调高上限治不了。
 *      (排除项: 秩亏本身不致命 —— 1400×1200 秩亏通过; dither 成对近重复结构也不致命 ——
 *       1800×1500 带 dither 通过。是规模 + 动态范围一起才炸。)
 *
 * 现在的算法 = 随机子空间迭代 + Rayleigh-Ritz(与 calib/zlayer.c 的 SVD 段同族做法):
 *     V ← 正交化(randn(L, DIN)),  L = nv + 64(过采样)
 *     重复 NPOWER 次:  W = Y·Vᵀ ;  V ← 正交化(W·Y)          [= 对 C=YᵀY 做幂迭代]
 *     W = Y·Vᵀ ;  H = W·Wᵀ = Vᵀ·C·V  (L×L 对称正定)
 *     H = Z Λ Zᵀ  →  σ_i = √λ_i ,  Vt_i = (Z·V)_i
 * 好处: ①【整块 Gram 不再出现】(省 1.2-1.37 GB 与 4e12 flop) ②大矩阵只走矩阵乘(可并行、
 * 可上 BLAS) ③稠密特征分解只在 L=2112 阶做一次(约 4s, 稳在 tridiag/QL 的舒适区)
 * ④迭代次数固定 ⇒ 【没有"不收敛"这个失败模式】。
 *
 * ★语义声明不变★: 这段只喂 V₀=PCA / 门=pca2 两个候选。Ritz 向量是主子空间的近似基,
 * 与 gesdd 的精确奇异向量在【符号、简并子空间内的旋转、以及靠近截断处方向的混合】上不同 ——
 * 但 .py 的 215 条历史解算记录里 PCA/pca2 从未当选, 所以这条不确定性实际不落盘。
 * 随机起始块用【独立的 mt_t 实例 + 固定种子】, 不消耗也不扰动 .py 的三条流(seed 1/7/11),
 * 所以本工具仍然是确定性的: 同输入 → 同输出。 */
#define SVD_OVERSAMPLE 256
#define SVD_NPOWER      2
#define SVD_SEED  20250825u

typedef struct { const double *A; long long n; const char *what; long long bad; } fin_ctx;
static void fin_worker(void *vc, int i0, int i1) {
    fin_ctx *c = (fin_ctx *)vc;
    long long per = (c->n + 63) / 64;
    for (int b = i0; b < i1; b++) {
        long long s = (long long)b * per, e = s + per > c->n ? c->n : s + per;
        for (long long i = s; i < e; i++)
            if (!isfinite(c->A[i])) { if (c->bad < 0) c->bad = i; break; }
    }
}
/* 入口守卫: 上游(zcache / 锚 / dither)一旦带进 NaN/Inf, 整条解算就是垃圾, 且会在特征分解里
 * 表现成"不收敛"这种误导性症状。这里一次性查掉, 报清楚是哪个量。 */
static void guard_finite(const double *A, long long n, const char *what) {
    fin_ctx c = {A, n, what, -1};
    parallel_for(64, fin_worker, &c);
    if (c.bad >= 0)
        die("%s 含非有限值(第 %lld 个 = %g) —— 上游 zcache/锚/dither 已带进 NaN/Inf, 停车",
            what, c.bad, A[c.bad]);
}

static void guard_finite_f32(const float *A, long long n, const char *what) {
    for (long long i = 0; i < n; i++)
        if (!isfinite(A[i]))
            die("%s 第 %lld 个元素非有限(%g) —— 上游 zcache/锚已经带进 NaN/Inf, 停车", what, i, (double)A[i]);
}

/* 右看 Cholesky: G(n×n 对称正定, 行主序) → 原地出下三角 L(G = L·Lᵀ)。 */
typedef struct { double *G; const double *col; int n, k; } ch_ctx;
static void ch_worker(void *vc, int i0, int i1) {
    ch_ctx *c = (ch_ctx *)vc;
    const int n = c->n, k = c->k;
    for (int t = i0; t < i1; t++) {
        int i = k + 1 + t;
        double *gi = c->G + (size_t)i * n;
        double f = gi[k];
        if (f == 0.0) continue;
        /* 减的是 L[i][k]·L[j][k] —— 必须取【已缩放的第 k 列】。取第 k 行是错的:
         * 上三角还留着未缩放的原始对称值, 拿它更新会把因子算坏(首版就栽在这, 表现成
         * "起始块主元 ≤ 0" 这种看起来像数值退化的假象)。列先收拢成连续缓冲再用。 */
        for (int j = k + 1; j <= i; j++) gi[j] -= f * c->col[j];
    }
}
/* 返回 n(全部主元合格), 或第一个"数值上已落在前面行张成空间里"的行号。
 * 容差 = n·ε·max(初始对角), 与 LAPACK 判秩同款量级。 */
static int chol_lower_rr(double *G, int n) {
    double dmax = 0.0;
    for (int i = 0; i < n; i++) if (G[(size_t)i * n + i] > dmax) dmax = G[(size_t)i * n + i];
    const double tol = (double)n * 2.220446049250313e-16 * dmax;
    double *col = (double *)xmalloc((size_t)n * sizeof(double));
    int rank = n;
    for (int k = 0; k < n; k++) {
        double d = G[(size_t)k * n + k];
        if (!(d > tol)) { rank = k; break; }
        d = sqrt(d);
        G[(size_t)k * n + k] = d;
        for (int i = k + 1; i < n; i++) { double v = G[(size_t)i * n + k] / d; G[(size_t)i * n + k] = v; col[i] = v; }
        ch_ctx c = {G, col, n, k};
        if (n - k - 1 > 0) {
            if (n - k - 1 > 256) parallel_for(n - k - 1, ch_worker, &c);
            else ch_worker(&c, 0, n - k - 1);
        }
    }
    free(col);
    return rank;
}

/* 行正交化: V 是 Lr×N(每行一个基向量), 出 V·Vᵀ = I。CholQR 走两遍(CholQR2) ——
 * 一遍在条件数 >1e8 时精度不够, 两遍是数值稳的标准做法, 且全是三级 BLAS 形状。
 * 前代 L⁻¹V 按【列】切给线程: 行方向的顺序依赖在每段列内完整保留, 与串行逐位同。 */
#ifndef DQ_BLAS
typedef struct { double *V; const double *L; int Lr; long long N; } cq_ctx;
static void cq_worker(void *vc, int j0, int j1) {
    cq_ctx *c = (cq_ctx *)vc;
    for (int r = 0; r < c->Lr; r++) {
        double *vr = c->V + (size_t)r * c->N;
        const double *lr = c->L + (size_t)r * c->Lr;
        for (int t = 0; t < r; t++) {
            double f = lr[t];
            if (f == 0.0) continue;
            const double *vt = c->V + (size_t)t * c->N;
            for (int j = j0; j < j1; j++) vr[j] -= f * vt[j];
        }
        double di = 1.0 / lr[r];
        for (int j = j0; j < j1; j++) vr[j] *= di;
    }
}
#endif
static void orth_rows(double *V, int Lr, long long N, double *G, const char *what, mt_t *rng) {
    for (int pass = 0; pass < 2; pass++) {
        for (int round = 0; ; round++) {
            mm_nt(Lr, Lr, (int)N, V, (int)N, V, (int)N, G, Lr);
            int r = chol_lower_rr(G, Lr);
            if (r == Lr) break;
            /* 第 r 行起在数值上已落进前 r 行张成的空间 —— 只会发生在 Y 的数值秩 < L 时。
             * 这些方向对应 σ=0, numpy 的 gesdd 同样只是返回零空间里的【任意】正交补,
             * 所以正确做法是换一批新随机向量再正交化, 不是报错。用的是同一条确定性流,
             * 工具仍然同输入→同输出。 */
            if (round >= 3)
                die("CholQR(%s): 重随机化 4 轮后数值秩仍只有 %d < %d —— Y 的秩远低于所求方向数, "
                    "解算无意义, 停车。", what, r, Lr);
            for (long long i = (long long)r * N; i < (long long)Lr * N; i++) V[i] = mt_gauss(rng);
            fprintf(stderr, "[amp_solve_zc]   %s: 数值秩 %d < L=%d, 尾部 %d 行重随机化(第 %d 轮)\n",
                    what, r, Lr, Lr - r, round + 1);
        }
#ifdef DQ_BLAS
        /* 前代 L⁻¹V 就是 dtrsm(Left/Lower/NoTrans/NonUnit) —— 交给 BLAS, 三级形状。
         * 求和顺序与下面手写路不同, 属既有的"有/无 BLAS 两种构建不逐位"范畴。 */
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    Lr, (int)N, 1.0, G, Lr, V, (int)N);
#else
        cq_ctx c = {V, G, Lr, N};
        parallel_for((int)N, cq_worker, &c);
#endif
    }
}

/* ---- 小规模稠密对称特征分解(只在 L=nv+64 阶用, 约 2112) ----
 * Householder 三对角化 → 逐行累加特征向量 → 隐式位移 QL。特征向量按【行】存
 * (ZT[i] = 第 i 个特征向量): QL 旋转每次动相邻两行, 全连续访问; 按列存是 n×8 字节的
 * stride, 每次访问一个 cache line, 慢十倍以上。 */
typedef struct { double *A; const double *v, *w; int n, l; } h2_ctx;
static void h2_mv(void *vc, int j0, int j1) {          /* p[j] = Σ_k A[j][k]v[k], j ≤ l */
    h2_ctx *c = (h2_ctx *)vc;
    double *p = (double *)c->w;
    for (int j = j0; j < j1; j++) {
        const double *aj = c->A + (size_t)j * c->n;
        double s = 0.0;
        for (int k = 0; k <= c->l; k++) s += aj[k] * c->v[k];
        p[j] = s;
    }
}
static void h2_upd(void *vc, int j0, int j1) {         /* A -= v wᵀ + w vᵀ(两半都更新) */
    h2_ctx *c = (h2_ctx *)vc;
    for (int j = j0; j < j1; j++) {
        double *aj = c->A + (size_t)j * c->n;
        double vj = c->v[j], wj = c->w[j];
        for (int k = 0; k <= c->l; k++) aj[k] -= vj * c->w[k] + wj * c->v[k];
    }
}
static void pf_small(int n, pf_fn fn, void *ctx) {     /* 活太小就别开线程 */
    if (n < 512) fn(ctx, 0, n); else parallel_for(n, fn, ctx);
}
static void tridiag(double *A, int n, double *d, double *e, double *hv, double *hh) {
    double *p = (double *)xmalloc((size_t)n * sizeof(double));
    double *w = (double *)xmalloc((size_t)n * sizeof(double));
    for (int i = n - 1; i >= 1; i--) {
        int l = i - 1;
        double *v = hv + (size_t)i * n;
        d[i] = A[(size_t)i * n + i];
        memset(v, 0, (size_t)n * sizeof(double));
        hh[i] = 0.0;
        if (l == 0) { e[i] = A[(size_t)i * n + 0]; continue; }
        double scale = 0.0;
        for (int k = 0; k <= l; k++) scale += fabs(A[(size_t)i * n + k]);
        if (scale == 0.0) { e[i] = A[(size_t)i * n + l]; continue; }
        double h = 0.0;
        for (int k = 0; k <= l; k++) { v[k] = A[(size_t)i * n + k] / scale; h += v[k] * v[k]; }
        double f = v[l];
        double g = (f >= 0.0 ? -sqrt(h) : sqrt(h));
        e[i] = scale * g;
        h -= f * g;
        v[l] = f - g;
        hh[i] = h;
        h2_ctx c = {A, v, p, n, l};
        pf_small(l + 1, h2_mv, &c);
        double K = 0.0;
        for (int j = 0; j <= l; j++) { p[j] /= h; K += v[j] * p[j]; }
        K /= (2.0 * h);
        for (int j = 0; j <= l; j++) w[j] = p[j] - K * v[j];
        c.w = w;
        pf_small(l + 1, h2_upd, &c);
        for (int k = 0; k <= l; k++) { A[(size_t)i * n + k] = 0.0; A[(size_t)k * n + i] = 0.0; }
        A[(size_t)i * n + l] = e[i]; A[(size_t)l * n + i] = e[i];
    }
    d[0] = A[0]; e[0] = 0.0;
    free(p); free(w);
}
/* QT = H_1·H_2·⋯·H_{n-1} 逐行构造。H_i 只动 0..i-1 列, 而应用到第 i 步时 r ≥ i 的行
 * 还是单位行 ⇒ 只需遍历 r < i。 */
typedef struct { double *QT; const double *v; int n, l; double h; } qb_ctx;
static void qb_worker(void *vc, int r0, int r1) {
    qb_ctx *c = (qb_ctx *)vc;
    for (int r = r0; r < r1; r++) {
        double *q = c->QT + (size_t)r * c->n;
        double s = 0.0;
        for (int k = 0; k <= c->l; k++) s += q[k] * c->v[k];
        s /= c->h;
        for (int k = 0; k <= c->l; k++) q[k] -= s * c->v[k];
    }
}
static void build_QT(double *QT, int n, const double *hv, const double *hh) {
    memset(QT, 0, (size_t)n * n * sizeof(double));
    for (int r = 0; r < n; r++) QT[(size_t)r * n + r] = 1.0;
    for (int i = 1; i < n; i++) {
        if (hh[i] == 0.0) continue;
        qb_ctx c = {QT, hv + (size_t)i * n, n, i - 1, hh[i]};
        pf_small(i, qb_worker, &c);
    }
}
static double pythag(double a, double b) {
    double aa = fabs(a), ab = fabs(b);
    if (aa > ab) { double r = ab / aa; return aa * sqrt(1.0 + r * r); }
    if (ab == 0.0) return 0.0;
    { double r = aa / ab; return ab * sqrt(1.0 + r * r); }
}
/* 隐式位移 QL(EISPACK tql2 同式), 特征向量按行累加在 ZT 上。
 * ★收缩判据带绝对下限★: 纯相对判据 `|e[m]|+dd == dd` 在特征值动态范围超过 1/ε 时,
 * 小特征值那端永远收缩不了(e[m] 里是大特征值留下的 ~ε·‖T‖ 噪声) —— 这正是首版在真数据上
 * "第 0 个特征值 60 轮不收敛"的成因。LAPACK dsteqr 同样带这个绝对下限。 */
static void tqli_rows(double *d, double *e, int n, double *ZT) {
    const double EPS = 2.220446049250313e-16;
    for (int i = 0; i < n; i++)
        if (!isfinite(d[i]) || !isfinite(e[i]))
            die("tqli: 三对角阵含非有限值(i=%d d=%g e=%g) —— 上游已经坏了, 不是收敛问题", i, d[i], e[i]);
    double anorm = 0.0;
    for (int i = 0; i < n; i++) {
        double r = fabs(d[i]) + fabs(e[i]) + (i ? fabs(e[i - 1]) : 0.0);
        if (r > anorm) anorm = r;
    }
    const double afloor = EPS * anorm;          /* 绝对下限, 见上 */
    for (int i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (int l = 0; l < n; l++) {
        int iter = 0, m, i;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= EPS * dd + afloor) break;
            }
            if (m == l) break;
            if (iter++ == 100)
                die("tqli: 第 %d 个特征值 100 轮不收敛(m=%d e[m]=%.6g d[l]=%.6g ‖T‖≈%.6g)",
                    l, m, e[m], d[l], anorm);
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = pythag(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? fabs(r) : -fabs(r)));
            double s = 1.0, c = 1.0, p = 0.0;
            int brk = 0;
            for (i = m - 1; i >= l; i--) {
                double f = s * e[i], b = c * e[i];
                r = pythag(f, g);
                e[i + 1] = r;
                if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; brk = 1; break; }
                s = f / r; c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                double *zi = ZT + (size_t)i * n, *zj = ZT + (size_t)(i + 1) * n;
                for (int k = 0; k < n; k++) {
                    double fz = zj[k];
                    zj[k] = s * zi[k] + c * fz;
                    zi[k] = c * zi[k] - s * fz;
                }
            }
            if (brk) continue;
            d[l] -= p; e[l] = g; e[m] = 0.0;
        } while (m != l);
    }
}
typedef struct { double lam; int idx; } eig_t;
static int eig_cmp(const void *a, const void *b) {     /* 降序; 平局按下标稳定 */
    const eig_t *x = (const eig_t *)a, *y = (const eig_t *)b;
    if (x->lam > y->lam) return -1;
    if (x->lam < y->lam) return 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx);
}
/* 对称阵 H(n×n) 的全部特征对: 出 d(未排序)与 ZT(行=特征向量), H 被毁。 */
static void sym_eig_rows(double *H, int n, double *d, double *ZT) {
    double *e = (double *)xmalloc((size_t)n * 8);
    double *hv = (double *)xmalloc((size_t)n * n * 8);
    double *hh = (double *)xmalloc((size_t)n * 8);
    tridiag(H, n, d, e, hv, hh);
    build_QT(ZT, n, hv, hh);
    free(hv); free(hh);
    tqli_rows(d, e, n, ZT);
    free(e);
}

/* Y(na×DIN) 的前 nv 个右奇异向量 → Vt(nv×DIN, 行主序, 行正交)。
 * 同时可选出 sig[nv](降序奇异值), 传 NULL 表示不要。 */
static void top_right_singular(const double *Y, int na, long long DIN, int nv, double *Vt, double *sig) {
    guard_finite(Y, (long long)na * DIN, "SVD 输入 Xa−mean");
    int rmax = na < (int)DIN ? na : (int)DIN;
    if (nv > rmax) die("要 %d 个右奇异向量但 rank 上限只有 %d", nv, rmax);
    int L = nv + SVD_OVERSAMPLE; if (L > rmax) L = rmax;
    double *V = (double *)xmalloc((size_t)L * DIN * 8);
    double *W = (double *)xmalloc((size_t)L * na * 8);
    double *G = (double *)xmalloc((size_t)L * L * 8);
    mt_t rng; mt_seed(&rng, SVD_SEED);
    for (long long i = 0; i < (long long)L * DIN; i++) V[i] = mt_gauss(&rng);
    orth_rows(V, L, DIN, G, "起始块", &rng);
    /* ★每半步都正交化★, 不是每整步。整步才正交时, 一步的条件数是 (σ₁/σ_L)² ——
     * 衰减谱下这个数轻易破 1e8, 子空间会数值塌掉(实测 13106×12288 / 1/(1+i) 谱, 每整步
     * 正交时数值秩从 2112 塌到 1148, 尾部近千个方向被迫重随机化 = 那些方向直接算错了)。
     * 每半步正交后条件数降到 σ₁/σ_L, CholQR2 稳稳吃得下。代价约 +15% 矩阵乘。 */
    for (int it = 0; it < SVD_NPOWER; it++) {
        mm_nt(L, na, (int)DIN, V, (int)DIN, Y, (int)DIN, W, na);    /* W = V·Yᵀ  (L×na)  */
        orth_rows(W, L, na, G, "幂迭代(左)", &rng);
        mm_nn(L, (int)DIN, na, W, na, Y, (int)DIN, V, (int)DIN);    /* V = W·Y   (L×DIN) */
        orth_rows(V, L, DIN, G, "幂迭代(右)", &rng);
        fprintf(stderr, "[amp_solve_zc]   子空间迭代 %d/%d\n", it + 1, SVD_NPOWER);
    }
    mm_nt(L, na, (int)DIN, V, (int)DIN, Y, (int)DIN, W, na);
    double *H = (double *)xmalloc((size_t)L * L * 8);
    mm_nt(L, L, na, W, na, W, na, H, L);                            /* H = W·Wᵀ = VᵀCV   */
    free(W);
    double *d = (double *)xmalloc((size_t)L * 8);
    double *ZT = G;                                                 /* G 已用完, 复用 L×L */
    sym_eig_rows(H, L, d, ZT);
    free(H);
    eig_t *ord = (eig_t *)xmalloc((size_t)L * sizeof(eig_t));
    for (int i = 0; i < L; i++) { ord[i].lam = d[i]; ord[i].idx = i; }
    qsort(ord, (size_t)L, sizeof(eig_t), eig_cmp);
    double *Zs = (double *)xmalloc((size_t)nv * L * 8);
    for (int i = 0; i < nv; i++) {
        memcpy(Zs + (size_t)i * L, ZT + (size_t)ord[i].idx * L, (size_t)L * 8);
        if (sig) sig[i] = ord[i].lam > 0.0 ? sqrt(ord[i].lam) : 0.0;
    }
    mm_nn(nv, (int)DIN, L, Zs, L, V, (int)DIN, Vt, (int)DIN);       /* Vt = Zsel·V */
    free(Zs); free(ord); free(d); free(G); free(V);
}

/* ---------------- 锚读取(scripts/probe_layer_behavior.py: anchor_layer 的 fin 分支) ----
 * 只有 zcache 里没有 xcap 时才走这条退路(.py 第 31-35 行)。头 8 个 u32: hd[1]=S_file,
 * hd[3]=DIM; fin 段从 40 开始, 每层 S_file*DIM*4 字节。注意 .py 用 S_file 算层偏移、
 * 用 ntok(=zcache 的 S)算读取长度 —— 两者可以不等, 照抄。 */
static float *anchor_fin(const char *ap, int L, long long ntok, long long *DIM_out) {
    FILE *f = fopen(ap, "rb");
    if (!f) die("FileNotFoundError: %s", ap);
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8) die("锚头读不全: %s", ap);
    long long Sf = hd[1], DIM = hd[3];
    if (DIM <= 0 || Sf <= 0) die("锚头不合法: S=%lld DIM=%lld", Sf, DIM);
    long long fin_off = 40;
    if (fseeko(f, (off_t)(fin_off + (long long)L * Sf * DIM * 4), SEEK_SET)) die("锚 seek 失败");
    float *fin = (float *)xmalloc((size_t)ntok * (size_t)DIM * 4);
    if (fread(fin, 4, (size_t)(ntok * DIM), f) != (size_t)(ntok * DIM))
        die("锚 fin 读不全: L=%d ntok=%lld DIM=%lld", L, ntok, DIM);
    fclose(f);
    *DIM_out = DIM;
    return fin;
}

/* φ(x) = [x, x⊙x/rms, relu(x)] —— .py 的 _phi, 全程 f32:
 *   n = sqrt(mean_1(M*M)) + 1e-6      (mean 沿连续轴 ⇒ pairwise, f32 累加器)
 * 与引擎 ds4_zchain_zl_apply 的 din==3d 分支逐式一致。 */
typedef struct { const float *M; float *P; long long S, DX; } phi_ctx;
static void phi_worker(void *vc, int i0, int i1) {
    phi_ctx *c = (phi_ctx *)vc;
    long long DX = c->DX, DIN = 3 * DX;
    for (int i = i0; i < i1; i++) {
        const float *m = c->M + (size_t)i * DX;
        float *p = c->P + (size_t)i * DIN;
        float n = sqrtf(pw_sq_f32(m, DX) / (float)DX) + 1e-6f;
        for (long long j = 0; j < DX; j++) {
            float v = m[j];
            p[j] = v;
            p[DX + j] = (v * v) / n;
            p[2 * DX + j] = v > 0.0f ? v : 0.0f;
        }
    }
}

/* python int(s) 的严格语义(前后空白可去, 必须整串是十进制整数)。 */
static long long py_int(const char *s, const char *what) {
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
    if (!*p) die("ValueError: invalid literal for int() with base 10: '%s' (%s)", s, what);
    long long v = 0;
    for (; *p >= '0' && *p <= '9'; p++) v = v * 10 + (*p - '0');
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p) die("ValueError: invalid literal for int() with base 10: '%s' (%s)", s, what);
    return neg ? -v : v;
}

/* zrec 头: 116 字节, 名字在 0, u64 载荷长在 88, i32 1 在 112(.py 的 hdr())。 */
static void put_hdr(uint8_t *h, const char *nm, uint64_t psz) {
    memset(h, 0, 116);
    memcpy(h, nm, strlen(nm));
    memcpy(h + 88, &psz, 8);
    int32_t one = 1;
    memcpy(h + 112, &one, 4);
}

/* 生成谱可控的测试矩阵: Y = A·diag(s)·Bᵀ, s_c = 1/(1+c)^decay(decay=0 即平谱), 秩 = rank。 */
static void st_gen(double *Y, int na, long long DIN, int rank, double decay, uint32_t seed) {
    mt_t g; mt_seed(&g, seed);
    int rmax = na < (int)DIN ? na : (int)DIN;
    if (rank >= rmax) {          /* 满秩: 列缩放的 iid 高斯, O(na·DIN) 生成, 谱由 s_j 定 */
        for (int i = 0; i < na; i++) {
            double *r = Y + (size_t)i * DIN;
            for (long long j = 0; j < DIN; j++)
                r[j] = mt_gauss(&g) * (decay == 0.0 ? 1.0 : pow(1.0 / (1.0 + (double)j), decay));
        }
        return;
    }
    double *A = (double *)xmalloc((size_t)na * rank * 8);
    double *B = (double *)xmalloc((size_t)DIN * rank * 8);
    for (long long i = 0; i < (long long)na * rank; i++) A[i] = mt_gauss(&g);
    for (long long i = 0; i < DIN * rank; i++) B[i] = mt_gauss(&g);
    for (int c = 0; c < rank; c++) {
        double s = decay == 0.0 ? 1.0 : pow(1.0 / (1.0 + (double)c), decay);
        for (int i = 0; i < na; i++) A[(size_t)i * rank + c] *= s;
    }
    mm_nt(na, (int)DIN, rank, A, rank, B, rank, Y, (int)DIN);
    free(A); free(B);
}
/* 投影残差 ‖Y − (Y·Vᵀ)·V‖_F / ‖Y‖_F, 按行分块算, 不开 na×DIN 的大缓冲。
 * (不能用 ‖Y‖²−‖P‖² 那个恒等式: 残差接近 0 时是灾难性相消, 开方后只能验到 ~1e-8。) */
static double st_residual(const double *Y, int na, long long DIN, const double *V, int nv) {
    const int BLK = 256;
    if (na > 4096) na = 4096;   /* 抽样前 4096 行: 比值有代表性, 全量和 SVD 本身一样贵 */
    double *P = (double *)xmalloc((size_t)BLK * nv * 8);
    double *R = (double *)xmalloc((size_t)BLK * DIN * 8);
    double sy = 0.0, sr = 0.0;
    for (int b = 0; b < na; b += BLK) {
        int nb = b + BLK > na ? na - b : BLK;
        const double *Yb = Y + (size_t)b * DIN;
        mm_nt(nb, nv, (int)DIN, Yb, (int)DIN, V, (int)DIN, P, nv);
        mm_nn(nb, (int)DIN, nv, P, nv, V, (int)DIN, R, (int)DIN);
        for (long long i = 0; i < (long long)nb * DIN; i++) {
            double y = Yb[i], r = y - R[i];
            sy += y * y; sr += r * r;
        }
    }
    free(P); free(R);
    return sy > 0.0 ? sqrt(sr / sy) : 0.0;
}
static void selftest(int argc, char **argv) {
    printf("== RNG: np.random.RandomState(1).randn(8) ==\n");
    mt_t s; mt_seed(&s, 1);
    for (int i = 0; i < 8; i++) printf("%.17g\n", mt_gauss(&s));
    printf("== f16 舍入抽查(值 → 位) ==\n");
    const double tv[] = {1.0, -2.5, 6.103515625e-05, 1e-8, 65519.0, 65520.0, 1.0009765625, 1.00048828125};
    for (int i = 0; i < 8; i++) printf("%.17g -> 0x%04x\n", tv[i], f64_to_f16(tv[i]));

    /* 用法: --selftest [na DIN nv rank decay]。缺省是小号(rank ≤ nv, 投影残差判据可验)。 */
    int na   = argc > 2 ? (int)py_int(argv[2], "na")   : 800;
    long long DIN = argc > 3 ? py_int(argv[3], "DIN")  : 700;
    int nv   = argc > 4 ? (int)py_int(argv[4], "nv")   : 200;
    int rank = argc > 5 ? (int)py_int(argv[5], "rank") : 150;
    double decay = argc > 6 ? atof(argv[6]) : 1.0;
    int rmax = na < (int)DIN ? na : (int)DIN;
    if (rank > rmax) rank = rmax;
    printf("== SVD: Y %d×%lld  取前 %d 个右奇异向量  秩=%d  谱=%s  (L=%d, 幂迭代 %d 轮) ==\n",
           na, DIN, nv, rank, decay == 0.0 ? "平" : "1/(1+i)^decay", nv + SVD_OVERSAMPLE, SVD_NPOWER);
    fflush(stdout);
    double *Y = (double *)xmalloc((size_t)na * DIN * 8);
    st_gen(Y, na, DIN, rank, decay, 5);
    double *Vt = (double *)xmalloc((size_t)nv * DIN * 8);
    double *sig = (double *)xmalloc((size_t)nv * 8);
    double t = now_s();
    top_right_singular(Y, na, DIN, nv, Vt, sig);
    double dt = now_s() - t;
    /* 正交性 */
    double onmax = 0.0, dgmax = 0.0;
    int nchk = nv < 64 ? nv : 64;
    for (int i = 0; i < nchk; i++) {
        double q = pw_sq_f64(Vt + (size_t)i * DIN, DIN);
        if (fabs(q - 1.0) > dgmax) dgmax = fabs(q - 1.0);
        for (int j = i + 1; j < nchk; j++) {
            double dp = 0.0;
            for (long long k = 0; k < DIN; k++) dp += Vt[(size_t)i * DIN + k] * Vt[(size_t)j * DIN + k];
            if (fabs(dp) > onmax) onmax = fabs(dp);
        }
    }
    /* Ritz 残差 ‖Yᵀ(Y v) − σ²v‖ / σ²(抽前几个 + 截断处附近) */
    int probe[6] = {0, 1, nv / 4, nv / 2, 3 * nv / 4, nv - 1};
    double rrmax = 0.0, lam0 = sig[0] * sig[0];
    double *u = (double *)xmalloc((size_t)na * 8), *z = (double *)xmalloc((size_t)DIN * 8);
    for (int pi = 0; pi < 6; pi++) {
        int i = probe[pi];
        const double *v = Vt + (size_t)i * DIN;
        mm_nt(1, na, (int)DIN, v, (int)DIN, Y, (int)DIN, u, na);
        mm_nn(1, (int)DIN, na, u, na, Y, (int)DIN, z, (int)DIN);
        double lam = sig[i] * sig[i], num = 0.0;
        for (long long k = 0; k < DIN; k++) { double r = z[k] - lam * v[k]; num += r * r; }
        /* 按 λ_0 归一, 不按 λ_i: 零空间方向的 λ_i≈0 会把比值刷爆, 那不是精度问题 */
        double rr = lam0 > 0.0 ? sqrt(num) / lam0 : 0.0;
        if (rr > rrmax) rrmax = rr;
    }
    free(u); free(z);
    double res = st_residual(Y, na, DIN, Vt, nv);
    printf("耗时 %.1fs  前%d: |‖v‖²−1| max %.3e  正交 max %.3e   Ritz 残差 max %.3e\n",
           dt, nchk, dgmax, onmax, rrmax);
    printf("投影残差 ‖Y−(YVᵀ)V‖/‖Y‖ = %.6e%s   %s\n", res, na > 4096 ? "(前4096行)" : "",
           rank <= nv ? (res < 1e-10 ? "★秩≤nv, <1e-10 ✓" : "★秩≤nv 但 ≥1e-10 ✗")
                      : "(秩>nv: 该值 = 被截掉的尾部能量, 数学上不可能小, 见下)");
    if (rank > nv) {
        printf("  └ 满秩/高秩下这个判据不适用: 只取前 %d 个方向, 残差 = √(Σ_{i>%d}σ²/Σσ²)。\n"
               "    对本例的期望量级: 平谱≈√(1−nv/rank)=%.3f, 1/(1+i) 谱≈%.3f。\n"
               "    该场景要看的是上一行的【Ritz 残差】与【正交性】, 以及下面的 σ 与 numpy 对表。\n",
               nv, nv, sqrt(1.0 - (double)nv / rank), 0.0);
    }
    printf("σ[0..7] =");
    for (int i = 0; i < 8 && i < nv; i++) printf(" %.10g", sig[i]);
    printf("\nσ[nv-4..nv-1] =");
    for (int i = nv - 4; i < nv; i++) if (i >= 0) printf(" %.10g", sig[i]);
    printf("\n");
    free(Y); free(Vt); free(sig);
}

/* ================================ 主流程 ================================
 * 下面每一段都标了对应的 amp_solve.py 行号, 改的时候两边一起看。 */
int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) { selftest(argc, argv); return 0; }
    if (argc < 4) {
        fprintf(stderr, "用法: amp_solve_zc <anchor> <zcache_LXX.npz> <out_amprec.bin> [NFIT]\n");
        fprintf(stderr, "      amp_solve_zc --selftest [na DIN nv rank decay]\n");
        return 1;
    }
    const char *ap = argv[1], *zcp = argv[2], *outp = argv[3];       /* py:14 */
    long long NFIT = argc > 4 ? py_int(argv[4], "NFIT") : 0;         /* py:15 */

    /* py:16  L = int(os.path.basename(zcp).split("_L")[1][:2]) */
    const char *base = strrchr(zcp, '/'); base = base ? base + 1 : zcp;
    const char *tag = strstr(base, "_L");
    if (!tag) die("IndexError: list index out of range (文件名 %s 里没有 \"_L\")", base);
    char lbuf[3] = {0, 0, 0};
    strncpy(lbuf, tag + 2, 2);
    int L = (int)py_int(lbuf, "层号");

    /* py:17-19  zc = np.load(zcp); dH/prow/pw/pYQ; S,D = dH.shape */
    int zfd = open(zcp, O_RDONLY);
    if (zfd < 0) die("FileNotFoundError: %s", zcp);
    struct stat st;
    if (fstat(zfd, &st)) die("stat 失败: %s", zcp);
    long long zsz = st.st_size;
    const uint8_t *zb = (const uint8_t *)mmap(NULL, (size_t)zsz, PROT_READ, MAP_PRIVATE, zfd, 0);
    if (zb == MAP_FAILED) die("zcache mmap 失败: %s", zcp);
    npy_t adH, aprow, apw, apYQ, ayqe, axcap;
    if (npz_find(zb, zsz, "dH", &adH)) die("KeyError: dH is not a file in the archive");
    /* prow/pw/pYQ: .py 第 18 行无条件取出(缺就 KeyError), 但有 yqe 时其实用不上 —— 只查在,
     * 不物化(S=8192 时 pYQ 有 GB 量级, 白搬一遍)。 */
    if (npz_find(zb, zsz, "prow", &aprow)) die("KeyError: prow is not a file in the archive");
    if (npz_find(zb, zsz, "pw", &apw)) die("KeyError: pw is not a file in the archive");
    if (npz_find(zb, zsz, "pYQ", &apYQ)) die("KeyError: pYQ is not a file in the archive");
    long long S = adH.d0, D = adH.d1;                                /* py:19 */
    if (adH.nd != 2 || S <= 0 || D <= 0) die("dH 形状不合法");
    if (NFIT <= 0) NFIT = S * 8 / 10;                                /* py:20 */
    if (NFIT >= S) die("NFIT=%lld ≥ S=%lld: held 段为空, .py 会在 e0=0 上除零", NFIT, S);
    long long NEV = S - NFIT;

    float *dH = (float *)xmalloc((size_t)S * D * 4);
    npz_to_f32(&adH, dH, S * D);

    /* py:31-35  X0 = xcap(引擎真值) 优先, 否则退回 FP 锚 */
    float *X0 = NULL; long long DX = 0; const char *X0src = "xcap";
    if (!npz_find(zb, zsz, "xcap", &axcap)) {
        DX = axcap.d1;
        X0 = (float *)xmalloc((size_t)S * DX * 4);
        npz_to_f32(&axcap, X0, S * DX);
    } else {
        const char *xap = getenv("DS4_ZL_XANCHOR");                  /* .py 既有 env, 非新增 */
        X0 = anchor_fin(xap && *xap ? xap : ap, L, S, &DX);
        X0src = "锚 fin";
    }
    double t0 = now_s();                                             /* py:36 计时从这里起 */

    /* py:40-44  yq = 引擎真实部署态 routed 输出; 没有 yqe 时从 per-expert 片段重建 */
    float *yq = (float *)xcalloc((size_t)S * D, 4);
    if (!npz_find(zb, zsz, "yqe", &ayqe)) {
        npz_to_f32(&ayqe, yq, S * D);
    } else {
        long long npair = aprow.d0;
        if (apYQ.d0 != npair || apYQ.d1 != D) die("pYQ 形状 %lldx%lld 与 prow(%lld)/D(%lld) 不配",
                                                  apYQ.d0, apYQ.d1, npair, D);
        long long *prow = (long long *)xmalloc((size_t)npair * 8);
        npz_to_i64(&aprow, prow, npair);
        float *pw = (float *)xmalloc((size_t)npair * 4);
        npz_to_f32(&apw, pw, npair);
        /* np.add.at(yq, prow, pw[:,None]*pYQ): 乘积先在 f32 舍入, 再无缓冲逐对累加(顺序=对序) */
        for (long long i = 0; i < npair; i++) {
            long long t = prow[i];
            if (t < 0 || t >= S) die("zcache prow 越界: %lld", t);
            float w = pw[i];
            float *dst = yq + (size_t)t * D;
            const uint8_t *src = apYQ.data + (size_t)i * D * 4;
            for (long long j = 0; j < D; j++) {
                float v; memcpy(&v, src + j * 4, 4);
                dst[j] += w * v;
            }
        }
        free(prow); free(pw);
    }
    /* NaN 入口守卫: 坏值往下走会伪装成"held=nan → 层闸"或特征分解"不收敛", 症状全是误导的 */
    guard_finite_f32(dH, S * D, "zcache dH");
    guard_finite_f32(yq, S * D, "yq(yqe 或 add.at 重建)");
    guard_finite_f32(X0, S * DX, X0src);

    /* py:45  yfp = yq + dH (f32) */
    float *yfp = (float *)xmalloc((size_t)S * D * 4);
    for (long long i = 0; i < S * D; i++) yfp[i] = yq[i] + dH[i];

    /* py:51-56  X0 = _phi(X0); X = X0.astype(f64); DIN = X.shape[1] */
    long long DIN = 3 * DX;
    float *Phi = (float *)xmalloc((size_t)S * DIN * 4);
    { phi_ctx pc = {X0, Phi, S, DX}; parallel_for((int)S, phi_worker, &pc); }
    free(X0); X0 = NULL;
    double *X = (double *)xmalloc((size_t)S * DIN * 8);
    for (long long i = 0; i < S * DIN; i++) X[i] = (double)Phi[i];
    free(Phi);
    fprintf(stderr, "[amp_solve_zc] L%d S=%lld D=%lld DIN=%lld NFIT=%lld NEV=%lld 线程=%d\n",
            L, S, D, DIN, NFIT, NEV, n_threads());

    /* py:57  eps = sqrt(mean_0(yq[tr]²))*1e-2 + 1e-12   (全程 f32, axis0 ⇒ 朴素累加) */
    float *m2 = (float *)xcalloc((size_t)D, 4);
    for (long long i = 0; i < NFIT; i++) {
        const float *r = yq + (size_t)i * D;
        for (long long j = 0; j < D; j++) { float s = r[j] * r[j]; m2[j] += s; }
    }
    for (long long j = 0; j < D; j++) m2[j] = m2[j] / (float)NFIT;
    float *eps = (float *)xmalloc((size_t)D * 4);
    for (long long j = 0; j < D; j++) eps[j] = sqrtf(m2[j]) * 1e-2f + 1e-12f;

    /* py:58  R = (dH*yq/(yq²+eps²)).astype(f64) —— 括号里全是 f32, 最后才升 f64。
     * 这一处是精度阶梯最容易被"顺手改好"的地方(zlayer 转录清单 #2 同款坑)。 */
    double *R = (double *)xmalloc((size_t)S * D * 8);
    for (long long i = 0; i < S; i++) {
        const float *dh = dH + (size_t)i * D, *y = yq + (size_t)i * D;
        double *r = R + (size_t)i * D;
        for (long long j = 0; j < D; j++) {
            float e2 = eps[j] * eps[j];
            float num = dh[j] * y[j];
            float den = y[j] * y[j] + e2;
            r[j] = (double)(num / den);
        }
    }

    /* py:59  colw = sqrt(var_0(R[tr]) + 1e-12) * sqrt(mean_0(yq[tr]²) + 1e-12)
     * 左半 f64(R 已升), 右半 f32(m2 还是 f32), 乘积 f64。 */
    double *colw = (double *)xmalloc((size_t)D * 8);
    {
        double *mu = (double *)xcalloc((size_t)D, 8);
        for (long long i = 0; i < NFIT; i++) { const double *r = R + (size_t)i * D; for (long long j = 0; j < D; j++) mu[j] += r[j]; }
        for (long long j = 0; j < D; j++) mu[j] /= (double)NFIT;
        double *va = (double *)xcalloc((size_t)D, 8);
        for (long long i = 0; i < NFIT; i++) { const double *r = R + (size_t)i * D; for (long long j = 0; j < D; j++) { double d = r[j] - mu[j]; va[j] += d * d; } }
        for (long long j = 0; j < D; j++) va[j] /= (double)NFIT;
        for (long long j = 0; j < D; j++) colw[j] = sqrt(va[j] + 1e-12) * (double)sqrtf(m2[j] + 1e-12f);
        free(mu); free(va);
    }

    /* py:60-63  rms / dither 增广 Xa / Ra */
    long long na = 2 * NFIT;
    double *rms = (double *)xmalloc((size_t)NFIT * 8);
    for (long long i = 0; i < NFIT; i++) rms[i] = sqrt(pw_sq_f64(X + (size_t)i * DIN, DIN) / (double)DIN);
    double *Xa = (double *)xmalloc((size_t)na * DIN * 8);
    memcpy(Xa, X, (size_t)NFIT * DIN * 8);
    {
        mt_t r2; mt_seed(&r2, 1);                                    /* py:61 RandomState(1) */
        for (long long i = 0; i < NFIT; i++) {
            const double *src = X + (size_t)i * DIN;
            double *dst = Xa + (size_t)(NFIT + i) * DIN;
            double rm = rms[i];
            for (long long j = 0; j < DIN; j++) dst[j] = src[j] + (mt_gauss(&r2) * 0.04) * rm;
        }
    }
    free(rms);
    double *Ra = (double *)xmalloc((size_t)na * D * 8);
    for (long long i = 0; i < NFIT; i++) {
        const double *r = R + (size_t)i * D;
        double *a = Ra + (size_t)i * D, *b = Ra + (size_t)(NFIT + i) * D;
        for (long long j = 0; j < D; j++) { double v = r[j] * colw[j]; a[j] = v; b[j] = v; }
    }
    /* Xev = X[ev] 单独物化, 之后 X 就没用了 */
    double *Xev = (double *)xmalloc((size_t)NEV * DIN * 8);
    memcpy(Xev, X + (size_t)NFIT * DIN, (size_t)NEV * DIN * 8);
    free(X); free(R);

    /* py:64  e0 = ((yfp[ev]-yq[ev]).astype(f64)**2).sum() */
    const float *yq_ev = yq + (size_t)NFIT * D, *yfp_ev = yfp + (size_t)NFIT * D;
    double e0 = pw_sqdiff(yfp_ev, yq_ev, NEV * D);
    if (!(e0 > 0.0)) die("e0=%.17g: held 段 dH 全零, 挽回率无定义", e0);

    /* py:69  KMAX = 1024(引擎 ds4_zchain.c 解析的硬上限) */
    const int KMAX = 1024;
    if (2 * (long long)KMAX > na || 2 * (long long)KMAX > DIN)
        die("min(na=%lld, DIN=%lld) < 2*KMAX=%d: .py 会在 pca2 门上广播失败", na, DIN, 2 * KMAX);

    /* py:82-84  SVD(Xa − mean_0(Xa)) → CANDS/GATES 的 PCA 两段; scale */
    fprintf(stderr, "[amp_solve_zc] L%d SVD(%lld×%lld)…\n", L, na, DIN);
    double *V0[2], *A0[2];
    {
        double *Y = (double *)xmalloc((size_t)na * DIN * 8);
        double *mean = (double *)xcalloc((size_t)DIN, 8);
        for (long long i = 0; i < na; i++) { const double *r = Xa + (size_t)i * DIN; for (long long j = 0; j < DIN; j++) mean[j] += r[j]; }
        for (long long j = 0; j < DIN; j++) mean[j] /= (double)na;
        for (long long i = 0; i < na; i++) { const double *r = Xa + (size_t)i * DIN; double *d = Y + (size_t)i * DIN; for (long long j = 0; j < DIN; j++) d[j] = r[j] - mean[j]; }
        free(mean);
        double *Vt = (double *)xmalloc((size_t)(2 * KMAX) * DIN * 8);
        top_right_singular(Y, (int)na, DIN, 2 * KMAX, Vt, NULL);
        free(Y);
        /* 基一律按【行=基向量】存(= Vt 的自然布局), 投影走 NT gemm。 */
        V0[0] = (double *)xmalloc((size_t)KMAX * DIN * 8);
        A0[0] = (double *)xmalloc((size_t)KMAX * DIN * 8);
        memcpy(V0[0], Vt, (size_t)KMAX * DIN * 8);                          /* py:83 Vt[:KMAX].T */
        memcpy(A0[0], Vt + (size_t)KMAX * DIN, (size_t)KMAX * DIN * 8);     /* py:94 Vt[KMAX:2K].T */
        free(Vt);
    }
    /* py:83/95  rand = RandomState(7).randn(DIN,KMAX)/√DIN, rnd2 = RandomState(11) 同式。
     * numpy 按 C 序逐元素抽(先走完一行 KMAX 个再下一行), 这里转置着存, 抽取顺序不变。 */
    {
        double inv = 1.0 / sqrt((double)DIN);
        const uint32_t seeds[2] = {7, 11};
        double *dst[2];
        V0[1] = (double *)xmalloc((size_t)KMAX * DIN * 8);
        A0[1] = (double *)xmalloc((size_t)KMAX * DIN * 8);
        dst[0] = V0[1]; dst[1] = A0[1];
        for (int b = 0; b < 2; b++) {
            mt_t r; mt_seed(&r, seeds[b]);
            for (long long i = 0; i < DIN; i++)
                for (int c = 0; c < KMAX; c++) dst[b][(size_t)c * DIN + i] = mt_gauss(&r) * inv;
        }
    }
    double scale = sqrt(pw_sq_f64(Xa, na * DIN) / (double)(na * DIN));       /* py:84 */

    /* py:99-105 的四个投影(每个 (nm,gn) 组合里 .py 都重算一遍, 值完全一样) 提到循环外:
     * 纯粹省 4× 矩阵乘, 每个元素的操作数与舍入一模一样, 不是数值改动。 */
    fprintf(stderr, "[amp_solve_zc] L%d 投影 tanh(X·V/s)…\n", L);
    double *TVa[2], *TAa[2], *TVe[2], *TAe[2];
    for (int b = 0; b < 2; b++) {
        TVa[b] = (double *)xmalloc((size_t)na * KMAX * 8);
        TAa[b] = (double *)xmalloc((size_t)na * KMAX * 8);
        TVe[b] = (double *)xmalloc((size_t)NEV * KMAX * 8);
        TAe[b] = (double *)xmalloc((size_t)NEV * KMAX * 8);
        mm_nt((int)na, KMAX, (int)DIN, Xa, (int)DIN, V0[b], (int)DIN, TVa[b], KMAX);
        mm_nt((int)na, KMAX, (int)DIN, Xa, (int)DIN, A0[b], (int)DIN, TAa[b], KMAX);
        mm_nt((int)NEV, KMAX, (int)DIN, Xev, (int)DIN, V0[b], (int)DIN, TVe[b], KMAX);
        mm_nt((int)NEV, KMAX, (int)DIN, Xev, (int)DIN, A0[b], (int)DIN, TAe[b], KMAX);
        for (long long i = 0; i < na * KMAX; i++) { TVa[b][i] = tanh(TVa[b][i] / scale); TAa[b][i] = tanh(TAa[b][i] / scale); }
        for (long long i = 0; i < NEV * KMAX; i++) { TVe[b][i] = tanh(TVe[b][i] / scale); TAe[b][i] = tanh(TAe[b][i] / scale); }
    }
    free(Xa); free(Xev);

    /* py:97-117  held 网格: V₀ × 门 × λ × k, 判据 = held 行为挽回率, 严格 > 才换庄(平局保先) */
    static const char *CAND_NM[2] = {"PCA", "rand"};
    static const char *GATE_NM[2] = {"pca2", "rnd2"};
    static const double LAMS[7] = {1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0};
    static const int KS[9] = {16, 32, 64, 128, 256, 384, 512, 768, 1024};

    double *Za = (double *)xmalloc((size_t)na * KMAX * 8);
    double *ZaT = (double *)xmalloc((size_t)KMAX * na * 8);
    double *Ze = (double *)xmalloc((size_t)NEV * KMAX * 8);
    double *ZtZ = (double *)xmalloc((size_t)KMAX * KMAX * 8);
    double *G = (double *)xmalloc((size_t)KMAX * KMAX * 8);
    double *ZtR = (double *)xmalloc((size_t)KMAX * D * 8);
    double *ZtRT = (double *)xmalloc((size_t)D * KMAX * 8);
    double *UT = (double *)xmalloc((size_t)D * KMAX * 8);
    double *Uc = (double *)xmalloc((size_t)D * KMAX * 8);
    double *gd = (double *)xmalloc((size_t)NEV * D * 8);
    double *bestU = (double *)xmalloc((size_t)D * KMAX * 8);

    double best_rec = 0.0;                                            /* py:97 best = (0.0, None) */
    int have_best = 0, best_nm = 0, best_gn = 0, best_k = 0;
    double best_lam = 0.0;
    int combo = 0;

    for (int nm = 0; nm < 2; nm++) {
      for (int gn = 0; gn < 2; gn++) {
        combo++;
        for (long long i = 0; i < na * KMAX; i++) Za[i] = TVa[nm][i] * TAa[gn][i];
        for (long long i = 0; i < NEV * KMAX; i++) Ze[i] = TVe[nm][i] * TAe[gn][i];
        for (long long i = 0; i < na; i++)
            for (int c = 0; c < KMAX; c++) ZaT[(size_t)c * na + i] = Za[(size_t)i * KMAX + c];
        mm_nt(KMAX, KMAX, (int)na, ZaT, (int)na, ZaT, (int)na, ZtZ, KMAX);      /* Za.T @ Za  */
        mm_nn(KMAX, (int)D, (int)na, ZaT, (int)na, Ra, (int)D, ZtR, (int)D);    /* Za.T @ Ra_ */
        double trZ = 0.0;                                             /* xp.trace: 跨步视图 ⇒ 朴素求和 */
        for (int c = 0; c < KMAX; c++) trZ += ZtZ[(size_t)c * KMAX + c];
        for (long long j = 0; j < D; j++)
            for (int c = 0; c < KMAX; c++) ZtRT[(size_t)j * KMAX + c] = ZtR[(size_t)c * D + j];

        for (int li = 0; li < 7; li++) {
            double lam = LAMS[li];
            memcpy(G, ZtZ, (size_t)KMAX * KMAX * 8);                  /* py:110 G = ZtZ.copy() */
            double ridge = lam * trZ / (double)KMAX + 1e-10;          /* py:111 */
            for (int c = 0; c < KMAX; c++) G[(size_t)c * KMAX + c] += ridge;
            memcpy(UT, ZtRT, (size_t)D * KMAX * 8);
            lu_solve_T(G, KMAX, UT, (int)D, KMAX);                    /* py:112 np.linalg.solve */
            for (long long j = 0; j < D; j++) {
                double iw = colw[j];
                const double *u = UT + (size_t)j * KMAX;
                double *o = Uc + (size_t)j * KMAX;
                for (int c = 0; c < KMAX; c++) o[c] = u[c] / iw;      /* py:114 U[:k]/colw */
            }
            for (int ki = 0; ki < 9; ki++) {
                int k = KS[ki];
                mm_nt((int)NEV, (int)D, k, Ze, KMAX, Uc, KMAX, gd, (int)D);
                /* py:115-116  yh = yq_ev*(1+g.astype(f32)); rec = 1 − Σ(yfp−yh)²/e0 */
                double rec = 1.0 - pw_sqres(yfp_ev, yq_ev, gd, NEV * D) / e0;
                if (rec > best_rec) {
                    best_rec = rec; have_best = 1;
                    best_nm = nm; best_gn = gn; best_lam = lam; best_k = k;
                    memcpy(bestU, Uc, (size_t)D * KMAX * 8);
                }
            }
        }
        fprintf(stderr, "[amp_solve_zc] L%d 网格 %d/4 (%s×%s) 完, 当前最好 %.2f%% @%s/%s λ=%.1f k=%d\n",
                L, combo, CAND_NM[nm], GATE_NM[gn], best_rec * 100.0,
                have_best ? CAND_NM[best_nm] : "-", have_best ? GATE_NM[best_gn] : "-",
                best_lam, best_k);
    }
    }

    /* py:119-138  产物 */
    FILE *of = fopen(outp, "wb");
    if (!of) die("输出打不开: %s", outp);
    uint8_t hdr[116];
    if (best_rec <= 0.005 || !have_best) {                            /* py:124 */
        /* 116B 零载荷 "zl.AMP" 头 = 终态"层闸"标记(psz<16 合并器不入链); 空文件保留给
         * "中断未跑", zside 断点续跑据此区分重解。注意名字是 zl.AMP 不是 zl.AMPD —— .py
         * 两个分支写的名字不同, 照抄不统一。 */
        put_hdr(hdr, "zl.AMP", 0);
        if (fwrite(hdr, 1, 116, of) != 116) die("写出失败");
        fclose(of);
        printf("★L%d 放大器: held %.2f%% ≤0.5%% → 层闸(116B 终态标记) | %.0fs\n",
               L, best_rec * 100.0, now_s() - t0);
        fflush(stdout);
        return 0;
    }
    {
        int k = best_k;
        size_t nAV = (size_t)DIN * k, nU = (size_t)D * k;
        uint16_t *Aeng = (uint16_t *)xmalloc(nAV * 2);
        uint16_t *Ueng = (uint16_t *)xmalloc(nU * 2);
        uint16_t *Veng = (uint16_t *)xmalloc(nAV * 2);
        /* Ueng = (U/colw).T → D×k, 引擎读 hU[j*k+c] */
        for (long long j = 0; j < D; j++)
            for (int c = 0; c < k; c++) Ueng[(size_t)j * k + c] = f64_to_f16(bestU[(size_t)j * KMAX + c]);
        /* Veng/Aeng = 基的前 k 列 → DIN×k, 引擎读 hV[i*k+c]。基在内存里按行存(行=基向量),
         * 所以这里是一次转置。 */
        for (long long i = 0; i < DIN; i++)
            for (int c = 0; c < k; c++) {
                Veng[(size_t)i * k + c] = f64_to_f16(V0[best_nm][(size_t)c * DIN + i]);
                Aeng[(size_t)i * k + c] = f64_to_f16(A0[best_gn][(size_t)c * DIN + i]);
            }
        /* py:134  pay = pack('<IfII', k, scale, DIN, D) + A + U + V */
        uint8_t head[16];
        uint32_t u32; float f32s = (float)scale;
        u32 = (uint32_t)k;   memcpy(head + 0, &u32, 4);
        memcpy(head + 4, &f32s, 4);
        u32 = (uint32_t)DIN; memcpy(head + 8, &u32, 4);
        u32 = (uint32_t)D;   memcpy(head + 12, &u32, 4);
        uint64_t psz = 16 + (uint64_t)(nAV + nU + nAV) * 2;
        put_hdr(hdr, "zl.AMPD", psz);                                 /* py:135 type9 动态 z */
        if (fwrite(hdr, 1, 116, of) != 116 ||
            fwrite(head, 1, 16, of) != 16 ||
            fwrite(Aeng, 2, nAV, of) != nAV ||
            fwrite(Ueng, 2, nU, of) != nU ||
            fwrite(Veng, 2, nAV, of) != nAV) die("写出失败: %s", outp);
        fclose(of);
        printf("★L%d 放大器: held行为挽回 %.2f%% @V₀=%s 门=%s λ=%.1f k_L=%d 体积 %.1fMB | %.0fs\n",
               L, best_rec * 100.0, CAND_NM[best_nm], GATE_NM[best_gn], best_lam, k,
               (double)psz / 1048576.0, now_s() - t0);
        fflush(stdout);
    }
    return 0;
}
