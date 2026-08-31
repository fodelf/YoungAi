/* zlayer.c — 每层 z 侧车一体化(建缓存→四损失闭式解→注入 dql, 反修段主力)。
 *
 * 【转录来源】zlever/zlayer.py(672 行, 全仓 Python 清零已删, 见 git 历史)逐式转录, 2026-08-25。
 *   连带转录它 import 的两个函数: scripts/probe_layer_behavior.py 的 anchor_layer/swiglu,
 *   scripts/probe_behavior_spectrum.py 的 st_index/st_mxfp4/vq_slot/vq_dequant/st_raw/FP4T。
 *   MXFP4(HF I8 容器 + E8M0 scale) 的读取直接复用 quant/st_read.c 的 st_read_weight,
 *   不重写解码(与量化器/引擎同一份实现)。
 *
 * 【逐式对照声明】数值语义一律以 .py 为准, 不"改进":
 *   - swiglu 用 .py 的【双侧】clip(g,u 都夹 ±lim), 不是 ds4quant_fwd.c dq_expert_fp 的
 *     单侧 gate clip —— 这是已知分歧点, 见下方 zl_swiglu 注释。
 *   - 路由权重 w 在 w2 矩阵乘【之后】才乘上去(py: dH[rows] += w[:,None]*Yf), 不是
 *     dq_expert_fp 那样先折进 h —— 舍入位置不同, 所以本文件自带 zl_expert_fwd。
 *   - f32/f64 的每一次 astype 都照抄: ftA 支线的 Gram 在 f32 上算(py: Fa 是 float32,
 *     Fa@Fa.T 走 sgemm)然后才升 f64 进 solve; 线性支线的 Gram 全程 f64。
 *   - dither 用 numpy legacy RandomState(1).randn 的【逐位复刻】(MT19937 + legacy gauss),
 *     自检: ./zlayer --selftest-rng 打印前 8 个值, 应等于 np.random.RandomState(1).randn(8)。
 *   - fp16 载荷由 f64 直接舍入(py: A[:,:K].astype(np.float16), A 是 f64), 不走 f32 中转。
 *
 * 【第二期补齐(2026-08-25)】一期只做 XCAP 口径, 二期把 .py 剩下的四个模式全转录进来。
 *   ① 非 XCAP 口径: x=锚 fin, 学生=本进程重算量化专家(VQ blob 或 GGUF 切片),
 *      dH = Σw·Y_fp − Σw_q·Y_q。amp86/dspark86 战役走的就是这一路(见 scripts/amp86_spark.sh)。
 *   ② --gguf 标量模式: 学生权重从 GGUF 专家张量按字节均分切片再 dequant。
 *   ③ --xanchor / --addon 链模式: 链态锚(x_q/路由_q 从第二个锚读) +
 *      叠加式合并注入(既有 GE 乘入学生权重、既有 z 出力从 dH 扣除、新旧合并成单条记录)。
 *      ★照抄 .py 的缩进事实★: ADDON 的读取整块嵌在 `if XAP:` 里 —— 也就是说
 *      【只给 --addon 而不给 --xanchor 时, ADDON 完全不生效】。这不是笔误顺手
 *      改掉的地方: .py 是权威, 改了就等于 C 与 py 同参数下产物不同。C 里额外加一行提示,
 *      免得有人设了 ADDON 却以为生效了(提示不改行为)。
 *   ④ ERF 死层部件: 组合增益 < --erf-bar 时上的每专家 ΔW_w2 加权低秩补丁。
 *   仍然【不做】: cupy/GPU 路径(纯 CPU + 可选 BLAS)。
 *
 * 【二期新增的不可逐位点】(判定看打印读数与结构, 不看载荷字节):
 *   - ERF 的 randomized SVD 里有 QR 分解, ADDON 的合并里有两次 QR + 一次 SVD。
 *     numpy 的 np.linalg.qr 是 LAPACK geqrf/orgqr(Householder), 这里手写 Householder,
 *     两者的 Q/R 符号约定与舍入都不逐位。但 QR 只是中间基:
 *       ADDON: 最终产物 = Pc·Qcᵀ 的截断 SVD, 与用哪组 QR 基无关(数学上唯一, 差在末位);
 *       ERF:   最终产物 = 低秩补丁的乘积, 同理。
 *     所以【金标口径 = 打印的挽回率/专家数/记录长度 + 记录头逐字节】, U/V 载荷不逐位。
 *   - ERF 里 np.argsort(-_en) 是 numpy 默认 quicksort(不稳定); C 用稳定排序(键相同按
 *     下标升序)。_en 是连续浮点能量, 实测不会撞值; 真撞了两边的 tau 也一样(同值)。
 *
 * 【金标口径】SVD 的数值路径与 numpy 不同(见 zl_svd_lowrank): numpy 走 LAPACK gesdd,
 *   这里走子空间迭代 + Rayleigh-Ritz。子空间迭代对 rank-k 截断是逼近而非精确, 所以
 *   【不能逐位对拍 .py】。判定口径 = ①打印出来的 held 挽回率(0.1% 精度)对齐
 *   ②产物用 calib/rec_fidelity 复评(载荷重放挽回率)对齐。
 *
 * 用法(位置参数与 .py 一致; 原 DS4_ZL_* env 已随 2026-08-31 env 大扫除改为 --flag,
 * 值语义未动, 完整面板见 p3 用法打印; K 与 --ntok 必传, 静默默认已删):
 *   zlayer <hf> <layers_dir> <anchor> <L> <K> [inject=1] [XCAP目录] [PREV目录] --ntok N [...]
 *   zlayer --selftest-rng
 * 编译:
 *   gcc -O3 -march=native -o zlayer zlayer.c -lm -lpthread              (纯循环, 慢但能跑)
 *   gcc -O3 -march=native -DDQ_BLAS -o zlayer zlayer.c -lm -lpthread -framework Accelerate
 *   gcc -O3 -march=native -DDQ_BLAS -o zlayer zlayer.c -lm -lpthread -lopenblas   (Linux)
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
#include "../../src/common/ds4_amp_fmt.h"   /* 116 记录头/秩上限/fit切分: 与 ds4quant_run/引擎同一契约 */
#include <pthread.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/file.h>

/* macOS 没有 posix_fadvise: st_read.c 的页缓存旁路是 Linux 专属的 IO 优化(读完丢缓存),
 * Mac 上退化成 no-op —— 只影响读吞吐, 不碰任何数值。放在 include 之前, 免改 st_read.c。 */
#if defined(__APPLE__)
#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif
static int posix_fadvise(int fd, off_t off, off_t len, int adv) {
    (void)fd; (void)off; (void)len; (void)adv; return 0;
}
#endif

#include "../calib/st_read.c"   /* st_ctx / st_open / st_read_weight: HF MXFP4 读取 */

#ifdef DQ_BLAS
#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define D 4096                 /* py: D=4096(写死) */
#define NEXP 256               /* GE 门的专家数(py: Gg 是 256×256) */

/* ---------------- 通用小助手 ---------------- */

static void die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static void die(const char *fmt, ...) {
    /* 战役脚本按 grep -aE "XCAP|Error|Traceback|assert|★" 过滤输出 —— 致命信息必须带
     * 其中一个词, 否则失败在日志里是隐形的。 */
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "zlayer Error: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap); exit(2);
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int n_threads(void) {
    static int n = 0;
    if (!n) { long c = sysconf(_SC_NPROCESSORS_ONLN); n = (int)(c > 0 ? c : 1); if (n > 32) n = 32; }
    return n;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("malloc %zu 失败", n);
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s);
    if (!p) die("calloc %zu×%zu 失败", n, s);
    return p;
}

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

/* ---------------- 半精度转换 ---------------- */

static float f16_to_f32(uint16_t h) {
    int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = (float)ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = (float)ldexp((double)(m + 1024), e - 25);
    return s ? -v : v;
}

/* f64 → f16, round-to-nearest-even(与 numpy .astype(np.float16) 同).
 * 直接从 double 舍入: py 里 A/S/Bt/ge 都是 f64, 走 f32 中转会双重舍入。 */
static uint16_t f64_to_f16(double d) {
    uint64_t x; memcpy(&x, &d, 8);
    uint32_t sign = (uint32_t)((x >> 48) & 0x8000u);
    int e64 = (int)((x >> 52) & 0x7FF);
    uint64_t man = x & 0xFFFFFFFFFFFFFULL;              /* 52 位尾数 */
    if (e64 == 0x7FF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (e64 == 0) return (uint16_t)sign;                 /* f64 次正规 → f16 必为 ±0 */
    int e = e64 - 1023 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {                                        /* f16 次正规: m = (2^52+man)/2^(43-e) */
        if (e < -10) return (uint16_t)sign;
        int sh = 43 - e;                                 /* 43..53 */
        uint64_t full = man | (1ULL << 52);
        uint64_t m = full >> sh, rem = full & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
        if (rem > half || (rem == half && (m & 1))) m++;
        return (uint16_t)(sign | (uint32_t)m);
    }
    uint32_t h = ((uint32_t)e << 10) | (uint32_t)(man >> 42);
    uint64_t rem = man & ((1ULL << 42) - 1), half = 1ULL << 41;
    if (rem > half || (rem == half && (h & 1))) h++;     /* 进位可自然溢进指数位, 正确 */
    return (uint16_t)(sign | h);
}

/* ---------------- numpy legacy MT19937 + legacy gauss ----------------
 * np.random.RandomState(1).randn(...) 的逐位复刻。三处必须一模一样:
 *   ① 播种 = numpy 的 mt19937_seed(等价标准 init_genrand), 不是 init_by_array;
 *   ② 双精度 = (next>>5)*2^26 + (next>>6) 再除 2^53(两次抽 32 位);
 *   ③ 高斯 = Marsaglia polar, 【先返回 f*x2, 把 f*x1 存进缓存】(numpy legacy_gauss 的顺序)。
 * 任何一处写反, dither 增广行就整体换了一套 —— 而且不会报错, 只是解出来的 z 悄悄不一样。 */
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

/* ---------------- 矩阵乘(BLAS 可选, 无 BLAS 走并行三重循环) ----------------
 * ta/tb: 0=NoTrans 1=Trans。C = op(A)·op(B), alpha=1 beta=0, 行主序。
 * 无 BLAS 路的求和顺序与 BLAS 不同(BLAS 分块/向量化), f64 尾位会差 —— 两种构建之间
 * 不可逐位对拍, 但都在打印精度内。 */
typedef struct { int ta, tb, M, N, K, lda, ldb, ldc; const double *A, *B; double *C; } mm64_ctx;
static void mm64_worker(void *vc, int i0, int i1) {
    mm64_ctx *c = (mm64_ctx *)vc;
    for (int i = i0; i < i1; i++) {
        double *cr = c->C + (size_t)i * c->ldc;
        for (int j = 0; j < c->N; j++) cr[j] = 0.0;
        for (int k = 0; k < c->K; k++) {
            double a = c->ta ? c->A[(size_t)k * c->lda + i] : c->A[(size_t)i * c->lda + k];
            if (a == 0.0) continue;
            if (c->tb) {
                for (int j = 0; j < c->N; j++) cr[j] += a * c->B[(size_t)j * c->ldb + k]; }
            else { const double *br = c->B + (size_t)k * c->ldb;
                for (int j = 0; j < c->N; j++) cr[j] += a * br[j]; }
        }
    }
}
#ifdef ZL_CUDA
/* CUDA 卸载(zlayer_gpu.cu): 大 gemm 走 cuBLAS, 小的/GPU 不可用回落 CPU。
 * 契约=py 自身 cupy/numpy 双路(能力探测, 数值同语义求和序容差); 阈值只挑真热点。 */
extern int zg_dgemm(int, int, int, int, int, const double *, int, const double *, int, double *, int);
extern int zg_sgemm(int, int, int, int, int, const float *, int, const float *, int, float *, int);
#define ZG_MIN_FLOPS 5.0e8
static long zg_hits=0, zg_miss=0;
#endif
static void mm64(int ta, int tb, int M, int N, int K, const double *A, int lda,
                 const double *B, int ldb, double *C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) { for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[(size_t)i * ldc + j] = 0.0; return; }
#ifdef ZL_CUDA
    if ((double)M * N * K >= ZG_MIN_FLOPS) { if (zg_dgemm(ta, tb, M, N, K, A, lda, B, ldb, C, ldc)) { zg_hits++; return; } zg_miss++; }
#endif
#ifdef DQ_BLAS
    cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
#else
    mm64_ctx c = {ta, tb, M, N, K, lda, ldb, ldc, A, B, C};
    parallel_for(M, mm64_worker, &c);
#endif
}

typedef struct { int ta, tb, M, N, K, lda, ldb, ldc; const float *A, *B; float *C; } mm32_ctx;
static void mm32_worker(void *vc, int i0, int i1) {
    mm32_ctx *c = (mm32_ctx *)vc;
    for (int i = i0; i < i1; i++) {
        float *cr = c->C + (size_t)i * c->ldc;
        if (c->tb) {                       /* NT: 逐个点积, 与 sgemm 同语义 */
            for (int j = 0; j < c->N; j++) {
                const float *br = c->B + (size_t)j * c->ldb;
                const float *ar = c->A + (size_t)i * c->lda;
                float acc = 0.0f;
                for (int k = 0; k < c->K; k++) acc += ar[k] * br[k];
                cr[j] = acc;
            }
            continue;
        }
        for (int j = 0; j < c->N; j++) cr[j] = 0.0f;
        for (int k = 0; k < c->K; k++) {
            float a = c->ta ? c->A[(size_t)k * c->lda + i] : c->A[(size_t)i * c->lda + k];
            if (a == 0.0f) continue;
            const float *br = c->B + (size_t)k * c->ldb;
            for (int j = 0; j < c->N; j++) cr[j] += a * br[j];
        }
    }
}
static void mm32(int ta, int tb, int M, int N, int K, const float *A, int lda,
                 const float *B, int ldb, float *C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) { for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[(size_t)i * ldc + j] = 0.0f; return; }
#ifdef ZL_CUDA
    if ((double)M * N * K >= ZG_MIN_FLOPS) { if (zg_sgemm(ta, tb, M, N, K, A, lda, B, ldb, C, ldc)) { zg_hits++; return; } zg_miss++; }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
#else
    mm32_ctx c = {ta, tb, M, N, K, lda, ldb, ldc, A, B, C};
    parallel_for(M, mm32_worker, &c);
#endif
}

/* ---------------- Cholesky 解(对偶 ridge 的 np.linalg.solve 位) ----------------
 * py 用 np.linalg.solve(LU 部分主元), 这里用 Cholesky —— G 是 XaXaᵀ+ridge·I, 对称正定,
 * 两者数学等价, 尾位不同。分解单线程, 回代按 rhs 列并行(大头)。 */
typedef struct { const double *L; double *B; int n, nrhs; } tri_ctx;
static void tri_worker(void *vc, int c0, int c1) {
    tri_ctx *t = (tri_ctx *)vc;
    const double *A = t->L; double *B = t->B;
    const int n = t->n, nrhs = t->nrhs;
    for (int c = c0; c < c1; c++) {
        for (int i = 0; i < n; i++) {
            double s = B[(size_t)i * nrhs + c];
            const double *Ai = A + (size_t)i * n;
            for (int k = 0; k < i; k++) s -= Ai[k] * B[(size_t)k * nrhs + c];
            B[(size_t)i * nrhs + c] = s / Ai[i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = B[(size_t)i * nrhs + c];
            for (int k = i + 1; k < n; k++) s -= A[(size_t)k * n + i] * B[(size_t)k * nrhs + c];
            B[(size_t)i * nrhs + c] = s / A[(size_t)i * n + i];
        }
    }
}
#ifdef DQ_BLAS
/* LAPACK 分块 potrf/potrs(openblas/Accelerate 自带): 与下方手写 Cholesky 同数学,
 * 尾位不同=既有"Cholesky vs numpy LU"预算类。gprof 实锤: 手写回代双跨步访存占 95%
 * (3105s/解), 分块版是同规模 numpy 33s 的实现口径。
 * 行主序对称阵以列主序视角取 uplo='U' 等价; B 行主序[n×nrhs]需转置进出。 */
#if defined(__APPLE__)
extern void dpotrf_(const char *, const int *, double *, const int *, int *);
extern void dpotrs_(const char *, const int *, const int *, const double *, const int *, double *, const int *, int *);
#define ZL_DPOTRF dpotrf_
#define ZL_DPOTRS dpotrs_
#else
extern void scipy_dpotrf_(const char *, const int *, double *, const int *, int *);
extern void scipy_dpotrs_(const char *, const int *, const int *, const double *, const int *, double *, const int *, int *);
#define ZL_DPOTRF scipy_dpotrf_
#define ZL_DPOTRS scipy_dpotrs_
#endif
static int chol_solve(double *A, int n, double *B, int nrhs) {
    int info = 0; const char up = 'U';
    ZL_DPOTRF(&up, &n, A, &n, &info);
    if (info) return -1;
    double *Bt = (double *)xmalloc((size_t)n * nrhs * 8);
    for (int i = 0; i < n; i++)   /* rm[n×nrhs] → cm(=rm 的转置) */
        for (int c = 0; c < nrhs; c++) Bt[(size_t)c * n + i] = B[(size_t)i * nrhs + c];
    ZL_DPOTRS(&up, &n, &nrhs, A, &n, Bt, &n, &info);
    if (info) { free(Bt); return -1; }
    for (int i = 0; i < n; i++)
        for (int c = 0; c < nrhs; c++) B[(size_t)i * nrhs + c] = Bt[(size_t)c * n + i];
    free(Bt);
    return 0;
}
#else
static int chol_solve(double *A, int n, double *B, int nrhs) {
    for (int j = 0; j < n; j++) {
        double d = A[(size_t)j * n + j];
        for (int k = 0; k < j; k++) d -= A[(size_t)j * n + k] * A[(size_t)j * n + k];
        if (d <= 0) return -1;
        d = sqrt(d); A[(size_t)j * n + j] = d;
        for (int i = j + 1; i < n; i++) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; k++) s -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
            A[(size_t)i * n + j] = s / d;
        }
    }
    tri_ctx tc = {A, B, n, nrhs};
    parallel_for(nrhs, tri_worker, &tc);
    return 0;
}
#endif

/* 一般方阵 LU + 部分主元(GE 的 np.linalg.solve 位: Gg 只有 256×256, 直接照 LAPACK 语义走) */
static int lu_solve(double *A, int n, double *b) {
    int *piv = (int *)xmalloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) piv[i] = i;
    for (int k = 0; k < n; k++) {
        int p = k; double mx = fabs(A[(size_t)k * n + k]);
        for (int i = k + 1; i < n; i++) { double v = fabs(A[(size_t)i * n + k]); if (v > mx) { mx = v; p = i; } }
        if (mx == 0.0) { free(piv); return -1; }
        if (p != k) {
            for (int j = 0; j < n; j++) { double t = A[(size_t)k * n + j]; A[(size_t)k * n + j] = A[(size_t)p * n + j]; A[(size_t)p * n + j] = t; }
            double t = b[k]; b[k] = b[p]; b[p] = t;
        }
        double d = A[(size_t)k * n + k];
        for (int i = k + 1; i < n; i++) {
            double f = A[(size_t)i * n + k] / d;
            if (f == 0.0) continue;
            A[(size_t)i * n + k] = f;
            for (int j = k + 1; j < n; j++) A[(size_t)i * n + j] -= f * A[(size_t)k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int j = i + 1; j < n; j++) s -= A[(size_t)i * n + j] * b[j];
        b[i] = s / A[(size_t)i * n + i];
    }
    free(piv);
    return 0;
}

/* ★非 ADDON 注入前的底座对账(2026-08-31)★ 非 ADDON 解算把学生当"裸专家加权和" ——
 * zcache 既不读也不建模底座 dql 已落地的修正链 op; 回放却会先应用它们再叠新 z:
 * λ 族(bf.GL/GLdyn/GLhc/TREF)缩放同一块 routed, 而新 z 按"填满 teacher−裸学生全缺口"
 * 解, 重叠部分被修两遍; 既有 zl.RRR/zl.ERF 同理(回放逐条应用, 无末条胜出语义)。
 * bf.GE 例外: 回放只取末条=替换语义, 新解相对裸基自洽。发现未建模 op 即停车, 由人
 * 决定剥离(champ_reset 回纯净量化态)还是走 --addon, 不许静默叠加出双重修正。 */
static void zl_dql_guard(const char *ld, int L) {
    char p[1200]; snprintf(p, sizeof p, "%s/dql_L%02d.bin", ld, L);
    FILE *f = fopen(p, "rb");
    if (!f) return;                              /* dql 缺失由注入路自己报错 */
    uint8_t hd[12];
    if (fread(hd, 1, 12, f) != 12 || memcmp(hd, "DQL2", 4)) { fclose(f); return; }
    uint32_t nr; memcpy(&nr, hd + 8, 4);
    char bad[256] = ""; int nbad = 0;
    for (uint32_t i = 0; i < nr; i++) {
        uint8_t rh[DS4_AMP_REC_HDR];
        if (fread(rh, 1, DS4_AMP_REC_HDR, f) != DS4_AMP_REC_HDR) break;
        char nm[17]; memcpy(nm, rh, 16); nm[16] = 0;
        uint64_t psz; memcpy(&psz, rh + DS4_AMP_REC_OFF_PSZ, 8);
        int32_t vd; memcpy(&vd, rh + DS4_AMP_REC_OFF_VD, 4);
        if (fseeko(f, (off_t)psz, SEEK_CUR)) break;
        if (vd != 1) continue;
        /* 活 op 名单镜像 dsq_lfile parse_op_rec(bf.GE 之外全算; RRR 空壳 psz<16 是占位) */
        const int live = strstr(nm, ".GL") || strstr(nm, "TREF") || strstr(nm, "xlayer")
                      || strstr(nm, "zl.ERF") || (strstr(nm, "zl.RRR") && psz >= 16);
        if (!live) continue;
        if (nbad < 4) snprintf(bad + strlen(bad), sizeof bad - strlen(bad),
                               "%s%.16s", nbad ? "," : "", nm);
        nbad++;
    }
    fclose(f);
    if (nbad)
        die("L%d 底座 dql 已带 %d 条落地修正 op(%s%s) — 非 ADDON 解算不建模既有链, "
            "注入=同一残差修两遍。先还原纯净量化态(champ_reset/layers_quant)或走 --addon",
            L, nbad, bad, nbad > 4 ? ",…" : "");
}

