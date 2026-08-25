/* zlayer.c — 每层 z 侧车一体化(建缓存→四损失闭式解→注入 dql, 反修段主力)。
 *
 * 【转录来源】gguf-tools/go-onebit/zlever/zlayer.py (672 行) 逐式转录, 2026-08-25。
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
 * 【第一期范围】能编译能跑的最小完整闭环。明确【不做】(碰到就报错停车, 不静默换语义):
 *   - DS4_ZL_GGUF 标量模式(量化侧权重从 GGUF 切片) —— 未实现, 设了就拒跑。
 *   - DS4_ZL_XANCHOR / DS4_ZL_ADDON 链模式(链态锚 + 叠加式合并注入) —— 未实现, 拒跑。
 *   - ERF 死层部件 —— 未实现, 触发条件命中时打印一行"未实现, 跳过"继续走。
 *   - cupy/GPU 路径 —— 纯 CPU(+可选 BLAS)。
 *   - 非 XCAP 口径(x/y_q 走锚 fin + Python 重算量化专家) —— 第一期只做 XCAP。
 *
 * 【金标口径】SVD 的数值路径与 numpy 不同(见 zl_svd_lowrank): numpy 走 LAPACK gesdd,
 *   这里走子空间迭代 + Rayleigh-Ritz。子空间迭代对 rank-k 截断是逼近而非精确, 所以
 *   【不能逐位对拍 .py】。判定口径 = ①打印出来的 held 挽回率(0.1% 精度)对齐
 *   ②产物用 calib/rec_fidelity 复评(载荷重放挽回率)对齐。
 *
 * 用法(与 .py 完全一致):
 *   zlayer <hf> <layers_dir> <anchor> <L> [K=1024] [inject=1] [XCAP目录] [PREV目录]
 *   zlayer --selftest-rng
 * 环境变量(只认既有的这几个, 禁新增):
 *   DS4_ZL_NTOK=1716 DS4_ZL_NFIT=1287 DS4_ZL_FIT_RANGES DS4_ZL_EV_RANGE
 *   DS4_ZL_GE=1 DS4_ZL_GE_LAM=1e-3 DS4_ZL_FTA=1 DS4_ZL_ERF=1 DS4_ZL_GATE=0 DS4_ZL_SWLIM=10
 *   DS4_ZL_CACHE_ONLY=0(纯控制流, 不参与数值)
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

#include "../quant/st_read.c"   /* st_ctx / st_open / st_read_weight: HF MXFP4 读取 */

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
static void mm64(int ta, int tb, int M, int N, int K, const double *A, int lda,
                 const double *B, int ldb, double *C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) { for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[(size_t)i * ldc + j] = 0.0; return; }
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

/* ---------------- 对称特征分解: 循环 Jacobi(只用在 r×r 的小核上, r≤KMAX+64) ----------------
 * 出参: 特征值降序 w[r], 特征向量按列存 E[r][r](E[i*r+c] = 第 c 个向量的第 i 分量)。 */
static void jacobi_eig(double *A, int n, double *w, double *E) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) E[(size_t)i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
        double diag = 0.0;
        for (int p = 0; p < n; p++) diag += A[(size_t)p * n + p] * A[(size_t)p * n + p];
        if (off <= 1e-30 * (diag + 1e-300)) break;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = A[(size_t)p * n + q];
                if (fabs(apq) < 1e-300) continue;
                double app = A[(size_t)p * n + p], aqq = A[(size_t)q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; k++) {
                    double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq;
                    A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk;
                    A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {
                    double ekp = E[(size_t)k * n + p], ekq = E[(size_t)k * n + q];
                    E[(size_t)k * n + p] = c * ekp - s * ekq;
                    E[(size_t)k * n + q] = s * ekp + c * ekq;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) w[i] = A[(size_t)i * n + i];
    for (int a = 0; a < n; a++) {                     /* 降序(选择排序, n≤1088) */
        int best = a;
        for (int b = a + 1; b < n; b++) if (w[b] > w[best]) best = b;
        if (best == a) continue;
        double t = w[a]; w[a] = w[best]; w[best] = t;
        for (int k = 0; k < n; k++) { double e = E[(size_t)k * n + a]; E[(size_t)k * n + a] = E[(size_t)k * n + best]; E[(size_t)k * n + best] = e; }
    }
}

/* ---------------- 最小 npz(ZIP_STORED) 读写 ----------------
 * zcache_LXX.npz 是 np.savez 产物, 也要能被 amp_solve/rec_fidelity/各 probe 脚本读回去,
 * 所以读要吃 numpy 写的 ZIP64 局部头, 写要能被 python zipfile 校验通过(CRC32 必填)。
 * 读的部分与 calib/rec_fidelity.c 的解析器同源(那边全升 f64, 这里保留原 dtype 省内存)。 */
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void wr16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void wr32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

static uint32_t crc_tab[256];
static void crc_init(void) {
    if (crc_tab[1]) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_tab[i] = c;
    }
}
static uint32_t crc_upd(uint32_t crc, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crc_tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

typedef struct { const uint8_t *data; char dt[8]; long long d0, d1, nd; } npy_t;

/* 在 npz 映像里找 name.npy, 出参指向未压缩数据(零拷贝). 找不到返回 -1. */
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
/* 取字段并转成目标类型(数据在 npz 里可能非对齐, 一律 memcpy 逐元素搬) */
static void npz_to_f32(const npy_t *a, float *out, long long n) {
    if (!strcmp(a->dt, "<f4")) { memcpy(out, a->data, (size_t)n * 4); return; }
    if (!strcmp(a->dt, "<f8")) { for (long long i = 0; i < n; i++) { double v; memcpy(&v, a->data + i * 8, 8); out[i] = (float)v; } return; }
    if (!strcmp(a->dt, "<f2")) { for (long long i = 0; i < n; i++) { uint16_t v; memcpy(&v, a->data + i * 2, 2); out[i] = f16_to_f32(v); } return; }
    die("npz: dtype %s 不能转 f32", a->dt);
}
static void npz_to_i32(const npy_t *a, int *out, long long n) {
    if (!strcmp(a->dt, "<i4")) { for (long long i = 0; i < n; i++) { int32_t v; memcpy(&v, a->data + i * 4, 4); out[i] = v; } return; }
    if (!strcmp(a->dt, "<i8")) { for (long long i = 0; i < n; i++) { int64_t v; memcpy(&v, a->data + i * 8, 8); out[i] = (int)v; } return; }
    die("npz: dtype %s 不能转 i32", a->dt);
}

/* --- npz 写: 顺序追加 STORED 条目, 收尾写中央目录 --- */
typedef struct { char name[64]; uint32_t crc, size, off; } zent_t;
typedef struct { FILE *f; zent_t e[16]; int n; long long pos; } zwr_t;

static void npy_header(char *hdr, size_t cap, const char *dt, long long d0, long long d1, int nd, size_t *hlen) {
    char body[256];
    if (nd == 1) snprintf(body, sizeof body, "{'descr': '%s', 'fortran_order': False, 'shape': (%lld,), }", dt, d0);
    else snprintf(body, sizeof body, "{'descr': '%s', 'fortran_order': False, 'shape': (%lld, %lld), }", dt, d0, d1);
    size_t need = 10 + strlen(body) + 1;
    size_t pad = (64 - (need % 64)) % 64;
    size_t hl = strlen(body) + pad + 1;
    if (10 + hl > cap) die("npy 头太长");
    memcpy(hdr, "\x93NUMPY\x01\x00", 8);
    wr16((uint8_t *)hdr + 8, (uint32_t)hl);
    memcpy(hdr + 10, body, strlen(body));
    memset(hdr + 10 + strlen(body), ' ', pad);
    hdr[10 + hl - 1] = '\n';
    *hlen = 10 + hl;
}
/* data=NULL 时写 n 个零字节的数据体(pDY 是全零, 不必真开 169MB) */
static void zw_add(zwr_t *z, const char *name, const char *dt, long long d0, long long d1, int nd,
                   const void *data, size_t esz) {
    char hdr[256]; size_t hlen;
    npy_header(hdr, sizeof hdr, dt, d0, d1, nd, &hlen);
    size_t nel = (size_t)d0 * (size_t)(nd > 1 ? d1 : 1);
    size_t dsz = nel * esz, total = hlen + dsz;
    if (z->pos + (long long)total + 4096 > 0xF0000000LL)
        die("zcache 超过 4GB —— 本实现只写 ZIP32, 拒绝写出会被 numpy 读坏的文件");
    uint32_t crc = 0;
    crc = crc_upd(crc, hdr, hlen);
    if (data) crc = crc_upd(crc, data, dsz);
    else { static const uint8_t zbuf[65536] = {0}; size_t left = dsz;
           while (left) { size_t c = left > sizeof zbuf ? sizeof zbuf : left; crc = crc_upd(crc, zbuf, c); left -= c; } }
    char fn[64]; snprintf(fn, sizeof fn, "%s.npy", name);
    uint8_t lh[30] = {0};
    wr32(lh, 0x04034b50u); wr16(lh + 4, 20); wr16(lh + 6, 0); wr16(lh + 8, 0);
    wr16(lh + 10, 0); wr16(lh + 12, 0x21);          /* 固定时间戳: 产物字节可复现 */
    wr32(lh + 14, crc); wr32(lh + 18, (uint32_t)total); wr32(lh + 22, (uint32_t)total);
    wr16(lh + 26, (uint32_t)strlen(fn)); wr16(lh + 28, 0);
    zent_t *e = &z->e[z->n++];
    snprintf(e->name, sizeof e->name, "%s", fn);
    e->crc = crc; e->size = (uint32_t)total; e->off = (uint32_t)z->pos;
    fwrite(lh, 1, 30, z->f); fwrite(fn, 1, strlen(fn), z->f); fwrite(hdr, 1, hlen, z->f);
    if (data) fwrite(data, 1, dsz, z->f);
    else { static const uint8_t zbuf[65536] = {0}; size_t left = dsz;
           while (left) { size_t c = left > sizeof zbuf ? sizeof zbuf : left; fwrite(zbuf, 1, c, z->f); left -= c; } }
    z->pos += 30 + (long long)strlen(fn) + (long long)total;
}
static void zw_finish(zwr_t *z) {
    long long cd = z->pos;
    for (int i = 0; i < z->n; i++) {
        zent_t *e = &z->e[i];
        uint8_t ch[46] = {0};
        wr32(ch, 0x02014b50u); wr16(ch + 4, 20); wr16(ch + 6, 20); wr16(ch + 8, 0); wr16(ch + 10, 0);
        wr16(ch + 12, 0); wr16(ch + 14, 0x21);
        wr32(ch + 16, e->crc); wr32(ch + 20, e->size); wr32(ch + 24, e->size);
        wr16(ch + 28, (uint32_t)strlen(e->name)); wr16(ch + 30, 0); wr16(ch + 32, 0);
        wr16(ch + 34, 0); wr16(ch + 36, 0); wr32(ch + 38, 0); wr32(ch + 42, e->off);
        fwrite(ch, 1, 46, z->f); fwrite(e->name, 1, strlen(e->name), z->f);
        z->pos += 46 + (long long)strlen(e->name);
    }
    uint8_t eo[22] = {0};
    wr32(eo, 0x06054b50u); wr16(eo + 4, 0); wr16(eo + 6, 0);
    wr16(eo + 8, (uint32_t)z->n); wr16(eo + 10, (uint32_t)z->n);
    wr32(eo + 12, (uint32_t)(z->pos - cd)); wr32(eo + 16, (uint32_t)cd); wr16(eo + 20, 0);
    fwrite(eo, 1, 22, z->f);
}

/* ---------------- VQ 侧车(dql_vq_LXX.bin)反量化 ----------------
 * 逐式对应 probe_behavior_spectrum.vq_dequant:
 *   槽表在 blob+16, 每专家 3 个 u64 偏移(w1,w3,w2 顺序);
 *   载荷 = magic|dim,nc(u16)|rows,cols(u32) | 码本 f16[nc*dim] | 行乘子 f16[rows] | 索引位流。
 *   out[r][c] = cb[idx[r*(cols/dim) + c/dim]][c%dim] * gr[r]
 * 位流是小端 bit order(numpy unpackbits bitorder='little'): 第 i 个索引取 bit 偏移 i*nbit。 */
#define VQ_MAGIC 0x51565144u

static uint64_t vq_slot(const uint8_t *blob, size_t bsz, int e, int w) {
    size_t off = 16 + (size_t)(e * 3 + w) * 8;
    if (off + 8 > bsz) die("vq_slot 越界: e=%d w=%d", e, w);
    uint64_t v; memcpy(&v, blob + off, 8); return v;
}
/* 返回 malloc 的 f32 [rows*cols]; off==0 视为无该槽(返回 NULL, 与 py `if off else None` 同) */
static float *vq_dequant(const uint8_t *blob, size_t bsz, uint64_t off, long *R_out, long *C_out) {
    if (!off) return NULL;
    if (off + 16 > bsz) die("vq 载荷越界 off=%llu", (unsigned long long)off);
    uint32_t magic; memcpy(&magic, blob + off, 4);
    if (magic != VQ_MAGIC) die("vq 载荷 magic 不对: 0x%08x (assert)", magic);
    uint16_t dim, nc; memcpy(&dim, blob + off + 4, 2); memcpy(&nc, blob + off + 6, 2);
    uint32_t rows, cols; memcpy(&rows, blob + off + 8, 4); memcpy(&cols, blob + off + 12, 4);
    size_t p = (size_t)off + 16;
    size_t ncb = (size_t)nc * dim;
    float *cb = (float *)xmalloc(ncb * sizeof(float));
    for (size_t i = 0; i < ncb; i++) { uint16_t h; memcpy(&h, blob + p + i * 2, 2); cb[i] = f16_to_f32(h); }
    p += ncb * 2;
    float *gr = (float *)xmalloc((size_t)rows * sizeof(float));
    for (size_t i = 0; i < rows; i++) { uint16_t h; memcpy(&h, blob + p + i * 2, 2); gr[i] = f16_to_f32(h); }
    p += (size_t)rows * 2;
    int nbit = 1; while ((1 << nbit) < (int)nc) nbit++;     /* = max(1,(nc-1).bit_length()) */
    size_t nidx = (size_t)rows * cols / dim;
    size_t per = cols / dim;
    float *W = (float *)xmalloc((size_t)rows * cols * sizeof(float));
    for (size_t i = 0; i < nidx; i++) {
        uint32_t id;
        if (nbit == 8) { id = (p + i < bsz) ? blob[p + i] : 0u; }
        else {
            uint64_t bit = (uint64_t)i * (uint64_t)nbit, by = bit >> 3;
            uint32_t v = 0;
            for (int k = 0; k < 3; k++) if (p + by + k < bsz) v |= (uint32_t)blob[p + by + k] << (8 * k);
            id = (v >> (bit & 7)) & ((1u << nbit) - 1u);
        }
        if (id >= nc) id = nc - 1;
        size_t r = i / per;
        float g = gr[r];
        const float *c = cb + (size_t)id * dim;
        float *w = W + i * dim;
        for (int j = 0; j < dim; j++) w[j] = c[j] * g;
    }
    free(cb); free(gr);
    *R_out = (long)rows; *C_out = (long)cols;
    return W;
}

/* ---------------- 锚(DQA2)读取: probe_layer_behavior.anchor_layer 逐式 ---------------- */
typedef struct { int S, HCM, DIM, NL, VOCAB, NACT; } ameta_t;
static void anchor_layer(const char *ap, int L, int ntok, float **fin, int **ridx, float **rw, ameta_t *m) {
    FILE *f = fopen(ap, "rb");
    if (!f) die("锚打不开: %s", ap);
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8) die("锚头截断: %s", ap);
    if (hd[0] != 0x32415144u) die("%s 不是 DQA2 锚(magic 0x%08x)", ap, hd[0]);
    m->S = (int)hd[1]; m->HCM = (int)hd[2]; m->DIM = (int)hd[3];
    m->NL = (int)hd[4]; m->VOCAB = (int)hd[5]; m->NACT = (int)hd[6];
    if (m->DIM != D) die("锚 DIM=%d ≠ %d — .py 写死 D=4096, 口径不明拒跑", m->DIM, D);
    if (ntok > m->S) die("DS4_ZL_NTOK=%d > 锚 S=%d", ntok, m->S);
    long long fin_off = 40;
    long long ridx_off = fin_off + (long long)m->NL * m->S * m->DIM * 4;
    long long rw_off = ridx_off + (long long)m->NL * m->S * m->NACT * 4;
    *fin = (float *)xmalloc((size_t)ntok * D * sizeof(float));
    *ridx = (int *)xmalloc((size_t)ntok * m->NACT * sizeof(int));
    *rw = (float *)xmalloc((size_t)ntok * m->NACT * sizeof(float));
    if (fseeko(f, (off_t)(fin_off + (long long)L * m->S * m->DIM * 4), SEEK_SET) ||
        fread(*fin, 4, (size_t)ntok * D, f) != (size_t)ntok * D) die("锚 fin 读不满 L=%d", L);
    if (fseeko(f, (off_t)(ridx_off + (long long)L * m->S * m->NACT * 4), SEEK_SET) ||
        fread(*ridx, 4, (size_t)ntok * m->NACT, f) != (size_t)ntok * m->NACT) die("锚 ridx 读不满 L=%d", L);
    if (fseeko(f, (off_t)(rw_off + (long long)L * m->S * m->NACT * 4), SEEK_SET) ||
        fread(*rw, 4, (size_t)ntok * m->NACT, f) != (size_t)ntok * m->NACT) die("锚 rw 读不满 L=%d", L);
    fclose(f);
}

/* ---------------- swiglu / φ 提升 ----------------
 * ★分歧点★ ds4quant_fwd.c 的 dq_expert_fp 只夹 gate 的上侧(gg>swlim), .py 的 swiglu 是
 * 【双侧】夹(g 和 u 都夹到 [-lim,lim])。这里按 .py 写 —— 解算目标必须和 .py 的目标一致,
 * 否则 held 挽回率没法对拍。内层 exp 的 ±60 夹在 lim>0 时是死代码(g 已在 ±lim 内),
 * 只有 DS4_ZL_SWLIM=0(关截断)时才起作用, 照抄 CPU 路的 probe_layer_behavior.swiglu。 */
static float zl_swiglu1(float g, float u, float lim) {
    if (lim > 0) {
        if (g > lim) g = lim; else if (g < -lim) g = -lim;
        if (u > lim) u = lim; else if (u < -lim) u = -lim;
    }
    float gc = g > 60.0f ? 60.0f : (g < -60.0f ? -60.0f : g);
    return (g / (1.0f + expf(-gc))) * u;
}

/* zl_phi(M) = [M, M⊙M/rms, relu(M)], rms=sqrt(mean(M²,axis=1))+1e-6, 结果 f32。
 * 两个入口的中间精度不同, 照抄 .py:
 *   from_f64: py 传的是 f64 的 Xa —— 全程 f64 算, 最后一次性 astype(f32);
 *   from_f32: py 传的是 X[ev].astype(f32) —— 平方/除法都在 f32 上做。 */
typedef struct { const double *M; float *out; int d; } phi64_ctx;
static void phi64_worker(void *vc, int t0, int t1) {
    phi64_ctx *c = (phi64_ctx *)vc;
    for (int t = t0; t < t1; t++) {
        const double *m = c->M + (size_t)t * c->d;
        float *o = c->out + (size_t)t * 3 * c->d;
        double ss = 0;
        for (int j = 0; j < c->d; j++) ss += m[j] * m[j];
        double n = sqrt(ss / c->d) + 1e-6;
        for (int j = 0; j < c->d; j++) {
            o[j] = (float)m[j];
            o[c->d + j] = (float)((m[j] * m[j]) / n);
            o[2 * c->d + j] = (float)(m[j] > 0 ? m[j] : 0);
        }
    }
}
typedef struct { const float *M; float *out; int d; } phi32_ctx;
static void phi32_worker(void *vc, int t0, int t1) {
    phi32_ctx *c = (phi32_ctx *)vc;
    for (int t = t0; t < t1; t++) {
        const float *m = c->M + (size_t)t * c->d;
        float *o = c->out + (size_t)t * 3 * c->d;
        double ss = 0;                       /* 平方在 f32(与 numpy 同), 求和用 f64 累加器 */
        for (int j = 0; j < c->d; j++) { float sq = m[j] * m[j]; ss += (double)sq; }
        float n = (float)sqrtf((float)(ss / c->d)) + 1e-6f;
        for (int j = 0; j < c->d; j++) {
            o[j] = m[j];
            o[c->d + j] = (m[j] * m[j]) / n;
            o[2 * c->d + j] = m[j] > 0 ? m[j] : 0.0f;
        }
    }
}

/* ---------------- 低秩 SVD: 子空间迭代 + Rayleigh-Ritz ----------------
 * ★与 numpy 的差别★ .py 走 np.linalg.svd(LAPACK gesdd, 精确全谱)。这里只求前 r 个奇异
 * 三元组: 随机起始 → (Wᵀ 再 W) 幂迭代 12 轮(与 ds4quant_run.c z_solve_dual 尾部同款) →
 * 用 B=VᵀW 的小核 BBᵀ(r×r) 精确特征分解回收奇异值/向量(Rayleigh-Ritz)。
 * 精确性: 只要 V 张成的子空间收敛到前 r 个左奇异方向, 结果就等于精确截断; 收敛速度取决
 * 于 σ_r/σ_{r+p} 的间隙, 所以过采样 p=64。【故不可与 .py 逐位对拍】, 判定看 held 挽回率。
 * 出参: A[din*r](左, 列存 c), S[r](降序), Bt[r*Dout](右, 行 c)。 */
typedef struct { double *V; int rows, r; const double *G; int gcols; } cgs_ctx;

/* 块 Gram-Schmidt 正交化(带一次重正交): V[rows,r] 就地正交归一 */
static void orthonormalize(double *V, int rows, int r) {
    const int BS = 64;
    double *C = (double *)xmalloc((size_t)r * BS * sizeof(double));
    double *T = (double *)xmalloc((size_t)rows * BS * sizeof(double));
    for (int j0 = 0; j0 < r; j0 += BS) {
        int bs = j0 + BS > r ? r - j0 : BS;
        for (int pass = 0; pass < 2 && j0 > 0; pass++) {
            /* C[j0,bs] = Vprevᵀ · Vblk ; Vblk -= Vprev·C */
            mm64(1, 0, j0, bs, rows, V, r, V + j0, r, C, bs);
            mm64(0, 0, rows, bs, j0, V, r, C, bs, T, bs);
            for (int i = 0; i < rows; i++) {
                double *v = V + (size_t)i * r + j0;
                const double *t = T + (size_t)i * bs;
                for (int j = 0; j < bs; j++) v[j] -= t[j];
            }
        }
        for (int j = 0; j < bs; j++) {                  /* 块内 MGS */
            int col = j0 + j;
            for (int rep = 0; rep < 2; rep++) {
                for (int k = j0; k < col; k++) {
                    double d = 0;
                    for (int i = 0; i < rows; i++) d += V[(size_t)i * r + k] * V[(size_t)i * r + col];
                    for (int i = 0; i < rows; i++) V[(size_t)i * r + col] -= d * V[(size_t)i * r + k];
                }
            }
            double nr = 0;
            for (int i = 0; i < rows; i++) nr += V[(size_t)i * r + col] * V[(size_t)i * r + col];
            nr = sqrt(nr);
            if (nr < 1e-200) { for (int i = 0; i < rows; i++) V[(size_t)i * r + col] = (i == col % rows) ? 1.0 : 0.0; nr = 1.0; }
            double inv = 1.0 / nr;
            for (int i = 0; i < rows; i++) V[(size_t)i * r + col] *= inv;
        }
    }
    free(C); free(T);
}

static void zl_svd_lowrank(const double *W, int din, int dout, int r,
                           double **A_out, double **S_out, double **Bt_out) {
    if (r > din) r = din;
    if (r > dout) r = dout;
    double *V = (double *)xmalloc((size_t)din * r * sizeof(double));
    double *T = (double *)xmalloc((size_t)dout * r * sizeof(double));
    uint64_t seed = 0x5A5A1EEDULL;                       /* 固定种子: 同输入必同产物 */
    for (size_t i = 0; i < (size_t)din * r; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        V[i] = (double)((seed >> 40) & 0xFFFFFF) / 16777216.0 - 0.5;
    }
    orthonormalize(V, din, r);
    for (int it = 0; it < 12; it++) {
        mm64(1, 0, dout, r, din, W, dout, V, r, T, r);   /* T = Wᵀ V */
        mm64(0, 0, din, r, dout, W, dout, T, r, V, r);   /* V = W  T */
        orthonormalize(V, din, r);
    }
    double *B = (double *)xmalloc((size_t)r * dout * sizeof(double));
    mm64(1, 0, r, dout, din, V, r, W, dout, B, dout);    /* B = Vᵀ W  [r,dout] */
    double *C = (double *)xmalloc((size_t)r * r * sizeof(double));
    mm64(0, 1, r, r, dout, B, dout, B, dout, C, r);      /* C = B Bᵀ  [r,r] */
    double *lam = (double *)xmalloc((size_t)r * sizeof(double));
    double *E = (double *)xmalloc((size_t)r * r * sizeof(double));
    jacobi_eig(C, r, lam, E);
    free(C);
    double *S = (double *)xmalloc((size_t)r * sizeof(double));
    for (int c = 0; c < r; c++) S[c] = lam[c] > 0 ? sqrt(lam[c]) : 0.0;
    free(lam);
    double *A = (double *)xmalloc((size_t)din * r * sizeof(double));
    mm64(0, 0, din, r, r, V, r, E, r, A, r);             /* A = V E */
    double *Bt = (double *)xmalloc((size_t)r * dout * sizeof(double));
    mm64(1, 0, r, dout, r, E, r, B, dout, Bt, dout);     /* Bt = Eᵀ B, 下面按 1/S 归一 */
    double smax = S[0];
    for (int c = 0; c < r; c++) {
        double inv = (S[c] > 1e-13 * smax && S[c] > 0) ? 1.0 / S[c] : 0.0;
        if (inv == 0.0) S[c] = 0.0;
        double *row = Bt + (size_t)c * dout;
        for (int j = 0; j < dout; j++) row[j] *= inv;
    }
    free(V); free(T); free(B); free(E);
    *A_out = A; *S_out = S; *Bt_out = Bt;
}

/* ---------------- 单专家前向(FP 或量化侧共用) ----------------
 * py: Y = swiglu(x@w1ᵀ, x@w3ᵀ) @ w2ᵀ, 路由权重【不在这里乘】(外面 dH[rows]+=w*Y)。
 * w1/w3 是 [MOEI, D], w2 是 [D, MOEI]; x[n,D] → Y[n,D]。全 f32(与 numpy 同)。 */
static void zl_expert_fwd(const float *x, int n, const float *w1, const float *w3,
                          const float *w2, int MOEI, float lim, float *Y, float *g, float *u) {
    mm32(0, 1, n, MOEI, D, x, D, w1, D, g, MOEI);
    mm32(0, 1, n, MOEI, D, x, D, w3, D, u, MOEI);
    for (size_t i = 0; i < (size_t)n * MOEI; i++) g[i] = zl_swiglu1(g[i], u[i], lim);
    mm32(0, 1, n, D, MOEI, g, MOEI, w2, MOEI, Y, D);
}

/* ---------------- XCAP 捕获读取与对齐自检 ----------------
 * 引擎走 DS4_EVAL_IDS 时流首插了 BOS, 捕获会比锚多一行(NTOK+1)。差这一行就是整体错位
 * 一格 —— 每个 token 的 x 配到前一个 token 的目标上, 不报错、照样出挽回率。 */
static float *capload(const char *xcap, const char *nm, int L, int NTOK, int cols) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s_L%d", xcap, nm, L);
    FILE *f = fopen(p, "rb");
    if (!f) die("捕获打不开: %s", p);
    fseeko(f, 0, SEEK_END);
    long long sz = ftello(f);
    long long n = sz / 2 / cols;
    if (n != NTOK && n != NTOK + 1)
        die("assert 失败: %s: %lld 行, 既非 NTOK=%d 也非 NTOK+1(BOS) — 口径不明, 拒跑", p, n, NTOK);
    long long off = n - NTOK;                            /* 1 = 掐掉流首 BOS 行 */
    if (fseeko(f, (off_t)(off * cols * 2), SEEK_SET)) die("%s seek 失败", p);
    uint16_t *h = (uint16_t *)xmalloc((size_t)NTOK * cols * 2);
    if (fread(h, 2, (size_t)NTOK * cols, f) != (size_t)NTOK * cols) die("%s 读不满", p);
    fclose(f);
    float *o = (float *)xmalloc((size_t)NTOK * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)NTOK * cols; i++) o[i] = f16_to_f32(h[i]);
    free(h);
    return o;
}

/* 逐行余弦的中位数(a,b 各取前 min(len) 行)。py 用 f32 算、np.median 对偶数长度取中间两个
 * 的均值 —— 这里用 f64 累加, 差在 1e-7 量级, 判的是"对齐 vs 错位"的离散问题, 不影响判据。 */
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static double cosmed(const float *a, const float *b, int n, int cols) {
    double *v = (double *)xmalloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        const float *x = a + (size_t)i * cols, *y = b + (size_t)i * cols;
        double d = 0, nx = 0, ny = 0;
        for (int j = 0; j < cols; j++) { d += (double)x[j] * y[j]; nx += (double)x[j] * x[j]; ny += (double)y[j] * y[j]; }
        v[i] = d / (sqrt(nx) * sqrt(ny) + 1e-9);
    }
    qsort(v, (size_t)n, sizeof(double), cmp_dbl);
    double m = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    free(v);
    return m;
}
static double fro_norm(const float *a, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (double)a[i] * a[i];
    return sqrt(s);
}

/* ---------------- 116B 记录头(py 的 rec()) ---------------- */
static uint8_t *make_rec(const char *nm, const void *pay, size_t psz, size_t *out_len) {
    uint8_t *r = (uint8_t *)xcalloc(116 + psz, 1);
    memcpy(r, nm, strlen(nm));
    uint64_t p64 = psz; memcpy(r + 88, &p64, 8);
    int32_t one = 1; memcpy(r + 112, &one, 4);
    if (psz) memcpy(r + 116, pay, psz);
    *out_len = 116 + psz;
    return r;
}

/* ---------------- 区间解析: "a:b,c:d" → 行号数组 ---------------- */
static int *parse_ranges(const char *s, int *n_out) {
    int cap = 64, n = 0;
    int *v = (int *)xmalloc((size_t)cap * sizeof(int));
    const char *p = s;
    while (*p) {
        char *e1, *e2;
        long a = strtol(p, &e1, 10);
        if (*e1 != ':') die("区间语法错(要 a:b): %s", s);
        long b = strtol(e1 + 1, &e2, 10);
        for (long i = a; i < b; i++) {
            if (n == cap) { cap *= 2; v = (int *)realloc(v, (size_t)cap * sizeof(int)); if (!v) die("realloc"); }
            v[n++] = (int)i;
        }
        p = e2;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p && !(*p >= '0' && *p <= '9')) die("区间语法错: %s", s);
    }
    *n_out = n;
    return v;
}

static int env_int(const char *k, int dflt) { const char *v = getenv(k); return v ? atoi(v) : dflt; }
static double env_dbl(const char *k, double dflt) { const char *v = getenv(k); return v ? atof(v) : dflt; }

/* ======================================================================== *
 *                                  main                                    *
 * ======================================================================== */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);          /* py 全程 flush=True: 战役脚本靠管道实时 grep */
    crc_init();

    if (argc > 1 && !strcmp(argv[1], "--selftest-rng")) {
        /* 对拍: python3 -c "import numpy as np; print(np.random.RandomState(1).randn(8))"
         * 期望 1.62434536 -0.61175641 -0.52817175 -1.07296862 0.86540763 -2.3015387
         *      1.74481176 -0.7612069 */
        mt_t s; mt_seed(&s, 1);
        for (int i = 0; i < 8; i++) printf("%.17g\n", mt_gauss(&s));
        return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "用法: zlayer <hf> <layers_dir> <anchor> <L> [K=1024] [inject=1] [XCAP目录] [PREV目录]\n"
                        "      zlayer --selftest-rng\n");
        return 1;
    }
    const char *hf = argv[1], *ld = argv[2], *ap = argv[3];
    int L = atoi(argv[4]);
    int K = argc > 5 ? atoi(argv[5]) : 1024;
    int INJ = argc > 6 ? atoi(argv[6]) : 1;
    const char *XCAP = (argc > 7 && argv[7][0] && strcmp(argv[7], "-")) ? argv[7] : NULL;
    const char *PREV = (argc > 8 && argv[8][0] && strcmp(argv[8], "-")) ? argv[8] : NULL;

    /* 第一期未实现的支路: 宁可停车也不静默换语义(这一周的教训就是 Python↔C 悄悄分歧) */
    if (getenv("DS4_ZL_GGUF")) die("DS4_ZL_GGUF 标量模式第一期未实现 — 拒跑(用 .py 或等第二期)");
    if (getenv("DS4_ZL_XANCHOR")) die("DS4_ZL_XANCHOR 链模式第一期未实现 — 拒跑");
    if (getenv("DS4_ZL_ADDON")) die("DS4_ZL_ADDON 叠加模式第一期未实现 — 拒跑");
    if (!XCAP) die("第一期只做 XCAP 口径: 第 7 个位置参数(引擎捕获目录)必填");

    const int NTOK = env_int("DS4_ZL_NTOK", 1716);
    const int CACHE_ONLY = (getenv("DS4_ZL_CACHE_ONLY") &&
                            strcmp(getenv("DS4_ZL_CACHE_ONLY"), "") &&
                            strcmp(getenv("DS4_ZL_CACHE_ONLY"), "0")) ? 1 : 0;
    const float SWLIM = (float)env_dbl("DS4_ZL_SWLIM", 10.0);
    const int FTA = env_int("DS4_ZL_FTA", 1);
    const int GE_ON = env_int("DS4_ZL_GE", 1);
    const double GATE = env_dbl("DS4_ZL_GATE", 0.0);
    const double GELAM = env_dbl("DS4_ZL_GE_LAM", 1e-3);
    char path[1200];

    if (INJ == 2) {   /* 外挂模式断点续跑: zrec 已在则整层跳过(解算也省) */
        snprintf(path, sizeof path, "%s/zrec_L%02d.bin", ld, L);
        struct stat st;
        if (!stat(path, &st)) { printf("★L%d zrec 已存在(%lldB), 跳过\n", L, (long long)st.st_size); return 0; }
    }

    double t0 = now_s();
    float *X0fp = NULL, *rw = NULL; int *ridx = NULL; ameta_t am;
    anchor_layer(ap, L, NTOK, &X0fp, &ridx, &rw, &am);
    const int NACT = am.NACT;

    /* ★XCAP★ x 与被乘量都换成引擎真值(raw_ffn_in / raw_ffn_out), FP 侧仍走锚(教师)。
     * 教师路由 = FP 锚的 ridx/rw(2026-08-22 用户裁决: 换成量化路由等于换靶子)。 */
    float *X0 = capload(XCAP, "raw_ffn_in", L, NTOK, D);
    float *YQE = capload(XCAP, "raw_ffn_out", L, NTOK, D);

    if (PREV) {
        /* 第二轮反修: y_未修 = raw_ffn_out(带) / (1 + g_旧(新输入)) */
        snprintf(path, sizeof path, "%s/zrec_L%02d.bin", PREV, L);
        FILE *pf = fopen(path, "rb");
        if (!pf) die("上一轮记录打不开: %s", path);
        uint8_t hdr[116];
        if (fread(hdr, 1, 116, pf) != 116) die("%s 头截断", path);
        char nm[17]; memcpy(nm, hdr, 16); nm[16] = 0;
        if (!strstr(nm, "zl.AMPD")) die("assert 失败: L%d 上一轮记录不是 AMPD(%s), 无法还原增益", L, nm);
        uint64_t psz; memcpy(&psz, hdr + 88, 8);
        uint8_t *pay = (uint8_t *)xmalloc((size_t)psz);
        if (fread(pay, 1, (size_t)psz, pf) != psz) die("%s 载荷截断", path);
        fclose(pf);
        uint32_t pk, pdi, pdo; float psc;
        memcpy(&pk, pay, 4); memcpy(&psc, pay + 4, 4); memcpy(&pdi, pay + 8, 4); memcpy(&pdo, pay + 12, 4);
        if ((int)pdi != D) die("上一轮 din=%u ≠ %d — .py 的 X0@_V 在 din=3D 时也会崩, 口径不明拒跑", pdi, D);
        const uint16_t *h = (const uint16_t *)(pay + 16);
        size_t nA = (size_t)pdi * pk, nU = (size_t)pdo * pk;
        float *A = (float *)xmalloc(nA * 4), *U = (float *)xmalloc(nU * 4), *V = (float *)xmalloc(nA * 4);
        for (size_t i = 0; i < nA; i++) A[i] = f16_to_f32(h[i]);
        for (size_t i = 0; i < nU; i++) U[i] = f16_to_f32(h[nA + i]);
        for (size_t i = 0; i < nA; i++) V[i] = f16_to_f32(h[nA + nU + i]);
        float *pv = (float *)xmalloc((size_t)NTOK * pk * 4);
        float *pa = (float *)xmalloc((size_t)NTOK * pk * 4);
        mm32(0, 0, NTOK, (int)pk, D, X0, D, V, (int)pk, pv, (int)pk);
        mm32(0, 0, NTOK, (int)pk, D, X0, D, A, (int)pk, pa, (int)pk);
        for (size_t i = 0; i < (size_t)NTOK * pk; i++) pv[i] = tanhf(pv[i] / psc) * tanhf(pa[i] / psc);
        float *g = (float *)xmalloc((size_t)NTOK * D * 4);
        mm32(0, 1, NTOK, D, (int)pk, pv, (int)pk, U, (int)pk, g, D);
        double *ag = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
        for (size_t i = 0; i < (size_t)NTOK * D; i++) {
            float den = 1.0f + g[i];
            /* py: |den|<1e-3 时换成 sign(den)*1e-3 —— 注意 np.sign(0)=0, 换出来还是 0
             * (随后除零得 inf)。照抄, 不加"改进"。 */
            if (fabsf(den) < 1e-3f) den = (den > 0 ? 1e-3f : (den < 0 ? -1e-3f : 0.0f));
            YQE[i] = YQE[i] / den;
            ag[i] = fabs((double)g[i]);
        }
        qsort(ag, (size_t)NTOK * D, sizeof(double), cmp_dbl);
        size_t ng = (size_t)NTOK * D;
        double med = (ng % 2) ? ag[ng / 2] : 0.5 * (ag[ng / 2 - 1] + ag[ng / 2]);
        double pos = 0.99 * (double)(ng - 1);
        size_t lo = (size_t)pos; double fr = pos - (double)lo;
        double p99 = ag[lo] + fr * (ag[lo + 1 < ng ? lo + 1 : lo] - ag[lo]);
        printf("  L%d 第二轮: 已除回上一轮增益 |g_旧| 中位 %.5f p99 %.5f\n", L, med, p99);
        free(A); free(U); free(V); free(pv); free(pa); free(g); free(ag); free(pay);
    }

    /* ★对齐自检★ 判的是"两种对齐哪个对"的离散问题, 不是拿绝对余弦当质量闸:
     * 绝对值随层数衰减(L0 0.89 → L42 0.56), 拿常数当门会把深层全误拦。 */
    {
        double c_ok = cosmed(X0fp, X0, NTOK, D);
        double c_bad = cosmed(X0fp + (size_t)D, X0, NTOK - 1, D);
        if (!(c_ok > c_bad * 1.15))
            die("assert 失败: L%d XCAP 对齐自检失败: 采用对齐 %.4f 未明显优于错位版 %.4f — 口径可疑, 停车",
                L, c_ok, c_bad);
        printf("  L%d XCAP: 对齐 %.4f vs 错位 %.4f (%.1f×) |x|=%.1f |y_q|=%.1f\n",
               L, c_ok, c_bad, c_ok / (c_bad > 1e-6 ? c_bad : 1e-6),
               fro_norm(X0, (size_t)NTOK * D), fro_norm(YQE, (size_t)NTOK * D));
    }

    /* ---------------- 缓存: dH 与配对记录 ---------------- */
    const long long NPAIR = (long long)NTOK * NACT;
    float *dH = NULL, *pw = NULL, *pYQ = NULL;
    int *prow = NULL, *pe = NULL;
    long long npair = 0;
    char cache[1200];
    snprintf(cache, sizeof cache, "%s/zcache_L%02d.npz", ld, L);

    struct stat cst;
    if (!stat(cache, &cst) && cst.st_size > 0) {
        int fd = open(cache, O_RDONLY);
        if (fd < 0) die("zcache 打不开: %s", cache);
        size_t zsz = (size_t)cst.st_size;
        uint8_t *zb = (uint8_t *)mmap(NULL, zsz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (zb == MAP_FAILED) die("zcache mmap 失败");
        npy_t adH, aprow, ape, apw, apYQ;
        if (npz_find(zb, (long long)zsz, "dH", &adH) || npz_find(zb, (long long)zsz, "prow", &aprow) ||
            npz_find(zb, (long long)zsz, "pe", &ape) || npz_find(zb, (long long)zsz, "pw", &apw) ||
            npz_find(zb, (long long)zsz, "pYQ", &apYQ)) die("zcache 字段缺: %s", cache);
        if (adH.d0 != NTOK || adH.d1 != D) die("zcache dH 形状 %lldx%lld ≠ %dx%d", adH.d0, adH.d1, NTOK, D);
        npair = aprow.d0;
        dH = (float *)xmalloc((size_t)NTOK * D * 4);          npz_to_f32(&adH, dH, (long long)NTOK * D);
        prow = (int *)xmalloc((size_t)npair * sizeof(int));   npz_to_i32(&aprow, prow, npair);
        pe = (int *)xmalloc((size_t)npair * sizeof(int));     npz_to_i32(&ape, pe, npair);
        pw = (float *)xmalloc((size_t)npair * 4);             npz_to_f32(&apw, pw, npair);
        pYQ = (float *)xmalloc((size_t)npair * D * 4);        npz_to_f32(&apYQ, pYQ, npair * D);
        for (long long i = 0; i < npair; i++)                 /* 缓存是外部产物, 索引越界会写飞内存 */
            if (prow[i] < 0 || prow[i] >= NTOK || pe[i] < 0 || pe[i] >= NEXP)
                die("zcache 配对越界: prow=%d pe=%d (NTOK=%d)", prow[i], pe[i], NTOK);
        munmap(zb, zsz); close(fd);
    } else {
        st_ctx *sc = (st_ctx *)xmalloc(sizeof(st_ctx));
        st_open(sc, hf);
        snprintf(path, sizeof path, "%s/dql_vq_L%02d.bin", ld, L);
        int bfd = open(path, O_RDONLY);
        if (bfd < 0) die("VQ 侧车打不开: %s", path);
        struct stat bst; fstat(bfd, &bst);
        size_t bsz = (size_t)bst.st_size;
        const uint8_t *blob = (const uint8_t *)mmap(NULL, bsz, PROT_READ, MAP_PRIVATE, bfd, 0);
        if (blob == MAP_FAILED) die("VQ 侧车 mmap 失败: %s", path);

        int seen[NEXP]; memset(seen, 0, sizeof seen);
        for (long long i = 0; i < (long long)NTOK * NACT; i++) {
            int e = ridx[i];
            if (e < 0 || e >= NEXP) die("锚 ridx 越界: %d", e);
            seen[e] = 1;
        }
        int need[NEXP], nneed = 0;
        for (int e = 0; e < NEXP; e++) if (seen[e]) need[nneed++] = e;

        dH = (float *)xcalloc((size_t)NTOK * D, 4);
        prow = (int *)xmalloc((size_t)NPAIR * sizeof(int));
        pe = (int *)xmalloc((size_t)NPAIR * sizeof(int));
        pw = (float *)xmalloc((size_t)NPAIR * 4);
        pYQ = (float *)xmalloc((size_t)NPAIR * D * 4);

        int *rows = (int *)xmalloc((size_t)NTOK * NACT * sizeof(int));
        int *slots = (int *)xmalloc((size_t)NTOK * NACT * sizeof(int));
        float *xs = (float *)xmalloc((size_t)NTOK * D * 4);
        float *Y = (float *)xmalloc((size_t)NTOK * D * 4);
        float *gb = NULL, *ub = NULL;
        int MOEI = 0;

        for (int i = 0; i < nneed; i++) {
            int e = need[i];
            int nr = 0;
            for (int t = 0; t < NTOK; t++)                    /* np.where(ridx==e): 行升序, 行内槽升序 */
                for (int s = 0; s < NACT; s++)
                    if (ridx[(size_t)t * NACT + s] == e) { rows[nr] = t; slots[nr] = s; nr++; }
            if (!nr) continue;
            float *Wf[3], *Wq[3];
            static const char *NMS[3] = {"w1", "w3", "w2"};
            for (int wi = 0; wi < 3; wi++) {
                char tn[256];
                snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.%s.weight", L, e, NMS[wi]);
                long R = 0, C = 0;
                float *wf = st_read_weight(sc, tn, &R, &C);
                if (!wf) die("HF 权重读不到: %s", tn);
                long rq = 0, cq = 0;
                float *wq = vq_dequant(blob, bsz, vq_slot(blob, bsz, e, wi), &rq, &cq);
                if (wq && (R != rq || C != cq)) {             /* py: Wf.shape != Wq.shape → Wf = Wf.T */
                    float *tr = (float *)xmalloc((size_t)R * C * 4);
                    for (long r = 0; r < R; r++) for (long c = 0; c < C; c++) tr[(size_t)c * R + r] = wf[(size_t)r * C + c];
                    free(wf); wf = tr; long t = R; R = C; C = t;
                }
                Wf[wi] = wf; Wq[wi] = wq ? wq : wf;           /* 无 vq 槽 ⇒ 量化侧退回 FP(py 同) */
                if (wi == 0) {
                    if (C != D) die("w1 列数 %ld ≠ D=%d(L%d e%d) — 方向判定失败", C, D, L, e);
                    if (!MOEI) { MOEI = (int)R;
                        gb = (float *)xmalloc((size_t)NTOK * MOEI * 4);
                        ub = (float *)xmalloc((size_t)NTOK * MOEI * 4); }
                    if (R != MOEI) die("w1 行数 %ld ≠ MOEI=%d", R, MOEI);
                } else if (wi == 2 && (R != D || C != MOEI))
                    die("w2 形状 %ldx%ld ≠ %dx%d", R, C, D, MOEI);
            }
            /* FP 目标侧: 引擎真值 x + FP 锚路由 + FP 权重 */
            for (int j = 0; j < nr; j++) memcpy(xs + (size_t)j * D, X0 + (size_t)rows[j] * D, (size_t)D * 4);
            zl_expert_fwd(xs, nr, Wf[0], Wf[1], Wf[2], MOEI, SWLIM, Y, gb, ub);
            for (int j = 0; j < nr; j++) {
                float w = rw[(size_t)rows[j] * NACT + slots[j]];
                float *dst = dH + (size_t)rows[j] * D;
                const float *y = Y + (size_t)j * D;
                for (int d2 = 0; d2 < D; d2++) dst[d2] += w * y[d2];
            }
            /* 部署侧: XCAP 时 dH 不再减 Yq(直接用引擎 raw_ffn_out), 但 pYQ 仍要留给 GE */
            zl_expert_fwd(xs, nr, Wq[0], Wq[1], Wq[2], MOEI, SWLIM, Y, gb, ub);
            for (int j = 0; j < nr; j++) {
                prow[npair] = rows[j];
                pe[npair] = e;
                pw[npair] = rw[(size_t)rows[j] * NACT + slots[j]];
                memcpy(pYQ + (size_t)npair * D, Y + (size_t)j * D, (size_t)D * 4);
                npair++;
            }
            for (int wi = 0; wi < 3; wi++) { if (Wq[wi] != Wf[wi]) free(Wq[wi]); free(Wf[wi]); }
            if ((i + 1) % 64 == 0) printf("  L%d 缓存 …%d/%d\n", L, i + 1, nneed);
        }
        for (size_t i = 0; i < (size_t)NTOK * D; i++) dH[i] -= YQE[i];   /* dH = Σw·Y_fp(锚教师) − 引擎真实量化 routed */

        free(rows); free(slots); free(xs); free(Y); free(gb); free(ub);
        munmap((void *)blob, bsz); close(bfd);

        /* 原子换名: 并行 amp_solve 只见完整 zcache */
        char tmp[1300]; snprintf(tmp, sizeof tmp, "%s.tmp.npz", cache);
        zwr_t zw; memset(&zw, 0, sizeof zw);
        zw.f = fopen(tmp, "wb");
        if (!zw.f) die("zcache 写不开: %s", tmp);
        int64_t *prow64 = (int64_t *)xmalloc((size_t)npair * 8);
        for (long long i = 0; i < npair; i++) prow64[i] = prow[i];
        zw_add(&zw, "dH", "<f4", NTOK, D, 2, dH, 4);
        zw_add(&zw, "prow", "<i8", npair, 1, 1, prow64, 8);
        zw_add(&zw, "pe", "<i4", npair, 1, 1, pe, 4);
        zw_add(&zw, "pw", "<f4", npair, 1, 1, pw, 4);
        zw_add(&zw, "pYQ", "<f4", npair, D, 2, pYQ, 4);
        zw_add(&zw, "pDY", "<f4", npair, D, 2, NULL, 4);      /* py 存的就是全零 */
        zw_add(&zw, "yqe", "<f4", NTOK, D, 2, YQE, 4);
        zw_add(&zw, "xcap", "<f4", NTOK, D, 2, X0, 4);
        zw_finish(&zw);
        fclose(zw.f);
        free(prow64);
        if (rename(tmp, cache)) die("zcache 换名失败: %s", strerror(errno));
        free(sc);
    }
    double t1 = now_s();
    if (CACHE_ONLY) { printf("★L%d zcache-only: 就绪(%.0fs), 解算交外部\n", L, t1 - t0); return 0; }

    /* ---------------- 解算 ---------------- */
    double *X = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
    for (size_t i = 0; i < (size_t)NTOK * D; i++) X[i] = X0[i];

    int ntr = 0, nev = 0, *tr = NULL, *ev = NULL;
    const char *fr = getenv("DS4_ZL_FIT_RANGES"), *er = getenv("DS4_ZL_EV_RANGE");
    if (fr) {
        if (!er) die("DS4_ZL_FIT_RANGES 设了但 DS4_ZL_EV_RANGE 没设 — .py 同样会崩");
        tr = parse_ranges(fr, &ntr);
        ev = parse_ranges(er, &nev);
    } else {
        int NF = env_int("DS4_ZL_NFIT", 1287);
        ntr = NF; nev = NTOK - NF;
        tr = (int *)xmalloc((size_t)ntr * sizeof(int));
        ev = (int *)xmalloc((size_t)(nev > 0 ? nev : 1) * sizeof(int));
        for (int i = 0; i < ntr; i++) tr[i] = i;
        for (int i = 0; i < nev; i++) ev[i] = NF + i;
    }
    for (int i = 0; i < ntr; i++) if (tr[i] < 0 || tr[i] >= NTOK) die("fit 行号 %d 越界", tr[i]);
    for (int i = 0; i < nev; i++) if (ev[i] < 0 || ev[i] >= NTOK) die("ev 行号 %d 越界", ev[i]);

    /* colw = sqrt(var(dH[tr], axis=0) + 1e-12) —— ★是 f32★(py 里 dH 是 f32, .var(0) 不升精度),
     * 于是 Ra=dH[tr]*colw 也是 f32, 到 np.linalg.solve 才升 f64。这里照抄这个精度阶梯。
     * (均值/方差的求和用 f64 累加器, 与 numpy 的 f32 pairwise 差 ~1e-7 相对, 打印精度内。) */
    float *colw = (float *)xmalloc((size_t)D * 4);
    for (int j = 0; j < D; j++) {
        double m = 0;
        for (int i = 0; i < ntr; i++) m += dH[(size_t)tr[i] * D + j];
        m /= ntr;
        double v = 0;
        for (int i = 0; i < ntr; i++) { double d2 = dH[(size_t)tr[i] * D + j] - m; v += d2 * d2; }
        colw[j] = sqrtf((float)(v / ntr) + 1e-12f);
    }

    const int NN = 2 * ntr;
    double *Xa = (double *)xmalloc((size_t)NN * D * sizeof(double));
    float *Ra32 = (float *)xmalloc((size_t)NN * D * 4);
    {
        mt_t rng; mt_seed(&rng, 1);          /* np.random.RandomState(1) */
        for (int i = 0; i < ntr; i++) {
            const double *x = X + (size_t)tr[i] * D;
            memcpy(Xa + (size_t)i * D, x, (size_t)D * sizeof(double));
            double ss = 0;
            for (int j = 0; j < D; j++) ss += x[j] * x[j];
            double rms = sqrt(ss / D);
            double *xd = Xa + (size_t)(ntr + i) * D;
            /* randn 按 C 序逐元素抽 —— 行内先走完再下一行, 与 randn(len(tr),D) 一致 */
            for (int j = 0; j < D; j++) xd[j] = (x[j] + (mt_gauss(&rng) * 0.04) * rms) * 0.5;
            const float *dh = dH + (size_t)tr[i] * D;
            float *r0 = Ra32 + (size_t)i * D, *r1 = Ra32 + (size_t)(ntr + i) * D;
            for (int j = 0; j < D; j++) { r0[j] = dh[j] * colw[j]; r1[j] = r0[j] * 0.5f; }
        }
    }

    int kmax = 0;
    int KG[16], nKG = 0;
    {
        int cand[9] = {64, 128, 256, 384, 512, 768, 1024, 1536, K};
        for (int i = 0; i < 9; i++) {
            int k = cand[i];
            if (k > K || k > 1024) continue;
            int dup = 0;
            for (int j = 0; j < nKG; j++) if (KG[j] == k) dup = 1;
            if (!dup) KG[nKG++] = k;
        }
        for (int a = 0; a < nKG; a++) for (int b = a + 1; b < nKG; b++)
            if (KG[b] < KG[a]) { int t = KG[a]; KG[a] = KG[b]; KG[b] = t; }
        if (!nKG) { KG[nKG++] = K > 0 ? K : 0; }
        kmax = KG[nKG - 1];
    }

    /* --- 线性支线: 对偶 ridge(dither 增广 + colw 感知列权) → 折 colw → 低秩截断 --- */
    double *A = NULL, *S = NULL, *Bt = NULL, *Af = NULL, *Sf = NULL, *Bf = NULL;
    int rlin = 0, rfta = 0;
    {
        double *G = (double *)xmalloc((size_t)NN * NN * sizeof(double));
        mm64(0, 1, NN, NN, D, Xa, D, Xa, D, G, NN);
        double trc = 0;
        for (int i = 0; i < NN; i++) trc += G[(size_t)i * NN + i];
        double ridge = 3.0 * trc / D + 1e-10;
        for (int i = 0; i < NN; i++) G[(size_t)i * NN + i] += ridge;
        double *al = (double *)xmalloc((size_t)NN * D * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * D; i++) al[i] = Ra32[i];
        if (chol_solve(G, NN, al, D)) die("对偶 ridge Cholesky 失败(G 非正定) — 停车");
        free(G);
        double *Wz = (double *)xmalloc((size_t)D * D * sizeof(double));
        mm64(1, 0, D, D, NN, Xa, D, al, D, Wz, D);
        free(al);
        for (int i = 0; i < D; i++) { double *row = Wz + (size_t)i * D;
            for (int j = 0; j < D; j++) row[j] /= colw[j]; }
        if (kmax > 0) {
            rlin = kmax + 64; if (rlin > D) rlin = D;
            zl_svd_lowrank(Wz, D, D, rlin, &A, &S, &Bt);
        }
        free(Wz);
    }

    /* --- ftA 特征提升支线: φ(x)=[x, x⊙x/rms, relu(x)], 部署 din=3D --- */
    float *Phi_ev32 = NULL;
    const int DIN3 = 3 * D;
    if (FTA) {
        float *Fa = (float *)xmalloc((size_t)NN * DIN3 * 4);
        { phi64_ctx pc = {Xa, Fa, D}; parallel_for(NN, phi64_worker, &pc); }
        float *Gf32 = (float *)xmalloc((size_t)NN * NN * 4);
        mm32(0, 1, NN, NN, DIN3, Fa, DIN3, Fa, DIN3, Gf32, NN);     /* py: Fa 是 f32 → sgemm */
        double trc = 0;
        for (int i = 0; i < NN; i++) trc += Gf32[(size_t)i * NN + i];
        float rg = (float)(3.0 * (double)(float)trc / DIN3 + 1e-10);
        for (int i = 0; i < NN; i++) Gf32[(size_t)i * NN + i] += rg;
        double *Gf = (double *)xmalloc((size_t)NN * NN * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * NN; i++) Gf[i] = Gf32[i];
        free(Gf32);
        double *alf = (double *)xmalloc((size_t)NN * D * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * D; i++) alf[i] = Ra32[i];
        if (chol_solve(Gf, NN, alf, D)) die("ftA 支线 Cholesky 失败 — 停车");
        free(Gf);
        double *Fa64 = (double *)xmalloc((size_t)NN * DIN3 * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * DIN3; i++) Fa64[i] = Fa[i];
        free(Fa);
        double *Wf = (double *)xmalloc((size_t)DIN3 * D * sizeof(double));
        mm64(1, 0, DIN3, D, NN, Fa64, DIN3, alf, D, Wf, D);
        free(Fa64); free(alf);
        for (int i = 0; i < DIN3; i++) { double *row = Wf + (size_t)i * D;
            for (int j = 0; j < D; j++) row[j] /= colw[j]; }
        if (kmax > 0) {
            rfta = kmax + 64; if (rfta > D) rfta = D;
            zl_svd_lowrank(Wf, DIN3, D, rfta, &Af, &Sf, &Bf);
        }
        free(Wf);
        Phi_ev32 = (float *)xmalloc((size_t)nev * DIN3 * 4);
        float *Xev32 = (float *)xmalloc((size_t)nev * D * 4);
        for (int i = 0; i < nev; i++) for (int j = 0; j < D; j++) Xev32[(size_t)i * D + j] = (float)X[(size_t)ev[i] * D + j];
        { phi32_ctx pc = {Xev32, Phi_ev32, D}; parallel_for(nev, phi32_worker, &pc); }
        free(Xev32);
    }
    free(Xa);

    /* --- 活 k_L: held 段 k 曲线 --- */
    double e0 = 0;
    for (int i = 0; i < nev; i++) { const float *d2 = dH + (size_t)ev[i] * D;
        for (int j = 0; j < D; j++) e0 += (double)d2[j] * d2[j]; }

    double curve[16], curvef[16];
    for (int i = 0; i < nKG; i++) { curve[i] = 0.0; curvef[i] = 0.0; }
    {
        double *pbuf = (double *)xmalloc((size_t)nev * D * sizeof(double));
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1 && !(FTA && rfta > 0)) break;
            int r = pass ? rfta : rlin, din = pass ? DIN3 : D;
            const double *AA = pass ? Af : A, *SS = pass ? Sf : S, *BB = pass ? Bf : Bt;
            if (r <= 0) continue;
            double *Ph = (double *)xmalloc((size_t)nev * din * sizeof(double));
            if (pass) { for (size_t i = 0; i < (size_t)nev * din; i++) Ph[i] = Phi_ev32[i]; }
            else { for (int i = 0; i < nev; i++) memcpy(Ph + (size_t)i * D, X + (size_t)ev[i] * D, (size_t)D * sizeof(double)); }
            double *AS = (double *)xmalloc((size_t)din * r * sizeof(double));
            for (int i = 0; i < din; i++) for (int c = 0; c < r; c++) AS[(size_t)i * r + c] = AA[(size_t)i * r + c] * SS[c];
            double *Pev = (double *)xmalloc((size_t)nev * r * sizeof(double));
            mm64(0, 0, nev, r, din, Ph, din, AS, r, Pev, r);
            free(Ph); free(AS);
            for (int ki = 0; ki < nKG; ki++) {
                int k = KG[ki];
                if (k > 0) mm64(0, 0, nev, D, k, Pev, r, BB, D, pbuf, D);
                double e1 = 0;
                for (int i = 0; i < nev; i++) {
                    const float *d2 = dH + (size_t)ev[i] * D;
                    const double *p = pbuf + (size_t)i * D;
                    for (int j = 0; j < D; j++) {
                        float res = d2[j] - (k > 0 ? (float)p[j] : 0.0f);   /* py: 先降 f32 再平方 */
                        e1 += (double)res * res;
                    }
                }
                double v = 1.0 - e1 / e0;
                if (pass) curvef[ki] = v; else curve[ki] = v;
            }
            free(Pev);
        }
        free(pbuf);
    }

    double rz_lin = -1e300, rz_fta = -1e300;
    for (int i = 0; i < nKG; i++) { if (curve[i] > rz_lin) rz_lin = curve[i]; if (curvef[i] > rz_fta) rz_fta = curvef[i]; }
    if (!FTA) { rz_fta = 0.0; for (int i = 0; i < nKG; i++) curvef[i] = 0.0; }
    const char *FORM = "lin";
    double rz;
    if ((rz_lin > rz_fta ? rz_lin : rz_fta) <= GATE) { K = 0; rz = 0.0; }
    else if (rz_fta > rz_lin) {
        FORM = "ftA"; K = KG[0]; rz = curvef[0];
        for (int i = 1; i < nKG; i++) if (curvef[i] > rz) { rz = curvef[i]; K = KG[i]; }
    } else {
        K = KG[0]; rz = curve[0];
        for (int i = 1; i < nKG; i++) if (curve[i] > rz) { rz = curve[i]; K = KG[i]; }
    }
    printf("  L%d k曲线lin", L);
    for (int i = 0; i < nKG; i++) printf("%s%d:%.1f", i ? " " : " ", KG[i], curve[i] * 100);
    printf("\n");
    if (FTA) {
        printf("  L%d k曲线ftA", L);
        for (int i = 0; i < nKG; i++) printf("%s%d:%.1f", i ? " " : " ", KG[i], curvef[i] * 100);
        printf(" → 赢家=%s k_L=%d\n", FORM, K);
    } else printf("  L%d 纯z(ftA关) → k_L=%d\n", L, K);

    /* R = dH − 中标形态的 z 出力(f32, 与 py 的 .astype(np.float32) 同位置) */
    float *R = dH;
    if (K > 0) {
        int din = !strcmp(FORM, "ftA") ? DIN3 : D;
        int r = !strcmp(FORM, "ftA") ? rfta : rlin;
        const double *AA = !strcmp(FORM, "ftA") ? Af : A, *SS = !strcmp(FORM, "ftA") ? Sf : S,
                     *BB = !strcmp(FORM, "ftA") ? Bf : Bt;
        double *Ph = (double *)xmalloc((size_t)NTOK * din * sizeof(double));
        if (din == DIN3) {
            float *X32 = (float *)xmalloc((size_t)NTOK * D * 4);
            for (size_t i = 0; i < (size_t)NTOK * D; i++) X32[i] = (float)X[i];
            float *P32 = (float *)xmalloc((size_t)NTOK * DIN3 * 4);
            { phi32_ctx pc = {X32, P32, D}; parallel_for(NTOK, phi32_worker, &pc); }
            for (size_t i = 0; i < (size_t)NTOK * DIN3; i++) Ph[i] = P32[i];
            free(X32); free(P32);
        } else memcpy(Ph, X, (size_t)NTOK * D * sizeof(double));
        double *AS = (double *)xmalloc((size_t)din * K * sizeof(double));
        for (int i = 0; i < din; i++) for (int c = 0; c < K; c++) AS[(size_t)i * K + c] = AA[(size_t)i * r + c] * SS[c];
        double *Pv = (double *)xmalloc((size_t)NTOK * K * sizeof(double));
        mm64(0, 0, NTOK, K, din, Ph, din, AS, K, Pv, K);
        free(Ph); free(AS);
        double *prj = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
        mm64(0, 0, NTOK, D, K, Pv, K, BB, D, prj, D);
        free(Pv);
        R = (float *)xmalloc((size_t)NTOK * D * 4);
        for (size_t i = 0; i < (size_t)NTOK * D; i++) R[i] = dH[i] - (float)prj[i];
        free(prj);
    }

    /* tok_pairs: 每 token 的配对下标(CSR), 与 py 的 append 顺序一致(p 升序) */
    int *pcnt = (int *)xcalloc((size_t)NTOK + 1, sizeof(int));
    for (long long p = 0; p < npair; p++) pcnt[prow[p] + 1]++;
    for (int t = 0; t < NTOK; t++) pcnt[t + 1] += pcnt[t];
    int *pidx = (int *)xmalloc((size_t)(npair ? npair : 1) * sizeof(int));
    { int *fill = (int *)xmalloc(((size_t)NTOK + 1) * sizeof(int));
      memcpy(fill, pcnt, ((size_t)NTOK + 1) * sizeof(int));
      for (long long p = 0; p < npair; p++) pidx[fill[prow[p]]++] = (int)p;
      free(fill); }
    int maxpp = 0;
    for (int t = 0; t < NTOK; t++) if (pcnt[t + 1] - pcnt[t] > maxpp) maxpp = pcnt[t + 1] - pcnt[t];

    /* ---------------- GE: 每专家门(k 截断后残差 ridge) ---------------- */
    uint16_t ge16[NEXP];
    float gf[NEXP];
    double comb, eff;
    int USE_GE = 0;
    if (GE_ON) {
        double *Gg = (double *)xcalloc(NEXP * NEXP, sizeof(double));
        double *bg = (double *)xcalloc(NEXP, sizeof(double));
        double *va = (double *)xmalloc((size_t)(maxpp ? maxpp : 1) * D * sizeof(double));
        int *ve = (int *)xmalloc((size_t)(maxpp ? maxpp : 1) * sizeof(int));
        for (int i = 0; i < ntr; i++) {
            int t = tr[i], n = pcnt[t + 1] - pcnt[t];
            for (int a = 0; a < n; a++) {
                int p = pidx[pcnt[t] + a];
                ve[a] = pe[p];
                const float *y = pYQ + (size_t)p * D;
                double *v = va + (size_t)a * D;
                float w = pw[p];
                for (int j = 0; j < D; j++) v[j] = (double)(w * y[j]);   /* py: f32 乘积再升 f64 */
            }
            const float *rt = R + (size_t)t * D;
            for (int a = 0; a < n; a++) {
                const double *v = va + (size_t)a * D;
                double s = 0;
                for (int j = 0; j < D; j++) s += v[j] * rt[j];
                bg[ve[a]] += s;
                for (int b = a; b < n; b++) {
                    const double *vb = va + (size_t)b * D;
                    double d2 = 0;
                    for (int j = 0; j < D; j++) d2 += v[j] * vb[j];
                    Gg[(size_t)ve[a] * NEXP + ve[b]] += d2;
                    if (ve[a] != ve[b]) Gg[(size_t)ve[b] * NEXP + ve[a]] += d2;
                }
            }
        }
        free(va); free(ve);
        double gtr = 0;
        for (int i = 0; i < NEXP; i++) gtr += Gg[(size_t)i * NEXP + i];
        double shr = GELAM * (gtr / NEXP > 1.0 ? gtr / NEXP : 1.0);
        for (int i = 0; i < NEXP; i++) Gg[(size_t)i * NEXP + i] += shr;
        if (lu_solve(Gg, NEXP, bg)) die("GE 解算奇异 — 停车");
        for (int e = 0; e < NEXP; e++) { ge16[e] = f64_to_f16(1.0 + bg[e]); gf[e] = f16_to_f32(ge16[e]); }
        free(Gg); free(bg);
        /* 组合终验: z^L(K)+GE 在 held 段的总增益 */
        double eng = 0;
        double *rr = (double *)xmalloc((size_t)D * sizeof(double));
        for (int i = 0; i < nev; i++) {
            int t = ev[i];
            const float *rt = R + (size_t)t * D;
            for (int j = 0; j < D; j++) rr[j] = rt[j];
            for (int a = pcnt[t]; a < pcnt[t + 1]; a++) {
                int p = pidx[a];
                double g1 = (double)gf[pe[p]] - 1.0;
                const float *y = pYQ + (size_t)p * D;
                float w = pw[p];
                for (int j = 0; j < D; j++) rr[j] -= g1 * (double)(w * y[j]);
            }
            for (int j = 0; j < D; j++) eng += rr[j] * rr[j];
        }
        free(rr);
        comb = 1.0 - eng / e0;
        USE_GE = (comb >= rz - 1e-9) || (K == 0);
    } else {
        for (int e = 0; e < NEXP; e++) { ge16[e] = f64_to_f16(1.0); gf[e] = 1.0f; }
        comb = rz; USE_GE = 0;
    }
    eff = USE_GE ? comb : rz;

    /* 分域终验: held 前半=prog 后半=fin(注意 py 这里【总是】叠 GE 效应, GE 关时 gf≡1 无害) */
    if (nev >= 2) {
        int half = nev / 2;
        double dv[2];
        for (int side = 0; side < 2; side++) {
            const int *idx = side ? ev + half : ev;
            int n = side ? nev - half : half;
            if (n == 0) { dv[side] = NAN; continue; }
            double e0d = 0;
            for (int i = 0; i < n; i++) { const float *d2 = dH + (size_t)idx[i] * D;
                for (int j = 0; j < D; j++) e0d += (double)d2[j] * d2[j]; }
            if (e0d <= 0) { dv[side] = NAN; continue; }
            double ed = 0;
            double *rr = (double *)xmalloc((size_t)D * sizeof(double));
            for (int i = 0; i < n; i++) {
                int t = idx[i];
                const float *rt = (K == 0 ? dH : R) + (size_t)t * D;
                for (int j = 0; j < D; j++) rr[j] = rt[j];
                for (int a = pcnt[t]; a < pcnt[t + 1]; a++) {
                    int p = pidx[a];
                    double g1 = (double)gf[pe[p]] - 1.0;
                    const float *y = pYQ + (size_t)p * D;
                    float w = pw[p];
                    for (int j = 0; j < D; j++) rr[j] -= g1 * (double)(w * y[j]);
                }
                for (int j = 0; j < D; j++) ed += rr[j] * rr[j];
            }
            free(rr);
            dv[side] = 1.0 - ed / e0d;
        }
        printf("  L%d 分域组合: prog=%.1f%%  fin=%.1f%%  [组件门: z=%.1f z+GE=%.1f → %s]\n",
               L, dv[0] * 100, dv[1] * 100, rz * 100, comb * 100,
               (USE_GE && K > 0) ? "z+GE" : (K == 0 ? "GE-only" : "纯z"));
    }
    double t2 = now_s();

    /* ---------------- 注入载荷 ---------------- */
    uint8_t *add = NULL; size_t add_len = 0; int nrec_add = 0;
    if (USE_GE) {
        size_t rl; uint8_t *r = make_rec("bf.GE", ge16, sizeof ge16, &rl);
        add = (uint8_t *)realloc(add, add_len + rl); if (!add) die("realloc");
        memcpy(add + add_len, r, rl); add_len += rl; free(r);
        nrec_add++;
    }
    if (K > 0) {
        int ftaW = !strcmp(FORM, "ftA");
        int din = ftaW ? DIN3 : D, r = ftaW ? rfta : rlin;
        const double *AA = ftaW ? Af : A, *SS = ftaW ? Sf : S, *BB = ftaW ? Bf : Bt;
        size_t nh = (size_t)K + (size_t)D * K + (size_t)din * K;
        size_t psz = 16 + nh * 2;
        uint8_t *pay = (uint8_t *)xmalloc(psz);
        uint32_t u32k = (uint32_t)K, u32di = (uint32_t)din, u32do = (uint32_t)D;
        /* ★tr 槽写 0.5★ 引擎 type6 拿它当信任域上限(‖z 出力‖ ≤ tr·‖routed‖), 不是"关闭夹持"
         * 的大数 —— 今晨定的契约, 写 1e6 等于事实上无夹持。 */
        float tr05 = 0.5f;
        memcpy(pay, &u32k, 4); memcpy(pay + 4, &tr05, 4);
        memcpy(pay + 8, &u32di, 4); memcpy(pay + 12, &u32do, 4);
        uint16_t *h = (uint16_t *)(pay + 16);
        for (int c = 0; c < K; c++) h[c] = f64_to_f16(SS[c]);                       /* z[K] */
        uint16_t *U16 = h + K, *V16 = h + K + (size_t)D * K;
        for (int j = 0; j < D; j++) for (int c = 0; c < K; c++)
            U16[(size_t)j * K + c] = f64_to_f16(BB[(size_t)c * D + j]);             /* U=Bt[:K].T [D,K] */
        for (int i = 0; i < din; i++) for (int c = 0; c < K; c++)
            V16[(size_t)i * K + c] = f64_to_f16(AA[(size_t)i * r + c]);             /* V=A[:,:K] [din,K] */
        size_t rl; uint8_t *rec = make_rec("zl.RRR", pay, psz, &rl);
        free(pay);
        add = (uint8_t *)realloc(add, add_len + rl); if (!add) die("realloc");
        memcpy(add + add_len, rec, rl); add_len += rl; free(rec);
        nrec_add++;
    }

    /* ERF 死层部件: 第一期未实现。只在 .py 会进这个分支时提示, 免得日志里凭空多一行。 */
    if (eff < 0.01 && env_int("DS4_ZL_ERF", 1))
        printf("  L%d ERF死层部件: 第一期未实现, 跳过(组合 %.1f%% < 1%%)\n", L, eff * 100);

    char status[512]; snprintf(status, sizeof status, "解算完");
    if (INJ == 2) {
        snprintf(path, sizeof path, "%s/zrec_L%02d.bin", ld, L);
        FILE *zf = fopen(path, "wb");
        if (!zf) die("zrec 写不开: %s", path);
        if (eff <= GATE) {
            snprintf(status, sizeof status, "组合增益 %.1f%% ≤闸%.1f%% → 空 zrec(skip 标记)", comb * 100, GATE * 100);
        } else {
            if (add_len && fwrite(add, 1, add_len, zf) != add_len) die("zrec 写失败");
            snprintf(status, sizeof status, "zrec 落盘(+%d记录 %.1fMB)", nrec_add, add_len / 1048576.0);
        }
        fclose(zf);
    } else if (INJ) {
        char man[1300]; snprintf(man, sizeof man, "%s/zinject_manifest.txt", ld);
        char mlk[1400]; snprintf(mlk, sizeof mlk, "%s.lock", man);
        int lkfd = open(mlk, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lkfd < 0) die("账本锁打不开: %s", mlk);
        if (flock(lkfd, LOCK_EX)) die("账本上锁失败");     /* ★双路并发: 账本读写全程持锁 */
        int done = 0;
        FILE *mf = fopen(man, "r");
        if (mf) { char line[256];
            /* 空行/垃圾行不能当成 "0" —— 否则 L0 会被误判成已注入过而整层跳过 */
            while (fgets(line, sizeof line, mf)) {
                const char *q = line;
                while (*q == ' ' || *q == '\t') q++;
                if (!(*q == '-' || (*q >= '0' && *q <= '9'))) continue;
                if (atoi(q) == L) done = 1;
            }
            fclose(mf); }
        if (done) {
            snprintf(status, sizeof status, "已注入过, 跳过(回滚请按账本截断)");
        } else if (eff <= GATE) {
            mf = fopen(man, "a"); if (!mf) die("账本写不开");
            fprintf(mf, "%d -1 -1\n", L); fclose(mf);
            snprintf(status, sizeof status, "组合增益 %.1f%% ≤闸%.1f%% → 本层不注入(闸)", comb * 100, GATE * 100);
        } else {
            char dql[1300]; snprintf(dql, sizeof dql, "%s/dql_L%02d.bin", ld, L);
            struct stat ds;
            if (stat(dql, &ds)) die("dql 不存在: %s", dql);
            long long osz = (long long)ds.st_size;
            FILE *df = fopen(dql, "r+b");
            if (!df) die("dql 打不开(r+b): %s", dql);
            uint32_t n0;
            if (fseeko(df, 8, SEEK_SET) || fread(&n0, 4, 1, df) != 1) die("dql nrec 读不到");
            if (fseeko(df, 0, SEEK_END)) die("dql seek end 失败");
            if (add_len && fwrite(add, 1, add_len, df) != add_len) die("dql 追加失败");
            uint32_t n1 = n0 + (uint32_t)nrec_add;
            if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
            fclose(df);
            mf = fopen(man, "a"); if (!mf) die("账本写不开");
            fprintf(mf, "%d %lld %u\n", L, osz, n0); fclose(mf);
            snprintf(status, sizeof status, "注入完(+%d记录, 原长 %lld 入账本)", nrec_add, osz);
        }
        flock(lkfd, LOCK_UN); close(lkfd);
    }
    double t3 = now_s();

    double gemean;
    { float acc = 0.0f;                       /* numpy 对 f16 数组的 mean 用 f32 累加器 */
      for (int e = 0; e < NEXP; e++) acc += f16_to_f32(ge16[e]);
      gemean = (double)(acc / (float)NEXP); }
    printf("★L%d z侧车: held挽回 z^L %.1f%% 组合 %.1f%%  GE均值 %.4f  体积 %.1fMB | "
           "缓存 %.0fs 解算 %.0fs 总 %.0fs | %s\n",
           L, rz * 100, comb * 100, gemean, add_len / 1048576.0,
           t1 - t0, t2 - t1, t3 - t0, status);
    return 0;
}
