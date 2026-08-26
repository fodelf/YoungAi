/* ds4_z.c -- z 隐变量: closed-form low-rank latent correction.
 * Self-contained (stdlib only); see ds4_z.h for the module contract. */
#include "ds4_z.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* DQ_BLAS 快路(2026-08-26): 正规方程/Cholesky 换 BLAS/LAPACK 分块。数学不变,
 * 只有累加顺序的尾位差(zlayer chol_solve 36× 先例同款)。引擎构建不定义 DQ_BLAS,
 * 模块对引擎保持 stdlib-only; 反修解算器(zloss_solve)开 DQ_BLAS —— 4096² 正规
 * 方程纯标量 ~5min/层, 43 层等一夜, 不开就是浪费设备。 */
#ifdef DQ_BLAS
#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define DZ_DPOTRF dpotrf_
#define DZ_DPOTRS dpotrs_
#else
#include <cblas.h>
extern void scipy_dpotrf_(const char *, const int *, double *, const int *, int *);
extern void scipy_dpotrs_(const char *, const int *, const int *, const double *,
                          const int *, double *, const int *, int *);
#define DZ_DPOTRF scipy_dpotrf_
#define DZ_DPOTRS scipy_dpotrs_
#endif
#ifdef DQ_CUDA
/* zlayer_gpu.cu 的行主序 GPU 入口(2026-08-26 提速令"必须用GPU"): 能力探测契约,
 * 返 0 自动回 cblas/LAPACK。QR 正交化与 mgs 产的基不同但张成同一子空间 ——
 * U·diag(z)·Vᵀ 对基不变, 金标口径=selftest 收回率+针表打印精度。 */
extern int zg_dgemm(int, int, int, int, int, const double *, int, const double *, int,
                    double *, int);
extern int zg_sgemm(int, int, int, int, int, const float *, int, const float *, int,
                    float *, int);
extern int zg_spotrf_potrs_f64io(int n, int nrhs, const double *A, double *B);
extern int zg_sqr_orth(int d, int k, float *V);
#endif
static void dz_dgemm(int ta, int M, int N, int K, const double *A, int lda,
                     const double *B, int ldb, double *C, int ldc) {
#ifdef DQ_CUDA
    if (zg_dgemm(ta, 0, M, N, K, A, lda, B, ldb, C, ldc)) return;
#endif
    cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
}
static void dz_sgemm(int ta, int M, int N, int K, const float *A, int lda,
                     const float *B, int ldb, float *C, int ldc) {
#ifdef DQ_CUDA
    if (zg_sgemm(ta, 0, M, N, K, A, lda, B, ldb, C, ldc)) return;
#endif
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}
#endif

/* Deterministic LCG (Numerical Recipes constants): subspace-iteration init
 * must be reproducible, so no rand()/time seeding anywhere in this module. */
static inline uint32_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(*s >> 32);
}
static inline float lcg_unit(uint64_t *s) {
    return ((float)lcg_next(s) / 4294967296.0f) * 2.0f - 1.0f;   /* [-1,1) */
}

/* In-place Cholesky A = L L^T (A: d x d, lower triangle used). Returns 0, or
 * -1 if A is not positive definite (ridge lambda too small / degenerate X). */
static int cholesky(double *A, uint32_t d) {
    for (uint32_t j = 0; j < d; j++) {
        double diag = A[(size_t)j * d + j];
        for (uint32_t k = 0; k < j; k++) {
            double v = A[(size_t)j * d + k];
            diag -= v * v;
        }
        if (diag <= 0.0) return -1;
        diag = sqrt(diag);
        A[(size_t)j * d + j] = diag;
        for (uint32_t i = j + 1; i < d; i++) {
            double v = A[(size_t)i * d + j];
            for (uint32_t k = 0; k < j; k++) {
                v -= A[(size_t)i * d + k] * A[(size_t)j * d + k];
            }
            A[(size_t)i * d + j] = v / diag;
        }
    }
    return 0;
}

/* Solve L L^T w = b in place (b becomes w). */
static void cholesky_solve(const double *L, uint32_t d, double *b) {
    for (uint32_t i = 0; i < d; i++) {          /* forward: L y = b */
        double v = b[i];
        for (uint32_t k = 0; k < i; k++) v -= L[(size_t)i * d + k] * b[k];
        b[i] = v / L[(size_t)i * d + i];
    }
    for (uint32_t ii = d; ii-- > 0; ) {         /* backward: L^T w = y */
        double v = b[ii];
        for (uint32_t k = ii + 1; k < d; k++) v -= L[(size_t)k * d + ii] * b[k];
        b[ii] = v / L[(size_t)ii * d + ii];
    }
}

/* Modified Gram-Schmidt orthonormalization of the k columns of Q (d x k)
 * with reorthogonalization ("twice is enough", Kahan): one float pass leaves
 * cancellation noise ALIGNED with the earlier columns whenever the residual
 * is tiny -- and subspace iteration on a low-rank W drives trailing columns
 * into exactly that regime -- so a second projection pass is mandatory for a
 * usable basis. Columns whose residual collapses below 1e-4 of their
 * pre-projection norm carry no independent direction: re-seed them
 * deterministically (bounded retries; a genuinely exhausted column is left
 * zero, which downstream turns into z=0, a harmless dead direction). */
static void mgs(float *Q, uint32_t d, uint32_t k, uint64_t *seed) {
    for (uint32_t j = 0; j < k; j++) {
        for (int attempt = 0; ; attempt++) {
            double pre = 0.0;
            for (uint32_t i = 0; i < d; i++) {
                double v = Q[(size_t)i * k + j];
                pre += v * v;
            }
            pre = sqrt(pre);
            for (int pass = 0; pass < 2; pass++) {
                for (uint32_t p = 0; p < j; p++) {
                    double dot = 0.0;
                    for (uint32_t i = 0; i < d; i++)
                        dot += (double)Q[(size_t)i * k + p] * Q[(size_t)i * k + j];
                    for (uint32_t i = 0; i < d; i++)
                        Q[(size_t)i * k + j] -= (float)dot * Q[(size_t)i * k + p];
                }
            }
            double nrm = 0.0;
            for (uint32_t i = 0; i < d; i++) {
                double v = Q[(size_t)i * k + j];
                nrm += v * v;
            }
            nrm = sqrt(nrm);
            if (nrm > 1e-4 * pre && nrm > 1e-30) {
                for (uint32_t i = 0; i < d; i++) Q[(size_t)i * k + j] /= (float)nrm;
                break;
            }
            if (attempt >= 8) {                 /* exhausted: dead direction */
                for (uint32_t i = 0; i < d; i++) Q[(size_t)i * k + j] = 0.0f;
                break;
            }
            for (uint32_t i = 0; i < d; i++) Q[(size_t)i * k + j] = lcg_unit(seed);
        }
    }
}

/* 正交化入口: DQ_CUDA 有卡走 GPU QR(失败自动回落), 否则 mgs 标量参考路 */
static void dz_orth(float *V, uint32_t d, uint32_t k, uint64_t *seed) {
#if defined(DQ_BLAS) && defined(DQ_CUDA)
    if (zg_sqr_orth((int)d, (int)k, V)) return;
#endif
    mgs(V, d, k, seed);
}

/* ---- 秩截断: W(d_in×d_out) 的子空间迭代取 top-rank; W 被本函数消费(释放)。
 * V converges to the top-k left singular subspace of W, then
 * M = V^T W = diag(z) U^T. DQ_BLAS 下三个大矩阵乘走 sgemm(2026-08-26: 标量
 * 截断段单线程 100-300 GFLOP/次是针跑法的第二浪费源; mgs 仍标量, 占比小)。 */
static ds4_z *zl_truncate(float *W, uint32_t d_in, uint32_t d_out, uint32_t rank) {
    ds4_z *zl = calloc(1, sizeof(*zl));
    float *V = malloc((size_t)d_in * rank * sizeof(float));
    float *T = malloc((size_t)d_out * rank * sizeof(float));   /* W^T V */
    float *M = malloc((size_t)rank * d_out * sizeof(float));
    float *U = malloc((size_t)d_out * rank * sizeof(float));
    float *z = malloc((size_t)rank * sizeof(float));
    if (!zl || !V || !T || !M || !U || !z) {
        free(zl); free(V); free(T); free(M); free(U); free(z); free(W);
        return NULL;
    }
    uint64_t seed = 0x5A5A1EEDULL;
    for (size_t i = 0; i < (size_t)d_in * rank; i++) V[i] = lcg_unit(&seed);
    dz_orth(V, d_in, rank, &seed);
    for (int it = 0; it < 12; it++) {           /* 12 iters: ample for k<<d */
#ifdef DQ_BLAS
        /* T = WᵀV (d_out×k); V = W·T (d_in×k) — gemm, 数学同标量路(尾位差) */
        dz_sgemm(1, (int)d_out, (int)rank, (int)d_in, W, (int)d_out, V, (int)rank,
                 T, (int)rank);
        dz_sgemm(0, (int)d_in, (int)rank, (int)d_out, W, (int)d_out, T, (int)rank,
                 V, (int)rank);
#else
        /* T = W^T V  (d_out x k) */
        memset(T, 0, (size_t)d_out * rank * sizeof(float));
        for (uint32_t i = 0; i < d_in; i++) {
            const float *Wi = W + (size_t)i * d_out;
            const float *Vi = V + (size_t)i * rank;
            for (uint32_t j = 0; j < d_out; j++) {
                const float wij = Wi[j];
                if (wij == 0.0f) continue;
                float *Tj = T + (size_t)j * rank;
                for (uint32_t c = 0; c < rank; c++) Tj[c] += wij * Vi[c];
            }
        }
        /* V = W T  (d_in x k), then re-orthonormalize */
        for (uint32_t i = 0; i < d_in; i++) {
            const float *Wi = W + (size_t)i * d_out;
            float *Vi = V + (size_t)i * rank;
            for (uint32_t c = 0; c < rank; c++) Vi[c] = 0.0f;
            for (uint32_t j = 0; j < d_out; j++) {
                const float wij = Wi[j];
                if (wij == 0.0f) continue;
                const float *Tj = T + (size_t)j * rank;
                for (uint32_t c = 0; c < rank; c++) Vi[c] += wij * Tj[c];
            }
        }
#endif
        dz_orth(V, d_in, rank, &seed);
    }
    /* M = V^T W (k x d_out); z = row norms of M (descending by construction
     * up to iteration accuracy); U rows = normalized M rows. */
#ifdef DQ_BLAS
    dz_sgemm(1, (int)rank, (int)d_out, (int)d_in, V, (int)rank, W, (int)d_out,
             M, (int)d_out);
#else
    memset(M, 0, (size_t)rank * d_out * sizeof(float));
    for (uint32_t i = 0; i < d_in; i++) {
        const float *Wi = W + (size_t)i * d_out;
        const float *Vi = V + (size_t)i * rank;
        for (uint32_t c = 0; c < rank; c++) {
            const float v = Vi[c];
            if (v == 0.0f) continue;
            float *Mc = M + (size_t)c * d_out;
            for (uint32_t j = 0; j < d_out; j++) Mc[j] += v * Wi[j];
        }
    }
#endif
    for (uint32_t c = 0; c < rank; c++) {
        double nrm = 0.0;
        const float *Mc = M + (size_t)c * d_out;
        for (uint32_t j = 0; j < d_out; j++) nrm += (double)Mc[j] * Mc[j];
        nrm = sqrt(nrm);
        z[c] = (float)nrm;
        const float inv = nrm > 1e-20 ? (float)(1.0 / nrm) : 0.0f;
        for (uint32_t j = 0; j < d_out; j++) U[(size_t)j * rank + c] = Mc[j] * inv;
    }
    free(M);
    free(T);
    free(W);

    /* Sort directions by |z| descending so ds4_z_set_rank(k) keeps the top-k
     * (subspace iteration converges the subspace, not the ordering). */
    for (uint32_t a = 0; a < rank; a++) {
        uint32_t best = a;
        for (uint32_t b = a + 1; b < rank; b++) if (z[b] > z[best]) best = b;
        if (best != a) {
            float tz = z[a]; z[a] = z[best]; z[best] = tz;
            for (uint32_t i = 0; i < d_in; i++) {
                float tv = V[(size_t)i * rank + a];
                V[(size_t)i * rank + a] = V[(size_t)i * rank + best];
                V[(size_t)i * rank + best] = tv;
            }
            for (uint32_t j = 0; j < d_out; j++) {
                float tu = U[(size_t)j * rank + a];
                U[(size_t)j * rank + a] = U[(size_t)j * rank + best];
                U[(size_t)j * rank + best] = tu;
            }
        }
    }

    zl->U = U;
    zl->V = V;
    zl->z = z;
    zl->d_in = d_in;
    zl->d_out = d_out;
    zl->rank = rank;
    zl->k = rank;
    return zl;
}

int ds4_z_solve_multi(const float *X, const float *R, uint32_t n,
                      uint32_t d_in, uint32_t d_out, uint32_t rank,
                      const float *lambdas, uint32_t nl, ds4_z **out) {
    if (!X || !R || !lambdas || !out || nl == 0 ||
        n == 0 || d_in == 0 || d_out == 0 || rank == 0) return -1;
    if (rank > d_in) rank = d_in;
    if (rank > d_out) rank = d_out;
    for (uint32_t i = 0; i < nl; i++) out[i] = NULL;

    /* Normal equations in double: G = X^T X (d_in x d_in), B = X^T R
     * (d_in x d_out) — 与 λ 无关, 只算一次。Accumulation order is fixed. */
    double *G = calloc((size_t)d_in * d_in, sizeof(double));
    double *B = calloc((size_t)d_in * d_out, sizeof(double));
    if (!G || !B) { free(G); free(B); return -1; }
    int gram_done = 0; (void)gram_done;  /* 标量路不消费 */
#if defined(DQ_BLAS) && defined(DQ_CUDA)
    {   /* f32 Gram 上卡(GB10 f64=1:64; zlayer ftA f32 Gram 冠军先例), 升 f64 进解 */
        float *Gf = malloc((size_t)d_in * d_in * 4), *Bf = malloc((size_t)d_in * d_out * 4);
        if (Gf && Bf &&
            zg_sgemm(1, 0, (int)d_in, (int)d_in, (int)n, X, (int)d_in, X, (int)d_in, Gf, (int)d_in) &&
            zg_sgemm(1, 0, (int)d_in, (int)d_out, (int)n, X, (int)d_in, R, (int)d_out, Bf, (int)d_out)) {
            for (size_t i = 0; i < (size_t)d_in * d_in; i++) G[i] = Gf[i];
            for (size_t i = 0; i < (size_t)d_in * d_out; i++) B[i] = Bf[i];
            gram_done = 1;
        }
        free(Gf); free(Bf);
    }
#endif
#ifdef DQ_BLAS
    if (!gram_done) {   /* G = XᵀX, B = XᵀR — dgemm 双精(输入升 f64 后与标量路同一乘加集合) */
        double *Xd = malloc((size_t)n * d_in * sizeof(double));
        double *Rd = malloc((size_t)n * d_out * sizeof(double));
        if (!Xd || !Rd) { free(Xd); free(Rd); free(G); free(B); return -1; }
        for (size_t i = 0; i < (size_t)n * d_in; i++) Xd[i] = X[i];
        for (size_t i = 0; i < (size_t)n * d_out; i++) Rd[i] = R[i];
        dz_dgemm(1, (int)d_in, (int)d_in, (int)n, Xd, (int)d_in, Xd, (int)d_in,
                 G, (int)d_in);
        dz_dgemm(1, (int)d_in, (int)d_out, (int)n, Xd, (int)d_in, Rd, (int)d_out,
                 B, (int)d_out);
        free(Xd); free(Rd);
    }
#else
    for (uint32_t t = 0; t < n; t++) {
        const float *x = X + (size_t)t * d_in;
        const float *r = R + (size_t)t * d_out;
        for (uint32_t i = 0; i < d_in; i++) {
            const double xi = x[i];
            if (xi == 0.0) continue;
            double *Gi = G + (size_t)i * d_in;
            for (uint32_t j = i; j < d_in; j++) Gi[j] += xi * x[j];
            double *Bi = B + (size_t)i * d_out;
            for (uint32_t j = 0; j < d_out; j++) Bi[j] += xi * r[j];
        }
    }
#endif
    double tr = 0.0;                             /* ridge 量纲基 + 镜像下三角 */
    for (uint32_t i = 0; i < d_in; i++) tr += G[(size_t)i * d_in + i];
    for (uint32_t i = 0; i < d_in; i++)
        for (uint32_t j = i + 1; j < d_in; j++)
            G[(size_t)j * d_in + i] = G[(size_t)i * d_in + j];

    double *Ac = malloc((size_t)d_in * d_in * sizeof(double));
#ifdef DQ_BLAS
    double *Bt0 = malloc((size_t)d_in * d_out * sizeof(double));
    double *Btc = malloc((size_t)d_in * d_out * sizeof(double));
    if (Bt0)
        for (uint32_t i = 0; i < d_in; i++)
            for (uint32_t j = 0; j < d_out; j++)
                Bt0[(size_t)j * d_in + i] = B[(size_t)i * d_out + j];
    int ok = Ac && Bt0 && Btc;
#else
    double *col = malloc((size_t)d_in * sizeof(double));
    int ok = Ac && col;
#endif
    int rc = ok ? 0 : -1;
    for (uint32_t li = 0; rc == 0 && li < nl; li++) {
        memcpy(Ac, G, (size_t)d_in * d_in * sizeof(double));
        const double ridge = (double)lambdas[li] * (tr / (double)d_in) + 1e-10;
        for (uint32_t i = 0; i < d_in; i++) Ac[(size_t)i * d_in + i] += ridge;
        float *W = malloc((size_t)d_in * d_out * sizeof(float));
        if (!W) { rc = -1; break; }
#ifdef DQ_BLAS
        {   /* 分块 potrf/potrs: 行主序对称阵取 uplo='U' 列主序等价; B 转置进出。
             * DQ_CUDA 先试 cusolver(Ac 主机侧不被破坏), 失败回 LAPACK。 */
            int done = 0;
            memcpy(Btc, Bt0, (size_t)d_in * d_out * sizeof(double));
#ifdef DQ_CUDA
            /* GB10 FP64=1:64 阉割, f64 因子化在卡上跟 CPU 一样慢(计时定罪
             * 47s/53s); f32 因子化+无量纲 ridge(条件数≤1/λ)精度富余, 载荷 fp16 */
            done = zg_spotrf_potrs_f64io((int)d_in, (int)d_out, Ac, Btc);
#endif
            if (!done) {
                int info = 0, N = (int)d_in, nrhs = (int)d_out;
                const char up = 'U';
                DZ_DPOTRF(&up, &N, Ac, &N, &info);
                if (info) { free(W); rc = -1; break; }
                DZ_DPOTRS(&up, &N, &nrhs, Ac, &N, Btc, &N, &info);
                if (info) { free(W); rc = -1; break; }
            }
            for (uint32_t i = 0; i < d_in; i++)
                for (uint32_t j = 0; j < d_out; j++)
                    W[(size_t)i * d_out + j] = (float)Btc[(size_t)j * d_in + i];
        }
#else
        if (cholesky(Ac, d_in) != 0) { free(W); rc = -1; break; }
        for (uint32_t j = 0; j < d_out; j++) {   /* solve per output column */
            for (uint32_t i = 0; i < d_in; i++) col[i] = B[(size_t)i * d_out + j];
            cholesky_solve(Ac, d_in, col);
            for (uint32_t i = 0; i < d_in; i++) W[(size_t)i * d_out + j] = (float)col[i];
        }
#endif
        out[li] = zl_truncate(W, d_in, d_out, rank);   /* W 被消费 */
        if (!out[li]) rc = -1;
    }
#ifdef DQ_BLAS
    free(Bt0); free(Btc);
#else
    free(col);
#endif
    free(Ac); free(G); free(B);
    if (rc)
        for (uint32_t i = 0; i < nl; i++) { ds4_z_free(out[i]); out[i] = NULL; }
    return rc;
}

ds4_z *ds4_z_solve(const float *X, const float *R, uint32_t n,
                   uint32_t d_in, uint32_t d_out, uint32_t rank,
                   float lambda) {
    ds4_z *o = NULL;
    return ds4_z_solve_multi(X, R, n, d_in, d_out, rank, &lambda, 1, &o) == 0 ? o : NULL;
}

void ds4_z_apply(const ds4_z *zl, const float *x, float *y) {
    if (!zl || !x || !y || zl->k == 0) return;
    const uint32_t k = zl->k <= zl->rank ? zl->k : zl->rank;
    /* t = diag(z) V^T x  (k) */
    float t[512];
    float *tp = k <= 512 ? t : malloc((size_t)k * sizeof(float));
    if (!tp) return;
    for (uint32_t c = 0; c < k; c++) tp[c] = 0.0f;
    for (uint32_t i = 0; i < zl->d_in; i++) {
        const float xi = x[i];
        if (xi == 0.0f) continue;
        const float *Vi = zl->V + (size_t)i * zl->rank;
        for (uint32_t c = 0; c < k; c++) tp[c] += xi * Vi[c];
    }
    for (uint32_t c = 0; c < k; c++) tp[c] *= zl->z[c];
    /* y += U t */
    for (uint32_t j = 0; j < zl->d_out; j++) {
        const float *Uj = zl->U + (size_t)j * zl->rank;
        float acc = 0.0f;
        for (uint32_t c = 0; c < k; c++) acc += Uj[c] * tp[c];
        y[j] += acc;
    }
    if (tp != t) free(tp);
}

void ds4_z_set_rank(ds4_z *zl, uint32_t k) {
    if (!zl) return;
    zl->k = k <= zl->rank ? k : zl->rank;
}

const float *ds4_z_values(const ds4_z *zl, uint32_t *rank_out) {
    if (rank_out) *rank_out = zl ? zl->rank : 0;
    return zl ? zl->z : NULL;
}

/* "DS4Z" v1: header {magic, version, d_in, d_out, rank, k} u32 LE, then
 * U (d_out*rank), V (d_in*rank), z (rank) as f32. */
int ds4_z_save(const ds4_z *zl, const char *path) {
    if (!zl || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    const uint32_t hdr[6] = {0x5A345344u /* "DS4Z" LE */, 1u,
                             zl->d_in, zl->d_out, zl->rank, zl->k};
    int ok = fwrite(hdr, sizeof(hdr), 1, fp) == 1 &&
             fwrite(zl->U, sizeof(float), (size_t)zl->d_out * zl->rank, fp) ==
                 (size_t)zl->d_out * zl->rank &&
             fwrite(zl->V, sizeof(float), (size_t)zl->d_in * zl->rank, fp) ==
                 (size_t)zl->d_in * zl->rank &&
             fwrite(zl->z, sizeof(float), zl->rank, fp) == zl->rank;
    fclose(fp);
    return ok ? 0 : -1;
}

ds4_z *ds4_z_load(const char *path) {
    FILE *fp = path ? fopen(path, "rb") : NULL;
    if (!fp) return NULL;
    uint32_t hdr[6];
    ds4_z *zl = NULL;
    if (fread(hdr, sizeof(hdr), 1, fp) == 1 &&
        hdr[0] == 0x5A345344u && hdr[1] == 1u &&
        hdr[2] && hdr[3] && hdr[4] && hdr[5] <= hdr[4]) {
        zl = calloc(1, sizeof(*zl));
        if (zl) {
            zl->d_in = hdr[2]; zl->d_out = hdr[3];
            zl->rank = hdr[4]; zl->k = hdr[5];
            zl->U = malloc((size_t)zl->d_out * zl->rank * sizeof(float));
            zl->V = malloc((size_t)zl->d_in * zl->rank * sizeof(float));
            zl->z = malloc((size_t)zl->rank * sizeof(float));
            if (!zl->U || !zl->V || !zl->z ||
                fread(zl->U, sizeof(float), (size_t)zl->d_out * zl->rank, fp) !=
                    (size_t)zl->d_out * zl->rank ||
                fread(zl->V, sizeof(float), (size_t)zl->d_in * zl->rank, fp) !=
                    (size_t)zl->d_in * zl->rank ||
                fread(zl->z, sizeof(float), zl->rank, fp) != zl->rank) {
                ds4_z_free(zl);
                zl = NULL;
            }
        }
    }
    fclose(fp);
    return zl;
}

void ds4_z_free(ds4_z *zl) {
    if (!zl) return;
    free(zl->U);
    free(zl->V);
    free(zl->z);
    free(zl);
}
