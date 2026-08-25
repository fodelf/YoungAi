/* amp_solve.c — 乘性动态 z 放大器逐层闭式解算(C, 2026-08-23)。
 *
 * 铁律落地: 算法一律 C(feedback_c_not_python_shared_impl), 取代 zlever/amp_solve.py
 * (Python 版=技术债, 保留只作交叉对拍)。语义与引擎逐式一致:
 *
 *   引擎(ds4_zchain.c / ds4_cuda.cu type9 zl.AMPD):
 *     routed'_t = routed_t ⊙ (1 + Σ_c U[:,c] · tanh(A_c·φ(x_t)/s) · tanh(V_c·φ(x_t)/s))
 *     φ(x) = [x, x⊙x/rms(x), relu(x)]   (din==3d 时引擎在线展开, 与此处 phi_lift 逐式同)
 *
 *   目标 = ★四损失放大器口径(2026-08-23 用户裁决"是放大器不是修残差")★:
 *     L_align 主损失: 每 token 按 1/‖y*‖ 行归一化进最小二乘 ⇒ MSE 几何→cos 对齐几何
 *       (ALGORITHM.md §4: 对齐损失=1−mean cos, 直接优化方向=真 metric, 最关键);
 *     L_classify 感知列权: colw = y* 的 per-dim 判别方差权(保住驱动下游路由的维度);
 *     L_smooth: dither 增广 ×2(固定种子, 输入扰动下输出不跳变);
 *     L_fixed: 强收缩 ridge λ + 固定种子(钉死欠定方向)。
 *     g 的靶仍是比值域 R = dH·yq/(yq²+eps²)(乘性形式所需), 但行/列权与选择判据
 *     全部换成对齐口径: λ/k 网格判据 = held mean cos(ŷ,y*) 最大化, 报数=对齐度前→后。
 *     V₀/门 = 固定种子随机基 —— 依据: 215 条解算记录
 *     V₀=rand 100% 胜出、门=rnd2 98.6% 胜出, PCA 从未当选 ⇒ SVD 整支砍除。
 *     (随机数不必对齐 numpy: 基向量整块存进载荷, 引擎读的就是解算用的那份。)
 *
 * 输入(与 calib_run 同一 capnpy 目录, 全部引擎解码路真值 + C 教师):
 *   ffn_in_L{L}.npy   [S,D]  x(链态)      obase_v3_L{L}.npy [S,D] yq(部署字节 routed)
 *   routed_L{L}.npy   [S,D]  y*(teacher_routed: FP 专家 @ 引擎轨迹 = FP 自洽锚口径)
 * 输出: <out>/amp_L{NN}.bin —— zrec "zl.AMPD"(116B 头 + k,scale,din,dout + A|U|V fp16),
 *   经 zrec_to_zchain.py(纯字节编排, 不参与数值)合并成 --zchain 文件。
 *
 * 并行/cholesky 助手自带(与 solve_rrr.c 同构; 不改动在用工具的源码)。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "npy.h"

#define DM 4096

/* ---------- 小助手: 并行 for ---------- */
typedef void (*pf_fn)(void *ctx, int i0, int i1);
typedef struct { pf_fn fn; void *ctx; int i0, i1; } pf_arg;
static void *pf_tramp(void *a) { pf_arg *p = (pf_arg *)a; p->fn(p->ctx, p->i0, p->i1); return NULL; }
static void parallel_for(int n, int threads, pf_fn fn, void *ctx) {
    if (threads > n) threads = n > 0 ? n : 1;
    pthread_t th[128]; pf_arg pa[128];
    if (threads > 128) threads = 128;
    int per = (n + threads - 1) / threads, nt = 0;
    for (int t = 0; t < threads; t++) {
        int i0 = t * per, i1 = i0 + per > n ? n : i0 + per;
        if (i0 >= i1) break;
        pa[nt] = (pf_arg){fn, ctx, i0, i1};
        if (pthread_create(&th[nt], NULL, pf_tramp, &pa[nt])) { pa[nt].fn(ctx, i0, i1); continue; }
        nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

/* ---------- 小助手: cholesky 解 (A 对称正定 n×n, B n×nrhs, 原地出 X) ---------- */
typedef struct { const double *A; double *B; int n, nrhs; } tri_ctx;
static void tri_worker(void *vc, int c0, int c1) {
    tri_ctx *t = (tri_ctx *)vc;
    const double *A = t->A; double *B = t->B;
    const int n = t->n, nrhs = t->nrhs;
    for (int c = c0; c < c1; c++) {
        for (int i = 0; i < n; i++) {                 /* L y = b */
            double s = B[(size_t)i * nrhs + c];
            const double *Ai = A + (size_t)i * n;
            for (int k = 0; k < i; k++) s -= Ai[k] * B[(size_t)k * nrhs + c];
            B[(size_t)i * nrhs + c] = s / Ai[i];
        }
        for (int i = n - 1; i >= 0; i--) {            /* Lᵀ x = y */
            double s = B[(size_t)i * nrhs + c];
            for (int k = i + 1; k < n; k++) s -= A[(size_t)k * n + i] * B[(size_t)k * nrhs + c];
            B[(size_t)i * nrhs + c] = s / A[(size_t)i * n + i];
        }
    }
}
static void parallel_for(int n, int threads, void (*fn)(void *, int, int), void *ctx);
static int chol_solve(double *A, int n, double *B, int nrhs, int threads) {
    for (int j = 0; j < n; j++) {                     /* 分解: 单线程(0.4GF, ~1s) */
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
    tri_ctx tc = {A, B, n, nrhs};                     /* 回代: rhs 列独立 → 并行(60GF 大头) */
    parallel_for(nrhs, threads, tri_worker, &tc);
    return 0;
}

/* ---------- 小助手: splitmix64 + Box-Muller 高斯 ---------- */
static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double gauss(uint64_t *s, double *spare, int *has) {
    if (*has) { *has = 0; return *spare; }
    double u1 = ((sm64(s) >> 11) + 1.0) * (1.0 / 9007199254740993.0);
    double u2 = (sm64(s) >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1)), a = 2.0 * M_PI * u2;
    *spare = r * sin(a); *has = 1;
    return r * cos(a);
}

/* ---------- 小助手: f32 → f16 (round-to-nearest-even) ---------- */
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t h = m >> shift;
        if ((m >> (shift - 1)) & 1u) h++;
        return (uint16_t)(sign | h);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if (m & 0x1000u) h++;
    return h;
}

/* ---------- φ 提升: [x, x⊙x/rms, relu(x)] (与引擎 din==3d 分支逐式一致) ---------- */
static void phi_lift(const float *x, float *out, int d) {
    double ss = 0;
    for (int j = 0; j < d; j++) ss += (double)x[j] * x[j];
    float n = (float)sqrt(ss / d) + 1e-6f;
    for (int j = 0; j < d; j++) {
        out[j] = x[j];
        out[d + j] = x[j] * x[j] / n;
        out[2 * d + j] = x[j] > 0 ? x[j] : 0;
    }
}

typedef struct { const float *X0; float *X; int d; } phi_ctx;
static void phi_worker(void *vc, int t0, int t1) {
    phi_ctx *c = (phi_ctx *)vc;
    for (int t = t0; t < t1; t++) phi_lift(c->X0 + (size_t)t * c->d, c->X + (size_t)t * 3 * c->d, c->d);
}

typedef struct { float *B; int din; uint64_t salt; } base_ctx;
static void base_worker(void *vc, int k0, int k1) {
    base_ctx *c = (base_ctx *)vc;
    const float inv_sq = 1.0f / sqrtf((float)c->din);
    for (int k = k0; k < k1; k++) {
        uint64_t sd = c->salt * 0x9E3779B9ULL + (uint64_t)k; double sp = 0; int hs = 0;
        float *row = c->B + (size_t)k * c->din;
        for (int j = 0; j < c->din; j++) row[j] = (float)gauss(&sd, &sp, &hs) * inv_sq;
    }
}

typedef struct { const float *X; float *Xd; int din; uint64_t lsalt; } dith_ctx;
static void dith_worker(void *vc, int t0, int t1) {
    dith_ctx *c = (dith_ctx *)vc;
    for (int t = t0; t < t1; t++) {
        uint64_t sd = 0x5eedULL + c->lsalt * 1000003ULL + (uint64_t)t; double sp = 0; int hs = 0;
        const float *x = c->X + (size_t)t * c->din;
        float *xd = c->Xd + (size_t)t * c->din;
        double ss = 0;
        for (int j = 0; j < c->din; j++) ss += (double)x[j] * x[j];
        float rms = (float)sqrt(ss / c->din);
        for (int j = 0; j < c->din; j++) xd[j] = x[j] + (float)gauss(&sd, &sp, &hs) * 0.04f * rms;
    }
}

/* ---------- 特征 matmul worker: Z[r,c] = tanh(dot(Xrow r, Bt row c)/s) (r 块并行) ---------- */
typedef struct {
    const float *X;      /* [n, din] */
    const float *Vt;     /* [K, din] 行主(基转置) */
    const float *At;     /* [K, din] 门基; NULL=纯 tanh */
    float *Z;            /* [n, K] */
    int din, K;
    float inv_s;
} feat_ctx;
/* c-tile 分块: 每次让 CT 列的 V/A 基(2×CT×din×4B ≈ 3MB@CT=32)贴 L2, 本线程的行
 * 全部扫过这 CT 列再换下一 tile —— 基矩阵总读量从 每行×96MB 降到 每线程×96MB。
 * 内层 f32 四路累加(编译器 NEON 向量化), 精度对 tanh 特征足够(|dot|≲30, f32 尾差
 * 过 tanh 后被压平; held 网格对 ulp 不敏感)。 */
static void feat_worker(void *vc, int r0, int r1) {
    feat_ctx *c = (feat_ctx *)vc;
    const int CT = 32, din = c->din, K = c->K;
    for (int ct = 0; ct < K; ct += CT) {
        const int ce = ct + CT > K ? K : ct + CT;
        for (int r = r0; r < r1; r++) {
            const float *restrict x = c->X + (size_t)r * din;
            float *zr = c->Z + (size_t)r * K;
            for (int k = ct; k < ce; k++) {
                const float *restrict v = c->Vt + (size_t)k * din;
                float dv0 = 0, dv1 = 0, dv2 = 0, dv3 = 0;
                for (int j = 0; j + 4 <= din; j += 4) {
                    dv0 += x[j] * v[j];     dv1 += x[j + 1] * v[j + 1];
                    dv2 += x[j + 2] * v[j + 2]; dv3 += x[j + 3] * v[j + 3];
                }
                float g = tanhf((dv0 + dv1 + dv2 + dv3) * c->inv_s);
                if (c->At) {
                    const float *restrict a = c->At + (size_t)k * din;
                    float da0 = 0, da1 = 0, da2 = 0, da3 = 0;
                    for (int j = 0; j + 4 <= din; j += 4) {
                        da0 += x[j] * a[j];     da1 += x[j + 1] * a[j + 1];
                        da2 += x[j + 2] * a[j + 2]; da3 += x[j + 3] * a[j + 3];
                    }
                    g *= tanhf((da0 + da1 + da2 + da3) * c->inv_s);
                }
                zr[k] = g;
            }
        }
    }
}

/* ---------- ZtZ / ZtR 累加 worker (按 K 行块并行, f64 累加) ---------- */
typedef struct {
    const float *Z;      /* [n, K] */
    const float *R;      /* [n, D] 已乘 colw */
    double *ZtZ;         /* [K, K] 全局(归并后) */
    double *ZtR;         /* [K, D] */
    double **priv_zz;    /* [nth] 线程私有累加 */
    double **priv_zr;
    int n, K, D, nth;
} gram_ctx;
/* 按 t 分线程 + 线程私有累加: 每线程只流自己 t 块的 Z/R 一遍(总 ~270MB),
 * 原版按 a 分线程 = 每 a 行重流全量(~274GB)。私有区 slot 用 i0 定位。 */
static void gram_worker(void *vc, int t0, int t1) {
    gram_ctx *c = (gram_ctx *)vc;
    int slot = (int)((long)t0 * c->nth / (c->n > 0 ? c->n : 1));
    if (slot >= c->nth) slot = c->nth - 1;
    double *zz = c->priv_zz[slot], *zr = c->priv_zr[slot];
    for (int t = t0; t < t1; t++) {
        const float *zt = c->Z + (size_t)t * c->K;
        const float *rt = c->R + (size_t)t * c->D;
        for (int a = 0; a < c->K; a++) {
            const double za = zt[a];
            if (za == 0.0) continue;
            double *za_row = zz + (size_t)a * c->K;
            for (int b = a; b < c->K; b++) za_row[b] += za * zt[b];
            double *zr_row = zr + (size_t)a * c->D;
            for (int j = 0; j < c->D; j++) zr_row[j] += za * rt[j];
        }
    }
}

/* ---------- held 增量评估 worker: G += Ze[:,k0:k1] @ Uw[k0:k1,:] (ev 行并行) ---------- */
typedef struct {
    const float *Ze;     /* [ne, K] */
    const double *Uw;    /* [K, D] = U/colw */
    float *G;            /* [ne, D] 累加 */
    int K, D, c0, c1;
} acc_ctx;
static void acc_worker(void *vc, int t0, int t1) {
    acc_ctx *c = (acc_ctx *)vc;
    const int CT = 16;   /* 16×4096×8B = 512KB tile 驻 L2 */
    for (int ct = c->c0; ct < c->c1; ct += CT) {
        const int ce = ct + CT > c->c1 ? c->c1 : ct + CT;
        for (int t = t0; t < t1; t++) {
            const float *z = c->Ze + (size_t)t * c->K;
            float *g = c->G + (size_t)t * c->D;
            for (int k = ct; k < ce; k++) {
                const double zk = z[k];
                if (zk == 0.0) continue;
                const double *restrict u = c->Uw + (size_t)k * c->D;
                for (int j = 0; j < c->D; j++) g[j] += (float)(zk * u[j]);
            }
        }
    }
}

typedef struct {
    const float *yq, *yfp, *G;
    double *part;        /* [64] 每块一槽 Σ(yfp - yq(1+g))² */
    int D, nth, nev;
} rec_ctx;
static void rec_worker(void *vc, int t0, int t1) {
    rec_ctx *c = (rec_ctx *)vc;
    double s = 0;
    for (int t = t0; t < t1; t++) {
        const float *yq = c->yq + (size_t)t * c->D;
        const float *yf = c->yfp + (size_t)t * c->D;
        const float *g = c->G + (size_t)t * c->D;
        double d = 0, na = 0, nb = 0;
        for (int j = 0; j < c->D; j++) {
            double yh = (double)yq[j] * (1.0 + (double)g[j]);
            d += yh * yf[j]; na += yh * yh; nb += (double)yf[j] * yf[j];
        }
        s += (na > 0 && nb > 0) ? d / (sqrt(na) * sqrt(nb)) : 0;   /* cos(ŷ, y*) */
    }
    int per = (c->nev + c->nth - 1) / c->nth;
    c->part[(t0 / (per > 0 ? per : 1)) % 64] += s;
}

int main(int argc, char **argv) {
    const char *cap = NULL, *outd = NULL, *layers = NULL;
    int threads = 20, KMAX = 1024, nfit_arg = 0, feat_yq = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outd = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kmax") && i + 1 < argc) KMAX = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nfit") && i + 1 < argc) nfit_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--feat") && i + 1 < argc) feat_yq = !strcmp(argv[++i], "yq");
    }
    if (!cap || !outd || !layers) {
        fprintf(stderr, "用法: amp_solve --cap capnpy目录 --out zrec目录 --layers a-b [--threads N] [--kmax K]\n");
        return 2;
    }
    int l0 = 0, l1 = 0;
    if (sscanf(layers, "%d-%d", &l0, &l1) != 2) { l0 = l1 = atoi(layers); }

    const double LAMS[] = {1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0};
    const int NLAM = 7;
    const int KS[] = {16, 32, 64, 128, 256, 384, 512, 768, 1024};
    const int NKS = 9;

    for (int L = l0; L <= l1; L++) {
        time_t t0 = time(NULL);
        char p[1024]; npy_meta mx, mq, mf;
        /* --feat yq: 特征源换 ŷ(部署字节 routed_out) —— 四支柱"专家输出预测"。
         * 引擎侧落地 = 把 routed_out 绑为 zchain 特征输入(solve_rrr.h feat_yhat 同型)。 */
        snprintf(p, sizeof p, feat_yq ? "%s/obase_v3_L%d.npy" : "%s/ffn_in_L%d.npy", cap, L);
        float *X0 = npy_read_f32(p, &mx);
        snprintf(p, sizeof p, "%s/obase_v3_L%d.npy", cap, L);
        float *yq = npy_read_f32(p, &mq);
        snprintf(p, sizeof p, "%s/routed_L%d.npy", cap, L);
        float *yfp = npy_read_f32(p, &mf);
        if (!X0 || !yq || !yfp || mx.ndim != 2 || mx.shape[1] != DM ||
            mq.shape[0] != mx.shape[0] || mf.shape[0] != mx.shape[0]) {
            fprintf(stderr, "L%d: cap 张量缺/形状错\n", L);
            free(X0); free(yq); free(yfp); continue;
        }
        const int S = (int)mx.shape[0], D = DM, DIN = 3 * DM;
        const int NFIT = nfit_arg > 0 ? nfit_arg : S * 8 / 10;
        const int NEV = S - NFIT, NA = 2 * NFIT;

        /* φ 提升整表 [S, DIN] */
        float *X = malloc((size_t)S * DIN * sizeof(float));
        if (!X) { fprintf(stderr, "L%d OOM X\n", L); return 3; }
        { phi_ctx pc = {X0, X, D}; parallel_for(S, threads, phi_worker, &pc); }
        free(X0);

        /* eps / R / colw (训练段统计) */
        double *msq = calloc(D, sizeof(double));
        for (int t = 0; t < NFIT; t++) {
            const float *y = yq + (size_t)t * D;
            for (int j = 0; j < D; j++) msq[j] += (double)y[j] * y[j];
        }
        float *eps = malloc(D * sizeof(float));
        for (int j = 0; j < D; j++) eps[j] = (float)(sqrt(msq[j] / NFIT) * 1e-2 + 1e-12);
        float *R = malloc((size_t)S * D * sizeof(float));
        for (int t = 0; t < S; t++) {
            const float *y = yq + (size_t)t * D, *yf = yfp + (size_t)t * D;
            float *r = R + (size_t)t * D;
            for (int j = 0; j < D; j++) {
                float dh = yf[j] - y[j];
                r[j] = dh * y[j] / (y[j] * y[j] + eps[j] * eps[j]);
            }
        }
        /* L_classify 感知列权 = y* 的 per-dim 判别方差权(训练段), 归一到均值 1。
         * 旧版 std(R)·rms(yq) 是残差空间权 = 修残差框架, 已废(2026-08-23)。 */
        double *rm = calloc(D, sizeof(double)), *rv = calloc(D, sizeof(double));
        for (int t = 0; t < NFIT; t++) {
            const float *yf = yfp + (size_t)t * D;
            for (int j = 0; j < D; j++) { rm[j] += yf[j]; rv[j] += (double)yf[j] * yf[j]; }
        }
        double *colw = malloc(D * sizeof(double));
        double cwm = 0;
        for (int j = 0; j < D; j++) {
            double mean = rm[j] / NFIT, var = rv[j] / NFIT - mean * mean;
            colw[j] = sqrt((var > 0 ? var : 0) + 1e-12);
            cwm += colw[j];
        }
        cwm /= D;
        for (int j = 0; j < D; j++) colw[j] /= cwm;
        /* L_align 行权: α_t = 1/(‖y*_t‖ + ε‖·‖均值) —— √权乘进增广行两侧, MSE→cos 几何 */
        double *ralpha = malloc((size_t)S * sizeof(double));
        { double nm = 0;
          for (int t = 0; t < S; t++) {
              const float *yf = yfp + (size_t)t * D;
              double ss = 0;
              for (int j = 0; j < D; j++) ss += (double)yf[j] * yf[j];
              ralpha[t] = sqrt(ss); nm += ralpha[t];
          }
          nm /= S;
          for (int t = 0; t < S; t++) ralpha[t] = 1.0 / (ralpha[t] + 1e-3 * nm); }
        free(msq); free(rm); free(rv); free(eps);

        /* dither 增广: Xa = [X_tr; X_tr + 0.04·rms_row·N] ; Ra = [R_tr·colw] ×2 */
        float *Xa = malloc((size_t)NA * DIN * sizeof(float));
        float *Ra = malloc((size_t)NA * D * sizeof(float));
        if (!Xa || !Ra) { fprintf(stderr, "L%d OOM Xa\n", L); return 3; }
        memcpy(Xa, X, (size_t)NFIT * DIN * sizeof(float));
        {   /* 行独立种子 → 可并行(原单线程 8000 万次 Box-Muller ≈ 4s) */
            dith_ctx dc = {X, Xa + (size_t)NFIT * DIN, DIN, (uint64_t)L};
            parallel_for(NFIT, threads, dith_worker, &dc);
        }
        for (int t = 0; t < NFIT; t++) {
            const float *r = R + (size_t)t * D;
            float *ra = Ra + (size_t)t * D, *rb = Ra + (size_t)(NFIT + t) * D;
            const double w = ralpha[t];
            for (int j = 0; j < D; j++) ra[j] = rb[j] = (float)(r[j] * colw[j] * w);
        }

        /* scale = rms(Xa) */
        double ssq = 0;
        for (size_t i = 0; i < (size_t)NA * DIN; i += 97) ssq += (double)Xa[i] * Xa[i];  /* 步进抽样, 均值稳 */
        float scale = (float)sqrt(ssq / (((size_t)NA * DIN + 96) / 97));

        /* 随机基 Vt/At [KMAX, din] 行主(转置存, dot 连续) */
        float *Vt = malloc((size_t)KMAX * DIN * sizeof(float));
        float *At = malloc((size_t)KMAX * DIN * sizeof(float));
        {   /* 每基行独立种子 → 并行(原单线程 5000 万次 Box-Muller) */
            base_ctx bv = {Vt, DIN, 7};  parallel_for(KMAX, threads, base_worker, &bv);
            base_ctx ba = {At, DIN, 11}; parallel_for(KMAX, threads, base_worker, &ba);
        }

        /* 特征: Za [NA, K], Ze [NEV, K] */
        float *Za = malloc((size_t)NA * KMAX * sizeof(float));
        float *Ze = malloc((size_t)NEV * KMAX * sizeof(float));
        if (!Za || !Ze) { fprintf(stderr, "L%d OOM Z\n", L); return 3; }
        time_t tf0 = time(NULL);
        feat_ctx fc = {Xa, Vt, At, Za, DIN, KMAX, 1.0f / scale};
        parallel_for(NA, threads, feat_worker, &fc);
        feat_ctx fe = {X + (size_t)NFIT * DIN, Vt, At, Ze, DIN, KMAX, 1.0f / scale};
        parallel_for(NEV, threads, feat_worker, &fe);
        free(Xa);
        /* 行归一(L_align): 加权 LS 的 α 权 —— Za 行与目标行(Ra, 已乘)各带一次 α ⇒
         * 归一方程 (Zᵀdiag(α²)Z)U = Zᵀdiag(α²)R̃ = cos 几何的行归一化。 */
        for (int t = 0; t < NA; t++) {
            const double w = ralpha[t < NFIT ? t : t - NFIT];
            float *zr = Za + (size_t)t * KMAX;
            for (int k = 0; k < KMAX; k++) zr[k] = (float)(zr[k] * w);
        }
        time_t tf1 = time(NULL);

        /* ZtZ / ZtR */
        double *ZtZ = calloc((size_t)KMAX * KMAX, sizeof(double));
        double *ZtR = calloc((size_t)KMAX * D, sizeof(double));
        {
            int nth = threads > 20 ? 20 : threads;
            double *pz[20], *pr2[20];
            for (int i = 0; i < nth; i++) {
                pz[i] = calloc((size_t)KMAX * KMAX, sizeof(double));
                pr2[i] = calloc((size_t)KMAX * D, sizeof(double));
            }
            gram_ctx gc = {Za, Ra, ZtZ, ZtR, pz, pr2, NA, KMAX, D, nth};
            parallel_for(NA, nth, gram_worker, &gc);
            for (int i = 0; i < nth; i++) {
                for (size_t q = 0; q < (size_t)KMAX * KMAX; q++) ZtZ[q] += pz[i][q];
                for (size_t q = 0; q < (size_t)KMAX * D; q++) ZtR[q] += pr2[i][q];
                free(pz[i]); free(pr2[i]);
            }
        }
        for (int a = 0; a < KMAX; a++)                    /* 对称补全 */
            for (int b = 0; b < a; b++) ZtZ[(size_t)a * KMAX + b] = ZtZ[(size_t)b * KMAX + a];
        free(Za); free(Ra);
        double trZ = 0;
        for (int a = 0; a < KMAX; a++) trZ += ZtZ[(size_t)a * KMAX + a];

        /* 基线对齐度: mean cos(yq, y*) on held(放大器要抬的就是这个数) */
        double cos0 = 0;
        for (int t = NFIT; t < S; t++) {
            const float *y = yq + (size_t)t * D, *yf = yfp + (size_t)t * D;
            double d = 0, na = 0, nb = 0;
            for (int j = 0; j < D; j++) { d += (double)y[j] * yf[j]; na += (double)y[j] * y[j]; nb += (double)yf[j] * yf[j]; }
            cos0 += (na > 0 && nb > 0) ? d / (sqrt(na) * sqrt(nb)) : 0;
        }
        cos0 /= NEV;

        /* λ × k 网格 */
        double best_rec = -2; int best_lam = -1, best_k = 0;   /* 判据=held mean cos */
        double *Ubest = malloc((size_t)KMAX * D * sizeof(double));
        double *A_ch = malloc((size_t)KMAX * KMAX * sizeof(double));
        double *U = malloc((size_t)KMAX * D * sizeof(double));
        double *Uw = malloc((size_t)KMAX * D * sizeof(double));
        float *G = malloc((size_t)NEV * D * sizeof(float));
        double part[64];
        for (int li = 0; li < NLAM; li++) {
            memcpy(A_ch, ZtZ, (size_t)KMAX * KMAX * sizeof(double));
            double ridge = LAMS[li] * trZ / KMAX + 1e-10;
            for (int a = 0; a < KMAX; a++) A_ch[(size_t)a * KMAX + a] += ridge;
            memcpy(U, ZtR, (size_t)KMAX * D * sizeof(double));
            if (chol_solve(A_ch, KMAX, U, D, threads)) { fprintf(stderr, "L%d λ=%g chol 失败\n", L, LAMS[li]); continue; }
            for (int a = 0; a < KMAX; a++)
                for (int j = 0; j < D; j++) Uw[(size_t)a * D + j] = U[(size_t)a * D + j] / colw[j];
            memset(G, 0, (size_t)NEV * D * sizeof(float));
            int kprev = 0;
            for (int ki = 0; ki < NKS && KS[ki] <= KMAX; ki++) {
                acc_ctx ac = {Ze, Uw, G, KMAX, D, kprev, KS[ki]};
                parallel_for(NEV, threads, acc_worker, &ac);
                kprev = KS[ki];
                rec_ctx rc = {yq + (size_t)NFIT * D, yfp + (size_t)NFIT * D, G, part, D, threads, NEV};
                memset(part, 0, sizeof(double) * 64);
                parallel_for(NEV, threads, rec_worker, &rc);
                double csum = 0;
                for (int i = 0; i < 64; i++) csum += part[i];
                double rec = csum / NEV;               /* held mean cos(ŷ, y*) */
                if (rec > best_rec) {
                    best_rec = rec; best_lam = li; best_k = KS[ki];
                    memcpy(Ubest, U, (size_t)KMAX * D * sizeof(double));
                }
            }
        }
        free(A_ch); free(U); free(Uw); free(G); free(ZtZ); free(ZtR); free(Ze);
        fprintf(stderr, "L%d 段耗时: 准备 %lds 特征 %lds gram+网格 %lds\n", L,
                (long)(tf0 - t0), (long)(tf1 - tf0), (long)(time(NULL) - tf1));

        /* zrec 写出 (与 amp_solve.py hdr()/payload 逐字节同构) */
        snprintf(p, sizeof p, "%s/zrec_L%02d.bin", outd, L);
        FILE *f = fopen(p, "wb");
        if (!f) { fprintf(stderr, "L%d: %s 打不开\n", L, p); return 4; }
        uint8_t hdr[116]; memset(hdr, 0, sizeof hdr);
        if (best_rec <= cos0 + 2e-4 || best_lam < 0) {   /* 层闸: held 对齐度无增益 */
            memcpy(hdr, "zl.AMP", 6);
            uint64_t psz = 0; memcpy(hdr + 88, &psz, 8);
            int32_t one = 1; memcpy(hdr + 112, &one, 4);
            fwrite(hdr, 1, 116, f); fclose(f);
            printf("★L%d 放大器(C): held对齐 %.4f→%.4f 无增益 → 层闸 | %lds\n",
                   L, cos0, best_rec, (long)(time(NULL) - t0));
        } else {
            const int k = best_k;
            memcpy(hdr, "zl.AMPD", 7);
            uint64_t psz = 16 + (size_t)2 * DIN * k * 2 + (size_t)D * k * 2;
            memcpy(hdr + 88, &psz, 8);
            int32_t one = 1; memcpy(hdr + 112, &one, 4);
            fwrite(hdr, 1, 116, f);
            uint32_t ku = (uint32_t)k, dinu = (uint32_t)DIN, du = (uint32_t)D;
            fwrite(&ku, 4, 1, f); fwrite(&scale, 4, 1, f); fwrite(&dinu, 4, 1, f); fwrite(&du, 4, 1, f);
            uint16_t *h16 = malloc((size_t)DIN * k * sizeof(uint16_t));
            /* A: [DIN, k] 列 c = At 行 c */
            for (int j = 0; j < DIN; j++)
                for (int c = 0; c < k; c++) h16[(size_t)j * k + c] = f32_to_f16(At[(size_t)c * DIN + j]);
            fwrite(h16, 2, (size_t)DIN * k, f);
            /* U: [D, k] = (Ubest[:k]/colw)ᵀ */
            for (int j = 0; j < D; j++)
                for (int c = 0; c < k; c++)
                    h16[(size_t)j * k + c] = f32_to_f16((float)(Ubest[(size_t)c * D + j] / colw[j]));
            fwrite(h16, 2, (size_t)D * k, f);
            /* V: [DIN, k] 列 c = Vt 行 c */
            for (int j = 0; j < DIN; j++)
                for (int c = 0; c < k; c++) h16[(size_t)j * k + c] = f32_to_f16(Vt[(size_t)c * DIN + j]);
            fwrite(h16, 2, (size_t)DIN * k, f);
            free(h16); fclose(f);
            printf("★L%d 放大器(C): held对齐 cos %.4f→%.4f (Δ%+.4f) @λ=%.0f k_L=%d 体积 %.1fMB | %lds\n",
                   L, cos0, best_rec, best_rec - cos0, LAMS[best_lam], k,
                   (double)psz / (1 << 20), (long)(time(NULL) - t0));
        }
        fflush(stdout);
        free(X); free(yq); free(yfp); free(R); free(colw); free(ralpha); free(Vt); free(At); free(Ubest);
    }
    return 0;
}
