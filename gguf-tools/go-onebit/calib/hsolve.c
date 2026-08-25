/* hsolve.c — 层出口放大器全层解算(C, 2026-08-23 用户令"每一层叠加我的设计, 最后反修,
 * 看端到端输出")。
 *
 * 每层: W = ridge(x → dH_h, λ=3 全秩, 训练段 80%) → rsvd 低秩截断 k →
 *       zrec type10 载荷 A[DM,k]|B[HD,k] f16, 引擎语义 h += B·(Aᵀx)。
 * 依据(一层完整测试, L20): 层出口差的可修成分是线性的(held 还原率 +7.3 点,
 * 动态 z 在线性残差上零增量); 单层中间指标与端到端脱钩(良性漂移), 判决只看端到端 Σmin。
 *
 * 输入: 锚(H 段) + hdump(批量路, BOS 掐头对齐) + capnpy(ffn_in)。
 * 输出: <out>/zrec_L{NN}.bin — "zl.HXP"(type10) 116B 头 + <u32 k><u32 din><u32 hd>
 *       + A(din*k f16) + B(hd*k f16)。
 * 用法: hsolve --anchor F --hdump DIR --cap DIR --out DIR --layers a-b
 *              [--rank K=256] [--lam F=3] [--ntok N=8191] [--threads T]        */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "npy.h"

#define DM 4096

typedef void (*pf_fn)(void *, int, int);
typedef struct { pf_fn fn; void *ctx; int i0, i1; } pf_arg;
static void *pf_tramp(void *a) { pf_arg *p = (pf_arg *)a; p->fn(p->ctx, p->i0, p->i1); return NULL; }
static void parallel_for(int n, int threads, pf_fn fn, void *ctx) {
    if (threads > n) threads = n > 0 ? n : 1;
    if (threads > 64) threads = 64;
    pthread_t th[64]; pf_arg pa[64];
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

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mm = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) { if (e < -10) return (uint16_t)sign;
        mm |= 0x800000u; uint32_t sh = (uint32_t)(14 - e), h = mm >> sh;
        if ((mm >> (sh - 1)) & 1u) h++; return (uint16_t)(sign | h); }
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (mm >> 13));
    if (mm & 0x1000u) h++;
    return h;
}

typedef struct { long long S, HCM, DIM, NL, V, NACT; } ameta;
static float *anchor_H(const char *ap, int L, long long ntok, ameta *m) {
    FILE *f = fopen(ap, "rb");
    if (!f) { fprintf(stderr, "锚 %s 打不开\n", ap); exit(2); }
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8 || hd[0] != 0x32415144u) { fprintf(stderr, "%s 非 DQA2\n", ap); exit(2); }
    m->S = hd[1]; m->HCM = hd[2]; m->DIM = hd[3]; m->NL = hd[4]; m->V = hd[5]; m->NACT = hd[6];
    long long S = m->S, HCM = m->HCM, D = m->DIM, NL = m->NL, NACT = m->NACT;
    if (ntok > S) ntok = S;
    long long off = 40 + NL * S * D * 4 + NL * S * NACT * 4 * 2 + (long long)L * S * HCM * D * 4;
    fseek(f, (long)off, SEEK_SET);
    float *h = malloc((size_t)ntok * HCM * D * sizeof(float));
    if (!h || fread(h, 4, (size_t)ntok * HCM * D, f) != (size_t)ntok * HCM * D) {
        fprintf(stderr, "锚 H L%d 读断\n", L); exit(2);
    }
    fclose(f);
    return h;
}
static float *hdump_L(const char *dir, int L, long long ntok, long long hcm) {
    char p[1024];
    snprintf(p, sizeof p, "%s/h_L%02d.bin", dir, L);
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", p); exit(3); }
    fseek(f, 0, SEEK_END); long long sz = ftell(f);
    long long rows = sz / (hcm * DM * 4);
    if (rows < ntok + 1) { fprintf(stderr, "%s 行不足\n", p); exit(3); }
    fseek(f, (long)(1 * hcm * DM * 4), SEEK_SET);   /* 掐 BOS 行(xdiag 实锤), 保头 */
    float *h = malloc((size_t)ntok * hcm * DM * sizeof(float));
    if (!h || fread(h, 4, (size_t)ntok * hcm * DM, f) != (size_t)ntok * hcm * DM) {
        fprintf(stderr, "%s 读断\n", p); exit(3);
    }
    fclose(f);
    return h;
}

static double vcos(const float *a, const float *b, long long n) {
    double d = 0, na = 0, nb = 0;
    for (long long i = 0; i < n; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return (na > 0 && nb > 0) ? d / (sqrt(na) * sqrt(nb)) : 0;
}

/* Gram worker: f32 4 路累加(f64 逐点太慢), 线程私有 f64 汇总 */
typedef struct {
    const float *X, *Y;
    double **pzz, **pzy;
    int n, dm, hd, nth;
} gctx;
static void gworker(void *vc, int t0, int t1) {
    gctx *c = (gctx *)vc;
    int slot = (int)((long)t0 * c->nth / (c->n > 0 ? c->n : 1));
    if (slot >= c->nth) slot = c->nth - 1;
    double *zz = c->pzz[slot], *zy = c->pzy[slot];
    for (int t = t0; t < t1; t++) {
        const float *xt = c->X + (size_t)t * c->dm;
        const float *yt = c->Y + (size_t)t * c->hd;
        for (int a = 0; a < c->dm; a++) {
            const double xa = xt[a];
            if (xa == 0.0) continue;
            double *zr = zz + (size_t)a * c->dm;
            const float *xb = xt;
            for (int b = a; b < c->dm; b++) zr[b] += xa * xb[b];
            double *yr = zy + (size_t)a * c->hd;
            for (int j = 0; j < c->hd; j++) yr[j] += xa * yt[j];
        }
    }
}
typedef struct { const double *A; double *B; int n, nrhs; } tctx;
static void tworker(void *vc, int c0, int c1) {
    tctx *t = (tctx *)vc;
    const double *A = t->A; double *B = t->B;
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
static int chol_fact(double *A, int n) {
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
    return 0;
}

static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* rsvd 截断: W[dm, hd] ≈ A_out[dm,k]·B_out[hd,k]ᵀ
 * G=randn[hd,k+p]; Y=W G; QR(Y); Bm=QᵀW; eig(Bm Bmᵀ) → 截断。Jacobi on (k+p)². */
typedef struct { const double *W; const float *G; double *Y; int dm, hd, kp; } ymctx;
static void ymworker(void *vc, int r0, int r1) {
    ymctx *c = (ymctx *)vc;
    for (int r = r0; r < r1; r++) {
        const double *wr = c->W + (size_t)r * c->hd;
        double *yr = c->Y + (size_t)r * c->kp;
        for (int q = 0; q < c->kp; q++) yr[q] = 0;
        for (int j = 0; j < c->hd; j++) {
            const double wj = wr[j];
            if (wj == 0.0) continue;
            const float *gj = c->G + (size_t)j * c->kp;
            for (int q = 0; q < c->kp; q++) yr[q] += wj * gj[q];
        }
    }
}
typedef struct { const double *Q, *W; double *Bm; int dm, hd, kp; } bmctx;
static void bmworker(void *vc, int q0, int q1) {
    bmctx *c = (bmctx *)vc;
    for (int q = q0; q < q1; q++) {
        double *br = c->Bm + (size_t)q * c->hd;
        for (int j = 0; j < c->hd; j++) br[j] = 0;
        for (int r = 0; r < c->dm; r++) {
            const double qv = c->Q[(size_t)r * c->kp + q];
            if (qv == 0.0) continue;
            const double *wr = c->W + (size_t)r * c->hd;
            for (int j = 0; j < c->hd; j++) br[j] += qv * wr[j];
        }
    }
}
/* Jacobi 对称特征分解 (n<=320): A 破坏, V 出特征向量(列), d 出特征值 */
static void jacobi_eig(double *A, int n, double *V, double *d) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) V[(size_t)i * n + j] = i == j ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 30; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) off += A[(size_t)i * n + j] * A[(size_t)i * n + j];
        if (off < 1e-18) break;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                double apq = A[(size_t)p * n + q];
                if (fabs(apq) < 1e-15) continue;
                double app = A[(size_t)p * n + p], aqq = A[(size_t)q * n + q];
                double theta = 0.5 * (aqq - app) / apq;
                double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1));
                double cth = 1.0 / sqrt(t * t + 1), sth = t * cth;
                for (int i = 0; i < n; i++) {
                    double aip = A[(size_t)i * n + p], aiq = A[(size_t)i * n + q];
                    A[(size_t)i * n + p] = aip * cth - aiq * sth;
                    A[(size_t)i * n + q] = aip * sth + aiq * cth;
                }
                for (int i = 0; i < n; i++) {
                    double api = A[(size_t)p * n + i], aqi = A[(size_t)q * n + i];
                    A[(size_t)p * n + i] = api * cth - aqi * sth;
                    A[(size_t)q * n + i] = api * sth + aqi * cth;
                }
                for (int i = 0; i < n; i++) {
                    double vip = V[(size_t)i * n + p], viq = V[(size_t)i * n + q];
                    V[(size_t)i * n + p] = vip * cth - viq * sth;
                    V[(size_t)i * n + q] = vip * sth + viq * cth;
                }
            }
    }
    for (int i = 0; i < n; i++) d[i] = A[(size_t)i * n + i];
}

int main(int argc, char **argv) {
    const char *ap = NULL, *hd_dir = NULL, *cap = NULL, *outd = NULL, *layers = NULL;
    int K = 256, threads = 20;
    long long ntok = 8191;
    double lam = 3.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--anchor") && i + 1 < argc) ap = argv[++i];
        else if (!strcmp(argv[i], "--hdump") && i + 1 < argc) hd_dir = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outd = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lam") && i + 1 < argc) lam = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    if (!ap || !hd_dir || !cap || !outd || !layers) {
        fprintf(stderr, "用法: hsolve --anchor F --hdump DIR --cap DIR --out DIR --layers a-b [--rank K] [--lam F] [--ntok N]\n");
        return 2;
    }
    int l0, l1;
    if (sscanf(layers, "%d-%d", &l0, &l1) != 2) l0 = l1 = atoi(layers);

    for (int L = l0; L <= l1; L++) {
        char zp[1024];
        snprintf(zp, sizeof zp, "%s/zrec_L%02d.bin", outd, L);
        FILE *chk = fopen(zp, "rb");
        if (chk) { fclose(chk); printf("L%d 已在, 跳过\n", L); continue; }
        time_t t0 = time(NULL);
        ameta m;
        float *hf = anchor_H(ap, L, ntok, &m);
        long long n = ntok > m.S ? m.S : ntok;
        const int HD = (int)(m.HCM * m.DIM);
        float *hq = hdump_L(hd_dir, L, n, m.HCM);
        char p[1024]; npy_meta mx;
        snprintf(p, sizeof p, "%s/ffn_in_L%d.npy", cap, L);
        float *X = npy_read_f32(p, &mx);
        if (!X || mx.shape[0] < n) { fprintf(stderr, "L%d cap 缺\n", L); return 3; }
        const int NFIT = (int)(n * 8 / 10), NEV = (int)(n - NFIT);

        float *R = malloc((size_t)n * HD * sizeof(float));
        for (size_t q = 0; q < (size_t)n * HD; q++) R[q] = hf[q] - hq[q];

        /* Gram(训练段) → chol → W */
        int nth = threads > 20 ? 20 : threads;
        double *pzz[20], *pzy[20];
        for (int i = 0; i < nth; i++) { pzz[i] = calloc((size_t)DM * DM, sizeof(double)); pzy[i] = calloc((size_t)DM * HD, sizeof(double)); }
        gctx gc = {X, R, pzz, pzy, NFIT, DM, HD, nth};
        parallel_for(NFIT, nth, gworker, &gc);
        double *XX = calloc((size_t)DM * DM, sizeof(double));
        double *W = calloc((size_t)DM * HD, sizeof(double));
        for (int i = 0; i < nth; i++) {
            for (size_t q = 0; q < (size_t)DM * DM; q++) XX[q] += pzz[i][q];
            for (size_t q = 0; q < (size_t)DM * HD; q++) W[q] += pzy[i][q];
            free(pzz[i]); free(pzy[i]);
        }
        for (int a = 0; a < DM; a++)
            for (int b = 0; b < a; b++) XX[(size_t)a * DM + b] = XX[(size_t)b * DM + a];
        double trX = 0;
        for (int a = 0; a < DM; a++) trX += XX[(size_t)a * DM + a];
        for (int a = 0; a < DM; a++) XX[(size_t)a * DM + a] += lam * trX / DM + 1e-8;
        if (chol_fact(XX, DM)) { fprintf(stderr, "L%d chol 失败\n", L); return 4; }
        tctx tc = {XX, W, DM, HD};
        parallel_for(HD, threads, tworker, &tc);
        free(XX);

        /* rsvd 截断 W[dm,hd] → A[dm,K]·B[hd,K]ᵀ */
        const int KP = K + 16;
        float *G = malloc((size_t)HD * KP * sizeof(float));
        uint64_t sd = 0xabcdULL + (uint64_t)L;
        for (size_t i = 0; i < (size_t)HD * KP; i++) {
            /* 均匀±1 足够(rsvd 投影) */
            G[i] = (sm64(&sd) & 1) ? 1.0f : -1.0f;
        }
        double *Y = malloc((size_t)DM * KP * sizeof(double));
        ymctx yc = {W, G, Y, DM, HD, KP};
        parallel_for(DM, threads, ymworker, &yc);
        free(G);
        /* QR(Y) modified Gram-Schmidt (列) */
        for (int q = 0; q < KP; q++) {
            double nr = 0;
            for (int r = 0; r < DM; r++) nr += Y[(size_t)r * KP + q] * Y[(size_t)r * KP + q];
            nr = sqrt(nr) + 1e-30;
            for (int r = 0; r < DM; r++) Y[(size_t)r * KP + q] /= nr;
            for (int q2 = q + 1; q2 < KP; q2++) {
                double d = 0;
                for (int r = 0; r < DM; r++) d += Y[(size_t)r * KP + q] * Y[(size_t)r * KP + q2];
                for (int r = 0; r < DM; r++) Y[(size_t)r * KP + q2] -= d * Y[(size_t)r * KP + q];
            }
        }
        /* Bm = QᵀW [KP, HD] */
        double *Bm = malloc((size_t)KP * HD * sizeof(double));
        bmctx bc = {Y, W, Bm, DM, HD, KP};
        parallel_for(KP, threads, bmworker, &bc);
        /* eig(Bm Bmᵀ) */
        double *BB = malloc((size_t)KP * KP * sizeof(double));
        for (int a = 0; a < KP; a++)
            for (int b = a; b < KP; b++) {
                double s = 0;
                const double *ra = Bm + (size_t)a * HD, *rb = Bm + (size_t)b * HD;
                for (int j = 0; j < HD; j++) s += ra[j] * rb[j];
                BB[(size_t)a * KP + b] = BB[(size_t)b * KP + a] = s;
            }
        double *V = malloc((size_t)KP * KP * sizeof(double));
        double *ev = malloc(KP * sizeof(double));
        jacobi_eig(BB, KP, V, ev);
        int *ord = malloc(KP * sizeof(int));
        for (int i = 0; i < KP; i++) ord[i] = i;
        for (int a = 0; a < K; a++) {
            int mi = a;
            for (int b = a + 1; b < KP; b++) if (ev[ord[b]] > ev[ord[mi]]) mi = b;
            int t = ord[a]; ord[a] = ord[mi]; ord[mi] = t;
        }
        /* A_out[dm,K] = Q·U_B[:,topK];  B_out[hd,K] = Bmᵀ·U_B[:,topK]/√ev (=V_B·S) → 但
         * Δh = Bᵀ... 引擎语义 h += B·(Aᵀx): pred = W ᵀx ≈ B Aᵀ x ⇒ A[dm,K], B[hd,K],
         * A 列 = Q·u_i, B 列 = Bmᵀ u_i (未再缩放: W≈Q Bm = Σ (Q u_i)(Bmᵀ u_i)ᵀ 当 u 正交完备;
         * 截断到 topK 即所需)。 */
        float *Af = malloc((size_t)DM * K * sizeof(float));
        float *Bf = malloc((size_t)HD * K * sizeof(float));
        for (int c = 0; c < K; c++) {
            const int u = ord[c];
            for (int r = 0; r < DM; r++) {
                double s = 0;
                for (int q = 0; q < KP; q++) s += Y[(size_t)r * KP + q] * V[(size_t)q * KP + u];
                Af[(size_t)r * K + c] = (float)s;
            }
            for (int j = 0; j < HD; j++) {
                double s = 0;
                for (int q = 0; q < KP; q++) s += Bm[(size_t)q * HD + j] * V[(size_t)q * KP + u];
                Bf[(size_t)j * K + c] = (float)s;
            }
        }
        free(Y); free(Bm); free(BB); free(V); free(ev); free(ord);

        /* held 自检: 还原率 前→后(全秩 W 与低秩 A·B 各报) */
        double c0 = 0, c1 = 0, c2 = 0;
        {
            float *fx = malloc(HD * sizeof(float));
            double *zv = malloc(K * sizeof(double));
            for (long long t = NFIT; t < n; t += 4) {   /* 抽 1/4 评估足够 */
                const float *xt = X + (size_t)t * DM;
                const float *hqt = hq + (size_t)t * HD, *hft = hf + (size_t)t * HD;
                c0 += vcos(hqt, hft, HD);
                /* 全秩 */
                for (int j = 0; j < HD; j++) fx[j] = hqt[j];
                for (int a = 0; a < DM; a++) {
                    const double xa = xt[a];
                    if (xa == 0.0) continue;
                    const double *wr = W + (size_t)a * HD;
                    for (int j = 0; j < HD; j++) fx[j] += (float)(xa * wr[j]);
                }
                c1 += vcos(fx, hft, HD);
                /* 低秩 */
                for (int c = 0; c < K; c++) {
                    double s = 0;
                    for (int a = 0; a < DM; a++) s += (double)xt[a] * Af[(size_t)a * K + c];
                    zv[c] = s;
                }
                for (int j = 0; j < HD; j++) {
                    double s = 0;
                    const float *br = Bf + (size_t)j * K;
                    for (int c = 0; c < K; c++) s += zv[c] * br[c];
                    fx[j] = hqt[j] + (float)s;
                }
                c2 += vcos(fx, hft, HD);
            }
            free(fx); free(zv);
            long long ne = (NEV + 3) / 4;
            c0 /= ne; c1 /= ne; c2 /= ne;
        }
        printf("★L%d 层出口解算: held还原率 %.4f → 全秩 %.4f → 低秩k%d %.4f | %lds\n",
               L, c0, c1, K, c2, (long)(time(NULL) - t0));
        fflush(stdout);
        free(W);

        /* zrec type10: "zl.HXP" + <k, din, hd> + A|B f16 */
        FILE *f = fopen(zp, "wb");
        uint8_t hdr[116]; memset(hdr, 0, sizeof hdr);
        memcpy(hdr, "zl.HXP", 6);
        uint64_t psz = 12 + (size_t)DM * K * 2 + (size_t)HD * K * 2;
        memcpy(hdr + 88, &psz, 8);
        int32_t one = 1; memcpy(hdr + 112, &one, 4);
        fwrite(hdr, 1, 116, f);
        uint32_t ku = (uint32_t)K, du = DM, hu = (uint32_t)HD;
        fwrite(&ku, 4, 1, f); fwrite(&du, 4, 1, f); fwrite(&hu, 4, 1, f);
        uint16_t *h16 = malloc((size_t)HD * K * sizeof(uint16_t));
        for (size_t q = 0; q < (size_t)DM * K; q++) h16[q] = f32_to_f16(Af[q]);
        fwrite(h16, 2, (size_t)DM * K, f);
        for (size_t q = 0; q < (size_t)HD * K; q++) h16[q] = f32_to_f16(Bf[q]);
        fwrite(h16, 2, (size_t)HD * K, f);
        free(h16); fclose(f);

        free(hf); free(hq); free(X); free(R); free(Af); free(Bf);
    }
    return 0;
}
