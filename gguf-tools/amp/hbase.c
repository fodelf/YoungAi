/* hbase.c — 层出口口径的基线还原率曲线 + 可学性探针(C, 2026-08-23 重新设计第一步)。
 *
 * 观测点 = 层出口 h(hc 4 流拼接, 16384 维): attn/norm/shared/routed/hc 混合的
 * 全部量化效应之和 —— "放大器跟当层量化模型的一切计算一起算"的正确落点。
 *
 * ① 基线曲线(--layers a-b): 逐层报
 *      还原率 cos(h_q, h_fp) 均值/p5  与  rel = ‖h_q−h_fp‖/‖h_fp‖
 *    对标: 单层还原率 ≥0.90(用户裁决)。缺口地图定资源分配。
 * ② 可学性探针(--probe L): 层出口差 dH_h = h_fp − h_q 能否从 x(链态 MoE 输入)预测:
 *      ridge x→dH_h(λ 网格), held 段报 还原率 cos(h_q+pred, h_fp) 前→后。
 *    held 无增益 ⇒ 层出口口径下放大器也无肉, 方向判死; 有增益 ⇒ 值多少肉。
 *
 * 输入: 锚(H 段 [NL][S][HCM][DIM] f32) + hdump(h_L%02d.bin [n][HCM][DIM] f32,
 *       批量路可能多 BOS 行, 掐头对齐) + capnpy(ffn_in, 探针特征)。
 * 用法: hbase --anchor F --hdump DIR [--cap DIR] [--layers a-b] [--probe L]
 *             [--ntok N] [--threads T]                                        */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846  /* Linux -std=c11 严格模式 math.h 不给(Darwin 给) */
#endif
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

/* 锚头 */
typedef struct { long long S, HCM, DIM, NL, V, NACT; } ameta;
static int anchor_head(FILE *f, ameta *m) {
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8 || hd[0] != 0x32415144u) return -1;
    m->S = hd[1]; m->HCM = hd[2]; m->DIM = hd[3]; m->NL = hd[4]; m->V = hd[5]; m->NACT = hd[6];
    return 0;
}

/* 读锚 H 段某层 [S][HCM*DIM] */
static float *anchor_H(const char *ap, int L, long long ntok, ameta *m) {
    FILE *f = fopen(ap, "rb");
    if (!f) { fprintf(stderr, "锚 %s 打不开\n", ap); exit(2); }
    if (anchor_head(f, m)) { fprintf(stderr, "%s 不是 DQA2\n", ap); exit(2); }
    long long S = m->S, HCM = m->HCM, D = m->DIM, NL = m->NL, NACT = m->NACT;
    if (ntok > S) ntok = S;
    long long off = 40 + NL * S * D * 4 + NL * S * NACT * 4 * 2   /* fin + ridx + rw */
                  + (long long)L * S * HCM * D * 4;
    fseek(f, (long)off, SEEK_SET);
    float *h = malloc((size_t)ntok * HCM * D * sizeof(float));
    if (!h || fread(h, 4, (size_t)ntok * HCM * D, f) != (size_t)ntok * HCM * D) {
        fprintf(stderr, "锚 H L%d 读断\n", L); exit(2);
    }
    fclose(f);
    return h;
}

/* 读 hdump 某层(f32 raw), 掐头对齐到 ntok 行(批量路 BOS 行在流首) */
static float *hdump_L(const char *dir, int L, long long ntok, long long hcm) {
    char p[1024];
    snprintf(p, sizeof p, "%s/h_L%02d.bin", dir, L);
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", p); exit(3); }
    fseek(f, 0, SEEK_END); long long sz = ftell(f);
    long long rows = sz / (hcm * DM * 4);
    if (rows < ntok) { fprintf(stderr, "%s 行不足 %lld < %lld\n", p, rows, ntok); exit(3); }
    /* 对齐(xdiag 实锤 2026-08-23): hdump 行 t = 处理第 t 个输入的输出, 而输入流 =
     * [BOS, ids[0..]], 所以 hdump[t+1] ↔ 锚[t](锚无 BOS)。shift−1 对角 cos=0.998,
     * 0/+1 只有 0.29 —— 固定掐头 1 行(BOS), 再保头取 ntok。 */
    long long off_rows = 1;
    if (rows < ntok + 1) { fprintf(stderr, "%s 行不足(%lld < %lld+BOS)\n", p, rows, ntok); exit(3); }
    fseek(f, (long)(off_rows * hcm * DM * 4), SEEK_SET);
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

/* ---- 探针的并行 Gram: XtX[dm,dm], XtY[dm, hd] ---- */
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
            for (int b = a; b < c->dm; b++) zr[b] += xa * xt[b];
            double *yr = zy + (size_t)a * c->hd;
            for (int j = 0; j < c->hd; j++) yr[j] += xa * yt[j];
        }
    }
}

/* chol + 回代(列并行) */
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

/* held 预测累加: pred[t] = W^T x_t (行并行) */
typedef struct {
    const float *X; const double *W;   /* W[dm, hd] */
    float *P;
    int dm, hd;
} pctx;
static void pworker(void *vc, int t0, int t1) {
    pctx *c = (pctx *)vc;
    for (int t = t0; t < t1; t++) {
        const float *xt = c->X + (size_t)t * c->dm;
        float *pt = c->P + (size_t)t * c->hd;
        for (int j = 0; j < c->hd; j++) pt[j] = 0;
        for (int a = 0; a < c->dm; a++) {
            const double xa = xt[a];
            if (xa == 0.0) continue;
            const double *wr = c->W + (size_t)a * c->hd;
            for (int j = 0; j < c->hd; j++) pt[j] += (float)(xa * wr[j]);
        }
    }
}

/* ═══ 完整设计探针(2026-08-23 用户令"一层先把我的设计完整测试还原率") ═══
 * 动态 z(双 tanh 门, z=f(x)) + 四损失全件 + φ 特征, 目标=层出口, 指标=held 还原率。
 *   特征: φ(x)=[x,x²/rms,relu(x)] (12288) → z_c = tanh(φxV_c/s)·tanh(φxA_c/s), K=1024 随机基
 *   修正: Δh = U·z(x) (加性; 层出口有 massive 维, 乘性危险)
 *   L_align   → 行归一 1/‖h_fp‖ 进最小二乘(cos 几何 = 还原率本身)
 *   L_cls 感知 → 列权 std(h_fp per-dim) 归一(判别重要维)
 *   L_smooth  → dither 增广 ×2 (0.04·rms, 固定种子)
 *   L_fixed   → 强收缩 λ 网格 + 固定种子随机基
 *   λ×k held 网格自选, 报 held 还原率 前→后 + 各 k 的体积。 */
static uint64_t sm64_(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double gauss_(uint64_t *s, double *sp, int *hs) {
    if (*hs) { *hs = 0; return *sp; }
    double u1 = ((sm64_(s) >> 11) + 1.0) * (1.0 / 9007199254740993.0);
    double u2 = (sm64_(s) >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1)), a = 2.0 * M_PI * u2;
    *sp = r * sin(a); *hs = 1;
    return r * cos(a);
}
typedef struct { const float *X; float *P; int din; } phi2_ctx;
static void phi2_worker(void *vc, int t0, int t1) {
    phi2_ctx *c = (phi2_ctx *)vc;
    const int d = DM;
    for (int t = t0; t < t1; t++) {
        const float *x = c->X + (size_t)t * d;
        float *o = c->P + (size_t)t * c->din;
        double ss = 0;
        for (int j = 0; j < d; j++) ss += (double)x[j] * x[j];
        float nr = (float)sqrt(ss / d) + 1e-6f;
        for (int j = 0; j < d; j++) { o[j] = x[j]; o[d + j] = x[j] * x[j] / nr; o[2 * d + j] = x[j] > 0 ? x[j] : 0; }
    }
}
typedef struct { const float *X; float *Xd; int din; uint64_t salt; } dith2_ctx;
static void dith2_worker(void *vc, int t0, int t1) {
    dith2_ctx *c = (dith2_ctx *)vc;
    for (int t = t0; t < t1; t++) {
        uint64_t sd = 0x5eedULL + c->salt * 1000003ULL + (uint64_t)t; double sp = 0; int hs = 0;
        const float *x = c->X + (size_t)t * c->din;
        float *xd = c->Xd + (size_t)t * c->din;
        double ss = 0;
        for (int j = 0; j < c->din; j++) ss += (double)x[j] * x[j];
        float rms = (float)sqrt(ss / c->din);
        for (int j = 0; j < c->din; j++) xd[j] = x[j] + (float)gauss_(&sd, &sp, &hs) * 0.04f * rms;
    }
}
typedef struct { float *B; int din; uint64_t salt; } base2_ctx;
static void base2_worker(void *vc, int k0, int k1) {
    base2_ctx *c = (base2_ctx *)vc;
    const float inv = 1.0f / sqrtf((float)c->din);
    for (int k = k0; k < k1; k++) {
        uint64_t sd = c->salt * 0x9E3779B9ULL + (uint64_t)k; double sp = 0; int hs = 0;
        float *r = c->B + (size_t)k * c->din;
        for (int j = 0; j < c->din; j++) r[j] = (float)gauss_(&sd, &sp, &hs) * inv;
    }
}
typedef struct { const float *X, *Vt, *At; float *Z; int din, K; float inv_s; } feat2_ctx;
static void feat2_worker(void *vc, int r0, int r1) {
    feat2_ctx *c = (feat2_ctx *)vc;
    const int CT = 32, din = c->din, K = c->K;
    for (int ct = 0; ct < K; ct += CT) {
        const int ce = ct + CT > K ? K : ct + CT;
        for (int r = r0; r < r1; r++) {
            const float *restrict x = c->X + (size_t)r * din;
            float *zr = c->Z + (size_t)r * K;
            for (int k = ct; k < ce; k++) {
                const float *restrict v = c->Vt + (size_t)k * din;
                const float *restrict a = c->At + (size_t)k * din;
                float d0 = 0, d1 = 0, a0 = 0, a1 = 0;
                for (int j = 0; j + 2 <= din; j += 2) {
                    d0 += x[j] * v[j]; d1 += x[j + 1] * v[j + 1];
                    a0 += x[j] * a[j]; a1 += x[j + 1] * a[j + 1];
                }
                zr[k] = tanhf((d0 + d1) * c->inv_s) * tanhf((a0 + a1) * c->inv_s);
            }
        }
    }
}
typedef struct { const float *Z, *R; double **pzz, **pzy; int n, K, HD, nth; } g2ctx;
static void g2worker(void *vc, int t0, int t1) {
    g2ctx *c = (g2ctx *)vc;
    int slot = (int)((long)t0 * c->nth / (c->n > 0 ? c->n : 1));
    if (slot >= c->nth) slot = c->nth - 1;
    double *zz = c->pzz[slot], *zy = c->pzy[slot];
    for (int t = t0; t < t1; t++) {
        const float *zt = c->Z + (size_t)t * c->K;
        const float *rt = c->R + (size_t)t * c->HD;
        for (int a = 0; a < c->K; a++) {
            const double za = zt[a];
            if (za == 0.0) continue;
            double *zr = zz + (size_t)a * c->K;
            for (int b = a; b < c->K; b++) zr[b] += za * zt[b];
            double *yr = zy + (size_t)a * c->HD;
            for (int j = 0; j < c->HD; j++) yr[j] += za * rt[j];
        }
    }
}
typedef struct { const float *Ze; const double *U; float *G; int K, HD, c0, c1; } a2ctx;
static void a2worker(void *vc, int t0, int t1) {
    a2ctx *c = (a2ctx *)vc;
    const int CT = 8;
    for (int ct = c->c0; ct < c->c1; ct += CT) {
        const int ce = ct + CT > c->c1 ? c->c1 : ct + CT;
        for (int t = t0; t < t1; t++) {
            const float *z = c->Ze + (size_t)t * c->K;
            float *g = c->G + (size_t)t * c->HD;
            for (int k = ct; k < ce; k++) {
                const double zk = z[k];
                if (zk == 0.0) continue;
                const double *u = c->U + (size_t)k * c->HD;
                for (int j = 0; j < c->HD; j++) g[j] += (float)(zk * u[j]);
            }
        }
    }
}
static int full_design_probe(const char *ap, const char *hd_dir, const char *cap,
                             int L, long long ntok, int threads) {
    if (!cap) { fprintf(stderr, "--probe2 需要 --cap\n"); return 2; }
    time_t t0 = time(NULL);
    ameta m;
    float *hf = anchor_H(ap, L, ntok, &m);
    long long n = ntok > m.S ? m.S : ntok;
    const int HD = (int)(m.HCM * m.DIM);
    float *hq = hdump_L(hd_dir, L, n, m.HCM);
    char p[1024]; npy_meta mx;
    snprintf(p, sizeof p, "%s/ffn_in_L%d.npy", cap, L);
    float *X0 = npy_read_f32(p, &mx);
    if (!X0 || mx.shape[0] < n) { fprintf(stderr, "cap ffn_in 缺\n"); return 3; }
    const int NFIT = (int)(n * 8 / 10), NEV = (int)(n - NFIT), NA = 2 * NFIT;
    const int DIN = 3 * DM, K = 1024;

    /* φ 提升 + dither 增广(L_smooth/L_fixed) */
    float *PX = malloc((size_t)n * DIN * sizeof(float));
    phi2_ctx pc = {X0, PX, DIN};
    parallel_for((int)n, threads, phi2_worker, &pc);
    float *Xa = malloc((size_t)NA * DIN * sizeof(float));
    memcpy(Xa, PX, (size_t)NFIT * DIN * sizeof(float));
    dith2_ctx dc = {PX, Xa + (size_t)NFIT * DIN, DIN, (uint64_t)L};
    parallel_for(NFIT, threads, dith2_worker, &dc);

    /* 靶 dH_h */
    float *R = malloc((size_t)n * HD * sizeof(float));
    for (long long t = 0; t < n; t++) {
        const float *a = hf + (size_t)t * HD, *b = hq + (size_t)t * HD;
        for (int j = 0; j < HD; j++) R[(size_t)t * HD + j] = a[j] - b[j];
    }
    /* ★线性主干先行(组合设计)★: W = ridge(x→dH_h, λ=3 全秩, 训练段), 全部行扣除
     * W 预测 → 动态 z 解线性残差。层出口差的主可修成分实测是线性的(0.841 vs 0.821)。 */
    float *LIN = calloc((size_t)n * HD, sizeof(float));
    {
        int nth0 = threads > 20 ? 20 : threads;
        double *pz0[20], *py0[20];
        for (int i = 0; i < nth0; i++) { pz0[i] = calloc((size_t)DM * DM, sizeof(double)); py0[i] = calloc((size_t)DM * HD, sizeof(double)); }
        gctx g0 = {X0, R, pz0, py0, NFIT, DM, HD, nth0};
        parallel_for(NFIT, nth0, gworker, &g0);
        double *XX = calloc((size_t)DM * DM, sizeof(double));
        double *XY = calloc((size_t)DM * HD, sizeof(double));
        for (int i = 0; i < nth0; i++) {
            for (size_t q = 0; q < (size_t)DM * DM; q++) XX[q] += pz0[i][q];
            for (size_t q = 0; q < (size_t)DM * HD; q++) XY[q] += py0[i][q];
            free(pz0[i]); free(py0[i]);
        }
        for (int a = 0; a < DM; a++)
            for (int b = 0; b < a; b++) XX[(size_t)a * DM + b] = XX[(size_t)b * DM + a];
        double trX = 0;
        for (int a = 0; a < DM; a++) trX += XX[(size_t)a * DM + a];
        for (int a = 0; a < DM; a++) XX[(size_t)a * DM + a] += 3.0 * trX / DM + 1e-8;
        if (chol_fact(XX, DM)) { fprintf(stderr, "线性 chol 失败\n"); return 4; }
        tctx tl = {XX, XY, DM, HD};
        parallel_for(HD, threads, tworker, &tl);
        pctx pl = {X0, XY, LIN, DM, HD};
        parallel_for((int)n, threads, pworker, &pl);
        for (size_t q = 0; q < (size_t)n * HD; q++) R[q] -= LIN[q];
        free(XX); free(XY);
        printf("L%d 线性主干已扣除(λ=3 全秩), z 解残差 | %lds\n", L, (long)(time(NULL) - t0));
        fflush(stdout);
    }
    double *ralpha = malloc(n * sizeof(double));
    { double nm = 0;
      for (long long t = 0; t < n; t++) {
          const float *a = hf + (size_t)t * HD;
          double ss = 0;
          for (int j = 0; j < HD; j++) ss += (double)a[j] * a[j];
          ralpha[t] = sqrt(ss); nm += ralpha[t];
      }
      nm /= n;
      for (long long t = 0; t < n; t++) ralpha[t] = 1.0 / (ralpha[t] + 1e-3 * nm); }
    double *colw = malloc(HD * sizeof(double));
    { double *mv = calloc(HD, sizeof(double)), *vv = calloc(HD, sizeof(double));
      for (long long t = 0; t < NFIT; t++) {
          const float *a = hf + (size_t)t * HD;
          for (int j = 0; j < HD; j++) { mv[j] += a[j]; vv[j] += (double)a[j] * a[j]; }
      }
      double cm = 0;
      for (int j = 0; j < HD; j++) {
          double mean = mv[j] / NFIT, var = vv[j] / NFIT - mean * mean;
          colw[j] = sqrt((var > 0 ? var : 0) + 1e-12); cm += colw[j];
      }
      cm /= HD;
      for (int j = 0; j < HD; j++) colw[j] /= cm;
      free(mv); free(vv); }

    /* 随机基(固定种子) + 特征 */
    float *Vt = malloc((size_t)K * DIN * sizeof(float));
    float *At = malloc((size_t)K * DIN * sizeof(float));
    base2_ctx bv = {Vt, DIN, 7};  parallel_for(K, threads, base2_worker, &bv);
    base2_ctx ba = {At, DIN, 11}; parallel_for(K, threads, base2_worker, &ba);
    double ssq = 0;
    for (size_t i = 0; i < (size_t)NA * DIN; i += 97) ssq += (double)Xa[i] * Xa[i];
    float scale = (float)sqrt(ssq / (((size_t)NA * DIN + 96) / 97));
    float *Za = malloc((size_t)NA * K * sizeof(float));
    float *Ze = malloc((size_t)NEV * K * sizeof(float));
    feat2_ctx fc = {Xa, Vt, At, Za, DIN, K, 1.0f / scale};
    parallel_for(NA, threads, feat2_worker, &fc);
    feat2_ctx fe = {PX + (size_t)NFIT * DIN, Vt, At, Ze, DIN, K, 1.0f / scale};
    parallel_for(NEV, threads, feat2_worker, &fe);
    free(Xa); free(PX);

    /* 增广目标行(colw·α; dither 行同靶=L_smooth) + Za 行 ×α */
    float *Ra = malloc((size_t)NA * HD * sizeof(float));
    for (int t = 0; t < NFIT; t++) {
        const float *r = R + (size_t)t * HD;
        float *r1 = Ra + (size_t)t * HD, *r2 = Ra + (size_t)(NFIT + t) * HD;
        const double w = ralpha[t];
        for (int j = 0; j < HD; j++) r1[j] = r2[j] = (float)(r[j] * colw[j] * w);
    }
    for (int t = 0; t < NA; t++) {
        const double w = ralpha[t < NFIT ? t : t - NFIT];
        float *zr = Za + (size_t)t * K;
        for (int k = 0; k < K; k++) zr[k] = (float)(zr[k] * w);
    }

    /* Gram */
    int nth = threads > 20 ? 20 : threads;
    double *pzz[20], *pzy[20];
    for (int i = 0; i < nth; i++) { pzz[i] = calloc((size_t)K * K, sizeof(double)); pzy[i] = calloc((size_t)K * HD, sizeof(double)); }
    g2ctx gc = {Za, Ra, pzz, pzy, NA, K, HD, nth};
    parallel_for(NA, nth, g2worker, &gc);
    double *ZtZ = calloc((size_t)K * K, sizeof(double));
    double *ZtR = calloc((size_t)K * HD, sizeof(double));
    for (int i = 0; i < nth; i++) {
        for (size_t q = 0; q < (size_t)K * K; q++) ZtZ[q] += pzz[i][q];
        for (size_t q = 0; q < (size_t)K * HD; q++) ZtR[q] += pzy[i][q];
        free(pzz[i]); free(pzy[i]);
    }
    for (int a = 0; a < K; a++)
        for (int b = 0; b < a; b++) ZtZ[(size_t)a * K + b] = ZtZ[(size_t)b * K + a];
    double trZ = 0;
    for (int a = 0; a < K; a++) trZ += ZtZ[(size_t)a * K + a];
    free(Za); free(Ra);

    double c0v = 0, c0lin = 0;
    { float *tmp = malloc(HD * sizeof(float));
      for (long long t = NFIT; t < n; t++) {
          c0v += vcos(hq + (size_t)t * HD, hf + (size_t)t * HD, HD);
          const float *hqt = hq + (size_t)t * HD, *lt = LIN + (size_t)t * HD;
          for (int j = 0; j < HD; j++) tmp[j] = hqt[j] + lt[j];
          c0lin += vcos(tmp, hf + (size_t)t * HD, HD);
      }
      free(tmp); }
    c0v /= NEV; c0lin /= NEV;
    printf("L%d 组合探针: 裸基线 %.4f → 线性主干后 %.4f | 准备 %lds\n", L, c0v, c0lin, (long)(time(NULL) - t0));
    fflush(stdout);

    const double LAMS[] = {30.0, 10.0, 3.0, 1.0};
    const int KS[] = {128, 256, 512, 1024};
    double *A = malloc((size_t)K * K * sizeof(double));
    double *U = malloc((size_t)K * HD * sizeof(double));
    double *Uw = malloc((size_t)K * HD * sizeof(double));
    float *G = malloc((size_t)NEV * HD * sizeof(float));
    float *hfix = malloc(HD * sizeof(float));
    for (int li = 0; li < 4; li++) {
        memcpy(A, ZtZ, (size_t)K * K * sizeof(double));
        double ridge = LAMS[li] * trZ / K + 1e-10;
        for (int a = 0; a < K; a++) A[(size_t)a * K + a] += ridge;
        if (chol_fact(A, K)) { fprintf(stderr, "chol 失败 λ=%g\n", LAMS[li]); continue; }
        memcpy(U, ZtR, (size_t)K * HD * sizeof(double));
        tctx tc = {A, U, K, HD};
        parallel_for(HD, threads, tworker, &tc);
        for (int a = 0; a < K; a++)
            for (int j = 0; j < HD; j++) Uw[(size_t)a * HD + j] = U[(size_t)a * HD + j] / colw[j];
        memset(G, 0, (size_t)NEV * HD * sizeof(float));
        int kprev = 0;
        for (int ki = 0; ki < 4; ki++) {
            a2ctx ac = {Ze, Uw, G, K, HD, kprev, KS[ki]};
            parallel_for(NEV, threads, a2worker, &ac);
            kprev = KS[ki];
            double c1 = 0;
            for (long long t = 0; t < NEV; t++) {
                const float *hqt = hq + (size_t)(NFIT + t) * HD;
                const float *hft = hf + (size_t)(NFIT + t) * HD;
                const float *g = G + (size_t)t * HD;
                const float *lt = LIN + (size_t)(NFIT + t) * HD;
                for (int j = 0; j < HD; j++) hfix[j] = hqt[j] + lt[j] + g[j];
                c1 += vcos(hfix, hft, HD);
            }
            c1 /= NEV;
            double mb = ((double)DIN * KS[ki] * 2 + (double)HD * KS[ki]) * 2 / (1 << 20);
            printf("★L%d 完整设计: λ=%g k=%d held还原率 %.4f→%.4f (Δ%+.4f) 体积 %.0fMB | %lds\n",
                   L, LAMS[li], KS[ki], c0v, c1, c1 - c0v, mb, (long)(time(NULL) - t0));
            fflush(stdout);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *ap = NULL, *hd_dir = NULL, *cap = NULL, *layers = "0-42";
    int probe = -1, xdiag = -1, probe2 = -1, threads = 20;
    long long ntok = 8192;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--anchor") && i + 1 < argc) ap = argv[++i];
        else if (!strcmp(argv[i], "--hdump") && i + 1 < argc) hd_dir = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--probe") && i + 1 < argc) probe = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--xdiag") && i + 1 < argc) xdiag = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe2") && i + 1 < argc) probe2 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    if (!ap || !hd_dir) { fprintf(stderr, "用法: hbase --anchor F --hdump DIR [--cap DIR] [--layers a-b] [--probe L] [--ntok N]\n"); return 2; }

    int l0, l1;
    if (sscanf(layers, "%d-%d", &l0, &l1) != 2) l0 = l1 = atoi(layers);

    if (probe == -99) { /* --xdiag: 流配对/层偏/token 位移交叉诊断 */ }
    if (xdiag >= 0) {
        /* 交叉诊断: 锚H(层 L..L+1) × hdump h(层 L) 的 [流a×流b] cos 矩阵 + token±1 */
        int L = xdiag;
        ameta m;
        for (int dl = 0; dl <= 1; dl++) {
            float *hf = anchor_H(ap, L + dl, ntok, &m);
            long long n = ntok > m.S ? m.S : ntok;
            float *hq = hdump_L(hd_dir, L, n, m.HCM);
            printf("── 锚H_L%d × hdump_L%d, 流 cos 矩阵(行=锚流, 列=引擎流, token 对角) ──\n", L + dl, L);
            for (int a = 0; a < m.HCM; a++) {
                for (int b = 0; b < m.HCM; b++) {
                    double cs = 0;
                    for (long long t = 0; t < n; t += 8)
                        cs += vcos(hf + ((size_t)t * m.HCM + a) * m.DIM,
                                   hq + ((size_t)t * m.HCM + b) * m.DIM, m.DIM);
                    printf(" %7.4f", cs / ((n + 7) / 8));
                }
                printf("\n");
            }
            /* token ±1 位移(流对角均值) */
            for (int sh = -1; sh <= 1; sh++) {
                double cs = 0; long long cnt = 0;
                for (long long t = 8; t + 8 < n; t += 8) {
                    for (int a = 0; a < m.HCM; a++)
                        cs += vcos(hf + ((size_t)(t + sh) * m.HCM + a) * m.DIM,
                                   hq + ((size_t)t * m.HCM + a) * m.DIM, m.DIM);
                    cnt += m.HCM;
                }
                printf("  token shift %+d: %.4f\n", sh, cs / cnt);
            }
            free(hf); free(hq);
        }
        return 0;
    }
    if (probe2 >= 0) return full_design_probe(ap, hd_dir, cap, probe2, ntok, threads);
    if (probe < 0) {
        printf("%-4s %10s %10s %10s %10s\n", "L", "还原率cos", "p5", "rel", "‖h_fp‖");
        for (int L = l0; L <= l1; L++) {
            ameta m;
            float *hf = anchor_H(ap, L, ntok, &m);
            long long n = ntok > m.S ? m.S : ntok, hd = m.HCM * m.DIM;
            float *hq = hdump_L(hd_dir, L, n, m.HCM);
            double cs = 0, e2 = 0, f2 = 0;
            double *cbuf = malloc(n * sizeof(double));
            for (long long t = 0; t < n; t++) {
                const float *a = hf + (size_t)t * hd, *b = hq + (size_t)t * hd;
                cbuf[t] = vcos(b, a, hd);
                cs += cbuf[t];
                for (long long j = 0; j < hd; j++) {
                    double e = (double)b[j] - a[j];
                    e2 += e * e; f2 += (double)a[j] * a[j];
                }
            }
            /* p5 */
            for (long long a1 = 0; a1 < n; a1++) for (long long b1 = a1 + 1; b1 < n; b1++)
                if (cbuf[b1] < cbuf[a1]) { double tt = cbuf[a1]; cbuf[a1] = cbuf[b1]; cbuf[b1] = tt; }
            printf("%-4d %10.4f %10.4f %10.4f %10.1f\n",
                   L, cs / n, cbuf[n / 20], sqrt(e2 / f2), sqrt(f2 / n));
            fflush(stdout);
            free(hf); free(hq); free(cbuf);
        }
        return 0;
    }

    /* ---- 可学性探针 ---- */
    if (!cap) { fprintf(stderr, "--probe 需要 --cap\n"); return 2; }
    int L = probe;
    time_t t0 = time(NULL);
    ameta m;
    float *hf = anchor_H(ap, L, ntok, &m);
    long long n = ntok > m.S ? m.S : ntok, hd = m.HCM * m.DIM;
    float *hq = hdump_L(hd_dir, L, n, m.HCM);
    char p[1024]; npy_meta mx;
    snprintf(p, sizeof p, "%s/ffn_in_L%d.npy", cap, L);
    float *X = npy_read_f32(p, &mx);
    if (!X || mx.shape[0] < n) { fprintf(stderr, "cap ffn_in 缺/行不足\n"); return 3; }
    const int NFIT = (int)(n * 8 / 10), NEV = (int)(n - NFIT);

    /* dH_h = h_fp − h_q */
    float *dH = malloc((size_t)n * hd * sizeof(float));
    for (size_t q = 0; q < (size_t)n * hd; q++) dH[q] = hf[q] - hq[q];

    /* Gram(训练段) */
    int nth = threads > 20 ? 20 : threads;
    double *pzz[20], *pzy[20];
    for (int i = 0; i < nth; i++) {
        pzz[i] = calloc((size_t)DM * DM, sizeof(double));
        pzy[i] = calloc((size_t)DM * hd, sizeof(double));
        if (!pzz[i] || !pzy[i]) { fprintf(stderr, "OOM gram\n"); return 3; }
    }
    gctx gc = {X, dH, pzz, pzy, NFIT, DM, (int)hd, nth};
    parallel_for(NFIT, nth, gworker, &gc);
    double *XtX = calloc((size_t)DM * DM, sizeof(double));
    double *XtY = calloc((size_t)DM * hd, sizeof(double));
    for (int i = 0; i < nth; i++) {
        for (size_t q = 0; q < (size_t)DM * DM; q++) XtX[q] += pzz[i][q];
        for (size_t q = 0; q < (size_t)DM * hd; q++) XtY[q] += pzy[i][q];
        free(pzz[i]); free(pzy[i]);
    }
    for (int a = 0; a < DM; a++)
        for (int b = 0; b < a; b++) XtX[(size_t)a * DM + b] = XtX[(size_t)b * DM + a];
    double trX = 0;
    for (int a = 0; a < DM; a++) trX += XtX[(size_t)a * DM + a];

    /* 基线 held 还原率 */
    double c0 = 0;
    for (long long t = NFIT; t < n; t++)
        c0 += vcos(hq + (size_t)t * hd, hf + (size_t)t * hd, hd);
    c0 /= NEV;

    const double LAMS[] = {10.0, 3.0, 1.0};
    float *P = malloc((size_t)NEV * hd * sizeof(float));
    float *hfix = malloc(hd * sizeof(float));
    double *A = malloc((size_t)DM * DM * sizeof(double));
    double *W = malloc((size_t)DM * hd * sizeof(double));
    for (int li = 0; li < 3; li++) {
        memcpy(A, XtX, (size_t)DM * DM * sizeof(double));
        double ridge = LAMS[li] * trX / DM + 1e-8;
        for (int a = 0; a < DM; a++) A[(size_t)a * DM + a] += ridge;
        if (chol_fact(A, DM)) { fprintf(stderr, "chol 失败 λ=%g\n", LAMS[li]); continue; }
        memcpy(W, XtY, (size_t)DM * hd * sizeof(double));
        tctx tc = {A, W, DM, (int)hd};
        parallel_for((int)hd, threads, tworker, &tc);
        pctx pc = {X + (size_t)NFIT * DM, W, P, DM, (int)hd};
        parallel_for(NEV, threads, pworker, &pc);
        double c1 = 0, tr_c = 0;
        for (long long t = 0; t < NEV; t++) {
            const float *hqt = hq + (size_t)(NFIT + t) * hd;
            const float *hft = hf + (size_t)(NFIT + t) * hd;
            const float *pt = P + (size_t)t * hd;
            for (long long j = 0; j < hd; j++) hfix[j] = hqt[j] + pt[j];
            c1 += vcos(hfix, hft, hd);
        }
        c1 /= NEV;
        /* 训练段还原率(过拟合参照, 抽 512 行) */
        pctx pc2 = {X, W, P, DM, (int)hd};
        parallel_for(512, threads, pworker, &pc2);
        for (int t = 0; t < 512; t++) {
            const float *hqt = hq + (size_t)t * hd;
            const float *hft = hf + (size_t)t * hd;
            const float *pt = P + (size_t)t * hd;
            for (long long j = 0; j < hd; j++) hfix[j] = hqt[j] + pt[j];
            tr_c += vcos(hfix, hft, hd);
        }
        tr_c /= 512;
        printf("★L%d 层出口探针 λ=%g: held还原率 %.4f→%.4f (Δ%+.4f)  训练段→%.4f | %lds\n",
               L, LAMS[li], c0, c1, c1 - c0, tr_c, (long)(time(NULL) - t0));
        fflush(stdout);
    }
    return 0;
}
