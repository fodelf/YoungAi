/* ge_solve.c — GE 每专家门控解算(C, 2026-08-23)。zlayer.py 实证组合"z+GE"的 GE 半:
 *   g_e(标量, 256 个) 乘进被选专家的 gate 权重, 修 per-expert 系统性偏差(VQ 码本收缩
 *   偏置的正对药)。参数极少 ⇒ 小样本不过拟合。
 *
 * 数学(zlayer.py _GE_ON 段逐式):
 *   靶 R_t = y*_t − yq_t  (y* = FP 锚教师 routed, yq = 引擎部署字节 routed 聚合)
 *   共燃 Gram: G[a,b] = Σ_t 1[a,b∈S_t]·(w_a Y_a)·(w_b Y_b),  rhs[a] = Σ_t (w_a Y_a)·R_t
 *   ridge: G += gelam·tr(G)/256·I  (gelam=1e-3 产线; λ↑把增益往 1 收)
 *   ge = 1 + G⁻¹ rhs
 *   Y_e = 量化专家个体输出(VQ blob dequant + swiglu(clip±10) 前向 @ 引擎 x/引擎路由)
 *
 * 引擎落地: zchain type5 "bf.GE" — per-expert gain 乘进 router gate weight(512B f16)。
 * 输入: capnpy 目录(ffn_in/route/route_w/obase_v3/routed) + dql_vq_L%02d.bin 目录。
 * 输出: <out>/zrec_ge_L{NN}.bin (bf.GE 116B 头 + 256×f16)。
 * 用法: ge_solve --cap DIR --dql DIR --layers a-b --out DIR [--nfit N] [--threads T]
 *                [--gelam F] [--swlim F]                                          */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include "npy.h"
#include "vq_fmt.h"   /* repo 根(与引擎共用同一份解码) */

#define DM   4096
#define DFF  2048
#define TOPK 6
#define NEXP 256

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) { if (e < -10) return (uint16_t)sign;
        m |= 0x800000u; uint32_t sh = (uint32_t)(14 - e), h = m >> sh;
        if ((m >> (sh - 1)) & 1u) h++; return (uint16_t)(sign | h); }
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if (m & 0x1000u) h++;
    return h;
}

/* ---- 专家桶前向 worker: 量化 VQ 权重, pair_buf[t*TOPK+slot] = w·Y ---- */
typedef struct {
    const uint8_t *blob;
    const float *x;          /* [n, DM] 引擎链态 */
    const float *rw;         /* [n, TOPK] */
    float *pair;             /* [n*TOPK, DM] 输出 */
    const int *erows, *eslots;
    int cnt, e;
    float swlim;
} fw_task;
static void *fw_worker(void *arg) {
    fw_task *t = (fw_task *)arg;
    uint64_t o1 = ds4vq_slot(t->blob, t->e, 0);
    uint64_t o3 = ds4vq_slot(t->blob, t->e, 1);
    uint64_t o2 = ds4vq_slot(t->blob, t->e, 2);
    if (!o1 || !o3 || !o2) return (void *)1;
    float *w1 = malloc((size_t)DFF * DM * sizeof(float));
    float *w3 = malloc((size_t)DFF * DM * sizeof(float));
    float *w2 = malloc((size_t)DM * DFF * sizeof(float));
    if (ds4vq_dequant_f32(t->blob + o1, w1, DFF, DM) ||
        ds4vq_dequant_f32(t->blob + o3, w3, DFF, DM) ||
        ds4vq_dequant_f32(t->blob + o2, w2, DM, DFF)) { free(w1); free(w3); free(w2); return (void *)2; }
    float *h = malloc(DFF * sizeof(float));
    const float lim = t->swlim;
    for (int i = 0; i < t->cnt; i++) {
        const int r = t->erows[i], sl = t->eslots[i];
        const float *xr = t->x + (size_t)r * DM;
        for (int j = 0; j < DFF; j++) {
            double g = 0, u = 0;
            const float *g1 = w1 + (size_t)j * DM, *u3 = w3 + (size_t)j * DM;
            for (int d = 0; d < DM; d++) { g += (double)g1[d] * xr[d]; u += (double)u3[d] * xr[d]; }
            float gf = (float)g, uf = (float)u;
            if (lim > 0) { if (gf > lim) gf = lim; if (gf < -lim) gf = -lim;
                           if (uf > lim) uf = lim; if (uf < -lim) uf = -lim; }
            h[j] = (gf / (1.0f + expf(-fminf(fmaxf(gf, -60.f), 60.f)))) * uf;
        }
        float *out = t->pair + ((size_t)r * TOPK + sl) * DM;
        const float wgt = t->rw[(size_t)r * TOPK + sl];
        for (int d = 0; d < DM; d++) {
            double s = 0;
            const float *w2r = w2 + (size_t)d * DFF;
            for (int j = 0; j < DFF; j++) s += (double)w2r[j] * h[j];
            out[d] = (float)(s * wgt);
        }
    }
    free(w1); free(w3); free(w2); free(h);
    return NULL;
}

static int chol_solve_small(double *A, int n, double *b) {
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
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= A[(size_t)i * n + k] * b[k];
        b[i] = s / A[(size_t)i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= A[(size_t)k * n + i] * b[k];
        b[i] = s / A[(size_t)i * n + i];
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *cap = NULL, *dql = NULL, *outd = NULL, *layers = NULL;
    int threads = 20, nfit_arg = 0;
    double gelam = 1e-3;
    float swlim = 10.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = argv[++i];
        else if (!strcmp(argv[i], "--dql") && i + 1 < argc) dql = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outd = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nfit") && i + 1 < argc) nfit_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gelam") && i + 1 < argc) gelam = atof(argv[++i]);
        else if (!strcmp(argv[i], "--swlim") && i + 1 < argc) swlim = (float)atof(argv[++i]);
    }
    if (!cap || !dql || !outd || !layers) {
        fprintf(stderr, "用法: ge_solve --cap DIR --dql DIR --layers a-b --out DIR [--nfit N] [--threads T] [--gelam F] [--swlim F]\n");
        return 2;
    }
    int l0 = 0, l1 = 0;
    if (sscanf(layers, "%d-%d", &l0, &l1) != 2) l0 = l1 = atoi(layers);

    for (int L = l0; L <= l1; L++) {
        time_t t0 = time(NULL);
        char p[1024]; npy_meta mx, mi, mw, mq, mf;
        snprintf(p, sizeof p, "%s/ffn_in_L%d.npy", cap, L);   float *x   = npy_read_f32(p, &mx);
        snprintf(p, sizeof p, "%s/route_L%d.npy", cap, L);    float *id  = npy_read_f32(p, &mi);
        snprintf(p, sizeof p, "%s/route_w_L%d.npy", cap, L);  float *rw  = npy_read_f32(p, &mw);
        snprintf(p, sizeof p, "%s/obase_v3_L%d.npy", cap, L); float *yq  = npy_read_f32(p, &mq);
        snprintf(p, sizeof p, "%s/routed_L%d.npy", cap, L);   float *yf  = npy_read_f32(p, &mf);
        if (!x || !id || !rw || !yq || !yf || mx.shape[0] != mf.shape[0]) {
            fprintf(stderr, "L%d: cap 张量缺/行数不齐\n", L);
            free(x); free(id); free(rw); free(yq); free(yf); continue;
        }
        const int n = (int)mx.shape[0];
        const int NFIT = nfit_arg > 0 ? nfit_arg : n * 8 / 10;

        snprintf(p, sizeof p, "%s/dql_vq_L%02d.bin", dql, L);
        FILE *bf = fopen(p, "rb");
        if (!bf) { fprintf(stderr, "L%d: %s 打不开\n", L, p); return 3; }
        fseek(bf, 0, SEEK_END); long bsz = ftell(bf); fseek(bf, 0, SEEK_SET);
        uint8_t *blob = malloc((size_t)bsz);
        if (fread(blob, 1, (size_t)bsz, bf) != (size_t)bsz) { fprintf(stderr, "L%d blob 读断\n", L); return 3; }
        fclose(bf);
        if (!ds4vq_blob_ok(blob, (size_t)bsz)) { fprintf(stderr, "L%d blob 魔数错\n", L); return 3; }

        /* pair_buf: [n*TOPK, DM] = w·Y_e(量化个体输出) */
        float *pair = calloc((size_t)n * TOPK * DM, sizeof(float));
        if (!pair) { fprintf(stderr, "L%d OOM pair(%zu MB)\n", L, (size_t)n * TOPK * DM * 4 >> 20); return 3; }

        /* 专家分桶 */
        int *cnt = calloc(NEXP, sizeof(int));
        for (int t = 0; t < n; t++)
            for (int s = 0; s < TOPK; s++) { int e = (int)id[(size_t)t * TOPK + s]; if (e >= 0 && e < NEXP) cnt[e]++; }
        int **erows = malloc(NEXP * sizeof(int *)), **eslots = malloc(NEXP * sizeof(int *));
        int *fill = calloc(NEXP, sizeof(int));
        for (int e = 0; e < NEXP; e++) { erows[e] = malloc(cnt[e] * sizeof(int)); eslots[e] = malloc(cnt[e] * sizeof(int)); }
        for (int t = 0; t < n; t++)
            for (int s = 0; s < TOPK; s++) {
                int e = (int)id[(size_t)t * TOPK + s];
                if (e >= 0 && e < NEXP) { erows[e][fill[e]] = t; eslots[e][fill[e]] = s; fill[e]++; }
            }

        /* 桶前向(线程池: 每线程一批专家) */
        pthread_t th[64]; fw_task tk[NEXP]; int nth = threads > 64 ? 64 : threads;
        int einext = 0;
        int order[NEXP], no = 0;
        for (int e = 0; e < NEXP; e++) if (cnt[e] > 0) order[no++] = e;
        /* 简单静态划分: 逐专家串行分发到 nth 线程组 */
        for (int base = 0; base < no; base += nth) {
            int batch = no - base < nth ? no - base : nth;
            for (int i = 0; i < batch; i++) {
                int e = order[base + i];
                tk[i] = (fw_task){blob, x, rw, pair, erows[e], eslots[e], cnt[e], e, swlim};
                pthread_create(&th[i], NULL, fw_worker, &tk[i]);
            }
            for (int i = 0; i < batch; i++) pthread_join(th[i], NULL);
            einext = base + batch;
            if ((einext % 64) < nth) { fprintf(stderr, "  L%d 前向 …%d/%d\r", L, einext, no); fflush(stderr); }
        }
        fprintf(stderr, "\n");

        /* dH = y* − yq */
        float *dH = malloc((size_t)n * DM * sizeof(float));
        for (size_t q = 0; q < (size_t)n * DM; q++) dH[q] = yf[q] - yq[q];

        /* 共燃 Gram(训练段) */
        double *G = calloc((size_t)NEXP * NEXP, sizeof(double));
        double *rhs = calloc(NEXP, sizeof(double));
        for (int t = 0; t < NFIT; t++) {
            int es[TOPK]; const float *vs[TOPK]; int m = 0;
            for (int s = 0; s < TOPK; s++) {
                int e = (int)id[(size_t)t * TOPK + s];
                if (e < 0 || e >= NEXP) continue;
                es[m] = e; vs[m] = pair + ((size_t)t * TOPK + s) * DM; m++;
            }
            const float *rt = dH + (size_t)t * DM;
            for (int a = 0; a < m; a++) {
                double br = 0;
                for (int d = 0; d < DM; d++) br += (double)vs[a][d] * rt[d];
                rhs[es[a]] += br;
                for (int b = a; b < m; b++) {
                    double dd = 0;
                    for (int d = 0; d < DM; d++) dd += (double)vs[a][d] * vs[b][d];
                    G[(size_t)es[a] * NEXP + es[b]] += dd;
                    if (es[a] != es[b]) G[(size_t)es[b] * NEXP + es[a]] += dd;
                }
            }
        }
        double trG = 0;
        for (int e = 0; e < NEXP; e++) trG += G[(size_t)e * NEXP + e];
        double ridge = gelam * (trG / NEXP > 1.0 ? trG / NEXP : 1.0);
        for (int e = 0; e < NEXP; e++) G[(size_t)e * NEXP + e] += ridge;
        if (chol_solve_small(G, NEXP, rhs)) { fprintf(stderr, "L%d chol 失败\n", L); return 4; }

        /* held 评估: rel 能量挽回 + cos 前后 */
        double e0 = 0, e1 = 0, c0 = 0, c1 = 0;
        int nev = n - NFIT;
        for (int t = NFIT; t < n; t++) {
            const float *rt = dH + (size_t)t * DM;
            const float *yqt = yq + (size_t)t * DM, *yft = yf + (size_t)t * DM;
            double yh[4096];
            for (int d = 0; d < DM; d++) yh[d] = yqt[d];
            for (int s = 0; s < TOPK; s++) {
                int e = (int)id[(size_t)t * TOPK + s];
                if (e < 0 || e >= NEXP) continue;
                const float *v = pair + ((size_t)t * TOPK + s) * DM;
                const double g = rhs[e];
                for (int d = 0; d < DM; d++) yh[d] += g * v[d];
            }
            double da = 0, na = 0, nb = 0, db = 0, n2 = 0;
            for (int d = 0; d < DM; d++) {
                double eb = (double)rt[d]; e0 += eb * eb;
                double ea = (double)yft[d] - yh[d]; e1 += ea * ea;
                da += (double)yqt[d] * yft[d]; na += (double)yqt[d] * yqt[d]; nb += (double)yft[d] * yft[d];
                db += yh[d] * yft[d]; n2 += yh[d] * yh[d];
            }
            c0 += (na > 0 && nb > 0) ? da / (sqrt(na) * sqrt(nb)) : 0;
            c1 += (n2 > 0 && nb > 0) ? db / (sqrt(n2) * sqrt(nb)) : 0;
        }
        double rec = 1.0 - e1 / (e0 > 0 ? e0 : 1);
        double gabs = 0;
        for (int e = 0; e < NEXP; e++) gabs += fabs(rhs[e]);
        printf("★L%d GE(C): held挽回 %.2f%%  cos %.4f→%.4f (Δ%+.4f)  mean|g| %.4f | %lds\n",
               L, rec * 100, c0 / nev, c1 / nev, (c1 - c0) / nev, gabs / NEXP,
               (long)(time(NULL) - t0));
        fflush(stdout);

        /* zrec 写出: bf.GE + 256×f16(ge = 1+g) */
        snprintf(p, sizeof p, "%s/zrec_L%02d.bin", outd, L);
        FILE *f = fopen(p, "wb");
        uint8_t hdr[116]; memset(hdr, 0, sizeof hdr);
        memcpy(hdr, "bf.GE", 5);
        uint64_t psz = NEXP * 2; memcpy(hdr + 88, &psz, 8);
        int32_t one = 1; memcpy(hdr + 112, &one, 4);
        fwrite(hdr, 1, 116, f);
        uint16_t geh[NEXP];
        for (int e = 0; e < NEXP; e++) geh[e] = f32_to_f16((float)(1.0 + rhs[e]));
        fwrite(geh, 2, NEXP, f); fclose(f);

        free(x); free(id); free(rw); free(yq); free(yf); free(pair); free(dH);
        free(G); free(rhs); free(blob); free(cnt); free(fill);
        for (int e = 0; e < NEXP; e++) { free(erows[e]); free(eslots[e]); }
        free(erows); free(eslots);
    }
    return 0;
}
