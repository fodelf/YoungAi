/* kl_forensic.c — 逐位置 KL 分解取证器(C, 2026-08-25 Python→C 迁移 Wave A)。
 * 取代 zlever/kl_forensic.py(铁律: 链上不留 Python)。逐式同源:
 *   KL_i = Σ_v p_ref·(ln p_ref − ln p_stu)  (log-softmax f64, max 平移)
 *   ΔKL = KL_B − KL_A; 按参考 NLL 分桶(难度); 最差 1%/5% 集中度; top10 罪犯位置。
 * 锚格式 DQA2(同 anchor_metrics.c): 头 32B + idh 8B + fin/ridx/rw/H 跳过 + logits[S][V]。
 * 学生格式: 尾对齐 S*V*4, 允许 0/8B 头(与 .py 同容差)。
 * 用法: kl_forensic <anchor.bin> <ids.txt> <stuA.bin> <stuB.bin> [标签A 标签B] */
#define _GNU_SOURCE   /* glibc qsort_r 声明 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

static int S, V;

static float *map_ref(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8 || hd[0] != 0x32415144u) { fprintf(stderr, "%s: 不是 DQA2 锚\n", path); exit(2); }
    S = (int)hd[1]; V = (int)hd[5];
    int HCM = (int)hd[2], DIM = (int)hd[3], NL = (int)hd[4], NACT = (int)hd[6];
    long long off = 40 + (long long)NL * S * DIM * 4 + 2LL * NL * S * NACT * 4
                  + (long long)NL * S * HCM * DIM * 4;
    float *lg = malloc((size_t)S * V * 4);
    fseek(f, (long)off, SEEK_SET);
    if (fread(lg, 4, (size_t)S * V, f) != (size_t)S * V) { fprintf(stderr, "%s: logits 读不满\n", path); exit(2); }
    fclose(f);
    return lg;
}
static float *map_stu(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long long sz = ftell(f);
    long long off = sz - (long long)S * V * 4;
    if (off != 0 && off != 8) { fprintf(stderr, "%s 尺寸异常 off=%lld\n", path, off); exit(2); }
    fseek(f, (long)off, SEEK_SET);
    float *lg = malloc((size_t)S * V * 4);
    if (fread(lg, 4, (size_t)S * V, f) != (size_t)S * V) { fprintf(stderr, "%s: 读不满\n", path); exit(2); }
    fclose(f);
    return lg;
}

/* 行级: log-softmax(f64) 与 .py 完全同式(max 平移 + ln Σexp) */
static void logsm(const float *row, double *out) {
    double mx = -1e300;
    for (int v = 0; v < V; v++) if (row[v] > mx) mx = row[v];
    double s = 0;
    for (int v = 0; v < V; v++) { out[v] = (double)row[v] - mx; s += exp(out[v]); }
    double ls = log(s);
    for (int v = 0; v < V; v++) out[v] -= ls;
}

typedef struct { const float *R, *A, *B; const long *ids; double *klA, *klB, *nllR; int i0, i1; } warg_t;
static void *worker(void *p) {
    warg_t *w = p;
    double *lr = malloc((size_t)V * 8), *la = malloc((size_t)V * 8), *lb = malloc((size_t)V * 8);
    for (int i = w->i0; i < w->i1; i++) {
        logsm(w->R + (size_t)i * V, lr);
        logsm(w->A + (size_t)i * V, la);
        logsm(w->B + (size_t)i * V, lb);
        double ka = 0, kb = 0;
        for (int v = 0; v < V; v++) {
            double pr = exp(lr[v]);
            ka += pr * (lr[v] - la[v]);
            kb += pr * (lr[v] - lb[v]);
        }
        w->klA[i] = ka; w->klB[i] = kb;
        w->nllR[i] = -lr[w->ids[i + 1]];
    }
    free(lr); free(la); free(lb);
    return NULL;
}

static int cmp_desc_idx(const void *a, const void *b, void *ctx) {
    const double *d = ctx; int ia = *(const int *)a, ib = *(const int *)b;
    return d[ia] < d[ib] ? 1 : (d[ia] > d[ib] ? -1 : 0);
}
#if defined(__APPLE__)
static const double *g_d;
static int cmp_apple(void *ctx, const void *a, const void *b) { return cmp_desc_idx(a, b, (void *)ctx); }
#endif
static double med(double *v, int n) {   /* 破坏性中位(与 np.median 偶数取均值同) */
    /* 简单排序足够: n≈2652 */
    for (int i = 1; i < n; i++) { double x = v[i]; int j = i - 1; while (j >= 0 && v[j] > x) { v[j+1] = v[j]; j--; } v[j+1] = x; }
    return n & 1 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "用法: kl_forensic <anchor> <ids> <stuA> <stuB> [标签A 标签B]\n"); return 1; }
    const char *la = argc > 5 ? argv[5] : "A", *lb = argc > 6 ? argv[6] : "B";
    float *R = map_ref(argv[1]);
    long *ids = malloc((size_t)S * sizeof(long)); int nid = 0;
    { FILE *f = fopen(argv[2], "r"); if (!f) { fprintf(stderr, "ids 打不开\n"); return 2; }
      while (nid < S && fscanf(f, "%ld", &ids[nid]) == 1) nid++;
      fclose(f);
      if (nid != S) { fprintf(stderr, "ids %d != S %d\n", nid, S); return 2; } }
    float *A = map_stu(argv[3]), *B = map_stu(argv[4]);
    int n = S - 1;
    double *klA = malloc(n * 8), *klB = malloc(n * 8), *nllR = malloc(n * 8);
    int nth = 16; pthread_t th[16]; warg_t w[16];
    int per = (n + nth - 1) / nth;
    for (int t = 0; t < nth; t++) {
        w[t] = (warg_t){R, A, B, ids, klA, klB, nllR, t * per, (t + 1) * per > n ? n : (t + 1) * per};
        pthread_create(&th[t], NULL, worker, &w[t]);
    }
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);

    double mA = 0, mB = 0, tot = 0;
    double *d = malloc(n * 8), *dc = malloc(n * 8);
    for (int i = 0; i < n; i++) { mA += klA[i]; mB += klB[i]; d[i] = klB[i] - klA[i]; tot += d[i]; dc[i] = d[i]; }
    printf("n=%d  KL[%s]=%.5f  KL[%s]=%.5f  ΔKL(mean)=%+.5f  Δ(中位)=%+.5f\n",
           n, la, mA / n, lb, mB / n, tot / n, med(dc, n));
    static const double edges[] = {0, 0.1, 0.5, 1.5, 3.0, 6.0, 99};
    /* 桶标签逐字符复刻 .py 的 f"{a:>5}-{b:<6}"(混合 int/float 字面量) */
    static const char *blab[6] = {"    0-0.1   ", "  0.1-0.5   ", "  0.5-1.5   ",
                                  "  1.5-3.0   ", "  3.0-6.0   ", "  6.0-99    "};
    printf("%12s %5s %9s-KL %7s-KL %9s %9s\n", "ref-NLL桶", "n", la, lb, "ΔKL均值", "Δ总量占比");
    for (int j = 0; j < 6; j++) {
        int cnt = 0; double sa = 0, sb = 0, sd = 0;
        for (int i = 0; i < n; i++) if (nllR[i] >= edges[j] && nllR[i] < edges[j+1]) { cnt++; sa += klA[i]; sb += klB[i]; sd += d[i]; }
        if (!cnt) continue;
        printf("%s %5d %9.4f %9.4f %+9.4f %8.1f%%\n",
               blab[j], cnt, sa / cnt, sb / cnt, sd / cnt, tot != 0 ? sd / tot * 100 : 0);
    }
    int *ord = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) ord[i] = i;
#if defined(__APPLE__)
    g_d = d; qsort_r(ord, n, sizeof(int), (void *)d, cmp_apple);
#else
    qsort_r(ord, n, sizeof(int), cmp_desc_idx, d);
#endif
    for (int fi = 0; fi < 2; fi++) {
        double frac = fi ? 0.05 : 0.01;
        int k = (int)(n * frac); if (k < 1) k = 1;
        double s = 0; for (int i = 0; i < k; i++) s += d[ord[i]];
        printf("最差%.0f%%位置(%d个)承担 Δ 总量的 %.1f%%\n", frac * 100, k, tot != 0 ? s / tot * 100 : 0);
    }
    printf("top10 罪犯位置(pos tgt refNLL KLa→KLb):\n");
    for (int i = 0; i < 10 && i < n; i++) {
        int p = ord[i];
        printf("  pos=%d tgt=%ld refNLL=%.2f %.3f→%.3f (Δ%+.3f)\n",
               p, ids[p + 1], nllR[p], klA[p], klB[p], d[p]);
    }
    return 0;
}
