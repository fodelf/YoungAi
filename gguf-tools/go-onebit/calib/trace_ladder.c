/* trace_ladder.c — 单token逐层分叉梯分析器(C, 2026-08-25 Python→C 迁移 Wave A)。
 * 取代 zlever/trace_ladder.py。输入=trace 脚手架的 dump(traceH_*.bin: 逐条
 * <i32 L><i32 row><f32 H[HCM*DIM]>) + DQA2 锚的 H 区(FP 参考) + 武装链日志里的
 * [TRACE] type6 注入行。逐行输出逐层: relL2(armed vs bare / bare vs FP / armed vs FP)
 * + |zd| |routed| sc2, 相邻层 armed-vs-bare 跳变>0.05 标 ◀跳变。
 * 用法: trace_ladder <traceH_bare> <traceH_armed> <anchor.bin> <trace_armed.log> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAXROWS 16
typedef struct { int L, r; float *h; } rec_t;
static rec_t *load_dump(const char *p, int *nrec, int ROW) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", p); exit(2); }
    rec_t *rs = NULL; int n = 0, cap = 0;
    for (;;) {
        int32_t hd[2];
        if (fread(hd, 4, 2, f) != 2) break;
        float *h = malloc((size_t)ROW * 4);
        if (fread(h, 4, ROW, f) != (size_t)ROW) { free(h); break; }
        if (n == cap) { cap = cap ? cap * 2 : 256; rs = realloc(rs, cap * sizeof(rec_t)); }
        rs[n++] = (rec_t){hd[0], hd[1], h};
    }
    fclose(f); *nrec = n;
    return rs;
}
static float *find_rec(rec_t *rs, int n, int L, int r) {
    for (int i = 0; i < n; i++) if (rs[i].L == L && rs[i].r == r) return rs[i].h;
    return NULL;
}
static double rel(const float *x, const float *y, int n) {
    double e = 0, a = 0;
    for (int i = 0; i < n; i++) { double d = (double)x[i] - y[i]; e += d * d; a += (double)y[i] * y[i]; }
    return sqrt(e) / (sqrt(a) + 1e-30);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "用法: trace_ladder <bare.bin> <armed.bin> <anchor> <armed.log>\n"); return 1; }
    /* 锚头 → 维度 + H 区偏移 */
    FILE *af = fopen(argv[3], "rb");
    if (!af) { fprintf(stderr, "锚打不开\n"); return 2; }
    uint32_t hd[8];
    if (fread(hd, 4, 8, af) != 8 || hd[0] != 0x32415144u) { fprintf(stderr, "非 DQA2 锚\n"); return 2; }
    int S = hd[1], HCM = hd[2], DIM = hd[3], NL = hd[4], NACT = hd[6];
    int ROW = HCM * DIM;
    long long hbase = 40 + (long long)NL * S * DIM * 4 + 2LL * NL * S * NACT * 4;
    int nb, na;
    rec_t *B = load_dump(argv[1], &nb, ROW), *A = load_dump(argv[2], &na, ROW);
    /* zd 注入行: [TRACE] L=%d row=%d type6 |zd|=%f |routed|=%f sc2=%f */
    typedef struct { int L, r; double zd, rt, sc; } zline_t;
    zline_t *Z = NULL; int nz = 0, zc = 0;
    { FILE *lf = fopen(argv[4], "r"); char ln[512];
      if (lf) { while (fgets(ln, sizeof ln, lf)) {
          zline_t z; char *p = strstr(ln, "[TRACE] L=");
          if (p && sscanf(p, "[TRACE] L=%d row=%d type6 |zd|=%lf |routed|=%lf sc2=%lf",
                          &z.L, &z.r, &z.zd, &z.rt, &z.sc) == 5) {
              if (nz == zc) { zc = zc ? zc * 2 : 256; Z = realloc(Z, zc * sizeof(zline_t)); }
              Z[nz++] = z; } }
        fclose(lf); } }
    /* 追踪行集合 = dump 里出现过的 row 去重升序 */
    int rows[MAXROWS], nrows = 0;
    for (int i = 0; i < nb; i++) {
        int seen = 0;
        for (int j = 0; j < nrows; j++) if (rows[j] == B[i].r) seen = 1;
        if (!seen && nrows < MAXROWS) rows[nrows++] = B[i].r;
    }
    for (int i = 1; i < nrows; i++) { int x = rows[i], j = i - 1; while (j >= 0 && rows[j] > x) { rows[j+1] = rows[j]; j--; } rows[j+1] = x; }
    float *fp = malloc((size_t)ROW * 4);
    for (int ri = 0; ri < nrows; ri++) {
        int r = rows[ri];
        printf("===== row %d =====\n", r);
        printf("%3s %13s %11s %12s %8s %9s %5s\n", "L", "armed-vs-bare", "bare-vs-FP", "armed-vs-FP", "|zd|", "|routed|", "sc2");
        double prev = 0.0;
        for (int L = 0; L < NL; L++) {
            float *hb = find_rec(B, nb, L, r), *ha = find_rec(A, na, L, r);
            if (!hb || !ha) continue;
            fseek(af, (long)(hbase + ((long long)L * S + r) * ROW * 4), SEEK_SET);
            if (fread(fp, 4, ROW, af) != (size_t)ROW) break;
            double ab = rel(ha, hb, ROW), bf = rel(hb, fp, ROW), afp = rel(ha, fp, ROW);
            zline_t *z = NULL;
            for (int i = 0; i < nz; i++) if (Z[i].L == L && Z[i].r == r) { z = &Z[i]; break; }
            printf("%3d %13.4f %11.4f %12.4f ", L, ab, bf, afp);
            if (z) printf("%8.3f %9.2f %5.2f", z->zd, z->rt, z->sc);
            else   printf("%24s", "");
            printf("%s\n", ab - prev > 0.05 ? " ◀跳变" : "");
            prev = ab;
        }
    }
    return 0;
}
