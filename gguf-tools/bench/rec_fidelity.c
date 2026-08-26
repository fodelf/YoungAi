/* rec_fidelity.c — 注入载荷忠实度对拍器(C, 2026-08-25 Python→C 迁移 Wave A)。
 * 取代 zlever/rec_fidelity.py。把 dql 注入记录按回放公式(type5 GE / type6 zl.RRR 含
 * ftA φ 提升)重放在解算 zcache 的同一批 x 行上, 重评 held/fit 挽回率 —— S1↔S2 对拍仪。
 * zcache=.npz(np.savez 默认 ZIP_STORED 无压缩): 读取走 calib/npy.c 的 npz_get
 * (2026-08-26 并入, 全仓唯一 npz 实现; 原内嵌解析器逐行搬过去, 数值不变)。
 * 行集与发车 env 同式: FR=块0-23 剔前64, ER=块24-31 剔前64。
 * 用法: rec_fidelity <layers_dir> <L> [orig_len=855638144] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "npy.h"

#define D 4096

/* f16→f64: dql fp16 载荷(GE 门/z/U/V)解码用, 精确 IEEE 展开 */
static double f16_to_f64(uint16_t h) {
    int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = ldexp(m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexp(m + 1024, e - 25);
    return s ? -v : v;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: rec_fidelity <layers_dir> <L> [orig=855638144]\n"); return 1; }
    const char *ld = argv[1]; int L = atoi(argv[2]);
    long long orig = argc > 3 ? atoll(argv[3]) : 855638144LL;
    char p[1024];

    /* dql 注入记录(先扫描, 打印序与 .py 一致) */
    snprintf(p, sizeof p, "%s/dql_L%02d.bin", ld, L);
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", p); return 2; }
    fseek(f, 0, SEEK_END); long long dsz = ftell(f);
    printf("L%d 注入区 %lld B, 记录: ", L, dsz - orig);
    long long off = orig;
    typedef struct { char nm[17]; long long pay; uint64_t psz; } rr_t;
    rr_t recs[16]; int nrec = 0;
    while (off + 116 <= dsz && nrec < 16) {
        uint8_t hdr[116];
        fseek(f, (long)off, SEEK_SET);
        if (fread(hdr, 1, 116, f) != 116) break;
        memcpy(recs[nrec].nm, hdr, 16); recs[nrec].nm[16] = 0;
        memcpy(&recs[nrec].psz, hdr + 88, 8);
        recs[nrec].pay = off + 116;
        printf("%s%s[%lluB]", nrec ? " " : "", recs[nrec].nm, (unsigned long long)recs[nrec].psz);
        off += 116 + (long long)recs[nrec].psz;
        nrec++;
    }
    printf("\n");

    /* zcache */
    snprintf(p, sizeof p, "%s/zcache_L%02d.npz", ld, L);
    FILE *zf = fopen(p, "rb");
    if (!zf) { fprintf(stderr, "%s 打不开\n", p); return 2; }
    fseek(zf, 0, SEEK_END); long long zsz = ftell(zf); fseek(zf, 0, SEEK_SET);
    uint8_t *zbuf = malloc(zsz);
    if (fread(zbuf, 1, zsz, zf) != (size_t)zsz) return 2;
    fclose(zf);
    npz_arr dH, prow, pe, pw, pYQ, X;
    if (npz_get(zbuf, zsz, "dH", &dH) || npz_get(zbuf, zsz, "prow", &prow) ||
        npz_get(zbuf, zsz, "pe", &pe) || npz_get(zbuf, zsz, "pw", &pw) ||
        npz_get(zbuf, zsz, "pYQ", &pYQ)) { fprintf(stderr, "zcache 字段缺\n"); return 2; }
    int hasX = npz_get(zbuf, zsz, "xcap", &X) == 0;
    int NTOK = (int)dH.d0;
    printf("zcache: dH(%d, %d) X=%s 对数=%lld\n", NTOK, (int)dH.d1, hasX ? "有" : "无(锚fin口径)", (long long)prow.n);
    if (!hasX) { fprintf(stderr, "无 xcap 的 zcache(锚fin口径)本工具不重建 x — 与 .py 行为一致需锚, 拒跑\n"); return 3; }
    /* 残差 = dH 副本 */
    double *resid = malloc((size_t)NTOK * D * 8);
    memcpy(resid, dH.v, (size_t)NTOK * D * 8);

    for (int ri = 0; ri < nrec; ri++) {
        uint8_t *pay = malloc(recs[ri].psz);
        fseek(f, (long)recs[ri].pay, SEEK_SET);
        if (fread(pay, 1, recs[ri].psz, f) != recs[ri].psz) return 2;
        if (strstr(recs[ri].nm, "bf.GE") && recs[ri].psz >= 512) {
            const uint16_t *geh = (const uint16_t *)pay;
            int neff = 0;
            double ge[256];
            for (int e = 0; e < 256; e++) { ge[e] = f16_to_f64(geh[e]); if (fabs(ge[e] - 1.0) > 1e-4) neff++; }
            for (long long i = 0; i < prow.n; i++) {
                int t = (int)prow.v[i], e = (int)pe.v[i];
                double w = pw.v[i], g1 = ge[e] - 1.0;
                double *rt = resid + (size_t)t * D;
                const double *yq = pYQ.v + (size_t)i * D;
                for (int d2 = 0; d2 < D; d2++) rt[d2] -= g1 * w * yq[d2];
            }
            printf("  bf.GE: 活门=%d\n", neff);
        } else if (strstr(recs[ri].nm, "zl.RRR") && recs[ri].psz >= 16) {
            uint32_t k, din, dout; float tr;
            memcpy(&k, pay, 4); memcpy(&tr, pay + 4, 4); memcpy(&din, pay + 8, 4); memcpy(&dout, pay + 12, 4);
            const uint16_t *h = (const uint16_t *)(pay + 16);
            printf("  zl.RRR: k=%u tr=%.1e din=%u(%s) dout=%u\n", k, tr, din, din == 3 * D ? "ftA" : "lin", dout);
            /* pv = (φ(x)ᵀV)·z; zd = U·pv — 与回放同式(tr=大值时不夹, 与 .py 同: 不建模 clip) */
            double *z = malloc(k * 8), *U = malloc((size_t)dout * k * 8), *V = malloc((size_t)din * k * 8);
            for (uint32_t i = 0; i < k; i++) z[i] = f16_to_f64(h[i]);
            for (uint64_t i = 0; i < (uint64_t)dout * k; i++) U[i] = f16_to_f64(h[k + i]);
            for (uint64_t i = 0; i < (uint64_t)din * k; i++) V[i] = f16_to_f64(h[k + (uint64_t)dout * k + i]);
            double *phi = malloc((size_t)din * 8), *pv = malloc(k * 8);
            for (int t = 0; t < NTOK; t++) {
                const double *x = X.v + (size_t)t * D;
                if (din == 3 * D) {
                    double ss = 0; for (int d2 = 0; d2 < D; d2++) ss += x[d2] * x[d2];
                    /* .py: rms 用 float32 X 算 mean 后开方 +1e-6, φ 前两块 f32 精度 —
                     * 这里全 f64, 差在 1e-7 相对量级, 打印精度内一致 */
                    double nr = sqrt(ss / D) + 1e-6;
                    for (int d2 = 0; d2 < D; d2++) { phi[d2] = x[d2]; phi[D + d2] = x[d2] * x[d2] / nr; phi[2 * D + d2] = x[d2] > 0 ? x[d2] : 0; }
                } else memcpy(phi, x, (size_t)D * 8);
                for (uint32_t c = 0; c < k; c++) {
                    double a = 0;
                    for (uint32_t d2 = 0; d2 < din; d2++) a += phi[d2] * V[(size_t)d2 * k + c];
                    pv[c] = a * z[c];
                }
                double *rt = resid + (size_t)t * D;
                for (int d2 = 0; d2 < D; d2++) {
                    double a = 0; const double *Ud = U + (size_t)d2 * k;
                    for (uint32_t c = 0; c < k; c++) a += pv[c] * Ud[c];
                    rt[d2] -= a;
                }
            }
            free(z); free(U); free(V); free(phi); free(pv);
        }
        free(pay);
    }
    fclose(f);
    /* 行集 FR/ER + 挽回率 */
    double e0f = 0, e1f = 0, e0e = 0, e1e = 0;
    for (int b = 0; b < 32; b++) {
        for (int r = b * 256 + 64; r < (b + 1) * 256 && r < NTOK; r++) {
            const double *dh = dH.v + (size_t)r * D, *rs = resid + (size_t)r * D;
            double a = 0, e = 0;
            for (int d2 = 0; d2 < D; d2++) { a += dh[d2] * dh[d2]; e += rs[d2] * rs[d2]; }
            if (b < 24) { e0f += a; e1f += e; } else { e0e += a; e1e += e; }
        }
    }
    printf("★S2 载荷重放挽回: held(ev)=%.1f%%  fit(tr)=%.1f%%   [S1 解算自评对表: 日志值]\n",
           (1 - e1e / e0e) * 100, (1 - e1f / e0f) * 100);
    return 0;
}
