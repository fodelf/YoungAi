/* z_leverage_test.c — 用【用户自己的求解器】验证 z 方案的杠杆上限(2026-08-08)。
 *
 * 不再用我自己写的 SVD/numpy: 直接调 ds4_z.c 的闭式 ridge 解 + ds4_z.h 的秩梯子,
 * 并用 ds4_loss.c 的四损失(align/classify/smooth/fixed)作为评估口径。
 *
 * 输入(Python 侧准备的真实数据, 二进制 f32):
 *   X.bin  [n, d_in ]  层输入激活(锚 fin, 部署同源)
 *   R.bin  [n, d_out]  目标修正 = FP 层输出 − 量化层输出(整层 MoE, 含路由/SwiGLU/down)
 * 划分: 前 n_fit 行训练, 其余 held-out。
 *
 * 报: 各 rank 下 held-out 的 ①残差能量下降 ②四损失各项 ③等效 bpw ④z 体积/杠杆
 * 用法: z_leverage_test X.bin R.bin n d_in d_out n_fit lambda
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ds4_z.c"     /* 提供 ds4_z 结构 + static 原语 cholesky/mgs/lcg(量化器同款 include 方式) */
#include "ds4_loss.c"
#include "z_chain.inc"  /* 从 ds4quant_run.c 原样抽出的 z_solve_dual + z_solve_fourloss */

static float *rdbin(const char *p, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
    float *b = malloc(n * sizeof(float));
    if (fread(b, sizeof(float), n, f) != n) { fprintf(stderr, "short %s\n", p); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "用法见文件头\n"); return 2; }
    const char *xp = argv[1], *rp = argv[2];
    uint32_t n = atoi(argv[3]), din = atoi(argv[4]), dout = atoi(argv[5]), nfit = atoi(argv[6]);
    float lam = atof(argv[7]);
    uint32_t nho = n - nfit;
    float *X = rdbin(xp, (size_t)n * din), *R = rdbin(rp, (size_t)n * dout);
    const float *Xho = X + (size_t)nfit * din, *Rho = R + (size_t)nfit * dout;

    /* 基线: 不修正时 held-out 的残差能量 + 四损失(与"零向量预测"比) */
    double base = 0;
    for (size_t i = 0; i < (size_t)nho * dout; i++) base += (double)Rho[i] * Rho[i];
    float *zero = calloc((size_t)nho * dout, sizeof(float));
    float *wdim = malloc((size_t)dout * sizeof(float));
    ds4_loss_dim_variance(Rho, nho, dout, wdim);
    printf("held-out 基线: 残差能量 %.6g | align %.4f | cls %.6g\n",
           base, ds4_loss_align(zero, Rho, nho, dout),
           ds4_loss_classify(zero, Rho, wdim, nho, dout));
    free(zero);

    const uint32_t RMAX = (argc > 8) ? (uint32_t)atoi(argv[8]) : 128;
    /* ★用户设计全套: L_classify 列权(per-dim 方差) + L_smooth dither 增广 + L_fixed ridge ★ */
    float *colw = malloc((size_t)dout * sizeof(float));
    ds4_loss_dim_variance(R, nfit, dout, colw);
    printf("\n解 z(z_solve_fourloss: 四损失进求解, dither 增广 %u 行, λ=%.3g, rank<=%u, 训练 %u 行)…\n",
           nfit, lam, RMAX, nfit);
    ds4_z *zl = z_solve_fourloss(X, R, (int)nfit, (int)din, (int)dout, (int)RMAX, lam, colw, (int)nfit, 0.25f, 0x5A5A1EEDULL);
    if (!zl) { fprintf(stderr, "z_solve 返回 NULL(退化/OOM)\n"); return 3; }

    printf("\n%5s %14s %10s %9s %9s %11s %10s\n",
           "rank", "held残差能量", "挽回", "align", "cls", "等效Δbpw", "z体积/层");
    float *pred = malloc((size_t)dout * sizeof(float));
    float *P = malloc((size_t)nho * dout * sizeof(float));
    const uint32_t KS[] = {1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512};
    for (int ki = 0; ki < 12; ki++) {
        uint32_t k = KS[ki];
        if (k > zl->rank) break;
        ds4_z_set_rank(zl, k);
        double resid = 0;
        for (uint32_t i = 0; i < nho; i++) {
            memset(pred, 0, (size_t)dout * sizeof(float));
            ds4_z_apply(zl, Xho + (size_t)i * din, pred);
            for (uint32_t j = 0; j < dout; j++) {
                float e = Rho[(size_t)i * dout + j] - pred[j];
                P[(size_t)i * dout + j] = pred[j];
                resid += (double)e * e;
            }
        }
        double rtr = 0, btr = 0;                            /* 训练集还原度(诊断) */
        for (uint32_t i = 0; i < nfit; i++) {
            memset(pred, 0, (size_t)dout * sizeof(float));
            ds4_z_apply(zl, X + (size_t)i * din, pred);
            for (uint32_t j = 0; j < dout; j++) {
                float t = R[(size_t)i * dout + j];
                float e = t - pred[j];
                rtr += (double)e * e; btr += (double)t * t;
            }
        }
        double ftr = 1.0 - rtr / btr;
        double f = 1.0 - resid / base;                      /* 挽回比例 */
        double dbpw = f > 0 && f < 1 ? 0.5 * log2(1.0 / (1.0 - f)) : (f >= 1 ? 99 : 0);
        double vol = (double)k * (din + dout) * 2.0 / 1048576.0;   /* fp16 MiB/层 */
        printf("%5u %14.6g %9.1f%% %9.4f %9.4g %10.3f %9.2fM  [训练 %.1f%%]\n",
               k, resid, f * 100.0,
               ds4_loss_align(P, Rho, nho, dout),
               ds4_loss_classify(P, Rho, wdim, nho, dout),
               dbpw, vol, ftr * 100.0);
    }
    ds4_z_free(zl);
    return 0;
}
