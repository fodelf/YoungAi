/* zloss_cond.inc.c — ★条件化修正臂(用户理论获证配方, 只被 zloss_solve.c include)★
 *
 * 依据 fable5 2026-07-05 真尺判决: 全局线性 z rank16 = −5915%(彻底崩, 函数形态错)
 * vs 条件化 z(top1 专家分组 g_e+b_e, ~2MB) = +9.0pp 首次转正 ⇒ 用户理论
 * "Go 严格语法 ⟹ 修正按离散路由结构切换"获证。本片即该配方的 C 实现。 */

/* ★条件化 z(用户理论 2026-07-05 获证配方, fable5 line482 复活)★
 * 历史判决: 全局线性 z rank16 = −5915%(彻底崩, 函数形态错) vs 条件化 z
 * (top1 专家分组 g_e+b_e, ~2MB) = +9.0pp 首次转正 ⇒ "修正按离散路由结构切换"。
 * 本臂即那份配方: 按 token 的 top1 专家分组, 每组逐通道闭式解 增益 g[j] 与偏置 b[j]:
 *   corr[t][j] = g[e(t)][j]·Ys[t][j] + b[e(t)][j]
 * 闭式(中心化最小二乘, 组内): g_j = Σ R_c·Ys_c / Σ Ys_c², b_j = mean(R) − g_j·mean(Ys)。
 * 门=路由本身(部署免费), 参数 2×256×D fp16 ≈ 4MB/层。 */
static void run_cond_arm(const float *X, const float *R, const float *Ys,
                         const float *Yt_ev, const float *wv, const int *top1,
                         const int *fit, int nf, const int *ev, const int *mode0,
                         int nev, int ntok, const ds4_loss_weights *lw, double dscale,
                         uint64_t seed, int nth, int L, FILE *lf, best_t *best,
                         float *Yhat, float *Cb, float *Cp, double lam) {
    const int NE = 256;
    double *sy = xmalloc((size_t)NE * D * sizeof(double));   /* Σ Ys */
    double *sr = xmalloc((size_t)NE * D * sizeof(double));   /* Σ R  */
    double *syy = xmalloc((size_t)NE * D * sizeof(double));  /* Σ Ys² */
    double *syr = xmalloc((size_t)NE * D * sizeof(double));  /* Σ Ys·R */
    long *cnt = xmalloc((size_t)NE * sizeof(long));
    memset(sy, 0, (size_t)NE * D * sizeof(double)); memset(sr, 0, (size_t)NE * D * sizeof(double));
    memset(syy, 0, (size_t)NE * D * sizeof(double)); memset(syr, 0, (size_t)NE * D * sizeof(double));
    memset(cnt, 0, (size_t)NE * sizeof(long));
    for (int i = 0; i < nf; i++) {
        const int t = fit[i], e = top1[t];
        if (e < 0 || e >= NE) continue;
        const float *y = Ys + (size_t)t * D, *r = R + (size_t)t * D;
        double *a = sy + (size_t)e * D, *b2 = sr + (size_t)e * D;
        double *c = syy + (size_t)e * D, *d2 = syr + (size_t)e * D;
        for (int j = 0; j < D; j++) {
            a[j] += y[j]; b2[j] += r[j];
            c[j] += (double)y[j] * y[j]; d2[j] += (double)y[j] * r[j];
        }
        cnt[e]++;
    }
    float *gg = xmalloc((size_t)NE * D * sizeof(float));
    float *bb = xmalloc((size_t)NE * D * sizeof(float));
    int nact = 0;
    for (int e = 0; e < NE; e++) {
        const double n = (double)cnt[e];
        if (n < 8) {   /* 样本太少的组不修(避免拟合噪声), 退回零修正 */
            for (int j = 0; j < D; j++) { gg[(size_t)e * D + j] = 0.0f; bb[(size_t)e * D + j] = 0.0f; }
            continue;
        }
        nact++;
        double *a = sy + (size_t)e * D, *b2 = sr + (size_t)e * D;
        double *c = syy + (size_t)e * D, *d2 = syr + (size_t)e * D;
        /* 组内中心化: cov = Σyr − Σy·Σr/n ; var = Σy² − (Σy)²/n */
        double vr = 0; for (int j = 0; j < D; j++) vr += (c[j] - a[j] * a[j] / n);
        vr = vr / D + 1e-30;
        for (int j = 0; j < D; j++) {
            const double cov = d2[j] - a[j] * b2[j] / n;
            const double var = c[j] - a[j] * a[j] / n;
            const double g = cov / (var + lam * vr);
            gg[(size_t)e * D + j] = (float)g;
            bb[(size_t)e * D + j] = (float)((b2[j] - g * a[j]) / n);
        }
    }
    float *corr = xmalloc((size_t)ntok * D * sizeof(float));
    for (int t = 0; t < ntok; t++) {
        const int e = top1[t];
        float *o = corr + (size_t)t * D;
        if (e < 0 || e >= NE || cnt[e] < 8) { memset(o, 0, (size_t)D * 4); continue; }
        const float *y = Ys + (size_t)t * D;
        const float *g = gg + (size_t)e * D, *b2 = bb + (size_t)e * D;
        for (int j = 0; j < D; j++) o[j] = g[j] * y[j] + b2[j];
    }
    char nm[24]; snprintf(nm, sizeof nm, "cond");
    run_bc_arm(nm, gg, NE * D, corr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
               lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
    printf("    ↑cond λ=%.3g 活跃组=%d/%d vol=%.1fMB(g+b fp16)\n",
           lam, nact, NE, (double)2 * NE * D * 2 / 1e6);
    free(sy); free(sr); free(syy); free(syr); free(cnt);
    free(gg); free(bb); free(corr);
}
