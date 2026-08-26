/* zloss_gate.inc.c — 动态 z 的模式发现与 x 侧门(物理分片, 只被 zloss_solve.c include)。
 *
 * 模式 = 混合线性回归的硬 EM: 行标签按"哪张图预测得更好"迭代(T=6), 定死种子
 * + 双重启取 fit 总误差更小者, 全确定性。门 = EM 标签的 x 方向质心
 * (apply 时只有 x); 训练行按门重分配, 与部署 apply 完全同式(门错训练时就吃进去)。
 *
 * 为什么不是无监督聚类(v2 两次翻案固化): 模式藏在 x↔R 的联合关系里 ——
 * x 密度聚类与残差方向聚类都只看边缘分布, 对称分布下任意对径切分密度等价,
 * selftest 金标(种 sign(p·h) 双图)下两者门一致率都 ≈50% 掷硬币。 */
static void derive_modes(const float *X, const float *R, const int *fit, int nf,
                         int M, int maxk, double lam0, uint64_t seed, int L,
                         float *C, int *mode_fit) {
    float *Xf = xmalloc((size_t)nf * D * sizeof(float));
    float *Rf = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        memcpy(Xf + (size_t)i * D, X + (size_t)fit[i] * D, D * sizeof(float));
        memcpy(Rf + (size_t)i * D, R + (size_t)fit[i] * D, D * sizeof(float));
    }
    const int EMK = maxk < 32 ? maxk : 32, T = 6, RS = 2;
    int *lab = xmalloc((size_t)nf * sizeof(int));
    int *labbest = xmalloc((size_t)nf * sizeof(int));
    float *tmp = xmalloc(D * sizeof(float));
    double errbest = 1e300;
    for (int rs = 0; rs < RS; rs++) {
        uint64_t s = (seed + 1 + (uint64_t)rs) * 6364136223846793005ULL + 1442695040888963407ULL;
        for (int i = 0; i < nf; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            lab[i] = (int)((s >> 33) % (uint64_t)M);
        }
        double toterr = 1e300;
        for (int it = 0; it < T; it++) {
            ds4_z *wm[MAXM] = {0};
            for (int m = 0; m < M; m++) {
                int c = 0;
                for (int i = 0; i < nf; i++) if (lab[i] == m) c++;
                if (!c) continue;                     /* 空模式=零图参赛, 行可迁回 */
                float *Xm = xmalloc((size_t)c * D * sizeof(float));
                float *Rm = xmalloc((size_t)c * D * sizeof(float));
                int w = 0;
                for (int i = 0; i < nf; i++) if (lab[i] == m) {
                    memcpy(Xm + (size_t)w * D, Xf + (size_t)i * D, D * sizeof(float));
                    memcpy(Rm + (size_t)w * D, Rf + (size_t)i * D, D * sizeof(float));
                    w++;
                }
                wm[m] = ds4_z_solve(Xm, Rm, (uint32_t)c, D, D,
                                    (uint32_t)(EMK < c ? EMK : c), (float)lam0);
                free(Xm); free(Rm);
                if (!wm[m]) die("L%d M=%d EM 模式%d 解算失败", L, M, m);
            }
            toterr = 0;
            for (int i = 0; i < nf; i++) {
                const float *x = Xf + (size_t)i * D, *r = Rf + (size_t)i * D;
                double best = 1e300; int bm = lab[i];
                for (int m = 0; m < M; m++) {
                    double e = 0;
                    if (!wm[m]) { for (int j = 0; j < D; j++) e += (double)r[j] * r[j]; }
                    else {
                        memset(tmp, 0, D * sizeof(float));
                        ds4_z_apply(wm[m], x, tmp);
                        for (int j = 0; j < D; j++) { double d = (double)r[j] - tmp[j]; e += d * d; }
                    }
                    if (e < best) { best = e; bm = m; }
                }
                lab[i] = bm; toterr += best;
            }
            for (int m = 0; m < M; m++) if (wm[m]) ds4_z_free(wm[m]);
        }
        printf("  L%d M=%d EM 重启%d fit误差 %.4g\n", L, M, rs, toterr);
        if (toterr < errbest) { errbest = toterr; memcpy(labbest, lab, (size_t)nf * sizeof(int)); }
    }
    double *xacc = xmalloc((size_t)M * D * sizeof(double));
    memset(xacc, 0, (size_t)M * D * sizeof(double));
    for (int i = 0; i < nf; i++) {
        const float *x = Xf + (size_t)i * D;
        double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
        double inv = ss > 0 ? 1.0 / sqrt(ss) : 0.0;
        double *a = xacc + (size_t)labbest[i] * D;
        for (int j = 0; j < D; j++) a[j] += x[j] * inv;
    }
    for (int m = 0; m < M; m++) {
        double ss = 0; const double *a = xacc + (size_t)m * D;
        for (int j = 0; j < D; j++) ss += a[j] * a[j];
        double inv = ss > 0 ? 1.0 / sqrt(ss) : 0.0;
        float *c = C + (size_t)m * D;
        for (int j = 0; j < D; j++) c[j] = (float)(a[j] * inv);
    }
    int agree = 0;
    for (int i = 0; i < nf; i++) {
        mode_fit[i] = mode_assign(Xf + (size_t)i * D, C, M);
        if (mode_fit[i] == labbest[i]) agree++;
    }
    printf("  L%d M=%d 门一致率(x门 vs EM标签) %.1f%%\n", L, M, 100.0 * agree / nf);
    free(Xf); free(Rf); free(lab); free(labbest); free(tmp); free(xacc);
}
