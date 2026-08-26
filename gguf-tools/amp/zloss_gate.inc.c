/* zloss_gate.inc.c — zloss_solve 的门族分片(语义化命名): EM 模式门(x 侧) +
 * 语境条件化动态门(GEc)。只被 zloss_solve.c include, 解算器仍是一份代码。
 *
 * ★语境门 GEc(2026-08-26 段漂移实锤后加针)★ 折半诊断: L41 GEδ fit 内稳定
 * (cos 0.884)但 fit↔held 塌到 0.329, 且漂移深度单调(L20 0.775/L30 0.570)——
 * 每专家增益随语料段漂移, 静态参数抓不住 = 用户动态 z 理论在门参数层的证据。
 * 形态: δ_e(c) = δ0_e + Δ_e·c, c = 块内因果运行均值的低维 PCA 码(m 维)。
 * 基 P 只从 fit 块取(防 held 泄漏, snapshot PCA + sym_eig_topk 复用);
 * 推理侧 c = 当前序列 ffn_in 运行均值的同一投影, 因果可得, 可部署。
 * 解算: 参数 θ[256×(1+m)] 联合闭式 ridge(正规方程 1280², chol_solve_spd)。 */

/* c[ntok×m]: 因果块内运行均值 → fit 块 snapshot PCA 码(每维按特征值标准化) */
static float *ctx_build(const float *X, int ntok, int m, const uint8_t *isfit, int L) {
    const int BS = 256, nb = ntok / BS;
    double *bm = xmalloc((size_t)nb * D * sizeof(double));   /* 全块均值 */
    memset(bm, 0, (size_t)nb * D * sizeof(double));
    for (int b = 0; b < nb; b++) {
        for (int t = b * BS; t < (b + 1) * BS; t++) {
            const float *x = X + (size_t)t * D;
            double *o = bm + (size_t)b * D;
            for (int j = 0; j < D; j++) o[j] += x[j];
        }
        for (int j = 0; j < D; j++) bm[(size_t)b * D + j] /= BS;
    }
    int fb[64], nfb = 0;                       /* fit 块 = 含 fit 行的块 */
    for (int b = 0; b < nb && nfb < 64; b++) {
        int hit = 0;
        for (int t = b * BS; t < (b + 1) * BS; t++) if (isfit[t]) { hit = 1; break; }
        if (hit) fb[nfb++] = b;
    }
    if (m > nfb - 1) m = nfb - 1;
    double *mu = xmalloc(D * sizeof(double));
    memset(mu, 0, D * sizeof(double));
    for (int i = 0; i < nfb; i++)
        for (int j = 0; j < D; j++) mu[j] += bm[(size_t)fb[i] * D + j];
    for (int j = 0; j < D; j++) mu[j] /= nfb;
    double *Gm = xmalloc((size_t)nfb * nfb * sizeof(double));   /* snapshot Gram */
    for (int a = 0; a < nfb; a++)
        for (int b2 = a; b2 < nfb; b2++) {
            double s = 0;
            const double *pa = bm + (size_t)fb[a] * D, *pb = bm + (size_t)fb[b2] * D;
            for (int j = 0; j < D; j++) s += (pa[j] - mu[j]) * (pb[j] - mu[j]);
            Gm[(size_t)a * nfb + b2] = Gm[(size_t)b2 * nfb + a] = s;
        }
    double *ev = xmalloc((size_t)m * nfb * sizeof(double)), *el = xmalloc((size_t)m * sizeof(double));
    sym_eig_topk(Gm, nfb, m, ev, el, 300, 0x5EEDC0DEULL);
    float *P = xmalloc((size_t)m * D * sizeof(float));          /* D 空间基(行=一支) */
    for (int q = 0; q < m; q++) {
        double sc = 1.0 / sqrt(el[q] / nfb + 1e-20);            /* 标准化码尺度 */
        for (int j = 0; j < D; j++) {
            double s = 0;
            for (int a = 0; a < nfb; a++) s += ev[(size_t)q * nfb + a] * (bm[(size_t)fb[a] * D + j] - mu[j]);
            P[(size_t)q * D + j] = (float)(s * sc / (sqrt(el[q]) + 1e-20));
        }
    }
    float *c = xmalloc((size_t)ntok * (size_t)m * sizeof(float));
    double *run = xmalloc(D * sizeof(double));
    for (int b = 0; b < nb; b++) {                              /* 因果运行均值 */
        memset(run, 0, D * sizeof(double));
        for (int t = b * BS; t < (b + 1) * BS; t++) {
            const float *x = X + (size_t)t * D;
            for (int j = 0; j < D; j++) run[j] += x[j];
            double inv = 1.0 / (t - b * BS + 1);
            for (int q = 0; q < m; q++) {
                double s = 0;
                const float *pq = P + (size_t)q * D;
                for (int j = 0; j < D; j++) s += pq[j] * (run[j] * inv - mu[j]);
                c[(size_t)t * m + q] = (float)s;
            }
        }
    }
    printf("  L%d 语境码: fit块=%d m=%d 特征值前%d=[", L, nfb, m, m);
    for (int q = 0; q < m; q++) printf("%s%.3g", q ? " " : "", el[q]);
    printf("]\n");
    free(bm); free(mu); free(Gm); free(ev); free(el); free(P); free(run);
    return c;
}

/* GEc 联合闭式解: θ[e][0..m] — solve_ge 的语境条件化推广(cc=[1,c_t]) */
static void solve_gec(const zpairs *zp, const float *R, const float *c, int m,
                      const uint8_t *isfit, int ntok, double lam, int L, double *th) {
    const int P1 = 1 + m, N = 256 * P1;
    long long *rs = xmalloc((size_t)(ntok + 1) * sizeof(long long));
    memset(rs, 0, (size_t)(ntok + 1) * sizeof(long long));
    for (long long i = 0; i < zp->npair; i++) if (zp->prow[i] < ntok) rs[zp->prow[i] + 1]++;
    for (int t = 0; t < ntok; t++) rs[t + 1] += rs[t];
    long long *ord = xmalloc((size_t)zp->npair * sizeof(long long));
    { long long *cur = xmalloc((size_t)ntok * sizeof(long long));
      memcpy(cur, rs, (size_t)ntok * sizeof(long long));
      for (long long i = 0; i < zp->npair; i++)
          if (zp->prow[i] < ntok) ord[cur[zp->prow[i]]++] = i;
      free(cur); }
    double *A = xmalloc((size_t)N * N * sizeof(double));
    memset(A, 0, (size_t)N * N * sizeof(double));
    memset(th, 0, (size_t)N * sizeof(double));
    double cc[16]; cc[0] = 1.0;
    for (int t = 0; t < ntok; t++) {
        if (!isfit[t]) continue;
        for (int q = 0; q < m; q++) cc[1 + q] = c[(size_t)t * m + q];
        const float *r = R + (size_t)t * D;
        for (long long a = rs[t]; a < rs[t + 1]; a++) {
            long long i = ord[a];
            const float *yi = zp->pyq + (size_t)i * D;
            double wi = zp->pw[i]; int e = zp->pe[i];
            double bi = 0;
            for (int j = 0; j < D; j++) bi += (double)yi[j] * r[j];
            bi *= wi;
            for (int q = 0; q < P1; q++) th[(size_t)e * P1 + q] += bi * cc[q];
            for (long long b2 = a; b2 < rs[t + 1]; b2++) {
                long long k = ord[b2];
                const float *yk = zp->pyq + (size_t)k * D;
                double d = 0;
                for (int j = 0; j < D; j++) d += (double)yi[j] * yk[j];
                d *= wi * zp->pw[k];
                int f = zp->pe[k];
                for (int q1 = 0; q1 < P1; q1++)
                    for (int q2 = 0; q2 < P1; q2++) {
                        double v = d * cc[q1] * cc[q2];
                        A[((size_t)e * P1 + q1) * N + (size_t)f * P1 + q2] += v;
                        if (k != i) A[((size_t)f * P1 + q2) * N + (size_t)e * P1 + q1] += v;
                    }
            }
        }
    }
    double md = 0; for (int i = 0; i < N; i++) md += A[(size_t)i * N + i];
    md = md / N + 1e-30;
    for (int i = 0; i < N; i++) A[(size_t)i * N + i] += lam * md;
    if (chol_solve_spd(A, N, th, 1) != 0) {
        for (int i = 0; i < N; i++) A[(size_t)i * N + i] += 100 * lam * md;
        if (chol_solve_spd(A, N, th, 1) != 0) die("L%d GEc 正规方程非正定", L);
    }
    free(rs); free(ord); free(A);
}

static float *gec_corr_build(const zpairs *zp, const double *th, const float *c, int m,
                             int ntok) {
    const int P1 = 1 + m;
    float *corr = xmalloc((size_t)ntok * D * sizeof(float));
    memset(corr, 0, (size_t)ntok * D * sizeof(float));
    for (long long i = 0; i < zp->npair; i++) {
        int t = zp->prow[i]; if (t >= ntok) continue;
        double g = th[(size_t)zp->pe[i] * P1];
        for (int q = 0; q < m; q++) g += th[(size_t)zp->pe[i] * P1 + 1 + q] * c[(size_t)t * m + q];
        g *= zp->pw[i];
        if (fabs(g) < 1e-12) continue;
        float *o = corr + (size_t)t * D;
        const float *y = zp->pyq + (size_t)i * D;
        for (int j = 0; j < D; j++) o[j] += (float)(g * y[j]);
    }
    return corr;
}

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
                         const int *top1, float *C, int *mode_fit,
                         int *table, int *gate_route) {
    float *Xf = xmalloc((size_t)nf * D * sizeof(float));
    float *Rf = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        memcpy(Xf + (size_t)i * D, X + (size_t)fit[i] * D, D * sizeof(float));
        memcpy(Rf + (size_t)i * D, R + (size_t)fit[i] * D, D * sizeof(float));
    }
    /* 单重启+4 迭代(2026-08-26 提速定案: L20/L30 双重启 6 次实测全部收敛同解,
     * 第二重启纯烧机; 迭代 4 轮后标签已稳) */
    const int EMK = maxk < 32 ? maxk : 32, T = 4, RS = 1;
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
    /* 门候选①: x 方向质心(EM 标签的 x 侧线性读出) */
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
    int agree_x = 0;
    for (int i = 0; i < nf; i++)
        if (mode_assign(Xf + (size_t)i * D, C, M) == labbest[i]) agree_x++;
    /* 门候选②: 路由 top-1 专家查表(R=Σ所选专家误差图 ⇒ 模式天然路由承载;
     * 部署时路由先于专家计算, 免费信号)。空表专家回落全局多数标签。 */
    int agree_r = -1;
    if (top1 && table) {
        int *vt = xmalloc((size_t)256 * M * sizeof(int));
        memset(vt, 0, (size_t)256 * M * sizeof(int));
        int gcnt[MAXM]; memset(gcnt, 0, sizeof gcnt);
        for (int i = 0; i < nf; i++) { vt[(size_t)top1[fit[i]] * M + labbest[i]]++; gcnt[labbest[i]]++; }
        int gmaj = 0;
        for (int m = 1; m < M; m++) if (gcnt[m] > gcnt[gmaj]) gmaj = m;
        for (int e = 0; e < 256; e++) {
            int bm = -1, bc = 0;
            for (int m = 0; m < M; m++) if (vt[(size_t)e * M + m] > bc) { bc = vt[(size_t)e * M + m]; bm = m; }
            table[e] = bm < 0 ? gmaj : bm;
        }
        agree_r = 0;
        for (int i = 0; i < nf; i++) if (table[top1[fit[i]]] == labbest[i]) agree_r++;
        free(vt);
    }
    *gate_route = (agree_r > agree_x);
    for (int i = 0; i < nf; i++)
        mode_fit[i] = *gate_route ? table[top1[fit[i]]] : mode_assign(Xf + (size_t)i * D, C, M);
    printf("  L%d M=%d 门一致率 x=%.1f%% 路由=%.1f%% → 选%s门\n", L, M,
           100.0 * agree_x / nf, agree_r < 0 ? -1.0 : 100.0 * agree_r / nf,
           *gate_route ? "路由" : "x");
    free(Xf); free(Rf); free(lab); free(labbest); free(tmp); free(xacc);
}
