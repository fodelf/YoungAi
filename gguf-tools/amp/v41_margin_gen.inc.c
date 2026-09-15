/* v41_margin_gen.inc.c — 后训练第三件的【泛化】侧: 留一尺(尺 L) + 两针诊断(2026-09-14)。
 * v41_amp_run.c 单 TU include; 解算本体在 v41_margin_solve.inc.c(解算器只许有一份, 这里只是换着法子调它)。
 *
 * 【为什么要这个文件 —— 09-13/14 两轮实测逼出来的】
 * 第二版把"决策点翻不翻"变成线性方程之后, 训练日撬得动了(3 条样本 55%→87.67% 真前向; 8 条样本
 * 预测 57.26%→76.92%), 可是★判决点一个没翻★: 12 个候选、四档 λ、两种形态, val 全是 55.26%,
 * 一位小数都不差。这不是 bug, 是欠定最小二乘的必然 —— 解恒落在训练方程张成的子空间 span{A_i} 里,
 * 对没见过的决策点 j, Δm_j = Σ_i β_i·<A_j, A_i>; A_j 与训练集近正交 ⇒ 恒为 0。
 *
 * 于是第三版做三件事, 全在这个文件里:
 *   ① 尺 L(留一): 把"举一反三"变成★秒级能算、能进网格优化★的数。以前选候选按"训练拟合率",
 *      那是专门挑最会背题的那个(实测: 拟合最高的 λ=0.3 与拟合最低的 λ=10, val 一模一样)。
 *   ② 诊断 A: 把上面那个"近正交"直接量出来 —— 折外决策方程在折内子空间里的★投影占比★。
 *      它 ≈0 ⇒ 无论加多少天样本、λ 怎么调, 这个形态都不会泛化, 该去换形态而不是继续调参。
 *   ③ 诊断 B: 教训重不重复(专家重合率 / 词对交集)。判决点若全是一次性的具体数字, 它本来就不该
 *      被"学会", 尺 B 的天花板不是 100%。
 * 留一必须★按样本条分组★: 同一份报告里相邻决策点共享上下文与路由, 按点随机分折 = 自己给自己泄题。 */

/* (1) 词对重复度: 同一对 (该赢的 token, 对手 token) 在几条不同样本里出现过。
 * 为什么它是泛化的抓手: 方向表 C = α·inv·γ⊙(W[a]−W[b]) 完全由这一对词决定 —— 同一对词在不同
 * 日子里 C ★完全相同★, 沿它的修正天然跨样本可复用。"上调↔下调"这种教训会重复, "目标价 3.72"
 * 这种不会, 后者学了就是背题。权重 ×(1+ν·(rep−1)) 让重复的教训说话更响。 */
static void mg_pair_rep(mg_acc *ab, int ndec_fit) {
    int multi = 0;
    for (int i = 0; i < ndec_fit; i++) {
        int seen[64], ns = 0;                       /* 这一对词出现过的样本号(去重); 样本数远小于 64 */
        for (int j = 0; j < ndec_fit; j++) {
            if (ab->ta[j] != ab->ta[i] || ab->tb[j] != ab->tb[i]) continue;
            int dup = 0;
            for (int q = 0; q < ns; q++) if (seen[q] == ab->smp[j]) { dup = 1; break; }
            if (!dup && ns < 64) seen[ns++] = ab->smp[j];
        }
        ab->repw[i] = (float)ns;
        if (ns > 1) multi++;
    }
    printf("[后训练·词对] 决策点 %d 个, 其中 %d 个的 (该赢, 对手) 词对★跨条重复★(%.1f%%)\n",
           ndec_fit, multi, ndec_fit ? 100.0 * multi / ndec_fit : 0.0);
}

/* (2) 专家频率岭 w_e = (f_max+κ)/(f_e+κ), 再整体归一到均值 1。
 * ★归一这一步不是洁癖★: 不归一的话 κ 一变, 岭的整体强度跟着变, 于是"κ 的效果"和"λ 的效果"
 * 混在一起, 网格上读出来的东西没法解释。归一后 κ 只改【专家之间的相对贵贱】。
 * κ→∞ 退化成全 1(= 第二版行为), 所以 κ 网格里留一个大值就能自带对照组。 */
static float *mg_freq_we(ctx_t *c, mg_acc *ab, int ndec_fit, float kappa) {
    const int ne = c->n_expert, nu = ab->nu;
    int *f = calloc((size_t)ne, sizeof(int));
    float *we = malloc((size_t)ne * 4);
    if (!f || !we) { free(f); free(we); fprintf(stderr, "★频率岭缓冲失败★\n"); return NULL; }
    for (int i = 0; i < ndec_fit; i++)
        for (int k = 0; k < nu; k++) { const int e = ab->sel[(size_t)i * nu + k]; if (e >= 0 && e < ne) f[e]++; }
    int fmax = 0, nz = 0;
    for (int e = 0; e < ne; e++) { if (f[e] > fmax) fmax = f[e]; if (f[e]) nz++; }
    double sum = 0.0;
    for (int e = 0; e < ne; e++) { we[e] = (fmax + kappa) / (f[e] + kappa); sum += we[e]; }
    const float mean = (float)(sum / ne);
    for (int e = 0; e < ne; e++) we[e] /= mean;
    float *dwe = NULL;
    if (cudaMalloc((void **)&dwe, (size_t)ne * 4) || cudaMemcpy(dwe, we, (size_t)ne * 4, cudaMemcpyHostToDevice)) {
        fprintf(stderr, "★频率岭上传失败★\n"); free(f); free(we); return NULL; }
    printf("[后训练·频率岭 κ=%g] 决策行用到 %d/%d 个专家, 最热 %d 次; 岭权重范围 [%.3f, %.3f](均值 1)\n",
           kappa, nz, ne, fmax, we[0] < we[ne - 1] ? we[0] : we[ne - 1], (fmax + kappa) / (0 + kappa) / mean);
    free(f); free(we);
    return dwe;
}

/* (2.5) ★x 键控的门控方向★(back.md 段 4; 09-14 三针诊断逼出来的)。
 * gate[i][0] ≡ 1(恒等门 ⇒ R=1 严格退化成旧形态, 新旧可比); gate[i][r≥1] = σ((q_r·x_i − μ)/s),
 * q_r = 决策行隐状态 x 的第 r 个主成分(幂迭代 + 逐次去相关, 决策点才是我们要分情况对待的地方)。
 * 为什么用 x 的主成分当门: 我们要的是"在什么情况下修" —— x 就是这一层看到的全部上下文,
 * 它的主方向是这批决策点之间差别最大的方向, 也就是最有分辨力的"情况"。
 * 标准化到零均值单位方差再过 σ, 是为了让门真的在 [0.27, 0.73] 这段有区分度(不标准化时
 * q·x 的量级动辄上百, σ 全饱和成 0/1, 门就退化成硬开关, 解出来全是记忆)。 */
static float *mg_gates(ctx_t *c, mg_acc *ab, int ndec_fit, int R, float **hgate_out) {
    const int ns = ab->n, D = ab->D;
    float *hg = malloc((size_t)ns * R * 4);
    float *q = calloc((size_t)(R > 1 ? R - 1 : 1) * D, 4);
    double *xm = calloc((size_t)D, sizeof(double)), *v = malloc((size_t)D * sizeof(double)),
           *w = malloc((size_t)D * sizeof(double));
    if (!hg || !q || !xm || !v || !w) { free(hg); free(q); free(xm); free(v); free(w); return NULL; }
    for (int i = 0; i < ns * R; i++) hg[i] = 0.f;
    for (int i = 0; i < ns; i++) hg[(size_t)i * R] = 1.0f;            /* r=0: 恒等门 */
    if (R > 1) {
        for (int i = 0; i < ndec_fit; i++) for (int d = 0; d < D; d++) xm[d] += ab->xh[(size_t)i * D + d];
        for (int d = 0; d < D; d++) xm[d] /= (ndec_fit > 0 ? ndec_fit : 1);
        for (int r = 0; r < R - 1; r++) {
            for (int d = 0; d < D; d++) v[d] = sin((double)(d * 7 + r * 13) * 0.001) ;   /* 确定性起点: 同一份料解两次必须同结果 */
            for (int it = 0; it < 30; it++) {
                for (int d = 0; d < D; d++) w[d] = 0.0;
                for (int i = 0; i < ndec_fit; i++) {   /* w += (xᵀv)·x, x 已中心化 */
                    double dot = 0.0;
                    for (int d = 0; d < D; d++) dot += ((double)ab->xh[(size_t)i * D + d] - xm[d]) * v[d];
                    for (int d = 0; d < D; d++) w[d] += dot * ((double)ab->xh[(size_t)i * D + d] - xm[d]);
                }
                for (int pr = 0; pr < r; pr++) {       /* 去掉前面已取的主成分 */
                    double dot = 0.0;
                    for (int d = 0; d < D; d++) dot += w[d] * q[(size_t)pr * D + d];
                    for (int d = 0; d < D; d++) w[d] -= dot * q[(size_t)pr * D + d];
                }
                double nr = 0.0;
                for (int d = 0; d < D; d++) nr += w[d] * w[d];
                nr = sqrt(nr);
                if (!(nr > 0.0)) break;
                for (int d = 0; d < D; d++) v[d] = w[d] / nr;
            }
            for (int d = 0; d < D; d++) q[(size_t)r * D + d] = (float)v[d];
            /* 投影 → 用决策行的均值方差标准化 → σ。约束行用同一套 μ/s(它们不是"情况"的定义者) */
            double mu = 0.0, s2 = 0.0;
            float *proj = malloc((size_t)ns * 4);
            if (!proj) break;
            for (int i = 0; i < ns; i++) {
                double t = 0.0;
                for (int d = 0; d < D; d++) t += ((double)ab->xh[(size_t)i * D + d] - xm[d]) * v[d];
                proj[i] = (float)t;
            }
            for (int i = 0; i < ndec_fit; i++) mu += proj[i];
            mu /= (ndec_fit > 0 ? ndec_fit : 1);
            for (int i = 0; i < ndec_fit; i++) s2 += (proj[i] - mu) * (proj[i] - mu);
            const double sd = sqrt(s2 / (ndec_fit > 1 ? ndec_fit - 1 : 1)) + 1e-9;
            double gmin = 1e30, gmax = -1e30;
            for (int i = 0; i < ns; i++) {
                const double t = (proj[i] - mu) / sd;
                const float g = (float)(1.0 / (1.0 + exp(-t)));
                hg[(size_t)i * R + r + 1] = g;
                if (g < gmin) gmin = g;
                if (g > gmax) gmax = g;
            }
            printf("[后训练·门控] 第 %d 门: 决策行 σ 范围 [%.3f, %.3f]\n", r + 1, gmin, gmax);
            free(proj);
        }
    }
    float *dg = NULL;
    if (cudaMalloc((void **)&dg, (size_t)ns * R * 4) ||
        cudaMemcpy(dg, hg, (size_t)ns * R * 4, cudaMemcpyHostToDevice)) {
        fprintf(stderr, "★门控上传失败★\n"); free(hg); free(q); free(xm); free(v); free(w); return NULL; }
    free(q); free(xm); free(v); free(w);
    if (hgate_out) *hgate_out = hg; else free(hg);
    (void)c;
    return dg;
}

/* (3) 尺 L: 按样本条留一。每条样本轮流被抽出方程之外(权重 0), 解完只看★它自己★的决策点翻没翻。
 * 折数 = 样本条数; 每折一次 CG(秒级), 不落盘但照做 fp4 往返 —— 尺 L 与 gate 必须量同一个态。
 * 返回 0 成功; stL[0] = 折外翻转率(%), stL[1] = 折外点数, stL[2] = 总耗时(s)。 */
static int mg_loo(ctx_t *c, mg_acc *ab, mg_pairs *pp, int il, const char *out, mg_cfg *cf,
                  int ndec_fit, int ndec, int nsmp, float *dC, float *dm, const float *sprev,
                  float *dY, float *dLg, int *dIds, float *hLg, const unsigned char *base_win, float *stL) {
    const double t0 = now_s();
    int ok = 0, tot = 0, broke = 0, broke_tr = 0;
    const int keep_hold = cf->hold_smp, keep_w = cf->no_write;
    cf->no_write = 1;
    for (int m = 0; m < nsmp; m++) {
        int has = 0;
        for (int i = 0; i < ndec_fit && !has; i++) has = (ab->smp[i] == m);
        if (!has) continue;                       /* 这条样本一个可用决策点都没有: 跳过, 不算进分母 */
        float st[16] = { 0 };
        cf->hold_smp = m;
        if (mg_candidate(c, ab, pp, il, out, cf, ndec_fit, ndec, dC, dm, sprev, dY, dLg, dIds, hLg, base_win, st)) {
            fprintf(stderr, "★留一第 %d 折解算失败★\n", m); cf->hold_smp = keep_hold; cf->no_write = keep_w; return -1; }
        ok += (int)st[8]; tot += (int)st[9]; broke += (int)st[10]; broke_tr += (int)st[11];
    }
    cf->hold_smp = keep_hold; cf->no_write = keep_w;
    stL[0] = tot ? 100.f * ok / tot : 0.f;
    stL[1] = (float)tot;
    stL[2] = (float)(now_s() - t0);
    stL[3] = (float)broke; stL[4] = (float)broke_tr;
    return 0;
}

/* 主机侧 Cholesky: A(n×n, 行主) 原地分解成下三角 L。返回 0 成功, <0 = 不正定(加岭还不正定就是有 NaN)。 */
static int mg_chol(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; k++) s -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
            if (i == j) { if (!(s > 0.0)) return -1; A[(size_t)i * n + i] = sqrt(s); }
            else A[(size_t)i * n + j] = s / A[(size_t)j * n + j];
        }
        for (int j = i + 1; j < n; j++) A[(size_t)i * n + j] = 0.0;
    }
    return 0;
}
static void mg_chol_solve(const double *L, int n, const double *b, double *x) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[(size_t)i * n + k] * x[k];
        x[i] = s / L[(size_t)i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = x[i];
        for (int k = i + 1; k < n; k++) s -= L[(size_t)k * n + i] * x[k];
        x[i] = s / L[(size_t)i * n + i];
    }
}

/* (4) 诊断 A/B: 折外决策方程在折内子空间里的投影占比 + 专家重合 + 词对交集。
 * ★投影占比怎么读★: p_j = ‖P_span(折内 A) A_j‖² / ‖A_j‖²。解是最小范数解 ⇒ 它只能沿折内子空间动,
 * 所以 p_j 就是"这个判决点最多能被撬动多少"的上限刻度。p 全体接近 0 ⇒ 形态无泛化能力(换形态);
 * p 有相当一部分不小而尺 L 仍是 0 ⇒ 方向有、幅度不够(调 τ/ρ/λ 或加样本有戏)。 */
static int mg_diag(ctx_t *c, mg_acc *ab, int ndec_fit, int nsmp, const char *out,
                   float *dC, int *dsrc_buf) {
    const int nu = ab->nu, D = ab->D, ne = c->n_expert;
    const int np = ndec_fit;
    if (np < 4) { fprintf(stderr, "★诊断: 决策点只有 %d 个, 不做★\n", np); return 0; }
    /* 决策行的方向表与 src(每行一对, 与候选首轮同一套) */
    float *al = malloc((size_t)np * 4), *iv = malloc((size_t)np * 4);
    int *src = malloc((size_t)np * 4);
    float *G = malloc((size_t)np * np * 4);
    float *dG = NULL;
    if (!al || !iv || !src || !G || cudaMalloc((void **)&dG, (size_t)np * np * 4)) {
        fprintf(stderr, "★诊断缓冲失败★\n"); free(al); free(iv); free(src); free(G); return -1; }
    for (int i = 0; i < np; i++) { al[i] = ab->alpha[i]; iv[i] = ab->inv[i]; src[i] = i; }
    int rc = -1;
    if (!v41_klt_margin_dirs(c->klt, ab->ta, ab->tb, al, iv, np, D, dC) &&
        !cudaMemcpy(dsrc_buf, src, (size_t)np * 4, cudaMemcpyHostToDevice) &&
        !v41_gr_margin_gram(c->dye, c->drw, c->dsel, dC, dsrc_buf, np, nu, D, dG) &&
        !cudaMemcpy(G, dG, (size_t)np * np * 4, cudaMemcpyDeviceToHost)) rc = 0;
    cudaFree(dG); free(al); free(iv); free(src);
    if (rc) { free(G); fprintf(stderr, "★诊断 Gram 失败★\n"); return -1; }

    /* ★c 方向之间的 cos★: A_i = (专家侧的 ye) ⊗ c_i, 两个因子谁在正交是两条不同的路 ——
     * c 正交 ⇒ 连"固定方向只调专家权重"(F3)都没有共同方向可固定, 该换形态;
     * c 不正交而 A 正交 ⇒ 正交来自 ye, 那么把解限制在"共同 c 方向"上就是对症的先验。 */
    {
        float *hC = malloc((size_t)np * D * 4);
        if (hC && !cudaMemcpy(hC, dC, (size_t)np * D * 4, cudaMemcpyDeviceToHost)) {
            double *nrm = malloc((size_t)np * sizeof(double));
            double sum = 0.0, amax = -1.0; int cnt = 0, npos = 0;
            for (int i = 0; i < np; i++) {
                double e = 0.0;
                for (int d = 0; d < D; d++) e += (double)hC[(size_t)i * D + d] * hC[(size_t)i * D + d];
                nrm[i] = sqrt(e);
            }
            for (int i = 0; i < np; i++) for (int j = i + 1; j < np; j++) {
                if (!(nrm[i] > 0.0) || !(nrm[j] > 0.0)) continue;
                double dp2 = 0.0;
                for (int d = 0; d < D; d++) dp2 += (double)hC[(size_t)i * D + d] * hC[(size_t)j * D + d];
                const double cs = dp2 / (nrm[i] * nrm[j]);
                sum += cs; cnt++; if (cs > 0.0) npos++;
                if (fabs(cs) > amax) amax = fabs(cs);
            }
            if (cnt) printf("[诊断·c 方向] 决策点两两 cos: 均 %+.4f | |cos| 最大 %.4f | 同号比例 %.1f%%\n"
                            "  读法: 均值远离 0 ⇒ 存在共同方向(F3 有戏); 贴着 0 且最大也小 ⇒ 词对方向本身互不相干\n",
                            sum / cnt, amax, 100.0 * npos / cnt);
            free(nrm);
        }
        free(hC);
    }

    char dp[4300]; snprintf(dp, sizeof dp, "%s/diag.txt", out);
    FILE *df = fopen(dp, "w");
    if (df) fprintf(df, "# 留一诊断(back.md §4.8): 每条样本轮流当折外\n"
                        "# 样本 折外点数 投影占比均值 投影占比中位 专家重合%% 词对交集%%\n");
    double sum_p = 0.0, sum_e = 0.0, sum_w = 0.0; int cnt_p = 0, cnt_e = 0, cnt_w = 0;
    double *K = malloc((size_t)np * np * sizeof(double)), *b = malloc((size_t)np * sizeof(double)),
           *x = malloc((size_t)np * sizeof(double));
    int *idx = malloc((size_t)np * 4);
    double *pj = malloc((size_t)np * sizeof(double));
    if (!K || !b || !x || !idx || !pj) { free(G); free(K); free(b); free(x); free(idx); free(pj); if (df) fclose(df); return -1; }
    for (int m = 0; m < nsmp; m++) {
        int nin = 0, nh = 0;
        for (int i = 0; i < np; i++) if (ab->smp[i] != m) idx[nin++] = i;
        for (int i = 0; i < np; i++) if (ab->smp[i] == m) nh++;
        if (!nh || nin < 2) continue;
        /* 折内 Gram + 岭(相对对角均值 1e-6): 方程之间常有近重复, 不加岭 Cholesky 会掉 */
        double dm_avg = 0.0;
        for (int q = 0; q < nin; q++) dm_avg += G[(size_t)idx[q] * np + idx[q]];
        dm_avg /= nin;
        for (int q = 0; q < nin; q++) for (int r = 0; r < nin; r++)
            K[(size_t)q * nin + r] = (double)G[(size_t)idx[q] * np + idx[r]] + (q == r ? 1e-6 * dm_avg : 0.0);
        if (mg_chol(K, nin)) { fprintf(stderr, "★第 %d 折 Gram 不正定, 跳过★\n", m); continue; }
        double sp = 0.0, se = 0.0, sw = 0.0; int nps = 0;
        for (int i = 0; i < np; i++) {
            if (ab->smp[i] != m) continue;
            for (int q = 0; q < nin; q++) b[q] = G[(size_t)idx[q] * np + i];
            mg_chol_solve(K, nin, b, x);
            double num = 0.0;
            for (int q = 0; q < nin; q++) num += b[q] * x[q];
            const double den = G[(size_t)i * np + i];
            const double p = den > 0.0 ? num / den : 0.0;
            pj[nps++] = p; sp += p;
            /* 专家重合: 这一行的 6 个专家里, 有几个在折内决策行里出现过 */
            int hit = 0, tot = 0;
            for (int k = 0; k < nu; k++) {
                const int e = ab->sel[(size_t)i * nu + k];
                if (e < 0 || e >= ne) continue;
                tot++;
                for (int q = 0; q < nin; q++) {
                    int f = 0;
                    for (int k2 = 0; k2 < nu; k2++) if (ab->sel[(size_t)idx[q] * nu + k2] == e) { f = 1; break; }
                    if (f) { hit++; break; }
                }
            }
            se += tot ? (double)hit / tot : 0.0;
            /* 词对交集: 这一行的 (该赢, 对手) 对在折内出现过没有 */
            int wf = 0;
            for (int q = 0; q < nin; q++)
                if (ab->ta[idx[q]] == ab->ta[i] && ab->tb[idx[q]] == ab->tb[i]) { wf = 1; break; }
            sw += wf;
        }
        if (!nps) continue;
        for (int a = 1; a < nps; a++) { const double v = pj[a]; int q = a - 1; while (q >= 0 && pj[q] > v) { pj[q + 1] = pj[q]; q--; } pj[q + 1] = v; }
        const double med = pj[nps / 2];
        sum_p += sp; cnt_p += nps; sum_e += se; cnt_e += nps; sum_w += sw; cnt_w += nps;
        printf("  [诊断 折外=样本%d] %d 点: 投影占比 均 %.4f 中位 %.4f | 专家重合 %.1f%% | 词对交集 %.1f%%\n",
               m, nps, sp / nps, med, 100.0 * se / nps, 100.0 * sw / nps);
        if (df) fprintf(df, "%d %d %.6f %.6f %.2f %.2f\n", m, nps, sp / nps, med, 100.0 * se / nps, 100.0 * sw / nps);
    }
    if (cnt_p) {
        printf("★诊断汇总★ 折外 %d 点: ★投影占比均 %.4f★ | 专家重合 %.1f%% | 词对交集 %.1f%%\n"
               "  读法: 投影占比 = 折外决策方程能被折内子空间表达的比例, 它就是泛化上限的刻度;\n"
               "        ≈0 ⇒ 加样本/调 λ 都白费(要换形态), 明显 >0 而尺 L 仍 0 ⇒ 方向有、幅度不够。\n",
               cnt_p, sum_p / cnt_p, 100.0 * sum_e / cnt_e, 100.0 * sum_w / cnt_w);
        if (df) fprintf(df, "ALL %d %.6f - %.2f %.2f\n", cnt_p, sum_p / cnt_p, 100.0 * sum_e / cnt_e, 100.0 * sum_w / cnt_w);
    }
    if (df) fclose(df);
    free(G); free(K); free(b); free(x); free(idx); free(pj);
    return 0;
}
