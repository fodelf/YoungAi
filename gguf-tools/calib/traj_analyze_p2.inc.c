/* ------------------------------------------------------------------ */
/* PART 3 — cross-layer path structure                                 */
/* ------------------------------------------------------------------ */

/* mean per-token Jaccard of top-6 sets and primary(col0)-agreement fraction,
 * between layers La and Lb. If perm != NULL, Lb is indexed through perm[t]
 * (token-shuffled baseline that destroys per-token correlation). */
static void pair_stats(int La, int Lb, const int *perm,
                       double *mean_jac, double *prim_agree) {
    double sj = 0;
    long agree = 0;
    for (int t = 0; t < NTOK; t++) {
        int tb = perm ? perm[t] : t;
        const uint8_t *a = row_sort(La, t);
        const uint8_t *b = row_sort(Lb, tb);
        int inter = inter6(a, b);
        int uni = 2 * TOPK - inter;
        sj += (double)inter / (double)uni;
        if (row_orig(La, t)[0] == row_orig(Lb, tb)[0]) agree++;
    }
    *mean_jac = sj / NTOK;
    *prim_agree = (double)agree / NTOK;
}

/* chance primary-agreement = sum_e Pa(e)Pb(e) from the two primary histograms */
static double chance_primary_agree(int La, int Lb) {
    double pa[NEXPERT] = {0}, pb[NEXPERT] = {0};
    for (int t = 0; t < NTOK; t++) {
        pa[row_orig(La, t)[0]] += 1.0;
        pb[row_orig(Lb, t)[0]] += 1.0;
    }
    double s = 0;
    for (int e = 0; e < NEXPERT; e++) s += (pa[e] / NTOK) * (pb[e] / NTOK);
    return s;
}

static void part3(void) {
    printf("================================================================================\n");
    printf("PART 3  Cross-layer routing path structure\n");
    printf("  Jaccard = mean over tokens of |A&B|/|A|B| of the two layers' top-6 sets.\n");
    printf("  primAgree = fraction of tokens whose PRIMARY (top-1) expert id matches.\n");
    printf("  chance = expected primAgree if the two primaries were independent.\n");
    printf("--------------------------------------------------------------------------------\n");

    /* (a) specific pairs */
    struct { int a, b; const char *tag; } pairs[] = {
        {0, 1, "adj-shallow"},
        {20, 21, "adj-mid"},
        {0, 42, "far(0-42)"},
    };
    printf("  pair         layers       Jaccard  primAgree   chance   pA/chance\n");
    printf("  -----------  -----------  -------  ---------  -------  ----------\n");
    for (int i = 0; i < 3; i++) {
        double jac, agr;
        pair_stats(pairs[i].a, pairs[i].b, NULL, &jac, &agr);
        double ch = chance_primary_agree(pairs[i].a, pairs[i].b);
        printf("  %-11s  L%-2d <-> L%-2d  %7.4f  %9.4f  %7.4f  %8.1fx\n",
               pairs[i].tag, pairs[i].a, pairs[i].b, jac, agr, ch,
               ch > 0 ? agr / ch : 0.0);
        if (i == 0) { g_l0l1_agree = agr; g_l0l1_chance = ch; }
    }

    /* (b) shuffled baseline (token-permuted L0<->L1) + analytic random */
    int *perm = (int *)malloc((size_t)NTOK * sizeof(int));
    for (int t = 0; t < NTOK; t++) perm[t] = t;
    uint64_t st = 0xBADC0FFEEULL;
    for (int t = NTOK - 1; t > 0; t--) {        /* Fisher-Yates */
        int j = (int)(splitmix64(&st) % (uint64_t)(t + 1));
        int tmp = perm[t]; perm[t] = perm[j]; perm[j] = tmp;
    }
    double sjac, sagr;
    pair_stats(0, 1, perm, &sjac, &sagr);
    free(perm);
    /* analytic: two independent uniform random 6-subsets of 256 */
    double rnd_inter = (double)(TOPK * TOPK) / (double)NEXPERT;     /* E|A&B| */
    double rnd_jac = rnd_inter / (2.0 * TOPK - rnd_inter);
    g_rnd_jac = rnd_jac;
    printf("  %-11s  L0  <-> L1   %7.4f  %9.4f  %7s  %-10s\n",
           "shuffled", sjac, sagr, "-", "(perm)");
    printf("  %-11s  %-11s  %7.4f  %9.4f  %7s  %-10s\n",
           "analytic", "any random", rnd_jac, 1.0 / NEXPERT, "-", "(indep)");

    /* (c) Jaccard vs layer separation (locality decay) */
    printf("\n  Jaccard / primAgree vs layer separation (avg over all valid pairs):\n");
    printf("    dL |  #pairs |  Jaccard   primAgree\n");
    printf("   -----+--------+----------------------\n");
    int seps[] = {1, 2, 4, 8, 16, 21, 42};
    for (int si = 0; si < (int)(sizeof(seps) / sizeof(seps[0])); si++) {
        int d = seps[si];
        double accj = 0, acca = 0;
        int np = 0;
        for (int L = 0; L + d < NLAYERS; L++) {
            double jac, agr;
            pair_stats(L, L + d, NULL, &jac, &agr);
            accj += jac; acca += agr; np++;
        }
        if (d == 1) g_dL1_jac = accj / np;
        printf("   %4d | %6d | %8.4f   %8.4f\n", d, np, accj / np, acca / np);
    }
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* PART 4 — activation manifold rank (PCA via Gram trick)              */
/* ------------------------------------------------------------------ */

/* PCA of a subsampled, centered (N x D) activation matrix.
 *
 * We want the eigenspectrum of the DxD covariance C = Xc^T Xc (D=4096). With
 * only N=2048 centered samples, rank(C) <= N, and the nonzero eigenvalues of
 * C are exactly the eigenvalues of the much smaller NxN Gram G = Xc Xc^T, with
 * the same trace (total variance). So we diagonalize the 2048x2048 G instead of
 * a 4096x4096 covariance — 4x cheaper matvec, identical spectrum. cumE ratios
 * (partial eigenvalue sum / trace) are therefore exact covariance energy
 * fractions; only the per-eigenvalue values are power-iteration estimates. */
static void pca_report(const char *path, const char *label, int idx) {
    double t0 = wall_now();
    npy_meta m;
    float *raw = npy_read_f32(path, &m);   /* ~200 MB float32 */
    if (!raw) { fprintf(stderr, "cannot read %s\n", path); return; }
    if (m.ndim != 2 || m.shape[0] != NTOK || m.shape[1] != FFN_DIM) {
        fprintf(stderr, "%s: unexpected shape\n", path); free(raw); return;
    }

    const int N = PCA_SAMPLES, D = FFN_DIM;
    double *X = (double *)malloc((size_t)N * D * sizeof(double));   /* 67 MB */
    if (!X) { fprintf(stderr, "OOM X\n"); free(raw); return; }

    /* subsample N rows evenly (stride PCA_STRIDE) */
    for (int i = 0; i < N; i++) {
        const float *src = raw + (size_t)(i * PCA_STRIDE) * D;
        double *dst = X + (size_t)i * D;
        for (int f = 0; f < D; f++) dst[f] = (double)src[f];
    }
    free(raw);   /* drop the 200 MB buffer ASAP */

    /* center per feature */
    double *mean = (double *)calloc((size_t)D, sizeof(double));
    for (int i = 0; i < N; i++) {
        const double *r = X + (size_t)i * D;
        for (int f = 0; f < D; f++) mean[f] += r[f];
    }
    for (int f = 0; f < D; f++) mean[f] /= N;
    for (int i = 0; i < N; i++) {
        double *r = X + (size_t)i * D;
        for (int f = 0; f < D; f++) r[f] -= mean[f];
    }
    free(mean);

    /* Gram G = Xc Xc^T  (N x N, symmetric PSD) */
    double *G = (double *)malloc((size_t)N * N * sizeof(double));   /* 33.5 MB */
    if (!G) { fprintf(stderr, "OOM G\n"); free(X); return; }
    for (int i = 0; i < N; i++) {
        const double *ri = X + (size_t)i * D;
        for (int k = i; k < N; k++) {
            const double *rk = X + (size_t)k * D;
            double s = 0.0;
            for (int f = 0; f < D; f++) s += ri[f] * rk[f];
            G[(size_t)i * N + k] = s;
            G[(size_t)k * N + i] = s;
        }
    }
    free(X);

    double trace = 0.0;
    for (int i = 0; i < N; i++) trace += G[(size_t)i * N + i];

    /* top-K eigenpairs (we only use the eigenvalues) */
    double *evecs = (double *)malloc((size_t)PCA_K * N * sizeof(double));
    double *evals = (double *)malloc((size_t)PCA_K * sizeof(double));
    if (!evecs || !evals) { fprintf(stderr, "OOM eig\n"); free(G); free(evecs); free(evals); return; }

    fprintf(stderr, "  [PCA] %-18s eig(2048x2048, k=%d) ...\n", label, PCA_K);
    sym_eig_topk(G, N, PCA_K, evecs, evals, PCA_ITERS, PCA_SEED);
    free(G); free(evecs);

    /* cumulative energy fractions */
    double cum = 0.0;
    int rank95 = -1;
    double cumAt[6] = {0};       /* @8,16,32,64,128,K */
    int marks[6] = {8, 16, 32, 64, 128, PCA_K};
    int mi = 0;
    for (int a = 0; a < PCA_K; a++) {
        double ev = evals[a] > 0 ? evals[a] : 0.0;  /* clamp tiny negatives */
        cum += ev;
        double frac = cum / trace;
        if (rank95 < 0 && frac >= 0.95) rank95 = a + 1;
        while (mi < 6 && a + 1 == marks[mi]) { cumAt[mi] = frac; mi++; }
    }
    double lam1_frac = (evals[0] > 0 ? evals[0] : 0.0) / trace;

    char r95[24];
    if (rank95 > 0) snprintf(r95, sizeof(r95), "%d", rank95);
    else            snprintf(r95, sizeof(r95), ">%d", PCA_K);

    printf("  %-18s | %10.3e | %7s | %.4f %.4f %.4f %.4f %.4f | %.4f | %.4f\n",
           label, trace, r95, cumAt[0], cumAt[1], cumAt[2], cumAt[3], cumAt[4],
           cumAt[5], lam1_frac);
    fprintf(stderr, "  [PCA] %-18s done (%.1fs, rank95=%s)\n", label, wall_now() - t0, r95);

    if (idx >= 0 && idx < 6) {
        snprintf(g_pca_label[idx], sizeof(g_pca_label[idx]), "%s", label);
        g_pca_cum128[idx] = cumAt[4];
        g_pca_lam1[idx] = lam1_frac;
        g_pca_rank95[idx] = rank95;   /* -1 if not reached within PCA_K */
    }
    free(evals);
}

static void part4(const char *cap) {
    printf("================================================================================\n");
    printf("PART 4  Activation manifold effective rank  (PCA, %d tokens subsampled, Gram trick)\n", PCA_SAMPLES);
    printf("  rank95 = # principal components to reach 95%% of total variance (lower = lower-dim)\n");
    printf("  cumE@k = cumulative variance fraction in the top-k components; lam1 = top-1 share\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  tensor             |   totalVar | rank95  | cumE@8 @16    @32    @64    @128  | cumE@K | lam1\n");
    printf("  -------------------+------------+---------+----------------------------------+--------+-------\n");

    int layers[] = {0, 21, 42};
    for (int i = 0; i < 3; i++) {
        int L = layers[i];
        char path[1024], label[64];
        snprintf(path, sizeof(path), "%s/ffn_in_L%d.npy", cap, L);
        snprintf(label, sizeof(label), "ffn_in  L%d", L);
        pca_report(path, label, 2 * i);
        snprintf(path, sizeof(path), "%s/ffn_out_L%d.npy", cap, L);
        snprintf(label, sizeof(label), "ffn_out L%d", L);
        pca_report(path, label, 2 * i + 1);
    }
    printf("  (K=%d components computed; rank95 shown as '>%d' if 95%% not reached within top-%d)\n\n",
           PCA_K, PCA_K, PCA_K);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    const char *cap = NULL;   /* 必传: 旧默认是某台机的 /tmp 取料目录, 漏传=静默读错机器 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cap") == 0 && i + 1 < argc) cap = argv[++i];
    }
    if (!cap) { fprintf(stderr, "traj_analyze: --cap <取料目录> 必传(无默认)\n"); return 2; }

    printf("################################################################################\n");
    printf("# Go-domain trajectory (go-trace) structure analysis\n");
    printf("# capture dir: %s\n", cap);
    printf("# %d layers, %d Go tokens, %d routed experts, top-%d, FFN dim %d\n",
           NLAYERS, NTOK, NEXPERT, TOPK, FFN_DIM);
    printf("################################################################################\n\n");

    if (load_all_routes(cap) != 0) return 1;

    part1();
    part2();
    part3();
    part4(cap);

    /* ------------------------------------------------------------------ */
    /* SUMMARY                                                             */
    /* ------------------------------------------------------------------ */
    int global_dead = 0, used_all = 0, used_some = 0;
    for (int e = 0; e < NEXPERT; e++) {
        if (g_layers_fired[e] == 0) global_dead++; else used_some++;
        if (g_layers_fired[e] == NLAYERS) used_all++;
    }
    /* averages over the three ffn_in / three ffn_out PCA runs */
    double in_cum128 = (g_pca_cum128[0] + g_pca_cum128[2] + g_pca_cum128[4]) / 3.0;
    double out_lam1_last = g_pca_lam1[5];   /* ffn_out L42 */
    double out_lam1_mid = g_pca_lam1[3];    /* ffn_out L21 */

    printf("################################################################################\n");
    printf("# SUMMARY — low-dimensional Go regularities found (honest reading of the tables)\n");
    printf("################################################################################\n");
    printf("  [usage]   Go exercises %d / 256 expert-id slots somewhere; %d are network-dead\n",
           used_some, global_dead);
    printf("            (never fire anywhere); %d fire in ALL %d layers. Per layer only ~%.0f\n",
           used_all, NLAYERS, sum_dead / NLAYERS);
    printf("            experts are Go-dead, and ~%.0f / %.0f / %.0f experts carry 50/90/99%% of\n",
           sum_n50 / NLAYERS, sum_n90 / NLAYERS, sum_n99 / NLAYERS);
    printf("            firings (mean H/8 = %.3f). => usage is concentrated but NOT sparse:\n",
           sum_Hnorm / NLAYERS);
    printf("            ~95%% coverage still needs ~%.0f of 256 experts; few are droppable.\n",
           sum_n95 / NLAYERS);
    printf("  [shallow] STRONGEST low-dim signal: layers L0-L2 collapse to only %ld distinct\n",
           distinct_L[0]);
    printf("            top-6 combos (~%.0f tokens/combo, %.1f%% unique) — early routing is\n",
           (double)NTOK / distinct_L[0], 100.0 * distinct_L[0] / NTOK);
    printf("            near-deterministic in token identity. Deep layers L3-L42 are %.0f-%.0f%%\n",
           100.0 * distinct_L[3] / NTOK, 100.0 * max_distinct / NTOK);
    printf("            unique (mean %.1f%%): combination diversity is high once past the input.\n",
           100.0 * sum_frac_unique / NLAYERS);
    printf("  [path]    Cross-layer top-6 SET overlap is at chance everywhere: avg Jaccard at\n");
    printf("            separation 1 = %.4f vs random %.4f. So expert SETS are ~independent\n",
           g_dL1_jac, g_rnd_jac);
    printf("            per layer — the Go 'trajectory' is NOT a continuous expert path. Only\n");
    printf("            a mild PRIMARY-expert correlation survives in the shallowest pair\n");
    printf("            (L0<->L1 primAgree %.4f = %.1fx chance), decaying to chance by mid/deep.\n",
           g_l0l1_agree, g_l0l1_chance > 0 ? g_l0l1_agree / g_l0l1_chance : 0.0);
    printf("  [rank]    FFN INPUTS (residual) are HIGH rank: 95%% needs >%d PCA dirs, cumE@128\n",
           PCA_K);
    printf("            only ~%.2f. FFN OUTPUTS shed rank with depth: L21-out has a dominant\n",
           in_cum128);
    printf("            direction (lam1=%.2f) and L42-out is effectively RANK-1 (lam1=%.3f),\n",
           out_lam1_mid, out_lam1_last);
    printf("            but its total variance is ~10^4x larger => driven by a few massive-\n");
    printf("            activation outlier tokens (a late-layer artifact, not Go-generic).\n");
    printf("  --------------------------------------------------------------------------------\n");
    printf("  Takeaway for a Go quantizer: the exploitable regularities are (1) per-layer\n");
    printf("  usage concentration (~%.0f experts cover 99%%, ~%.0f dead/layer) and (2) the\n",
           sum_n99 / NLAYERS, sum_dead / NLAYERS);
    printf("  ~%ld-combo shallow-routing collapse (L0-L2). There is NO low-dim cross-layer\n",
           distinct_L[0]);
    printf("  expert path and NO low input-activation rank to lean on; routed-expert SETS in\n");
    printf("  deep layers stay high-entropy and near-unique per token.\n");
    printf("################################################################################\n");
    return 0;
}
