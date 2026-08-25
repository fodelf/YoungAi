/* one layer of the aggregate pipeline; returns 0 on success.
 * feat_yhat: fit the latent on φ=ŷ (routed_out) instead of φ=x — runtime-
 * realizable by binding routed_out as the corr kernel's x buffer.
 * dump_sel: non-NULL => stop after assembly and persist the compact arrays. */
static int run_rrr_layer(const char *hf_dir, const char *cap_dir, int L,
                         int n_x, int n_test_max, int chunk, double heldout_frac,
                         int n_threads, int feat_yhat, const char *dump_sel,
                         const char *obase_dir, rrr_params *rp) {
    char path[1024];
    npy_meta mx, mi, mr;
    int rc = 1;
    time_t t0 = time(NULL);

    snprintf(path, sizeof path, "%s/ffn_in_L%d.npy", cap_dir, L);
    float *x_all = npy_read_f32(path, &mx);
    snprintf(path, sizeof path, "%s/route_L%d.npy", cap_dir, L);
    float *ids_all = npy_read_f32(path, &mi);
    snprintf(path, sizeof path, "%s/routed_L%d.npy", cap_dir, L);
    float *ref_all = npy_read_f32(path, &mr);
    if (!x_all || !ids_all || !ref_all ||
        mx.ndim != 2 || mx.shape[1] != DM || mi.ndim != 2 || mi.shape[1] != RRR_TOPK ||
        mr.ndim != 2 || mr.shape[1] != DM || mi.shape[0] != mx.shape[0] || mr.shape[0] != mx.shape[0]) {
        fprintf(stderr, "L%d: bad cap tensors (need ffn_in/route/routed)\n", L);
        free(x_all); free(ids_all); free(ref_all);
        return 1;
    }
    int64_t n_tok = mx.shape[0];

    /* held-out split on CHUNK boundaries: last frac chunks are test */
    int n_chunks = (int)(n_tok / chunk); if (n_chunks < 1) n_chunks = 1;
    int n_te_ch = heldout_frac > 0 ? (int)(heldout_frac * n_chunks + 0.5) : 0;
    if (n_te_ch >= n_chunks) n_te_ch = n_chunks - 1;
    int64_t pool = (int64_t)(n_chunks - n_te_ch) * chunk;
    if (pool > n_tok) pool = n_tok;
    int ntr = n_x < pool ? n_x : (int)pool;
    int nte = 0;
    if (n_te_ch > 0) {
        int64_t avail = n_tok - pool;
        nte = n_test_max < avail ? n_test_max : (int)avail;
    }
    int n_sel = ntr + nte;

    /* gate weights: prefer captured route_w (post-E1); else recompute exactly */
    float *rw_sel = malloc((size_t)n_sel * RRR_TOPK * sizeof(float));
    float *route_sel = malloc((size_t)n_sel * RRR_TOPK * sizeof(float));
    float *Xsel = malloc((size_t)n_sel * DM * sizeof(float));
    float *Yref = malloc((size_t)n_sel * DM * sizeof(float));
    float *Yhat = calloc((size_t)n_sel * DM, sizeof(float));
    if (!rw_sel || !route_sel || !Xsel || !Yref || !Yhat) { fprintf(stderr, "L%d: OOM sel arrays\n", L); goto out; }
    for (int s = 0; s < n_sel; s++) {
        int64_t t = s < ntr ? (int64_t)s : pool + (s - ntr);
        memcpy(Xsel + (size_t)s * DM, x_all + (size_t)t * DM, (size_t)DM * sizeof(float));
        memcpy(Yref + (size_t)s * DM, ref_all + (size_t)t * DM, (size_t)DM * sizeof(float));
        memcpy(route_sel + (size_t)s * RRR_TOPK, ids_all + (size_t)t * RRR_TOPK, RRR_TOPK * sizeof(float));
    }
    free(x_all); x_all = NULL;
    free(ref_all); ref_all = NULL;

    {
        npy_meta mw;
        snprintf(path, sizeof path, "%s/route_w_L%d.npy", cap_dir, L);
        float *rw_all = npy_read_f32(path, &mw);
        if (rw_all && mw.ndim == 2 && mw.shape[1] == RRR_TOPK && mw.shape[0] == n_tok) {
            for (int s = 0; s < n_sel; s++) {
                int64_t t = s < ntr ? (int64_t)s : pool + (s - ntr);
                memcpy(rw_sel + (size_t)s * RRR_TOPK, rw_all + (size_t)t * RRR_TOPK, RRR_TOPK * sizeof(float));
            }
            free(rw_all);
            printf("L%d: gate weights from captured route_w\n", L);
        } else {
            free(rw_all);
            hf_db *db = hf_open(hf_dir);
            char gname[256]; int64_t gn = 0;
            snprintf(gname, sizeof gname, "layers.%d.ffn.gate.weight", L);
            float *gate_w = db ? hf_read_f32(db, gname, &gn) : NULL;
            if (!gate_w || gn != (int64_t)NEXP * DM) {
                fprintf(stderr, "L%d: router gate read failed\n", L);
                free(gate_w); if (db) hf_close(db);
                goto out;
            }
            for (int s = 0; s < n_sel; s++) {
                const float *x = Xsel + (size_t)s * DM;
                const float *idf = route_sel + (size_t)s * RRR_TOPK;
                double sc[RRR_TOPK], ssum = 0;
                for (int e = 0; e < RRR_TOPK; e++) {
                    int id = (int)idf[e];
                    double zv = 0;
                    if (id >= 0 && id < NEXP) {
                        const float *g = gate_w + (size_t)id * DM;
                        for (int j = 0; j < DM; j++) zv += (double)g[j] * (double)x[j];
                    }
                    sc[e] = rrr_sqrtsoftplus(zv);
                    ssum += sc[e];
                }
                if (!(ssum > 0)) ssum = 1.0;
                for (int e = 0; e < RRR_TOPK; e++)
                    rw_sel[(size_t)s * RRR_TOPK + e] = (float)(sc[e] / ssum * 1.5);
            }
            free(gate_w); hf_close(db);
            printf("L%d: gate weights recomputed (sqrtsoftplus@ids, norm_topk, x1.5)\n", L);
        }
    }

    /* ŷ from obase file (deployed-bytes student, e.g. v3 GPTQ-sign 底座): HF 现场
     * go1b(HF) 模拟只能重现 naive-sign 学生, 对符号重选后的底座是错的学生 —— 有
     * obase 就直读, 跳过整个装配. 文件 = obase_v3_L{L}.npy [n_tok, DM] (f16 npy). */
    int yhat_from_file = 0;
    if (obase_dir && obase_dir[0]) {
        npy_meta mo;
        snprintf(path, sizeof path, "%s/obase_v3_L%d.npy", obase_dir, L);
        float *ob = npy_read_f32(path, &mo);
        if (ob && mo.ndim == 2 && mo.shape[1] == DM && mo.shape[0] == n_tok) {
            for (int s = 0; s < n_sel; s++) {
                int64_t t = s < ntr ? (int64_t)s : pool + (s - ntr);
                memcpy(Yhat + (size_t)s * DM, ob + (size_t)t * DM, (size_t)DM * sizeof(float));
            }
            free(ob);
            yhat_from_file = 1;
            printf("L%d: ŷ from obase (deployed-bytes student)  ntr=%d nte=%d\n", L, ntr, nte);
            fflush(stdout);
        } else {
            free(ob);
            fprintf(stderr, "L%d: obase 缺失/形状不符: %s (need [%lld,%d])\n",
                    L, path, (long long)n_tok, DM);
            goto out;
        }
    }

    /* CSR inverted index expert -> (row, weight) */
    if (!yhat_from_file) {
        int *cnt = calloc(NEXP + 1, sizeof(int));
        int *pair_off = malloc((NEXP + 1) * sizeof(int));
        int *pair_row = malloc((size_t)n_sel * RRR_TOPK * sizeof(int));
        float *pair_w = malloc((size_t)n_sel * RRR_TOPK * sizeof(float));
        if (!cnt || !pair_off || !pair_row || !pair_w) { free(cnt); free(pair_off); free(pair_row); free(pair_w); goto out; }
        for (int s = 0; s < n_sel; s++)
            for (int e = 0; e < RRR_TOPK; e++) {
                int id = (int)route_sel[(size_t)s * RRR_TOPK + e];
                if (id >= 0 && id < NEXP) cnt[id]++;
            }
        pair_off[0] = 0;
        for (int e = 0; e < NEXP; e++) pair_off[e + 1] = pair_off[e] + cnt[e];
        memset(cnt, 0, (NEXP + 1) * sizeof(int));
        for (int s = 0; s < n_sel; s++)
            for (int e = 0; e < RRR_TOPK; e++) {
                int id = (int)route_sel[(size_t)s * RRR_TOPK + e];
                if (id < 0 || id >= NEXP) continue;
                int at = pair_off[id] + cnt[id]++;
                pair_row[at] = s;
                pair_w[at] = rw_sel[(size_t)s * RRR_TOPK + e];
            }
        free(cnt);

        /* ŷ assembly: expert-band workers, worker-local accumulators */
        pthread_t th[NEXP]; rrr_ctx ctx[64];
        int nthr = n_threads > 64 ? 64 : n_threads;
        int per = (NEXP + nthr - 1) / nthr, nt = 0, bad = 0;
        for (int e0 = 0; e0 < NEXP; e0 += per) {
            rrr_ctx *c = &ctx[nt];
            memset(c, 0, sizeof *c);
            c->hf_dir = hf_dir; c->layer = L;
            c->e_begin = e0; c->e_end = e0 + per < NEXP ? e0 + per : NEXP;
            c->Xsel = Xsel; c->pair_row = pair_row; c->pair_w = pair_w; c->pair_off = pair_off;
            c->n_sel = n_sel;
            c->acc = calloc((size_t)n_sel * DM, sizeof(float));
            if (!c->acc) { bad = 1; break; }
            if (pthread_create(&th[nt], NULL, rrr_yhat_worker, c) != 0) { bad = 1; free(c->acc); break; }
            nt++;
        }
        int done_exp = 0;
        for (int t = 0; t < nt; t++) {
            pthread_join(th[t], NULL);
            bad += ctx[t].rc; done_exp += ctx[t].done;
            const float *a = ctx[t].acc;
            for (size_t i = 0; i < (size_t)n_sel * DM; i++) Yhat[i] += a[i];
            free(ctx[t].acc);
        }
        free(pair_off); free(pair_row); free(pair_w);
        if (bad) { fprintf(stderr, "L%d: rrr ŷ worker(s) failed\n", L); goto out; }
        printf("L%d: ŷ assembled  ntr=%d nte=%d experts_touched=%d  (%lds)\n",
               L, ntr, nte, done_exp, (long)(time(NULL) - t0));
        fflush(stdout);
    }

    if (dump_sel) {   /* two-stage split: assembly host stops here */
        rc = sel_write(dump_sel, L, ntr, nte, Xsel, Yhat, Yref, route_sel);
        goto out;
    }

    /* optional next-layer router weight for the Q metric */
    {
        float *Wr = NULL;
        if (rp->alpha_router > 0 && L + 1 <= 42) {
            hf_db *db = hf_open(hf_dir);
            char gname[256]; int64_t gn = 0;
            snprintf(gname, sizeof gname, "layers.%d.ffn.gate.weight", L + 1);
            Wr = db ? hf_read_f32(db, gname, &gn) : NULL;
            if (Wr && gn != (int64_t)NEXP * DM) { free(Wr); Wr = NULL; }
            if (db) hf_close(db);
            printf("L%d: Q-metric alpha=%.3f Wr(L+1) %s\n", L, rp->alpha_router, Wr ? "loaded" : "MISSING -> alpha off");
        }
        rp->Wr = Wr;
        if (!Wr) rp->alpha_router = 0;

        int *itr = malloc((size_t)ntr * sizeof(int));
        int *ite = nte > 0 ? malloc((size_t)nte * sizeof(int)) : NULL;
        if (!itr || (nte > 0 && !ite)) { free(itr); free(ite); free(Wr); goto out; }
        for (int i = 0; i < ntr; i++) itr[i] = i;
        for (int i = 0; i < nte; i++) ite[i] = ntr + i;
        rp->chunk = chunk;

        /* closed-form scalar-gain baseline: y' = (1+s)·ŷ, s = Σ⟨T,ŷ⟩/Σ⟨ŷ,ŷ⟩ on
         * train — quantifies how much of the gap is pure 1-bit norm collapse. */
        {
            double shy = 0, shh = 0;
            for (int s = 0; s < ntr; s++) {
                const float *h = Yhat + (size_t)s * DM, *r = Yref + (size_t)s * DM;
                for (int j = 0; j < DM; j++) { shy += (double)h[j] * (double)r[j]; shh += (double)h[j] * (double)h[j]; }
            }
            double gain = shh > 0 ? shy / shh : 1.0;   /* (1+s) = ⟨y*,ŷ⟩/⟨ŷ,ŷ⟩ */
            double csum = 0, num = 0, den = 0; long cnt = 0;
            for (int s = ntr; s < ntr + nte; s++) {
                const float *h = Yhat + (size_t)s * DM, *r = Yref + (size_t)s * DM;
                double dot = 0, np2 = 0, nr2 = 0;
                for (int j = 0; j < DM; j++) {
                    double pj = gain * (double)h[j], rj = (double)r[j];
                    dot += pj * rj; np2 += pj * pj; nr2 += rj * rj;
                    double e2 = rj - pj; num += e2 * e2; den += rj * rj;
                }
                if (np2 > 0 && nr2 > 0) { csum += dot / (sqrt(np2) * sqrt(nr2)); cnt++; }
            }
            if (cnt)
                printf("L%d: scalar-gain diag: (1+s)=%.4f  TE cos=%.5f rel=%.4f (cos unchanged by scalar; rel isolates norm collapse)\n",
                       L, gain, csum / cnt, den > 0 ? sqrt(num / den) : 0);
        }

        z_layer z; rrr_report rep;
        time_t t1 = time(NULL);
        const float *feat = feat_yhat ? Yhat : Xsel;
        int src = rrr_solve(feat, Yhat, Yref, route_sel, RRR_TOPK,
                            itr, ntr, ite, nte, DM, NEXP, rp, &z, &rep);
        free(itr); free(ite); free(Wr); rp->Wr = NULL;
        if (src) { fprintf(stderr, "L%d: rrr_solve rc=%d\n", L, src); goto out; }
        printf("%-4d | %3d | %6.4f | TR base_cos %7.5f -> corr %7.5f (rel %6.4f -> %6.4f) | TE base_cos %7.5f -> corr %7.5f (rel %6.4f -> %6.4f) | solve %lds\n",
               L, rep.d_l, rep.energy_at_k,
               rep.base_cos_tr, rep.corr_cos_tr, rep.base_rel_tr, rep.corr_rel_tr,
               rep.base_cos_te, rep.corr_cos_te, rep.base_rel_te, rep.corr_rel_te,
               (long)(time(NULL) - t1));
        rrr_dump_z(&z, L);
        z_layer_free(&z);
    }
    rc = 0;
out:
    free(x_all); free(ids_all); free(ref_all);
    free(rw_sel); free(route_sel); free(Xsel); free(Yref); free(Yhat);
    return rc;
}

int main(int argc, char **argv) {
    const char *hf_dir = "hf/DeepSeek-V4-Flash-Base";
    const char *cap_dir = "/private/tmp/m1_ds4/cap_m1";
    const char *layers_s = "0,4,8";
    int n_x = 64, n_threads = 6, max_rank = 64, rank = 8, ntrain = 0, n_feat = 1024;
    int n_exp = NEXP;  /* --n-experts: subset of experts for fast iteration (default all 256) */
    int no_solve = 0;  /* --no-solve: only the 1-bit base_cos/relL2 (skip the ~100s four-loss solve) */
    double energy = 0.95, lambda = -1.0;
    double w_align = -1, w_smooth = -1, w_classify = -1, w_fixed = -1;  /* -1 = keep solver default */
    const char *solver = "hv";  /* hv | perexpert | denoise | genrf | rrr (aggregate four-loss, held-out) */
    double heldout_frac = 0.0, lam_c = 1e-2, alpha_router = 0.0;
    int ntest = 1024, chunk = 512, eig_iters = 0, no_procrustes = 0;
    const char *feat_s = "x";           /* rrr latent features: x | yhat */
    const char *umode_s = "rrr";        /* shared basis: rrr | pca (feature PCA) */
    const char *gostats_path = NULL;    /* L_fix per-expert E[x²] .dat  */
    const char *dump_sel = NULL;        /* two-stage: write assembly, skip solve */
    const char *load_sel = NULL;        /* two-stage: solve from a sel dump      */
    const char *obase_dir = NULL;       /* ŷ 直读部署字节 obase (v3 底座必用)     */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hf") && i+1<argc) hf_dir = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i+1<argc) cap_dir = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i+1<argc) layers_s = argv[++i];
        else if (!strcmp(argv[i], "--nx") && i+1<argc) n_x = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1<argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--energy") && i+1<argc) energy = atof(argv[++i]);
        else if (!strcmp(argv[i], "--maxrank") && i+1<argc) max_rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i+1<argc) lambda = atof(argv[++i]);
        else if (!strcmp(argv[i], "--w-align") && i+1<argc) w_align = atof(argv[++i]);
        else if (!strcmp(argv[i], "--w-smooth") && i+1<argc) w_smooth = atof(argv[++i]);
        else if (!strcmp(argv[i], "--w-classify") && i+1<argc) w_classify = atof(argv[++i]);
        else if (!strcmp(argv[i], "--w-fixed") && i+1<argc) w_fixed = atof(argv[++i]);
        else if (!strcmp(argv[i], "--solver") && i+1<argc) solver = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i+1<argc) rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ntrain") && i+1<argc) ntrain = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nfeat") && i+1<argc) n_feat = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--n-experts") || !strcmp(argv[i], "--nexp")) && i+1<argc) {
            n_exp = atoi(argv[++i]); if (n_exp < 1) n_exp = 1; if (n_exp > NEXP) n_exp = NEXP;
        }
        else if (!strcmp(argv[i], "--no-solve")) no_solve = 1;
        else if (!strcmp(argv[i], "--heldout-frac") && i+1<argc) heldout_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ntest") && i+1<argc) ntest = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk") && i+1<argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lam-c") && i+1<argc) lam_c = atof(argv[++i]);
        else if (!strcmp(argv[i], "--alpha-router") && i+1<argc) alpha_router = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eig-iters") && i+1<argc) eig_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-procrustes")) no_procrustes = 1;
        else if (!strcmp(argv[i], "--feat") && i+1<argc) feat_s = argv[++i];
        else if (!strcmp(argv[i], "--umode") && i+1<argc) umode_s = argv[++i];
        else if (!strcmp(argv[i], "--go-stats") && i+1<argc) gostats_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-sel") && i+1<argc) dump_sel = argv[++i];
        else if (!strcmp(argv[i], "--load-sel") && i+1<argc) load_sel = argv[++i];
        else if (!strcmp(argv[i], "--obase-dir") && i+1<argc) obase_dir = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > n_exp) n_threads = n_exp;

    if (gostats_path && gostats_load(gostats_path)) return 2;

    printf("calib_run: hf=%s cap=%s layers=%s n_x=%d n_exp=%d threads=%d energy=%.3f maxrank=%d%s\n",
           hf_dir, cap_dir, layers_s, n_x, n_exp, n_threads, energy, max_rank,
           g_gs_n ? " [go-stats L_fix scales]" : "");
    printf("hypothesis: per-layer 256-expert 1-bit Δo low-dim + recoverable by z^ℓ\n\n");

    /* parse layer list */
    int layers[64], nL = 0;
    { char buf[256]; strncpy(buf, layers_s, sizeof buf - 1); buf[sizeof buf -1]=0;
      for (char *t = strtok(buf, ","); t && nL < 64; t = strtok(NULL, ",")) layers[nL++] = atoi(t); }

    if (!strcmp(solver, "rrr")) {
        rrr_params rp; rrr_params_default(&rp);
        rp.k = rank;
        rp.lambda_rel = lambda >= 0 ? lambda : 1e-4;
        rp.lam_c_rel = lam_c;
        rp.w_smooth = w_smooth >= 0 ? w_smooth : 0.0;
        rp.w_align = w_align >= 0 ? w_align : 0.0;
        rp.alpha_router = alpha_router;
        rp.procrustes = !no_procrustes;
        rp.eig_iters = eig_iters;
        rp.threads = n_threads;
        int feat_yhat = !strcmp(feat_s, "yhat");
        rp.umode = !strcmp(umode_s, "pca") ? 1 : 0;
        printf("solver=rrr (aggregate four-loss): rank=%d lambda=%.2e lam_c=%.2e w_smooth=%.2e alpha=%.2f heldout=%.2f chunk=%d ntest=%d feat=%s umode=%s\n\n",
               rank, rp.lambda_rel, lam_c, rp.w_smooth, alpha_router, heldout_frac, chunk, ntest, feat_s, umode_s);
        int badL = 0;
        for (int li = 0; li < nL; li++) {
            if (load_sel)
                badL += run_rrr_layer_from_sel(load_sel, layers[li], chunk, feat_yhat, &rp) != 0;
            else
                badL += run_rrr_layer(hf_dir, cap_dir, layers[li], n_x, ntest, chunk,
                                      heldout_frac, n_threads, feat_yhat, dump_sel, obase_dir, &rp) != 0;
        }
        printf("\ndone.\n");
        return badL ? 1 : 0;
    }

    printf("%-4s | %-10s | %-5s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
           "L", "tot_energy", "d_l", "cumE@d_l", "base_relL2", "corr_relL2", "base_cos", "corr_cos");
    printf("-----+------------+-------+-----------+-----------+------------+-----------+----------\n");

    size_t cell = (size_t)n_exp * n_x * DM;
    double *o_ref = malloc(cell * sizeof(double));
    double *o_hat = malloc(cell * sizeof(double));
    double *delta = malloc(cell * sizeof(double));
    float  *x_f32 = malloc((size_t)n_x * DM * sizeof(float));
    double *x_d   = malloc((size_t)n_x * DM * sizeof(double));
    if (!o_ref || !o_hat || !delta || !x_f32 || !x_d) { fprintf(stderr, "OOM main arrays (%.1f GiB)\n", 3.0*cell*8/1e9); return 1; }

    for (int li = 0; li < nL; li++) {
        int L = layers[li];
        char path[512];
        snprintf(path, sizeof path, "%s/ffn_in_L%d.npy", cap_dir, L);
        npy_meta m;
        float *xfull = npy_read_f32(path, &m);
        if (!xfull || m.ndim != 2 || m.shape[1] != DM) { fprintf(stderr, "L%d: bad %s\n", L, path); free(xfull); continue; }
        int64_t total = m.shape[0];
        int stride = (int)(total / n_x); if (stride < 1) stride = 1;
        for (int i = 0; i < n_x; i++) {
            const float *src = xfull + (size_t)((int64_t)i * stride % total) * DM;
            for (int j = 0; j < DM; j++) { x_f32[(size_t)i*DM+j] = src[j]; x_d[(size_t)i*DM+j] = (double)src[j]; }
        }
        free(xfull);

        /* fan out over experts */
        pthread_t th[NEXP]; worker_ctx ctx[NEXP];
        int per = (n_exp + n_threads - 1) / n_threads, nt = 0;
        for (int e0 = 0; e0 < n_exp; e0 += per) {
            worker_ctx *c = &ctx[nt];
            c->hf_dir = hf_dir; c->layer = L; c->e_begin = e0; c->e_end = (e0+per<n_exp)?e0+per:n_exp;
            c->n_x = n_x; c->x_f32 = x_f32; c->o_ref = o_ref; c->o_hat = o_hat; c->delta = delta;
            c->rc = 1; c->done_experts = 0;
            if (pthread_create(&th[nt], NULL, worker, c) != 0) { fprintf(stderr, "pthread_create\n"); return 1; }
            nt++;
        }
        int bad = 0, did = 0;
        for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); bad += ctx[t].rc; did += ctx[t].done_experts; }
        if (bad) { fprintf(stderr, "L%d: %d worker(s) failed (experts done=%d) — skipping\n", L, bad, did); continue; }

        /* baseline + solve + fidelity */
        double b_rel, b_cos;
        baseline_metrics(o_ref, o_hat, delta, n_exp, n_x, DM, &b_rel, &b_cos);

        if (no_solve) {  /* fast 1-bit probe: base_cos/relL2 only, skip the ~100s four-loss solve */
            printf("%-4d | %10s | %5s | %9s | %9.4f | %10s | %9.5f | %8s\n",
                   L, "-", "-", "-", b_rel, "-", b_cos, "-");
            continue;
        }

        double corr_rel, corr_cos, totE = 0, cumFrac = 0; int dl_disp; int have_spec = 0;
        z_spectrum spec; memset(&spec, 0, sizeof spec);
        if (!strcmp(solver, "perexpert")) {
            double lam = (lambda >= 0) ? lambda : 1.0;
            pe_result pr = solve_perexpert(delta, x_d, o_ref, o_hat, n_exp, n_x, DM, rank, lam);
            if (!pr.ok) { fprintf(stderr, "L%d: solve_perexpert failed\n", L); continue; }
            corr_rel = pr.rel_l2; corr_cos = pr.cos_mean; dl_disp = rank;
        } else if (!strcmp(solver, "denoise")) {
            int ntr = (ntrain > 0) ? ntrain : (n_x * 3 / 4);
            double lam = (lambda >= 0) ? lambda : 1.0;
            dn_result dr = solve_denoise(o_ref, o_hat, n_exp, n_x, DM, rank, lam, ntr);  /* HELD-OUT eval */
            if (!dr.ok) { fprintf(stderr, "L%d: solve_denoise failed\n", L); continue; }
            corr_rel = dr.rel_l2; corr_cos = dr.cos_mean; dl_disp = rank;
        } else if (!strcmp(solver, "genrf")) {
            int ntr = (ntrain > 0) ? ntrain : (n_x * 3 / 4);
            double lam = (lambda >= 0) ? lambda : 1.0;
            gr_result gr = solve_genrf(o_ref, o_hat, x_d, n_exp, n_x, DM, n_feat, lam, ntr, 1234ULL);  /* HELD-OUT */
            if (!gr.ok) { fprintf(stderr, "L%d: solve_genrf failed\n", L); continue; }
            corr_rel = gr.rel_l2; corr_cos = gr.cos_mean; dl_disp = n_feat;
        } else {
            hv_params p; hv_params_default(&p);
            p.energy_threshold = energy; p.max_rank = max_rank;
            if (lambda >= 0) p.lambda = lambda;
            if (w_align >= 0)    p.w_align    = w_align;
            if (w_smooth >= 0)   p.w_smooth   = w_smooth;
            if (w_classify >= 0) p.w_classify = w_classify;
            if (w_fixed >= 0)    p.w_fixed    = w_fixed;
            z_layer z; z_fidelity fid; memset(&z, 0, sizeof z);
            int rc = hv_solve(delta, x_d, o_ref, NULL, NULL, n_exp, n_x, DM, 0, &p, &z, &spec);
            if (rc) { fprintf(stderr, "L%d: hv_solve rc=%d\n", L, rc); continue; }
            hv_fidelity(&z, delta, x_d, o_hat, n_exp, n_x, DM, &fid);  /* n_exp (not NEXP): arrays sized to n_exp -> OOB/segfault when n_exp<256 */
            /* Dump z^L {d_model,n_exp,d_l, U,V,C,b,beta,delta} as float32 for GGUF emit + ds4 apply. */
            const char *zdir = getenv("DS4_Z_DUMP_DIR");
            if (zdir && zdir[0]) {
                char zpath[1024];
                snprintf(zpath, sizeof zpath, "%s/z_L%d.bin", zdir, L);
                FILE *zf = fopen(zpath, "wb");
                if (zf) {
                    int32_t hdr[3] = { z.d_model, z.n_exp, z.d_l };
                    fwrite(hdr, sizeof(int32_t), 3, zf);
                    size_t nU=(size_t)z.d_model*z.d_l, nV=(size_t)z.d_l*z.d_model, nC=(size_t)z.n_exp*z.d_l;
                    #define ZWF(a,n) do{ for(size_t _i=0;_i<(n);_i++){ float _v=(float)(a)[_i]; fwrite(&_v,4,1,zf);} }while(0)
                    ZWF(z.U,nU); ZWF(z.V,nV); ZWF(z.C,nC); ZWF(z.b,(size_t)z.d_model); ZWF(z.beta,(size_t)z.n_exp); ZWF(z.delta,(size_t)z.n_exp);
                    #undef ZWF
                    fclose(zf);
                    fprintf(stderr, "L%d: z dumped d_l=%d -> %s\n", L, z.d_l, zpath);
                }
            }
            double cumE = 0; for (int k = 0; k < spec.d_l && k < spec.n; k++) cumE += spec.evals[k];
            cumFrac = spec.total_energy > 0 ? cumE / spec.total_energy : 0;
            corr_rel = fid.rel_l2; corr_cos = fid.cosine_mean; dl_disp = spec.d_l; totE = spec.total_energy; have_spec = 1;
            z_layer_free(&z);
        }
        printf("%-4d | %10.3e | %5d | %9.4f | %9.4f | %10.4f | %9.5f | %8.5f\n",
               L, totE, dl_disp, cumFrac, b_rel, corr_rel, b_cos, corr_cos);
        if (have_spec) {
            printf("      eig[0..7]:");
            for (int k = 0; k < 8 && k < spec.n; k++) printf(" %.3e", spec.evals[k]);
            printf("\n");
        }
        z_spectrum_free(&spec);
    }

    free(o_ref); free(o_hat); free(delta); free(x_f32); free(x_d);
    printf("\ndone.\n");
    return 0;
}
