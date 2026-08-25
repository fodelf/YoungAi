/* main solve                                                           */
/* ------------------------------------------------------------------ */
int rrr_solve(const float *X, const float *Yhat, const float *Yref,
              const float *route, int topk,
              const int *idx_tr, int n_train, const int *idx_te, int n_test,
              int d, int n_exp,
              const rrr_params *pin, z_layer *z, rrr_report *rep) {
    if (!X || !Yhat || !Yref || !route || !idx_tr || n_train < 8 || d < 1 || n_exp < 1 || n_exp > 1024)
        return 1;
    rrr_params p = *pin;
    if (p.k < 1) p.k = 1;
    if (p.k > d) p.k = d;
    if (p.threads < 1) p.threads = 1;
    const int k = p.k;
    int rc = 1;

    memset(z, 0, sizeof *z);
    memset(rep, 0, sizeof *rep);
    rep->n_train = n_train; rep->n_test = n_test; rep->d_l = k;

    /* ---- assemble contiguous train X/T (+ optional whitening + diffs) ---- */
    /* diff pairs: adjacent capture positions (idx differs by 1) inside one
     * chunk — L_smooth as real augmented samples sqrt(w)·(Δx, ΔT). */
    int n_pairs = 0;
    if (p.w_smooth > 0.0 && p.chunk > 1) {
        for (int s = 1; s < n_train; s++)
            if (idx_tr[s] == idx_tr[s-1] + 1 && (idx_tr[s] % p.chunk) != 0) n_pairs++;
    }
    const int n_gram = n_train + n_pairs;
    qmet qm; memset(&qm, 0, sizeof qm);
    float *Xs = (float *)malloc((size_t)n_gram * d * sizeof(float));
    float *Ts = (float *)malloc((size_t)n_gram * d * sizeof(float));
    double *bfit = (double *)calloc((size_t)d, sizeof(double));
    if (!Xs || !Ts || !bfit) goto cleanup0;

    for (int s = 0; s < n_train; s++) {
        int t = idx_tr[s];
        const float *x = X + (size_t)t * d;
        const float *h = Yhat + (size_t)t * d;
        const float *r = Yref + (size_t)t * d;
        float *xd = Xs + (size_t)s * d, *td = Ts + (size_t)s * d;
        for (int j = 0; j < d; j++) {
            xd[j] = x[j];
            double tv = (double)r[j] - (double)h[j];
            td[j] = (float)tv;
            bfit[j] += tv;
        }
    }
    for (int j = 0; j < d; j++) bfit[j] /= (double)n_train;
    for (int s = 0; s < n_train; s++) {           /* center targets */
        float *td = Ts + (size_t)s * d;
        for (int j = 0; j < d; j++) td[j] = (float)((double)td[j] - bfit[j]);
    }

    /* L_align: inverse-norm token reweight — scale each (x,T) row by √a_t so
     * the LS stops being hijacked by the few giant-norm tokens (deep layers).
     * Same weights flow through the C-stage/Procrustes coordinates since they
     * are computed from these scaled rows. Held-out eval stays unscaled. */
    if (p.w_align > 0.0) {
        double w = p.w_align > 1.0 ? 1.0 : p.w_align;
        double *tn = (double *)malloc((size_t)n_train * sizeof(double));
        if (!tn) goto cleanup0;
        double mean_tn = 0;
        for (int s = 0; s < n_train; s++) {
            const float *td = Ts + (size_t)s * d;
            double q = 0; for (int j = 0; j < d; j++) q += (double)td[j] * (double)td[j];
            tn[s] = sqrt(q);
            mean_tn += tn[s];
        }
        mean_tn /= (double)n_train;
        for (int s = 0; s < n_train; s++) {
            double a = (1.0 - w) + w * (mean_tn / (tn[s] > 1e-12 ? tn[s] : 1e-12));
            double sa = sqrt(a);
            float *xd = Xs + (size_t)s * d, *td = Ts + (size_t)s * d;
            for (int j = 0; j < d; j++) { xd[j] = (float)(sa * xd[j]); td[j] = (float)(sa * td[j]); }
        }
        free(tn);
    }

    /* optional Q-metric whitening of the centered targets */
    if (p.alpha_router > 0.0 && p.Wr) {
        if (qmet_build(&qm, p.Wr, n_exp, d, p.seed ^ 0x51AEDULL)) goto cleanup0;
    }
    if (qm.ok) {
        double *tv = (double *)malloc((size_t)d * sizeof(double));
        double *tn = (double *)malloc((size_t)n_exp * sizeof(double));
        if (!tv || !tn) { free(tv); free(tn); goto cleanup0; }
        for (int s = 0; s < n_train; s++) {
            float *td = Ts + (size_t)s * d;
            for (int j = 0; j < d; j++) tv[j] = (double)td[j];
            qmet_apply(&qm, p.alpha_router, +1, tv, tn);
            for (int j = 0; j < d; j++) td[j] = (float)tv[j];
        }
        free(tv); free(tn);
    }

    if (n_pairs > 0) {
        double sw = sqrt(p.w_smooth);
        int w = n_train;
        for (int s = 1; s < n_train; s++) {
            if (!(idx_tr[s] == idx_tr[s-1] + 1 && (idx_tr[s] % p.chunk) != 0)) continue;
            const float *xa = Xs + (size_t)(s-1) * d, *xb = Xs + (size_t)s * d;
            const float *ta = Ts + (size_t)(s-1) * d, *tb = Ts + (size_t)s * d;
            float *xd = Xs + (size_t)w * d, *td = Ts + (size_t)w * d;
            for (int j = 0; j < d; j++) {
                xd[j] = (float)(sw * ((double)xb[j] - (double)xa[j]));
                td[j] = (float)(sw * ((double)tb[j] - (double)ta[j]));
            }
            w++;
        }
    }

    /* ---- shared low-rank basis (U, V) --------------------------------- */
    /* umode 0 (RRR): ridge LS map MT=(XᵀX+λI)⁻¹XᵀT, U = top-k eig of the
     * fitted covariance YfitᵀYfit, V = UᵀM.
     * umode 1 (PCA): U = top-k eig of the FEATURE covariance XᵀX (φ=ŷ →
     * output-space basis), V = Uᵀ — no d×d map fit at all; magnitudes come
     * entirely from the C-stage + Procrustes. */
    {
        double *G  = (double *)calloc((size_t)d * d, sizeof(double));
        double *R  = (double *)calloc((size_t)d * d, sizeof(double));
        double *evecs = (double *)malloc((size_t)k * d * sizeof(double));
        double *evals = (double *)malloc((size_t)k * sizeof(double));
        if (!G || !R || !evecs || !evals) { free(G); free(R); free(evecs); free(evals); goto cleanup0; }
        gram_ctx gc = { Xs, Ts, n_gram, d, G, R };
        parallel_for(d, p.threads, gram_worker, &gc);

        if (p.umode == 1) {
            /* PCA of the feature: mirror the (pre-ridge) Gram and eig it */
            double trF = 0;
            for (int i = 0; i < d; i++) {
                trF += G[(size_t)i * d + i];
                for (int j = 0; j < i; j++) G[(size_t)j * d + i] = G[(size_t)i * d + j];
            }
            sym_eig_topk(G, d, k, evecs, evals, p.eig_iters, p.seed ^ 0xE16ULL);
            double topE = 0; for (int i = 0; i < k; i++) topE += evals[i] > 0 ? evals[i] : 0;
            rep->energy_at_k = trF > 0 ? topE / trF : 0;
            free(G); free(R); R = NULL; G = NULL;
        } else {
            double *Gk = (double *)malloc((size_t)d * d * sizeof(double)); /* pristine copies: chol is */
            double *Rk = (double *)malloc((size_t)d * d * sizeof(double)); /* in-place + corrupts B on fail */
            if (!Gk || !Rk) { free(G); free(R); free(Gk); free(Rk); free(evecs); free(evals); goto cleanup0; }
            memcpy(Gk, G, (size_t)d * d * sizeof(double));
            memcpy(Rk, R, (size_t)d * d * sizeof(double));

            double lam_rel = p.lambda_rel > 0 ? p.lambda_rel : 1e-4;
            int solved = 0;
            for (int attempt = 0; attempt < 4 && !solved; attempt++) {
                loss_fixed(G, d, lam_rel, 1.0, p.seed);
                if (chol_solve_mt(G, d, R, d, p.threads) == 0) { solved = 1; break; }
                memcpy(G, Gk, (size_t)d * d * sizeof(double));
                memcpy(R, Rk, (size_t)d * d * sizeof(double));
                lam_rel *= 10.0;
                fprintf(stderr, "[rrr] chol failed, ridge -> %.2e\n", lam_rel);
            }
            free(Gk); free(Rk);
            free(G); G = NULL;
            if (!solved) { free(R); free(evecs); free(evals); goto cleanup0; }
            /* R now holds MT = [d_in×d_out] */

            float *Yfit = (float *)malloc((size_t)n_train * d * sizeof(float));
            double *F = (double *)calloc((size_t)d * d, sizeof(double));
            if (!Yfit || !F) { free(R); free(Yfit); free(F); free(evecs); free(evals); goto cleanup0; }
            fit_ctx fc = { Xs, R, Yfit, n_train, d };
            parallel_for(n_train, p.threads, fit_worker, &fc);
            fcov_ctx xc = { Yfit, n_train, d, F };
            parallel_for(d, p.threads, fcov_worker, &xc);
            double trF = 0;
            for (int i = 0; i < d; i++) {
                trF += F[(size_t)i * d + i];
                for (int j = 0; j < i; j++) F[(size_t)j * d + i] = F[(size_t)i * d + j];  /* mirror for eig matvec */
            }
            sym_eig_topk(F, d, k, evecs, evals, p.eig_iters, p.seed ^ 0xE16ULL);
            free(F);
            free(Yfit);
            double topE = 0; for (int i = 0; i < k; i++) topE += evals[i] > 0 ? evals[i] : 0;
            rep->energy_at_k = trF > 0 ? topE / trF : 0;
        }

        /* ---- z arrays ------------------------------------------------ */
        z->d_model = d; z->n_exp = n_exp; z->d_l = k;
        z->U = (double *)calloc((size_t)d * k, sizeof(double));
        z->V = (double *)calloc((size_t)k * d, sizeof(double));
        z->C = (double *)calloc((size_t)n_exp * k, sizeof(double));
        z->b = (double *)calloc((size_t)d, sizeof(double));
        z->beta = (double *)calloc((size_t)n_exp, sizeof(double));
        z->delta = (double *)calloc((size_t)n_exp, sizeof(double));
        if (!z->U || !z->V || !z->C || !z->b || !z->beta || !z->delta) { free(R); free(evecs); free(evals); goto cleanup_z; }

        for (int kk = 0; kk < k; kk++)
            for (int j = 0; j < d; j++) z->U[(size_t)j * k + kk] = evecs[(size_t)kk * d + j];
        if (p.umode == 1) {
            /* V = Uᵀ: latent v = coordinates of the feature in its own basis */
            for (int kk = 0; kk < k; kk++)
                for (int i = 0; i < d; i++) z->V[(size_t)kk * d + i] = evecs[(size_t)kk * d + i];
        } else {
            /* V = Uᵀ·M  (M row o, col i = MT[i][o]) */
            for (int kk = 0; kk < k; kk++) {
                const double *u = evecs + (size_t)kk * d;
                double *vr = z->V + (size_t)kk * d;
                for (int i = 0; i < d; i++) {
                    const double *mi = R + (size_t)i * d;
                    double s = 0; for (int o = 0; o < d; o++) s += u[o] * mi[o];
                    vr[i] = s;
                }
            }
        }
        free(R);

        /* ---- latent coords on train: v=V·x, g=Uᵀ·Tc ------------------ */
        double *v = (double *)malloc((size_t)n_train * k * sizeof(double));
        double *g = (double *)malloc((size_t)n_train * k * sizeof(double));
        if (!v || !g) { free(v); free(g); free(evecs); free(evals); goto cleanup_z; }
        for (int s = 0; s < n_train; s++) {
            const float *x = Xs + (size_t)s * d;
            const float *tc = Ts + (size_t)s * d;
            for (int kk = 0; kk < k; kk++) {
                const double *vr = z->V + (size_t)kk * d;
                const double *u  = evecs + (size_t)kk * d;
                double sv = 0, sg = 0;
                for (int j = 0; j < d; j++) { sv += vr[j] * (double)x[j]; sg += u[j] * (double)tc[j]; }
                v[(size_t)s * k + kk] = sv;
                g[(size_t)s * k + kk] = sg;
            }
        }
        free(evecs);

        /* ---- C-stage: per coordinate, 256×256 co-firing LS, prior 1/topk */
        {
            double *A = (double *)malloc((size_t)n_exp * n_exp * sizeof(double));
            double *rhs = (double *)malloc((size_t)n_exp * sizeof(double));
            if (!A || !rhs) { free(A); free(rhs); free(v); free(g); free(evals); goto cleanup_z; }
            const double prior = 1.0 / (double)topk;
            for (int kk = 0; kk < k; kk++) {
                memset(A, 0, (size_t)n_exp * n_exp * sizeof(double));
                memset(rhs, 0, (size_t)n_exp * sizeof(double));
                for (int s = 0; s < n_train; s++) {
                    double vk = v[(size_t)s * k + kk];
                    if (vk == 0.0) continue;
                    double v2 = vk * vk, vg = vk * g[(size_t)s * k + kk];
                    const float *rt = route + (size_t)idx_tr[s] * topk;
                    int ids[16], nid = 0;
                    for (int e = 0; e < topk && e < 16; e++) {
                        int id = (int)rt[e];
                        if (id >= 0 && id < n_exp) ids[nid++] = id;
                    }
                    for (int a = 0; a < nid; a++) {
                        rhs[ids[a]] += vg;
                        for (int b2 = 0; b2 < nid; b2++) A[(size_t)ids[a] * n_exp + ids[b2]] += v2;
                    }
                }
                double md = 0; for (int e = 0; e < n_exp; e++) md += A[(size_t)e * n_exp + e];
                md /= (double)n_exp;
                double lamc = (p.lam_c_rel > 0 ? p.lam_c_rel : 1e-2) * (md > 0 ? md : 1.0);
                for (int e = 0; e < n_exp; e++) {
                    A[(size_t)e * n_exp + e] += lamc;
                    rhs[e] += lamc * prior;              /* shrink toward C=1/topk */
                }
                if (chol_solve_spd(A, n_exp, rhs, 1) != 0) {
                    for (int e = 0; e < n_exp; e++) z->C[(size_t)e * k + kk] = prior;  /* degenerate: neutral */
                    continue;
                }
                for (int e = 0; e < n_exp; e++) z->C[(size_t)e * k + kk] = rhs[e];
            }
            free(A); free(rhs);
        }

        /* ---- Procrustes polish in coord space: g ≈ s·Ω·f -------------- */
        if (p.procrustes) {
            double *f = (double *)malloc((size_t)n_train * k * sizeof(double));
            double *W = (double *)calloc((size_t)k * k, sizeof(double));
            if (f && W) {
                double fnorm2 = 0;
                for (int s = 0; s < n_train; s++) {
                    const float *rt = route + (size_t)idx_tr[s] * topk;
                    for (int kk = 0; kk < k; kk++) {
                        double sc = 0;
                        for (int e = 0; e < topk; e++) {
                            int id = (int)rt[e];
                            if (id >= 0 && id < n_exp) sc += z->C[(size_t)id * k + kk];
                        }
                        double fv = sc * v[(size_t)s * k + kk];
                        f[(size_t)s * k + kk] = fv;
                        fnorm2 += fv * fv;
                    }
                }
                for (int s = 0; s < n_train; s++)
                    for (int a = 0; a < k; a++) {
                        double ga = g[(size_t)s * k + a];
                        if (ga == 0.0) continue;
                        for (int b2 = 0; b2 < k; b2++) W[(size_t)a * k + b2] += ga * f[(size_t)s * k + b2];
                    }
                /* SVD of W (k×k) via eig of WᵀW */
                double *WtW = (double *)calloc((size_t)k * k, sizeof(double));
                double *Qv = (double *)malloc((size_t)k * k * sizeof(double));
                double *sv2 = (double *)malloc((size_t)k * sizeof(double));
                double *P = (double *)malloc((size_t)k * k * sizeof(double));
                double *Om = (double *)malloc((size_t)k * k * sizeof(double));
                if (WtW && Qv && sv2 && P && Om) {
                    for (int a = 0; a < k; a++)
                        for (int b2 = 0; b2 < k; b2++) {
                            double s = 0;
                            for (int i = 0; i < k; i++) s += W[(size_t)i * k + a] * W[(size_t)i * k + b2];
                            WtW[(size_t)a * k + b2] = s;
                        }
                    sym_eig_topk(WtW, k, k, Qv, sv2, 0, p.seed ^ 0x9C0FULL);
                    double smax = sv2[0] > 0 ? sqrt(sv2[0]) : 0, ssum = 0;
                    for (int i = 0; i < k; i++) {
                        double si = sv2[i] > 0 ? sqrt(sv2[i]) : 0;
                        ssum += si;
                        const double *qi = Qv + (size_t)i * k;
                        if (si > 1e-12 * (smax > 0 ? smax : 1.0)) {
                            for (int m = 0; m < k; m++) {
                                double s = 0;
                                for (int b2 = 0; b2 < k; b2++) s += W[(size_t)m * k + b2] * qi[b2];
                                P[(size_t)m * k + i] = s / si;
                            }
                        } else {
                            for (int m = 0; m < k; m++) P[(size_t)m * k + i] = qi[m]; /* rank-deficient guard */
                        }
                    }
                    for (int m = 0; m < k; m++)
                        for (int c2 = 0; c2 < k; c2++) {
                            double s = 0;
                            for (int i = 0; i < k; i++) s += P[(size_t)m * k + i] * Qv[(size_t)i * k + c2];
                            Om[(size_t)m * k + c2] = s;
                        }
                    double scale = fnorm2 > 0 ? ssum / fnorm2 : 1.0;
                    /* U ← s·U·Ω */
                    double *urow = (double *)malloc((size_t)k * sizeof(double));
                    if (urow) {
                        for (int j = 0; j < d; j++) {
                            double *uj = z->U + (size_t)j * k;
                            for (int c2 = 0; c2 < k; c2++) {
                                double s = 0;
                                for (int m = 0; m < k; m++) s += uj[m] * Om[(size_t)m * k + c2];
                                urow[c2] = scale * s;
                            }
                            memcpy(uj, urow, (size_t)k * sizeof(double));
                        }
                        free(urow);
                    }
                }
                free(WtW); free(Qv); free(sv2); free(P); free(Om);
            }
            free(f); free(W);
        }
        free(v); free(g); free(evals);
    }

    /* ---- un-whiten U if Q-metric was used ---------------------------- */
    if (qm.ok) {
        double *col = (double *)malloc((size_t)d * sizeof(double));
        double *tn = (double *)malloc((size_t)n_exp * sizeof(double));
        if (col && tn) {
            for (int kk = 0; kk < k; kk++) {
                for (int j = 0; j < d; j++) col[j] = z->U[(size_t)j * k + kk];
                qmet_apply(&qm, p.alpha_router, -1, col, tn);
                for (int j = 0; j < d; j++) z->U[(size_t)j * k + kk] = col[j];
            }
        }
        free(col); free(tn);
    }

    /* ---- runtime b convention: kernel adds n_valid·b ------------------ */
    for (int j = 0; j < d; j++) z->b[j] = bfit[j] / (double)topk;

    /* ---- evaluate train + held-out ----------------------------------- */
    {
        double *corr = (double *)malloc((size_t)d * sizeof(double));
        double *vt = (double *)malloc((size_t)k * sizeof(double));
        if (corr && vt) {
            double c0, r0, c1, r1;
            agg_metrics(Yhat, Yref, NULL, idx_tr, n_train, d, &c0, &r0);
            rep->base_cos_tr = c0; rep->base_rel_tr = r0;
            /* corrected: per-token */
            double csum = 0, num = 0, den = 0; long cnt = 0;
            for (int s = 0; s < n_train; s++) {
                int t = idx_tr[s];
                token_corr(z->U, z->V, z->C, z->b, X + (size_t)t * d, route + (size_t)t * topk,
                           topk, d, k, n_exp, corr, vt);
                const float *h = Yhat + (size_t)t * d, *r = Yref + (size_t)t * d;
                double dot = 0, np = 0, nr = 0;
                for (int j = 0; j < d; j++) {
                    double pj = (double)h[j] + corr[j], rj = (double)r[j];
                    dot += pj * rj; np += pj * pj; nr += rj * rj;
                    double e = rj - pj; num += e * e; den += rj * rj;
                }
                if (np > 0 && nr > 0) { csum += dot / (sqrt(np) * sqrt(nr)); cnt++; }
            }
            rep->corr_cos_tr = cnt ? csum / cnt : 0;
            rep->corr_rel_tr = den > 0 ? sqrt(num / den) : 0;
            if (idx_te && n_test > 0) {
                agg_metrics(Yhat, Yref, NULL, idx_te, n_test, d, &c1, &r1);
                rep->base_cos_te = c1; rep->base_rel_te = r1;
                csum = num = den = 0; cnt = 0;
                for (int s = 0; s < n_test; s++) {
                    int t = idx_te[s];
                    token_corr(z->U, z->V, z->C, z->b, X + (size_t)t * d, route + (size_t)t * topk,
                               topk, d, k, n_exp, corr, vt);
                    const float *h = Yhat + (size_t)t * d, *r = Yref + (size_t)t * d;
                    double dot = 0, np = 0, nr = 0;
                    for (int j = 0; j < d; j++) {
                        double pj = (double)h[j] + corr[j], rj = (double)r[j];
                        dot += pj * rj; np += pj * pj; nr += rj * rj;
                        double e = rj - pj; num += e * e; den += rj * rj;
                    }
                    if (np > 0 && nr > 0) { csum += dot / (sqrt(np) * sqrt(nr)); cnt++; }
                }
                rep->corr_cos_te = cnt ? csum / cnt : 0;
                rep->corr_rel_te = den > 0 ? sqrt(num / den) : 0;
            }
        }
        free(corr); free(vt);
    }

    rc = 0;
    goto cleanup1;

cleanup_z:
    z_layer_free(z);
cleanup1:
cleanup0:
    if (qm.ok) qmet_free(&qm);
    free(Xs); free(Ts); free(bfit);
    return rc;
}

