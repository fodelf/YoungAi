/* ====================================================================== */
/* FULL HF DRIVER (excluded from the synthetic self-test build).          */
/* ====================================================================== */
#ifndef CALIBDIAG_TEST

typedef enum { TGT_DELTA = 0, TGT_OREF = 1 } solve_target_t;

/* ---- shared worker context (identical to calib_run.c) ------------------ */
typedef struct {
    const char *hf_dir;
    int    layer;
    int    e_begin, e_end;     /* this worker's expert band [e_begin,e_end)  */
    int    n_x;
    const float *x_f32;        /* [n_x*DM] subsampled Go inputs (float)      */
    double *o_ref;             /* [NEXP*n_x*DM] out (this band's slots)      */
    double *o_hat;             /* [NEXP*n_x*DM] out                          */
    double *delta;             /* [NEXP*n_x*DM] out                          */
    int    rc;                 /* 0 ok, nonzero on failure                   */
    int    done_experts;
} worker_ctx;

/* go1b round-trip a weight matrix [rows×cols] → w_hat (the 1-bit reconstruction). */
static void quant_dequant(const float *w, float *wh, int rows, int cols, unsigned char *scratch) {
    size_t rb = go1b_row_bytes(cols);
    go1b_quantize(w, scratch, rows, cols);
    for (int r = 0; r < rows; r++)
        go1b_dequantize_row(scratch + (size_t)r * rb, wh + (size_t)r * cols, cols);
}

static void *worker(void *arg) {
    worker_ctx *c = (worker_ctx *)arg;
    c->rc = 1;
    hf_db *db = hf_open(c->hf_dir);
    if (!db) { fprintf(stderr, "[worker] hf_open failed\n"); return NULL; }

    /* reusable per-thread buffers (1-bit reconstructions + scratch + temps) */
    float *gh = malloc((size_t)DF * DM * sizeof(float));
    float *uh = malloc((size_t)DF * DM * sizeof(float));
    float *dh = malloc((size_t)DM * DF * sizeof(float));
    size_t scratch_n = (size_t)DF * go1b_row_bytes(DM);
    size_t s2 = (size_t)DM * go1b_row_bytes(DF);
    if (s2 > scratch_n) scratch_n = s2;
    unsigned char *scratch = malloc(scratch_n);
    float *oref_t = malloc((size_t)DM * sizeof(float));
    float *ohat_t = malloc((size_t)DM * sizeof(float));
    if (!gh || !uh || !dh || !scratch || !oref_t || !ohat_t) { fprintf(stderr, "[worker] OOM buffers\n"); goto done; }

    for (int e = c->e_begin; e < c->e_end; e++) {
        char nm[256];
        int64_t n;
        /* HF naming: w1=gate w3=up [DF×DM], w2=down [DM×DF]; FP8 dequant inside hf_read. */
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1.weight", c->layer, e);
        float *gate = hf_read_f32(db, nm, &n);
        if (!gate || n != (int64_t)DF * DM) { fprintf(stderr, "[worker] read %s n=%lld\n", nm, (long long)(gate?n:-1)); free(gate); goto done; }
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3.weight", c->layer, e);
        float *up = hf_read_f32(db, nm, &n);
        if (!up || n != (int64_t)DF * DM) { fprintf(stderr, "[worker] read up\n"); free(gate); free(up); goto done; }
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2.weight", c->layer, e);
        float *down = hf_read_f32(db, nm, &n);
        if (!down || n != (int64_t)DM * DF) { fprintf(stderr, "[worker] read down\n"); free(gate); free(up); free(down); goto done; }

        /* strict-1-bit reconstruction of this expert's three matrices */
        quant_dequant(gate, gh, DF, DM, scratch);
        quant_dequant(up,   uh, DF, DM, scratch);
        quant_dequant(down, dh, DM, DF, scratch);

        for (int i = 0; i < c->n_x; i++) {
            const float *x = c->x_f32 + (size_t)i * DM;
            expert_forward_f32(x, gate, up, down, oref_t, DM, DF);
            expert_forward_f32(x, gh,   uh,   dh,   ohat_t, DM, DF);
            size_t base = ((size_t)e * c->n_x + i) * DM;
            for (int j = 0; j < DM; j++) {
                double r = (double)oref_t[j], h = (double)ohat_t[j];
                c->o_ref[base + j] = r;
                c->o_hat[base + j] = h;
                c->delta[base + j] = r - h;
            }
        }
        free(gate); free(up); free(down);
        c->done_experts++;
    }
    c->rc = 0;
done:
    free(gh); free(uh); free(dh); free(scratch); free(oref_t); free(ohat_t);
    hf_close(db);
    return NULL;
}

/* baseline (no correction) metrics: rel-L2 = ‖Δo‖/‖o_ref‖, mean cosine(ô,o_ref). */
static void baseline_metrics(const double *o_ref, const double *o_hat, const double *delta,
                             int n_exp, int n_x, int d_model, double *rel_l2, double *cos_mean) {
    double num = 0, den = 0, csum = 0; long cnt = 0;
    for (long ei = 0; ei < (long)n_exp * n_x; ei++) {
        const double *r = o_ref + (size_t)ei * d_model;
        const double *h = o_hat + (size_t)ei * d_model;
        const double *d = delta + (size_t)ei * d_model;
        double dot = 0, nr = 0, nh = 0, dd = 0, rr = 0;
        for (int j = 0; j < d_model; j++) { dot += r[j]*h[j]; nr += r[j]*r[j]; nh += h[j]*h[j]; dd += d[j]*d[j]; rr += r[j]*r[j]; }
        num += dd; den += rr;
        if (nr > 0 && nh > 0) { csum += dot / (sqrt(nr) * sqrt(nh)); cnt++; }
    }
    *rel_l2 = den > 0 ? sqrt(num / den) : 0;
    *cos_mean = cnt ? csum / cnt : 0;
}

int main(int argc, char **argv) {
    const char *hf_dir = "hf/DeepSeek-V4-Flash-Base";
    const char *cap_dir = "/private/tmp/m1_ds4/cap_m1";
    const char *layers_s = "0,4,8";
    int n_x = 64, n_threads = 6, max_rank = 64;
    double energy = 0.95, lambda = -1.0;
    solve_target_t target = TGT_DELTA;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hf") && i+1<argc) hf_dir = argv[++i];
        else if (!strcmp(argv[i], "--cap") && i+1<argc) cap_dir = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i+1<argc) layers_s = argv[++i];
        else if (!strcmp(argv[i], "--nx") && i+1<argc) n_x = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1<argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--energy") && i+1<argc) energy = atof(argv[++i]);
        else if (!strcmp(argv[i], "--maxrank") && i+1<argc) max_rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i+1<argc) lambda = atof(argv[++i]);
        else if (!strcmp(argv[i], "--target") && i+1<argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "delta")) target = TGT_DELTA;
            else if (!strcmp(v, "oref")) target = TGT_OREF;
            else { fprintf(stderr, "--target must be delta|oref (got %s)\n", v); return 2; }
        }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > NEXP) n_threads = NEXP;
    int kmaxc = max_rank; if (kmaxc > DM) kmaxc = DM; if (kmaxc < 1) kmaxc = 1;
    double reg_lambda = (lambda >= 0.0) ? lambda : 1e-4;
    const char *target_s = (target == TGT_OREF) ? "oref(direct-gen)" : "delta(1-bit residual)";

    printf("calib_diag: hf=%s cap=%s layers=%s n_x=%d threads=%d energy=%.3f maxrank=%d target=%s\n",
           hf_dir, cap_dir, layers_s, n_x, n_threads, energy, max_rank, target_s);
    printf("deep diagnostics: out-PCA(o_ref,o_hat) | ORACLE Δo rank-d | x->Δo linear R^2 | hv-solve(%s)\n\n",
           target == TGT_OREF ? "GENERATE o_ref from x, no 1-bit base" : "correct the 1-bit residual");

    /* parse layer list */
    int layers[64], nL = 0;
    { char buf[256]; strncpy(buf, layers_s, sizeof buf - 1); buf[sizeof buf -1]=0;
      for (char *t = strtok(buf, ","); t && nL < 64; t = strtok(NULL, ",")) layers[nL++] = atoi(t); }

    /* the hv_solve/hv_fidelity table headers from calib_run; corr_* columns now
     * reflect the active --target (1-bit residual correction, or direct gen). */
    printf("%-4s | %-10s | %-5s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
           "L", "tot_energy", "d_l", "cumE@d_l", "base_relL2", "corr_relL2", "base_cos", "corr_cos");
    printf("-----+------------+-------+-----------+-----------+------------+-----------+----------\n");

    size_t cell = (size_t)NEXP * n_x * DM;
    double *o_ref = malloc(cell * sizeof(double));
    double *o_hat = malloc(cell * sizeof(double));
    double *delta = malloc(cell * sizeof(double));
    float  *x_f32 = malloc((size_t)n_x * DM * sizeof(float));
    double *x_d   = malloc((size_t)n_x * DM * sizeof(double));
    /* scratch for the new metrics: one [DM*DM] covariance/Gram buffer (reused
     * across o_ref/o_hat/Δo and the regression) + the top-kmax eigenpairs. */
    double *work_dd = malloc((size_t)DM * DM * sizeof(double));
    double *evecs   = malloc((size_t)kmaxc * DM * sizeof(double));
    double *evals   = malloc((size_t)kmaxc * sizeof(double));
    double *mu      = malloc((size_t)DM * sizeof(double));
    if (!o_ref || !o_hat || !delta || !x_f32 || !x_d || !work_dd || !evecs || !evals || !mu) {
        fprintf(stderr, "OOM main arrays (~%.1f GiB for o_ref/o_hat/Δo + %.1f GiB scratch)\n",
                3.0*cell*8/1e9, (double)DM*DM*8/1e9); return 1;
    }

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
        int per = (NEXP + n_threads - 1) / n_threads, nt = 0;
        for (int e0 = 0; e0 < NEXP; e0 += per) {
            worker_ctx *c = &ctx[nt];
            c->hf_dir = hf_dir; c->layer = L; c->e_begin = e0; c->e_end = (e0+per<NEXP)?e0+per:NEXP;
            c->n_x = n_x; c->x_f32 = x_f32; c->o_ref = o_ref; c->o_hat = o_hat; c->delta = delta;
            c->rc = 1; c->done_experts = 0;
            if (pthread_create(&th[nt], NULL, worker, c) != 0) { fprintf(stderr, "pthread_create\n"); return 1; }
            nt++;
        }
        int bad = 0, did = 0;
        for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); bad += ctx[t].rc; did += ctx[t].done_experts; }
        if (bad) { fprintf(stderr, "L%d: %d worker(s) failed (experts done=%d) — skipping\n", L, bad, did); continue; }

        long N = (long)NEXP * n_x;

        /* ---- existing Δo table: baseline + hv_solve + hv_fidelity --------- */
        double b_rel, b_cos;
        baseline_metrics(o_ref, o_hat, delta, NEXP, n_x, DM, &b_rel, &b_cos);

        hv_params p; hv_params_default(&p);
        p.energy_threshold = energy; p.max_rank = max_rank;
        if (lambda >= 0) p.lambda = lambda;

        /* metric 5: --target switches WHAT hv_solve reconstructs. delta = correct
         * the 1-bit residual (o^corr = ô + corr). oref = DIRECT GENERATION:
         * feed o_ref as the residual target and o_hat=NULL to hv_fidelity, so the
         * fit reconstructs o_ref straight from x with no 1-bit base. */
        const double *solve_target = (target == TGT_OREF) ? o_ref : delta;
        const double *fid_ohat     = (target == TGT_OREF) ? NULL  : o_hat;

        z_layer z; z_spectrum spec; z_fidelity fid;
        memset(&z, 0, sizeof z); memset(&spec, 0, sizeof spec);
        int rc = hv_solve(solve_target, x_d, o_ref, NULL, NULL, NEXP, n_x, DM, 0, &p, &z, &spec);
        if (rc) { fprintf(stderr, "L%d: hv_solve rc=%d\n", L, rc); continue; }
        hv_fidelity(&z, solve_target, x_d, fid_ohat, NEXP, n_x, DM, &fid);

        double cumE = 0; for (int k = 0; k < spec.d_l && k < spec.n; k++) cumE += spec.evals[k];
        double cumFrac = spec.total_energy > 0 ? cumE / spec.total_energy : 0;

        printf("%-4d | %10.3e | %5d | %9.4f | %9.4f | %10.4f | %9.5f | %8.5f\n",
               L, spec.total_energy, spec.d_l, cumFrac, b_rel, fid.rel_l2, b_cos, fid.cosine_mean);
        printf("      eig[0..7]:");
        for (int k = 0; k < 8 && k < spec.n; k++) printf(" %.3e", spec.evals[k]);
        printf("\n");
        z_layer_free(&z); z_spectrum_free(&spec);

        /* ---- NEW deep diagnostics --------------------------------------- */
        double cumE_r[4], cumE_h[4], cumE_d[4], tr_r, tr_h, tr_d;

        /* metric 1: is the Go expert OUTPUT itself low-dim? */
        int er_ref = effrank_compute(o_ref, N, DM, kmaxc, energy, DIAG_SEED, DIAG_EIG_ITERS,
                                     work_dd, mu, evecs, evals, cumE_r, &tr_r);
        printf("   [o_ref] out eff-rank(cumE>=%.2f)=", energy);
        if (er_ref < 0) printf(">%d", kmaxc); else printf("%d", er_ref);
        printf("  cumE@8/16/32/64="); pv(cumE_r[0]); printf("/"); pv(cumE_r[1]); printf("/"); pv(cumE_r[2]); printf("/"); pv(cumE_r[3]);
        printf("  trace=%.3e\n", tr_r);

        /* metric 2: is the 1-bit OUTPUT itself low-dim? */
        int er_hat = effrank_compute(o_hat, N, DM, kmaxc, energy, DIAG_SEED, DIAG_EIG_ITERS,
                                     work_dd, mu, evecs, evals, cumE_h, &tr_h);
        printf("   [o_hat] out eff-rank(cumE>=%.2f)=", energy);
        if (er_hat < 0) printf(">%d", kmaxc); else printf("%d", er_hat);
        printf("  cumE@8/16/32/64="); pv(cumE_h[0]); printf("/"); pv(cumE_h[1]); printf("/"); pv(cumE_h[2]); printf("/"); pv(cumE_h[3]);
        printf("  trace=%.3e\n", tr_h);

        /* Δo eigenbasis (also reports Δo's own eff-rank for context), then
         * metrics 3 (oracle) + 4 (x->Δo R^2) reuse mu/evecs/trace. */
        int er_d = effrank_compute(delta, N, DM, kmaxc, energy, DIAG_SEED, DIAG_EIG_ITERS,
                                   work_dd, mu, evecs, evals, cumE_d, &tr_d);
        printf("   [Δo]    eff-rank(cumE>=%.2f)=", energy);
        if (er_d < 0) printf(">%d", kmaxc); else printf("%d", er_d);
        printf("  cumE@8/16/32/64="); pv(cumE_d[0]); printf("/"); pv(cumE_d[1]); printf("/"); pv(cumE_d[2]); printf("/"); pv(cumE_d[3]);
        printf("  trace=%.3e\n", tr_d);

        double oracle[4], meanR2; int d16, sok;
        compute_delta_metrics(delta, x_d, NEXP, n_x, DM, mu, evecs, kmaxc, tr_d, reg_lambda,
                              work_dd, oracle, &meanR2, &d16, &sok);
        printf("   [Δo] ORACLE rank-d recon rel-L2 (perfect coords): d8="); pv(oracle[0]);
        printf(" d16="); pv(oracle[1]); printf(" d32="); pv(oracle[2]); printf(" d64="); pv(oracle[3]); printf("\n");
        printf("   [Δo] x->Δo LINEAR predictability: mean R^2 over top-%d coords = ", d16);
        if (sok) printf("%.4f\n", meanR2); else printf("n/a (regression solve failed)\n");
        printf("\n");
    }

    free(o_ref); free(o_hat); free(delta); free(x_f32); free(x_d);
    free(work_dd); free(evecs); free(evals); free(mu);
    printf("done.\n");
    return 0;
}

#else  /* CALIBDIAG_TEST */
/* ====================================================================== */
/* SYNTHETIC SELF-TEST — validates the NEW math only. No HF, no model.     */
/*   cc -DCALIBDIAG_TEST -O2 -std=c11 calib_diag.c hiddenvar_solve.c \      */
/*      linalg_small.c -o calib_diag_test -lm                              */
/* ====================================================================== */

/* deterministic PRNG (splitmix64) → Gaussian, for building synthetic data. */
static uint64_t tsm(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double tunit(uint64_t *s) { return (double)(tsm(s) >> 11) * (1.0 / 9007199254740992.0); }
static double tgauss(uint64_t *s) {
    double u1 = tunit(s), u2 = tunit(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
/* m unit-orthonormal rows of length n (modified Gram-Schmidt on Gaussians). */
static void orthonormal_rows(double *out, int m, int n, uint64_t *s) {
    for (int r = 0; r < m; r++) {
        double *vr = out + (size_t)r * n;
        for (int t = 0; t < n; t++) vr[t] = tgauss(s);
        for (int q = 0; q < r; q++) {
            const double *vq = out + (size_t)q * n;
            double d = 0.0;
            for (int t = 0; t < n; t++) d += vr[t] * vq[t];
            for (int t = 0; t < n; t++) vr[t] -= d * vq[t];
        }
        double nr = 0.0;
        for (int t = 0; t < n; t++) nr += vr[t] * vr[t];
        nr = sqrt(nr);
        if (nr > 0.0) for (int t = 0; t < n; t++) vr[t] /= nr;
    }
}

int main(void) {
    /* small dims so the d∈{8,16,32} probes are meaningful (d=64 → n/a). */
    const int d = 48, n_exp = 6, n_x = 150;
    const int r_ref = 6, r_hat = 4, r_delta = 20;  /* true ranks we will recover */
    const int maxrank = 32;
    const double energy = 0.95;
    uint64_t s = 0xC0FFEE1234567890ULL;
    long N = (long)n_exp * n_x;

    double *o_ref = malloc((size_t)N * d * sizeof(double));
    double *o_hat = malloc((size_t)N * d * sizeof(double));
    double *delta = malloc((size_t)N * d * sizeof(double));
    double *x     = malloc((size_t)n_x * d * sizeof(double));

    /* x: Gaussian, then CENTERED (x̄=0 → Δo below is mean-zero, so the oracle
     * denominator equals the centered energy and rel-L2 = sqrt(1−cumE)) and
     * WHITENED ((1/n_x)Σ x xᵀ = I). Whitening + an orthonormal Qd gives Δo's
     * r_delta latent coordinates EQUAL variance, so Δo's spectrum is flat over
     * exactly r_delta dims — the regime where the oracle residual decreases
     * smoothly with d (rather than collapsing to ~0 well before d=r_delta). */
    for (size_t t = 0; t < (size_t)n_x * d; t++) x[t] = tgauss(&s);
    { double mu[64]; for (int j = 0; j < d; j++) mu[j] = 0.0;
      for (int i = 0; i < n_x; i++) for (int j = 0; j < d; j++) mu[j] += x[(size_t)i*d+j];
      for (int j = 0; j < d; j++) mu[j] /= (double)n_x;
      for (int i = 0; i < n_x; i++) for (int j = 0; j < d; j++) x[(size_t)i*d+j] -= mu[j]; }
    {   /* whiten: S=(1/n_x)Σ x xᵀ; W=S^{-1/2}=Σ_m λ_m^{-1/2} q_m q_mᵀ; x <- W x */
        double *S = calloc((size_t)d * d, sizeof(double));
        for (int i = 0; i < n_x; i++)
            for (int a = 0; a < d; a++)
                for (int c = 0; c < d; c++)
                    S[(size_t)a*d+c] += x[(size_t)i*d+a] * x[(size_t)i*d+c];
        for (size_t t = 0; t < (size_t)d*d; t++) S[t] /= (double)n_x;
        double *qv = malloc((size_t)d * d * sizeof(double));
        double *lv = malloc((size_t)d * sizeof(double));
        sym_eig_topk(S, d, d, qv, lv, 400, 0x5EEDULL);
        double *W = calloc((size_t)d * d, sizeof(double));
        for (int mEig = 0; mEig < d; mEig++) {
            double lm = lv[mEig] > 1e-12 ? lv[mEig] : 1e-12;
            double inv = 1.0 / sqrt(lm);
            const double *qm = qv + (size_t)mEig * d;
            for (int a = 0; a < d; a++)
                for (int c = 0; c < d; c++)
                    W[(size_t)a*d+c] += inv * qm[a] * qm[c];
        }
        double *xw = malloc((size_t)n_x * d * sizeof(double));
        for (int i = 0; i < n_x; i++)
            for (int a = 0; a < d; a++) {
                double acc = 0.0;
                for (int c = 0; c < d; c++) acc += W[(size_t)a*d+c] * x[(size_t)i*d+c];
                xw[(size_t)i*d+a] = acc;
            }
        memcpy(x, xw, (size_t)n_x * d * sizeof(double));
        free(S); free(qv); free(lv); free(W); free(xw);
    }

    /* o_ref: rank-r_ref output (orthonormal basis B_ref) with a DECREASING but
     * comparable per-coordinate variance schedule (so it takes all r_ref dims to
     * reach 95% energy → effective rank == r_ref) + tiny full-dim noise. */
    double *Bref = malloc((size_t)r_ref * d * sizeof(double));   /* rows = basis cols */
    orthonormal_rows(Bref, r_ref, d, &s);
    for (int e = 0; e < n_exp; e++)
        for (int i = 0; i < n_x; i++) {
            double coord[64];
            for (int k = 0; k < r_ref; k++) {
                double v = 1.0 - 0.4 * (double)k / (double)(r_ref - 1); /* 1.0 → 0.6 */
                coord[k] = sqrt(v) * tgauss(&s);
            }
            double *o = o_ref + ((size_t)e * n_x + i) * d;
            for (int j = 0; j < d; j++) {
                double acc = 0.0;
                for (int k = 0; k < r_ref; k++) acc += Bref[(size_t)k*d+j] * coord[k];
                o[j] = acc + 1e-3 * tgauss(&s);
            }
        }

    /* o_hat: independent rank-r_hat block, same construction (validates metric 2). */
    double *Bhat = malloc((size_t)r_hat * d * sizeof(double));
    orthonormal_rows(Bhat, r_hat, d, &s);
    for (int e = 0; e < n_exp; e++)
        for (int i = 0; i < n_x; i++) {
            double coord[64];
            for (int k = 0; k < r_hat; k++) {
                double v = 1.0 - 0.4 * (double)k / (double)(r_hat - 1);
                coord[k] = sqrt(v) * tgauss(&s);
            }
            double *o = o_hat + ((size_t)e * n_x + i) * d;
            for (int j = 0; j < d; j++) {
                double acc = 0.0;
                for (int k = 0; k < r_hat; k++) acc += Bhat[(size_t)k*d+j] * coord[k];
                o[j] = acc + 1e-3 * tgauss(&s);
            }
        }

    /* Δo: PERFECTLY LINEAR in x and SHARED across experts (no per-expert / no
     * noise) → Δo_{e,i} = Dd · (Qd · x_i), rank r_delta. This makes:
     *   - metric 4 R^2 ≈ 1 (a single shared linear map recovers it exactly), and
     *   - metric 3 oracle residual strictly decrease in d, hitting ~0 once d≥r_delta. */
    double *Dd = malloc((size_t)r_delta * d * sizeof(double));   /* rows = output basis cols */
    double *Qd = malloc((size_t)r_delta * d * sizeof(double));   /* x → r_delta map (rows)   */
    orthonormal_rows(Dd, r_delta, d, &s);
    orthonormal_rows(Qd, r_delta, d, &s);   /* orthonormal rows + whitened x ⇒ flat latent variance */
    for (int i = 0; i < n_x; i++) {
        const double *xi = x + (size_t)i * d;
        double y[64];
        for (int k = 0; k < r_delta; k++) {
            double acc = 0.0;
            for (int j = 0; j < d; j++) acc += Qd[(size_t)k*d+j] * xi[j];
            y[k] = acc;
        }
        double row[64];
        for (int j = 0; j < d; j++) {
            double acc = 0.0;
            for (int k = 0; k < r_delta; k++) acc += Dd[(size_t)k*d+j] * y[k];
            row[j] = acc;
        }
        for (int e = 0; e < n_exp; e++) {              /* identical across experts */
            double *dv = delta + ((size_t)e * n_x + i) * d;
            for (int j = 0; j < d; j++) dv[j] = row[j];
        }
    }

    /* ---- run the new metrics ---- */
    int kmax = maxrank; if (kmax > d) kmax = d;
    double *Sigma = malloc((size_t)d * d * sizeof(double));
    double *evecs = malloc((size_t)kmax * d * sizeof(double));
    double *evals = malloc((size_t)kmax * sizeof(double));
    double *mu    = malloc((size_t)d * sizeof(double));
    double cumE_r[4], cumE_h[4], cumE_d[4], tr_r, tr_h, tr_d;

    int er_ref = effrank_compute(o_ref, N, d, kmax, energy, DIAG_SEED, DIAG_EIG_ITERS,
                                 Sigma, mu, evecs, evals, cumE_r, &tr_r);
    int er_hat = effrank_compute(o_hat, N, d, kmax, energy, DIAG_SEED, DIAG_EIG_ITERS,
                                 Sigma, mu, evecs, evals, cumE_h, &tr_h);
    /* Δo last so mu/evecs hold Δo's eigenbasis for metrics 3/4. */
    int er_d = effrank_compute(delta, N, d, kmax, energy, DIAG_SEED, DIAG_EIG_ITERS,
                               Sigma, mu, evecs, evals, cumE_d, &tr_d);
    double oracle[4], meanR2; int d16, sok;
    compute_delta_metrics(delta, x, n_exp, n_x, d, mu, evecs, kmax, tr_d, 1e-6,
                          Sigma, oracle, &meanR2, &d16, &sok);

    printf("== calib_diag synthetic self-test (d=%d n_exp=%d n_x=%d) ==\n", d, n_exp, n_x);
    printf("\n-- metric 1: o_ref out PCA (true rank %d) --\n", r_ref);
    printf("   eff-rank(cumE>=%.2f) = %d   cumE@8/16/32 = %.4f / %.4f / %.4f   trace=%.3e\n",
           energy, er_ref, cumE_r[0], cumE_r[1], cumE_r[2], tr_r);
    printf("\n-- metric 2: o_hat out PCA (true rank %d) --\n", r_hat);
    printf("   eff-rank(cumE>=%.2f) = %d   cumE@8/16/32 = %.4f / %.4f / %.4f   trace=%.3e\n",
           energy, er_hat, cumE_h[0], cumE_h[1], cumE_h[2], tr_h);
    printf("\n-- metric 3: ORACLE rank-d Δo reconstruction (true rank %d) --\n", r_delta);
    printf("   Δo eff-rank=%d   rel-L2: d8=%.4f  d16=%.4f  d32=%.4f  d64=", er_d,
           oracle[0], oracle[1], oracle[2]); pv(oracle[3]); printf("\n");
    printf("\n-- metric 4: x->Δo linear predictability (Δo built linear in x) --\n");
    printf("   mean R^2 over top-%d coords = %.6f  (solve_ok=%d)\n", d16, meanR2, sok);

    /* ---- assertions ---- */
    int a1 = (er_ref == r_ref);
    int a2 = (er_hat == r_hat);
    int a3 = (oracle[0] > oracle[1]) && (oracle[1] > oracle[2]) && (oracle[2] < 0.05);
    int a4 = sok && (meanR2 > 0.99);

    printf("\n== Assertions ==\n");
    printf("  (1) o_ref eff-rank == true  : %s  (%d vs %d)\n", a1?"PASS":"FAIL", er_ref, r_ref);
    printf("  (2) o_hat eff-rank == true  : %s  (%d vs %d)\n", a2?"PASS":"FAIL", er_hat, r_hat);
    printf("  (3) oracle resid ↓ with d   : %s  (%.4f > %.4f > %.4f, d32<0.05)\n",
           a3?"PASS":"FAIL", oracle[0], oracle[1], oracle[2]);
    printf("  (4) x->Δo R^2 ≈ 1           : %s  (%.6f)\n", a4?"PASS":"FAIL", meanR2);

    int all = a1 && a2 && a3 && a4;
    printf("\nRESULT: %s\n", all ? "ALL PASS" : "FAIL");

    free(o_ref); free(o_hat); free(delta); free(x);
    free(Bref); free(Bhat); free(Dd); free(Qd);
    free(Sigma); free(evecs); free(evals); free(mu);
    return all ? 0 : 1;
}
#endif /* CALIBDIAG_TEST */
