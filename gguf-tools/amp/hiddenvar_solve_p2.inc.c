/* ================================================================== */
/* reconstruction fidelity (表 B)                                      */
/* ================================================================== */
void hv_fidelity(const z_layer *z, const double *delta_o, const double *x,
                 const double *o_hat, int n_exp, int n_x, int d_model,
                 z_fidelity *out) {
    const size_t NE = (size_t)n_exp, NX = (size_t)n_x, DM = (size_t)d_model;
    const size_t DL = (size_t)z->d_l;

    /* f_{i,k} = (V x_i)_k */
    double *f = (double *)malloc(NX * DL * sizeof(double));
    for (size_t i = 0; i < NX; i++) {
        const double *xi = x + i * DM;
        double *fr = f + i * DL;
        for (size_t k = 0; k < DL; k++) {
            const double *vk = z->V + k * DM;
            double sv = 0.0;
            for (size_t j = 0; j < DM; j++) sv += vk[j] * xi[j];
            fr[k] = sv;
        }
    }

    double num2 = 0.0, den2 = 0.0, cossum = 0.0;
    size_t cnt = 0;
    double *corr = (double *)malloc(DM * sizeof(double));
    for (size_t e = 0; e < NE; e++)
        for (size_t i = 0; i < NX; i++) {
            const double *fr = f + i * DL;
            /* corr_j = Σ_k U[j][k]·C[e][k]·f[i][k] + β_e + b_j */
            for (size_t j = 0; j < DM; j++) {
                double acc = 0.0;
                for (size_t k = 0; k < DL; k++)
                    acc += z->U[j * DL + k] * z->C[e * DL + k] * fr[k];
                corr[j] = acc + z->beta[e] + z->b[j];
            }
            const double *d = delta_o + (e * NX + i) * DM;
            const double *oh = o_hat ? (o_hat + (e * NX + i) * DM) : NULL;
            /* o^corr = ô + corr ; o_ref = ô + Δo ; (ô=0 if not provided). */
            double dn = 0, rn = 0, dot = 0, cn = 0;
            for (size_t j = 0; j < DM; j++) {
                double base = oh ? oh[j] : 0.0;
                double oc = base + corr[j];
                double orf = base + d[j];
                double df = oc - orf;
                dn += df * df; rn += orf * orf; dot += oc * orf; cn += oc * oc;
            }
            num2 += dn; den2 += rn;
            double dc = sqrt(cn) * sqrt(rn);
            if (dc > 0.0) { cossum += dot / dc; cnt++; }
        }

    out->rel_l2      = (den2 > 0.0) ? sqrt(num2 / den2) : 0.0;
    out->cosine_mean = (cnt > 0) ? cossum / (double)cnt : 1.0;
    out->n_exp = n_exp; out->n_x = n_x;
    free(f); free(corr);
}

/* ================================================================== */
/* flat-file (de)serialization of z^ℓ                                  */
/* ================================================================== */
/* Offline interchange format: 16-byte header {magic,d_model,n_exp,d_l} then
 * the six arrays as raw little-endian float64 in struct order.
 *
 * TODO(integration): emit as blk.%d.corr_{U,V,C,b,beta,delta} GGUF tensors
 * (SPEC.md §9 P2.1) — write into the same GGUF file as Θ_fix via the reused
 * write_full_gguf / KV writer; this flat file is only the solver-side dump. */
#define ZL_MAGIC 0x315A565Au   /* 'Z','V','Z','1' */

int z_layer_save(const z_layer *z, const char *path) {
    if (!z || !path) return 1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 1;
    uint32_t magic = ZL_MAGIC;
    int32_t hdr[3] = { z->d_model, z->n_exp, z->d_l };
    size_t DM = (size_t)z->d_model, DL = (size_t)z->d_l, NE = (size_t)z->n_exp;
    int ok = 1;
    ok &= (fwrite(&magic, sizeof(magic), 1, fp) == 1);
    ok &= (fwrite(hdr, sizeof(int32_t), 3, fp) == 3);
    ok &= (fwrite(z->U,    sizeof(double), DM * DL, fp) == DM * DL);
    ok &= (fwrite(z->V,    sizeof(double), DL * DM, fp) == DL * DM);
    ok &= (fwrite(z->C,    sizeof(double), NE * DL, fp) == NE * DL);
    ok &= (fwrite(z->b,    sizeof(double), DM,      fp) == DM);
    ok &= (fwrite(z->beta, sizeof(double), NE,      fp) == NE);
    ok &= (fwrite(z->delta,sizeof(double), NE,      fp) == NE);
    fclose(fp);
    return ok ? 0 : 1;
}

int z_layer_load(z_layer *z, const char *path) {
    if (!z || !path) return 1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 1;
    uint32_t magic = 0;
    int32_t hdr[3] = {0, 0, 0};
    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != ZL_MAGIC ||
        fread(hdr, sizeof(int32_t), 3, fp) != 3) { fclose(fp); return 1; }
    z->d_model = hdr[0]; z->n_exp = hdr[1]; z->d_l = hdr[2];
    size_t DM = (size_t)z->d_model, DL = (size_t)z->d_l, NE = (size_t)z->n_exp;
    z->U     = (double *)malloc(DM * DL * sizeof(double));
    z->V     = (double *)malloc(DL * DM * sizeof(double));
    z->C     = (double *)malloc(NE * DL * sizeof(double));
    z->b     = (double *)malloc(DM * sizeof(double));
    z->beta  = (double *)malloc(NE * sizeof(double));
    z->delta = (double *)malloc(NE * sizeof(double));
    int ok = z->U && z->V && z->C && z->b && z->beta && z->delta;
    ok = ok && (fread(z->U,    sizeof(double), DM * DL, fp) == DM * DL);
    ok = ok && (fread(z->V,    sizeof(double), DL * DM, fp) == DL * DM);
    ok = ok && (fread(z->C,    sizeof(double), NE * DL, fp) == NE * DL);
    ok = ok && (fread(z->b,    sizeof(double), DM,      fp) == DM);
    ok = ok && (fread(z->beta, sizeof(double), NE,      fp) == NE);
    ok = ok && (fread(z->delta,sizeof(double), NE,      fp) == NE);
    fclose(fp);
    if (!ok) { z_layer_free(z); return 1; }
    return 0;
}

void z_layer_free(z_layer *z) {
    if (!z) return;
    free(z->U); free(z->V); free(z->C); free(z->b); free(z->beta); free(z->delta);
    z->U = z->V = z->C = z->b = z->beta = z->delta = NULL;
}
void z_spectrum_free(z_spectrum *s) {
    if (!s) return;
    free(s->evals);
    s->evals = NULL;
}

/* ================================================================== */
/* SELF-TEST (synthetic; no model)                                     */
/*   cc -std=c99 -O2 -DHVSOLVE_TEST hiddenvar_solve.c linalg_small.c \  */
/*      -o a5test -lm                                                   */
/* ================================================================== */
#ifdef HVSOLVE_TEST

/* Linux -std=c11 严格模式下 math.h 不给 M_PI(Darwin 给) — 自测段自带兜底。 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* deterministic Gaussian via splitmix64 + Box-Muller (test data only). */
static double rnd_gauss(uint64_t *s) {
    double u1 = sm64_unit(s), u2 = sm64_unit(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Build m unit-orthonormal vectors of length n (rows of `out`, m×n) by
 * modified Gram-Schmidt on random vectors. */
static void rand_orthonormal_rows(double *out, int m, int n, uint64_t *s) {
    for (int r = 0; r < m; r++) {
        double *vr = out + (size_t)r * n;
        for (int t = 0; t < n; t++) vr[t] = rnd_gauss(s);
        for (int q = 0; q < r; q++) {                 /* subtract earlier rows */
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
    /* ---- synthetic KNOWN low-rank structure (SPEC.md §6 model class) ---- */
    const int d_model = 32, d0 = 4, n_exp = 8, n_x = 200;
    const size_t DM = d_model, D0 = d0, NE = n_exp, NX = n_x;
    uint64_t s = 0xBADC0FFEE0DDF00DULL;

    /* U0: d_model×d0 with ORTHONORMAL COLUMNS (store columns via rows-of-Uᵀ) */
    double *U0t = malloc(D0 * DM * sizeof(double));     /* d0×d_model rows */
    rand_orthonormal_rows(U0t, d0, d_model, &s);        /* row k = column k of U0 */
    /* V0: d0×d_model with ORTHONORMAL ROWS */
    double *V0 = malloc(D0 * DM * sizeof(double));
    rand_orthonormal_rows(V0, d0, d_model, &s);
    /* C0: n_exp×d0 distinct positive gains in [0.7,1.6] */
    double *C0 = malloc(NE * D0 * sizeof(double));
    for (size_t t = 0; t < NE * D0; t++) C0[t] = 0.7 + 0.9 * sm64_unit(&s);
    /* b0: small global bias */
    double *b0 = malloc(DM * sizeof(double));
    for (size_t j = 0; j < DM; j++) b0[j] = 0.1 * rnd_gauss(&s);

    /* x pool: random, then CENTER (x̄=0) and WHITEN (Σ_i x_ix_iᵀ = n_x·I) so
     * the residual PCA basis aligns with U0's column space up to a permutation
     * — i.e. the diagonal-C model can represent the data exactly (clean test). */
    double *x = malloc(NX * DM * sizeof(double));
    for (size_t t = 0; t < NX * DM; t++) x[t] = rnd_gauss(&s);
    {   /* center */
        double *mu = calloc(DM, sizeof(double));
        for (size_t i = 0; i < NX; i++) for (size_t j = 0; j < DM; j++) mu[j] += x[i*DM+j];
        for (size_t j = 0; j < DM; j++) mu[j] /= (double)NX;
        for (size_t i = 0; i < NX; i++) for (size_t j = 0; j < DM; j++) x[i*DM+j] -= mu[j];
        free(mu);
        /* whiten: S = (1/n_x)Σ x xᵀ ; W = S^{-1/2} = Σ_m λ_m^{-1/2} q_m q_mᵀ */
        double *S = calloc(DM * DM, sizeof(double));
        for (size_t i = 0; i < NX; i++)
            for (size_t a = 0; a < DM; a++)
                for (size_t c = 0; c < DM; c++)
                    S[a*DM+c] += x[i*DM+a] * x[i*DM+c];
        for (size_t t = 0; t < DM*DM; t++) S[t] /= (double)NX;
        double *qv = malloc(DM * DM * sizeof(double));
        double *lv = malloc(DM * sizeof(double));
        sym_eig_topk(S, d_model, d_model, qv, lv, 400, 0x5EEDULL);
        double *W = calloc(DM * DM, sizeof(double));
        for (size_t m = 0; m < DM; m++) {
            double lm = lv[m] > 1e-12 ? lv[m] : 1e-12;
            double inv = 1.0 / sqrt(lm);
            const double *qm = qv + m * DM;
            for (size_t a = 0; a < DM; a++)
                for (size_t c = 0; c < DM; c++)
                    W[a*DM+c] += inv * qm[a] * qm[c];
        }
        double *xw = malloc(NX * DM * sizeof(double));
        for (size_t i = 0; i < NX; i++)
            for (size_t a = 0; a < DM; a++) {
                double acc = 0.0;
                for (size_t c = 0; c < DM; c++) acc += W[a*DM+c] * x[i*DM+c];
                xw[i*DM+a] = acc;
            }
        memcpy(x, xw, NX * DM * sizeof(double));
        free(S); free(qv); free(lv); free(W); free(xw);
    }

    /* synthesize Δo_{e,i} = U0·diag(C0_e)·(V0 x_i) + b0 (+ tiny noise) */
    const double noise = 1e-3;
    double *delta_o = malloc(NE * NX * DM * sizeof(double));
    {
        double *Vx = malloc(D0 * sizeof(double));
        for (size_t e = 0; e < NE; e++)
            for (size_t i = 0; i < NX; i++) {
                const double *xi = x + i * DM;
                for (size_t k = 0; k < D0; k++) {       /* V0 x */
                    const double *v0k = V0 + k * DM;
                    double acc = 0.0;
                    for (size_t j = 0; j < DM; j++) acc += v0k[j] * xi[j];
                    Vx[k] = C0[e*D0+k] * acc;           /* diag(C0_e)·(V0 x) */
                }
                double *d = delta_o + (e*NX+i)*DM;       /* U0·(...) + b0 */
                for (size_t j = 0; j < DM; j++) {
                    double acc = 0.0;
                    for (size_t k = 0; k < D0; k++) acc += U0t[k*DM+j] * Vx[k];
                    d[j] = acc + b0[j] + noise * rnd_gauss(&s);
                }
            }
        free(Vx);
    }

    /* ---- run the solver (defaults: align/smooth off → cleanest recovery) -- */
    hv_params P; hv_params_default(&P);
    z_layer z; z_spectrum sp;
    int rc = hv_solve(delta_o, x, NULL, NULL, NULL,
                      n_exp, n_x, d_model, 0, &P, &z, &sp);
    printf("hv_solve rc=%d\n", rc);

    /* ---- 表 P: energy spectrum + chosen d_ℓ ---- */
    printf("\n== Table P (residual covariance spectrum) ==\n");
    printf(" true d0 = %d   chosen d_l = %d   (threshold %.2f)\n",
           d0, sp.d_l, sp.threshold);
    printf(" %-4s %14s %10s %10s\n", "rank", "eigenvalue", "frac", "cum-frac");
    double cum = 0.0;
    int show = sp.n < 8 ? sp.n : 8;
    for (int k = 0; k < show; k++) {
        double fr = sp.evals[k] / sp.total_energy;
        cum += fr;
        printf(" %-4d %14.6e %9.4f%% %9.4f%%\n", k+1, sp.evals[k],
               100.0*fr, 100.0*cum);
    }

    /* ---- 表 B: reconstruction fidelity (o_hat=NULL → corr vs Δo) ---- */
    z_fidelity fid;
    hv_fidelity(&z, delta_o, x, NULL, n_exp, n_x, d_model, &fid);
    printf("\n== Table B (reconstruction fidelity) ==\n");
    printf(" rel-L2 = %.6e    cosine_mean = %.8f\n", fid.rel_l2, fid.cosine_mean);

    /* ---- serializer round-trip (flat file) ---- */
    /* 自测临时文件: 固定 /tmp 路径(重构交接账: 原硬编码 Mac 会话 scratchpad,
     * spark 上不存在 → save rc!=0 假 FAIL)。 */
    const char *zpath = "/tmp/ds4_hvsolve_selftest_z_layer.bin";
    int sv = z_layer_save(&z, zpath);
    z_layer z2; memset(&z2, 0, sizeof z2);
    int ld = z_layer_load(&z2, zpath);
    z_fidelity fid2; memset(&fid2, 0, sizeof fid2);
    if (ld == 0) hv_fidelity(&z2, delta_o, x, NULL, n_exp, n_x, d_model, &fid2);
    printf("\n== Serializer round-trip ==\n save rc=%d  load rc=%d  "
           "d_l(load)=%d  rel-L2(load)=%.6e\n", sv, ld, z2.d_l, fid2.rel_l2);

    /* ---- demonstrate losses 3+4 active don't break the fit ---- */
    hv_params P2; hv_params_default(&P2);
    P2.w_smooth = 1e-2; P2.w_align = 0.5;
    z_layer z3; z_spectrum sp3;
    hv_solve(delta_o, x, NULL, NULL, NULL, n_exp, n_x, d_model, 0, &P2, &z3, &sp3);
    z_fidelity fid3;
    hv_fidelity(&z3, delta_o, x, NULL, n_exp, n_x, d_model, &fid3);
    printf("\n== Losses smooth+align engaged (w_smooth=1e-2, w_align=0.5) ==\n");
    printf(" d_l=%d  rel-L2 = %.6e   cosine_mean = %.8f\n",
           sp3.d_l, fid3.rel_l2, fid3.cosine_mean);

    /* ---- ASSERTIONS ---- */
    int a_ok = (sp.d_l == d0);
    int b_ok = (fid.rel_l2 < 0.05);
    int c_ok = (fid.cosine_mean > 0.99);
    int rt_ok = (sv == 0 && ld == 0 && fabs(fid2.rel_l2 - fid.rel_l2) < 1e-12);
    printf("\n== Assertions ==\n");
    printf(" (a) d_l == d0          : %s  (%d vs %d)\n", a_ok?"PASS":"FAIL", sp.d_l, d0);
    printf(" (b) rel-L2 < 5%%        : %s  (%.4e)\n", b_ok?"PASS":"FAIL", fid.rel_l2);
    printf(" (c) cosine > 0.99      : %s  (%.6f)\n", c_ok?"PASS":"FAIL", fid.cosine_mean);
    printf(" (+) serializer roundtrip: %s\n", rt_ok?"PASS":"FAIL");

    int all = a_ok && b_ok && c_ok && rt_ok;
    printf("\nRESULT: %s\n", all ? "ALL PASS" : "FAIL");

    z_layer_free(&z); z_layer_free(&z2); z_layer_free(&z3);
    z_spectrum_free(&sp); z_spectrum_free(&sp3);
    free(U0t); free(V0); free(C0); free(b0); free(x); free(delta_o);
    return all ? 0 : 1;
}
#endif /* HVSOLVE_TEST */
