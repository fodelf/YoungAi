/* hiddenvar_solve.c — per-layer closed-form solve of z^ℓ for the Go-domain
 * 1-bit quantizer (SPEC.md §5/§6/§7). See hiddenvar_solve.h for the model and
 * the four-loss overview. Pure C99; links only linalg_small.h.
 *
 * NOTHING here trains: every quantity is a closed-form least-squares /
 * eigenvalue / mean computation. The only "iteration" is inside the linked
 * sym_eig_topk (power iteration for the PCA basis) and the bounded ridge-bump
 * retry of the SPD solve — neither is gradient descent.
 */
#include "hiddenvar_solve.h"
#include "linalg_small.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* deterministic dither PRNG (NOT rand()) for loss_fixed               */
/* ------------------------------------------------------------------ */
/* Local splitmix64, identical algorithm to linalg_small's internal one but
 * private here (the header exposes no PRNG). Used solely to dither the ridge
 * diagonal so degenerate directions get distinct, reproducible values. */
static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double sm64_unit(uint64_t *s) { /* uniform double in [0,1) */
    return (double)(sm64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* ================================================================== */
/* THE FOUR LOSSES                                                     */
/* ================================================================== */

/* Loss 1 — classification / routing loss → δ (SPEC.md §7.1).
 *
 * The 1-bit experts perturb the router's sqrtsoftplus logits, shifting which
 * experts land in the top-6. Model the fix as a per-expert additive bias δ_e
 * added back to the 1-bit router probabilities, shared across calibration
 * rows i. The least-squares objective
 *      min_δ Σ_i ‖ probs_ref[i] − (probs_1bit[i] + δ) ‖²
 * decouples per expert and is minimized by the mean gap:
 *      δ_e = mean_i ( probs_ref[i][e] − probs_1bit[i][e] ).
 * This is exactly the bias that realigns expected selection with the
 * reference, restoring the original top-6. Closed form, one pass. */
void loss_classify(const double *probs_ref, const double *probs_1bit,
                   int n_route, int n_exp, double w_classify,
                   double *delta_out) {
    for (int e = 0; e < n_exp; e++) delta_out[e] = 0.0;
    if (!probs_ref || !probs_1bit || n_route <= 0)
        return;                       /* router probs unavailable → δ=0 */
    for (int e = 0; e < n_exp; e++) {
        double acc = 0.0;
        for (int i = 0; i < n_route; i++)
            acc += probs_ref[(size_t)i * n_exp + e]
                 - probs_1bit[(size_t)i * n_exp + e];
        delta_out[e] = w_classify * (acc / (double)n_route);
    }
}

/* Loss 2 — fixed / noise loss (SPEC.md §7.2).
 *
 * Two jobs, both about pinning the otherwise under-determined solve:
 *  (1) Tikhonov ridge  w·λ·I  added to the normal-equations Gram. Guarantees
 *      positive-definiteness for chol_solve_spd and damps directions the Go
 *      calibration pool does not excite (anti-overfit). λ is taken RELATIVE
 *      to the mean Gram diagonal so it is scale-free across layers.
 *  (2) a deterministic per-diagonal DITHER from splitmix64(seed) — never
 *      rand()/time(). It perturbs each ridge entry by ≤0.05%, giving exactly
 *      degenerate directions distinct values so the solution is unique AND
 *      bit-reproducible from `seed` alone. */
void loss_fixed(double *gram, int n, double lambda_rel, double w_fixed,
                uint64_t seed) {
    if (n <= 0 || !gram) return;
    double md = 0.0;
    for (int i = 0; i < n; i++) md += gram[(size_t)i * n + i];
    md /= (double)n;
    if (md <= 0.0) md = 1.0;                       /* fall back to unit scale */
    double ridge = w_fixed * lambda_rel * md;
    uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < n; i++) {
        double u = sm64_unit(&st);                 /* deterministic in [0,1) */
        double dith = ridge * (1.0 + 1e-3 * (u - 0.5)); /* ±0.05% */
        gram[(size_t)i * n + i] += dith;
    }
}

/* Loss 3 — smoothness loss (SPEC.md §7.3).
 *
 * Penalize first differences of the FITTED correction trajectory along the
 * calibration order i. For the shared input map V the per-expert offset q_e
 * cancels in a difference, leaving
 *      Σ_{e,i} ‖ V x_{i+1} − V x_i ‖²  =  Σ_k v_kᵀ [ n_exp·Σ_i Δx_iΔx_iᵀ ] v_k ,
 *   Δx_i = x_{i+1} − x_i .
 * That quadratic form folds purely into the Gram (target 0 → no RHS change),
 * keeping the correction from jumping between adjacent Go tokens / positions.
 * O(n_x·d_model²). */
void loss_smooth(double *gram, const double *x, int n_x, int d_model,
                 int n_exp, double w_smooth) {
    if (w_smooth == 0.0 || n_x < 2 || !gram || !x) return;
    double c = w_smooth * (double)n_exp;
    size_t DM = (size_t)d_model;
    for (int i = 0; i + 1 < n_x; i++) {
        const double *xa = x + (size_t)i * DM;
        const double *xb = x + (size_t)(i + 1) * DM;
        /* gram += c · Δx Δxᵀ  (accumulate full symmetric) */
        for (size_t a = 0; a < DM; a++) {
            double da = xb[a] - xa[a];
            if (da == 0.0) continue;
            double cda = c * da;
            double *grow = gram + a * DM;
            for (size_t b = 0; b < DM; b++)
                grow[b] += cda * (xb[b] - xa[b]);
        }
    }
}

/* Loss 4 — alignment / direction loss (SPEC.md §7.4).
 *
 * A SINGLE closed-form reweight pass (≤1 pass — norms computed once, no
 * iteration). Weight each sample (e,i) by the inverse of its reference-output
 * norm so large-magnitude outputs do not dominate the squared error and the
 * per-expert OUTPUT DIRECTION (hence logit ranking) is preserved:
 *      a_{e,i} = (1 − w) + w · ( mean‖ref‖ / (‖ref_{e,i}‖ + ε) ).
 * w=0 → uniform weights (alignment off). The weights then enter the weighted
 * normal equations of the per-expert C/β refinement (stage 5), which is where
 * per-expert magnitude/direction is set and where per-sample weights are cheap
 * and keep the shared-V solve factorable. */
void loss_align(const double *ref, int n_exp, int n_x, int d_model,
                double w_align, double *weights_out) {
    size_t N = (size_t)n_exp * (size_t)n_x;
    if (!ref || w_align == 0.0) {
        for (size_t t = 0; t < N; t++) weights_out[t] = 1.0;
        return;
    }
    size_t DM = (size_t)d_model;
    double mean_norm = 0.0;
    for (size_t t = 0; t < N; t++) {                /* pass: norms + their mean */
        const double *v = ref + t * DM;
        double s = 0.0;
        for (size_t j = 0; j < DM; j++) s += v[j] * v[j];
        double nr = sqrt(s);
        weights_out[t] = nr;                        /* stash norm temporarily */
        mean_norm += nr;
    }
    mean_norm /= (double)N;
    if (mean_norm <= 0.0) mean_norm = 1.0;
    const double eps = 1e-12;
    for (size_t t = 0; t < N; t++) {                /* blend (closes the pass) */
        double nr = weights_out[t];
        weights_out[t] = (1.0 - w_align) + w_align * (mean_norm / (nr + eps));
    }
}

/* ================================================================== */
/* defaults                                                            */
/* ================================================================== */
void hv_params_default(hv_params *p) {
    if (!p) return;
    p->energy_threshold = 0.95;   /* SPEC.md §6.3 default energy fraction      */
    p->max_rank         = 64;     /* cap; d_ℓ expected ≪ d_model               */
    p->lambda           = 1e-4;   /* tiny relative ridge — pins, barely biases */
    p->w_classify       = 1.0;
    p->w_fixed          = 1.0;
    p->w_smooth         = 0.0;    /* off by default → cleanest LS recovery     */
    p->w_align          = 0.0;    /* off by default → uniform weights          */
    p->seed             = 0x0123456789ABCDEFULL;
    p->eig_iters        = 200;
}

/* ================================================================== */
/* the staged closed-form solve                                        */
/* ================================================================== */
int hv_solve(const double *delta_o, const double *x, const double *o_ref,
             const double *probs_ref, const double *probs_1bit,
             int n_exp, int n_x, int d_model, int n_route,
             const hv_params *params, z_layer *out, z_spectrum *spec_out) {
    if (!delta_o || !x || !out || n_exp <= 0 || n_x <= 0 || d_model <= 0)
        return 1;

    hv_params P;
    if (params) P = *params; else hv_params_default(&P);

    const size_t NE = (size_t)n_exp, NX = (size_t)n_x, DM = (size_t)d_model;
    #define DO_AT(e, i) (delta_o + (((size_t)(e) * NX + (size_t)(i)) * DM))

    /* ---------- Stage 1: global bias b = mean_{e,i} Δo ----------------
     * R_{e,i} = Δo_{e,i} − b is the residual to model with the low-rank map. */
    double *b = (double *)calloc(DM, sizeof(double));
    for (size_t e = 0; e < NE; e++)
        for (size_t i = 0; i < NX; i++) {
            const double *d = DO_AT(e, i);
            for (size_t j = 0; j < DM; j++) b[j] += d[j];
        }
    {
        double inv = 1.0 / ((double)NE * (double)NX);
        for (size_t j = 0; j < DM; j++) b[j] *= inv;
    }

    /* ---------- Stage 2: output covariance Σ = Σ_{e,i} R Rᵀ -----------
     * Σ is d_model×d_model; its top eigenvectors are the shared output basis
     * U and its spectrum sets the per-layer rank d_ℓ (SPEC.md §6.3 表 P).
     * Dominant offline cost: O(n_exp·n_x·d_model²). Accumulate the upper
     * triangle then mirror (sym_eig_topk reads the full matrix). */
    double *Sigma = (double *)calloc(DM * DM, sizeof(double));
    {
        double *R = (double *)malloc(DM * sizeof(double));
        for (size_t e = 0; e < NE; e++)
            for (size_t i = 0; i < NX; i++) {
                const double *d = DO_AT(e, i);
                for (size_t j = 0; j < DM; j++) R[j] = d[j] - b[j];
                if (P.w_align > 0.0) {   /* direction-aware basis (loss 4 → U):
                                          * unit-normalize residual so the PCA captures
                                          * the shared DIRECTION subspace, not the
                                          * magnitude subspace (else U is L2-dominated). */
                    double nr = 0.0;
                    for (size_t j = 0; j < DM; j++) nr += R[j] * R[j];
                    nr = sqrt(nr);
                    if (nr > 1e-12) { double inv = 1.0 / nr; for (size_t j = 0; j < DM; j++) R[j] *= inv; }
                }
                for (size_t a = 0; a < DM; a++) {
                    double ra = R[a];
                    if (ra == 0.0) continue;
                    double *srow = Sigma + a * DM;
                    for (size_t c = a; c < DM; c++) srow[c] += ra * R[c];
                }
            }
        for (size_t a = 0; a < DM; a++)
            for (size_t c = a + 1; c < DM; c++)
                Sigma[c * DM + a] = Sigma[a * DM + c];      /* mirror */
        free(R);
    }
    double trace = 0.0;                              /* = Σ of ALL eigenvalues */
    for (size_t a = 0; a < DM; a++) trace += Sigma[a * DM + a];

    /* ---------- Stage 2b: eigen-decompose, pick d_ℓ by energy ---------- */
    int kmax = P.max_rank;
    if (kmax > d_model) kmax = d_model;
    if (kmax < 1) kmax = 1;
    double *evecs = (double *)malloc((size_t)kmax * DM * sizeof(double));
    double *evals = (double *)malloc((size_t)kmax * sizeof(double));
    sym_eig_topk(Sigma, d_model, kmax, evecs, evals, P.eig_iters, P.seed);
    free(Sigma);

    /* d_ℓ = smallest rank whose cumulative eigen-energy ≥ threshold. The
     * denominator is the EXACT trace (all eigenvalues), so the fraction is
     * correct even though we only computed the top kmax pairs. */
    double total = (trace > 0.0) ? trace : 1.0;
    int d_l = 1;
    {
        double cum = 0.0;
        for (int k = 0; k < kmax; k++) {
            cum += (evals[k] > 0.0) ? evals[k] : 0.0;
            d_l = k + 1;
            if (cum / total >= P.energy_threshold) break;
        }
        if (d_l < 1) d_l = 1;
        if (d_l > kmax) d_l = kmax;                 /* capped at max_rank */
    }
    const size_t DL = (size_t)d_l;

    /* U (d_model×d_l): column k = k-th eigenvector (row k of evecs). */
    double *U = (double *)malloc(DM * DL * sizeof(double));
    for (size_t k = 0; k < DL; k++) {
        const double *ev = evecs + k * DM;
        for (size_t j = 0; j < DM; j++) U[j * DL + k] = ev[j];
    }

    /* ---------- Stage 3: project residuals to coordinates p = Uᵀ R ----
     * U has orthonormal columns, so working in coordinates makes the per-coord
     * regressions independent (the diagonal-C structure of the model). */
    double *p = (double *)malloc(NE * NX * DL * sizeof(double));
    {
        double *Rv = (double *)malloc(DM * sizeof(double));
        for (size_t e = 0; e < NE; e++)
            for (size_t i = 0; i < NX; i++) {
                const double *d = DO_AT(e, i);
                for (size_t j = 0; j < DM; j++) Rv[j] = d[j] - b[j];
                double *pp = p + (e * NX + i) * DL;
                for (size_t k = 0; k < DL; k++) {
                    double s = 0.0;
                    for (size_t j = 0; j < DM; j++) s += U[j * DL + k] * Rv[j];
                    pp[k] = s;
                }
            }
        free(Rv);
    }

    /* ---------- Stage 4: shared input feature V (pooled ridge) ---------
     * Per coordinate k the model is p_{e,i,k} ≈ v_k·x_i + q_{e,k} (q_e = the
     * per-expert offset in coord space). Eliminating q_e analytically (its
     * optimum is q_{e,k}=p̄_{e,k} − x̄·v_k; x_i is SHARED across experts so
     * x̄ is common) leaves the centered ridge regression
     *   ( n_exp·Σ_i(x_i−x̄)(x_i−x̄)ᵀ  +  smooth  +  λI ) v_k
     *        = Σ_i (x_i−x̄) · Σ_e(p_{e,i,k}−p̄_{e,k}) .
     * The Gram is the SAME for all k → one Cholesky, d_l right-hand sides.
     * loss_smooth and loss_fixed fold directly into this Gram. */
    double *xbar = (double *)calloc(DM, sizeof(double));
    for (size_t i = 0; i < NX; i++) {
        const double *xi = x + i * DM;
        for (size_t j = 0; j < DM; j++) xbar[j] += xi[j];
    }
    for (size_t j = 0; j < DM; j++) xbar[j] /= (double)NX;

    double *pbar = (double *)calloc(NE * DL, sizeof(double));   /* p̄_{e,k} */
    for (size_t e = 0; e < NE; e++)
        for (size_t i = 0; i < NX; i++) {
            const double *pp = p + (e * NX + i) * DL;
            double *pb = pbar + e * DL;
            for (size_t k = 0; k < DL; k++) pb[k] += pp[k];
        }
    for (size_t e = 0; e < NE; e++)
        for (size_t k = 0; k < DL; k++) pbar[e * DL + k] /= (double)NX;

    /* s_{i,k} = Σ_e (p_{e,i,k} − p̄_{e,k})  (expert-summed centered coord) */
    double *s = (double *)calloc(NX * DL, sizeof(double));
    for (size_t e = 0; e < NE; e++)
        for (size_t i = 0; i < NX; i++) {
            const double *pp = p + (e * NX + i) * DL;
            double *sr = s + i * DL;
            for (size_t k = 0; k < DL; k++) sr[k] += pp[k] - pbar[e * DL + k];
        }

    /* base Gram = n_exp · Σ_i (x_i−x̄)(x_i−x̄)ᵀ */
    double *Gbase = (double *)calloc(DM * DM, sizeof(double));
    {
        double *xc = (double *)malloc(DM * sizeof(double));
        for (size_t i = 0; i < NX; i++) {
            const double *xi = x + i * DM;
            for (size_t j = 0; j < DM; j++) xc[j] = xi[j] - xbar[j];
            for (size_t a = 0; a < DM; a++) {
                double xa = xc[a];
                if (xa == 0.0) continue;
                double *gr = Gbase + a * DM;
                for (size_t c = a; c < DM; c++) gr[c] += xa * xc[c];
            }
        }
        for (size_t a = 0; a < DM; a++)
            for (size_t c = a + 1; c < DM; c++) Gbase[c * DM + a] = Gbase[a * DM + c];
        for (size_t t = 0; t < DM * DM; t++) Gbase[t] *= (double)n_exp;
        free(xc);
    }
    loss_smooth(Gbase, x, n_x, d_model, n_exp, P.w_smooth);  /* fold loss 3 */

    /* RHS H (d_model×d_l): H[j][k] = Σ_i (x_i−x̄)[j] · s_{i,k} */
    double *Hbase = (double *)calloc(DM * DL, sizeof(double));
    for (size_t i = 0; i < NX; i++) {
        const double *xi = x + i * DM;
        const double *sr = s + i * DL;
        for (size_t j = 0; j < DM; j++) {
            double xc = xi[j] - xbar[j];
            if (xc == 0.0) continue;
            double *hr = Hbase + j * DL;
            for (size_t k = 0; k < DL; k++) hr[k] += xc * sr[k];
        }
    }

    /* SPD solve with bounded ridge-bump retry (per linalg_small.h guidance):
     * loss_fixed adds the ridge+dither; if still not PD, grow λ and retry. */
    double *A = (double *)malloc(DM * DM * sizeof(double));
    double *B = (double *)malloc(DM * DL * sizeof(double));
    double lam = P.lambda;
    int rc = 1;
    for (int attempt = 0; attempt < 10 && rc != 0; attempt++) {
        memcpy(A, Gbase, DM * DM * sizeof(double));
        loss_fixed(A, d_model, lam, P.w_fixed, P.seed);     /* fold loss 2 */
        memcpy(B, Hbase, DM * DL * sizeof(double));
        rc = chol_solve_spd(A, d_model, B, d_l);
        if (rc != 0) lam *= 10.0;
    }
    if (rc != 0) {                                          /* give up cleanly */
        free(A); free(B); free(Gbase); free(Hbase); free(s);
        free(pbar); free(xbar); free(p); free(U);
        free(evecs); free(evals); free(b);
        return 2;
    }
    /* B now holds the solution X (d_model×d_l), X[j][k]=v_k[j]=V[k][j]. */
    double *V = (double *)malloc(DL * DM * sizeof(double));
    for (size_t k = 0; k < DL; k++)
        for (size_t j = 0; j < DM; j++) V[k * DM + j] = B[j * DL + k];
    free(A); free(B); free(Gbase); free(Hbase); free(s); free(pbar);

    /* ---------- Stage 5: per-expert diagonal gain C_e + offset --------
     * Feature f_{i,k} = (V x_i)_k (shared across experts → precompute once).
     * For each expert e, coord k: weighted 1-D regression of p on f,
     *   min Σ_i a_{e,i}( p_{e,i,k} − C_{e,k} f_{i,k} − q_{e,k} )² + λ-ridge,
     * closed-form (loss_align supplies the weights a_{e,i}). */
    double *f = (double *)malloc(NX * DL * sizeof(double));
    for (size_t i = 0; i < NX; i++) {
        const double *xi = x + i * DM;
        double *fr = f + i * DL;
        for (size_t k = 0; k < DL; k++) {
            const double *vk = V + k * DM;
            double sv = 0.0;
            for (size_t j = 0; j < DM; j++) sv += vk[j] * xi[j];
            fr[k] = sv;
        }
    }
    double *w = (double *)malloc(NE * NX * sizeof(double));
    loss_align(o_ref ? o_ref : delta_o, n_exp, n_x, d_model, P.w_align, w); /* loss 4 */

    double *C = (double *)malloc(NE * DL * sizeof(double));
    double *q = (double *)malloc(NE * DL * sizeof(double));  /* coord offset */
    for (size_t e = 0; e < NE; e++) {
        for (size_t k = 0; k < DL; k++) {
            double Sa = 0, Sf = 0, Sp = 0, Sff = 0, Sfp = 0;
            for (size_t i = 0; i < NX; i++) {
                double a  = w[e * NX + i];
                double fi = f[i * DL + k];
                double pi = p[(e * NX + i) * DL + k];
                Sa += a; Sf += a * fi; Sp += a * pi;
                Sff += a * fi * fi; Sfp += a * fi * pi;
            }
            /* weighted 1-D normal equations; tiny relative ridge (fixed loss)
             * keeps C finite when f_{·,k} is near-constant. */
            double denom = Sa * Sff - Sf * Sf + P.lambda * Sa * Sff;
            double Cek, qek;
            if (Sa <= 0.0 || fabs(denom) < 1e-300) {
                Cek = 0.0;
                qek = (Sa > 0.0) ? Sp / Sa : 0.0;
            } else {
                Cek = (Sa * Sfp - Sf * Sp) / denom;
                qek = (Sp - Cek * Sf) / Sa;
            }
            C[e * DL + k] = Cek;
            q[e * DL + k] = qek;
        }
    }
    free(f); free(w);

    /* β_e from the coord-space offset q_e (SPEC.md §5: offset = β_e·1 + b).
     * The output-space offset is U·q_e; the model can only spend a scalar
     * (β_e·1) on the per-expert part, so β_e is the constant best matching
     * U·q_e → β_e = mean_j (U·q_e)_j = (1/d_model) Σ_k (Σ_j U[j][k]) q_{e,k}.
     * (Components of U·q_e orthogonal to 1 are unrepresentable and dropped;
     *  on well-centered pools q_e≈0 so this term is small.) */
    double *beta = (double *)calloc(NE, sizeof(double));
    {
        double *colsumU = (double *)calloc(DL, sizeof(double));
        for (size_t j = 0; j < DM; j++)
            for (size_t k = 0; k < DL; k++) colsumU[k] += U[j * DL + k];
        for (size_t e = 0; e < NE; e++) {
            double acc = 0.0;
            for (size_t k = 0; k < DL; k++) acc += colsumU[k] * q[e * DL + k];
            beta[e] = acc / (double)DM;
        }
        free(colsumU);
    }
    free(q);

    /* ---------- Stage 6: router δ (loss_classify) -------------------- */
    double *delta = (double *)calloc(NE, sizeof(double));
    loss_classify(probs_ref, probs_1bit, (n_route > 0) ? n_route : n_x,
                  n_exp, P.w_classify, delta);

    /* ---------- populate outputs ------------------------------------ */
    out->d_model = d_model; out->n_exp = n_exp; out->d_l = d_l;
    out->U = U; out->V = V; out->C = C; out->b = b;
    out->beta = beta; out->delta = delta;

    if (spec_out) {
        spec_out->n = kmax;
        spec_out->evals = (double *)malloc((size_t)kmax * sizeof(double));
        memcpy(spec_out->evals, evals, (size_t)kmax * sizeof(double));
        spec_out->total_energy = trace;
        spec_out->d_l = d_l;
        spec_out->threshold = P.energy_threshold;
    }

    free(p); free(xbar); free(evecs); free(evals);
    #undef DO_AT
    return 0;
}

