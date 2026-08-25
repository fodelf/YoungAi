/* calib_diag.c — DEEPER diagnostics for the Go-domain 1-bit hypothesis test.
 *
 * A variant of calib_run.c. The first run on the SHALLOWEST layer (L0) was
 * NEGATIVE: the Δo eigen-spectrum was nearly flat (rank-64 captured ~10% of the
 * energy) and a linear correction barely helped. Before declaring the per-layer
 * hidden variable z^ℓ dead we want to know *why*, by separating three distinct
 * questions that the single calib_run rel-L2 number conflated:
 *
 *   (1) Is the Go expert OUTPUT o_ref itself low-dimensional?      → out PCA
 *   (2) Is the 1-bit output ô itself low-dimensional?              → out PCA
 *   (3) Is Δo low-dim *given perfect coordinates* (ignore x)?      → ORACLE PCA
 *   (4) Can x LINEARLY predict Δo's top coordinates?               → ridge R^2
 *   (5) Could we skip the 1-bit base and GENERATE o_ref from x?    → --target oref
 *
 * The data pipeline (HF-streaming worker that fills o_ref / o_hat / Δo double
 * arrays of shape [256][n_x][4096]) and the arg parsing are reused verbatim
 * from calib_run.c. calib_run's existing baseline + hv_solve + hv_fidelity Δo
 * table is kept; this driver adds the five metrics above on top.
 *
 * The new math reuses linalg_small (sym_eig_topk for PCA, chol_solve_spd for the
 * ridge regression) and hiddenvar_solve (hv_solve / hv_fidelity for metric 5).
 *
 * Build (full driver):  see the compile line in the task / gguf-tools Makefile.
 * Self-test (synthetic, no HF):  cc -DCALIBDIAG_TEST -O2 -std=c11 calib_diag.c \
 *                                   hiddenvar_solve.c linalg_small.c -o t -lm
 *
 * Pure C11. No C++.
 */
#include "hiddenvar_solve.h"
#include "linalg_small.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CALIBDIAG_TEST
/* Model-side headers + threading are only needed by the full HF driver; the
 * synthetic self-test links just hiddenvar_solve.c + linalg_small.c, so guard
 * them out to keep that build free of undefined symbols. */
#include "npy.h"
#include "hf_read.h"
#include "onebit_quant.h"
#include "layer_probe.h"
#include <pthread.h>
#endif

#define DM 4096   /* hidden_size        */
#define DF 2048   /* moe_intermediate   */
#define NEXP 256  /* routed experts     */

/* deterministic seed + power-iteration budget for the new PCA calls (kept
 * separate from hv_solve's own seed so the two are independently reproducible). */
#define DIAG_SEED      0x0123456789ABCDEFULL
#define DIAG_EIG_ITERS 200

/* ====================================================================== */
/* NEW DIAGNOSTIC MATH — shared by the full driver and the synthetic test. */
/* All routines take the width `d` as a parameter (4096 in production, tiny  */
/* in the self-test) and are pure double-precision linear algebra.          */
/* ====================================================================== */

/* Mean of N row-major rows of width d → mu[d]. */
static void rows_mean(const double *A, long N, int d, double *mu) {
    for (int j = 0; j < d; j++) mu[j] = 0.0;
    for (long t = 0; t < N; t++) {
        const double *a = A + (size_t)t * d;
        for (int j = 0; j < d; j++) mu[j] += a[j];
    }
    double inv = (N > 0) ? 1.0 / (double)N : 0.0;
    for (int j = 0; j < d; j++) mu[j] *= inv;
}

/* Centered covariance Sigma[d*d] = Σ_t (a_t − mu)(a_t − mu)ᵀ (full symmetric,
 * accumulated upper-triangle then mirrored — sym_eig_topk reads the whole
 * matrix). Returns trace(Sigma) = Σ of ALL eigenvalues (exact total energy). */
static double centered_cov(const double *A, long N, int d,
                           const double *mu, double *Sigma) {
    for (size_t t = 0; t < (size_t)d * d; t++) Sigma[t] = 0.0;
    double *r = (double *)malloc((size_t)d * sizeof(double));
    for (long t = 0; t < N; t++) {
        const double *a = A + (size_t)t * d;
        for (int j = 0; j < d; j++) r[j] = a[j] - mu[j];
        for (int aa = 0; aa < d; aa++) {
            double ra = r[aa];
            if (ra == 0.0) continue;
            double *srow = Sigma + (size_t)aa * d;
            for (int c = aa; c < d; c++) srow[c] += ra * r[c];
        }
    }
    for (int aa = 0; aa < d; aa++)
        for (int c = aa + 1; c < d; c++)
            Sigma[(size_t)c * d + aa] = Sigma[(size_t)aa * d + c];
    free(r);
    double tr = 0.0;
    for (int aa = 0; aa < d; aa++) tr += Sigma[(size_t)aa * d + aa];
    return tr;
}

/* Smallest rank in [1,kmax] whose cumulative (negatives clamped to 0) energy
 * fraction reaches `thresh`; −1 if not reached within the kmax computed pairs. */
static int rank_at_energy(const double *evals, int kmax, double trace, double thresh) {
    if (trace <= 0.0) return -1;
    double cum = 0.0;
    for (int k = 0; k < kmax; k++) {
        cum += (evals[k] > 0.0) ? evals[k] : 0.0;
        if (cum / trace >= thresh) return k + 1;
    }
    return -1;
}

/* Cumulative energy fraction captured by the top-d eigenpairs, or −1 ("n/a")
 * when d exceeds the number of pairs we actually computed (kmax). */
static double cumE_at(const double *evals, int kmax, double trace, int d) {
    if (d > kmax) return -1.0;
    if (trace <= 0.0) return 0.0;
    double cum = 0.0;
    for (int k = 0; k < d; k++) cum += (evals[k] > 0.0) ? evals[k] : 0.0;
    return cum / trace;
}

/* Metrics 1/2 core: PCA of a centered [N×d] activation block.
 *   Fills mu[d], evecs[kmax*d], evals[kmax] (top eigenpairs), trace, and the
 *   cumE_out[4] fractions at d∈{8,16,32,64} (−1 = n/a). kmax MUST be ≤ d.
 *   Returns the effective rank at `energy` (−1 if > kmax). The evecs/mu/trace
 *   are left populated so a caller (Δo) can reuse them for metrics 3/4. */
static int effrank_compute(const double *A, long N, int d, int kmax, double energy,
                           uint64_t seed, int iters, double *Sigma, double *mu,
                           double *evecs, double *evals,
                           double cumE_out[4], double *trace_out) {
    rows_mean(A, N, d, mu);
    double tr = centered_cov(A, N, d, mu, Sigma);
    sym_eig_topk(Sigma, d, kmax, evecs, evals, iters, seed);
    int dl[4] = {8, 16, 32, 64};
    for (int t = 0; t < 4; t++) cumE_out[t] = cumE_at(evals, kmax, tr, dl[t]);
    if (trace_out) *trace_out = tr;
    return rank_at_energy(evals, kmax, tr, energy);
}

/* Metrics 3 + 4 core, on the residual Δo.
 *
 *   3 (ORACLE rank-d reconstruction): give the reconstructor PERFECT coordinate
 *     knowledge — project each centered residual onto Δo's OWN top-d eigenbasis
 *     and add the global mean (which the z^ℓ model gets for free as the bias b):
 *         Δo_hat = mu + U_d Uᵀ_d (Δo − mu).
 *     residual rel-L2(d) = ‖Δo − Δo_hat‖_F / ‖Δo‖_F. Since the coordinates are
 *     the exact projections (not predicted from x), this is the BEST any rank-d
 *     correction can do; a high residual here means Δo is genuinely high-rank.
 *
 *   4 (x→Δo predictability): take the top-d16 (≤16) Δo eigen-coordinates
 *     p_{e,i} = Uᵀ(Δo_{e,i} − mu) and ridge-regress them on x with ONE shared
 *     linear map (mirroring the z^ℓ model's shared V·x). x_i is common to all
 *     experts, so the Gram and RHS collapse to per-i sums (the same algebra
 *     hv_solve uses). Closed form via chol_solve_spd on (n_exp·Σ xc xcᵀ + λI).
 *     Reports the mean R^2 over the d16 coordinates. R^2≈1 ⇒ x linearly predicts
 *     Δo; R^2≈0 ⇒ the residual is not a (shared) linear function of x.
 *
 *   evecs : kmax×d top eigenvectors of the centered Δo covariance (row k = u_k).
 *   trace : centered total energy Σ‖Δo−mu‖² (= Σ of all eigenvalues).
 *   Gscratch : caller-owned [d*d] workspace (the Δo covariance buffer can be
 *              reused here — its eigenpairs are already extracted into evecs).
 *   oracle_out[4] : rel-L2 at d∈{8,16,32,64} (−1 = n/a, d > kmax).
 *   *meanR2_out, *d16_out, *solve_ok set on return. */
static void compute_delta_metrics(const double *delta, const double *x_d,
                                  int n_exp, int n_x, int d,
                                  const double *mu, const double *evecs, int kmax,
                                  double trace, double lambda_rel, double *Gscratch,
                                  double oracle_out[4], double *meanR2_out,
                                  int *d16_out, int *solve_ok) {
    int d16 = (kmax < 16) ? kmax : 16;
    if (d16 > d) d16 = d;

    double *sum_c2 = (double *)calloc((size_t)kmax, sizeof(double)); /* Σ_ei c_k²  */
    double *S      = (double *)calloc((size_t)n_x * d16, sizeof(double)); /* Σ_e c   */
    double *Q      = (double *)calloc((size_t)n_x * d16, sizeof(double)); /* Σ_e c²  */
    double *r      = (double *)malloc((size_t)d * sizeof(double));
    double *xbar   = (double *)calloc((size_t)d, sizeof(double));
    double *xc     = (double *)malloc((size_t)d * sizeof(double));
    double denom_full = 0.0;   /* Σ_ei ‖Δo‖²  (the thing the oracle reconstructs) */

    for (int i = 0; i < n_x; i++) {
        const double *xi = x_d + (size_t)i * d;
        for (int j = 0; j < d; j++) xbar[j] += xi[j];
    }
    for (int j = 0; j < d; j++) xbar[j] /= (double)n_x;

    /* Single projection pass: per (e,i) compute the kmax eigen-coordinates of the
     * centered residual once, feeding both the oracle energy split (sum_c2) and
     * the per-i expert sums (S,Q) used by the regression. */
    for (int e = 0; e < n_exp; e++)
        for (int i = 0; i < n_x; i++) {
            const double *dv = delta + ((size_t)e * n_x + i) * d;
            double nrm2 = 0.0;
            for (int j = 0; j < d; j++) { double rv = dv[j] - mu[j]; r[j] = rv; nrm2 += dv[j] * dv[j]; }
            denom_full += nrm2;
            for (int k = 0; k < kmax; k++) {
                const double *uk = evecs + (size_t)k * d;
                double c = 0.0;
                for (int j = 0; j < d; j++) c += uk[j] * r[j];
                sum_c2[k] += c * c;
                if (k < d16) {
                    S[(size_t)i * d16 + k] += c;
                    Q[(size_t)i * d16 + k] += c * c;
                }
            }
        }

    /* ---- metric 3: oracle residual at d∈{8,16,32,64} -------------------- */
    int dl[4] = {8, 16, 32, 64};
    for (int t = 0; t < 4; t++) {
        int D = dl[t];
        if (D > kmax) { oracle_out[t] = -1.0; continue; }
        double cap = 0.0;
        for (int k = 0; k < D; k++) cap += sum_c2[k];     /* captured centered energy */
        double resid = trace - cap;
        if (resid < 0.0) resid = 0.0;                     /* numerical guard */
        oracle_out[t] = (denom_full > 0.0) ? sqrt(resid / denom_full) : 0.0;
    }

    /* ---- metric 4: shared ridge regression of p(top-d16) on x ----------- */
    /* G = n_exp · Σ_i (x_i−x̄)(x_i−x̄)ᵀ   (XᵀX with x repeated over experts). */
    for (size_t t = 0; t < (size_t)d * d; t++) Gscratch[t] = 0.0;
    for (int i = 0; i < n_x; i++) {
        const double *xi = x_d + (size_t)i * d;
        for (int j = 0; j < d; j++) xc[j] = xi[j] - xbar[j];
        for (int a = 0; a < d; a++) {
            double xa = xc[a];
            if (xa == 0.0) continue;
            double *gr = Gscratch + (size_t)a * d;
            for (int c = a; c < d; c++) gr[c] += xa * xc[c];
        }
    }
    for (int a = 0; a < d; a++)
        for (int c = a + 1; c < d; c++)
            Gscratch[(size_t)c * d + a] = Gscratch[(size_t)a * d + c];
    for (size_t t = 0; t < (size_t)d * d; t++) Gscratch[t] *= (double)n_exp;

    /* RHS H[d*d16]: H[j][k] = Σ_i (x_i−x̄)[j] · S_i[k]   (S_i = Σ_e p_{e,i}). */
    double *H = (double *)calloc((size_t)d * d16, sizeof(double));
    for (int i = 0; i < n_x; i++) {
        const double *xi = x_d + (size_t)i * d;
        const double *Sr = S + (size_t)i * d16;
        for (int j = 0; j < d; j++) {
            double xcj = xi[j] - xbar[j];
            if (xcj == 0.0) continue;
            double *hr = H + (size_t)j * d16;
            for (int k = 0; k < d16; k++) hr[k] += xcj * Sr[k];
        }
    }

    double meandiag = 0.0;
    for (int a = 0; a < d; a++) meandiag += Gscratch[(size_t)a * d + a];
    meandiag /= (double)d;
    if (meandiag <= 0.0) meandiag = 1.0;
    double lam = (lambda_rel >= 0.0) ? lambda_rel : 1e-4;

    double *A = (double *)malloc((size_t)d * d * sizeof(double));
    double *B = (double *)malloc((size_t)d * d16 * sizeof(double));
    int rc = 1;
    double lamc = lam;
    for (int att = 0; att < 8 && rc != 0; att++) {          /* ridge-bump retry */
        memcpy(A, Gscratch, (size_t)d * d * sizeof(double));
        for (int a = 0; a < d; a++) A[(size_t)a * d + a] += lamc * meandiag;
        memcpy(B, H, (size_t)d * d16 * sizeof(double));
        rc = chol_solve_spd(A, d, B, d16);                  /* B → W [d×d16]    */
        if (rc != 0) lamc *= 10.0;
    }

    double meanR2 = 0.0;
    int okr = 0;
    if (rc == 0) {
        /* p_hat_i[k] = Σ_j (x_i−x̄)[j]·W[j][k] (depends only on i; x shared).
         * SS_res_k = Σ_i ( Q_i[k] − 2·p_hat·S_i[k] + n_exp·p_hat² ).
         * SS_tot_k = Σ_ei p² = sum_c2[k] (p is mean-zero over all samples). */
        double *SSres = (double *)calloc((size_t)d16, sizeof(double));
        double *phat  = (double *)malloc((size_t)d16 * sizeof(double));
        for (int i = 0; i < n_x; i++) {
            const double *xi = x_d + (size_t)i * d;
            for (int k = 0; k < d16; k++) phat[k] = 0.0;
            for (int j = 0; j < d; j++) {
                double xcj = xi[j] - xbar[j];
                if (xcj == 0.0) continue;
                const double *bj = B + (size_t)j * d16;
                for (int k = 0; k < d16; k++) phat[k] += xcj * bj[k];
            }
            const double *Sr = S + (size_t)i * d16;
            const double *Qr = Q + (size_t)i * d16;
            for (int k = 0; k < d16; k++) {
                double ph = phat[k];
                SSres[k] += Qr[k] - 2.0 * ph * Sr[k] + (double)n_exp * ph * ph;
            }
        }
        for (int k = 0; k < d16; k++) {
            double sstot = sum_c2[k];
            double r2 = (sstot > 0.0) ? 1.0 - SSres[k] / sstot : 0.0;
            meanR2 += r2;
        }
        meanR2 /= (double)d16;
        okr = 1;
        free(SSres); free(phat);
    }

    *meanR2_out = meanR2;
    *d16_out = d16;
    *solve_ok = okr;
    free(A); free(B); free(H);
    free(sum_c2); free(S); free(Q); free(r); free(xbar); free(xc);
}

/* print a fraction-or-n/a cell (−1 → "n/a"). */
static void pv(double v) { if (v < 0.0) printf("n/a"); else printf("%.4f", v); }

