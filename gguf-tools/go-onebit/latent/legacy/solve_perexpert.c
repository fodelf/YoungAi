/* solve_perexpert.c — per-expert INDEPENDENT low-rank corrector.
 *
 * See solve_perexpert.h for the model and staged closed-form derivation. The
 * implementation is deliberately memory-frugal: all O(d²) and O(n_x·d) scratch
 * is allocated ONCE and reused across experts, and the shared Gram (Σ x xᵀ+λI)
 * is inverted ONCE (chol_solve_spd with an identity RHS) so each expert costs
 * only matmuls + one small eigendecomposition. Peak memory is therefore
 * independent of n_exp.
 *
 * Pure C11, row-major, double precision. No C++, no gradient descent.
 */
#include "solve_perexpert.h"
#include "linalg_small.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Tiny row-major GEMM helpers (offline quantizer scale — clarity over */
/* blocking). All matrices row-major, double precision.                */
/* ------------------------------------------------------------------ */

/* C[m×n] = A[m×k] · B[k×n] */
static void gemm_nn(double *restrict C, const double *A, const double *B,
                    int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) s += A[i * k + p] * B[p * n + j];
            C[i * n + j] = s;
        }
}

/* C[m×n] = A[m×k] · B[n×k]ᵀ   (B stored n×k row-major) */
static void gemm_nt(double *restrict C, const double *A, const double *B,
                    int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) s += A[i * k + p] * B[j * k + p];
            C[i * n + j] = s;
        }
}

/* C[m×n] = A[k×m]ᵀ · B[k×n]   (A stored k×m row-major) */
static void gemm_tn(double *restrict C, const double *A, const double *B,
                    int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) s += A[p * m + i] * B[p * n + j];
            C[i * n + j] = s;
        }
}

/* ------------------------------------------------------------------ */

pe_result solve_perexpert(const double *delta_o, const double *x,
                          const double *o_ref, const double *o_hat,
                          int n_exp, int n_x, int d_model,
                          int rank, double lambda) {
    pe_result res = { 0, 0.0, 0.0 };

    if (!delta_o || !x || n_exp < 1 || n_x < 1 || d_model < 1)
        return res;            /* ok stays 0 */

    const int d = d_model;
    const size_t dd = (size_t)d * (size_t)d;

    int r = rank;
    if (r < 0) r = 0;
    if (r > d) r = d;          /* sym_eig_topk also clamps, but bound buffers */

    /* ---- shared buffers (allocated once, reused across experts) ---- */
    double *Gxx   = malloc(dd * sizeof(double));        /* Σ_i x_i x_iᵀ        */
    double *Gtmp  = malloc(dd * sizeof(double));        /* Gxx+λI → L (in chol)*/
    double *Ginv  = malloc(dd * sizeof(double));        /* (Gxx+λI)⁻¹          */
    double *Be    = malloc(dd * sizeof(double));        /* Σ_i Δo_{e,i} x_iᵀ   */
    double *Mfull = malloc(dd * sizeof(double));        /* M_e^full            */
    double *Tmp   = malloc(dd * sizeof(double));        /* M_full·Gxx          */
    double *S     = malloc(dd * sizeof(double));        /* M·Gxx·Mᵀ (sym PSD)  */
    double *Mtr   = malloc(dd * sizeof(double));        /* M_e^trunc           */
    double *evecs = malloc(dd * sizeof(double));        /* top-r eigvecs (r×d) */
    double *evals = malloc((size_t)d * sizeof(double));
    double *coeff = malloc(dd * sizeof(double));        /* Qᵀ·M_full   (r×d)   */
    double *Y     = malloc((size_t)n_x * (size_t)d * sizeof(double)); /* M_tr·xᵀ */
    double *be    = malloc((size_t)d * sizeof(double)); /* per-expert bias     */

    if (!Gxx || !Gtmp || !Ginv || !Be || !Mfull || !Tmp || !S || !Mtr ||
        !evecs || !evals || !coeff || !Y || !be) {
        goto cleanup;          /* ok stays 0 */
    }

    /* ---- shared Gram and its (single) inverse ----------------------
     * Gxx = Σ_i x_i x_iᵀ = xᵀ·x  (x is n_x×d). Inverting (Gxx+λI) ONCE with
     * an identity RHS gives Ginv reusable by every expert via a matmul, so we
     * never re-factor per expert. λ is grown geometrically if the ridge is
     * still not positive-definite (per chol_solve_spd's contract). */
    gemm_tn(Gxx, x, x, d, n_x, d);

    {
        double lam = lambda;
        int chol_ok = 0;
        for (int att = 0; att < 8 && !chol_ok; att++) {
            memcpy(Gtmp, Gxx, dd * sizeof(double));
            for (int a = 0; a < d; a++) Gtmp[(size_t)a * d + a] += lam;
            memset(Ginv, 0, dd * sizeof(double));
            for (int a = 0; a < d; a++) Ginv[(size_t)a * d + a] = 1.0;
            if (chol_solve_spd(Gtmp, d, Ginv, d) == 0) chol_ok = 1;
            else lam = (lam > 0.0) ? lam * 10.0 : 1e-6;
        }
        if (!chol_ok) goto cleanup;   /* ok stays 0 */
    }

    /* ---- per-expert fit + global metric accumulation --------------- */
    double num_sq = 0.0;   /* Σ ‖pred − ref‖²                              */
    double den_sq = 0.0;   /* Σ ‖ref‖²                                     */
    double sum_cos = 0.0;  /* Σ_(e,i) cos(pred, ref)                       */

    for (int e = 0; e < n_exp; e++) {
        const double *delta_e = delta_o + (size_t)e * n_x * d;
        const double *ohat_e  = o_hat ? o_hat + (size_t)e * n_x * d : NULL;
        const double *oref_e  = o_ref ? o_ref + (size_t)e * n_x * d : NULL;

        /* 1) ridge map: B_e = Δo_eᵀ·x ;  M_full = B_e·Ginv. */
        gemm_tn(Be, delta_e, x, d, n_x, d);
        gemm_nn(Mfull, Be, Ginv, d, d, d);

        /* 2) rank-r truncation via reduced-rank regression.
         *    S = M_full·Gxx·M_fullᵀ (fitted-output covariance, sym PSD).
         *    Q = top-r eigvecs of S (output directions).
         *    M_trunc = Q·Qᵀ·M_full. */
        if (r > 0) {
            gemm_nn(Tmp, Mfull, Gxx, d, d, d);   /* Tmp = M_full·Gxx        */
            gemm_nt(S, Tmp, Mfull, d, d, d);     /* S   = Tmp·M_fullᵀ       */

            /* symmetrize defensively (round-off) before the eig solve */
            for (int a = 0; a < d; a++)
                for (int b2 = a + 1; b2 < d; b2++) {
                    double m = 0.5 * (S[(size_t)a * d + b2] + S[(size_t)b2 * d + a]);
                    S[(size_t)a * d + b2] = m;
                    S[(size_t)b2 * d + a] = m;
                }

            uint64_t seed = 0xD1B54A32D192ED03ULL +
                            (uint64_t)e * 0x9E3779B97F4A7C15ULL;
            sym_eig_topk(S, d, r, evecs, evals, 0 /*default iters*/, seed);

            gemm_nn(coeff, evecs, Mfull, r, d, d);   /* coeff = Qᵀ·M_full   */
            gemm_tn(Mtr, evecs, coeff, d, r, d);     /* M_tr  = Q·coeff     */
        } else {
            memset(Mtr, 0, dd * sizeof(double));     /* rank 0 → pure bias  */
        }

        /* Y = M_trunc·xᵀ  (Y[i] = M_trunc·x_i), then bias b_e = mean(Δo−Y). */
        gemm_nt(Y, x, Mtr, n_x, d, d);
        for (int c = 0; c < d; c++) {
            double s = 0.0;
            for (int i = 0; i < n_x; i++)
                s += delta_e[(size_t)i * d + c] - Y[(size_t)i * d + c];
            be[c] = s / (double)n_x;
        }

        /* 3) accumulate fidelity. corr = M_trunc·x_i + b_e. */
        for (int i = 0; i < n_x; i++) {
            const double *yrow  = Y + (size_t)i * d;
            const double *dorow = delta_e + (size_t)i * d;
            const double *ohrow = ohat_e ? ohat_e + (size_t)i * d : NULL;
            const double *orrow = oref_e ? oref_e + (size_t)i * d : NULL;

            double dot = 0.0, np = 0.0, nr = 0.0, diff2 = 0.0;
            for (int c = 0; c < d; c++) {
                double corr = yrow[c] + be[c];
                double pr, rf;
                if (ohrow) {
                    double oh = ohrow[c];
                    pr = oh + corr;
                    rf = orrow ? orrow[c] : (oh + dorow[c]);
                } else {
                    pr = corr;
                    rf = dorow[c];   /* measure correction directly vs Δo */
                }
                double dlt = pr - rf;
                diff2 += dlt * dlt;
                dot   += pr * rf;
                np    += pr * pr;
                nr    += rf * rf;
            }
            num_sq += diff2;
            den_sq += nr;
            double denom = sqrt(np * nr);
            sum_cos += (denom > 1e-300) ? (dot / denom) : 0.0;
        }
    }

    res.ok       = 1;
    res.rel_l2   = (den_sq > 0.0) ? sqrt(num_sq / den_sq) : 0.0;
    res.cos_mean = sum_cos / ((double)n_exp * (double)n_x);

cleanup:
    free(Gxx);  free(Gtmp);  free(Ginv);  free(Be);    free(Mfull);
    free(Tmp);  free(S);     free(Mtr);   free(evecs); free(evals);
    free(coeff); free(Y);    free(be);
    return res;
}

/* ================================================================== */
/* Synthetic self-test (no model). Build:                              */
/*   cc -O2 -std=c11 -Wall -Wextra -DPEREXPERT_TEST solve_perexpert.c  */
/*      linalg_small.c -o petest -lm && ./petest                       */
/* ================================================================== */
#ifdef PEREXPERT_TEST
#include <stdio.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* deterministic splitmix64 → no rand()/time(); reproducible test data. */
static uint64_t g_sm;
static uint64_t sm_next(void) {
    uint64_t z = (g_sm += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double sm_u01(void) {           /* [0,1) */
    return (double)(sm_next() >> 11) * (1.0 / 9007199254740992.0);
}
static double sm_normal(void) {        /* Box–Muller standard normal */
    double u1 = sm_u01(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = sm_u01();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(void) {
    const int d = 32, r0 = 3, n_exp = 8, n_x = 200;
    const double noise = 1e-3, lambda = 1e-3;

    g_sm = 0xC0FFEEULL;

    double *x   = malloc((size_t)n_x * d * sizeof(double));
    double *del = malloc((size_t)n_exp * n_x * d * sizeof(double));
    double *U0  = malloc((size_t)d * r0 * sizeof(double));
    double *V0  = malloc((size_t)r0 * d * sizeof(double));
    double *M0  = malloc((size_t)d * d * sizeof(double));
    double *b0  = malloc((size_t)d * sizeof(double));
    assert(x && del && U0 && V0 && M0 && b0);

    /* shared activation pool, column-centered so the constant bias b0_e is
     * cleanly separable from the linear map (Σ_i x_i = 0). */
    for (int i = 0; i < n_x; i++)
        for (int j = 0; j < d; j++) x[i * d + j] = sm_normal();
    for (int j = 0; j < d; j++) {
        double mean = 0.0;
        for (int i = 0; i < n_x; i++) mean += x[i * d + j];
        mean /= (double)n_x;
        for (int i = 0; i < n_x; i++) x[i * d + j] -= mean;
    }

    /* per-expert ground truth: Δo_e = U0_e·(V0_e·x_i) + b0_e + tiny noise,
     * with M0_e = U0_e·V0_e exactly rank r0. */
    for (int e = 0; e < n_exp; e++) {
        for (int a = 0; a < d; a++)
            for (int k = 0; k < r0; k++) U0[a * r0 + k] = sm_normal();
        for (int k = 0; k < r0; k++)
            for (int j = 0; j < d; j++) V0[k * d + j] = sm_normal();
        for (int j = 0; j < d; j++) b0[j] = 0.5 * sm_normal();

        for (int a = 0; a < d; a++)            /* M0 = U0·V0 (rank r0)  */
            for (int j = 0; j < d; j++) {
                double s = 0.0;
                for (int k = 0; k < r0; k++) s += U0[a * r0 + k] * V0[k * d + j];
                M0[a * d + j] = s;
            }

        double *de = del + (size_t)e * n_x * d;
        for (int i = 0; i < n_x; i++)
            for (int a = 0; a < d; a++) {
                double s = b0[a] + noise * sm_normal();
                for (int j = 0; j < d; j++) s += M0[a * d + j] * x[i * d + j];
                de[i * d + a] = s;
            }
    }

    /* exact rank → near-perfect reconstruction. */
    pe_result r3 = solve_perexpert(del, x, NULL, NULL, n_exp, n_x, d, r0, lambda);
    /* under-rank → strictly worse, but finite and stable. */
    pe_result r1 = solve_perexpert(del, x, NULL, NULL, n_exp, n_x, d, 1, lambda);

    printf("per-expert independent low-rank solver — synthetic self-test\n");
    printf("  d_model=%d  n_exp=%d  n_x=%d  true_rank=%d  noise=%.0e\n",
           d, n_exp, n_x, r0, noise);
    printf("  rank=%d : ok=%d  rel_l2=%.6e  cos_mean=%.6f\n",
           r0, r3.ok, r3.rel_l2, r3.cos_mean);
    printf("  rank=%d : ok=%d  rel_l2=%.6e  cos_mean=%.6f\n",
           1, r1.ok, r1.rel_l2, r1.cos_mean);

    /* assertions */
    assert(r3.ok == 1);
    assert(r3.rel_l2 < 0.05);
    assert(r3.cos_mean > 0.99);

    assert(r1.ok == 1);
    assert(isfinite(r1.rel_l2));
    assert(isfinite(r1.cos_mean));
    assert(r1.rel_l2 > r3.rel_l2);          /* under-rank is worse        */
    assert(r1.rel_l2 < 10.0);               /* ...but stable, not blown up*/

    printf("ALL TESTS PASSED\n");

    free(x); free(del); free(U0); free(V0); free(M0); free(b0);
    return 0;
}
#endif /* PEREXPERT_TEST */
