/* solve_denoise.c — shared reduced-rank "denoiser" map  o_ref ≈ M·ô + b.
 *
 * See solve_denoise.h for the model and the staged closed-form derivation. The
 * single most important property of this file is the TRAIN/TEST split: a
 * d_model×d_model map fit on few tokens overfits, so the only trustworthy
 * fidelity numbers come from held-out tokens. The covariances are accumulated
 * over the TRAIN tokens only, in a single streaming pass, and the reported
 * rel_l2 / cos are measured over the TEST tokens only.
 *
 * Memory: the cost is the d×d feature/cross covariances and a few d×d scratch
 * maps (all allocated once). Forming them from tens-of-thousands of pooled
 * samples is O(N·d²) — the dominant cost (~minutes at d_model=4096) — but it is
 * ONE shared solve, not a per-expert one, so peak memory is O(d²), independent
 * of n_exp and n_x.
 *
 * Pure C11, row-major, double precision. No C++, no gradient descent. Links
 * only linalg_small.h (chol_solve_spd + sym_eig_topk).
 */
#include "solve_denoise.h"
#include "linalg_small.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Deterministic seed for sym_eig_topk. There is exactly ONE shared eigensolve
 * (not a per-expert one), so a single fixed seed makes the whole result
 * bit-reproducible. */
#define DN_EIG_SEED 0xD1B54A32D192ED03ULL

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
            for (int p = 0; p < k; p++) s += A[(size_t)i * k + p] * B[(size_t)p * n + j];
            C[(size_t)i * n + j] = s;
        }
}

/* C[m×n] = A[m×k] · B[n×k]ᵀ   (B stored n×k row-major) */
static void gemm_nt(double *restrict C, const double *A, const double *B,
                    int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) s += A[(size_t)i * k + p] * B[(size_t)j * k + p];
            C[(size_t)i * n + j] = s;
        }
}

/* C[m×n] = A[k×m]ᵀ · B[k×n]   (A stored k×m row-major, B stored k×n row-major) */
static void gemm_tn(double *restrict C, const double *A, const double *B,
                    int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) s += A[(size_t)p * m + i] * B[(size_t)p * n + j];
            C[(size_t)i * n + j] = s;
        }
}

/* C[m×n] += A[k×m]ᵀ · B[k×n]   (accumulating; used to pool the train-token
 * second moments across experts without materializing the pooled matrix). */
static void gemm_tn_acc(double *restrict C, const double *A, const double *B,
                        int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < k; p++) s += A[(size_t)p * m + i] * B[(size_t)p * n + j];
            C[(size_t)i * n + j] += s;
        }
}

/* ------------------------------------------------------------------ */

dn_result solve_denoise(const double *o_ref, const double *o_hat,
                        int n_exp, int n_x, int d_model,
                        int rank, double lambda, int n_train) {
    dn_result res = { 0, 0.0, 0.0, 0, 0 };

    if (!o_ref || !o_hat || n_exp < 1 || n_x < 1 || d_model < 1)
        return res;                       /* ok stays 0 */
    if (n_train < 1 || n_train >= n_x)    /* need ≥1 train AND ≥1 held-out token */
        return res;

    const int    d  = d_model;
    const size_t dd = (size_t)d * (size_t)d;
    const double N  = (double)n_exp * (double)n_train;   /* pooled train count */
    const int    n_test = n_exp * (n_x - n_train);

    int r = rank;
    if (r < 0) r = 0;
    if (r > d) r = d;                     /* sym_eig_topk clamps too, but bound buffers */
    const int ralloc = (r > 0) ? r : 1;   /* never malloc(0) */

    res.rank   = r;
    res.n_test = n_test;

    /* ---- buffers (allocated once; a couple are reused across lifetimes) ---- */
    double *Sff   = malloc(dd * sizeof(double));   /* Σ ô ôᵀ      → Ccen (in place) */
    double *Sfy   = malloc(dd * sizeof(double));   /* Σ ô o_refᵀ  → Cyfᵀ (in place) */
    double *Cffr  = malloc(dd * sizeof(double));   /* Cff = Ccen+λI (chol & So copy) */
    double *Mft   = malloc(dd * sizeof(double));   /* RHS Cyfᵀ → M_fullᵀ; then Tmp   */
    double *Mfull = malloc(dd * sizeof(double));   /* M_full (transpose of Mft)      */
    double *So    = malloc(dd * sizeof(double));   /* M_full·Cff·M_fullᵀ; then M_r   */
    double *evecs = malloc((size_t)ralloc * d * sizeof(double)); /* top-r eigvecs Q  */
    double *coeff = malloc((size_t)ralloc * d * sizeof(double)); /* Q·M_full  (r×d)  */
    double *evals = malloc((size_t)d * sizeof(double));
    double *fb    = malloc((size_t)d * sizeof(double));          /* mean ô  (train)  */
    double *yb    = malloc((size_t)d * sizeof(double));          /* mean o_ref(train)*/
    double *bvec  = malloc((size_t)d * sizeof(double));          /* bias b           */

    if (!Sff || !Sfy || !Cffr || !Mft || !Mfull || !So ||
        !evecs || !coeff || !evals || !fb || !yb || !bvec)
        goto cleanup;                     /* ok stays 0 */

    /* ---- single streaming pass over TRAIN tokens, pooled over experts ------
     * For expert e the block o_*[e] is (n_x × d) row-major, so its first
     * n_train rows ARE the train tokens and are contiguous; gemm_tn_acc with
     * k=n_train reads exactly those rows and never touches a test token. */
    memset(Sff, 0, dd * sizeof(double));
    memset(Sfy, 0, dd * sizeof(double));
    for (int a = 0; a < d; a++) { fb[a] = 0.0; yb[a] = 0.0; }

    for (int e = 0; e < n_exp; e++) {
        const double *fh = o_hat + (size_t)e * n_x * d;   /* train rows: [0,n_train) */
        const double *fr = o_ref + (size_t)e * n_x * d;
        gemm_tn_acc(Sff, fh, fh, d, n_train, d);          /* Σ ô ôᵀ      */
        gemm_tn_acc(Sfy, fh, fr, d, n_train, d);          /* Σ ô o_refᵀ  */
        for (int i = 0; i < n_train; i++) {
            const double *rh = fh + (size_t)i * d;
            const double *rr = fr + (size_t)i * d;
            for (int a = 0; a < d; a++) { fb[a] += rh[a]; yb[a] += rr[a]; }
        }
    }
    for (int a = 0; a < d; a++) { fb[a] /= N; yb[a] /= N; }

    /* Center the moments: Ccen = Σ(ô−fb)(ô−fb)ᵀ = Sff − N·fb·fbᵀ (in place in
     * Sff), and Cyfᵀ = Σ(ô−fb)(o_ref−yb)ᵀ = Sfy − N·fb·ybᵀ (in place in Sfy). */
    for (int a = 0; a < d; a++) {
        double fa = fb[a];
        double *crow = Sff + (size_t)a * d;
        double *trow = Sfy + (size_t)a * d;
        for (int b = 0; b < d; b++) {
            crow[b] -= N * fa * fb[b];     /* Ccen[a][b]  */
            trow[b] -= N * fa * yb[b];     /* Cyfᵀ[a][b]  */
        }
    }
    double *Ccen = Sff;                    /* centered feature covariance (no ridge) */
    double *CyfT = Sfy;                    /* RHS template (left untouched by solve) */

    /* ridge λ relative to the covariance scale (scale-safety) */
    double mean_diag = 0.0;
    for (int a = 0; a < d; a++) mean_diag += Ccen[(size_t)a * d + a];
    mean_diag /= (double)d;
    if (!(mean_diag > 0.0)) mean_diag = 1.0;            /* degenerate guard */
    double lam = (lambda > 0.0 ? lambda : 0.0) * mean_diag;

    /* ---- full ridge map: solve Cff·M_fullᵀ = Cyfᵀ, growing λ on non-PD ------
     * chol_solve_spd destroys A (→L) and overwrites the RHS with the solution,
     * so each attempt rebuilds Cff = Ccen+λI in Cffr and a fresh RHS copy in
     * Mft. CyfT (in Sfy) is the untouched template. */
    int    solved   = 0;
    double lam_used = 0.0;
    for (int att = 0; att < 8 && !solved; att++) {
        memcpy(Cffr, Ccen, dd * sizeof(double));
        for (int a = 0; a < d; a++) Cffr[(size_t)a * d + a] += lam;
        memcpy(Mft, CyfT, dd * sizeof(double));
        if (chol_solve_spd(Cffr, d, Mft, d) == 0) { solved = 1; lam_used = lam; }
        else lam = (lam > 0.0) ? lam * 10.0 : 1e-6 * mean_diag;
    }
    if (!solved) goto cleanup;            /* ok stays 0 */
    /* Mft now holds M_fullᵀ. Rebuild Cff = Ccen+λ_used·I for the truncation. */
    memcpy(Cffr, Ccen, dd * sizeof(double));
    for (int a = 0; a < d; a++) Cffr[(size_t)a * d + a] += lam_used;

    /* M_full = (M_fullᵀ)ᵀ */
    for (int a = 0; a < d; a++)
        for (int b = 0; b < d; b++)
            Mfull[(size_t)a * d + b] = Mft[(size_t)b * d + a];

    /* ---- rank-r reduced-rank truncation -----------------------------------
     * So = M_full·Cff·M_fullᵀ (fitted-output covariance, symmetric PSD); its
     * top-r eigenvectors Q are the dominant output directions, and
     * M_r = Qᵀ·Q·M_full is the optimal rank-r reduced-rank regression. */
    if (r > 0) {
        double *Tmp = Mft;                 /* Mft (=M_fullᵀ) no longer needed → scratch */
        gemm_nn(Tmp, Mfull, Cffr, d, d, d);    /* Tmp = M_full·Cff        */
        gemm_nt(So,  Tmp,   Mfull, d, d, d);   /* So  = Tmp·M_fullᵀ        */

        /* symmetrize defensively (round-off) before the eigensolve */
        for (int a = 0; a < d; a++)
            for (int b = a + 1; b < d; b++) {
                double m = 0.5 * (So[(size_t)a * d + b] + So[(size_t)b * d + a]);
                So[(size_t)a * d + b] = m;
                So[(size_t)b * d + a] = m;
            }

        sym_eig_topk(So, d, r, evecs, evals, 0 /*default iters*/, DN_EIG_SEED);

        gemm_nn(coeff, evecs, Mfull, r, d, d); /* coeff = Q·M_full   (r×d) */
        gemm_tn(So,    evecs, coeff, d, r, d); /* M_r   = Qᵀ·coeff   (d×d) → So */
    } else {
        memset(So, 0, dd * sizeof(double));    /* rank 0 → pure bias (pred = yb) */
    }
    double *Mr = So;                       /* the final rank-r denoiser map */

    /* bias b = yb − M_r·fb */
    for (int a = 0; a < d; a++) {
        const double *mrow = Mr + (size_t)a * d;
        double s = 0.0;
        for (int c = 0; c < d; c++) s += mrow[c] * fb[c];
        bvec[a] = yb[a] - s;
    }

    /* ---- HELD-OUT evaluation: TEST tokens [n_train,n_x) only ---------------
     * pred = M_r·ô + b ; metrics accumulated over every test (e,i). */
    double num_sq = 0.0;   /* Σ ‖pred − o_ref‖²  */
    double den_sq = 0.0;   /* Σ ‖o_ref‖²         */
    double sum_cos = 0.0;  /* Σ cos(pred, o_ref) */
    for (int e = 0; e < n_exp; e++) {
        const double *oh = o_hat + (size_t)e * n_x * d + (size_t)n_train * d;
        const double *orf = o_ref + (size_t)e * n_x * d + (size_t)n_train * d;
        for (int i = 0; i < n_x - n_train; i++) {
            const double *ohrow = oh  + (size_t)i * d;
            const double *orrow = orf + (size_t)i * d;
            double dot = 0.0, np = 0.0, nr = 0.0, diff2 = 0.0;
            for (int a = 0; a < d; a++) {
                const double *mrow = Mr + (size_t)a * d;
                double pa = bvec[a];
                for (int c = 0; c < d; c++) pa += mrow[c] * ohrow[c];
                double rf = orrow[a];
                double dl = pa - rf;
                diff2 += dl * dl;
                dot   += pa * rf;
                np    += pa * pa;
                nr    += rf * rf;
            }
            num_sq  += diff2;
            den_sq  += nr;
            double denom = sqrt(np * nr);
            sum_cos += (denom > 1e-300) ? (dot / denom) : 0.0;
        }
    }

    res.ok       = 1;
    res.rel_l2   = (den_sq > 0.0) ? sqrt(num_sq / den_sq) : 0.0;
    res.cos_mean = (n_test > 0) ? sum_cos / (double)n_test : 0.0;

cleanup:
    free(Sff);   free(Sfy);   free(Cffr);  free(Mft);  free(Mfull);
    free(So);    free(evecs); free(coeff); free(evals);
    free(fb);    free(yb);    free(bvec);
    return res;
}

/* ================================================================== */
/* Synthetic self-test (no model). Build:                              */
/*   cc -O2 -std=c11 -Wall -Wextra -DDENOISE_TEST solve_denoise.c      */
/*      linalg_small.c -o dntest -lm && ./dntest                       */
/*                                                                     */
/* Two cases prove the held-out methodology:                           */
/*   (1) LEARNABLE   — o_ref = M0·ô + b0 + tiny noise, M0 rank 4. A     */
/*       rank-4 fit GENERALIZES: held-out rel_l2 collapses, cos→1.      */
/*   (2) UNLEARNABLE — o_ref = ô + INDEPENDENT large noise. A rank-8    */
/*       fit cannot predict the noise on held-out tokens, so the        */
/*       held-out rel_l2 stays at the noise floor (≉ 0): the held-out    */
/*       metric REFUSES to reward overfitting.                          */
/* ================================================================== */
#ifdef DENOISE_TEST
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
static double sm_u01(void) {            /* [0,1) */
    return (double)(sm_next() >> 11) * (1.0 / 9007199254740992.0);
}
static double sm_normal(void) {         /* Box–Muller standard normal */
    double u1 = sm_u01(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = sm_u01();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* (1) LEARNABLE: one SHARED rank-r0 map M0 (=U0·V0) plus bias b0, applied to
 *     random ô, with tiny additive noise. Held-out rel_l2 must collapse. */
static int test_learnable(void) {
    const int    d = 32, r0 = 4, n_exp = 8, n_x = 200, n_train = 150;
    const double noise = 1e-3, lambda = 1e-3;

    double *oh = malloc((size_t)n_exp * n_x * d * sizeof(double));
    double *of = malloc((size_t)n_exp * n_x * d * sizeof(double));
    double *U0 = malloc((size_t)d * r0 * sizeof(double));
    double *V0 = malloc((size_t)r0 * d * sizeof(double));
    double *M0 = malloc((size_t)d * d * sizeof(double));
    double *b0 = malloc((size_t)d * sizeof(double));
    assert(oh && of && U0 && V0 && M0 && b0);

    /* one shared rank-r0 ground-truth map (+bias) for all experts/tokens */
    for (int a = 0; a < d; a++)
        for (int k = 0; k < r0; k++) U0[a * r0 + k] = sm_normal();
    for (int k = 0; k < r0; k++)
        for (int j = 0; j < d; j++) V0[k * d + j] = sm_normal();
    for (int a = 0; a < d; a++)
        for (int j = 0; j < d; j++) {
            double s = 0.0;
            for (int k = 0; k < r0; k++) s += U0[a * r0 + k] * V0[k * d + j];
            M0[a * d + j] = s;                       /* M0 exactly rank r0 */
        }
    for (int j = 0; j < d; j++) b0[j] = 0.5 * sm_normal();

    /* ô ~ N(0,1); o_ref = M0·ô + b0 + tiny noise */
    for (int e = 0; e < n_exp; e++)
        for (int i = 0; i < n_x; i++) {
            double *rh = oh + ((size_t)e * n_x + i) * d;
            double *rr = of + ((size_t)e * n_x + i) * d;
            for (int c = 0; c < d; c++) rh[c] = sm_normal();
            for (int a = 0; a < d; a++) {
                double s = b0[a] + noise * sm_normal();
                for (int c = 0; c < d; c++) s += M0[a * d + c] * rh[c];
                rr[a] = s;
            }
        }

    dn_result r = solve_denoise(of, oh, n_exp, n_x, d, r0, lambda, n_train);

    printf("(1) LEARNABLE  (o_ref = M0·ô + b0 + noise, rank(M0)=%d)\n", r0);
    printf("    d=%d n_exp=%d n_x=%d  TRAIN=%d  TEST=%d  fit rank=%d\n",
           d, n_exp, n_x, n_train, r.n_test, r.rank);
    printf("    HELD-OUT TEST:  ok=%d  rel_l2=%.6e  cos=%.6f\n",
           r.ok, r.rel_l2, r.cos_mean);

    int pass = r.ok && (r.n_test == n_exp * (n_x - n_train)) &&
               (r.rel_l2 < 0.10) && (r.cos_mean > 0.95);
    printf("    => %s  (expect rel_l2<0.10 AND cos>0.95: GENERALIZES)\n\n",
           pass ? "PASS" : "FAIL");

    free(oh); free(of); free(U0); free(V0); free(M0); free(b0);
    return pass;
}

/* (2) UNLEARNABLE CONTROL: o_ref = ô + INDEPENDENT large noise (the noise is
 *     not a function of ô). No map can predict the held-out noise, so the
 *     held-out rel_l2 must stay at the noise floor ≈ σ/√(1+σ²), NOT ~0. */
static int test_control(void) {
    const int    d = 32, n_exp = 8, n_x = 200, n_train = 150;
    const double sigma = 2.0, lambda = 1e-3;
    const int    rank = 8;

    double *oh = malloc((size_t)n_exp * n_x * d * sizeof(double));
    double *of = malloc((size_t)n_exp * n_x * d * sizeof(double));
    assert(oh && of);

    for (int e = 0; e < n_exp; e++)
        for (int i = 0; i < n_x; i++) {
            double *rh = oh + ((size_t)e * n_x + i) * d;
            double *rr = of + ((size_t)e * n_x + i) * d;
            for (int c = 0; c < d; c++) {
                rh[c] = sm_normal();                 /* ô ~ N(0,1)                 */
                rr[c] = rh[c] + sigma * sm_normal(); /* o_ref = ô + INDEP. noise   */
            }
        }

    dn_result r = solve_denoise(of, oh, n_exp, n_x, d, rank, lambda, n_train);

    double floor = sigma / sqrt(1.0 + sigma * sigma);   /* irreducible held-out rel_l2 */
    printf("(2) UNLEARNABLE CONTROL  (o_ref = ô + N(0,%.1f²) independent noise)\n", sigma);
    printf("    d=%d n_exp=%d n_x=%d  TRAIN=%d  TEST=%d  fit rank=%d\n",
           d, n_exp, n_x, n_train, r.n_test, r.rank);
    printf("    HELD-OUT TEST:  ok=%d  rel_l2=%.6e  cos=%.6f\n",
           r.ok, r.rel_l2, r.cos_mean);
    printf("    noise floor σ/√(1+σ²) = %.6f  (a perfect map cannot beat this)\n",
           floor);

    /* The held-out error must NOT collapse to ~0 (overfitting would have, had
     * we scored on TRAIN). It stays near the noise floor. */
    int pass = r.ok && (r.rel_l2 > 0.5);
    printf("    => %s  (expect rel_l2 does NOT collapse to ~0: refuses overfitting)\n\n",
           pass ? "PASS" : "FAIL");

    free(oh); free(of);
    return pass;
}

int main(void) {
    g_sm = 0xC0FFEEULL;
    printf("shared reduced-rank denoiser (o_ref ≈ M·ô + b) — held-out self-test\n\n");
    int a = test_learnable();
    int b = test_control();
    printf("RESULT: learnable=%s  control=%s  =>  %s\n",
           a ? "PASS" : "FAIL", b ? "PASS" : "FAIL",
           (a && b) ? "ALL PASS" : "FAIL");
    return (a && b) ? 0 : 1;
}
#endif /* DENOISE_TEST */
