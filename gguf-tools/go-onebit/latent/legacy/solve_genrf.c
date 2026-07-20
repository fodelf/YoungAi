/* solve_genrf.c — closed-form random-feature NONLINEAR generator (see
 * solve_genrf.h for the why and the math).
 *
 * Pipeline (no SGD):
 *   1. Deterministically materialize a fixed random nonlinear map R (n_feat×2d),
 *      bias c (n_feat) from `seed` via splitmix64 → Box-Muller Gaussians.
 *   2. ONE streaming pass over TRAIN tokens accumulates the uncentered moments
 *      S_pp = Σ phi·phiᵀ (lower triangle), S_py = Σ phi·o_refᵀ, and the means
 *      s_p = Σ phi, s_y = Σ o_ref.  phi = SiLU(R·[ô;x]+c) is recomputed per token
 *      and never stored, so peak memory is O(n_feat² + n_feat·d_model).
 *   3. Form the CENTERED Gram via the rank-1 corrections
 *        Gpp = S_pp − s_p·s_pᵀ/N + λI ,  Gpy = S_py − s_p·s_yᵀ/N
 *      (centering folded into the intercept b), solve Gpp·Z = Gpy by Cholesky
 *      (chol_solve_spd), giving the ridge readout M = Zᵀ and b = ȳ − M·φ̄.
 *   4. Recompute phi on the HELD-OUT TEST tokens and score global rel_l2 and the
 *      mean per-sample cosine.  The fit never touches test tokens.
 */
#include "solve_genrf.h"
#include "linalg_small.h"

#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define GR_TWO_PI         6.283185307179586476925286766559
#define GR_BIAS_STD       0.1   /* c ~ N(0, 0.1²): small offset, breaks feature symmetry */
#define GR_MAX_RIDGE_TRIES 8     /* ×10 ridge growth attempts before declaring non-PD     */

/* ---- deterministic Gaussian RNG: splitmix64 + cached Box-Muller ---- */
typedef struct { uint64_t s; double cache; int has_cache; } gr_rng;

static inline uint64_t gr_sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
/* uniform in the open interval (0,1) — never 0 (safe for log) nor 1 */
static inline double gr_uniform(gr_rng *r) {
    uint64_t z = gr_sm64(&r->s);
    return ((double)(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static double gr_normal(gr_rng *r) {
    if (r->has_cache) { r->has_cache = 0; return r->cache; }
    double u1 = gr_uniform(r), u2 = gr_uniform(r);
    double mag = sqrt(-2.0 * log(u1));
    r->cache = mag * sin(GR_TWO_PI * u2);
    r->has_cache = 1;
    return mag * cos(GR_TWO_PI * u2);
}

/* phi = SiLU(R·f + c), with f = [ o_hat (d) ; x (d) ] formed implicitly.
 * SiLU(z) = z·sigmoid(z); for very negative z, exp(-z) overflows to +inf so the
 * sigmoid underflows to 0 and phi → 0 (no NaN as long as z is finite). */
static void gr_features(double *phi, const double *oh, const double *xx,
                        const double *R, const double *c,
                        int n_feat, int d_model) {
    int twoD = 2 * d_model;
    for (int k = 0; k < n_feat; k++) {
        const double *Rk = R + (size_t)k * twoD;
        double z = c[k];
        for (int j = 0; j < d_model; j++) z += Rk[j]          * oh[j];
        for (int j = 0; j < d_model; j++) z += Rk[d_model + j] * xx[j];
        double sig = 1.0 / (1.0 + exp(-z));
        phi[k] = z * sig;
    }
}

gr_result solve_genrf(const double *o_ref, const double *o_hat, const double *x,
                      int n_exp, int n_x, int d_model, int n_feat,
                      double lambda, int n_train, uint64_t seed) {
    gr_result res = { 0, 0.0, 0.0, 0, 0 };

    if (!o_ref || !o_hat || !x) return res;
    if (n_exp < 1 || n_x < 2 || d_model < 1 || n_feat < 1) return res;
    if (n_train < 1 || n_train > n_x - 1) return res;
    if (lambda < 0.0) return res;

    const int    d    = d_model, nf = n_feat, twoD = 2 * d_model;
    const double Ntr  = (double)n_exp * (double)n_train;

    const size_t szR  = (size_t)nf * (size_t)twoD;  /* random map R           */
    const size_t szPP = (size_t)nf * (size_t)nf;    /* Gram (n_feat²)         */
    const size_t szPY = (size_t)nf * (size_t)d;     /* cross-Gram / readout   */

    double *R   = malloc(szR  * sizeof(double));
    double *c   = malloc((size_t)nf * sizeof(double));
    double *Spp = calloc(szPP, sizeof(double));     /* Σ phi·phiᵀ (lower tri) */
    double *Spy = calloc(szPY, sizeof(double));     /* Σ phi·o_refᵀ           */
    double *sp  = calloc((size_t)nf, sizeof(double)); /* Σ phi                */
    double *sy  = calloc((size_t)d,  sizeof(double)); /* Σ o_ref              */
    double *phi = malloc((size_t)nf * sizeof(double));
    double *Gpp = malloc(szPP * sizeof(double));    /* centered Gram, chol-destroyed */
    double *Gpy = malloc(szPY * sizeof(double));    /* RHS → Z = Gpp⁻¹·Gpy    */
    double *M   = malloc(szPY * sizeof(double));    /* readout, d×nf (= Zᵀ)   */
    double *b   = malloc((size_t)d * sizeof(double));

    if (!R || !c || !Spp || !Spy || !sp || !sy || !phi || !Gpp || !Gpy || !M || !b) {
        free(R); free(c); free(Spp); free(Spy); free(sp); free(sy);
        free(phi); free(Gpp); free(Gpy); free(M); free(b);
        return res;
    }

    /* ---- fixed random nonlinear feature map (deterministic in `seed`) ----
     * R entries ~ N(0, 1/(2d)) so z = R·f is O(1) when f is O(1); c ~ N(0,small). */
    gr_rng rng = { seed, 0.0, 0 };
    const double r_std = 1.0 / sqrt((double)twoD);
    for (size_t i = 0; i < szR; i++) R[i] = gr_normal(&rng) * r_std;
    for (int    k = 0; k < nf;  k++) c[k] = gr_normal(&rng) * GR_BIAS_STD;

    /* ---- TRAIN: one streaming pass over tokens [0,n_train) of every expert ---- */
    for (int e = 0; e < n_exp; e++) {
        for (int i = 0; i < n_train; i++) {
            const double *oh = o_hat + ((size_t)e * n_x + i) * d;
            const double *xr = x     + (size_t)i * d;
            const double *yr = o_ref + ((size_t)e * n_x + i) * d;
            gr_features(phi, oh, xr, R, c, nf, d);
            for (int k = 0; k < nf; k++) {
                double pk = phi[k];
                sp[k] += pk;
                double *Sk = Spp + (size_t)k * nf;
                for (int j = 0; j <= k; j++) Sk[j] += pk * phi[j];   /* lower triangle */
                double *Yk = Spy + (size_t)k * d;
                for (int j = 0; j < d;  j++) Yk[j] += pk * yr[j];
            }
            for (int j = 0; j < d; j++) sy[j] += yr[j];
        }
    }

    /* ---- ridge readout via centered normal equations, with PD-retry ----
     * Centered Gram: Gpp = S_pp − s_p·s_pᵀ/N (+λI), Gpy = S_py − s_p·s_yᵀ/N.
     * chol_solve_spd reads only the lower triangle but we fill both for safety;
     * it overwrites Gpy with Z = Gpp⁻¹·Gpy, so we rebuild from the moments on each
     * retry. λ is relative to the mean centered diagonal (scale-safe). */
    double eff_lambda = (lambda > 0.0) ? lambda : 0.0;
    int solved = 0;
    for (int attempt = 0; attempt < GR_MAX_RIDGE_TRIES && !solved; attempt++) {
        double dmean = 0.0;
        for (int k = 0; k < nf; k++) dmean += Spp[(size_t)k * nf + k] - sp[k] * sp[k] / Ntr;
        dmean /= (double)nf;
        if (dmean < 1e-12) dmean = 1e-12;
        double lam_abs = eff_lambda * dmean;

        for (int k = 0; k < nf; k++) {
            const double *Sk = Spp + (size_t)k * nf;
            for (int j = 0; j <= k; j++) {
                double v = Sk[j] - sp[k] * sp[j] / Ntr;
                if (j == k) v += lam_abs;
                Gpp[(size_t)k * nf + j] = v;
                Gpp[(size_t)j * nf + k] = v;
            }
        }
        for (int k = 0; k < nf; k++) {
            const double *Sk = Spy + (size_t)k * d;
            double       *Gk = Gpy + (size_t)k * d;
            for (int j = 0; j < d; j++) Gk[j] = Sk[j] - sp[k] * sy[j] / Ntr;
        }

        if (chol_solve_spd(Gpp, nf, Gpy, d) == 0) solved = 1;
        else eff_lambda = (eff_lambda > 0.0) ? eff_lambda * 10.0 : 1e-6;
    }
    if (!solved) {
        free(R); free(c); free(Spp); free(Spy); free(sp); free(sy);
        free(phi); free(Gpp); free(Gpy); free(M); free(b);
        return res;  /* ok stays 0: Gram never positive-definite */
    }

    /* M = Zᵀ (d×nf) and b = ȳ − M·φ̄.  Z (=Gpy) is n_feat×d_model, M[j][k]=Z[k][j]. */
    for (int j = 0; j < d; j++) {
        double mb = 0.0;
        for (int k = 0; k < nf; k++) {
            double mjk = Gpy[(size_t)k * d + j];
            M[(size_t)j * nf + k] = mjk;
            mb += mjk * (sp[k] / Ntr);
        }
        b[j] = sy[j] / Ntr - mb;
    }

    /* ---- HELD-OUT TEST: recompute phi on tokens [n_train,n_x) and score ---- */
    double sse = 0.0, refsq = 0.0, cos_sum = 0.0;
    long n_test = 0;
    for (int e = 0; e < n_exp; e++) {
        for (int i = n_train; i < n_x; i++) {
            const double *oh = o_hat + ((size_t)e * n_x + i) * d;
            const double *xr = x     + (size_t)i * d;
            const double *yr = o_ref + ((size_t)e * n_x + i) * d;
            gr_features(phi, oh, xr, R, c, nf, d);
            double dot = 0.0, pn = 0.0, rn = 0.0;
            for (int j = 0; j < d; j++) {
                const double *Mj = M + (size_t)j * nf;
                double p = b[j];
                for (int k = 0; k < nf; k++) p += Mj[k] * phi[k];
                double diff = p - yr[j];
                sse   += diff * diff;
                refsq += yr[j] * yr[j];
                dot   += p * yr[j];
                pn    += p * p;
                rn    += yr[j] * yr[j];
            }
            double denom = sqrt(pn) * sqrt(rn);
            if (denom > 0.0) cos_sum += dot / denom;
            n_test++;
        }
    }

    res.ok       = 1;
    res.rel_l2   = (refsq > 0.0)  ? sqrt(sse / refsq)        : 0.0;
    res.cos_mean = (n_test > 0)   ? cos_sum / (double)n_test : 0.0;
    res.n_feat   = nf;
    res.n_test   = (int)n_test;

    free(R); free(c); free(Spp); free(Spy); free(sp); free(sy);
    free(phi); free(Gpp); free(Gpy); free(M); free(b);
    return res;
}

/* =====================================================================
 * Synthetic self-test (no model):  cc ... -DGENRF_TEST ...
 *   (1) NONLINEAR LEARNABLE : o_ref = A·SiLU(B·f) + tiny noise   → held-out cos > 0.9
 *   (2) LINEAR control      : o_ref = C·ô + tiny noise           → held-out cos > 0.9
 *   (3) UNLEARNABLE control : o_ref = ô + large independent noise → held-out rel_l2 ≫ 0
 * ===================================================================== */
#ifdef GENRF_TEST
#include <stdio.h>

static void gr_fill_normal(double *a, size_t n, gr_rng *r, double scale) {
    for (size_t i = 0; i < n; i++) a[i] = gr_normal(r) * scale;
}
static double gr_silu(double z) { return z / (1.0 + exp(-z)); }

int main(void) {
    const int d = 24, ne = 6, nx = 400, ntr = 300, nf = 512, twoD = 2 * 24;
    const size_t NEXD = (size_t)ne * nx * d;
    const double lambda = 1e-3;
    int fails = 0;

    /* ---------- Case 1: NONLINEAR LEARNABLE ---------- */
    {
        const int k = 4;   /* small hidden width */
        const double bscale = 0.6 / sqrt((double)twoD);  /* B·f ~ 0.6: gentle, smoothly nonlinear
                                                          * SiLU regime (real quadratic content, but
                                                          * inside the random-feature RKHS) */
        double *oh = malloc(NEXD * sizeof(double));
        double *xx = malloc((size_t)nx * d * sizeof(double));
        double *yr = malloc(NEXD * sizeof(double));
        double *B  = malloc((size_t)k * twoD * sizeof(double));
        double *A  = malloc((size_t)d * k * sizeof(double));
        double *fb = malloc((size_t)twoD * sizeof(double));
        double *hh = malloc((size_t)k * sizeof(double));
        gr_rng rg = { 0xABCDEF0123456789ULL, 0.0, 0 };
        gr_fill_normal(oh, NEXD, &rg, 1.0);
        gr_fill_normal(xx, (size_t)nx * d, &rg, 1.0);
        gr_fill_normal(B, (size_t)k * twoD, &rg, bscale);
        gr_fill_normal(A, (size_t)d * k, &rg, 1.0);
        double ss = 0.0; size_t cnt = 0;
        for (int e = 0; e < ne; e++) for (int i = 0; i < nx; i++) {
            const double *o = oh + ((size_t)e * nx + i) * d;
            const double *xv = xx + (size_t)i * d;
            for (int j = 0; j < d; j++) fb[j]     = o[j];
            for (int j = 0; j < d; j++) fb[d + j] = xv[j];
            for (int t = 0; t < k; t++) {
                double z = 0.0; for (int j = 0; j < twoD; j++) z += B[t * twoD + j] * fb[j];
                hh[t] = gr_silu(z);
            }
            double *y = yr + ((size_t)e * nx + i) * d;
            for (int j = 0; j < d; j++) {
                double v = 0.0; for (int t = 0; t < k; t++) v += A[j * k + t] * hh[t];
                y[j] = v; ss += v * v; cnt++;
            }
        }
        double nstd = 0.01 * sqrt(ss / (double)cnt);             /* ~1% noise */
        for (size_t m = 0; m < NEXD; m++) yr[m] += gr_normal(&rg) * nstd;

        gr_result r = solve_genrf(yr, oh, xx, ne, nx, d, nf, lambda, ntr, 0x55AA55AA55AA55AAULL);
        printf("[case1 nonlinear-learnable] ok=%d  held-out cos=%.4f  rel_l2=%.4f  n_feat=%d n_test=%d\n",
               r.ok, r.cos_mean, r.rel_l2, r.n_feat, r.n_test);
        if (!(r.ok && r.cos_mean > 0.9)) { printf("  FAIL: expected held-out cos > 0.9\n"); fails++; }
        else                              printf("  PASS\n");
        free(oh); free(xx); free(yr); free(B); free(A); free(fb); free(hh);
    }

    /* ---------- Case 2: LINEAR control ---------- */
    {
        double *oh = malloc(NEXD * sizeof(double));
        double *xx = malloc((size_t)nx * d * sizeof(double));
        double *yr = malloc(NEXD * sizeof(double));
        double *C  = malloc((size_t)d * d * sizeof(double));
        gr_rng rg = { 0x1357246813572468ULL, 0.0, 0 };
        gr_fill_normal(oh, NEXD, &rg, 1.0);
        gr_fill_normal(xx, (size_t)nx * d, &rg, 1.0);
        gr_fill_normal(C, (size_t)d * d, &rg, 1.0 / sqrt((double)d));        /* C·ô ~ O(1) */
        double ss = 0.0; size_t cnt = 0;
        for (int e = 0; e < ne; e++) for (int i = 0; i < nx; i++) {
            const double *o = oh + ((size_t)e * nx + i) * d;
            double *y = yr + ((size_t)e * nx + i) * d;
            for (int j = 0; j < d; j++) {
                double v = 0.0; for (int m = 0; m < d; m++) v += C[j * d + m] * o[m];
                y[j] = v; ss += v * v; cnt++;
            }
        }
        double nstd = 0.01 * sqrt(ss / (double)cnt);
        for (size_t m = 0; m < NEXD; m++) yr[m] += gr_normal(&rg) * nstd;

        gr_result r = solve_genrf(yr, oh, xx, ne, nx, d, nf, lambda, ntr, 0x77BB77BB77BB77BBULL);
        printf("[case2 linear-control]      ok=%d  held-out cos=%.4f  rel_l2=%.4f  n_feat=%d n_test=%d\n",
               r.ok, r.cos_mean, r.rel_l2, r.n_feat, r.n_test);
        if (!(r.ok && r.cos_mean > 0.9)) { printf("  FAIL: expected held-out cos > 0.9\n"); fails++; }
        else                              printf("  PASS\n");
        free(oh); free(xx); free(yr); free(C);
    }

    /* ---------- Case 3: UNLEARNABLE control ---------- */
    {
        double *oh = malloc(NEXD * sizeof(double));
        double *xx = malloc((size_t)nx * d * sizeof(double));
        double *yr = malloc(NEXD * sizeof(double));
        gr_rng rg = { 0x0F0F0F0F0F0F0F0FULL, 0.0, 0 };
        gr_fill_normal(oh, NEXD, &rg, 1.0);
        gr_fill_normal(xx, (size_t)nx * d, &rg, 1.0);
        double ss = 0.0; for (size_t m = 0; m < NEXD; m++) ss += oh[m] * oh[m];
        double nstd = 2.0 * sqrt(ss / (double)NEXD);             /* large, INDEPENDENT of f */
        for (size_t m = 0; m < NEXD; m++) yr[m] = oh[m] + gr_normal(&rg) * nstd;

        gr_result r = solve_genrf(yr, oh, xx, ne, nx, d, nf, lambda, ntr, 0x99CC99CC99CC99CCULL);
        printf("[case3 unlearnable-control] ok=%d  held-out rel_l2=%.4f  cos=%.4f  n_feat=%d n_test=%d\n",
               r.ok, r.rel_l2, r.cos_mean, r.n_feat, r.n_test);
        if (!(r.ok && r.rel_l2 > 0.5)) { printf("  FAIL: expected held-out rel_l2 > 0.5 (no overfit reward)\n"); fails++; }
        else                            printf("  PASS\n");
        free(oh); free(xx); free(yr);
    }

    printf("\n%s (%d failure%s)\n",
           fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
#endif /* GENRF_TEST */
