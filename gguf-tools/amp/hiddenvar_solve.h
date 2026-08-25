/* hiddenvar_solve.h — per-layer CLOSED-FORM solver for the Go-domain 1-bit
 * quantizer's "hidden variable" z^ℓ (SPEC.md §5/§6/§7).
 *
 * The 1-bit fixed model Θ_fix reproduces each routed expert's output only
 * approximately: ô_e(x) ≈ o_e(x). The per-layer correction restores Go-domain
 * behavior in ACTIVATION space (not weight values) via a shared low-rank map
 * plus per-expert diagonal gains:
 *
 *     o_e^corr(x) = ô_e(x) + U·diag(C_e)·(V·x) + β_e·1 + b
 *
 *   U (d_model×d_ℓ), V (d_ℓ×d_model), b (d_model)  : per-layer shared
 *   C (n_exp×d_ℓ), β (n_exp)                        : per-expert coefficients
 *   δ (n_exp)                                       : router-logit correction
 *
 * We FIT this to the residuals Δo_{e,i} = o_ref_{e,i} − ô_{e,i} measured by the
 * layer probe on a shared Go activation pool {x_i}, in STAGED CLOSED FORM —
 * NO gradient descent, NO epochs (this is quantization, not training). The
 * objective is the weighted sum of FOUR losses, each its own C function:
 *
 *   loss_classify (routing)   → δ                      (mean router-prob gap)
 *   loss_fixed    (noise)     → Tikhonov λI + dither   (pins underdetermined)
 *   loss_smooth   (smoothness)→ first-difference Gram   (no jumps across i)
 *   loss_align    (direction) → inverse-norm reweight   (preserves ranking)
 *
 * Solve = global-mean bias → output PCA basis U & rank d_ℓ → coordinate
 * projection → pooled ridge for V → per-expert 1-D regressions for C/β →
 * router δ. One closed-form pass, deterministic (splitmix64 dither, no rand()).
 *
 * Pure C99. Links only against linalg_small.h (chol_solve_spd + sym_eig_topk).
 * All matrices are row-major, double precision.
 */
#ifndef HIDDENVAR_SOLVE_H
#define HIDDENVAR_SOLVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* The per-layer hidden variable z^ℓ (SPEC.md §4 product ②).           */
/* ------------------------------------------------------------------ */
typedef struct {
    int    d_model;   /* output/input width (4096 in production)              */
    int    n_exp;     /* routed experts (256 in production)                   */
    int    d_l;       /* per-layer rank d_ℓ (chosen by energy threshold)      */
    double *U;        /* [d_model*d_l] row-major, U[j*d_l+k] (output basis)   */
    double *V;        /* [d_l*d_model] row-major, V[k*d_model+j] (in feature) */
    double *C;        /* [n_exp*d_l]   row-major, per-expert diagonal gains   */
    double *b;        /* [d_model]     global output bias                     */
    double *beta;     /* [n_exp]       per-expert scalar offset               */
    double *delta;    /* [n_exp]       router-logit correction                */
} z_layer;

/* Eigen-spectrum of the residual covariance Σ (SPEC.md §6.3 "表 P"):
 * how the per-layer effective rank d_ℓ was chosen. */
typedef struct {
    int    n;             /* number of eigenvalues captured (= requested kmax)*/
    double *evals;        /* [n] eigenvalues, descending                      */
    double total_energy;  /* trace(Σ) = sum of ALL eigenvalues (exact)        */
    int    d_l;           /* chosen rank                                      */
    double threshold;     /* energy fraction threshold used                   */
} z_spectrum;

/* Reconstruction fidelity of o^corr vs o_ref, per layer (SPEC.md "表 B"). */
typedef struct {
    double rel_l2;        /* ‖o^corr − o_ref‖_F / ‖o_ref‖_F (global)          */
    double cosine_mean;   /* mean over (e,i) of cos(o^corr_{e,i}, o_ref_{e,i})*/
    int    n_exp, n_x;
} z_fidelity;

/* Solver knobs (SPEC.md §8 旋钮). hv_params_default() fills sane values that
 * reproduce a clean closed-form least-squares solve. */
typedef struct {
    double   energy_threshold; /* d_ℓ = smallest rank with this energy frac   */
    int      max_rank;         /* cap on d_ℓ / # eigenpairs requested         */
    double   lambda;           /* Tikhonov ridge, RELATIVE to mean Gram diag  */
    double   w_classify;       /* weight of loss_classify (→ δ)               */
    double   w_fixed;          /* weight of loss_fixed (ridge + dither)       */
    double   w_smooth;         /* weight of loss_smooth (first-difference)    */
    double   w_align;          /* weight of loss_align (inverse-norm reweight)*/
    uint64_t seed;            /* eig start vectors + fixed-loss dither        */
    int      eig_iters;        /* power-iteration budget (<=0 → lib default)  */
} hv_params;

void hv_params_default(hv_params *p);

/* ------------------------------------------------------------------ */
/* The FOUR losses — each contributes to the assembled normal equations */
/* / objective (SPEC.md §7). Documented in the .c at the point of use.   */
/* ------------------------------------------------------------------ */

/* Loss 1 — classification/routing. Closed-form per-expert additive bias
 * δ_e = w · mean_i(probs_ref[i][e] − probs_1bit[i][e]) that realigns the
 * 1-bit router's expected top-6 selection with the reference. Probabilities
 * are [n_route][n_exp] row-major. If either array is NULL → δ=0. */
void loss_classify(const double *probs_ref, const double *probs_1bit,
                   int n_route, int n_exp, double w_classify,
                   double *delta_out /*[n_exp]*/);

/* Loss 2 — fixed/noise. Adds the Tikhonov ridge w·λ·I to a square n×n Gram
 * (λ relative to the mean diagonal, so scale-free) PLUS a deterministic
 * per-diagonal splitmix64 dither (seeded by `seed`, never rand()) that pins
 * the under-determined directions to a unique, bit-reproducible solution. */
void loss_fixed(double *gram, int n, double lambda_rel, double w_fixed,
                uint64_t seed);

/* Loss 3 — smoothness. Folds the first-difference penalty
 * w·n_exp·Σ_i (x_{i+1}−x_i)(x_{i+1}−x_i)ᵀ into the d_model×d_model Gram
 * (target 0, no RHS), so the fitted correction does not jump between
 * adjacent calibration positions i. */
void loss_smooth(double *gram, const double *x, int n_x, int d_model,
                 int n_exp, double w_smooth);

/* Loss 4 — alignment. Single closed-form reweight pass: per-sample weight
 * a_{e,i} = (1−w) + w·(mean‖ref‖ / ‖ref_{e,i}‖) so large-norm outputs do not
 * dominate and per-expert direction (logit ranking) is preserved. `ref` is
 * [n_exp][n_x][d_model] (o_ref if available, else Δo). w=0 → uniform. */
void loss_align(const double *ref, int n_exp, int n_x, int d_model,
                double w_align, double *weights_out /*[n_exp*n_x]*/);

/* ------------------------------------------------------------------ */
/* The staged closed-form solve.                                       */
/* ------------------------------------------------------------------ */

/* Solve z^ℓ for one layer.
 *   delta_o   : [n_exp][n_x][d_model] residuals Δo = o_ref − ô (required)
 *   x         : [n_x][d_model] shared Go activation pool (required)
 *   o_ref     : [n_exp][n_x][d_model] reference outputs, or NULL. Used only
 *               for loss_align norms; NULL → align falls back to Δo norms.
 *   probs_ref,
 *   probs_1bit: [n_route][n_exp] router probabilities, or NULL → δ=0.
 *   out       : populated; arrays malloc'd here, free with z_layer_free().
 *   spec_out  : optional (may be NULL); free with z_spectrum_free().
 * Returns 0 on success, nonzero on bad args / non-PD even after ridge growth.
 */
int hv_solve(const double *delta_o, const double *x, const double *o_ref,
             const double *probs_ref, const double *probs_1bit,
             int n_exp, int n_x, int d_model, int n_route,
             const hv_params *params, z_layer *out, z_spectrum *spec_out);

/* Reconstruction fidelity (表 B). o_hat is the 1-bit output ô
 * [n_exp][n_x][d_model], or NULL (treated as 0 → measures the correction
 * directly against Δo, which is what the synthetic self-test uses). */
void hv_fidelity(const z_layer *z, const double *delta_o, const double *x,
                 const double *o_hat, int n_exp, int n_x, int d_model,
                 z_fidelity *out);

/* Flat-file (de)serialization of z^ℓ — the offline solver's interchange
 * format. GGUF-tensor embedding is a later integration step (see .c). */
int  z_layer_save(const z_layer *z, const char *path);
int  z_layer_load(z_layer *z, const char *path);
void z_layer_free(z_layer *z);
void z_spectrum_free(z_spectrum *s);

#ifdef __cplusplus
}
#endif

#endif /* HIDDENVAR_SOLVE_H */
