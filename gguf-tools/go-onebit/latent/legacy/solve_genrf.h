/* solve_genrf.h — closed-form RANDOM-FEATURE NONLINEAR generator M_φ for the
 * Go-domain 1-bit expert quantizer.
 *
 * Motivation. LINEAR activation-space reconstruction of the 1-bit expert error
 * failed: neither correcting Δo = o_ref − ô from the layer input x (high-rank,
 * x-unpredictable) nor refining ô itself by ONE shared linear map (solve_denoise)
 * beat the ~0.55 held-out cosine baseline — a linear o_ref ≈ M·ô + b simply does
 * not have the capacity. The next lever is a small per-layer NONLINEAR generator
 *
 *     o_ref ≈ M_φ( ô , x )                       (ONE generator, SHARED over experts)
 *
 * fit WITHOUT gradient descent via a random-feature kernel approximation: lift the
 * concatenated feature f = [ô ; x] through a FIXED random nonlinear map and fit a
 * closed-form ridge readout on top:
 *
 *     f    = [ o_hat_{e,i} (d_model) ; x_i (d_model) ]        (length 2·d_model)
 *     z    = R·f + c                                          (R fixed N(0,1/2d), c small)
 *     phi  = SiLU(z) = z·sigmoid(z)                           (length n_feat)
 *     o_ref ≈ M·phi + b                                       (M is d_model×n_feat)
 *
 * The readout (M,b) is the ridge least-squares solution of the normal equations
 *
 *     Gpp = Σ_train (phi−φ̄)(phi−φ̄)ᵀ + λI       (n_feat×n_feat, λ rel. to mean diag)
 *     Gpy = Σ_train (phi−φ̄)(o_ref−ȳ)ᵀ           (n_feat×d_model)
 *     M   = (Gpp⁻¹·Gpy)ᵀ        (via chol_solve_spd)     b = ȳ − M·φ̄
 *
 * This is the no-SGD CLOSED-FORM INITIALIZATION of the generator (a light
 * fine-tuning pass comes later). Because R is a fixed nonlinear map and the
 * readout is linear in phi, the whole fit stays closed form, yet M_φ can
 * represent nonlinear functions of (ô,x) that the linear denoiser cannot.
 *
 * CRITICAL — held-out evaluation. A d_model×n_feat readout (n_feat≫d_model) over
 * a handful of tokens overfits, after which IN-SAMPLE error is meaningless.
 * solve_genrf fits ONLY on TRAIN tokens [0,n_train) and reports rel_l2 / cos on
 * the HELD-OUT TEST tokens [n_train,n_x). Only those numbers say whether the
 * generator GENERALIZES. The synthetic self-test (-DGENRF_TEST) proves the
 * methodology on three controls: a genuinely NONLINEAR learnable target (held-out
 * cosine collapses toward 1 — random features approximate the nonlinear map), a
 * LINEAR target (also recovered — random features represent linear maps), and an
 * independent large-noise target (held-out rel_l2 stays at the noise floor — the
 * held-out metric refuses to reward overfitting).
 *
 * Memory. Samples are streamed: one pass over TRAIN accumulates the Gram matrices
 * (never storing per-sample phi), and TEST recomputes phi on the fly. Peak working
 * set is O(n_feat² + n_feat·d_model).
 *
 * Pure C11, row-major double precision. Links only linalg_small.h
 * (chol_solve_spd). No C++, no gradient descent, no external libraries.
 */
#ifndef SOLVE_GENRF_H
#define SOLVE_GENRF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Held-out fidelity of the random-feature nonlinear generator. */
typedef struct {
    int    ok;        /* 1 on success; 0 on bad args / non-PD Gram / OOM          */
    double rel_l2;    /* HELD-OUT ‖pred − o_ref‖_F / ‖o_ref‖_F over test (e,i)    */
    double cos_mean;  /* mean over test (e,i) of cos(pred_{e,i}, o_ref_{e,i})     */
    int    n_feat;    /* number of random features actually used                  */
    int    n_test;    /* held-out samples scored = n_exp·(n_x − n_train)          */
} gr_result;

/* Fit ONE shared random-feature generator  o_ref ≈ M·SiLU(R·[ô;x]+c) + b  on the
 * TRAIN split and report fidelity on the HELD-OUT TEST split.
 *
 *   o_ref   : [n_exp][n_x][d_model] reference (high-precision) outputs (required)
 *   o_hat   : [n_exp][n_x][d_model] 1-bit outputs ô                    (required)
 *   x       : [n_x][d_model] layer inputs, SHARED across experts       (required)
 *   n_exp,n_x,d_model : dimensions (n_exp≥1, n_x≥2, d_model≥1)
 *   n_feat  : number of fixed random features (≥1; default in practice ~1024)
 *   lambda  : Tikhonov ridge, RELATIVE to the mean diagonal of the centered phi
 *             covariance (scale-safe); grown automatically (×10) if the Gram is
 *             not positive-definite within working precision
 *   n_train : tokens [0,n_train) of EACH expert are TRAIN (pooled across experts);
 *             tokens [n_train,n_x) are HELD-OUT TEST. Requires 1 ≤ n_train ≤ n_x−1.
 *   seed    : seeds the deterministic random feature map (R,c); identical seed ⇒
 *             bit-identical features. No rand()/time() is used.
 *
 * Samples are POOLED over all experts e (one shared generator, not per-expert).
 * The fit never reads test tokens; the eval never reads train tokens.
 *
 * Returns gr_result; on bad arguments / numerical failure ok=0 and the metric
 * fields are 0.
 */
gr_result solve_genrf(const double *o_ref, const double *o_hat, const double *x,
                      int n_exp, int n_x, int d_model, int n_feat,
                      double lambda, int n_train, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* SOLVE_GENRF_H */
