/* solve_denoise.h — shared reduced-rank "denoiser" map for the Go-domain
 * 1-bit expert quantizer.
 *
 * Motivation. Correcting the 1-bit expert error Δo = o_ref − ô from the layer
 * INPUT x failed: Δo is high-rank and x-unpredictable (R²≈0.04). The remaining
 * lever treats the 1-bit OUTPUT ô itself as the feature and asks whether a
 * single per-layer linear map can REFINE ô toward the high-precision reference
 * o_ref. The two are only ~0.59 cosine-aligned, so a learned map may sharpen
 * them:
 *
 *     o_ref ≈ M · ô + b ,   rank(M) ≤ rank      (ONE map, SHARED over experts)
 *
 * fit in CLOSED FORM (reduced-rank ridge regression — no gradient descent):
 *
 *   fb  = mean ô over TRAIN tokens        yb = mean o_ref over TRAIN tokens
 *   Cff = Σ_train (ô−fb)(ô−fb)ᵀ + λ·I     (d×d feature covariance; λ ridge,
 *                                          scaled relative to mean diag)
 *   Cyf = Σ_train (o_ref−yb)(ô−fb)ᵀ        (d×d cross-covariance)
 *   M_full : Cff·M_fullᵀ = Cyfᵀ            (full ridge map, via chol_solve_spd)
 *   So  = M_full·Cff·M_fullᵀ ; Q = top-`rank` eigvecs(So)  (output directions)
 *   M_r = Qᵀ·Q·M_full                       (rank-`rank` reduced-rank map)
 *   b   = yb − M_r·fb
 *
 * CRITICAL — held-out evaluation. A d_model×d_model map fit on a handful of
 * tokens overfits, after which the IN-SAMPLE error is meaningless. solve_denoise
 * fits ONLY on TRAIN tokens [0,n_train) and reports rel_l2 / cos on the HELD-OUT
 * TEST tokens [n_train,n_x). Only those numbers say whether the map GENERALIZES.
 * The synthetic self-test (-DDENOISE_TEST) proves the methodology on two
 * controls: a learnable rank-r map (held-out error collapses) and an unlearnable
 * independent-noise target (held-out error stays at the noise floor — the
 * held-out metric refuses to reward overfitting).
 *
 * Pure C11, row-major double precision. Links only linalg_small.h
 * (chol_solve_spd + sym_eig_topk). No C++, no gradient descent.
 */
#ifndef SOLVE_DENOISE_H
#define SOLVE_DENOISE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Held-out fidelity of the shared denoiser map. */
typedef struct {
    int    ok;        /* 1 on success; 0 on bad args / non-PD Cff / OOM         */
    double rel_l2;    /* HELD-OUT ‖pred − o_ref‖_F / ‖o_ref‖_F over test (e,i)  */
    double cos_mean;  /* mean over test (e,i) of cos(pred_{e,i}, o_ref_{e,i})   */
    int    rank;      /* effective map rank actually used (clamped to [0,d])    */
    int    n_test;    /* held-out samples scored = n_exp·(n_x − n_train)        */
} dn_result;

/* Fit ONE shared rank-`rank` denoiser  o_ref ≈ M·ô + b  on the TRAIN split and
 * report fidelity on the HELD-OUT TEST split.
 *
 *   o_ref   : [n_exp][n_x][d_model] reference (high-precision) outputs (required)
 *   o_hat   : [n_exp][n_x][d_model] 1-bit outputs ô                    (required)
 *   n_exp,n_x,d_model : dimensions (all ≥ 1)
 *   rank    : target shared-map rank (clamped to [0, d_model])
 *   lambda  : Tikhonov ridge, RELATIVE to the mean diagonal of the centered
 *             feature covariance (scale-safe); grown automatically if Cff is
 *             not positive-definite within working precision
 *   n_train : tokens [0,n_train) of EACH expert are TRAIN (pooled across all
 *             experts); tokens [n_train,n_x) are HELD-OUT TEST. Requires
 *             1 ≤ n_train ≤ n_x − 1.
 *
 * Samples are POOLED over all experts e (one shared map, not per-expert). The
 * fit never reads test tokens; the eval never reads train tokens.
 *
 * Returns dn_result; on bad arguments / numerical failure ok=0 and the metric
 * fields are 0.
 */
dn_result solve_denoise(const double *o_ref, const double *o_hat,
                        int n_exp, int n_x, int d_model,
                        int rank, double lambda, int n_train);

#ifdef __cplusplus
}
#endif

#endif /* SOLVE_DENOISE_H */
