/* solve_perexpert.h — PER-EXPERT INDEPENDENT low-rank corrector (alternative
 * solver for the Go-domain 1-bit quantizer).
 *
 * Where hiddenvar_solve.h fits a per-layer SHARED low-rank basis (U,V) with
 * per-expert diagonal gains, this solver gives EACH routed expert its OWN
 * independent rank-`rank` map M_e plus bias b_e. The question it answers is
 * capacity: does an independent per-expert map materially out-reconstruct the
 * shared-basis form on the same residual pool?
 *
 * For each expert e the residual is fit, in CLOSED FORM (no gradient descent,
 * no epochs), as
 *
 *     Δo_e(x) ≈ M_e · x + b_e ,   rank(M_e) ≤ rank
 *
 * via three staged steps:
 *
 *   1. Ridge map (full rank, reduced-rank-regression OLS stage):
 *        M_e^full = (Σ_i Δo_{e,i} x_iᵀ) · (Σ_i x_i x_iᵀ + λI)⁻¹
 *      The Gram G = Σ_i x_i x_iᵀ + λI is SHARED across experts (x is a shared
 *      activation pool), so it is factored ONCE — actually inverted once via
 *      chol_solve_spd with an identity RHS — and the inverse is reused for
 *      every expert by a matmul. Experts are processed one at a time so peak
 *      memory is O(d_model² + n_x·d_model), independent of n_exp.
 *
 *   2. Rank-`rank` truncation (reduced-rank-regression projection stage):
 *      eigendecompose the fitted-output covariance S_e = M_e^full·Gxx·M_e^fullᵀ
 *      with sym_eig_topk, take its top-`rank` output directions Q, and project
 *        M_e^trunc = Q·Qᵀ·M_e^full   (rank(M_e^trunc) ≤ rank).
 *      This is the optimal rank-`rank` reduced-rank regression of the residual
 *      under the shared input covariance.
 *
 *   3. Bias:  b_e = mean_i( Δo_{e,i} − M_e^trunc · x_i ).
 *
 * Pure C99/C11, row-major double precision. Links only against
 * linalg_small.h (chol_solve_spd + sym_eig_topk).
 */
#ifndef SOLVE_PEREXPERT_H
#define SOLVE_PEREXPERT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Global reconstruction fidelity of the per-expert corrector. */
typedef struct {
    int    ok;        /* 1 on success; 0 on bad args / non-PD Gram / OOM      */
    double rel_l2;    /* global ‖pred − ref‖_F / ‖ref‖_F over all (e,i)       */
    double cos_mean;  /* mean over (e,i) of cos(pred_{e,i}, ref_{e,i})        */
} pe_result;

/* Fit one independent rank-`rank` corrector per expert and report global
 * fidelity.
 *
 *   delta_o : [n_exp][n_x][d_model] residuals Δo = o_ref − ô   (required)
 *   x       : [n_x][d_model] SHARED activation pool            (required)
 *   o_ref   : [n_exp][n_x][d_model] reference outputs, or NULL
 *   o_hat   : [n_exp][n_x][d_model] 1-bit outputs ô,        or NULL
 *   n_exp,n_x,d_model : dimensions (all ≥ 1)
 *   rank    : target per-expert map rank (clamped to [0, d_model])
 *   lambda  : absolute Tikhonov ridge added to the diagonal of the Gram
 *
 * Fidelity convention (matches hv_fidelity):
 *   - o_hat != NULL : pred = ô + corr,  ref = o_ref (or ô+Δo if o_ref==NULL).
 *   - o_hat == NULL : pred = corr,      ref = Δo  (o_ref ignored; may be NULL).
 *
 * Returns pe_result; on bad arguments / numerical failure ok=0 and the metric
 * fields are 0.
 */
pe_result solve_perexpert(const double *delta_o, const double *x,
                          const double *o_ref, const double *o_hat,
                          int n_exp, int n_x, int d_model,
                          int rank, double lambda);

#ifdef __cplusplus
}
#endif

#endif /* SOLVE_PEREXPERT_H */
