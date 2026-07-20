/* solve_rrr.h — AGGREGATE-level closed-form solver for the per-layer hidden
 * variable z^ℓ, replacing the dense per-expert assembly of hiddenvar_solve.
 *
 * Fits the runtime correction EXACTLY in the semantics of
 * kernel_dsv4_corr_apply (metal/moe.metal):
 *
 *     y'_t = ŷ_t + Σ_{e∈S_t} U·(C_e ⊙ (V·x_t)) + n_valid·b        (no gate
 *     weights inside the sum; n_valid = topk = 6 on the decode path)
 *
 * against per-TOKEN aggregate targets T_t = y*_t − ŷ_t, where y* is the
 * teacher's routed-only MoE output (cap routed_L, gate weights already baked
 * in) and ŷ is the student 1-bit aggregate on the same routing trajectory.
 *
 * FOUR-LOSS mapping (all closed form, no SGD / epochs / alternation):
 *   L_align   → reduced-rank regression: ridge normal equations
 *               Mᵀ=(XᵀX+λI)⁻¹XᵀT, then rank-k truncation via the top-k
 *               eigenvectors of F = Ŷ_fitᵀŶ_fit (RRR optimum), then a k×k
 *               scale+rotation Procrustes polish folded into U.
 *   L_cls     → optional output-metric augmentation Q = I + α²·W_rᵀW_r
 *               (W_r = NEXT layer's router weight): whiten targets with
 *               Q^{1/2} (rank-256 Woodbury correction, 256×256 eig), run RRR
 *               in the whitened space, un-whiten U with Q^{-1/2}. δ itself is
 *               solved elsewhere (needs teacher router logits).
 *   L_smooth  → first-difference sample augmentation: (Δx, ΔT) pairs of
 *               chunk-internal adjacent tokens enter XᵀX AND XᵀT with weight
 *               w_smooth (real RHS, unlike hiddenvar_solve's zero-RHS fold).
 *   L_fix     → Tikhonov ridge relative to mean Gram diagonal + splitmix64
 *               dither (reuses loss_fixed) pinning underdetermined directions.
 *
 * C-stage: per latent coordinate k, the kernel's unweighted per-selected-
 * expert sum makes the 256 gains C[·][k] JOINTLY determined by the firing
 * pattern: solve the 256×256 co-firing system
 *     A[e,e'] = Σ_t 1[e∈S_t]1[e'∈S_t]·v_t[k]²,  rhs[e] = Σ_t 1[e∈S_t]·v_t[k]·g_t[k]
 * with shrinkage toward the NEUTRAL PRIOR C=1/topk (Σ_{e∈S_t}C_e = 1 exactly
 * reproduces the rank-k map U·(V·x); never-fired experts fall back to it, so
 * unseen routing degrades to the RRR map instead of to zero correction).
 *
 * Output-side conventions expected by the runtime (do NOT change here):
 *   z.b stored = b_fit / topk   (kernel adds n_valid·b)
 *   beta = delta = 0            (delta needs teacher router logits: E1/E7)
 *
 * Pure C99; links linalg_small (chol_solve_spd, sym_eig_topk) and reuses the
 * z_layer container + loss_fixed from hiddenvar_solve.h.
 */
#ifndef SOLVE_RRR_H
#define SOLVE_RRR_H

#include <stdint.h>
#include "hiddenvar_solve.h"   /* z_layer, loss_fixed */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int      k;            /* target rank d_ℓ (per-layer adjustable)          */
    double   lambda_rel;   /* ridge for the d×d Gram, relative to mean diag   */
    double   lam_c_rel;    /* ridge for the 256×256 C-stage co-firing solves  */
    double   w_smooth;     /* first-difference augmentation weight (0 = off)  */
    double   w_align;      /* L_align inverse-norm token reweight: row weight
                              a_t=(1−w)+w·(mean‖T‖/‖T_t‖), rows scaled √a_t.
                              0 = energy-weighted LS (big-norm tokens dominate —
                              deep layers have extreme norm dispersion and the
                              per-token direction of small tokens gets sacrificed);
                              1 = fully norm-equalized.                         */
    int      chunk;        /* tokens per capture chunk (diff pairs stay inside)*/
    double   alpha_router; /* Q-metric weight (0 = off)                        */
    const float *Wr;       /* [n_exp×d] NEXT layer router weight, or NULL      */
    int      procrustes;   /* 1 = k×k scale+rotation polish                    */
    uint64_t seed;         /* eig start vectors + ridge dither                 */
    int      eig_iters;    /* power-iteration budget (<=0 → lib default)       */
    int      threads;      /* Gram/matmul worker threads                       */
    int      umode;        /* 0 = RRR-fitted (U,V);  1 = PCA of the FEATURE:
                              U = top-k feature eigvecs, V = Uᵀ — only sensible
                              when the feature lives in output space (φ=ŷ);
                              maximally sample-efficient (no d×d map fit), the
                              C-stage then learns per-direction/expert gains. */
} rrr_params;

void rrr_params_default(rrr_params *p);

typedef struct {
    int    n_train, n_test;
    int    d_l;
    double energy_at_k;      /* Σ top-k evals / trace(F): x-predictable share  */
    /* token-level aggregate fidelity: cos(ŷ+corr, y*) and ‖·‖ rel-L2 */
    double base_cos_tr, base_rel_tr, corr_cos_tr, corr_rel_tr;
    double base_cos_te, base_rel_te, corr_cos_te, corr_rel_te;
} rrr_report;

/* Solve one layer.
 *   X     [n×d]   f32 token inputs x_t (post-RMSNorm ffn_norm)
 *   Yhat  [n×d]   f32 student 1-bit routed aggregate ŷ_t
 *   Yref  [n×d]   f32 teacher routed-only gold y*_t
 *   route [n×topk] f32-decoded expert ids (from cap route_L)
 *   idx_tr/idx_te : token indices (into the n rows) for train / held-out test.
 *   Token order inside idx_tr must be capture order; adjacent tokens with
 *   idx differing by 1 and same chunk feed the L_smooth diff pairs.
 * Returns 0 on success; fills z (arrays malloc'd, free with z_layer_free)
 * and rep. */
int rrr_solve(const float *X, const float *Yhat, const float *Yref,
              const float *route, int topk,
              const int *idx_tr, int n_train, const int *idx_te, int n_test,
              int d, int n_exp,
              const rrr_params *p, z_layer *z, rrr_report *rep);

#ifdef __cplusplus
}
#endif

#endif /* SOLVE_RRR_H */
