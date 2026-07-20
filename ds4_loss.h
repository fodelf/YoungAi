/* =========================================================================
 * ds4_loss -- 四损失: the four calibration losses of the go-onebit scheme
 * (gguf-tools/go-onebit/ALGORITHM.md §4), as a self-contained module.
 * =========================================================================
 *
 * Pure functions over float arrays -- no engine, no autograd, no allocation
 * beyond the caller's buffers. They score a predictor's outputs against
 * reference outputs during closed-form post-training / calibration sweeps:
 *
 *   1. L_align    = 1 - mean cos(yhat, y)      -- THE metric, optimized直接.
 *   2. L_classify = per-dim variance-weighted MSE -- keeps the discriminative
 *                   dimensions that drive downstream routing.
 *   3. L_smooth   = mean ||M(x+delta) - M(x)||^2 -- input robustness; the
 *                   fixed-seed dither generator lives here so every caller
 *                   perturbs identically (reproducible sweeps).
 *   4. L_fixed    = weight-decay energy (mean w^2) -- pins the solution,
 *                   fights the mid-layer overfit enemy.
 *
 * All batch inputs are row-major (n rows x d dims). NaN-free inputs are the
 * caller's contract (quantized forwards can produce them; filter first). */
#ifndef DS4_LOSS_H
#define DS4_LOSS_H

#include <stddef.h>
#include <stdint.h>

/* 1. L_align = 1 - mean_i cos(yhat_i, y_i). Zero-norm rows count as cos 0
 * (maximally misaligned) rather than poisoning the mean with NaN. */
float ds4_loss_align(const float *yhat, const float *y, uint32_t n, uint32_t d);

/* 2. L_classify = mean_i sum_j w[j] * (yhat_ij - y_ij)^2 / sum_j w[j].
 * w = per-dim importance; pass NULL for plain MSE. */
float ds4_loss_classify(const float *yhat, const float *y,
                        const float *w, uint32_t n, uint32_t d);

/* Per-dim variance of Y (n x d) -> w[d]: the standard importance weights for
 * L_classify ("按 y 的 per-dim 方差加权"). */
void ds4_loss_dim_variance(const float *y, uint32_t n, uint32_t d, float *w);

/* 3. L_smooth = mean_i ||y_pert_i - y_base_i||^2 / d. The caller evaluates
 * its predictor twice (at x and at x + delta); this scores the difference. */
float ds4_loss_smooth(const float *y_base, const float *y_pert,
                      uint32_t n, uint32_t d);

/* Fixed-seed dither for L_smooth / L_fixed augmentation: writes d uniform
 * values in [-scale, scale] derived ONLY from (seed, row) -- byte-identical
 * across runs and machines, per the "固定种子" requirement. */
void ds4_loss_dither(uint64_t seed, uint32_t row, float scale,
                     float *delta, uint32_t d);

/* 4. L_fixed = mean_j w_j^2 over n_params parameters (weight-decay energy). */
float ds4_loss_fixed(const float *params, size_t n_params);

/* Weighted total: Loss = w_a*L_align + w_c*L_classify + w_s*L_smooth +
 * w_f*L_fixed (ALGORITHM.md §4 weighting). */
typedef struct {
    float w_align;
    float w_classify;
    float w_smooth;
    float w_fixed;
} ds4_loss_weights;

float ds4_loss_total(const ds4_loss_weights *w, float l_align,
                     float l_classify, float l_smooth, float l_fixed);

#endif /* DS4_LOSS_H */
