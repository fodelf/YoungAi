/* =========================================================================
 * ds4_posttrain -- 后训练优化: closed-form, zero-training post-training
 * quantization optimization (gguf-tools/go-onebit/ALGORITHM.md §6.1), as a
 * self-contained module.
 * =========================================================================
 *
 * The validated lever: OUTPUT-optimal activation-aware per-row 1-bit scale.
 * Generic 1-bit uses s_i = mean|w_i| (weight-space optimal); this module
 * solves the scale that is optimal in OUTPUT space over real calibration
 * activations X (measured on go1b per-expert base: cosine 0.55 -> ~0.70):
 *
 *   b_i  = sign(w_i)
 *   s_i  = sum_x (w_i . x)(b_i . x) / sum_x (b_i . x)^2
 *
 * i.e. the least-squares regression of the true row output onto the 1-bit
 * row output. Everything is closed-form -- no gradients, no training, per
 * the 恢复原始能力/零训练 iron law. The same shape extends to any calibration
 * domain by swapping X (Go today, gin/React corpora tomorrow).
 *
 * Row-major everywhere; W is (rows x d), X is (n x d). */
#ifndef DS4_POSTTRAIN_H
#define DS4_POSTTRAIN_H

#include <stdint.h>

/* Activation-aware scale for ONE row. Returns s_i (0 when the 1-bit output
 * energy is degenerate -- caller should fall back to mean|w|). */
float ds4_posttrain_row_scale(const float *w_row, uint32_t d,
                              const float *X, uint32_t n);

/* Weight-space fallback scale s_i = mean|w_i| (the generic 1-bit baseline;
 * exposed so callers can A/B and so degenerate rows have a defined value). */
float ds4_posttrain_row_scale_meanabs(const float *w_row, uint32_t d);

/* Quantize one row to sign bits + activation-aware scale. bits[] receives d
 * entries of +1/-1 stored as int8 (packing is the file format's job, not the
 * math module's). Returns the scale used (activation-aware when X != NULL
 * and non-degenerate, else mean|w|). */
float ds4_posttrain_quant_row(const float *w_row, uint32_t d,
                              const float *X, uint32_t n, int8_t *bits);

/* Output-space cosine of the quantized row vs the original row over X:
 * cos( X w , s * X b ). The per-row accept metric for any post-train sweep
 * (scale choice cancels in cosine; it reports the DIRECTION recovery). */
float ds4_posttrain_row_cosine(const float *w_row, uint32_t d,
                               const int8_t *bits,
                               const float *X, uint32_t n);

#endif /* DS4_POSTTRAIN_H */
