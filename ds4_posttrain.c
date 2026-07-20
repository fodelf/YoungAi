/* ds4_posttrain.c -- 后训练优化: closed-form activation-aware 1-bit scales
 * (ALGORITHM.md §6.1). Self-contained; see ds4_posttrain.h. */
#include "ds4_posttrain.h"

#include <math.h>
#include <stddef.h>

float ds4_posttrain_row_scale_meanabs(const float *w_row, uint32_t d) {
    if (!w_row || d == 0) return 0.0f;
    double acc = 0.0;
    for (uint32_t j = 0; j < d; j++) acc += fabs((double)w_row[j]);
    return (float)(acc / (double)d);
}

float ds4_posttrain_row_scale(const float *w_row, uint32_t d,
                              const float *X, uint32_t n) {
    if (!w_row || !X || d == 0 || n == 0) return 0.0f;
    /* s = sum_x (w.x)(b.x) / sum_x (b.x)^2, b = sign(w). sign(0) := +1 so
     * the bit pattern is deterministic and matches ds4_posttrain_quant_row. */
    double num = 0.0, den = 0.0;
    for (uint32_t t = 0; t < n; t++) {
        const float *x = X + (size_t)t * d;
        double wx = 0.0, bx = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            const double xj = x[j];
            wx += (double)w_row[j] * xj;
            bx += (w_row[j] < 0.0f ? -xj : xj);
        }
        num += wx * bx;
        den += bx * bx;
    }
    if (den <= 1e-30) return 0.0f;              /* degenerate: caller falls back */
    return (float)(num / den);
}

float ds4_posttrain_quant_row(const float *w_row, uint32_t d,
                              const float *X, uint32_t n, int8_t *bits) {
    if (!w_row || !bits || d == 0) return 0.0f;
    for (uint32_t j = 0; j < d; j++) bits[j] = w_row[j] < 0.0f ? -1 : 1;
    float s = X && n ? ds4_posttrain_row_scale(w_row, d, X, n) : 0.0f;
    if (s == 0.0f) s = ds4_posttrain_row_scale_meanabs(w_row, d);
    return s;
}

float ds4_posttrain_row_cosine(const float *w_row, uint32_t d,
                               const int8_t *bits,
                               const float *X, uint32_t n) {
    if (!w_row || !bits || !X || d == 0 || n == 0) return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t t = 0; t < n; t++) {
        const float *x = X + (size_t)t * d;
        double wx = 0.0, bx = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            wx += (double)w_row[j] * x[j];
            bx += bits[j] < 0 ? -(double)x[j] : (double)x[j];
        }
        dot += wx * bx;
        na += wx * wx;
        nb += bx * bx;
    }
    const double den = sqrt(na) * sqrt(nb);
    return den > 1e-30 ? (float)(dot / den) : 0.0f;
}
