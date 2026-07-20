/* ds4_loss.c -- 四损失 (ALGORITHM.md §4). Self-contained; see ds4_loss.h. */
#include "ds4_loss.h"

#include <math.h>
#include <stddef.h>

float ds4_loss_align(const float *yhat, const float *y, uint32_t n, uint32_t d) {
    if (!yhat || !y || n == 0 || d == 0) return 0.0f;
    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const float *a = yhat + (size_t)i * d;
        const float *b = y + (size_t)i * d;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            dot += (double)a[j] * b[j];
            na += (double)a[j] * a[j];
            nb += (double)b[j] * b[j];
        }
        const double den = sqrt(na) * sqrt(nb);
        acc += den > 1e-30 ? dot / den : 0.0;   /* dead row = cos 0, not NaN */
    }
    return (float)(1.0 - acc / (double)n);
}

float ds4_loss_classify(const float *yhat, const float *y,
                        const float *w, uint32_t n, uint32_t d) {
    if (!yhat || !y || n == 0 || d == 0) return 0.0f;
    double wsum = 0.0;
    if (w) {
        for (uint32_t j = 0; j < d; j++) wsum += w[j];
        if (wsum <= 1e-30) w = NULL;            /* degenerate weights = plain MSE */
    }
    if (!w) wsum = (double)d;
    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const float *a = yhat + (size_t)i * d;
        const float *b = y + (size_t)i * d;
        double row = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            const double e = (double)a[j] - b[j];
            row += (w ? (double)w[j] : 1.0) * e * e;
        }
        acc += row / wsum;
    }
    return (float)(acc / (double)n);
}

void ds4_loss_dim_variance(const float *y, uint32_t n, uint32_t d, float *w) {
    if (!y || !w || n == 0 || d == 0) return;
    for (uint32_t j = 0; j < d; j++) w[j] = 0.0f;
    /* two-pass for numerical sanity at calibration sizes */
    for (uint32_t j = 0; j < d; j++) {
        double mean = 0.0;
        for (uint32_t i = 0; i < n; i++) mean += y[(size_t)i * d + j];
        mean /= (double)n;
        double var = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            const double e = (double)y[(size_t)i * d + j] - mean;
            var += e * e;
        }
        w[j] = (float)(var / (double)n);
    }
}

float ds4_loss_smooth(const float *y_base, const float *y_pert,
                      uint32_t n, uint32_t d) {
    if (!y_base || !y_pert || n == 0 || d == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < (size_t)n * d; i++) {
        const double e = (double)y_pert[i] - y_base[i];
        acc += e * e;
    }
    return (float)(acc / ((double)n * (double)d));
}

void ds4_loss_dither(uint64_t seed, uint32_t row, float scale,
                     float *delta, uint32_t d) {
    if (!delta || d == 0) return;
    /* splitmix-style stream keyed by (seed, row): every caller that agrees on
     * (seed, row) perturbs identically -- the "固定种子 dither" contract. */
    uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(row + 1u));
    for (uint32_t j = 0; j < d; j++) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t x = s;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x ^= x >> 31;
        delta[j] = (((float)(uint32_t)x / 4294967296.0f) * 2.0f - 1.0f) * scale;
    }
}

float ds4_loss_fixed(const float *params, size_t n_params) {
    if (!params || n_params == 0) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < n_params; i++) acc += (double)params[i] * params[i];
    return (float)(acc / (double)n_params);
}

float ds4_loss_total(const ds4_loss_weights *w, float l_align,
                     float l_classify, float l_smooth, float l_fixed) {
    if (!w) return l_align + l_classify + l_smooth + l_fixed;
    return w->w_align * l_align + w->w_classify * l_classify +
           w->w_smooth * l_smooth + w->w_fixed * l_fixed;
}
