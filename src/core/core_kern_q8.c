/* core_kern_q8.c — q8_0 热核 dot/matvec/matmul worker(同TU铁律) (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

static inline int32_t dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (n == 32) {
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(a),      vld1q_s8(b));
        acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
        return vaddvq_s32(acc);
    }
#endif
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);
            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;

            int32x4_t dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0),      vld1q_s8(xq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + 16));

            int32x4_t dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1),      vld1q_s8(xq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + 16));

            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), f16_to_f32(scale_bits0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), f16_to_f32(scale_bits1) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(qs),      vld1q_s8(xqb));
            dot = vdotq_s32(dot, vld1q_s8(qs + 16), vld1q_s8(xqb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), f16_to_f32(scale_bits) * xscale[b]);
        }

        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif

    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot_i8_32(qs, xq + i0, n);
    }
    return acc;
}

static inline void dot_q8_0_row_2(
        const uint8_t *row,
        const int8_t  *xq0,
        const float   *xscale0,
        const int8_t  *xq1,
        const float   *xscale1,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(qs0),      vld1q_s8(xq0 + b * 32));
            d00 = vdotq_s32(d00, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(qs1),      vld1q_s8(xq0 + (b + 1) * 32));
            d01 = vdotq_s32(d01, vld1q_s8(qs1 + 16), vld1q_s8(xq0 + (b + 1) * 32 + 16));

            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(qs0),      vld1q_s8(xq1 + b * 32));
            d10 = vdotq_s32(d10, vld1q_s8(qs0 + 16), vld1q_s8(xq1 + b * 32 + 16));
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(qs1),      vld1q_s8(xq1 + (b + 1) * 32));
            d11 = vdotq_s32(d11, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + (b + 1) * 32 + 16));

            const float s0 = f16_to_f32(scale_bits0);
            const float s1 = f16_to_f32(scale_bits1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), s0 * xscale0[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), s1 * xscale0[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), s0 * xscale1[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), s1 * xscale1[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(qs),      vld1q_s8(xq0 + b * 32));
            d0 = vdotq_s32(d0, vld1q_s8(qs + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(qs),      vld1q_s8(xq1 + b * 32));
            d1 = vdotq_s32(d1, vld1q_s8(qs + 16), vld1q_s8(xq1 + b * 32 + 16));

            const float s0 = f16_to_f32(scale_bits);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), s0 * xscale0[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), s0 * xscale1[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    *out0 = dot_q8_0_row(row, xq0, xscale0, in_dim, blocks);
    *out1 = dot_q8_0_row(row, xq1, xscale1, in_dim, blocks);
}

static inline DS4_MAYBE_UNUSED void dot_q8_0_row_pair(
        const uint8_t *row0,
        const uint8_t *row1,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t s00, s01, s10, s11;
            memcpy(&s00, row0 + b * 34, sizeof(s00));
            memcpy(&s01, row0 + (b + 1) * 34, sizeof(s01));
            memcpy(&s10, row1 + b * 34, sizeof(s10));
            memcpy(&s11, row1 + (b + 1) * 34, sizeof(s11));

            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;
            const int8x16_t xv00 = vld1q_s8(xq0);
            const int8x16_t xv01 = vld1q_s8(xq0 + 16);
            const int8x16_t xv10 = vld1q_s8(xq1);
            const int8x16_t xv11 = vld1q_s8(xq1 + 16);

            const int8_t *q00 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q01 = (const int8_t *)(row0 + (b + 1) * 34 + 2);
            const int8_t *q10 = (const int8_t *)(row1 + b * 34 + 2);
            const int8_t *q11 = (const int8_t *)(row1 + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(q00),      xv00);
            d00 = vdotq_s32(d00, vld1q_s8(q00 + 16), xv01);
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(q01),      xv10);
            d01 = vdotq_s32(d01, vld1q_s8(q01 + 16), xv11);
            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(q10),      xv00);
            d10 = vdotq_s32(d10, vld1q_s8(q10 + 16), xv01);
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(q11),      xv10);
            d11 = vdotq_s32(d11, vld1q_s8(q11 + 16), xv11);

            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), f16_to_f32(s00) * xscale[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), f16_to_f32(s01) * xscale[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), f16_to_f32(s10) * xscale[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), f16_to_f32(s11) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t s0, s1;
            memcpy(&s0, row0 + b * 34, sizeof(s0));
            memcpy(&s1, row1 + b * 34, sizeof(s1));
            const int8_t *xqb = xq + b * 32;
            const int8x16_t xv0 = vld1q_s8(xqb);
            const int8x16_t xv1 = vld1q_s8(xqb + 16);
            const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(q0),      xv0);
            d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), xv1);
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(q1),      xv0);
            d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), xv1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), f16_to_f32(s0) * xscale[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), f16_to_f32(s1) * xscale[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t s0_bits;
        uint16_t s1_bits;
        memcpy(&s0_bits, row0 + b * 34, sizeof(s0_bits));
        memcpy(&s1_bits, row1 + b * 34, sizeof(s1_bits));
        const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
        const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc0 += f16_to_f32(s0_bits) * xscale[b] * (float)dot_i8_32(q0, xq + i0, n);
        acc1 += f16_to_f32(s1_bits) * xscale[b] * (float)dot_i8_32(q1, xq + i0, n);
    }
    *out0 = acc0;
    *out1 = acc1;
}

void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) {
            xq[i0 + i] = 0;
        }
    }
}

static void quantize_q8_0_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    quantize_q8_0_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        quantize_q8_0_activation(ctx->x + t * ctx->in_dim,
                                 ctx->xq + t * ctx->blocks * 32,
                                 ctx->xscale + t * ctx->blocks,
                                 ctx->in_dim);
    }
}

void quantize_q8_0_activation_batch(
        const float *x,
        int8_t      *xq,
        float       *xscale,
        uint64_t     n_tok,
        uint64_t     in_dim) {
    quantize_q8_0_batch_ctx ctx = {
        .x = x,
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    ds4_parallel_for(n_tok, quantize_q8_0_batch_worker, &ctx);
}

void matvec_q8_0_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint64_t o = ctx->row0 + r;
        const uint8_t *row = ctx->data + o * ctx->blocks * 34;
        ctx->out[r] = dot_q8_0_row(row, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
    }
}

void matvec_q8_0_pair_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_pair_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        dot_q8_0_row_pair(row0, row1, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks,
                          ctx->out0 + r, ctx->out1 + r);
    }
}

void matvec_q8_0_grouped_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_grouped_ctx *ctx = vctx;

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;
        const int8_t *xq = ctx->xq + group * ctx->blocks * 32;
        const float *xscale = ctx->xscale + group * ctx->blocks;
        ctx->out[idx] = dot_q8_0_row(row, xq, xscale, ctx->in_dim, ctx->blocks);
    }
}

void matmul_q8_0_grouped_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_grouped_batch_ctx *ctx = vctx;

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;

        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const uint64_t xbase0 = (t * ctx->n_groups + group) * ctx->blocks;
            const uint64_t xbase1 = ((t + 1) * ctx->n_groups + group) * ctx->blocks;
            dot_q8_0_row_2(row,
                           ctx->xq + xbase0 * 32,
                           ctx->xscale + xbase0,
                           ctx->xq + xbase1 * 32,
                           ctx->xscale + xbase1,
                           ctx->group_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group,
                           ctx->out + (t + 1) * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group);
        }
        for (; t < ctx->n_tok; t++) {
            const uint64_t xbase = (t * ctx->n_groups + group) * ctx->blocks;
            ctx->out[t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group] =
                dot_q8_0_row(row,
                             ctx->xq + xbase * 32,
                             ctx->xscale + xbase,
                             ctx->group_dim,
                             ctx->blocks);
        }
    }
}

void matmul_q8_0_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_batch_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row = ctx->data + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            dot_q8_0_row_2(row,
                           ctx->xq + t * ctx->blocks * 32,
                           ctx->xscale + t * ctx->blocks,
                           ctx->xq + (t + 1) * ctx->blocks * 32,
                           ctx->xscale + (t + 1) * ctx->blocks,
                           ctx->in_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->out_dim + r,
                           ctx->out + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            ctx->out[t * ctx->out_dim + r] =
                dot_q8_0_row(row,
                             ctx->xq + t * ctx->blocks * 32,
                             ctx->xscale + t * ctx->blocks,
                             ctx->in_dim,
                             ctx->blocks);
        }
    }
}

void matmul_q8_0_pair_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_pair_batch_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const int8_t *xq0 = ctx->xq + t * ctx->blocks * 32;
            const float *xscale0 = ctx->xscale + t * ctx->blocks;
            const int8_t *xq1 = ctx->xq + (t + 1) * ctx->blocks * 32;
            const float *xscale1 = ctx->xscale + (t + 1) * ctx->blocks;
            dot_q8_0_row_2(row0, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out0 + t * ctx->out_dim + r,
                           ctx->out0 + (t + 1) * ctx->out_dim + r);
            dot_q8_0_row_2(row1, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out1 + t * ctx->out_dim + r,
                           ctx->out1 + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            const int8_t *xq = ctx->xq + t * ctx->blocks * 32;
            const float *xscale = ctx->xscale + t * ctx->blocks;
            dot_q8_0_row_pair(row0, row1, xq, xscale, ctx->in_dim, ctx->blocks,
                              ctx->out0 + t * ctx->out_dim + r,
                              ctx->out1 + t * ctx->out_dim + r);
        }
    }
}

/* Multiply selected Q8_0 rows by an activation that has already been quantized
 * once.  This avoids repeated activation quantization for paired projections. */
