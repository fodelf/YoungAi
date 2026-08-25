/* core_kern_f16.c — deq_q2K/embed/rms_norm/matvec_f16(热核同TU) (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static void deq_q2K_row_f32(const uint8_t *row, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = row + b * 84u;
        const uint8_t *sc = blk, *qs = blk + 16;
        uint16_t hd, hm;
        memcpy(&hd, blk + 80, 2);
        memcpy(&hm, blk + 82, 2);
        const float d = f16_to_f32(hd), dm = f16_to_f32(hm);
        float *o = out + b * 256u;
        for (int j = 0; j < 16; j++) {
            const float dj = d * (float)(sc[j] & 0xF), mj = dm * (float)(sc[j] >> 4);
            for (int ii = 0; ii < 16; ii++) {
                const int idx = j * 16 + ii;
                const int qpos = (idx / 128) * 32 + (idx % 32);
                const int q = (qs[qpos] >> ((idx % 128) / 32 * 2)) & 3;
                o[idx] = dj * (float)q - mj;
            }
        }
    }
}

void embed_token_f16(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    ds4_tensor *te = w->token_embd;
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }

    const uint64_t stride = te->dim[0];
    if (te->bytes * 256u == te->elements * 84u) {   /* 文件真身是 q2_K(全q2 影子) */
        const uint8_t *qbase = (const uint8_t *)tensor_data(m, te);
        const uint64_t nblk = stride / 256u;
        deq_q2K_row_f32(qbase + (uint64_t)token * nblk * 84u, nblk, out);
        return;
    }

    const uint16_t *base = tensor_data(m, te);
    const uint16_t *row = base + (uint64_t)token * stride;

    for (uint64_t i = 0; i < stride; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}

/* RMSNorm without a learned scale, used by hyper-connection control vectors. */
void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
}

/* Standard DS4 RMSNorm with learned per-channel scale. */
void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

/* Normalize each attention head independently after Q projection. */
void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];

        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static inline float dot_f16_row(const uint16_t *row, const float *x, uint64_t n) {
#if defined(__ARM_NEON)
    uint64_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t hv = vreinterpretq_f16_u16(vld1q_u16(row + i));
        const float32x4_t h0 = vcvt_f32_f16(vget_low_f16(hv));
        const float32x4_t h1 = vcvt_f32_f16(vget_high_f16(hv));
        acc0 = vfmaq_f32(acc0, h0, vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, h1, vld1q_f32(x + i + 4));
    }

    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#endif
}

static void matvec_f16_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f16_ctx *ctx = vctx;

    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *row = ctx->data + o * ctx->in_dim;
        ctx->out[o] = dot_f16_row(row, ctx->x, ctx->in_dim);
    }
}

/* Dense F16 matvec for small control projections such as HC and router heads. */
void matvec_f16(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) ds4_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    matvec_f16_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = in_dim,
    };

    const uint64_t ops = in_dim * out_dim;
    const uint64_t min_rows = ops >= 262144 ? 1 : 512;
    ds4_parallel_for_min_rows(out_dim, matvec_f16_worker, &ctx, min_rows);
}

void matvec_f16_serial(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) ds4_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint16_t *data = tensor_data(m, w);
    for (uint64_t o = 0; o < out_dim; o++) {
        out[o] = dot_f16_row(data + o * in_dim, x, in_dim);
    }
}

