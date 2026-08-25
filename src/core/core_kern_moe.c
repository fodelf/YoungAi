/* core_kern_moe.c — iq2xxs/q2_K 专家矩阵向量核 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
float tensor_1d_value(const ds4_model *m, const ds4_tensor *t, uint64_t i) {
    if (i >= t->elements) ds4_die("tensor scalar index is out of bounds");
    if (t->type == 0) {
        const float *p = tensor_data(m, t);
        return p[i];
    }
    if (t->type == 1) {
        const uint16_t *p = tensor_data(m, t);
        return f16_to_f32(p[i]);
    }
    ds4_die("unsupported tensor scalar type");
    return 0.0f;
}

float tensor_2d_value(const ds4_model *m, const ds4_tensor *t, uint64_t x, uint64_t y) {
    if (t->ndim != 2 || x >= t->dim[0] || y >= t->dim[1]) {
        ds4_die("tensor 2D index is out of bounds");
    }
    return tensor_1d_value(m, t, y * t->dim[0] + x);
}

/* Locate one expert's 2D matrix inside a 3D GGUF expert tensor. */
const uint8_t *tensor_expert_bytes(
        const ds4_model  *m,
        const ds4_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes) {
    if (w->ndim != 3) ds4_die("expected a 3D expert tensor");
    if (expert >= w->dim[2]) ds4_die("expert id is outside expert tensor");

    *in_dim = w->dim[0];
    *out_dim = w->dim[1];

    const gguf_type_info *info = tensor_type(w->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported expert tensor type");
    const uint64_t blocks = (*in_dim + info->block_elems - 1) / info->block_elems;
    *row_bytes = blocks * info->block_bytes;

    const uint64_t expert_bytes = *out_dim * *row_bytes;
    return (const uint8_t *)tensor_data(m, w) + (uint64_t)expert * expert_bytes;
}

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *base0;
    const uint8_t *base1;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes0;
    uint64_t row_bytes1;
} matvec_iq2_xxs_pair_ctx;

static void matvec_iq2_xxs_pair_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_pair_ctx *ctx = vctx;
    for (uint64_t row = row0; row < row1; row++) {
        const block_iq2_xxs *br0 = (const block_iq2_xxs *)(ctx->base0 + row * ctx->row_bytes0);
        const block_iq2_xxs *br1 = (const block_iq2_xxs *)(ctx->base1 + row * ctx->row_bytes1);
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &ctx->out0[row], &ctx->out1[row], br0, br1, ctx->xq);
    }
}

/* Project one routed expert's gate and up matrices.  Both are IQ2_XXS and
 * share the same Q8_K activation. */
void matvec_iq2_xxs_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const ds4_model  *m,
        const ds4_tensor *w0,
        const ds4_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type != 16 || w1->type != 16) ds4_die("expected IQ2_XXS expert tensors");

    uint64_t in_dim0, out_dim0, row_bytes0;
    uint64_t in_dim1, out_dim1, row_bytes1;
    const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &row_bytes0);
    const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &row_bytes1);
    if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired IQ2_XXS expert tensors do not match");
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    matvec_iq2_xxs_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .base0 = base0,
        .base1 = base1,
        .xq = xq,
        .in_dim = in_dim0,
        .row_bytes0 = row_bytes0,
        .row_bytes1 = row_bytes1,
    };
    ds4_parallel_for(out_dim0, matvec_iq2_xxs_pair_worker, &ctx);
}


typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_mid_ctx;

static void matvec_iq2_xxs_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

/* Build all selected expert hidden vectors: IQ2_XXS gate/up, clamp, SwiGLU,
 * and router weight.  The down projection runs later on the quantized mids. */
void matvec_iq2_xxs_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != 16 || up_w->type != 16) ds4_die("expected IQ2_XXS expert tensors");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_iq2_xxs_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired IQ2_XXS expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("IQ2_XXS expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_iq2_xxs_mid_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
} matvec_q2_k_ctx;

static void matvec_q2_k_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_ctx *ctx = vctx;
    for (uint64_t row = row0; row < row1; row++) {
        const block_q2_K *br = (const block_q2_K *)(ctx->base + row * ctx->row_bytes);
        ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &ctx->out[row], br, ctx->xq);
    }
}

/* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */
void matvec_q2_k_expert(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");

    uint64_t in_dim, out_dim, row_bytes;
    const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
    if (in_dim % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

    matvec_q2_k_ctx ctx = {
        .out = out,
        .base = base,
        .xq = xq,
        .in_dim = in_dim,
        .row_bytes = row_bytes,
    };
    ds4_parallel_for(out_dim, matvec_q2_k_worker, &ctx);

    free(xq);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q2_k_accum_ctx;

static void matvec_q2_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q2_K *br = (const block_q2_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

/* Accumulate all selected experts' Q2_K down projections directly into the
 * 4096-wide MoE output. */
void matvec_q2_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q2_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q2_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q2_k_accum_worker, &ctx);
}


void matvec_iq2_xxs_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_iq2_xxs_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}


void quantize_mid_pairs_worker(void *vctx, uint64_t p0, uint64_t p1) {
    quantize_mid_pairs_ctx *ctx = vctx;
    for (uint64_t p = p0; p < p1; p++) {
        ds4_quantize_row_q8_K(ctx->mid + p * ctx->down_in_dim,
                              ctx->midq + p * ctx->down_blocks,
                              (int64_t)ctx->down_in_dim);
    }
}

typedef struct {
    float *down_pair;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_down_ctx;

static DS4_MAYBE_UNUSED void matvec_q2_k_batch_down_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q2_k_batch_down_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];
        const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim,
                                  ctx->down_pair + (uint64_t)pair_id * ctx->out_dim + row,
                                  br, xq);
        }
    }
}


void matvec_q2_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

typedef struct {
    float *moe;
    const float *down_pair;
    uint32_t n_tok;
    uint64_t out_dim;
} sum_down_pairs_ctx;

static DS4_MAYBE_UNUSED void sum_down_pairs_worker(void *vctx, uint64_t row0, uint64_t row1) {
    sum_down_pairs_ctx *ctx = vctx;
    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t token = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)token * ctx->out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint64_t pair_id = (uint64_t)token * DS4_N_EXPERT_USED + slot;
            acc += ctx->down_pair[pair_id * ctx->out_dim + row];
        }
        ctx->moe[idx] = acc;
    }
}
