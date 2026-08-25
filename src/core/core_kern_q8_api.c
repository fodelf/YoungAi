/* core_kern_q8_api.c — q8_0 matvec/matmul API + matvec_f32/any (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static void matvec_q8_0_rows_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    if (row0 > out_dim || n_rows > out_dim - row0) ds4_die("Q8_0 row range is outside tensor");
    const uint64_t ctx_blocks = (in_dim + 31) / 32;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = row0,
        .blocks = ctx_blocks,
    };
    ds4_parallel_for(n_rows, matvec_q8_0_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale) {
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, 0, w->dim[1]);
}

/* Compute two Q8_0 projections from the same input, used by gate/up and
 * compressor kv/score pairs. */
void matvec_q8_0_pair_prequant(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const int8_t    * xq,
        const float     * xscale) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    matvec_q8_0_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = tensor_data(m, w0),
        .data1 = tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    ds4_parallel_for(w0->dim[1], matvec_q8_0_pair_worker, &ctx);
}

static void matmul_q8_0_batch_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    matmul_q8_0_batch_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w->dim[0],
        .out_dim = w->dim[1],
        .blocks = (w->dim[0] + 31) / 32,
    };
    ds4_parallel_for(ctx.out_dim, matmul_q8_0_batch_worker, &ctx);
}

static void matmul_q8_0_pair_batch_prequant(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    matmul_q8_0_pair_batch_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = tensor_data(m, w0),
        .data1 = tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w0->dim[0],
        .out_dim = w0->dim[1],
        .blocks = (w0->dim[0] + 31) / 32,
    };
    ds4_parallel_for(ctx.out_dim, matmul_q8_0_pair_batch_worker, &ctx);
}

/* Batched Q8_0 matmul for prefill: quantize all token activations, then scan
 * weight rows once per output channel. */
void matmul_q8_0_batch(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_batch_prequant(out, m, w, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}

void matmul_q8_0_pair_batch(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const float     * x,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_pair_batch_prequant(out0, out1, m, w0, w1, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}

static void matvec_q8_0_rows(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t ctx_blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)ctx_blocks * 32);
    float *xscale = xmalloc((size_t)ctx_blocks * sizeof(xscale[0]));

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, row0, n_rows);

    free(xscale);
    free(xq);
}

/* Single-token Q8_0 matvec, used heavily in decode. */
void matvec_q8_0(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    matvec_q8_0_rows(out, m, w, x, 0, w->dim[1]);
}


/* Decode scratch owns this temporary activation quantization so generation
 * can assert that the hot path performs no malloc. */
static void cpu_decode_quantize_q8_0(
        ds4_cpu_decode_scratch * scratch,
        const float            * x,
        uint64_t                 in_dim) {
    if (in_dim > scratch->q8_cap) ds4_die("CPU decode Q8_0 scratch buffer is too small");
    quantize_q8_0_activation(x, scratch->q8_xq, scratch->q8_xscale, in_dim);
}

void matvec_q8_0_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    cpu_decode_quantize_q8_0(scratch, x, w->dim[0]);
    matvec_q8_0_prequant(out, m, w, scratch->q8_xq, scratch->q8_xscale);
}

void matvec_q8_0_pair_decode_scratch(
        float                  * out0,
        float                  * out1,
        const ds4_model        * m,
        const ds4_tensor       * w0,
        const ds4_tensor       * w1,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    cpu_decode_quantize_q8_0(scratch, x, w0->dim[0]);
    matvec_q8_0_pair_prequant(out0, out1, m, w0, w1, scratch->q8_xq, scratch->q8_xscale);
}

void matvec_any_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    if (w->type == 8) {
        matvec_q8_0_decode_scratch(out, m, w, x, scratch);
    } else {
        matvec_any(out, m, w, x);
    }
}

void matvec_q8_0_grouped_rows(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_groups * blocks * 32);
    float *xscale = xmalloc((size_t)n_groups * blocks * sizeof(xscale[0]));

    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 xq + (uint64_t)g * blocks * 32,
                                 xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);

    free(xscale);
    free(xq);
}

void matvec_q8_0_grouped_rows_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        uint32_t                 n_groups,
        uint64_t                 group_dim,
        uint64_t                 rank,
        ds4_cpu_decode_scratch * scratch) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }
    if ((uint64_t)n_groups * group_dim > scratch->q8_cap) {
        ds4_die("CPU decode grouped Q8_0 scratch buffer is too small");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 scratch->q8_xq + (uint64_t)g * blocks * 32,
                                 scratch->q8_xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = scratch->q8_xq,
        .xscale = scratch->q8_xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);
}

void matmul_q8_0_grouped_batch(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          n_tok,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * n_groups * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * n_groups * blocks * sizeof(xscale[0]));

    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            const uint64_t xbase = (t * n_groups + g) * blocks;
            quantize_q8_0_activation(x + t * n_groups * group_dim + (uint64_t)g * group_dim,
                                     xq + xbase * 32,
                                     xscale + xbase,
                                     group_dim);
        }
    }

    matmul_q8_0_grouped_batch_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .n_groups = n_groups,
        .group_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matmul_q8_0_grouped_batch_worker, &ctx);

    free(xscale);
    free(xq);
}

typedef struct {
    float *out;
    const float *data;
    const float *x;
    uint64_t in_dim;
} matvec_f32_ctx;

static void matvec_f32_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f32_ctx *ctx = vctx;

    for (uint64_t o = row0; o < row1; o++) {
        double acc = 0.0;
        const float *row = ctx->data + o * ctx->in_dim;
        for (uint64_t i = 0; i < ctx->in_dim; i++) {
            acc += (double)row[i] * ctx->x[i];
        }
        ctx->out[o] = (float)acc;
    }
}

static void matvec_f32(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 0 || w->ndim != 2) ds4_die("expected a 2D F32 tensor");

    matvec_f32_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = w->dim[0],
    };
    ds4_parallel_for(w->dim[1], matvec_f32_worker, &ctx);
}

/* Dispatch for dense F32/F16/Q8_0 tensors used by auxiliary projections. */
void matvec_any(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    switch (w->type) {
    case 0: matvec_f32(out, m, w, x); break;
    case 1: matvec_f16(out, m, w, x); break;
    case 8: matvec_q8_0(out, m, w, x); break;
    default:
        ds4_die("unsupported tensor type for dense matvec");
    }
}

