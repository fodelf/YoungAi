/* metal_moe_reap.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_kargs_sum_rows;

typedef struct {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
} ds4_gpu_dsv4_moe_sum6_args;

static int ds4_gpu_encode_moe_sum6(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum6_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 6u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum6_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum6_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

int ds4_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    if (n_expert == 6 &&
        ds4_gpu_encode_moe_sum6(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    ds4_gpu_bin_args first =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!ds4_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    ds4_gpu_bin_args accum =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!ds4_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_encode_get_rows_i32_token_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        table,
        NSUInteger           table_off,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_off,
        const int32_t       *token_inline,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        uint32_t             hash_rows,
        uint32_t             n_cols,
        uint32_t             n_tokens) {
    if (!cb || !table || !selected || hash_rows == 0 || n_cols == 0 || n_tokens == 0) return 0;
    if (!tokens && !token_inline) return 0;

    const uint64_t table_row_bytes = (uint64_t)n_cols * sizeof(int32_t);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int64_t)n_cols,
        .ne00 = (int64_t)n_cols,
        .nb01 = table_row_bytes,
        .nb02 = (uint64_t)hash_rows * table_row_bytes,
        .nb03 = (uint64_t)hash_rows * table_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = table_row_bytes,
        .nb2 = (uint64_t)n_tokens * table_row_bytes,
        .nb3 = (uint64_t)n_tokens * table_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_cols;
    const NSUInteger max_threads = g_get_rows_i32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;
    const NSUInteger nw0 = ((NSUInteger)n_cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:table offset:table_off atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_off atIndex:2];
    } else {
        [enc setBytes:token_inline length:sizeof(*token_inline) atIndex:2];
    }
    [enc setBuffer:selected offset:selected_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_get_rows_f32_router_weights(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        probs,
        NSUInteger           probs_off,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             n_expert,
        uint32_t             n_expert_used,
        uint32_t             n_tokens) {
    if (!cb || !probs || !selected || !weights || n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;

    const uint64_t probs_token_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_row_bytes = (uint64_t)n_expert_used * sizeof(int32_t);
    const uint64_t weights_row_bytes = (uint64_t)n_expert_used * sizeof(float);
    ds4_gpu_get_rows_args args = {
        .ne00t = 1,
        .ne00 = 1,
        .nb01 = sizeof(float),
        .nb02 = probs_token_bytes,
        .nb03 = (uint64_t)n_tokens * probs_token_bytes,
        .ne10 = (int64_t)n_expert_used,
        .nb10 = sizeof(int32_t),
        .nb11 = selected_row_bytes,
        .nb12 = (uint64_t)n_tokens * selected_row_bytes,
        .nb1 = sizeof(float),
        .nb2 = weights_row_bytes,
        .nb3 = (uint64_t)n_tokens * weights_row_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:probs offset:probs_off atIndex:1];
    [enc setBuffer:selected offset:selected_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_expert_used, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || width == 0 || rows == 0) return 0;

    const uint64_t src_row_bytes = (uint64_t)width * sizeof(float);
    ds4_gpu_kargs_sum_rows args = {
        .ne00 = (int64_t)width,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)rows * src_row_bytes,
        .nb03 = (uint64_t)rows * src_row_bytes,
        .ne0 = 1,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = sizeof(float),
        .nb2 = (uint64_t)rows * sizeof(float),
        .nb3 = (uint64_t)rows * sizeof(float),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_sum_rows_f32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)args.ne00 && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)args.ne00) nth = (NSUInteger)args.ne00;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_sum_rows_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
