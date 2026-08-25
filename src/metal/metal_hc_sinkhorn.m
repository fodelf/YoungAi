/* metal_hc_sinkhorn.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
} ds4_gpu_hc_split_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_hc_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float eps;
} ds4_gpu_hc_split_weighted_sum_args;

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_hc == 0 || n_hc > 16) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t mix_tensor_bytes = ds4_gpu_tensor_bytes(mix);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (!mixbuf || !outbuf ||
            mix_tensor_bytes < mix_bytes ||
            out_tensor_bytes < mix_bytes) {
            fprintf(stderr, "ds4: Metal HC split received undersized activation buffers\n");
            return 0;
        }
        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset) {
            fprintf(stderr, "ds4: Metal HC split parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        if (!scalebuf || !basebuf) return 0;

        uint64_t n_rows64 = mix_tensor_bytes / mix_bytes;
        const uint64_t out_rows64 = out_tensor_bytes / mix_bytes;
        if (out_rows64 < n_rows64) n_rows64 = out_rows64;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC split row count is outside supported range\n");
            return 0;
        }

        ds4_gpu_hc_split_args args = {
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb01 = mix_bytes,
            .nb1 = mix_bytes,
            .eps = eps,
        };
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_rows64));
        const NSUInteger n_tg = ((NSUInteger)n_rows64 + nth - 1u) / nth;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_split_sinkhorn_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sinkhorn")) return 0;
    }

    return 1;
}

static int ds4_gpu_hc_weighted_sum_strided(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint64_t                weight_offset,
        uint64_t                weight_row_stride,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0 ||
        weight_row_stride < (uint64_t)n_hc * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes || out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC weighted sum output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / out_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC weighted sum token count is outside supported range\n");
            return 0;
        }

        const uint64_t x_row_values = (uint64_t)n_hc * n_embd;
        if (x_row_values == 0 ||
            x_row_values > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (x_row_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / ((uint64_t)n_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC weighted sum activation size overflow\n");
            return 0;
        }

        const uint64_t x_bytes = n_tokens64 * x_row_values * sizeof(float);
        const uint64_t w_last = weight_offset +
                                (n_tokens64 - 1u) * weight_row_stride +
                                (uint64_t)n_hc * sizeof(float);
        if (!xbuf || !wbuf || !outbuf ||
            ds4_gpu_tensor_bytes(residual_hc) < x_bytes ||
            ds4_gpu_tensor_bytes(weights) < w_last) {
            fprintf(stderr, "ds4: Metal HC weighted sum received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_weighted_sum_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_w0 = sizeof(float),
            .nb_w1 = weight_row_stride,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
        };
        const uint64_t n_elem = (uint64_t)n_embd * n_tokens64;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_weighted_sum_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) + (NSUInteger)weight_offset atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, label)) return 0;
    }

    return 1;
}

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return ds4_gpu_hc_weighted_sum_strided(out,
                                             residual_hc,
                                             weights,
                                             0,
                                             (uint64_t)n_hc * sizeof(float),
                                             n_embd,
                                             n_hc,
                                             "HC weighted sum");
}

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return ds4_gpu_hc_weighted_sum_strided(out,
                                             residual_hc,
                                             split,
                                             0,
                                             mix_hc * sizeof(float),
                                             n_embd,
                                             n_hc,
                                             "HC weighted sum split");
}

/* Release decode fused HC pre-sublayer operation.  The graph driver owns the
 * optional reference fallback so this function stays a direct fused dispatch. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc == 0) {
        return 0;
    }
    if (n_hc != 4) {
        fprintf(stderr, "ds4: Metal fused HC split/sum is specialized for HC=4\n");
        return 0;
    }

    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
            out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal fused HC split/sum output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_rows64 = out_tensor_bytes / out_row_bytes;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX ||
            n_rows64 > UINT64_MAX / mix_bytes ||
            n_rows64 > UINT64_MAX / residual_row_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum row count is outside supported range\n");
            return 0;
        }

        const uint64_t mix_total_bytes = n_rows64 * mix_bytes;
        const uint64_t residual_total_bytes = n_rows64 * residual_row_bytes;
        if (!mixbuf || !splitbuf || !xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(mix) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(split) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < residual_total_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum received undersized activation buffers\n");
            return 0;
        }

        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset) {
            fprintf(stderr, "ds4: Metal fused HC split/sum parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        if (!scalebuf || !basebuf) return 0;

        ds4_gpu_hc_split_weighted_sum_args args = {
            .n_embd = (int64_t)n_embd,
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb_mix1 = mix_bytes,
            .nb_split1 = mix_bytes,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = residual_row_bytes,
            .nb0 = sizeof(float),
            .nb1 = out_row_bytes,
            .eps = eps,
        };

        NSUInteger nth = g_hc_split_weighted_sum_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n_embd) nth = (NSUInteger)n_embd;
        if (nth == 0) nth = 1u;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_split_weighted_sum_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:4];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_hc * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows64, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sum fused")) return 0;
    }

    return 1;
}
