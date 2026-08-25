/* metal_hc_norm.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

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
    uint64_t nb_norm1;
    float eps;
    float norm_eps;
} ds4_gpu_hc_split_weighted_sum_norm_args;

/* Decode-only HC-pre plus the immediately following weighted RMSNorm.  This is
 * intentionally specialized for DS4's fixed HC=4, embd=4096 shape; larger
 * batched prefill keeps using the existing two-stage path. */
int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd != 4096 || n_hc != 4) {
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
        id<MTLBuffer> normbuf = ds4_gpu_tensor_buffer(norm_out);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
            out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_rows64 = out_tensor_bytes / out_row_bytes;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX ||
            n_rows64 > UINT64_MAX / mix_bytes ||
            n_rows64 > UINT64_MAX / residual_row_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm row count is outside supported range\n");
            return 0;
        }

        const uint64_t mix_total_bytes = n_rows64 * mix_bytes;
        const uint64_t residual_total_bytes = n_rows64 * residual_row_bytes;
        const uint64_t out_total_bytes = n_rows64 * out_row_bytes;
        if (!mixbuf || !splitbuf || !xbuf || !outbuf || !normbuf ||
            ds4_gpu_tensor_bytes(mix) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(split) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < residual_total_bytes ||
            ds4_gpu_tensor_bytes(norm_out) < out_total_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm received undersized activation buffers\n");
            return 0;
        }

        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size || out_row_bytes > model_size - norm_weight_offset) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        uint64_t norm_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        id<MTLBuffer> normwbuf = ds4_gpu_wrap_model_range(model_map, model_size, norm_weight_offset, out_row_bytes, &norm_inner);
        if (!scalebuf || !basebuf || !normwbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_hc_split_weighted_sum_norm_pipeline,
                                   "kernel_dsv4_hc_split_weighted_sum_norm4");
        if (!pipeline) return 0;

        ds4_gpu_hc_split_weighted_sum_norm_args args = {
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
            .nb_norm1 = out_row_bytes,
            .eps = eps,
            .norm_eps = norm_eps,
        };

        NSUInteger nth = ds4_gpu_rms_norm_threads(n_embd);
        if (nth > pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm requires %lu threads but pipeline supports %lu\n",
                    (unsigned long)nth,
                    (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:4];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc setBuffer:normwbuf offset:(NSUInteger)norm_inner atIndex:7];
        [enc setBuffer:normbuf offset:ds4_gpu_tensor_offset(norm_out) atIndex:8];
        [enc setThreadgroupMemoryLength:((NSUInteger)n_embd + 4u + 32u) * sizeof(float)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows64, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sum/norm fused")) return 0;
    }

    return 1;
}

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !pre || !model_map || n_hc == 0) return 0;

    @autoreleasepool {
        if ((n_hc % 4u) != 0) {
            fprintf(stderr, "ds4: Metal output HC weights requires a multiple-of-4 HC width\n");
            return 0;
        }

        id<MTLBuffer> prebuf = ds4_gpu_tensor_buffer(pre);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (row_bytes == 0 || out_tensor_bytes < row_bytes || out_tensor_bytes % row_bytes != 0) {
            fprintf(stderr, "ds4: Metal output HC weights size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX ||
            n_tokens64 > UINT64_MAX / row_bytes) {
            fprintf(stderr, "ds4: Metal output HC weights token count is outside supported range\n");
            return 0;
        }

        const uint64_t bytes = n_tokens64 * row_bytes;
        if (!prebuf || !outbuf ||
            ds4_gpu_tensor_bytes(pre) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal output HC weights received undersized buffers\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                            scale_offset, sizeof(float),
                                                            &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           base_offset, row_bytes,
                                                           &base_inner);
        if (!scalebuf || !basebuf) return 0;

        const uint32_t n_tokens = (uint32_t)n_tokens64;
        ds4_gpu_bin_args mul_args = ds4_gpu_make_bin_rows_args(n_hc, n_tokens, 1);
        ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(n_hc, n_tokens, n_hc);
        ds4_gpu_unary_args sigmoid_args = ds4_gpu_make_unary_rows_args(n_hc, n_tokens, 1, 0.0f, 0.0f);
        ds4_gpu_unary_args scale_args = ds4_gpu_make_unary_rows_args(n_hc, n_tokens, 1, 1.0f, eps);

        NSUInteger mul_nth_max = g_bin_mul_scalar_pipeline.maxTotalThreadsPerThreadgroup;
        if (mul_nth_max > 256u) mul_nth_max = 256u;
        NSUInteger mul_nth = 1u;
        while (2u * mul_nth < (NSUInteger)mul_args.ne0 && mul_nth < mul_nth_max) {
            mul_nth *= 2u;
        }

        NSUInteger add_nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (add_nth_max > 256u) add_nth_max = 256u;
        NSUInteger add_nth = 1u;
        while (2u * add_nth < (NSUInteger)add_args.ne0 && add_nth < add_nth_max) {
            add_nth *= 2u;
        }

        NSUInteger unary_nth_max = g_unary_sigmoid_pipeline.maxTotalThreadsPerThreadgroup;
        if (unary_nth_max > 256u) unary_nth_max = 256u;
        NSUInteger unary_nth = (NSUInteger)sigmoid_args.ne00;
        if (unary_nth > unary_nth_max) unary_nth = unary_nth_max;
        if (unary_nth == 0) unary_nth = 1u;
        const NSUInteger unary_nk0 = ((NSUInteger)sigmoid_args.ne00 + unary_nth - 1u) / unary_nth;
        const NSUInteger out_offset = ds4_gpu_tensor_offset(out);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);

        [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
        [enc setBytes:&mul_args length:sizeof(mul_args) atIndex:0];
        [enc setBuffer:prebuf offset:ds4_gpu_tensor_offset(pre) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:outbuf offset:out_offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mul_args.ne01,
                                              (NSUInteger)mul_args.ne02,
                                              (NSUInteger)mul_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(mul_nth, 1, 1)];

        [enc setComputePipelineState:g_add_pipeline];
        [enc setBytes:&add_args length:sizeof(add_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:2];
        [enc setBuffer:outbuf offset:out_offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)add_args.ne01,
                                              (NSUInteger)add_args.ne02,
                                              (NSUInteger)add_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(add_nth, 1, 1)];

        [enc setComputePipelineState:g_unary_sigmoid_pipeline];
        [enc setBytes:&sigmoid_args length:sizeof(sigmoid_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:outbuf offset:out_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(unary_nk0 * (NSUInteger)sigmoid_args.ne01,
                                              (NSUInteger)sigmoid_args.ne02,
                                              (NSUInteger)sigmoid_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(unary_nth, 1, 1)];

        [enc setComputePipelineState:g_unary_scale_pipeline];
        [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:outbuf offset:out_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(unary_nk0 * (NSUInteger)scale_args.ne01,
                                              (NSUInteger)scale_args.ne02,
                                              (NSUInteger)scale_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(unary_nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "output HC weights")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> postbuf = ds4_gpu_tensor_buffer(post);
        id<MTLBuffer> combbuf = ds4_gpu_tensor_buffer(comb);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t comb_values = (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            comb_values > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (comb_values * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t post_bytes = n_tokens64 * (uint64_t)n_hc * sizeof(float);
        const uint64_t comb_bytes = n_tokens64 * comb_values * sizeof(float);
        if (!blockbuf || !resbuf || !postbuf || !combbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(post) < post_bytes ||
            ds4_gpu_tensor_bytes(comb) < comb_bytes) {
            fprintf(stderr, "ds4: Metal HC expand received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = (uint64_t)n_hc * n_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:postbuf offset:ds4_gpu_tensor_offset(post) atIndex:3];
        [enc setBuffer:combbuf offset:ds4_gpu_tensor_offset(comb) atIndex:4];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand")) return 0;
    }

    return 1;
}
