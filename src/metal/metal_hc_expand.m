/* metal_hc_expand.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand split output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand split token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            mix_hc > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (mix_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand split activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t split_bytes = n_tokens64 * mix_hc * sizeof(float);
        if (!blockbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes) {
            fprintf(stderr, "ds4: Metal HC expand split received undersized activation buffers\n");
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
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
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
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:3];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:4];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand split")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> addbuf = ds4_gpu_tensor_buffer(block_add);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand add split output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand add split token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            mix_hc > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (mix_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand add split activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t split_bytes = n_tokens64 * mix_hc * sizeof(float);
        if (!blockbuf || !addbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(block_add) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes) {
            fprintf(stderr, "ds4: Metal HC expand add split received undersized activation buffers\n");
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
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 1,
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
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:3];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:4];
        [enc setBuffer:addbuf offset:ds4_gpu_tensor_offset(block_add) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand add split")) return 0;
    }

    return 1;
}

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        const ds4_gpu_tensor *corr_delta,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out ||
        !residual_hc || !split || n_embd == 0 || n_hc == 0 ||
        n_hc != 4 || out_dim != n_embd || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(shared_mid);
        id<MTLBuffer> sharedbuf = ds4_gpu_tensor_buffer(shared_out);
        id<MTLBuffer> routedbuf = ds4_gpu_tensor_buffer(routed_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        const uint64_t shared_mid_bytes = in_dim * sizeof(float);
        const uint64_t embd_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t split_bytes = mix_hc * sizeof(float);

        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal shared-down HC fusion weight range is outside the mapped model\n");
            return 0;
        }
        if (!midbuf || !sharedbuf || !routedbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(shared_mid) < shared_mid_bytes ||
            ds4_gpu_tensor_bytes(shared_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(routed_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes ||
            ds4_gpu_tensor_bytes(out_hc) < hc_bytes) {
            fprintf(stderr, "ds4: Metal shared-down HC fusion received undersized buffers\n");
            return 0;
        }
        id<MTLBuffer> deltabuf = nil;
        if (corr_delta) {
            deltabuf = ds4_gpu_tensor_buffer(corr_delta);
            if (!deltabuf || ds4_gpu_tensor_bytes(corr_delta) < embd_bytes) {
                fprintf(stderr, "ds4: Metal shared-down HC fusion corr_delta undersized\n");
                return 0;
            }
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes,
                                                        &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        mv_args.nr0 = mv_dispatch.nr0;

        ds4_gpu_hc_expand_args hc_args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = 1,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 1,
            .has_corr_delta = corr_delta ? 1 : 0,
        };

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_shared_down_hc_expand4_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&hc_args length:sizeof(hc_args) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(shared_mid) atIndex:3];
        [enc setBuffer:sharedbuf offset:ds4_gpu_tensor_offset(shared_out) atIndex:4];
        [enc setBuffer:routedbuf offset:ds4_gpu_tensor_offset(routed_out) atIndex:5];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:6];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:7];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:8];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:9];
        /* dummy-bind routed_out when absent: flag 0 means the kernel never reads it */
        [enc setBuffer:(deltabuf ? deltabuf : routedbuf)
                offset:(deltabuf ? ds4_gpu_tensor_offset(corr_delta) : ds4_gpu_tensor_offset(routed_out))
               atIndex:10];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                              (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "shared-down HC expand fused")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !model_map || !x || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0 || n_hc != 4 || out_dim != n_embd ||
        (in_dim & 31u) != 0 || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t embd_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t split_bytes = mix_hc * sizeof(float);

        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8 HC fusion weight range is outside the mapped model\n");
            return 0;
        }
        if (!xbuf || !blockbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(block_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes ||
            ds4_gpu_tensor_bytes(out_hc) < hc_bytes) {
            fprintf(stderr, "ds4: Metal Q8 HC fusion received undersized buffers\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes,
                                                        &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        mv_args.nr0 = mv_dispatch.nr0;

        ds4_gpu_hc_expand_args hc_args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = 1,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_q8_hc_expand4_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&hc_args length:sizeof(hc_args) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:4];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:5];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:6];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:7];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:8];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                              (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8 HC expand fused")) return 0;
    }

    return 1;
}
