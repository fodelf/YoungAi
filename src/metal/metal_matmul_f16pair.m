/* metal_matmul_f16pair.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok != 1 || (in_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outabuf = ds4_gpu_tensor_buffer(out_a);
        id<MTLBuffer> outbbuf = ds4_gpu_tensor_buffer(out_b);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outabuf || !outbbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out_a) < out_bytes ||
            ds4_gpu_tensor_bytes(out_b) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 paired matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_a_offset > model_size || weight_bytes > model_size - weight_a_offset ||
            weight_b_offset > model_size || weight_bytes > model_size - weight_b_offset) {
            fprintf(stderr, "ds4: Metal F16 paired matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_a = 0;
        uint64_t inner_b = 0;
        id<MTLBuffer> wabuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_a_offset, weight_bytes,
                                                         &inner_a);
        id<MTLBuffer> wbbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_b_offset, weight_bytes,
                                                         &inner_b);
        if (!wabuf || !wbbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
        if (ds4_gpu_use_compressor_pair_nr4() &&
            (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
            mv_dispatch.nr0 = 4;
            mv_dispatch.smem = 32u * 4u * sizeof(float);
        }
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_f16_f32_pair_4", mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wabuf offset:(NSUInteger)inner_a atIndex:1];
        [enc setBuffer:wbbuf offset:(NSUInteger)inner_b atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outabuf offset:ds4_gpu_tensor_offset(out_a) atIndex:4];
        [enc setBuffer:outbbuf offset:ds4_gpu_tensor_offset(out_b) atIndex:5];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 paired matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX || n_tok != 1) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(float);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_f32_mv_args(in_dim, out_dim, 1);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 1);
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F32 tensor matvec")) return 0;
    }

    return 1;
}
