/* metal_attention_entry1.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
} ds4_gpu_dsv4_indexed_attention_args;

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0) return 0;

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_raw_heads(&cb,
                                                                heads,
                                                                sinks_buf,
                                                                (NSUInteger)sinks_inner,
                                                                q,
                                                                raw_kv,
                                                                n_tokens,
                                                                window,
                                                                n_head,
                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill raw attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                     heads,
                                                                     sinks_buf,
                                                                     (NSUInteger)sinks_inner,
                                                                     q,
                                                                     raw_kv,
                                                                     n_tokens,
                                                                     pos0,
                                                                     n_raw,
                                                                     raw_cap,
                                                                     raw_start,
                                                                     window,
                                                                     n_head,
                                                                     head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode raw batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || (n_comp != 0 && !comp_kv) ||
        (use_comp_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       (NSUInteger)sinks_inner,
                                                                       q,
                                                                       raw_kv,
                                                                       comp_kv,
                                                                       comp_mask,
                                                                       use_comp_mask,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       n_comp,
                                                                       window,
                                                                       ratio,
                                                                       n_head,
                                                                       head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode mixed batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_kv || !topk ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 || top_k > n_comp || (top_k & (top_k - 1u)) != 0 ||
        ratio == 0 || n_head == 0 || head_dim != 512) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal indexed attention sinks range is outside the mapped model\n");
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * row_bytes;
        const uint64_t raw_bytes = (uint64_t)raw_cap * row_bytes;
        const uint64_t comp_bytes = (uint64_t)n_comp * DS4_GPU_COMP_ROW_BYTES;   /* 压缩缓存行格式, 见 ds4_gpu_core.h */
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        if (!qbuf || !rawbuf || !compbuf || !topkbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal indexed mixed attention received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        id<MTLComputePipelineState> sort_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline,
                                    "kernel_dsv4_sort_i32_rows_asc");
        const bool decode_one_token = n_tokens == 1u;
        id<MTLComputePipelineState> attn_pipeline =
            decode_one_token ?
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_rb16_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8_rb16") :
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8");
        if (!sort_pipeline || !attn_pipeline) return 0;
        if ((NSUInteger)top_k > sort_pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal indexed attention top-k exceeds sort threadgroup limit\n");
            return 0;
        }
        /*
         * Fast decode attends to the same full top-k compressed rows but keeps
         * them in score order, avoiding a chronological sort dispatch.
         * --quality restores the sorted order for stricter reproducibility.
         */
        const bool skip_decode_sort = !g_quality_mode && decode_one_token;
        if (!skip_decode_sort &&
            !ds4_gpu_ensure_scratch_buffer(&g_indexed_topk_buffer,
                                             &g_indexed_topk_bytes,
                                             (NSUInteger)topk_bytes,
                                             "ds4_indexed_topk_sorted")) {
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args sort_args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)top_k,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(int32_t),
            .nb1 = (uint64_t)top_k * sizeof(int32_t),
        };
        ds4_gpu_dsv4_indexed_attention_args attn_args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .n_raw = n_raw,
            .raw_cap = raw_cap,
            .raw_start = raw_start,
            .n_comp = n_comp,
            .top_k = top_k,
            .pos0 = pos0,
            .window = window,
            .ratio = ratio,
            .q_token_stride = (uint64_t)n_head * row_bytes,
            .q_head_stride = row_bytes,
            .raw_row_stride = row_bytes,
            .comp_row_stride = DS4_GPU_COMP_ROW_BYTES,
            .topk_token_stride = (uint64_t)top_k * sizeof(int32_t),
            .dst_token_stride = (uint64_t)n_head * row_bytes,
            .dst_head_stride = row_bytes,
            .scale = 1.0f / sqrtf((float)head_dim),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = nil;
        if (!skip_decode_sort) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:sort_pipeline];
            [enc setBytes:&sort_args length:sizeof(sort_args) atIndex:0];
            [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
            [enc setBuffer:g_indexed_topk_buffer offset:0 atIndex:2];
            [enc setThreadgroupMemoryLength:(NSUInteger)top_k * sizeof(int32_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(top_k, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_kv) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(comp_kv) atIndex:3];
        [enc setBuffer:skip_decode_sort ? topkbuf : g_indexed_topk_buffer
              offset:skip_decode_sort ? ds4_gpu_tensor_offset(topk) : 0
             atIndex:4];
        [enc setBuffer:sinks_buf offset:(NSUInteger)sinks_inner atIndex:5];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:6];
        [enc setThreadgroupMemoryLength:(decode_one_token ? 16u : 1u) *
                                        128u * 4u * sizeof(uint16_t)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, ((NSUInteger)n_head + 7u) / 8u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph indexed mixed attention heads")) return 0;
    }

    return 1;
}
