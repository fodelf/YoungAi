/* metal_attention_entry2.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        ratio == 0 || (n_comp != 0 && !comp_kv)) {
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

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(&cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_kv_f16,
                                                                                NULL,
                                                                                0,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill static mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !comp_kv || !comp_mask || !model_map ||
        n_tokens == 0 || n_comp == 0 || ratio == 0) {
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

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(&cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_kv_f16,
                                                                                comp_mask,
                                                                                1,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill masked mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        n_raw > UINT32_MAX - n_comp || n_raw + n_comp > 8192u ||
        (n_comp != 0 && !comp_kv) ||
        (use_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                    (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
        const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
        if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal graph attention heads sink range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
        id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
        if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
            (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal graph attention heads received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size, sinks_offset, sink_bytes, &sinks_inner);
        if (!sinks_buf) return 0;

        if (n_comp == 0) {
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;

            if (!ds4_gpu_encode_flash_attention_raw_heads(cb,
                                                            heads,
                                                            sinks_buf,
                                                            (NSUInteger)sinks_inner,
                                                            q,
                                                            raw_kv,
                                                            n_raw,
                                                            raw_cap,
                                                            raw_start,
                                                            n_head,
                                                            head_dim)) {
                return 0;
            }

            if (!ds4_gpu_finish_command_buffer(cb, owned, "graph raw attention heads")) return 0;
            return 1;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_gathered_heads(cb,
                                                             heads,
                                                             sinks_buf,
                                                             (NSUInteger)sinks_inner,
                                                             q,
                                                             raw_kv,
                                                             n_raw,
                                                             raw_cap,
                                                             raw_start,
                                                             comp_kv,
                                                             comp_kv_f16,
                                                             n_comp,
                                                             comp_mask,
                                                             use_mask,
                                                             n_head,
                                                             head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph attention heads")) return 0;
    }

    return 1;
}
