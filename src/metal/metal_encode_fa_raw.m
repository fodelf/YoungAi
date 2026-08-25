/* metal_encode_fa_raw.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_encode_flash_attention_raw_heads(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *heads,
        id<MTLBuffer>         sinks_buf,
        NSUInteger            sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || raw_cap < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t heads_bytes = q_bytes;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < heads_bytes) {
        fprintf(stderr, "ds4: Metal DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_raw, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }
    memset([mask_buffer contents], 0, mask_bytes);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_raw % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, (n_raw % ncpsg) != 0,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        const uint32_t tail_elems = tail_rows * head_dim;
        const uint32_t head_elems = head_rows * head_dim;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }

        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_elems)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_elems))) {
            return 0;
        }

        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }

    if ((n_raw % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

void ds4_gpu_fill_raw_prefill_mask(uint16_t *mask, uint32_t n_tokens, uint32_t window) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_tokens;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= q;
            const bool in_window = window == 0 || q - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

void ds4_gpu_fill_raw_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  pos0,
        uint32_t  window) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* The caller has already copied the SWA ring into logical order when it
     * wraps, so key row k represents first_raw_pos + k. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_raw;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

void ds4_gpu_fill_mixed_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  n_comp,
        uint32_t  pos0,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* Raw keys are laid out by logical position; compressed keys follow them. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
        const uint32_t n_visible = (qpos + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_raw + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

void ds4_gpu_fill_static_mixed_prefill_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_comp,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_tokens + n_comp;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= q;
            const bool in_window = window == 0 || q - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }

        const uint32_t n_visible = (q + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_tokens + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}
