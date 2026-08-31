/* metal_encode_fa_prefill_vec.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
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


    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    if (n_comp) {
        if (!ds4_gpu_encode_copy_to_f16_1d(cb,
                                           compbuf,
                                           ds4_gpu_tensor_offset(comp_kv),
                                           comp_kv_f16 != 0,
                                           g_flash_attn_kv_buffer,
                                           (NSUInteger)n_tokens * row_bytes_f16,
                                           n_comp * head_dim)) {
            return 0;
        }
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, has_kvpad,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        enc = ds4_gpu_compute_encoder(cb);
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
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
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

    enc = ds4_gpu_compute_encoder(cb);
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
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
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

int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (n_tokens >= DS4_METAL_FA_LONG_MIN_TOKENS) {
        return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(cbp,
                                                                                       heads,
                                                                                       sinks_buf,
                                                                                       sinks_offset,
                                                                                       q,
                                                                                       raw_kv,
                                                                                       comp_kv,
                                                                                       comp_kv_f16,
                                                                                       comp_mask,
                                                                                       use_comp_mask,
                                                                                       n_tokens,
                                                                                       n_comp,
                                                                                       window,
                                                                                       ratio,
                                                                                       n_head,
                                                                                       head_dim);
    }
    return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(cbp,
                                                                           heads,
                                                                           sinks_buf,
                                                                           sinks_offset,
                                                                           q,
                                                                           raw_kv,
                                                                           comp_kv,
                                                                           comp_kv_f16,
                                                                           comp_mask,
                                                                           use_comp_mask,
                                                                           n_tokens,
                                                                           n_comp,
                                                                           window,
                                                                           ratio,
                                                                           n_head,
                                                                           head_dim);
}
