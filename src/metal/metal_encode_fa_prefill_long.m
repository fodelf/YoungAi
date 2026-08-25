/* metal_encode_fa_prefill_long.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(
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
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer = ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "static_mixed_nonvec", (name), n_tokens, n_comp, n_keys, \
                    n_head, head_dim, window, ratio, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");
    if (n_comp &&
        !ds4_gpu_encode_copy_to_f16_1d(cb,
                                       compbuf,
                                       ds4_gpu_tensor_offset(comp_kv),
                                       comp_kv_f16 != 0,
                                       g_flash_attn_kv_buffer,
                                       (NSUInteger)n_tokens * row_bytes_f16,
                                       n_comp * head_dim)) {
        return 0;
    }
    if (n_comp) {
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_comp");
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");
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
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_comp_copy");
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

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
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = nil;
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("block_map");

    ds4_gpu_flash_attn_vec_args args = {
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

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}
