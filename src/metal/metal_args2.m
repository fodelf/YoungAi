/* metal_args2.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return 0;
    }
}

const char *ds4_gpu_mv_ext_name(int q8, int16_t r1ptg) {
    if (q8) {
        switch (r1ptg) {
        case 2: return "kernel_mul_mv_ext_q8_0_f32_r1_2";
        case 3: return "kernel_mul_mv_ext_q8_0_f32_r1_3";
        case 4: return "kernel_mul_mv_ext_q8_0_f32_r1_4";
        case 5: return "kernel_mul_mv_ext_q8_0_f32_r1_5";
        default: return NULL;
        }
    }

    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f16_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f16_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f16_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f16_f32_r1_5";
    default: return NULL;
    }
}

ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_3d_args(
        uint32_t n0,
        uint32_t n1,
        uint32_t n2,
        float    eps) {
    const uint64_t row_bytes = (uint64_t)n0 * sizeof(float);
    const uint64_t plane_bytes = row_bytes * n1;
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n0,
        .ne00_t = (int32_t)(n0 / 4u),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = plane_bytes * n2,
        .eps = eps,
        .nef1 = { (int32_t)n1, 1, 1 },
        .nef2 = { (int32_t)n2, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { plane_bytes, row_bytes, row_bytes },
        .nbf3 = { plane_bytes * n2, row_bytes, row_bytes },
    };
}

NSUInteger ds4_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

NSUInteger ds4_gpu_rms_norm_pipeline_threads(
        uint32_t                  n,
        id<MTLComputePipelineState> pipeline) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger max_threads = pipeline ? [pipeline maxTotalThreadsPerThreadgroup] : 1024u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

ds4_gpu_rope_tail_batch_args ds4_gpu_make_rope_tail_args(
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        bool     inverse,
        float    freq_base,
        float    freq_scale,
        float    ext_factor,
        float    attn_factor,
        float    beta_fast,
        float    beta_slow) {
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t tok_bytes = (uint64_t)n_head * row_bytes;
    return (ds4_gpu_rope_tail_batch_args) {
        .ne00 = head_dim,
        .ne01 = n_head,
        .ne02 = n_tok,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = tok_bytes,
        .nb03 = (uint64_t)n_tok * tok_bytes,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = tok_bytes,
        .nb3 = (uint64_t)n_tok * tok_bytes,
        .n_dims = (int32_t)n_rot,
        .mode = 0,
        .n_ctx_orig = (int32_t)n_ctx_orig,
        .inverse = inverse ? 1 : 0,
        .freq_base = freq_base,
        .freq_scale = freq_scale,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .src2 = false,
    };
}

int ds4_gpu_encode_rope_tail_inplace(
        id<MTLCommandBuffer>                 cb,
        id<MTLBuffer>                        xbuf,
        NSUInteger                           xoff,
        const ds4_gpu_rope_tail_batch_args *args,
        uint32_t                             n_tok,
        uint32_t                             n_head,
        uint32_t                             head_dim,
        uint32_t                             pos0,
        uint32_t                             pos_step) {
    int32_t pos_stack[256];
    int32_t *pos = pos_stack;
    if (n_tok > (uint32_t)(sizeof(pos_stack) / sizeof(pos_stack[0]))) {
        pos = malloc((size_t)n_tok * sizeof(*pos));
        if (!pos) {
            fprintf(stderr, "ds4: failed to allocate Metal RoPE position buffer\n");
            return 0;
        }
    }
    for (uint32_t t = 0; t < n_tok; t++) pos[t] = (int32_t)(pos0 + t * pos_step);

    const NSUInteger pos_bytes = (NSUInteger)n_tok * sizeof(*pos);
    id<MTLBuffer> posbuf = nil;
    if (pos_bytes > 4096u) {
        /*
         * Metal inline setBytes data is meant for small constants. Long prefill
         * RoPE calls need thousands of positions; passing that much inline can
         * make the Apple driver abort the process instead of reporting a normal
         * API error.
         */
        posbuf = ds4_gpu_new_transient_buffer(pos_bytes, "ds4_rope_positions");
        if (!posbuf) {
            if (pos != pos_stack) free(pos);
            return 0;
        }
        memcpy([posbuf contents], pos, pos_bytes);
    }

    const NSUInteger nth = (NSUInteger)(head_dim < 256u ? head_dim : 256u);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_rope_tail_batch_pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    if (posbuf) {
        [enc setBuffer:posbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:pos length:pos_bytes atIndex:2];
    }
    [enc setBuffer:xbuf offset:xoff atIndex:3];
    [enc setBuffer:xbuf offset:xoff atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
         threadsPerThreadgroup:MTLSizeMake(nth ? nth : 1u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    if (pos != pos_stack) free(pos);
    return 1;
}
