/* metal_norm_rope.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    int32_t  q_n;
    int32_t  q_n4;
    int32_t  kv_n;
    int32_t  kv_n4;
    uint64_t q_row_stride;
    uint64_t kv_row_stride;
    float    eps;
} ds4_gpu_qkv_rms_norm_args;

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_rot;
} ds4_gpu_dsv4_fp8_kv_quantize_args;

typedef struct {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
} ds4_gpu_dsv4_indexer_qat_args;

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> rowbuf = ds4_gpu_tensor_buffer(row);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_bytes = row_bytes * n_hc;
        if (!rowbuf || !outbuf ||
            ds4_gpu_tensor_bytes(row) < row_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal HC repeat received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  rowbuf,
                                                  ds4_gpu_tensor_offset(row),
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC repeat")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal plain RMS norm received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "plain RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        const uint64_t bytes = row_bytes * rows;
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal weighted RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal weighted RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, row_bytes, &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "weighted RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_out || !kv || q_n == 0 || kv_n == 0 || rows == 0 ||
        (q_n & 3u) != 0 || (kv_n & 3u) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> kvoutbuf = ds4_gpu_tensor_buffer(kv_out);

        const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
        const uint64_t kv_row_bytes = (uint64_t)kv_n * sizeof(float);
        if (!qbuf || !qoutbuf || !kvbuf || !kvoutbuf ||
            ds4_gpu_tensor_bytes(q) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(q_out) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv) < kv_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv_out) < kv_row_bytes * rows) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
            kv_weight_offset > model_size || kv_row_bytes > model_size - kv_weight_offset) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm weight range is outside the mapped model\n");
            return 0;
        }

        uint64_t q_inner_offset = 0;
        uint64_t kv_inner_offset = 0;
        id<MTLBuffer> q_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                          q_weight_offset, q_row_bytes,
                                                          &q_inner_offset);
        if (!q_wbuf) return 0;
        id<MTLBuffer> kv_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           kv_weight_offset, kv_row_bytes,
                                                           &kv_inner_offset);
        if (!kv_wbuf) return 0;

        ds4_gpu_qkv_rms_norm_args args = {
            .q_n = (int32_t)q_n,
            .q_n4 = (int32_t)(q_n / 4u),
            .kv_n = (int32_t)kv_n,
            .kv_n4 = (int32_t)(kv_n / 4u),
            .q_row_stride = q_row_bytes,
            .kv_row_stride = kv_row_bytes,
            .eps = eps,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_qkv_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:q_wbuf offset:(NSUInteger)q_inner_offset atIndex:2];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:3];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:4];
        [enc setBuffer:kv_wbuf offset:(NSUInteger)kv_inner_offset atIndex:5];
        [enc setBuffer:kvoutbuf offset:ds4_gpu_tensor_offset(kv_out) atIndex:6];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(q_n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "fused q/kv RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || (head_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal head RMS norm received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_3d_args(head_dim, n_head, n_tok, eps);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_pipeline_threads(head_dim, g_rms_norm_plain_pipeline), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "head RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u) != 0) {
        return 0;
    }
    if (n_rot == 0) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal RoPE received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rope_tail_batch_args args = ds4_gpu_make_rope_tail_args(
            n_tok, n_head, head_dim, n_rot, n_ctx_orig, inverse,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_rope_tail_inplace(cb,
                                                xbuf,
                                                ds4_gpu_tensor_offset(x),
                                                &args,
                                                n_tok,
                                                n_head,
                                                head_dim,
                                                pos0,
                                                1)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "RoPE tail")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (n_rot == head_dim) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 FP8 KV quantize received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_fp8_kv_quantize_args args = {
            .ne00 = head_dim,
            .ne01 = n_tok,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)head_dim * sizeof(float),
            .nb02 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb03 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)head_dim * sizeof(float),
            .nb2 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb3 = (uint64_t)n_tok * head_dim * sizeof(float),
            .n_rot = (int32_t)n_rot,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_fp8_kv_quantize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 FP8 KV quantize")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_rows == 0) return 0;
    if (head_dim != 128u) {   /* shader 布局写死 128 宽(absbuf=scratch+128); 静默跳过=QAT 悄没做 */
        fprintf(stderr, "ds4: Metal DSV4 indexer QAT expects 128-wide rows (got %u), skipped\n", head_dim);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_rows * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 indexer QAT received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_indexer_qat_args args = {
            .n_rows = n_rows,
            .head_dim = head_dim,
            .row_stride = (uint64_t)head_dim * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_qat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 indexer Hadamard+FP4")) return 0;
    }

    return 1;
}
