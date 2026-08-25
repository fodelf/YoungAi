/* metal_kv_store.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_set_rows_args;

typedef struct {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
} ds4_gpu_dsv4_kv_fp8_store_args;

static void ds4_gpu_set_rows_thread_shape(
        uint32_t    width,
        NSUInteger *nth_out,
        NSUInteger *nrptg_out) {
    const NSUInteger nk0 = width ? (NSUInteger)width : 1u;
    const NSUInteger max_threads = g_set_rows_f32_i32_pipeline
        ? (NSUInteger)g_set_rows_f32_i32_pipeline.maxTotalThreadsPerThreadgroup
        : 1024u;

    NSUInteger nth = 32u;
    while (nth < nk0 && nth < max_threads) {
        nth *= 2u;
    }

    NSUInteger nrptg = 1u;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1u) / nk0;
        nth = nk0;
        if (nrptg * nth > max_threads) {
            nrptg--;
        }
    }

    if (nth > nk0) nth = nk0;
    if (nth == 0u) nth = 1u;
    if (nrptg == 0u) nrptg = 1u;

    *nth_out = nth;
    *nrptg_out = nrptg;
}

static int ds4_gpu_encode_f16_round_copy_for_raw_store(
        id<MTLCommandBuffer>   cb,
        const ds4_gpu_tensor *src,
        uint32_t               n) {
    id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
    const uint64_t src_bytes = (uint64_t)n * sizeof(float);
    if (!srcbuf || ds4_gpu_tensor_bytes(src) < src_bytes) {
        fprintf(stderr, "ds4: Metal raw KV store received undersized source buffer\n");
        return 0;
    }
    if (!ds4_gpu_ensure_scratch_buffer(&g_f16_round_scratch_buffer,
                                         &g_f16_round_scratch_bytes,
                                         (NSUInteger)n * sizeof(uint16_t),
                                         "ds4_f16_round_scratch") ||
        !ds4_gpu_ensure_scratch_buffer(&g_raw_store_round_buffer,
                                         &g_raw_store_round_bytes,
                                         (NSUInteger)n * sizeof(float),
                                         "ds4_raw_store_round")) {
        return 0;
    }

    ds4_gpu_cpy_args f32_to_f16 =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
    ds4_gpu_cpy_args f16_to_f32 =
        ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
    const NSUInteger nth_f32_f16 = ds4_gpu_cpy_threads(n, g_cpy_f32_f16_pipeline);
    const NSUInteger nth_f16_f32 = ds4_gpu_cpy_threads(n, g_cpy_f16_f32_pipeline);
    const NSUInteger groups_f32_f16 = ((NSUInteger)n + nth_f32_f16 - 1u) / nth_f32_f16;
    const NSUInteger groups_f16_f32 = ((NSUInteger)n + nth_f16_f32 - 1u) / nth_f16_f32;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&f32_to_f16 length:sizeof(f32_to_f16) atIndex:0];
    [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
    [enc setBuffer:g_f16_round_scratch_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_f32_f16, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth_f32_f16, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f32_pipeline];
    [enc setBytes:&f16_to_f32 length:sizeof(f16_to_f32) atIndex:0];
    [enc setBuffer:g_f16_round_scratch_buffer offset:0 atIndex:1];
    [enc setBuffer:g_raw_store_round_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_f16_f32, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth_f16_f32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_set_rows_f32_i32(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *dst,
        id<MTLBuffer>        srcbuf,
        NSUInteger           src_off,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             dst_rows,
        uint32_t             width) {
    id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
    const uint64_t dst_bytes = (uint64_t)dst_rows * width * sizeof(float);
    const uint64_t src_bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!dstbuf || !srcbuf || !rows || n_rows == 0 || width == 0 ||
        ds4_gpu_tensor_bytes(dst) < dst_bytes ||
        src_bytes > NSUIntegerMax - src_off) {
        fprintf(stderr, "ds4: Metal DS4 set_rows received invalid buffers\n");
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rows_bytes = (uint64_t)n_rows * sizeof(int32_t);
    ds4_gpu_set_rows_args args = {
        .nk0 = (int32_t)width,
        .ne01 = (int32_t)n_rows,
        .nb01 = row_bytes,
        .nb02 = (uint64_t)n_rows * row_bytes,
        .nb03 = (uint64_t)n_rows * row_bytes,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(int32_t),
        .nb11 = rows_bytes,
        .nb12 = rows_bytes,
        .nb1 = row_bytes,
        .nb2 = (uint64_t)dst_rows * row_bytes,
        .nb3 = (uint64_t)dst_rows * row_bytes,
    };

    NSUInteger nth;
    NSUInteger nrptg;
    ds4_gpu_set_rows_thread_shape(width, &nth, &nrptg);

    id<MTLBuffer> rowsbuf = nil;
    if (rows_bytes > 4096u) {
        rowsbuf = ds4_gpu_new_transient_buffer((NSUInteger)rows_bytes, "ds4_set_rows_indices");
        if (!rowsbuf) return 0;
        memcpy([rowsbuf contents], rows, (NSUInteger)rows_bytes);
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_set_rows_f32_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:srcbuf offset:src_off atIndex:1];
    if (rowsbuf) {
        [enc setBuffer:rowsbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:rows length:(NSUInteger)rows_bytes atIndex:2];
    }
    [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(dst) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_rows + nrptg - 1u) / nrptg, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, nrptg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !a || !b || !out || n == 0) return 0;

    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    ds4_gpu_bin_args args = {
        .ne00 = (int32_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = row_bytes,
        .nb13 = row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };

    NSUInteger nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)n && nth < nth_max) {
        nth *= 2u;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_add_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV store received undersized destination buffer\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t row_i32 = (int32_t)row;
        if (!ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, head_dim) ||
            !ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               &row_i32,
                                               1,
                                               raw_cap,
                                               head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV DS4 set_rows store")) return 0;
    }

    return 1;
}

/* Release decode fused KV finalizer.  Reference paths are selected by the C
 * graph driver; this Objective-C entry point always means "use the fused
 * Metal kernel." */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !raw_cache || raw_cap == 0 || row >= raw_cap || head_dim == 0 ||
        n_rot > head_dim || raw_cap > INT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_cache);
        const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (!kvbuf || !rawbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal fused KV FP8/raw-store received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_kv_fp8_store_args args = {
            .head_dim = (int32_t)head_dim,
            .n_rot = (int32_t)n_rot,
            .raw_row = (int32_t)row,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_kv_fp8_store_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_cache) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "KV FP8/raw-store fused")) return 0;
    }

    return 1;
}

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV batch store received undersized destination buffer\n");
            return 0;
        }

        int32_t rows_stack[512];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate raw KV set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            rows[t] = (int32_t)((pos0 + t) % raw_cap);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        const uint64_t n = (uint64_t)n_tokens * head_dim;
        const int ok = n <= UINT32_MAX &&
            ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, (uint32_t)n) &&
            ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               rows,
                                               n_tokens,
                                               raw_cap,
                                               head_dim);
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV batch DS4 set_rows store")) return 0;
    }

    return 1;
}
