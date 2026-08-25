/* metal_compressor_store.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
} ds4_gpu_dsv4_compressor_store_one_args;

int ds4_gpu_encode_compressor_score_with_ape(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        score_src,
        NSUInteger           score_src_offset,
        id<MTLBuffer>        score_dst,
        NSUInteger           score_dst_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        uint32_t             n_tokens) {
    if (!cb || !score_src || !score_dst || !apebuf ||
        width == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    const uint64_t total_elems64 = (uint64_t)n_tokens * width;
    if (total_elems64 > UINT32_MAX) {
        fprintf(stderr, "ds4: Metal compressor APE add received too many elements\n");
        return 0;
    }
    const uint32_t total_elems = (uint32_t)total_elems64;
    const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                         &g_compressor_store_ape_bytes,
                                         scratch_bytes,
                                         "ds4_compressor_store_ape")) {
        return 0;
    }

    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    uint32_t copied_rows = 0;
    uint32_t pos_mod = pos0 % ratio;
    while (copied_rows < n_tokens) {
        uint32_t seg_rows = ratio - pos_mod;
        if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
        const uint32_t seg_elems = seg_rows * width;
        const NSUInteger src_off = ape_offset + (NSUInteger)pos_mod * width * elem_ape;
        const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
        int ok;
        if (ape_type == 1u) {
            ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        } else {
            ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        }
        if (!ok) return 0;
        copied_rows += seg_rows;
        pos_mod = 0;
    }

    return ds4_gpu_encode_add_f32_1d(cb,
                                       score_src,
                                       score_src_offset,
                                       g_compressor_store_ape_buffer,
                                       0,
                                       score_dst,
                                       score_dst_offset,
                                       total_elems);
}

int ds4_gpu_encode_compressor_set_rows_projected(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             state_rows) {
    if (!cb || !state_kv || !state_score || !kvbuf || !scorebuf ||
        !apebuf || !rows || width == 0 || n_rows == 0 || state_rows == 0) {
        return 0;
    }

    const NSUInteger score_scratch_bytes = (NSUInteger)n_rows * width * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                         &g_compressor_store_score_bytes,
                                         score_scratch_bytes,
                                         "ds4_compressor_store_score")) {
        return 0;
    }

    return ds4_gpu_encode_compressor_score_with_ape(cb,
                                                      scorebuf,
                                                      score_offset,
                                                      g_compressor_store_score_buffer,
                                                      0,
                                                      apebuf,
                                                      ape_offset,
                                                      ape_type,
                                                      width,
                                                      ratio,
                                                      pos0,
                                                      n_rows) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_kv,
                                             kvbuf,
                                             kv_offset,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_score,
                                             g_compressor_store_score_buffer,
                                             0,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width);
}

int ds4_gpu_compressor_store_one_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                width,
        uint32_t                ratio,
        uint32_t                pos) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        width == 0 || ratio == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_hot_pipeline(g_dsv4_compressor_store_one_pipeline,
                                "kernel_dsv4_compressor_store_one");
    if (!pipeline) return 0;

    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        ds4_gpu_tensor_bytes(kv) < row_bytes ||
        ds4_gpu_tensor_bytes(sc) < row_bytes ||
        ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    uint64_t ape_inner = 0;
    id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                       ape_offset, ape_bytes,
                                                       &ape_inner);
    id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
    id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!apebuf || !kvbuf || !scbuf || !statekvbuf || !statescbuf) return 0;

    ds4_gpu_dsv4_compressor_store_one_args args = {
        .width = width,
        .ratio = ratio,
        .pos = pos,
        .ape_type = ape_type,
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;

    const NSUInteger nth = 256u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
    [enc setBuffer:scbuf offset:ds4_gpu_tensor_offset(sc) atIndex:2];
    [enc setBuffer:apebuf offset:(NSUInteger)ape_inner atIndex:3];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:4];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)width + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return ds4_gpu_finish_command_buffer(cb, owned, "compressor one-row store");
}

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor batch APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        if (!kvbuf || !scbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor batch store received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const uint64_t total_elems64 = (uint64_t)n_tokens * width;
        if (total_elems64 > UINT32_MAX || state_rows > INT32_MAX) {
            fprintf(stderr, "ds4: Metal compressor batch store received too many elements\n");
            return 0;
        }
        const uint32_t total_elems = (uint32_t)total_elems64;
        const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                             &g_compressor_store_ape_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_ape") ||
            !ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                             &g_compressor_store_score_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_score")) {
            return 0;
        }

        int32_t rows_stack[16];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate compressor set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t pos_mod = (pos0 + t) % ratio;
            rows[t] = (int32_t)(ratio == 4u ? ratio + pos_mod : pos_mod);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        int ok = 1;
        uint32_t copied_rows = 0;
        uint32_t pos_mod = pos0 % ratio;
        while (ok && copied_rows < n_tokens) {
            uint32_t seg_rows = ratio - pos_mod;
            if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
            const uint32_t seg_elems = seg_rows * width;
            const NSUInteger src_off = (NSUInteger)ape_inner +
                                       (NSUInteger)pos_mod * width * elem_ape;
            const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
            if (ape_type == 1u) {
                ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            } else {
                ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            }
            copied_rows += seg_rows;
            pos_mod = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_add_f32_1d(cb,
                                             scbuf,
                                             ds4_gpu_tensor_offset(sc),
                                             g_compressor_store_ape_buffer,
                                             0,
                                             g_compressor_store_score_buffer,
                                             0,
                                             total_elems);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_kv,
                                                   kvbuf,
                                                   ds4_gpu_tensor_offset(kv),
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_score,
                                                   g_compressor_store_score_buffer,
                                                   0,
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "compressor batch DS4 store")) return 0;
    }

    return 1;
}
