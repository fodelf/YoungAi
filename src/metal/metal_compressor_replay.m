/* metal_compressor_replay.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint32_t n_comp = n_tokens / ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor replay tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            ds4_gpu_tensor_bytes(comp_cache) < comp_bytes) {
            fprintf(stderr, "ds4: Metal compressor replay received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        const NSUInteger score_bytes = (NSUInteger)n_tokens * width * sizeof(float);
        const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
        if (ok && (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                    &g_compressor_store_score_bytes,
                                                    score_bytes,
                                                    "ds4_compressor_store_score") ||
                   !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                    &g_compressor_pool_kv_bytes,
                                                    pack_bytes,
                                                    "ds4_compressor_pool_kv") ||
                   !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                    &g_compressor_pool_score_bytes,
                                                    pack_bytes,
                                                    "ds4_compressor_pool_score"))) {
            ok = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                            scbuf,
                                                            ds4_gpu_tensor_offset(sc),
                                                            g_compressor_store_score_buffer,
                                                            0,
                                                            apebuf,
                                                            (NSUInteger)ape_inner,
                                                            ape_type,
                                                            width,
                                                            ratio,
                                                            pos0,
                                                            n_tokens);
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_kv_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_score_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                ds4_gpu_negative_infinity());
        }

        const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
        const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
        const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
        const uint64_t dst_plane_stride = 8ull * dst_row_stride;
        const NSUInteger state_off = ds4_gpu_tensor_offset(state_kv);
        const NSUInteger state_score_off = ds4_gpu_tensor_offset(state_score);

        if (ok) {
            /*
             * The aligned nonzero ratio-4 path replays the current ubatch
             * compressor, but seeds the first compressed row with the previous
             * compressor state. Rows 0..3 are the previous half, rows 4..7 are
             * the current half.
             */
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statekvbuf,
                                                 state_off,
                                                 g_compressor_pool_kv_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statescbuf,
                                                 state_score_off,
                                                 g_compressor_pool_score_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv) +
                                                         (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_kv_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_score_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok && n_comp > 1u) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv),
                                                 g_compressor_pool_kv_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 0,
                                                 g_compressor_pool_score_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok) {
            ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                    comp_cache,
                                                    g_compressor_pool_kv_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    g_compressor_pool_score_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    8,
                                                    head_dim,
                                                    n_comp);
        }
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                state_off,
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                state_score_off,
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const uint32_t prev_start = n_tokens - ratio;
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0 + prev_start,
                                                                 rows_prev,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill-state APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_tail);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_tail);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(sc_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor prefill-state received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows[4] = { 0, 1, 2, 3 };
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv_tail),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc_tail),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 rows,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}
