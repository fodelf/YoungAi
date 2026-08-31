/* metal_compressor_update.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_cur);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_cur);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        if (!kvbuf || !scbuf || !compbuf ||
            ds4_gpu_tensor_bytes(kv_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (emit && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor update received undersized buffers\n");
            return 0;
        }

        if (!ds4_gpu_compressor_store_one_tensor(kv_cur,
                                                 sc_cur,
                                                 state_kv,
                                                 state_score,
                                                 model_map,
                                                 model_size,
                                                 ape_offset,
                                                 ape_type,
                                                 width,
                                                 ratio,
                                                 pos)) {
            return 0;
        }
        if (!emit) return 1;

        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                comp_cache,
                (uint64_t)comp_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
        if (!comp_row_view) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_compressor_pool(cb,
                                                  comp_row_view,
                                                  state_kv,
                                                  state_score,
                                                  head_dim,
                                                  ratio);
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor DS4 softmax pool");
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view,
                                                       comp_row_view,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       1,
                                                       rms_eps) != 0;
        }
        if (ok) {
            const uint32_t comp_pos = pos + 1u - ratio;
            ok = ds4_gpu_rope_tail_tensor(comp_row_view,
                                            1,
                                            1,
                                            head_dim,
                                            n_rot,
                                            comp_pos,
                                            n_ctx_orig,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            beta_fast,
                                            beta_slow) != 0;
        }
        if (ok && ratio == 4u) {
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb &&
                 ds4_gpu_encode_compressor_shift_ratio4(cb,
                                                          state_kv,
                                                          state_score,
                                                          width);
            if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor ratio4 state shift");
        }
        ds4_gpu_tensor_free(comp_row_view);
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value) {
    if (!cb || !buf || width == 0 || rows == 0 || (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, 1, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth_max = g_unary_fill_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
