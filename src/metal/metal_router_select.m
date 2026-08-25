/* metal_router_select.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
} ds4_gpu_dsv4_router_select_one_args;

int ds4_gpu_encode_router_select(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *selected,
        ds4_gpu_tensor     *weights,
        ds4_gpu_tensor     *probs,
        id<MTLBuffer>         logitsbuf,
        NSUInteger            logits_off,
        id<MTLBuffer>         biasbuf,
        NSUInteger            bias_off,
        id<MTLBuffer>         hashbuf,
        NSUInteger            hash_off,
        id<MTLBuffer>         tokensbuf,
        NSUInteger            tokens_off,
        const int32_t        *single_token,
        uint32_t              hash_rows,
        uint32_t              n_tokens,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale,
        bool                  has_bias,
        bool                  hash_mode) {
    id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
    id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
    id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
    const NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
    const NSUInteger weights_off = ds4_gpu_tensor_offset(weights);
    const NSUInteger probs_off = ds4_gpu_tensor_offset(probs);

    if (!cb || !selectedbuf || !weightsbuf || !probsbuf || !logitsbuf ||
        n_tokens == 0 || n_expert == 0 || n_expert_used == 0) return 0;

    const NSUInteger probs_bytes = (NSUInteger)n_tokens * (NSUInteger)n_expert * sizeof(float);
    const bool flash_router_fast_path =
        n_expert == 256u &&
        n_expert_used == 6u &&
        fabsf(expert_weight_scale - 1.5f) <= 1.0e-6f;

    int ok = 0;
    if (flash_router_fast_path &&
        !g_quality_mode && n_tokens == 1 &&
        getenv("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION") == NULL) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        id<MTLComputePipelineState> router_finalize_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_finalize_one_pipeline,
                                    "kernel_dsv4_router_finalize_one");
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!softplus_sqrt_pipeline || !router_finalize_pipeline || !router_weights_pipeline) return 0;

        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
        if (!ok) return 0;

        const bool use_token_buffer = single_token == NULL;
        ds4_gpu_dsv4_router_select_one_args args = {
            .has_bias = has_bias ? 1u : 0u,
            .hash_mode = hash_mode ? 1u : 0u,
            .use_token_buffer = use_token_buffer ? 1u : 0u,
            .token = single_token ? (uint32_t)*single_token : 0u,
            .hash_rows = hash_rows,
        };

        const float zero_f32 = 0.0f;
        const int32_t zero_i32 = 0;
        if ((has_bias && !biasbuf) ||
            (hash_mode && !hashbuf) ||
            (use_token_buffer && !tokensbuf)) {
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_finalize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:probsbuf offset:probs_off atIndex:1];
        if (has_bias) {
            [enc setBuffer:biasbuf offset:bias_off atIndex:2];
        } else {
            [enc setBytes:&zero_f32 length:sizeof(zero_f32) atIndex:2];
        }
        if (hash_mode) {
            [enc setBuffer:hashbuf offset:hash_off atIndex:3];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:3];
        }
        if (use_token_buffer) {
            [enc setBuffer:tokensbuf offset:tokens_off atIndex:4];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:4];
        }
        [enc setBuffer:selectedbuf offset:selected_off atIndex:5];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) + 256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    const NSUInteger sum_bytes = (NSUInteger)n_tokens * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_router_weight_sum_buffer,
                                         &g_router_weight_sum_bytes,
                                         sum_bytes,
                                         "ds4_router_weight_sum")) {
        return 0;
    }

    if (flash_router_fast_path && !g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        ok = softplus_sqrt_pipeline &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
    } else {
        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_softplus_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f) &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_sqrt_pipeline,
                                             probsbuf,
                                             probs_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f);
    }
    if (!ok) return 0;

    if (hash_mode) {
        ok = ds4_gpu_encode_get_rows_i32_token_rows(cb,
                                                      hashbuf,
                                                      hash_off,
                                                      tokensbuf,
                                                      tokens_off,
                                                      single_token,
                                                      selectedbuf,
                                                      selected_off,
                                                      hash_rows,
                                                      n_expert_used,
                                                      n_tokens);
    } else {
        ds4_gpu_tensor *score_tensor = probs;
        DS4MetalTensor *selection_view = nil;

        if (has_bias) {
            if (!biasbuf ||
                !ds4_gpu_ensure_scratch_buffer(&g_router_selection_buffer,
                                                 &g_router_selection_bytes,
                                                 probs_bytes,
                                                 "ds4_router_selection")) {
                return 0;
            }

            ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(n_expert, n_tokens, n_expert);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_add_pipeline,
                                               &add_args,
                                               probsbuf,
                                               probs_off,
                                               biasbuf,
                                               bias_off,
                                               g_router_selection_buffer,
                                               0);
            if (!ok) return 0;

            selection_view = [DS4MetalTensor new];
            selection_view.buffer = g_router_selection_buffer;
            selection_view.offset = 0;
            selection_view.bytes = probs_bytes;
            selection_view.owner = 0;
            score_tensor = (__bridge ds4_gpu_tensor *)selection_view;
        }

        ok = ds4_gpu_indexer_topk_tensor(selected, score_tensor, n_expert, n_tokens, n_expert_used) != 0;
    }
    if (!ok) return 0;

    if (flash_router_fast_path && !g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!router_weights_pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    ok = ds4_gpu_encode_get_rows_f32_router_weights(cb,
                                                      probsbuf,
                                                      probs_off,
                                                      selectedbuf,
                                                      selected_off,
                                                      weightsbuf,
                                                      weights_off,
                                                      n_expert,
                                                      n_expert_used,
                                                      n_tokens) &&
         ds4_gpu_encode_sum_rows_f32(cb,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       n_expert_used,
                                       n_tokens) &&
         ds4_gpu_encode_unary_f32_rows(cb,
                                         g_unary_clamp_pipeline,
                                         g_router_weight_sum_buffer,
                                         0,
                                         g_router_weight_sum_buffer,
                                         0,
                                         1,
                                         n_tokens,
                                         0,
                                         6.103515625e-5f,
                                         ds4_gpu_positive_infinity());
    if (!ok) return 0;

    ds4_gpu_bin_args div_args = ds4_gpu_make_bin_rowwise_scalar_args(n_expert_used, n_tokens);
    const float scale = expert_weight_scale;
    ds4_gpu_bin_args scale_args = ds4_gpu_make_bin_rows_args(n_expert_used, n_tokens, 1);

    ok = ds4_gpu_encode_bin_f32_rows(cb,
                                       g_bin_div_row_pipeline,
                                       &div_args,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       weightsbuf,
                                       weights_off);
    if (!ok) return 0;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
    [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:1];
    [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)scale_args.ne1,
                                          (NSUInteger)scale_args.ne2,
                                          (NSUInteger)scale_args.ne3)
         threadsPerThreadgroup:MTLSizeMake(ds4_gpu_bin_threads(n_expert_used, g_bin_mul_scalar_pipeline), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

/* Upload the reduced-expert original-id -> compact-slot LUT into a small resident
 * GPU buffer. n_layer * 256 int16 (~21 KiB for 43 layers). Called once at load
 * for a shrunken model; a full model never calls this so the translation kernel
 * stays a no-op. Replaces any previous LUT. */
int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!lut || n_layer == 0) return 0;
    @autoreleasepool {
        const NSUInteger bytes = (NSUInteger)n_layer * 256u * sizeof(int16_t);
        id<MTLBuffer> buf = [g_device newBufferWithBytes:lut
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        if (!buf) {
            fprintf(stderr, "ds4: failed to allocate Metal expert keep-map LUT (%llu bytes)\n",
                    (unsigned long long)bytes);
            return 0;
        }
        buf.label = @"ds4_expert_keep_lut";
        g_expert_keep_lut_buffer = buf;
        g_expert_keep_lut_layers = n_layer;
    }
    return 1;
}

/* B2b fail-loud dump (react-go-execution-plan M-1.2), runs at process exit when
 * verify is on. Mode P masks every cold expert out of routing, so a non-zero
 * clamp tally is a correctness failure: a dropped expert was selected and the
 * translate kernel silently re-routed it to slot 0. Lists every offending
 * (layer, original-expert id) so a leak points straight at the masking gap. */
static void ds4_gpu_route_clamp_dump(void) {
    if (g_route_clamp_verify != 1 || !g_route_clamp_buf) return;
    const uint32_t *cnt = (const uint32_t *)g_route_clamp_buf.contents;
    uint64_t total = 0;
    for (uint32_t i = 0; i < g_route_clamp_layers * 256u; i++) total += cnt[i];
    if (total == 0) {
        fprintf(stderr,
                "ds4: [B2b] route-translate clamp verify: 0 cold-expert clamps "
                "across the run -- Mode P masking is tight (clamp count = 0)\n");
        return;
    }
    fprintf(stderr,
            "ds4: [B2b] *** ROUTE-TRANSLATE CLAMP LEAK *** %llu cold-expert "
            "selection(s) clamped to slot 0 -- Mode P masking LEAKED:\n",
            (unsigned long long)total);
    for (uint32_t L = 0; L < g_route_clamp_layers; L++) {
        for (uint32_t e = 0; e < 256u; e++) {
            const uint32_t c = cnt[L * 256u + e];
            if (c)
                fprintf(stderr,
                        "ds4: [B2b]   layer %02u expert %03u clamped %u time(s)\n",
                        L, e, c);
        }
    }
}

/* B2b: read DS4_VERIFY_ROUTE_CLAMP once; on first enable register the atexit
 * fail-loud dump. The tally buffer is allocated lazily by ensure_buf() below once
 * the kept-layer count is known. Default (unset) => verify off => byte-identical
 * routing (kernel skips the atomic). */
int ds4_gpu_route_clamp_verify_enabled(void) {
    if (g_route_clamp_verify < 0) {
        g_route_clamp_verify = ds4_gpu_env_bool("DS4_VERIFY_ROUTE_CLAMP") > 0 ? 1 : 0;
        if (g_route_clamp_verify == 1) {
            atexit(ds4_gpu_route_clamp_dump);
            fprintf(stderr,
                    "ds4: [B2b] route-translate clamp verify enabled "
                    "(DS4_VERIFY_ROUTE_CLAMP=1) -- Mode P clamp count must be 0\n");
        }
    }
    return g_route_clamp_verify;
}

/* B2b: lazily allocate the [n_layer*256] uint32 clamp tally, sized to the kept-
 * layer count (DeepSeek V4 = 43 layers, ~43 KiB). Allocated for any keep-map
 * model so the kernel always has a bound buffer at index 3; the atomic write is
 * gated by args.verify, so non-verify runs leave it untouched. */
id<MTLBuffer> ds4_gpu_route_clamp_ensure_buf(void) {
    if (!g_route_clamp_buf || g_route_clamp_layers != g_expert_keep_lut_layers) {
        const size_t n = (size_t)g_expert_keep_lut_layers * 256u * sizeof(uint32_t);
        g_route_clamp_buf = [g_device newBufferWithLength:n
                                                  options:MTLResourceStorageModeShared];
        if (g_route_clamp_buf) {
            memset(g_route_clamp_buf.contents, 0, n);
            g_route_clamp_layers = g_expert_keep_lut_layers;
        }
    }
    return g_route_clamp_buf;
}
