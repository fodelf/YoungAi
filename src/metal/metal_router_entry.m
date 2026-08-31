/* metal_router_entry.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* Rewrite a routed-expert selection tensor from original ids (0..255) to the
 * compact slots of a shrunken model's expert tensors, in place. Runs between
 * router selection and the routed-MoE matvec. No-op (returns 1) when no keep-map
 * LUT has been set, so a full model is unaffected. The kernel maps both top-k and
 * first-3-layer hash selections (both live in the same `selected` tensor) and
 * clamps dropped/out-of-range experts to slot 0 so the matvec never indexes out
 * of bounds. */
int ds4_gpu_translate_expert_ids(
        ds4_gpu_tensor       *selected,
        uint32_t                layer,
        uint32_t                n_expert_used,
        uint32_t                n_tokens,
        uint32_t                n_total_expert) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_expert_keep_lut_buffer) return 1;   /* full model: nothing to translate */
    if (!selected || n_expert_used == 0 || n_tokens == 0 || n_total_expert == 0) return 0;
    if (layer >= g_expert_keep_lut_layers) {
        fprintf(stderr, "ds4: expert id translation layer %u out of LUT range %u\n",
                layer, g_expert_keep_lut_layers);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t need = (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
        if (!selbuf || ds4_gpu_tensor_bytes(selected) < need) {
            fprintf(stderr, "ds4: Metal expert id translation received an undersized selection buffer\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_route_translate_pipeline,
                                    "kernel_dsv4_route_translate");
        if (!pipeline) return 0;

        id<MTLBuffer> clampbuf = ds4_gpu_route_clamp_ensure_buf();
        if (!clampbuf) {
            fprintf(stderr, "ds4: route-translate clamp tally alloc failed\n");
            return 0;   /* kernel requires a bound buffer at index 3 */
        }
        struct {
            uint32_t layer;
            uint32_t n_expert_used;
            uint32_t n_tokens;
            uint32_t n_total_expert;
            uint32_t verify;
        } args = { layer, n_expert_used, n_tokens, n_total_expert, 0u };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:g_expert_keep_lut_buffer offset:0 atIndex:1];
        [enc setBuffer:selbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:clampbuf offset:0 atIndex:3];
        const NSUInteger total = (NSUInteger)n_tokens * n_expert_used;
        NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > total) tg = total;
        if (tg == 0) tg = 1;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "expert id translation")) return 0;
    }
    return 1;
}

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        uint32_t                layer) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert_used == 0) return 0;
    if (hash_mode && token >= hash_rows) return 0;
    /* Decode-token note for the exact hash-layer staging (project.md P2.1). */
    ds4_gpu_expert_router_note((int)token, hash_mode ? 1 : 0);
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal router select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
            if (g_expert_keep_lut_buffer && model_map && n_expert <= 256u &&
                bias_offset + bias_bytes <= model_size) {
                /* Reduced-expert (shrunken) model: route_translate maps a dropped expert
                 * (keep-lut == -1) to "safe slot 0", so any token routed to dropped experts
                 * collapses its top-6 into duplicate slot-0 -> "number soup".  Mask the
                 * dropped here (before top-k), on a tiny per-call Shared copy of the
                 * model bias, so the router only ever selects kept experts.
                 * k16/shrunken fix: index the keep-LUT by the REAL layer (il), NOT a
                 * first-seen ordinal -- an inconsistent mask vs gather is "number soup". */
                const float *model_bias = (const float *)((const char *)model_map + bias_offset);
                id<MTLBuffer> combined = [g_device newBufferWithLength:bias_bytes
                                                               options:MTLResourceStorageModeShared];
                if (!combined) return 0;
                float *cbp = (float *)combined.contents;
                const int16_t *keeplut = (layer < g_expert_keep_lut_layers)
                    ? (const int16_t *)g_expert_keep_lut_buffer.contents + (uint64_t)layer * 256u : NULL;
                for (uint32_t i = 0; i < n_expert; i++) {
                    cbp[i] = model_bias[i];
                    if (keeplut && keeplut[i] < 0) cbp[i] = -1e30f;   /* shrunken model: dropped -> never selected */
                }
                biasbuf = combined;
                bias_set_offset = 0;
            } else {
                biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
                if (!biasbuf) return 0;
                bias_set_offset = (NSUInteger)bias_inner;
            }
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        const int32_t token_i32 = (int32_t)token;
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      nil,
                                                      0,
                                                      &token_i32,
                                                      hash_rows,
                                                      1,
                                                      n_expert,
                                                      n_expert_used,
                                                      expert_weight_scale,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens,
        uint32_t                layer) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map ||
        n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        id<MTLBuffer> tokensbuf = ds4_gpu_tensor_buffer(tokens);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf || !tokensbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_tokens * n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * n_expert_used * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_tokens * n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(tokens) < (uint64_t)n_tokens * sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal router batch select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
            if (g_expert_keep_lut_buffer && model_map && n_expert <= 256u &&
                bias_offset + bias_bytes <= model_size) {
                /* Reduced-expert (shrunken) model: route_translate maps a dropped expert
                 * (keep-lut == -1) to "safe slot 0", so any token routed to dropped experts
                 * collapses its top-6 into duplicate slot-0 -> "number soup".  Mask the
                 * dropped here (before top-k), on a tiny per-call Shared copy of the
                 * model bias, so the router only ever selects kept experts.
                 * k16/shrunken fix: index the keep-LUT by the REAL layer (il), NOT a
                 * first-seen ordinal -- an inconsistent mask vs gather is "number soup". */
                const float *model_bias = (const float *)((const char *)model_map + bias_offset);
                id<MTLBuffer> combined = [g_device newBufferWithLength:bias_bytes
                                                               options:MTLResourceStorageModeShared];
                if (!combined) return 0;
                float *cbp = (float *)combined.contents;
                const int16_t *keeplut = (layer < g_expert_keep_lut_layers)
                    ? (const int16_t *)g_expert_keep_lut_buffer.contents + (uint64_t)layer * 256u : NULL;
                for (uint32_t i = 0; i < n_expert; i++) {
                    cbp[i] = model_bias[i];
                    if (keeplut && keeplut[i] < 0) cbp[i] = -1e30f;   /* shrunken model: dropped -> never selected */
                }
                biasbuf = combined;
                bias_set_offset = 0;
            } else {
                biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
                if (!biasbuf) return 0;
                bias_set_offset = (NSUInteger)bias_inner;
            }
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      tokensbuf,
                                                      ds4_gpu_tensor_offset(tokens),
                                                      NULL,
                                                      hash_rows,
                                                      n_tokens,
                                                      n_expert,
                                                      n_expert_used,
                                                      expert_weight_scale,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

/* Dynamic routed-expert residency route. The host computes a footprint-vs-budget
 * verdict at load time and pushes it via ds4_gpu_set_expert_offload(): a model
 * that fits RAM runs resident (direct GPU read, no per-layer CPU gather); only
 * an over-budget model streams through the A3 CPU gather into small resident
 * scratch buffers. */
int g_expert_offload_verdict = -1;
