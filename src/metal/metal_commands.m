/* metal_commands.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_repeat_args;

int ds4_gpu_begin_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_cb = [g_queue commandBuffer];
    return g_batch_cb != nil;
}

int ds4_gpu_flush_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    [g_pending_cbs addObject:cb];

    g_batch_cb = [g_queue commandBuffer];
    if (!g_batch_cb) {
        (void)ds4_gpu_wait_pending_command_buffers("command batch");
        [g_transient_buffers removeAllObjects];
        return 0;
    }
    return 1;
}

int ds4_gpu_end_commands(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    return ds4_gpu_finish_command_buffer(cb, 1, "command batch");
}

/* project.md P0.1 barrier-tax recovery: the A3 expert path drains the batch
 * before every routed layer's CPU gather (the gather must read this layer's
 * router output, and may overwrite scratch the previous MoE still reads).
 * The classic drain commits + waitUntilCompleted, paying the slow per-CB
 * scheduling/status path every routed layer (~43x per decoded token).  The
 * fast drain reuses the MTLSharedEvent host-wait fast path (same precedent as
 * the TP rendezvous below: ~150ms -> <50us per sync): signal at the end of
 * the batch, commit without waiting, and block on waitUntilSignaledValue,
 * which gives the same "all prior GPU work complete, writes visible"
 * guarantee.  Command-buffer status/error checking is deferred one drain: the
 * CB is pushed on g_pending_cbs and swept by the next drain after its event
 * wait, when the CB is already complete and the sweep is free. */
static id<MTLSharedEvent> g_a3_drain_event;

static uint64_t g_a3_drain_event_value;

static int ds4_gpu_expert_event_drain_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_EVENT_DRAIN") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: A3 expert drain uses MTLSharedEvent fast host wait "
                    "(DS4_METAL_EXPERT_EVENT_DRAIN=1)\n");
        }
    }
    return cached;
}

static int ds4_gpu_end_commands_event(const char *label) {
    if (!g_batch_cb) return 0;
    if (!g_a3_drain_event) {
        g_a3_drain_event = [g_device newSharedEvent];
        if (!g_a3_drain_event) return ds4_gpu_end_commands();
    }
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    const uint64_t value = ++g_a3_drain_event_value;
    [cb encodeSignalEvent:g_a3_drain_event value:value];
    [cb commit];
    if (![g_a3_drain_event waitUntilSignaledValue:value timeoutMS:60000]) {
        fprintf(stderr,
                "ds4: Metal %s event drain timed out; falling back to waitUntilCompleted\n",
                label);
        if (!ds4_gpu_wait_command_buffer(cb, label)) return 0;
    }
    /* Everything queued before the signal is complete: sweep the deferred
     * status checks (instant now) and release transient buffer references. */
    int ok = ds4_gpu_wait_pending_command_buffers(label);
    [g_pending_cbs addObject:cb];
    [g_transient_buffers removeAllObjects];
    return ok;
}

int ds4_gpu_expert_drain_commands(const char *label) {
    if (ds4_gpu_expert_event_drain_enabled()) return ds4_gpu_end_commands_event(label);
    return ds4_gpu_end_commands();
}

/* Tensor-parallel host/GPU rendezvous. Instead of draining the whole pipeline
 * with waitUntilCompleted (slow per-CB scheduling path), we signal a
 * MTLSharedEvent at the end of the current batch and let the host wait on that
 * specific value — the MTLSharedEvent.waitUntilSignaledValue fast path
 * (Anukari/Apple precedent: ~150ms -> <50us per sync). The caller flushes the
 * batch (commit, no wait) right after signalling so the GPU runs and fires the
 * event while the host proceeds. */
static id<MTLSharedEvent> g_tp_event;

static uint64_t g_tp_event_value;

uint64_t ds4_gpu_tp_signal_after_batch(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    if (!g_tp_event) {
        g_tp_event = [g_device newSharedEvent];
        if (!g_tp_event) return 0;
    }
    uint64_t value = ++g_tp_event_value; /* values are reserved nonzero (0 == error) */
    [g_batch_cb encodeSignalEvent:g_tp_event value:value];
    return value;
}

int ds4_gpu_tp_host_wait(uint64_t value) {
    if (!g_tp_event || value == 0) return 0;
    uint64_t timeout_ms = 60000;
    const char *env = getenv("DS4_TP_EVENT_TIMEOUT_MS");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && v > 0) timeout_ms = (uint64_t)v;
    }
    return [g_tp_event waitUntilSignaledValue:value timeoutMS:timeout_ms] ? 1 : 0;
}

int ds4_gpu_flash_attn_stage_profile_boundary(
        id<MTLCommandBuffer> __strong *cbp,
        const char           *mode,
        const char           *stage,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              n_keys,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              window,
        uint32_t              ratio,
        double               *stage_t0) {
    if (!cbp || !*cbp || !stage_t0 || !stage) return 0;
    if (ds4_gpu_end_commands() == 0) return 0;

    const double now_ms = ds4_gpu_now_ms();
    const char *filter = getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER");
    const int print_stage =
        !filter || !filter[0] ||
        strstr(stage, filter) != NULL ||
        (mode && strstr(mode, filter) != NULL);
    if (print_stage) {
        fprintf(stderr,
                "ds4: Metal FlashAttention prefill stage mode=%s tokens=%u comp=%u "
                "keys=%u heads=%u dim=%u window=%u ratio=%u %s=%.3f ms\n",
                mode ? mode : "unknown",
                n_tokens,
                n_comp,
                n_keys,
                n_head,
                head_dim,
                window,
                ratio,
                stage,
                now_ms - *stage_t0);
    }
    *stage_t0 = now_ms;

    if (ds4_gpu_begin_commands() == 0) return 0;
    int owned = 0;
    *cbp = ds4_gpu_command_buffer(&owned);
    return *cbp != nil && owned == 0;
}

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return ds4_gpu_end_commands();
    if ([g_pending_cbs count] != 0) {
        int ok = ds4_gpu_wait_pending_command_buffers("synchronize");
        [g_transient_buffers removeAllObjects];
        return ok;
    }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;
    return ds4_gpu_finish_command_buffer(cb, 1, "synchronize");
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        if (g_batch_cb) {
            ds4_gpu_close_batch_encoder();
            [g_batch_cb commit];
            [g_batch_cb waitUntilCompleted];
            g_batch_cb = nil;
        }
        (void)ds4_gpu_wait_pending_command_buffers("cleanup");
        [g_transient_buffers removeAllObjects];
        g_tp_event = nil;
        g_tp_event_value = 0;
        g_a3_drain_event = nil;
        g_a3_drain_event_value = 0;
        g_set_rows_f32_i32_pipeline = nil;
        g_get_rows_f32_pipeline = nil;
        g_get_rows_f16_pipeline = nil;
        g_get_rows_i32_pipeline = nil;
        g_repeat_f32_pipeline = nil;
        g_concat_pipeline = nil;
        g_cpy_f32_f32_pipeline = nil;
        g_cpy_f32_f16_pipeline = nil;
        g_cpy_f16_f32_pipeline = nil;
        g_swiglu_pipeline = nil;
        g_add_pipeline = nil;
        g_moe_sum6_pipeline = nil;
        g_mul_pipeline = nil;
        g_bin_mul_scalar_pipeline = nil;
        g_bin_div_row_pipeline = nil;
        g_unary_sigmoid_pipeline = nil;
        g_unary_silu_pipeline = nil;
        g_unary_softplus_pipeline = nil;
        g_unary_sqrt_pipeline = nil;
        g_unary_clamp_pipeline = nil;
        g_unary_scale_pipeline = nil;
        g_unary_fill_pipeline = nil;
        g_unary_fill_f16_pipeline = nil;
        g_rms_norm_pipeline = nil;
        g_rms_norm_plain_pipeline = nil;
        g_dsv4_qkv_rms_norm_pipeline = nil;
        g_hc_split_sinkhorn_pipeline = nil;
        g_hc_split_weighted_sum_pipeline = nil;
        g_hc_split_weighted_sum_norm_pipeline = nil;
        g_hc_weighted_sum_pipeline = nil;
        g_hc_expand_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q2_k_pipeline = nil;
        g_moe_mul_mv_id_q2_k_sum6_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q4_k_sum6_pipeline = nil;
        g_rope_tail_batch_pipeline = nil;
        g_dsv4_fp8_kv_quantize_pipeline = nil;
        g_dsv4_indexer_qat_pipeline = nil;
        g_dsv4_kv_fp8_store_pipeline = nil;
        g_dsv4_ratio4_shift_pipeline = nil;
        g_dsv4_softmax_pool_pipeline = nil;
        g_soft_max_f32_pipeline = nil;
        g_soft_max_f32_4_pipeline = nil;
        g_argsort_f32_i32_desc_pipeline = nil;
        g_argsort_merge_f32_i32_desc_pipeline = nil;
        g_sum_rows_f32_f32_pipeline = nil;
        g_dsv4_topk_mask_pipeline = nil;
        g_dsv4_topk_mask_scatter_pipeline = nil;
        g_dsv4_indexer_weighted_sum_pipeline = nil;
        g_dsv4_indexer_score_one_direct_pipeline = nil;
        g_dsv4_compressor_store_one_pipeline = nil;
        g_dsv4_sort_i32_rows_asc_pipeline = nil;
        g_dsv4_indexed_attention_heads8_pipeline = nil;
        g_dsv4_indexed_attention_heads8_rb16_pipeline = nil;
        g_dsv4_softplus_sqrt_pipeline = nil;
        g_dsv4_router_finalize_one_pipeline = nil;
        g_dsv4_router_weights_one_pipeline = nil;
        g_dsv4_route_translate_pipeline = nil;
        g_dsv4_hc_expand4_pipeline = nil;
        g_flash_attn_mask_buffer = nil;
        g_flash_attn_pad_buffer = nil;
        g_flash_attn_tmp_buffer = nil;
        g_flash_attn_blk_buffer = nil;
        g_flash_attn_ring_buffer = nil;
        g_flash_attn_kv_buffer = nil;
        g_compressor_pool_kv_buffer = nil;
        g_compressor_pool_score_buffer = nil;
        g_compressor_pool_score_cont_buffer = nil;
        g_compressor_pool_softmax_buffer = nil;
        g_compressor_pool_product_buffer = nil;
        g_compressor_store_ape_buffer = nil;
        g_compressor_store_score_buffer = nil;
        g_embed_rows_buffer = nil;
        g_router_selection_buffer = nil;
        g_router_weight_sum_buffer = nil;
        g_expert_keep_lut_buffer = nil;
        g_expert_keep_lut_layers = 0;
        g_indexer_head_scores_buffer = nil;
        g_indexer_topk_buffer = nil;
        g_indexed_topk_buffer = nil;
        g_f16_round_scratch_buffer = nil;
        g_raw_store_round_buffer = nil;
        g_moe_gate_scratch_buffer = nil;
        g_moe_down_scratch_buffer = nil;
        g_moe_id_map_buffer = nil;
        g_attn_out_group_ids_buffer = nil;
        g_moe_scratch_gate = nil;
        g_moe_scratch_up = nil;
        g_moe_scratch_down = nil;
        g_model_map_ptr = NULL;
        g_model_map_size = 0;
        g_model_mapped_offset = 0;
        g_model_mapped_size = 0;
        g_model_mapped_max_tensor_bytes = 0;
        g_tensor_alloc_live_bytes = 0;
        g_tensor_alloc_peak_bytes = 0;
        g_flash_attn_mask_bytes = 0;
        g_flash_attn_pad_bytes = 0;
        g_flash_attn_tmp_bytes = 0;
        g_flash_attn_blk_bytes = 0;
        g_flash_attn_ring_bytes = 0;
        g_flash_attn_kv_bytes = 0;
        g_compressor_pool_kv_bytes = 0;
        g_compressor_pool_score_bytes = 0;
        g_compressor_pool_score_cont_bytes = 0;
        g_compressor_pool_softmax_bytes = 0;
        g_compressor_pool_product_bytes = 0;
        g_compressor_store_ape_bytes = 0;
        g_compressor_store_score_bytes = 0;
        g_embed_rows_bytes = 0;
        g_router_selection_bytes = 0;
        g_router_weight_sum_bytes = 0;
        g_indexer_head_scores_bytes = 0;
        g_indexer_topk_bytes = 0;
        g_indexed_topk_bytes = 0;
        g_f16_round_scratch_bytes = 0;
        g_raw_store_round_bytes = 0;
        g_moe_gate_scratch_bytes = 0;
        g_moe_down_scratch_bytes = 0;
        g_moe_id_map_bytes = 0;
        g_attn_out_group_ids_bytes = 0;
        g_moe_scratch_gate_bytes = 0;
        g_moe_scratch_up_bytes = 0;
        g_moe_scratch_down_bytes = 0;
        g_model_wrap_count = 0;
        g_model_wrap_bytes = 0;
        g_model_wrap_max_bytes = 0;
        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();
        [g_pipeline_cache removeAllObjects];
        g_pipeline_cache = nil;
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache = nil;
        g_transient_buffers = nil;
        g_pending_cbs = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

int ds4_gpu_encode_get_rows_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !tokens || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int32_t)n_embd,
        .ne00 = (int32_t)n_embd,
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)n_vocab * src_row_bytes,
        .nb03 = (uint64_t)n_vocab * src_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = dst_row_bytes,
        .nb2 = (uint64_t)n_tokens * dst_row_bytes,
        .nb3 = (uint64_t)n_tokens * dst_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_repeat_hc_embedding(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        rows,
        NSUInteger           rows_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_tokens,
        uint32_t             n_embd,
        uint32_t             n_hc) {
    if (!cb || !rows || !out || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;

    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    ds4_gpu_repeat_args args = {
        .ne00 = (int32_t)n_embd,
        .ne01 = 1,
        .ne02 = (int32_t)n_tokens,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = embd_bytes,
        .nb02 = embd_bytes,
        .nb03 = (uint64_t)n_tokens * embd_bytes,
        .ne0 = (int32_t)n_embd,
        .ne1 = (int32_t)n_hc,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = embd_bytes,
        .nb2 = (uint64_t)n_hc * embd_bytes,
        .nb3 = (uint64_t)n_tokens * n_hc * embd_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_repeat_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_repeat_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:rows offset:rows_offset atIndex:1];
    [enc setBuffer:out offset:out_offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(n_hc, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
