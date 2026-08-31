/* metal_expert_stream.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_gather_experts_run(
        const void *model_map,
        uint32_t    layer_index,
        uint32_t    n_active,
        const uint32_t *active_ids,
        uint8_t    *gate_dst,
        uint8_t    *up_dst,
        uint8_t    *down_dst,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (!model_map || n_expert_total == 0 || !active_ids || n_active == 0) return 0;
    if (!gate_dst || !up_dst || !down_dst) return 0;

    const uint8_t *map = (const uint8_t *)model_map;

    /* Resolve cached switches here, on the serial entry path, so the
     * lazily-initialized statics never race with gather worker threads. */
    int use_pread = ds4_gpu_expert_pread_enabled();
    int pread_fd = -1;
    if (use_pread) {
        if (g_expert_gather_nocache_call) pread_fd = ds4_gpu_expert_pread_fd_nocache();
        if (pread_fd < 0) pread_fd = ds4_gpu_expert_pread_fd();
    }
    if (use_pread && pread_fd < 0) use_pread = 0;
    const uint32_t gather_threads = ds4_gpu_expert_gather_threads();
    const uint32_t total_units = n_active * 3u;
    /* Cursor racing pays on big prefill batches (hundreds of units) and on
     * decode layers the staging could not cover (unregistered routers, layer
     * 0 without a predecessor): their gather window leaves the link idle
     * anyway, and the tail guard keeps remote off the layer's critical end.
     * Staged decode layers skip racing: 8 local threads claim the 18-unit
     * cursor instantly and a ~2ms remote round trip only adds tail. */
    /* NOTE: do NOT gate this on staging readiness.  Letting a low-readiness
     * layer fall back to cursor racing steals the fetch connections from the
     * NEXT layer's staging, which then enters ITS gather low on readiness --
     * the fallback cascades down all layers and converts the whole lookahead
     * pipeline back into in-layer racing (measured: 2.03 -> 1.80 t/s).
     * Armed == staged; the in-flight waits harvest what arrives late. */
    const int layer_is_staged =
        ds4_gpu_expert_stage_enabled() &&
        layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        g_stage[layer_index & 1u].gen != 0 &&
        g_stage[layer_index & 1u].layer == layer_index;
    /* Wave 32: units floor.  The old `|| !stage_enabled` arm made EVERY
     * gather race on the worker once accept-mode rfetch went live (worker
     * has no FETCH_HOST => staging off => arm always true).  Worker decode
     * gathers (18 units) finish locally in 2-6ms; a TB round trip to the
     * busy coordinator disk is 6ms+ of pure tail (measured: 426/1104 decode
     * gathers raced, smoke 2.13 -> 2.08 the moment accept mode connected).
     * Keep racing for batch-sized work (>= 96 units: prefill + verify) and
     * for the coordinator's unstaged-decode-layer path; small unstaged
     * gathers stay local. */
    const int remote_on =
        ds4_gpu_expert_remote_fetch_slots() > 0 && !g_rf_link_down &&
        (total_units >= 96u ||
         (ds4_gpu_expert_stage_enabled() && !layer_is_staged));
    g_gather_active = 1;   /* prefetch read-ahead yields while we own the SSD */

    ds4_metal_expert_gather_ctx ctx = {
        .model_map = model_map,
        .map = map,
        .gate_dst = gate_dst,
        .up_dst = up_dst,
        .down_dst = down_dst,
        .active_ids = active_ids,
        .n_active = n_active,
        .n_expert_total = n_expert_total,
        .layer_index = layer_index,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .next_slot = 0,
        .done = 0,
        .use_pread = use_pread,
        .pread_fd = pread_fd,
        .ok = 1,
    };

    /* Frequency-pinned cache: mlock this layer's hot experts resident (once) so the
     * gather below reads them from RAM, not cold SSD.  Routing unchanged (bit-exact). */
    ds4_gpu_expert_pin_mlock_layer(layer_index, model_map, gate_offset, up_offset,
                                   down_offset, gate_expert_bytes, down_expert_bytes);

    /* Publish the shared cursor so the remote fetch workers can pull units
     * from the peer's SSD in parallel with the local pread threads. */
    if (remote_on) {
        pthread_mutex_lock(&g_rf_mu);
        g_rf_ctx = &ctx;
        pthread_cond_broadcast(&g_rf_cv);
        pthread_mutex_unlock(&g_rf_mu);
    }

    int pool_ok = 0;
    uint32_t nth = gather_threads;
    if (nth > total_units) nth = total_units;
    if (nth > 1u) pool_ok = ds4_gpu_expert_gather_pool_run(&ctx, nth);
    if (!pool_ok) {
        /* Thread pool unavailable/failed: the calling thread drains the same
         * shared cursor inline (remote workers may still help). */
        (void)ds4_gpu_expert_gather_temp_worker(&ctx);
    }

    if (remote_on) {
        /* Local workers are done; remote workers may still have claimed units
         * in flight.  Wait for completion, then retire the ctx (the workers
         * only compare the pointer, never dereference it after retirement). */
        while (ctx.done < total_units) usleep(100);
        pthread_mutex_lock(&g_rf_mu);
        g_rf_ctx = NULL;
        pthread_cond_broadcast(&g_rf_cv);
        pthread_mutex_unlock(&g_rf_mu);
    }
    g_gather_active = 0;
    return ctx.ok;
}

/* Gather the full active set into the shared MoE scratch at slots [0, n_active).
 * Thin wrapper preserving the original signature/behavior (ensure scratch, then
 * gather into g_moe_scratch_*). */
int ds4_gpu_load_layer_experts_to_scratch(
        const void *model_map,
        uint32_t    layer_index,
        uint32_t    n_active,
        const uint32_t *active_ids,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (!ds4_gpu_ensure_moe_scratch(n_active, gate_expert_bytes, down_expert_bytes)) return 0;
    return ds4_gpu_gather_experts_run(model_map, layer_index, n_active, active_ids,
                                      (uint8_t *)g_moe_scratch_gate.contents,
                                      (uint8_t *)g_moe_scratch_up.contents,
                                      (uint8_t *)g_moe_scratch_down.contents,
                                      gate_offset, up_offset, down_offset,
                                      gate_expert_bytes, down_expert_bytes, n_expert_total);
}

/* Pass 1 of the selected-expert compaction: collect the unique active expert
 * ids of this layer call (first-appearance order) without touching the
 * selected-id buffer, so the caller can pick gather vs full-layer streaming
 * before committing to a slot remap. */
int ds4_gpu_collect_active_experts(
        id<MTLBuffer> selectedbuf,
        NSUInteger    selected_off,
        uint32_t      n_picks,
        uint32_t      n_expert_total,
        uint32_t     *active_ids,
        uint32_t      active_cap,
        uint32_t     *n_active_out) {
    if (!selectedbuf || !active_ids || !n_active_out || active_cap == 0) return 0;
    if (selectedbuf.storageMode != MTLStorageModeShared) {
        fprintf(stderr,
                "ds4: A3 expert offload requires Shared-storage selected buffer (got mode %lu)\n",
                (unsigned long)selectedbuf.storageMode);
        return 0;
    }
    if (n_expert_total == 0 || n_expert_total > active_cap ||
        active_cap > DS4_METAL_ACTIVE_EXPERTS_MAX) return 0;
    const int32_t *sel_cpu =
        (const int32_t *)((const uint8_t *)selectedbuf.contents + (size_t)selected_off);
    int16_t seen_lut[DS4_METAL_ACTIVE_EXPERTS_MAX];
    for (uint32_t i = 0; i < active_cap; i++) seen_lut[i] = -1;
    uint32_t n_active = 0;
    for (uint32_t i = 0; i < n_picks; i++) {
        int32_t raw = sel_cpu[i];
        uint32_t id = (raw >= 0 && (uint32_t)raw < n_expert_total) ? (uint32_t)raw : 0u;
        if (seen_lut[id] < 0) {
            if (n_active >= active_cap) return 0;
            seen_lut[id] = 0;
            active_ids[n_active++] = id;
        }
    }
    *n_active_out = n_active;
    return n_active != 0;
}
