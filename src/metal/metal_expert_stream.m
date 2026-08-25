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

    /* Resolve cached env switches here, on the serial entry path, so the
     * lazily-initialized statics never race with gather worker threads. */
    const int io_profile = ds4_gpu_expert_io_profile_enabled();
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
        .io_profile = io_profile,
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
    /* Record this layer's gathered (=now page-cache-hot) expert set so the NEXT
     * forward's router can bias its top-k toward them (cache-aware routing). No-op
     * when lambda=0. */
    ds4_gpu_router_cache_note_gather(layer_index, active_ids, n_active);
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

/* ---- project.md P1.2: prefill full-layer sequential streaming ----
 *
 * A prefill chunk of >=128 tokens activates nearly all 256 experts of every
 * routed layer, so the per-expert gather degenerates into ~256 scattered
 * multi-MiB reads.  When the active set covers at least
 * DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT (default 60) percent of the layer,
 * DS4_METAL_EXPERT_FULL_LAYER_STREAM=1 streams the whole gate/up/down expert
 * tensors into scratch as three long sequential reads (chunked across the
 * gather threads) and keeps the original expert ids — no slot remap, the MoE
 * kernels index scratch exactly like the resident direct path. */

int ds4_gpu_expert_stream_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_FULL_LAYER_STREAM") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: prefill full-layer expert streaming enabled "
                    "(threshold %llu%%, chunk %lluMiB)\n",
                    (unsigned long long)ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT", 60u),
                    (unsigned long long)ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_CHUNK_MB", 16u));
        }
    }
    return cached;
}

uint32_t ds4_gpu_expert_stream_threshold_pct(void) {
    static uint32_t cached;
    static int initialized;
    if (!initialized) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT", 60u);
        if (v < 1u) v = 1u;
        if (v > 100u) v = 100u;
        cached = (uint32_t)v;
        initialized = 1;
    }
    return cached;
}

/* Wave 31: full-layer streaming reads the whole 1728MiB tensor set from the
 * LOCAL disk only -- the stream path has no remote-fetch racing.  Measured on
 * the 49-token verify batch: mini L0-2 streamed at 1.9GB/s = ~950ms/layer
 * while the neighboring raced gathers did ~620MiB in ~130ms at 5-6.8GB/s
 * aggregated over both SSDs.  Whenever racing is live, dense activation is
 * exactly when racing pays the most (>=345 units), so skip streaming and let
 * the raced gather take it.  DS4_METAL_EXPERT_STREAM_RFETCH_BYPASS=0 restores
 * the old always-stream behavior. */
int ds4_gpu_expert_stream_rfetch_bypass(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_METAL_EXPERT_STREAM_RFETCH_BYPASS");
        cached = (env && env[0] == '0' && env[1] == '\0') ? 0 : 1;
    }
    if (!cached) return 0;
    return ds4_gpu_expert_remote_fetch_slots() > 0 && !g_rf_link_down;
}

static uint64_t ds4_gpu_expert_stream_chunk_bytes(void) {
    static uint64_t cached;
    if (cached == 0) {
        uint64_t mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_CHUNK_MB", 16u);
        if (mb < 1u) mb = 1u;
        if (mb > 256u) mb = 256u;
        cached = mb << 20;
    }
    return cached;
}

static void ds4_gpu_expert_stream_copy_chunk(ds4_metal_expert_stream_ctx *ctx, uint64_t chunk) {
    uint32_t seg = 0;
    uint64_t base = 0;
    while (seg < 3u && chunk >= base + ctx->seg_chunks[seg]) {
        base += ctx->seg_chunks[seg];
        seg++;
    }
    if (seg >= 3u) {
        ctx->ok = 0;
        return;
    }
    const uint64_t off = (chunk - base) * ctx->chunk_bytes;
    uint64_t len = ctx->seg_len[seg] - off;
    if (len > ctx->chunk_bytes) len = ctx->chunk_bytes;
    const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
    if (ctx->use_pread &&
        ds4_gpu_pread_full(ctx->pread_fd,
                           ctx->seg_dst[seg] + off,
                           ctx->seg_src[seg] + off,
                           (size_t)len)) {
        if (ctx->io_profile) {
            __sync_fetch_and_add(&g_io_prof_pread_ns,
                                 (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
            __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
        }
        return;
    }
    if (ctx->use_pread) __sync_fetch_and_add(&g_io_prof_pread_fallbacks, 1u);
    memcpy(ctx->seg_dst[seg] + off, ctx->map + ctx->seg_src[seg] + off, (size_t)len);
    if (ctx->io_profile) {
        __sync_fetch_and_add(&g_io_prof_copy_ns,
                             (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
        __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
    }
}

static void *ds4_gpu_expert_stream_worker(void *arg) {
    ds4_metal_expert_stream_ctx *ctx = (ds4_metal_expert_stream_ctx *)arg;
    for (;;) {
        const uint64_t chunk = __sync_fetch_and_add(&ctx->next_chunk, 1ull);
        if (chunk >= ctx->total_chunks) break;
        ds4_gpu_expert_stream_copy_chunk(ctx, chunk);
        if (!ctx->ok) break;
    }
    return NULL;
}

int ds4_gpu_stream_layer_experts_to_scratch(
        const void *model_map,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (!model_map || n_expert_total == 0) return 0;
    const uint64_t gate_total = (uint64_t)n_expert_total * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)n_expert_total * down_expert_bytes;
    if (gate_total == 0 || down_total == 0 ||
        gate_total > NSUIntegerMax || down_total > NSUIntegerMax) {
        return 0;
    }
    /* Same scratch pool as the per-expert gather: with ~all experts active the
     * gather already sizes scratch to the full layer, so streaming does not
     * raise the peak footprint. */
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_gate,
                                       &g_moe_scratch_gate_bytes,
                                       (NSUInteger)gate_total,
                                       "ds4_moe_scratch_gate") ||
        !ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_up,
                                       &g_moe_scratch_up_bytes,
                                       (NSUInteger)gate_total,
                                       "ds4_moe_scratch_up") ||
        !ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_down,
                                       &g_moe_scratch_down_bytes,
                                       (NSUInteger)down_total,
                                       "ds4_moe_scratch_down")) {
        return 0;
    }
    uint8_t *gate_dst = (uint8_t *)g_moe_scratch_gate.contents;
    uint8_t *up_dst = (uint8_t *)g_moe_scratch_up.contents;
    uint8_t *down_dst = (uint8_t *)g_moe_scratch_down.contents;
    if (!gate_dst || !up_dst || !down_dst) return 0;

    const int io_profile = ds4_gpu_expert_io_profile_enabled();
    int use_pread = ds4_gpu_expert_pread_enabled();
    const int pread_fd = use_pread ? ds4_gpu_expert_pread_fd() : -1;
    if (use_pread && pread_fd < 0) use_pread = 0;

    ds4_metal_expert_stream_ctx ctx = {
        .map = (const uint8_t *)model_map,
        .use_pread = use_pread,
        .pread_fd = pread_fd,
        .io_profile = io_profile,
        .seg_src = { gate_offset, up_offset, down_offset },
        .seg_dst = { gate_dst, up_dst, down_dst },
        .seg_len = { gate_total, gate_total, down_total },
        .chunk_bytes = ds4_gpu_expert_stream_chunk_bytes(),
        .next_chunk = 0,
        .ok = 1,
    };
    for (uint32_t s = 0; s < 3u; s++) {
        ctx.seg_chunks[s] = (ctx.seg_len[s] + ctx.chunk_bytes - 1u) / ctx.chunk_bytes;
        ctx.total_chunks += ctx.seg_chunks[s];
    }

    uint32_t nth = ds4_gpu_expert_gather_threads();
    if ((uint64_t)nth > ctx.total_chunks) nth = (uint32_t)ctx.total_chunks;
    pthread_t th[16];
    uint32_t created = 0;
    g_gather_active = 1;   /* prefetch read-ahead yields while we own the SSD */
    if (nth > 1u) {
        for (uint32_t i = 0; i + 1u < nth && i < 16u; i++) {
            if (pthread_create(&th[created], NULL, ds4_gpu_expert_stream_worker, &ctx) == 0) {
                created++;
            }
        }
    }
    /* The calling thread always participates; the shared cursor guarantees
     * every chunk is copied exactly once even if thread creation failed. */
    (void)ds4_gpu_expert_stream_worker(&ctx);
    for (uint32_t i = 0; i < created; i++) (void)pthread_join(th[i], NULL);
    g_gather_active = 0;
    return ctx.ok;
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
    if (n_expert_total == 0 || n_expert_total > active_cap || active_cap > 1024u) return 0;
    const int32_t *sel_cpu =
        (const int32_t *)((const uint8_t *)selectedbuf.contents + (size_t)selected_off);
    int16_t seen_lut[1024];
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

int ds4_gpu_cmp_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a;
    const uint32_t y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int ds4_gpu_expert_sort_ids_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_SORT_IDS") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: expert gather slots sorted by expert id (sequential cold reads)\n");
        }
    }
    return cached;
}
