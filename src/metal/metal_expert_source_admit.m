/* metal_expert_source_admit.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

static void ds4_gpu_expert_source_admit(
        const void *model_map,
        uint32_t layer,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_expert_source_entries || expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) return;

    void *gate_base = NULL, *up_base = NULL, *down_base = NULL;
    size_t gate_len = 0, up_len = 0, down_len = 0;
    ds4_gpu_expert_source_align_range(model_map,
                                      gate_offset + (uint64_t)expert * gate_expert_bytes,
                                      gate_expert_bytes,
                                      &gate_base,
                                      &gate_len);
    ds4_gpu_expert_source_align_range(model_map,
                                      up_offset + (uint64_t)expert * gate_expert_bytes,
                                      gate_expert_bytes,
                                      &up_base,
                                      &up_len);
    ds4_gpu_expert_source_align_range(model_map,
                                      down_offset + (uint64_t)expert * down_expert_bytes,
                                      down_expert_bytes,
                                      &down_base,
                                      &down_len);
    const uint64_t charge = g_expert_source_hard_copy ?
        (gate_expert_bytes + gate_expert_bytes + down_expert_bytes) :
        ((uint64_t)gate_len + (uint64_t)up_len + (uint64_t)down_len);
    const uint64_t budget = ds4_gpu_expert_source_effective_budget();
    if (charge == 0 || charge > budget) return;

    while (g_expert_source_tail >= 0 && g_expert_source_used_bytes + charge > budget) {
        ds4_gpu_expert_source_evict_tail();
    }
    if (g_expert_source_used_bytes + charge > budget) return;

    const int32_t idx = ds4_gpu_expert_source_free_entry();
    if (idx < 0) return;
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];

    const double t0 = ds4_gpu_now_ms();
    (void)madvise(gate_base, gate_len, MADV_WILLNEED);
    (void)madvise(up_base, up_len, MADV_WILLNEED);
    (void)madvise(down_base, down_len, MADV_WILLNEED);
    if (g_expert_source_hard_copy) {
        uint8_t *gcopy = (uint8_t *)malloc((size_t)gate_expert_bytes);
        uint8_t *ucopy = (uint8_t *)malloc((size_t)gate_expert_bytes);
        uint8_t *dcopy = (uint8_t *)malloc((size_t)down_expert_bytes);
        if (!gcopy || !ucopy || !dcopy) {
            free(gcopy); free(ucopy); free(dcopy);
            return;
        }
        const uint8_t *map = (const uint8_t *)model_map;
        memcpy(gcopy, map + gate_offset + (uint64_t)expert * gate_expert_bytes, (size_t)gate_expert_bytes);
        memcpy(ucopy, map + up_offset + (uint64_t)expert * gate_expert_bytes, (size_t)gate_expert_bytes);
        memcpy(dcopy, map + down_offset + (uint64_t)expert * down_expert_bytes, (size_t)down_expert_bytes);
        e->hard_copy = true;
        e->gate_copy = gcopy;
        e->up_copy = ucopy;
        e->down_copy = dcopy;
        e->gate_bytes = (size_t)gate_expert_bytes;
        e->up_bytes = (size_t)gate_expert_bytes;
        e->down_bytes = (size_t)down_expert_bytes;
    }
    bool gl = false, ul = false, dl = false;
    (void)ds4_gpu_expert_source_try_mlock(gate_base, gate_len, &gl);
    (void)ds4_gpu_expert_source_try_mlock(up_base, up_len, &ul);
    (void)ds4_gpu_expert_source_try_mlock(down_base, down_len, &dl);
    g_expert_source_admit_ms += ds4_gpu_now_ms() - t0;

    e->used = true;
    e->model_map = model_map;
    e->layer = layer;
    e->expert = expert;
    e->charged_bytes = charge;
    e->gate_base = gate_base;
    e->up_base = up_base;
    e->down_base = down_base;
    e->gate_len = gate_len;
    e->up_len = up_len;
    e->down_len = down_len;
    e->gate_locked = gl;
    e->up_locked = ul;
    e->down_locked = dl;
    ds4_gpu_expert_source_lru_link_head(idx);
    if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        g_expert_source_index[layer][expert] = idx;
    }
    g_expert_source_used_bytes += charge;
    if (g_expert_source_used_bytes > g_expert_source_peak_bytes) g_expert_source_peak_bytes = g_expert_source_used_bytes;
    g_expert_source_admissions++;
    if (gl || ul || dl) g_expert_source_locked_entries++;
    else g_expert_source_soft_entries++;
}

void ds4_gpu_expert_source_cache_note(
        const void *model_map,
        uint32_t layer,
        uint32_t n_active,
        const uint32_t *active_ids,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!ds4_gpu_expert_source_is_enabled() || !ds4_gpu_expert_source_layer_allowed(layer) ||
        !active_ids || n_active == 0 || layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
        return;
    }
    if (!ds4_gpu_expert_source_init()) return;

    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t expert = active_ids[i];
        if (expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) continue;
        g_expert_source_requests++;
        int32_t idx = ds4_gpu_expert_source_find(model_map, layer, expert);
        if (idx >= 0) {
            g_expert_source_hits++;
            ds4_gpu_expert_source_lru_touch(idx);
            continue;
        }
        g_expert_source_misses++;
        uint32_t seen = ++g_expert_source_seen[layer][expert];
        if (seen >= g_expert_source_admit_after) {
            if (g_expert_source_async_prefetch && !g_expert_source_hard_copy) {
                ds4_gpu_expert_source_prefetch_enqueue(model_map,
                                                       expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes);
            } else {
                ds4_gpu_expert_source_admit(model_map,
                                            layer,
                                            expert,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
            }
        }
    }
    if (g_expert_source_interval != 0 &&
        (g_expert_source_requests % g_expert_source_interval) == 0) {
        ds4_gpu_expert_source_print("live");
    }
}

const ds4_metal_expert_source_entry *ds4_gpu_expert_source_hard_find_ex(
        const void *model_map,
        uint32_t layer,
        uint32_t expert,
        bool touch_lru) {
    if (!g_expert_source_hard_copy || !g_expert_source_entries) return NULL;
    int32_t idx = ds4_gpu_expert_source_find(model_map, layer, expert);
    if (idx < 0) return NULL;
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    if (!e->hard_copy || !e->gate_copy || !e->up_copy || !e->down_copy) return NULL;
    if (touch_lru) ds4_gpu_expert_source_lru_touch(idx);
    return e;
}

static const ds4_metal_expert_source_entry *ds4_gpu_expert_source_hard_find(
        const void *model_map,
        uint32_t layer,
        uint32_t expert) {
    return ds4_gpu_expert_source_hard_find_ex(model_map, layer, expert, true);
}

uint32_t ds4_gpu_expert_gather_threads(void) {
    static int initialized;
    static uint32_t threads;
    if (!initialized) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_GATHER_THREADS", 1u);
        if (v < 1u) v = 1u;
        if (v > 16u) v = 16u;
        threads = (uint32_t)v;
        initialized = 1;
        if (threads > 1u && ds4_gpu_expert_offload_enabled() && !ds4_gpu_expert_offload_direct_enabled()) {
            fprintf(stderr,
                    "ds4: A3 expert CPU gather parallel copy enabled: %u threads\n",
                    threads);
        }
    }
    return threads;
}

/* ---- project.md P0.1/P1.1: expert IO instrumentation + single-copy pread ----
 *
 * Measured bottleneck (execution log 2026-06-10): the A3 gather moves cold
 * expert bytes at ~1.5GB/s because every byte pays a 16KiB mmap page fault
 * (SSD -> page cache) plus a memcpy (page cache -> Shared scratch), serialized
 * behind a per-layer command drain.  DS4_METAL_EXPERT_PREAD=1 replaces the
 * fault+memcpy double copy with one pread() per expert tensor straight into
 * the scratch MTLBuffer.  DS4_METAL_EXPERT_PREAD_NOCACHE=1 additionally reads
 * through a separate F_NOCACHE descriptor so the ~1.7GiB/token cold stream
 * stops evicting hot page-cache pages (A/B knob, off by default).
 * DS4_METAL_EXPERT_IO_PROFILE=1 prints one ds4-io line per routed-MoE layer
 * call decomposing wall time into fault/memcpy/pread/drain so the account in
 * project.md P0.1 can be settled from a normal speed run. */

int ds4_gpu_pread_full(int fd, void *dst, uint64_t src_off, size_t len) {
    uint8_t *p = (uint8_t *)dst;
    while (len > 0) {
        ssize_t r = pread(fd, p, len, (off_t)src_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += (size_t)r;
        src_off += (uint64_t)r;
        len -= (size_t)r;
    }
    return 1;
}

/* Resolve once from the serial gather entry (before worker threads spawn) so
 * the cached env lookups never race. */
int ds4_gpu_expert_pread_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        {
            /* auto-adapt (no baked script envs): streaming/offload models get
             * the single-copy pread win by default — token byte-exact, ~2.2x
             * measured; resident models don't need it. Env still overrides. */
            int e = ds4_gpu_env_bool("DS4_METAL_EXPERT_PREAD");
            cached = (e < 0) ? (ds4_gpu_expert_offload_enabled() ? 1 : 0) : (e > 0 ? 1 : 0);
        }
        if (cached && (g_model_fd < 0 || g_model_fd_conflict)) {
            fprintf(stderr,
                    "ds4: DS4_METAL_EXPERT_PREAD=1 requested but no unambiguous model fd; "
                    "falling back to mmap gather\n");
            cached = 0;
        }
        if (cached) {
            fprintf(stderr, "ds4: expert gather single-copy pread enabled (fd=%d)\n",
                    g_model_fd);
        }
    }
    return cached;
}

/* Wave 24: verify/prefill batch gathers read hundreds of MiB of cold expert
 * bytes per layer with ~zero reuse (batch hit_mib==0 in every profile run),
 * yet a kc=12 verify round streams ~5.8GiB through the coordinator page
 * cache and evicts the mmap-resident backbone (Q8 attention/shared) pages.
 * The next single-token frame then re-faults the backbone inside the GPU
 * command execution: decode drain_ms ballooned 14.5 -> 46ms/layer (spikes to
 * 855ms on the first layers after a verify), inflating round-1 from ~550ms to
 * ~2.3s.  Route batch-site cold preads through a separate F_NOCACHE
 * descriptor so one-shot batch bytes stop evicting hot pages.  Default on;
 * DS4_METAL_EXPERT_BATCH_NOCACHE=0 restores the shared cached fd (A/B). */
int g_expert_gather_nocache_call;

/* set on the serial encode path per gather call */

int ds4_gpu_expert_batch_nocache_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_METAL_EXPERT_BATCH_NOCACHE");
        cached = (v && *v && v[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* Wave 25 A/B verdict: NOCACHE won on first-touch prefill chunks (smoke
 * 2.02 -> 2.09) but lost on kc<=16 verify rounds (code-edit 1.68 -> 1.48):
 * verify unions overlap decode-hot experts and neighbouring rounds, so
 * bypassing the cache re-reads warm bytes from the slow mini SSD.  Keep
 * NOCACHE for big (prefill-sized) batches only; backbone protection against
 * verify eviction moves to the mlock pin (see ds4_gpu_backbone_mlock_register). */
uint32_t ds4_gpu_expert_batch_nocache_min_tokens(void) {
    static uint32_t cached;
    static int init;
    if (!init) {
        const char *v = getenv("DS4_METAL_EXPERT_BATCH_NOCACHE_MIN");
        uint64_t n = v ? strtoull(v, NULL, 10) : 0;
        /* Wave 35: 24 was calibrated when K=16 capped verify at 17 rows --
         * verify could never trip it.  The K=64 ladder now sends kc=25/33
         * verify batches (the two big rounds, 11.8s of 21.3s r2 total) which
         * 24 misclassifies as prefill-sized cold streams: their unions lose
         * both the warm bytes of the neighbouring round (consecutive copy
         * rounds share most experts) and the decode-hot overlap -- exactly
         * the case the wave-25 verdict measured as a loss (1.68 vs 1.48).
         * 64 keeps every verify batch (protocol max 64 rows) on the cached
         * fd; prefill frames (128 tokens) stay NOCACHE. */
        if (n == 0) n = 64;
        if (n > UINT32_MAX) n = UINT32_MAX;
        cached = (uint32_t)n;
        init = 1;
    }
    return cached;
}
