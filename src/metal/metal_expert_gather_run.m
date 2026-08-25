/* metal_expert_gather_run.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

static void *ds4_gpu_expert_gather_pool_worker(void *arg) {
    ds4_metal_expert_gather_worker_arg *warg = (ds4_metal_expert_gather_worker_arg *)arg;
    ds4_metal_expert_gather_pool *pool = warg->pool;
    const uint32_t worker_index = warg->index;
    uint64_t seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mu);
        while (!pool->shutdown && pool->generation == seen_generation) {
            pthread_cond_wait(&pool->cv, &pool->mu);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mu);
            break;
        }
        seen_generation = pool->generation;
        const uint32_t participates = worker_index < pool->target_workers;
        ds4_metal_expert_gather_ctx *ctx = participates ? pool->ctx : NULL;
        pthread_mutex_unlock(&pool->mu);

        if (!participates) continue;
        for (;;) {
            const uint32_t unit = __sync_fetch_and_add(&ctx->next_slot, 1u);
            if (unit >= ctx->n_active * 3u) break;
            ds4_gpu_expert_gather_copy_unit(ctx, unit);
            __sync_fetch_and_add(&ctx->done, 1u);
        }

        pthread_mutex_lock(&pool->mu);
        if (pool->active_workers > 0) pool->active_workers--;
        if (pool->active_workers == 0) {
            pool->completed_generation = seen_generation;
            pthread_cond_signal(&pool->done_cv);
        }
        pthread_mutex_unlock(&pool->mu);
    }
    return NULL;
}

void *ds4_gpu_expert_gather_temp_worker(void *arg) {
    ds4_metal_expert_gather_ctx *ctx = (ds4_metal_expert_gather_ctx *)arg;
    for (;;) {
        const uint32_t unit = __sync_fetch_and_add(&ctx->next_slot, 1u);
        if (unit >= ctx->n_active * 3u) break;
        ds4_gpu_expert_gather_copy_unit(ctx, unit);
        __sync_fetch_and_add(&ctx->done, 1u);
    }
    return NULL;
}

int ds4_gpu_expert_gather_pool_run(ds4_metal_expert_gather_ctx *ctx, uint32_t nth) {
    if (nth == 0 || nth > 16u) return 0;
    pthread_t th[16];
    int created = 0;
    for (uint32_t i = 0; i < nth; i++) {
        if (pthread_create(&th[i], NULL, ds4_gpu_expert_gather_temp_worker, ctx) == 0) {
            created++;
        }
    }
    for (int i = 0; i < created; i++) {
        (void)pthread_join(th[i], NULL);
    }
    return created == (int)nth ? ctx->ok : 0;
}

/* Ensure the three routed-MoE expert scratch buffers are sized for n_active
 * expert slots (grow-and-keep). */
int ds4_gpu_ensure_moe_scratch(uint32_t n_active,
                                      uint64_t gate_expert_bytes,
                                      uint64_t down_expert_bytes) {
    const uint64_t gate_total = (uint64_t)n_active * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)n_active * down_expert_bytes;
    if (n_active == 0 || gate_total > NSUIntegerMax || down_total > NSUIntegerMax) return 0;
    return ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_gate,
                                         &g_moe_scratch_gate_bytes,
                                         (NSUInteger)gate_total,
                                         "ds4_moe_scratch_gate") &&
           ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_up,
                                         &g_moe_scratch_up_bytes,
                                         (NSUInteger)gate_total,
                                         "ds4_moe_scratch_up") &&
           ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_down,
                                         &g_moe_scratch_down_bytes,
                                         (NSUInteger)down_total,
                                         "ds4_moe_scratch_down");
}

/* Core expert gather: stream the n_active experts named by active_ids[] into
 * the caller-provided gate/up/down destination buffers (expert at index i lands
 * at slot i, i.e. dst + (uint64_t)i*expert_bytes).  The destination is a
 * parameter so the P-OVL two-pass path can gather a disjoint slot sub-range of
 * the shared scratch -- call with (active_ids + lo, dst + lo*expert_bytes,
 * n_active = hi - lo) and the per-unit workers (slot = unit/3) land each expert
 * at its global scratch slot without needing any range awareness themselves. */
/* ---- Frequency-pinned expert cache (quality-safe; project.md PC.2) ----------
 * mlock the file-mmap regions of a per-layer coding-domain hot-expert set so they
 * stay resident in the page cache and the routed gather's pread hits them WARM
 * (RAM copy) instead of cold (SSD).  Routing is UNCHANGED -> bit-exact output;
 * this only changes which expert bytes are already in RAM.  Distinct from the
 * rejected paths: not the LRU source cache (evicts hot under churn -> 4.4% hit),
 * not the resident pool (slow scattered GPU read, wave-51), not anon hard_copy
 * (starves page cache).  Pin set is the frequency top-K profiled over diverse
 * coding (top-K covers ~31%/55% of requests at K=32/64).  Budget-capped so the
 * wired set stays under the watchdog. */
static bool     g_expert_pin[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][256];

static int      g_expert_pin_parsed = -1;

/* -1 uninit, 0 off, 1 on */
static bool     g_expert_pin_mlocked[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint64_t g_expert_pin_mlock_used;

static uint64_t g_expert_pin_mlock_budget;

static int ds4_gpu_expert_pin_enabled(void) {
    if (g_expert_pin_parsed < 0) {
        g_expert_pin_parsed = 0;
        g_expert_pin_mlock_budget =
            ds4_gpu_env_u64("DS4_EXPERT_PIN_MLOCK_MB", 0u) * 1024ull * 1024ull;
        const char *path = getenv("DS4_EXPERT_PIN_FILE");
        if (path && *path && g_expert_pin_mlock_budget > 0) {
            FILE *f = fopen(path, "r");
            if (f) {
                /* Accepts both "20:1,2,..." per line and the gen_pinned.py emit
                 * ("L20:...;L0:..." -- optional L prefix, ';' record separator,
                 * possibly one single line). File order per layer is frequency
                 * order, so the per-layer DS4_EXPERT_PIN_TOPK cut keeps the
                 * hottest K (uniform per-layer budget instead of the old
                 * whole-file first-come truncation). */
                const uint32_t topk = (uint32_t)ds4_gpu_env_u64("DS4_EXPERT_PIN_TOPK", 256u);
                uint32_t added[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS] = {0};
                char line[8192];
                while (fgets(line, sizeof(line), f)) {
                    char *save = NULL;
                    for (char *grp = strtok_r(line, ";\n", &save); grp;
                         grp = strtok_r(NULL, ";\n", &save)) {
                        char *colon = strchr(grp, ':');
                        if (!colon) continue;
                        *colon = '\0';
                        char *ls = grp;
                        while (*ls == ' ') ls++;
                        if (*ls == 'L' || *ls == 'l') ls++;
                        const uint32_t L = (uint32_t)strtoul(ls, NULL, 10);
                        if (L >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) continue;
                        char *p = colon + 1;
                        while (*p) {
                            char *end = NULL;
                            const uint32_t e = (uint32_t)strtoul(p, &end, 10);
                            if (end == p) break;
                            if (e < 256u && added[L] < topk && !g_expert_pin[L][e]) {
                                g_expert_pin[L][e] = true;
                                added[L]++;
                                g_expert_pin_parsed = 1;
                            }
                            p = end;
                            while (*p == ',' || *p == ' ') p++;
                        }
                    }
                }
                fclose(f);
            }
        }
        if (g_expert_pin_parsed == 1)
            fprintf(stderr, "ds4: frequency-pinned expert cache enabled (file=%s budget=%llu MiB) "
                    "-- quality-safe (routing unchanged, bit-exact)\n",
                    path, (unsigned long long)(g_expert_pin_mlock_budget / (1024ull * 1024ull)));
    }
    return g_expert_pin_parsed;
}

/* Lock-free hot check for the gather workers: the pin table and the per-layer
 * mlock marker are written on the serial model-load/encode path only, so by
 * the time gather worker threads run they are read-only — no lock needed. */
int ds4_gpu_expert_pin_hot(uint32_t layer_index, uint32_t expert_id) {
    return g_expert_pin_parsed == 1 &&
           layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
           g_expert_pin_mlocked[layer_index] &&
           expert_id < 256u && g_expert_pin[layer_index][expert_id];
}

/* mlock this layer's pinned experts (gate/up/down file regions) once, budget-capped.
 * Faults the pins in on first touch (one-time warm cost) then keeps them wired so
 * subsequent gathers read them from RAM. */
void ds4_gpu_expert_pin_mlock_layer(uint32_t layer_index, const void *model_map,
                                           uint64_t gate_offset, uint64_t up_offset,
                                           uint64_t down_offset, uint64_t gate_expert_bytes,
                                           uint64_t down_expert_bytes) {
    if (ds4_gpu_expert_pin_enabled() != 1 || !model_map) return;
    if (layer_index >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS || g_expert_pin_mlocked[layer_index]) return;
    g_expert_pin_mlocked[layer_index] = true;   /* attempt once per layer regardless */
    const uint8_t *base = (const uint8_t *)model_map;
    uint64_t pinned = 0;
    for (uint32_t e = 0; e < 256u; e++) {
        if (!g_expert_pin[layer_index][e]) continue;
        if (g_expert_pin_mlock_used >= g_expert_pin_mlock_budget) break;
        const uint64_t off[3] = { gate_offset + (uint64_t)e * gate_expert_bytes,
                                  up_offset   + (uint64_t)e * gate_expert_bytes,
                                  down_offset + (uint64_t)e * down_expert_bytes };
        const uint64_t len[3] = { gate_expert_bytes, gate_expert_bytes, down_expert_bytes };
        for (int r = 0; r < 3; r++) {
            if (g_expert_pin_mlock_used + len[r] > g_expert_pin_mlock_budget) continue;
            if (mlock(base + off[r], (size_t)len[r]) == 0) {
                g_expert_pin_mlock_used += len[r];
                pinned += len[r];
            }
        }
    }
    if (pinned)
        fprintf(stderr, "ds4: layer %u pinned %.1f MiB hot experts (total wired %.2f/%.2f GiB)\n",
                layer_index, (double)pinned / (1024.0 * 1024.0),
                (double)g_expert_pin_mlock_used / (1024.0 * 1024.0 * 1024.0),
                (double)g_expert_pin_mlock_budget / (1024.0 * 1024.0 * 1024.0));
}

/* Residual-sidecar twin of the frequency pin above (DS4_RESID_PIN_MLOCK_MB, default
 * 0 = off). The go1b residual gather is a single-thread memcpy straight off the
 * sidecar mmap (metal.m:21874/22117) -- measured 18% of coordinator decode wall,
 * mostly page-fault stalls, because the 9.2 GiB sidecar competes for page cache it
 * never wins. Pin the residual slot ranges of the SAME pin-file experts (the
 * sidecar's hot-64 superset) so those memcpys run at RAM speed. Residency-only:
 * bytes and results are bit-exact. Shares DS4_EXPERT_PIN_FILE via g_expert_pin[][]
 * (so DS4_EXPERT_PIN_TOPK sizes both pins); budget-capped separately. */
static bool     g_resid_pin_mlocked[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint64_t g_resid_pin_mlock_used;

void ds4_gpu_resid_pin_mlock_layer(uint32_t layer_index,
                                          const void *res_gate_ptr,
                                          const void *res_up_ptr,
                                          const void *res_down_ptr,
                                          const float *res_lut,
                                          uint64_t gate_expert_bytes,
                                          uint64_t down_expert_bytes) {
    static int64_t budget = -1;
    if (budget < 0) budget = (int64_t)(ds4_gpu_env_u64("DS4_RESID_PIN_MLOCK_MB", 0u) * 1024ull * 1024ull);
    if (budget == 0 || ds4_gpu_expert_pin_enabled() != 1) return;
    if (!res_gate_ptr || !res_up_ptr || !res_down_ptr) return;
    if (layer_index >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS || g_resid_pin_mlocked[layer_index]) return;
    g_resid_pin_mlocked[layer_index] = true;   /* attempt once per layer regardless */
    uint64_t pinned = 0;
    for (uint32_t e = 0; e < 256u; e++) {
        if (!g_expert_pin[layer_index][e]) continue;
        const int64_t slot = res_lut ? (int64_t)res_lut[e] : (int64_t)e;
        if (slot < 0) continue;               /* sparse residual: not a hot expert */
        if (g_resid_pin_mlock_used >= (uint64_t)budget) break;
        const uint8_t *ptr[3] = { (const uint8_t *)res_gate_ptr + (uint64_t)slot * gate_expert_bytes,
                                  (const uint8_t *)res_up_ptr   + (uint64_t)slot * gate_expert_bytes,
                                  (const uint8_t *)res_down_ptr + (uint64_t)slot * down_expert_bytes };
        const uint64_t len[3] = { gate_expert_bytes, gate_expert_bytes, down_expert_bytes };
        for (int r = 0; r < 3; r++) {
            if (g_resid_pin_mlock_used + len[r] > (uint64_t)budget) continue;
            if (mlock(ptr[r], (size_t)len[r]) == 0) {
                g_resid_pin_mlock_used += len[r];
                pinned += len[r];
            }
        }
    }
    if (pinned)
        fprintf(stderr, "ds4: layer %u pinned %.1f MiB residual hot experts (resid wired %.2f GiB)\n",
                layer_index, (double)pinned / (1024.0 * 1024.0),
                (double)g_resid_pin_mlock_used / (1024.0 * 1024.0 * 1024.0));
}
