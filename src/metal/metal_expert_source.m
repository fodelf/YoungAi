/* metal_expert_source.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

#define DS4_METAL_EXPERT_SOURCE_MAX_ENTRIES 65536u

static int g_expert_source_enabled = -1;

static bool g_expert_source_init_attempted;

static bool g_expert_source_summary_registered;

static bool g_expert_source_mlock_disabled;

static bool g_expert_source_mlock_reported;

static uint64_t g_expert_source_budget_bytes;

/* configured MB cap (upper bound) */
static int      g_expert_source_dynamic;

/* size against live headroom, not the fixed cap */
static uint64_t g_expert_source_dyn_margin_bytes;

/* keep this far below the 90% self-kill line */
uint64_t g_expert_source_used_bytes;

uint64_t g_expert_source_peak_bytes;

static uint32_t g_expert_source_layer_start;

static uint32_t g_expert_source_layer_end;

uint32_t g_expert_source_admit_after;

uint32_t g_expert_source_interval;

static int g_expert_source_use_mlock;

int g_expert_source_hard_copy;

ds4_metal_expert_source_entry *g_expert_source_entries;

static uint32_t g_expert_source_cap;

static int32_t g_expert_source_head = -1;

int32_t g_expert_source_tail = -1;

int32_t g_expert_source_index[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

uint32_t g_expert_source_seen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

uint64_t g_expert_source_requests;

uint64_t g_expert_source_hits;

uint64_t g_expert_source_misses;

uint64_t g_expert_source_admissions;

static uint64_t g_expert_source_evictions;

uint64_t g_expert_source_soft_entries;

uint64_t g_expert_source_locked_entries;

static uint64_t g_expert_source_lock_failures;

double   g_expert_source_admit_ms;

int      g_expert_source_async_prefetch;

static pthread_mutex_t g_expert_source_pf_mu = PTHREAD_MUTEX_INITIALIZER;

static pthread_cond_t  g_expert_source_pf_cv = PTHREAD_COND_INITIALIZER;

static pthread_t g_expert_source_pf_thread;

static int g_expert_source_pf_thread_started;

static int g_expert_source_pf_shutdown;

static ds4_metal_expert_source_prefetch_req *g_expert_source_pf_q;

static uint32_t g_expert_source_pf_qcap;

static uint32_t g_expert_source_pf_qhead;

static uint32_t g_expert_source_pf_qtail;

static uint32_t g_expert_source_pf_qcount;

static uint64_t g_expert_source_pf_enqueued;

static uint64_t g_expert_source_pf_dropped;

static uint64_t g_expert_source_pf_done;

static double   g_expert_source_pf_ms;

int ds4_gpu_expert_source_is_enabled(void) {
    if (g_expert_source_enabled < 0) {
        const uint64_t mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_MB", 0);
        g_expert_source_enabled = mb != 0 ? 1 : 0;
        if (g_expert_source_enabled) {
            g_expert_source_budget_bytes = mb * 1024ull * 1024ull;
            g_expert_source_layer_start = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_START", 3u);
            g_expert_source_layer_end = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_END", 127u);
            if (g_expert_source_layer_end < g_expert_source_layer_start) {
                g_expert_source_layer_end = g_expert_source_layer_start;
            }
            g_expert_source_admit_after = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_ADMIT_AFTER", 2u);
            if (g_expert_source_admit_after == 0) g_expert_source_admit_after = 1;
            g_expert_source_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_INTERVAL", 64u);
            g_expert_source_use_mlock = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_MLOCK") > 0;
            g_expert_source_hard_copy = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_HARD_COPY") > 0;
            g_expert_source_async_prefetch = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_ASYNC") > 0;
            /* Dynamic sizing: grow the cache into the live phys_footprint
             * headroom (up to the configured MB as a ceiling) instead of always
             * holding the fixed MB.  Keeps the cache useful when there is slack
             * under the 12G budget and yields it back as the working set grows. */
            g_expert_source_dynamic = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_DYNAMIC") > 0;
            g_expert_source_dyn_margin_bytes =
                ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_DYN_MARGIN_MB", 768u) * 1024ull * 1024ull;
        }
    }
    return g_expert_source_enabled;
}

/* Effective byte budget for the source cache.  In dynamic mode it is the live
 * headroom below the internal 90%-of-budget self-kill line (minus a margin),
 * excluding the cache's own bytes, capped by the configured MB.  The live
 * phys_footprint is sampled at most every 50ms so the per-expert admit path
 * does not call task_info() hundreds of times per layer. */
uint64_t ds4_gpu_expert_source_effective_budget(void) {
    if (!g_expert_source_dynamic) return g_expert_source_budget_bytes;
    static uint64_t cached_footprint;
    static double   cached_ms;
    const double now = ds4_gpu_now_ms();
    if (cached_footprint == 0 || now - cached_ms > 50.0) {
        cached_footprint = ds4_runtime_phys_footprint_bytes();
        cached_ms = now;
    }
    const uint64_t mem_budget = ds4_runtime_mem_budget_bytes();
    if (mem_budget == 0 || cached_footprint == 0) {
        return g_expert_source_budget_bytes;   /* no live info: fall back to fixed cap */
    }
    /* Internal watchdog self-kills at 90% of the budget; stay a margin below. */
    uint64_t ceiling = mem_budget / 10ull * 9ull;
    if (ceiling <= g_expert_source_dyn_margin_bytes) return 0;
    ceiling -= g_expert_source_dyn_margin_bytes;
    const uint64_t used = g_expert_source_used_bytes;
    const uint64_t other = cached_footprint > used ? cached_footprint - used : 0;  /* footprint minus this cache */
    if (other >= ceiling) return 0;            /* no headroom: evict everything */
    uint64_t dyn = ceiling - other;
    if (dyn > g_expert_source_budget_bytes) dyn = g_expert_source_budget_bytes;  /* MB is the hard ceiling */
    return dyn;
}

int ds4_gpu_expert_source_layer_allowed(uint32_t layer) {
    return layer >= g_expert_source_layer_start && layer <= g_expert_source_layer_end;
}

void ds4_gpu_expert_source_print(const char *tag) {
    if (g_expert_source_requests == 0) return;
    const double hit_pct = 100.0 * (double)g_expert_source_hits / (double)g_expert_source_requests;
    fprintf(stderr,
            "ds4: expert-source-cache %s: requests=%llu hit=%.2f%% hits=%llu misses=%llu "
            "admit=%llu evict=%llu used=%.2f/%.2f MiB peak=%.2f MiB "
            "layers=%u:%u admit_after=%u mlock=%s hard_copy=%s async=%s locked=%llu soft=%llu lock_fail=%llu admit=%.3f ms pf=%llu/%llu drop=%llu pf_ms=%.3f\n",
            tag ? tag : "live",
            (unsigned long long)g_expert_source_requests,
            hit_pct,
            (unsigned long long)g_expert_source_hits,
            (unsigned long long)g_expert_source_misses,
            (unsigned long long)g_expert_source_admissions,
            (unsigned long long)g_expert_source_evictions,
            (double)g_expert_source_used_bytes / (1024.0 * 1024.0),
            (double)g_expert_source_budget_bytes / (1024.0 * 1024.0),
            (double)g_expert_source_peak_bytes / (1024.0 * 1024.0),
            g_expert_source_layer_start,
            g_expert_source_layer_end,
            g_expert_source_admit_after,
            (g_expert_source_use_mlock && !g_expert_source_mlock_disabled) ? "on" : "off",
            g_expert_source_hard_copy ? "on" : "off",
            g_expert_source_async_prefetch ? "on" : "off",
            (unsigned long long)g_expert_source_locked_entries,
            (unsigned long long)g_expert_source_soft_entries,
            (unsigned long long)g_expert_source_lock_failures,
            g_expert_source_admit_ms,
            (unsigned long long)g_expert_source_pf_done,
            (unsigned long long)g_expert_source_pf_enqueued,
            (unsigned long long)g_expert_source_pf_dropped,
            g_expert_source_pf_ms);
}

static void ds4_gpu_expert_source_summary(void) {
    ds4_gpu_expert_source_print("summary");
}

static size_t ds4_gpu_page_size(void) {
    static size_t page;
    if (page == 0) {
        long p = sysconf(_SC_PAGESIZE);
        page = p > 0 ? (size_t)p : (size_t)4096;
    }
    return page;
}

void ds4_gpu_expert_source_align_range(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        void **base_out,
        size_t *len_out) {
    const size_t page = ds4_gpu_page_size();
    const uintptr_t addr = (uintptr_t)model_map + (uintptr_t)offset;
    const uintptr_t start = addr & ~((uintptr_t)page - 1u);
    const uintptr_t end = (addr + (uintptr_t)bytes + (uintptr_t)page - 1u) & ~((uintptr_t)page - 1u);
    *base_out = (void *)start;
    *len_out = (size_t)(end - start);
}

static void ds4_gpu_expert_source_lru_unlink(int32_t idx) {
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    if (e->prev >= 0) g_expert_source_entries[e->prev].next = e->next;
    if (e->next >= 0) g_expert_source_entries[e->next].prev = e->prev;
    if (g_expert_source_head == idx) g_expert_source_head = e->next;
    if (g_expert_source_tail == idx) g_expert_source_tail = e->prev;
    e->prev = -1;
    e->next = -1;
}

void ds4_gpu_expert_source_lru_link_head(int32_t idx) {
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    e->prev = -1;
    e->next = g_expert_source_head;
    if (g_expert_source_head >= 0) g_expert_source_entries[g_expert_source_head].prev = idx;
    g_expert_source_head = idx;
    if (g_expert_source_tail < 0) g_expert_source_tail = idx;
}

void ds4_gpu_expert_source_lru_touch(int32_t idx) {
    if (idx == g_expert_source_head) return;
    ds4_gpu_expert_source_lru_unlink(idx);
    ds4_gpu_expert_source_lru_link_head(idx);
}

static void ds4_gpu_expert_source_unlock_entry(ds4_metal_expert_source_entry *e) {
    if (!e) return;
    if (e->gate_locked) (void)munlock(e->gate_base, e->gate_len);
    if (e->up_locked) (void)munlock(e->up_base, e->up_len);
    if (e->down_locked) (void)munlock(e->down_base, e->down_len);
    free(e->gate_copy);
    free(e->up_copy);
    free(e->down_copy);
    e->gate_copy = e->up_copy = e->down_copy = NULL;
    e->gate_locked = e->up_locked = e->down_locked = false;
}

void ds4_gpu_expert_source_evict_tail(void) {
    const int32_t idx = g_expert_source_tail;
    if (idx < 0) return;
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    ds4_gpu_expert_source_lru_unlink(idx);
    if (e->layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        e->expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS &&
        g_expert_source_index[e->layer][e->expert] == idx) {
        g_expert_source_index[e->layer][e->expert] = -1;
    }
    ds4_gpu_expert_source_unlock_entry(e);
    if (g_expert_source_used_bytes >= e->charged_bytes) {
        g_expert_source_used_bytes -= e->charged_bytes;
    } else {
        g_expert_source_used_bytes = 0;
    }
    memset(e, 0, sizeof(*e));
    e->prev = -1;
    e->next = -1;
    g_expert_source_evictions++;
}

int32_t ds4_gpu_expert_source_find(const void *model_map, uint32_t layer, uint32_t expert) {
    if (!g_expert_source_entries) return -1;
    if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        const int32_t idx = g_expert_source_index[layer][expert];
        if (idx >= 0 && (uint32_t)idx < g_expert_source_cap) {
            ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
            if (e->used && e->model_map == model_map && e->layer == layer && e->expert == expert) {
                return idx;
            }
        }
    }
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        ds4_metal_expert_source_entry *e = &g_expert_source_entries[i];
        if (e->used && e->model_map == model_map && e->layer == layer && e->expert == expert) {
            if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_source_index[layer][expert] = (int32_t)i;
            }
            return (int32_t)i;
        }
    }
    return -1;
}

int32_t ds4_gpu_expert_source_free_entry(void) {
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        if (!g_expert_source_entries[i].used) return (int32_t)i;
    }
    ds4_gpu_expert_source_evict_tail();
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        if (!g_expert_source_entries[i].used) return (int32_t)i;
    }
    return -1;
}

int ds4_gpu_expert_source_init(void) {
    if (!ds4_gpu_expert_source_is_enabled()) return 0;
    if (g_expert_source_entries) return 1;
    if (g_expert_source_init_attempted) return 0;
    g_expert_source_init_attempted = true;

    g_expert_source_cap = DS4_METAL_EXPERT_SOURCE_MAX_ENTRIES;
    g_expert_source_entries = calloc(g_expert_source_cap, sizeof(g_expert_source_entries[0]));
    if (!g_expert_source_entries) return 0;
    for (uint32_t l = 0; l < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; l++) {
        for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
            g_expert_source_index[l][e] = -1;
        }
    }
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        g_expert_source_entries[i].prev = -1;
        g_expert_source_entries[i].next = -1;
    }
    if (!g_expert_source_summary_registered) {
        atexit(ds4_gpu_expert_source_summary);
        g_expert_source_summary_registered = true;
    }
    fprintf(stderr,
            "ds4: expert-source-cache enabled: %.2f MiB %s LRU, layers %u:%u, "
            "admit_after=%u, mlock=%s. GPU still reads compact A3 scratch; cache keeps expert source hot.\n",
            (double)g_expert_source_budget_bytes / (1024.0 * 1024.0),
            g_expert_source_hard_copy ? "hard-copy" : "source-page",
            g_expert_source_layer_start,
            g_expert_source_layer_end,
            g_expert_source_admit_after,
            g_expert_source_use_mlock ? "on" : "off");
    return 1;
}

int ds4_gpu_expert_source_try_mlock(void *base, size_t len, bool *locked) {
    *locked = false;
    if (!g_expert_source_use_mlock || g_expert_source_mlock_disabled) return 1;
    if (mlock(base, len) == 0) {
        *locked = true;
        return 1;
    }
    g_expert_source_lock_failures++;
    if (!g_expert_source_mlock_reported) {
        fprintf(stderr,
                "ds4: expert-source-cache mlock failed (%s); continuing with soft madvise/page-cache LRU\n",
                strerror(errno));
        g_expert_source_mlock_reported = true;
    }
    /* macOS often has a small mlock rlimit. Disable further attempts after the
     * first failure so hot-loop admission does not repeatedly pay syscall costs. */
    g_expert_source_mlock_disabled = true;
    return 0;
}

static void *ds4_gpu_expert_source_prefetch_main(void *arg) {
    (void)arg;
    for (;;) {
        ds4_metal_expert_source_prefetch_req req = { 0 };
        pthread_mutex_lock(&g_expert_source_pf_mu);
        while (!g_expert_source_pf_shutdown && g_expert_source_pf_qcount == 0) {
            pthread_cond_wait(&g_expert_source_pf_cv, &g_expert_source_pf_mu);
        }
        if (g_expert_source_pf_shutdown && g_expert_source_pf_qcount == 0) {
            pthread_mutex_unlock(&g_expert_source_pf_mu);
            break;
        }
        req = g_expert_source_pf_q[g_expert_source_pf_qhead];
        g_expert_source_pf_qhead = (g_expert_source_pf_qhead + 1u) % g_expert_source_pf_qcap;
        g_expert_source_pf_qcount--;
        pthread_mutex_unlock(&g_expert_source_pf_mu);

        void *gate_base = NULL, *up_base = NULL, *down_base = NULL;
        size_t gate_len = 0, up_len = 0, down_len = 0;
        ds4_gpu_expert_source_align_range(req.model_map,
                                          req.gate_offset + (uint64_t)req.expert * req.gate_expert_bytes,
                                          req.gate_expert_bytes,
                                          &gate_base,
                                          &gate_len);
        ds4_gpu_expert_source_align_range(req.model_map,
                                          req.up_offset + (uint64_t)req.expert * req.gate_expert_bytes,
                                          req.gate_expert_bytes,
                                          &up_base,
                                          &up_len);
        ds4_gpu_expert_source_align_range(req.model_map,
                                          req.down_offset + (uint64_t)req.expert * req.down_expert_bytes,
                                          req.down_expert_bytes,
                                          &down_base,
                                          &down_len);
        const double t0 = ds4_gpu_now_ms();
        (void)madvise(gate_base, gate_len, MADV_WILLNEED);
        (void)madvise(up_base, up_len, MADV_WILLNEED);
        (void)madvise(down_base, down_len, MADV_WILLNEED);
        const double dt = ds4_gpu_now_ms() - t0;
        __sync_fetch_and_add(&g_expert_source_pf_done, 1ull);
        g_expert_source_pf_ms += dt;
    }
    return NULL;
}

static void ds4_gpu_expert_source_prefetch_summary(void) {
    pthread_mutex_lock(&g_expert_source_pf_mu);
    g_expert_source_pf_shutdown = 1;
    pthread_cond_broadcast(&g_expert_source_pf_cv);
    pthread_mutex_unlock(&g_expert_source_pf_mu);
}

static int ds4_gpu_expert_source_prefetch_init(void) {
    if (!g_expert_source_async_prefetch) return 0;
    if (g_expert_source_pf_thread_started) return 1;
    g_expert_source_pf_qcap = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_ASYNC_QUEUE", 8192u);
    if (g_expert_source_pf_qcap < 64u) g_expert_source_pf_qcap = 64u;
    g_expert_source_pf_q = calloc(g_expert_source_pf_qcap, sizeof(g_expert_source_pf_q[0]));
    if (!g_expert_source_pf_q) return 0;
    if (pthread_create(&g_expert_source_pf_thread, NULL, ds4_gpu_expert_source_prefetch_main, NULL) != 0) {
        free(g_expert_source_pf_q);
        g_expert_source_pf_q = NULL;
        return 0;
    }
    pthread_detach(g_expert_source_pf_thread);
    g_expert_source_pf_thread_started = 1;
    atexit(ds4_gpu_expert_source_prefetch_summary);
    return 1;
}

void ds4_gpu_expert_source_prefetch_enqueue(
        const void *model_map,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!model_map || !ds4_gpu_expert_source_prefetch_init()) return;
    ds4_metal_expert_source_prefetch_req req = {
        .model_map = model_map,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .expert = expert,
    };
    pthread_mutex_lock(&g_expert_source_pf_mu);
    if (g_expert_source_pf_qcount == g_expert_source_pf_qcap) {
        g_expert_source_pf_dropped++;
    } else {
        g_expert_source_pf_q[g_expert_source_pf_qtail] = req;
        g_expert_source_pf_qtail = (g_expert_source_pf_qtail + 1u) % g_expert_source_pf_qcap;
        g_expert_source_pf_qcount++;
        g_expert_source_pf_enqueued++;
        pthread_cond_signal(&g_expert_source_pf_cv);
    }
    pthread_mutex_unlock(&g_expert_source_pf_mu);
}
