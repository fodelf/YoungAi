/* metal_expert_pool.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

static int g_expert_pool_enabled = -1;

bool g_expert_pool_init_attempted;

bool g_expert_pool_summary_registered;

id<MTLBuffer> g_expert_pool_gate;

id<MTLBuffer> g_expert_pool_up;

id<MTLBuffer> g_expert_pool_down;

uint64_t g_expert_pool_budget_bytes;

uint64_t g_expert_pool_gate_expert_bytes;

uint64_t g_expert_pool_down_expert_bytes;

uint64_t g_expert_pool_slot_bytes;

uint32_t g_expert_pool_slots;

ds4_metal_expert_pool_entry *g_expert_pool_entries;

ds4_metal_expert_pool_meta g_expert_pool_meta[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

int32_t g_expert_pool_head = -1;

int32_t g_expert_pool_tail = -1;

uint64_t g_expert_pool_calls;

uint64_t g_expert_pool_requests;

uint64_t g_expert_pool_hits;

uint64_t g_expert_pool_misses;

uint64_t g_expert_pool_inflight_hits;

uint64_t g_expert_pool_prefetches;

uint64_t g_expert_pool_prefetch_hits;

uint64_t g_expert_pool_prefetch_misses;

uint64_t g_expert_pool_prefetch_drops;

uint64_t g_expert_pool_prefetch_copied;

uint64_t g_expert_pool_sync_waits;

uint64_t g_expert_pool_fallbacks;

uint64_t g_expert_pool_miss_copy_bytes;

double   g_expert_pool_miss_copy_ms;

double   g_expert_pool_wait_ms;

double   g_expert_pool_prefetch_copy_ms;

uint32_t g_expert_pool_interval;

uint32_t g_expert_pool_layer_start;

uint32_t g_expert_pool_layer_end;

uint32_t g_expert_pool_lookahead;

uint32_t g_expert_pool_prefetch_top;

int      g_expert_pool_prefetch_self;

int      g_expert_pool_prefetch_adjacent;

int      g_expert_pool_wait_inflight;

int      g_expert_pool_foreground_fill;

int      g_expert_pool_warm_batch;

int      g_expert_pool_prefetch_evict;

int      g_expert_pool_hit_only;

uint32_t g_expert_pool_admit_after;

int      g_expert_pool_hotlock_enabled;

uint32_t g_expert_pool_hotlock_top;

uint64_t g_expert_pool_admission_skips;

uint64_t g_expert_pool_clock;

bool     g_expert_pool_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

static bool     g_expert_pool_static_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

static bool     g_expert_pool_dynamic_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

uint32_t g_expert_pool_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint32_t g_expert_pool_static_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint32_t g_expert_pool_dynamic_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

uint32_t g_expert_pool_pinned_total;

uint32_t g_expert_pool_static_pinned_total;

uint32_t g_expert_pool_dynamic_pinned_total;

uint64_t g_expert_pool_pinned_requests;

uint64_t g_expert_pool_pinned_hits;

uint64_t g_expert_pool_pinned_misses;

uint64_t g_expert_pool_pinned_prefetches;

uint64_t g_expert_pool_auto_pin_updates;

static bool     g_expert_pool_pinned_env_parsed;

bool     g_expert_pool_pinned_layer_queued[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

uint32_t g_expert_pool_auto_pin_top;

uint32_t g_expert_pool_auto_pin_min_req;

uint32_t g_expert_pool_auto_pin_interval;

uint32_t g_expert_pool_pin_reserve;

uint32_t g_expert_pool_hotlist_top;

uint32_t g_expert_pool_hotlist_interval;

uint64_t g_expert_pool_auto_pin_last_req[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

uint32_t g_expert_pool_last_active_n[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

uint32_t g_expert_pool_last_active[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX];

uint64_t g_expert_pool_hot_count[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

pthread_mutex_t g_expert_pool_mu = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t  g_expert_pool_cv = PTHREAD_COND_INITIALIZER;

bool g_expert_pool_prefetch_thread_started;

bool g_expert_pool_prefetch_shutdown;

bool g_expert_pool_cycle_guard_reported;

ds4_metal_expert_prefetch_req *g_expert_pool_prefetch_queue;

uint32_t g_expert_pool_prefetch_qcap;

uint32_t g_expert_pool_prefetch_qhead;

uint32_t g_expert_pool_prefetch_qtail;

uint32_t g_expert_pool_prefetch_qcount;

uint32_t ds4_gpu_expert_pool_min_layer_slots(void) {
    static int initialized;
    static uint32_t min_slots;
    if (!initialized) {
        /* A layer needs at least the six routed experts used by one token, but
         * in practice routing jitters across tokens.  Twelve slots/layer gives
         * real temporal LRU room while still fitting several layers in the
         * worker budget.  Override with 6 for maximum layer coverage. */
        min_slots = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_MIN_LAYER_SLOTS", 12u);
        if (min_slots < 6u) min_slots = 6u;
        if (min_slots > 4096u) min_slots = 4096u;
        initialized = 1;
    }
    return min_slots;
}

uint32_t ds4_gpu_expert_pool_requested_layers(void) {
    return g_expert_pool_layer_end >= g_expert_pool_layer_start ?
        (g_expert_pool_layer_end - g_expert_pool_layer_start + 1u) : 1u;
}

uint32_t ds4_gpu_expert_pool_served_layers(void) {
    const uint32_t requested = ds4_gpu_expert_pool_requested_layers();
    const uint32_t min_slots = ds4_gpu_expert_pool_min_layer_slots();
    if (g_expert_pool_slots < min_slots) return 0;
    uint32_t served = g_expert_pool_slots / min_slots;
    if (served > requested) served = requested;
    return served;
}

int ds4_gpu_expert_pool_layer_served(uint32_t layer, uint32_t *cap_out) {
    const uint32_t served = ds4_gpu_expert_pool_served_layers();
    if (served == 0) return 0;
    const uint32_t requested = ds4_gpu_expert_pool_requested_layers();
    (void)requested;
    /* Cache the tail of the configured layer range by default.  This is the
     * output-side worker bottleneck in the q2 two-host topology. */
    const uint32_t served_end = g_expert_pool_layer_end;
    const uint32_t served_start = served_end + 1u >= served ?
        (served_end + 1u - served) : g_expert_pool_layer_start;
    if (layer < served_start || layer > served_end) return 0;
    const uint32_t idx = layer - served_start;
    const uint32_t base = g_expert_pool_slots / served;
    const uint32_t rem = g_expert_pool_slots % served;
    uint32_t cap = base + (idx < rem ? 1u : 0u);
    if (cap < 6u) return 0;
    if (cap_out) *cap_out = cap;
    return 1;
}

int ds4_gpu_expert_pool_layer_allowed(uint32_t layer) {
    return layer >= g_expert_pool_layer_start && layer <= g_expert_pool_layer_end;
}

static int ds4_gpu_expert_pool_pin_sep(char c) {
    return c == ';' || c == '|' || c == '/' || c == '\n';
}

int ds4_gpu_expert_pool_add_pin(uint32_t layer, uint32_t expert, bool dynamic) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS ||
        expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        return 0;
    }
    int added = 0;
    if (!g_expert_pool_pinned[layer][expert]) {
        g_expert_pool_pinned[layer][expert] = true;
        g_expert_pool_pinned_per_layer[layer]++;
        g_expert_pool_pinned_total++;
        g_expert_pool_pinned_layer_queued[layer] = false;
        added = 1;
    }
    if (g_expert_pool_entries) {
        for (uint32_t slot = 0; slot < g_expert_pool_slots; slot++) {
            ds4_metal_expert_pool_entry *entry = &g_expert_pool_entries[slot];
            if (entry->used && entry->layer == layer && entry->expert == expert) {
                entry->pinned = true;
            }
        }
    }
    if (dynamic) {
        if (!g_expert_pool_static_pinned[layer][expert] &&
            !g_expert_pool_dynamic_pinned[layer][expert]) {
            g_expert_pool_dynamic_pinned[layer][expert] = true;
            g_expert_pool_dynamic_pinned_per_layer[layer]++;
            g_expert_pool_dynamic_pinned_total++;
        }
    } else if (!g_expert_pool_static_pinned[layer][expert]) {
        if (g_expert_pool_dynamic_pinned[layer][expert]) {
            g_expert_pool_dynamic_pinned[layer][expert] = false;
            if (g_expert_pool_dynamic_pinned_per_layer[layer] > 0) {
                g_expert_pool_dynamic_pinned_per_layer[layer]--;
            }
            if (g_expert_pool_dynamic_pinned_total > 0) g_expert_pool_dynamic_pinned_total--;
        }
        g_expert_pool_static_pinned[layer][expert] = true;
        g_expert_pool_static_pinned_per_layer[layer]++;
        g_expert_pool_static_pinned_total++;
    }
    return added;
}

static void ds4_gpu_expert_pool_parse_pinned_env(void) {
    if (g_expert_pool_pinned_env_parsed) return;
    g_expert_pool_pinned_env_parsed = true;
    const char *spec = getenv("DS4_METAL_EXPERT_POOL_PINNED");
    if (!spec || !spec[0]) return;

    const char *p = spec;
    while (*p) {
        while (*p && (isspace((unsigned char)*p) || ds4_gpu_expert_pool_pin_sep(*p))) p++;
        if (!*p) break;
        if (*p == 'L' || *p == 'l') p++;
        char *endp = NULL;
        unsigned long layer_ul = strtoul(p, &endp, 10);
        if (endp == p) {
            while (*p && !ds4_gpu_expert_pool_pin_sep(*p)) p++;
            continue;
        }
        p = endp;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p != ':' && *p != '=') {
            while (*p && !ds4_gpu_expert_pool_pin_sep(*p)) p++;
            continue;
        }
        p++;

        while (*p) {
            while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
            if (ds4_gpu_expert_pool_pin_sep(*p) || !*p) break;
            if (*p == 'E' || *p == 'e') p++;
            endp = NULL;
            unsigned long expert_ul = strtoul(p, &endp, 10);
            if (endp == p) {
                while (*p && *p != ',' && !ds4_gpu_expert_pool_pin_sep(*p)) p++;
                continue;
            }
            p = endp;
            if (layer_ul < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                expert_ul < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                (void)ds4_gpu_expert_pool_add_pin((uint32_t)layer_ul,
                                                  (uint32_t)expert_ul,
                                                  false);
            }
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p == ',') {
                p++;
                continue;
            }
            if (ds4_gpu_expert_pool_pin_sep(*p) || !*p) break;
            p++;
        }
    }

    if (g_expert_pool_pinned_total != 0) {
        fprintf(stderr,
                "ds4: expert-pool pinned whitelist loaded: %u experts from DS4_METAL_EXPERT_POOL_PINNED\n",
                g_expert_pool_pinned_total);
    }
}

int ds4_gpu_expert_pool_is_pinned(uint32_t layer, uint32_t expert) {
    return layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
           expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS &&
           g_expert_pool_pinned[layer][expert];
}

static int ds4_gpu_expert_pool_entry_ready(uint32_t layer, uint32_t expert, int *pinned_out) {
    bool ready = false;
    bool pinned = false;
    if (g_expert_pool_entries) {
        for (uint32_t slot = 0; slot < g_expert_pool_slots; slot++) {
            ds4_metal_expert_pool_entry *entry = &g_expert_pool_entries[slot];
            if (entry->used && entry->layer == layer && entry->expert == expert) {
                if (entry->ready) ready = true;
                if (entry->pinned) pinned = true;
            }
        }
    }
    if (pinned_out) *pinned_out = pinned ? 1 : 0;
    return ready ? 1 : 0;
}

void ds4_gpu_expert_pool_print_hotlist(const char *tag, uint32_t top, bool include_pinned_empty) {
    if (top == 0) return;
    if (top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;

    for (uint32_t layer = 0; layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; layer++) {
        uint64_t layer_req = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            layer_req += g_expert_pool_hot_count[layer][expert];
        }
        if (layer_req == 0 && (!include_pinned_empty || g_expert_pool_pinned_per_layer[layer] == 0)) continue;

        uint32_t layer_cap = 0;
        const int served = ds4_gpu_expert_pool_layer_served(layer, &layer_cap);
        fprintf(stderr,
                "ds4: expert-pool layer %02u: tag=%s requests=%llu pinned=%u static=%u dynamic=%u served=%d cap=%u\n",
                layer,
                tag ? tag : "live",
                (unsigned long long)layer_req,
                g_expert_pool_pinned_per_layer[layer],
                g_expert_pool_static_pinned_per_layer[layer],
                g_expert_pool_dynamic_pinned_per_layer[layer],
                served,
                layer_cap);

        bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
        uint32_t printed_count = 0;
        while (printed_count < top) {
            int best = -1;
            uint64_t best_req = 0;
            for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
                if (printed[expert]) continue;
                const uint64_t req = g_expert_pool_hot_count[layer][expert];
                if (req > best_req) {
                    best_req = req;
                    best = (int)expert;
                }
            }
            if (best < 0 || best_req == 0) break;
            printed[(uint32_t)best] = true;
            printed_count++;
            int entry_pinned = 0;
            const int resident = ds4_gpu_expert_pool_entry_ready(layer, (uint32_t)best, &entry_pinned);
            fprintf(stderr,
                    "ds4: expert-pool   L%02u E%03u req=%llu pinned=%d static=%d dynamic=%d resident=%d entry_pin=%d\n",
                    layer,
                    (uint32_t)best,
                    (unsigned long long)best_req,
                    ds4_gpu_expert_pool_is_pinned(layer, (uint32_t)best),
                    g_expert_pool_static_pinned[layer][best] ? 1 : 0,
                    g_expert_pool_dynamic_pinned[layer][best] ? 1 : 0,
                    resident,
                    entry_pinned);
        }

        if (!include_pinned_empty) continue;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS && printed_count < top; expert++) {
            if (!g_expert_pool_pinned[layer][expert] || printed[expert]) continue;
            printed_count++;
            int entry_pinned = 0;
            const int resident = ds4_gpu_expert_pool_entry_ready(layer, expert, &entry_pinned);
            fprintf(stderr,
                    "ds4: expert-pool   L%02u E%03u req=0 pinned=1 static=%d dynamic=%d resident=%d entry_pin=%d\n",
                    layer,
                    expert,
                    g_expert_pool_static_pinned[layer][expert] ? 1 : 0,
                    g_expert_pool_dynamic_pinned[layer][expert] ? 1 : 0,
                    resident,
                    entry_pinned);
        }
    }
}

int ds4_gpu_expert_pool_is_enabled(void) {
    if (g_expert_pool_enabled < 0) {
        const uint64_t mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_MB", 0);
        g_expert_pool_enabled = mb != 0 ? 1 : 0;
        if (g_expert_pool_enabled) {
            g_expert_pool_budget_bytes = mb * 1024ull * 1024ull;
            g_expert_pool_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_INTERVAL", 64u);
            g_expert_pool_layer_start = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_LAYER_START", 0u);
            g_expert_pool_layer_end = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_LAYER_END", 127u);
            if (g_expert_pool_layer_end < g_expert_pool_layer_start) {
                g_expert_pool_layer_end = g_expert_pool_layer_start;
            }
            /* A nonzero DS4_METAL_EXPERT_POOL_MB should mean a real foreground
             * LRU cache: every miss is admitted and the least-recently-used entry
             * is evicted only when the pool is full.  Predictor prefetch is an
             * explicit opt-in layer on top of that; defaulting it on caused the
             * async copier to churn the pool and hide whether LRU itself helps. */
            g_expert_pool_lookahead = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PREFETCH_LOOKAHEAD", 0u);
            if (g_expert_pool_lookahead > 16u) g_expert_pool_lookahead = 16u;
            g_expert_pool_prefetch_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PREFETCH_TOP", 0u);
            if (g_expert_pool_prefetch_top > 64u) g_expert_pool_prefetch_top = 64u;
            g_expert_pool_prefetch_self = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_PREFETCH_SELF") > 0;
            g_expert_pool_prefetch_adjacent = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_PREFETCH_ADJACENT") > 0;
            g_expert_pool_wait_inflight = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_WAIT_INFLIGHT") > 0;
            int fg_fill_env = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_FOREGROUND_FILL");
            g_expert_pool_foreground_fill = fg_fill_env < 0 ? 1 : (fg_fill_env > 0);
            g_expert_pool_warm_batch = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_WARM_BATCH") > 0;
            g_expert_pool_prefetch_evict = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_PREFETCH_EVICT") > 0;
            int hit_only_env = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_HIT_ONLY");
            g_expert_pool_hit_only = hit_only_env < 0 ? 1 : (hit_only_env > 0);
            g_expert_pool_admit_after = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_ADMIT_AFTER", 1u);
            if (g_expert_pool_admit_after == 0) g_expert_pool_admit_after = 1u;
            g_expert_pool_hotlock_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_HOTLOCK_TOP", 0u);
            if (g_expert_pool_hotlock_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_pool_hotlock_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
            }
            g_expert_pool_hotlock_enabled = g_expert_pool_hotlock_top != 0 ? 1 : 0;
            g_expert_pool_auto_pin_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_AUTO_PIN_TOP", 0u);
            if (g_expert_pool_auto_pin_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_pool_auto_pin_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
            }
            g_expert_pool_auto_pin_min_req = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_AUTO_PIN_MIN_REQ", 3u);
            if (g_expert_pool_auto_pin_min_req == 0) g_expert_pool_auto_pin_min_req = 1u;
            g_expert_pool_auto_pin_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_AUTO_PIN_INTERVAL", 32u);
            g_expert_pool_pin_reserve = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PIN_RESERVE", 8u);
            g_expert_pool_hotlist_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_HOTLIST_TOP", 16u);
            if (g_expert_pool_hotlist_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_pool_hotlist_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
            }
            g_expert_pool_hotlist_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_HOTLIST_INTERVAL", g_expert_pool_interval);
            ds4_gpu_expert_pool_parse_pinned_env();
        }
    }
    return g_expert_pool_enabled;
}
