/* metal_expert_profile.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* host AUTO verdict: -1 unset, else 0/1 */
static int g_expert_offload_cached  = -1;

/* resolved decision (env override or verdict) */

void ds4_gpu_set_expert_offload(int enabled) {
    g_expert_offload_verdict = enabled ? 1 : 0;
    g_expert_offload_cached = -1;   /* re-resolve on next query with the new verdict */
}

uint64_t ds4_gpu_recommended_max_working_set_bytes(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_device == nil) return 0;
    return (uint64_t)[g_device recommendedMaxWorkingSetSize];
}

/* Live GPU working-set size (model wired + graph scratch). Compared against
 * recommendedMaxWorkingSetSize this is the absolute evidence for whether a
 * slice fits in VRAM without paging. */
uint64_t ds4_gpu_current_allocated_bytes(void) {
    if (g_device == nil) return 0;
    return (uint64_t)[g_device currentAllocatedSize];
}

int ds4_gpu_expert_offload_enabled(void) {
    if (g_expert_offload_cached < 0) {
        const char *v = getenv("DS4_METAL_EXPERT_OFFLOAD");
        int decision;
        if (v && v[0]) {
            decision = !(v[0] == '0' && v[1] == '\0') ? 1 : 0;   /* explicit override */
        } else {
            decision = (g_expert_offload_verdict >= 0) ? g_expert_offload_verdict : 0;
        }
        g_expert_offload_cached = decision;
        if (decision) {
            fprintf(stderr,
                    "ds4: routed-expert offload ON (mmap views non-resident, per-layer CPU "
                    "gather; over-budget model or DS4_METAL_EXPERT_OFFLOAD=1).\n");
        } else {
            fprintf(stderr,
                    "ds4: routed experts RESIDENT (wired, direct GPU read, no per-layer gather; "
                    "model fits budget or DS4_METAL_EXPERT_OFFLOAD=0).\n");
        }
    }
    return g_expert_offload_cached;
}

int ds4_gpu_expert_offload_direct_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_OFFLOAD_DIRECT") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: DS4_METAL_EXPERT_OFFLOAD_DIRECT=1: q2 routed experts bypass A3 CPU-gather; "
                    "GPU reads selected rows directly from non-resident mmap views.\n");
        } else if (ds4_gpu_expert_offload_enabled()) {
            fprintf(stderr,
                    "ds4: q2 routed experts use A3 CPU-gather scratch; set "
                    "DS4_METAL_EXPERT_OFFLOAD_DIRECT=1 to avoid per-layer CPU gather barriers.\n");
        }
    }
    return cached;
}

#define DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES 65536u

static int g_expert_profile_enabled = -1;

static bool g_expert_profile_initialized;

static bool g_expert_profile_summary_registered;

static uint64_t g_expert_profile_cache_capacity_bytes;

static uint64_t g_expert_profile_cache_used_bytes;

static uint64_t g_expert_profile_cache_peak_bytes;

static uint64_t g_expert_profile_cache_slots;

static uint64_t g_expert_profile_calls;

static uint64_t g_expert_profile_routed_picks;

static uint64_t g_expert_profile_unique_requests;

static uint64_t g_expert_profile_hits;

static uint64_t g_expert_profile_misses;

static uint64_t g_expert_profile_actual_copy_bytes;

static uint64_t g_expert_profile_simulated_miss_bytes;

static double   g_expert_profile_copy_ms;

static uint32_t g_expert_profile_interval;

static uint32_t g_expert_profile_top;

static bool     g_expert_profile_all;

static bool     g_expert_profile_memory_seen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint32_t g_expert_profile_layer_experts[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint64_t g_expert_profile_gate_bytes[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint64_t g_expert_profile_down_bytes[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static uint64_t g_expert_profile_slot_bytes[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

static ds4_metal_expert_profile_layer g_expert_profile_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

ds4_metal_expert_profile_slot g_expert_profile_slot[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

static int32_t g_expert_profile_cache_index[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

static ds4_metal_expert_profile_cache_entry g_expert_profile_cache[DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES];

static int32_t g_expert_profile_cache_head = -1;

static int32_t g_expert_profile_cache_tail = -1;

static int32_t g_expert_profile_cache_free = -1;

uint64_t ds4_gpu_env_u64(const char *name, uint64_t defval) {
    const char *v = getenv(name);
    if (!v || !v[0]) return defval;
    char *endp = NULL;
    unsigned long long parsed = strtoull(v, &endp, 10);
    return endp != v ? (uint64_t)parsed : defval;
}

static void ds4_gpu_expert_profile_print_live(const char *tag) {
    if (g_expert_profile_unique_requests == 0) return;
    const double hit_pct = 100.0 * (double)g_expert_profile_hits /
                           (double)g_expert_profile_unique_requests;
    const double saved_pct = g_expert_profile_actual_copy_bytes ?
        100.0 * (double)(g_expert_profile_actual_copy_bytes -
                         (g_expert_profile_simulated_miss_bytes < g_expert_profile_actual_copy_bytes ?
                          g_expert_profile_simulated_miss_bytes : g_expert_profile_actual_copy_bytes)) /
        (double)g_expert_profile_actual_copy_bytes : 0.0;
    fprintf(stderr,
            "ds4: expert-profile %s: calls=%llu picks=%llu unique=%llu "
            "sim_lru_hit=%.2f%% hits=%llu misses=%llu cache=%.2f/%.2f MiB peak=%.2f MiB "
            "actual_copy=%.2f GiB sim_miss=%.2f GiB saved=%.2f%% copy=%.3f ms\n",
            tag ? tag : "live",
            (unsigned long long)g_expert_profile_calls,
            (unsigned long long)g_expert_profile_routed_picks,
            (unsigned long long)g_expert_profile_unique_requests,
            hit_pct,
            (unsigned long long)g_expert_profile_hits,
            (unsigned long long)g_expert_profile_misses,
            (double)g_expert_profile_cache_used_bytes / (1024.0 * 1024.0),
            (double)g_expert_profile_cache_capacity_bytes / (1024.0 * 1024.0),
            (double)g_expert_profile_cache_peak_bytes / (1024.0 * 1024.0),
            (double)g_expert_profile_actual_copy_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)g_expert_profile_simulated_miss_bytes / (1024.0 * 1024.0 * 1024.0),
            saved_pct,
            g_expert_profile_copy_ms);
}

static void ds4_gpu_expert_profile_dump_layers(void) {
    const uint32_t top = g_expert_profile_all ? DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS : g_expert_profile_top;
    for (uint32_t il = 0; il < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; il++) {
        ds4_metal_expert_profile_layer *ls = &g_expert_profile_layer[il];
        if (ls->unique_requests == 0) continue;
        const uint64_t slot_bytes = g_expert_profile_slot_bytes[il];
        const double layer_hit = 100.0 * (double)ls->hits / (double)ls->unique_requests;
        fprintf(stderr,
                "ds4: expert-profile layer %02u: experts=%u slot=%.3f MiB "
                "(gate=%.3f up=%.3f down=%.3f) layer_experts=%.3f GiB "
                "calls=%llu picks=%llu unique=%llu hit=%.2f%% misses=%llu "
                "actual=%.3f GiB sim_miss=%.3f GiB copy=%.3f ms\n",
                il,
                g_expert_profile_layer_experts[il],
                (double)slot_bytes / (1024.0 * 1024.0),
                (double)g_expert_profile_gate_bytes[il] / (1024.0 * 1024.0),
                (double)g_expert_profile_gate_bytes[il] / (1024.0 * 1024.0),
                (double)g_expert_profile_down_bytes[il] / (1024.0 * 1024.0),
                (double)(slot_bytes * (uint64_t)g_expert_profile_layer_experts[il]) /
                    (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)ls->calls,
                (unsigned long long)ls->routed_picks,
                (unsigned long long)ls->unique_requests,
                layer_hit,
                (unsigned long long)ls->misses,
                (double)ls->actual_copy_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)ls->simulated_miss_bytes / (1024.0 * 1024.0 * 1024.0),
                ls->copy_ms);

        bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
        uint32_t printed_count = 0;
        while (printed_count < top) {
            int best = -1;
            uint64_t best_req = 0;
            for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
                if (printed[e]) continue;
                const uint64_t req = g_expert_profile_slot[il][e].requests;
                if (req > best_req) {
                    best_req = req;
                    best = (int)e;
                }
            }
            if (best < 0 || best_req == 0) break;
            printed[(uint32_t)best] = true;
            printed_count++;
            const uint64_t miss = g_expert_profile_slot[il][best].misses;
            const uint64_t hit = best_req - miss;
            const double hp = 100.0 * (double)hit / (double)best_req;
            const bool resident = g_expert_profile_cache_index[il][best] >= 0;
            fprintf(stderr,
                    "ds4: expert-profile   L%02u E%03u req=%llu hit=%llu miss=%llu "
                    "hit=%.2f%% actual=%.3f MiB sim_miss=%.3f MiB resident=%d\n",
                    il,
                    (uint32_t)best,
                    (unsigned long long)best_req,
                    (unsigned long long)hit,
                    (unsigned long long)miss,
                    hp,
                    (double)(best_req * slot_bytes) / (1024.0 * 1024.0),
                    (double)(miss * slot_bytes) / (1024.0 * 1024.0),
                    resident ? 1 : 0);
        }
    }
}

static void ds4_gpu_expert_profile_summary(void) {
    if (!g_expert_profile_initialized || g_expert_profile_unique_requests == 0) return;
    ds4_gpu_expert_profile_print_live("summary");
    ds4_gpu_expert_profile_dump_layers();
}

static void ds4_gpu_expert_profile_init(void) {
    if (g_expert_profile_initialized) return;
    g_expert_profile_initialized = true;

    for (uint32_t il = 0; il < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; il++) {
        for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
            g_expert_profile_cache_index[il][e] = -1;
        }
    }
    for (uint32_t i = 0; i < DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES; i++) {
        g_expert_profile_cache[i].next = (i + 1u < DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES) ?
                                         (int32_t)(i + 1u) : -1;
        g_expert_profile_cache[i].prev = -1;
    }
    g_expert_profile_cache_free = 0;

    const uint64_t cache_mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_PROFILE_CACHE_MB", 1024u);
    g_expert_profile_cache_capacity_bytes = cache_mb * 1024ull * 1024ull;
    g_expert_profile_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_PROFILE_INTERVAL", 512u);
    g_expert_profile_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_PROFILE_TOP", 8u);
    if (g_expert_profile_top == 0) g_expert_profile_top = 1;
    if (g_expert_profile_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        g_expert_profile_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
    }
    g_expert_profile_all = ds4_gpu_env_bool("DS4_METAL_EXPERT_PROFILE_ALL") > 0;

    if (!g_expert_profile_summary_registered) {
        atexit(ds4_gpu_expert_profile_summary);
        g_expert_profile_summary_registered = true;
    }
    fprintf(stderr,
            "ds4: expert-offload profiler enabled: simulated LRU cache %.2f MiB, "
            "top=%u%s, live interval=%u MoE calls. Note: this profiles hit rate; "
            "A3 still copies active experts to scratch today.\n",
            (double)g_expert_profile_cache_capacity_bytes / (1024.0 * 1024.0),
            g_expert_profile_top,
            g_expert_profile_all ? " (ALL nonzero experts at exit)" : "",
            g_expert_profile_interval);
}

int ds4_gpu_expert_profile_is_enabled(void) {
    if (g_expert_profile_enabled < 0) {
        g_expert_profile_enabled = ds4_gpu_env_bool("DS4_METAL_EXPERT_OFFLOAD_PROFILE") > 0 ? 1 : 0;
        if (g_expert_profile_enabled) ds4_gpu_expert_profile_init();
    }
    return g_expert_profile_enabled;
}

static void ds4_gpu_expert_profile_cache_unlink(int32_t idx) {
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    if (e->prev >= 0) g_expert_profile_cache[e->prev].next = e->next;
    if (e->next >= 0) g_expert_profile_cache[e->next].prev = e->prev;
    if (g_expert_profile_cache_head == idx) g_expert_profile_cache_head = e->next;
    if (g_expert_profile_cache_tail == idx) g_expert_profile_cache_tail = e->prev;
    e->prev = -1;
    e->next = -1;
}

static void ds4_gpu_expert_profile_cache_link_head(int32_t idx) {
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    e->prev = -1;
    e->next = g_expert_profile_cache_head;
    if (g_expert_profile_cache_head >= 0) g_expert_profile_cache[g_expert_profile_cache_head].prev = idx;
    g_expert_profile_cache_head = idx;
    if (g_expert_profile_cache_tail < 0) g_expert_profile_cache_tail = idx;
}

static void ds4_gpu_expert_profile_cache_touch(int32_t idx) {
    if (idx == g_expert_profile_cache_head) return;
    ds4_gpu_expert_profile_cache_unlink(idx);
    ds4_gpu_expert_profile_cache_link_head(idx);
}

static void ds4_gpu_expert_profile_cache_evict_tail(void) {
    const int32_t idx = g_expert_profile_cache_tail;
    if (idx < 0) return;
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    ds4_gpu_expert_profile_cache_unlink(idx);
    if (e->layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        e->expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        g_expert_profile_cache_index[e->layer][e->expert] = -1;
    }
    if (g_expert_profile_cache_used_bytes >= e->bytes) {
        g_expert_profile_cache_used_bytes -= e->bytes;
    } else {
        g_expert_profile_cache_used_bytes = 0;
    }
    if (g_expert_profile_cache_slots > 0) g_expert_profile_cache_slots--;
    e->used = false;
    e->bytes = 0;
    e->next = g_expert_profile_cache_free;
    e->prev = -1;
    g_expert_profile_cache_free = idx;
}

static int32_t ds4_gpu_expert_profile_cache_alloc_entry(void) {
    while (g_expert_profile_cache_free < 0 && g_expert_profile_cache_tail >= 0) {
        ds4_gpu_expert_profile_cache_evict_tail();
    }
    if (g_expert_profile_cache_free < 0) return -1;
    const int32_t idx = g_expert_profile_cache_free;
    g_expert_profile_cache_free = g_expert_profile_cache[idx].next;
    g_expert_profile_cache[idx].next = -1;
    g_expert_profile_cache[idx].prev = -1;
    return idx;
}

static void ds4_gpu_expert_profile_cache_insert(uint32_t layer, uint32_t expert, uint64_t bytes) {
    if (g_expert_profile_cache_capacity_bytes == 0 || bytes == 0 ||
        bytes > g_expert_profile_cache_capacity_bytes) {
        return;
    }
    while (g_expert_profile_cache_tail >= 0 &&
           g_expert_profile_cache_used_bytes + bytes > g_expert_profile_cache_capacity_bytes) {
        ds4_gpu_expert_profile_cache_evict_tail();
    }
    if (g_expert_profile_cache_used_bytes + bytes > g_expert_profile_cache_capacity_bytes) return;
    const int32_t idx = ds4_gpu_expert_profile_cache_alloc_entry();
    if (idx < 0) return;
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    e->used = true;
    e->layer = layer;
    e->expert = expert;
    e->bytes = bytes;
    ds4_gpu_expert_profile_cache_link_head(idx);
    g_expert_profile_cache_index[layer][expert] = idx;
    g_expert_profile_cache_used_bytes += bytes;
    g_expert_profile_cache_slots++;
    if (g_expert_profile_cache_used_bytes > g_expert_profile_cache_peak_bytes) {
        g_expert_profile_cache_peak_bytes = g_expert_profile_cache_used_bytes;
    }
}

void ds4_gpu_expert_profile_record(
        uint32_t layer_index,
        const uint32_t *active_ids,
        uint32_t n_active,
        uint32_t n_total_expert,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t routed_picks,
        double copy_ms) {
    if (!ds4_gpu_expert_profile_is_enabled()) return;
    if (!active_ids || n_active == 0) return;
    if (layer_index >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;

    const uint64_t slot_bytes = 2ull * gate_expert_bytes + down_expert_bytes;
    if (!g_expert_profile_memory_seen[layer_index]) {
        g_expert_profile_memory_seen[layer_index] = true;
        g_expert_profile_layer_experts[layer_index] = n_total_expert;
        g_expert_profile_gate_bytes[layer_index] = gate_expert_bytes;
        g_expert_profile_down_bytes[layer_index] = down_expert_bytes;
        g_expert_profile_slot_bytes[layer_index] = slot_bytes;
        fprintf(stderr,
                "ds4: expert-profile memory layer %02u: expert_slots=%u "
                "one_expert=%.3f MiB (gate=%.3f up=%.3f down=%.3f), "
                "all_layer_experts=%.3f GiB\n",
                layer_index,
                n_total_expert,
                (double)slot_bytes / (1024.0 * 1024.0),
                (double)gate_expert_bytes / (1024.0 * 1024.0),
                (double)gate_expert_bytes / (1024.0 * 1024.0),
                (double)down_expert_bytes / (1024.0 * 1024.0),
                (double)(slot_bytes * (uint64_t)n_total_expert) /
                    (1024.0 * 1024.0 * 1024.0));
    }

    ds4_metal_expert_profile_layer *ls = &g_expert_profile_layer[layer_index];
    ls->calls++;
    ls->routed_picks += routed_picks;
    ls->actual_copy_bytes += (uint64_t)n_active * slot_bytes;
    ls->copy_ms += copy_ms;

    g_expert_profile_calls++;
    g_expert_profile_routed_picks += routed_picks;
    g_expert_profile_actual_copy_bytes += (uint64_t)n_active * slot_bytes;
    g_expert_profile_copy_ms += copy_ms;

    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t expert = active_ids[i];
        if (expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) continue;
        g_expert_profile_slot[layer_index][expert].requests++;
        ls->unique_requests++;
        g_expert_profile_unique_requests++;

        int32_t idx = g_expert_profile_cache_index[layer_index][expert];
        if (idx >= 0 && g_expert_profile_cache[idx].used) {
            ds4_gpu_expert_profile_cache_touch(idx);
            ls->hits++;
            g_expert_profile_hits++;
        } else {
            g_expert_profile_slot[layer_index][expert].misses++;
            ls->misses++;
            ls->simulated_miss_bytes += slot_bytes;
            g_expert_profile_misses++;
            g_expert_profile_simulated_miss_bytes += slot_bytes;
            ds4_gpu_expert_profile_cache_insert(layer_index, expert, slot_bytes);
        }
    }

    if (g_expert_profile_interval != 0 &&
        (g_expert_profile_calls % g_expert_profile_interval) == 0) {
        ds4_gpu_expert_profile_print_live("live");
        /* PROFILE_ALL: also emit per-expert hot set periodically so a SIGTERM'd
         * worker (no atexit) still dumps its layers' hot experts to its log. */
        if (g_expert_profile_all) ds4_gpu_expert_profile_dump_layers();
    }
}
