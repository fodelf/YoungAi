/* metal_expert_pool_thread.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

static void *ds4_gpu_expert_pool_prefetch_main(void *arg) {
    (void)arg;
    for (;;) {
        ds4_metal_expert_prefetch_req req = { 0 };
        ds4_metal_expert_pool_meta meta = { 0 };
        int32_t slot = -1;

        pthread_mutex_lock(&g_expert_pool_mu);
        while (!g_expert_pool_prefetch_shutdown && g_expert_pool_prefetch_qcount == 0) {
            pthread_cond_wait(&g_expert_pool_cv, &g_expert_pool_mu);
        }
        if (g_expert_pool_prefetch_shutdown && g_expert_pool_prefetch_qcount == 0) {
            pthread_mutex_unlock(&g_expert_pool_mu);
            break;
        }
        req = g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead];
        memset(&g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead], 0,
               sizeof(g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead]));
        g_expert_pool_prefetch_qhead = (g_expert_pool_prefetch_qhead + 1u) % g_expert_pool_prefetch_qcap;
        g_expert_pool_prefetch_qcount--;
        g_expert_pool_prefetches++;

        slot = ds4_gpu_expert_pool_find(req.model_map, req.layer, req.expert);
        if (slot >= 0) {
            ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
            if (e->ready) {
                g_expert_pool_prefetch_hits++;
                e->last_used = ++g_expert_pool_clock;
                ds4_gpu_expert_pool_lru_touch(slot);
            } else {
                g_expert_pool_inflight_hits++;
            }
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        if (req.layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
            g_expert_pool_prefetch_drops++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        meta = g_expert_pool_meta[req.layer];
        if (!meta.used || meta.model_map != req.model_map || req.expert >= meta.n_expert_total) {
            g_expert_pool_prefetch_drops++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        if (!g_expert_pool_prefetch_evict) {
            for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
                if (!g_expert_pool_entries[i].used) {
                    slot = (int32_t)i;
                    break;
                }
            }
        } else {
            uint32_t req_cap = 0;
            if (!ds4_gpu_expert_pool_layer_served(req.layer, &req_cap)) {
                g_expert_pool_prefetch_drops++;
                pthread_mutex_unlock(&g_expert_pool_mu);
                continue;
            }
            slot = ds4_gpu_expert_pool_victim(req.model_map, req.layer, req_cap, NULL, 0);
        }
        if (slot < 0) {
            g_expert_pool_prefetch_drops++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
        e->used = true;
        e->ready = false;
        e->loading = true;
        e->busy = false;
        e->model_map = req.model_map;
        e->layer = req.layer;
        e->expert = req.expert;
        e->pinned = ds4_gpu_expert_pool_is_pinned(req.layer, req.expert);
        e->last_used = ++g_expert_pool_clock;
        ds4_gpu_expert_pool_lru_link_head(slot);
        g_expert_pool_prefetch_misses++;
        pthread_mutex_unlock(&g_expert_pool_mu);

        const double t0 = ds4_gpu_now_ms();
        const int ok = ds4_gpu_expert_pool_copy_slot(slot, &meta, req.expert);
        const double dt = ds4_gpu_now_ms() - t0;

        pthread_mutex_lock(&g_expert_pool_mu);
        e = &g_expert_pool_entries[slot];
        if (e->used && e->model_map == req.model_map && e->layer == req.layer &&
            e->expert == req.expert && e->loading) {
            if (ok) {
                e->ready = true;
                e->loading = false;
                g_expert_pool_prefetch_copied++;
                g_expert_pool_prefetch_copy_ms += dt;
            } else {
                ds4_gpu_expert_pool_lru_unlink(slot);
                memset(e, 0, sizeof(*e));
                e->prev = -1;
                e->next = -1;
                g_expert_pool_prefetch_drops++;
            }
            pthread_cond_broadcast(&g_expert_pool_cv);
        }
        pthread_mutex_unlock(&g_expert_pool_mu);
    }
    return NULL;
}

static void ds4_gpu_expert_pool_stop_prefetch(void) {
    if (!g_expert_pool_prefetch_thread_started) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    g_expert_pool_prefetch_shutdown = true;
    pthread_cond_broadcast(&g_expert_pool_cv);
    pthread_mutex_unlock(&g_expert_pool_mu);
}

int ds4_gpu_expert_pool_init(uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (!ds4_gpu_expert_pool_is_enabled()) return 0;
    const uint64_t slot_bytes = 2ull * gate_expert_bytes + down_expert_bytes;
    if (slot_bytes == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;

    if (g_expert_pool_slots != 0) {
        return g_expert_pool_gate_expert_bytes == gate_expert_bytes &&
               g_expert_pool_down_expert_bytes == down_expert_bytes;
    }
    if (g_expert_pool_init_attempted) return 0;
    g_expert_pool_init_attempted = true;

    uint64_t slots64 = g_expert_pool_budget_bytes / slot_bytes;
    if (slots64 == 0 || slots64 > (uint64_t)INT32_MAX) return 0;
    if (slots64 > (uint64_t)UINT32_MAX) slots64 = UINT32_MAX;
    const uint32_t slots = (uint32_t)slots64;
    const uint64_t gate_total = (uint64_t)slots * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)slots * down_expert_bytes;
    if (gate_total > NSUIntegerMax || down_total > NSUIntegerMax) return 0;

    @autoreleasepool {
        g_expert_pool_gate = [g_device newBufferWithLength:(NSUInteger)gate_total
                                                   options:MTLResourceStorageModeShared];
        g_expert_pool_up = [g_device newBufferWithLength:(NSUInteger)gate_total
                                                 options:MTLResourceStorageModeShared];
        g_expert_pool_down = [g_device newBufferWithLength:(NSUInteger)down_total
                                                   options:MTLResourceStorageModeShared];
        if (!g_expert_pool_gate || !g_expert_pool_up || !g_expert_pool_down) {
            fprintf(stderr,
                    "ds4: expert-pool failed to allocate %.2f MiB (%u slots); falling back to A3 scratch\n",
                    (double)(2ull * gate_total + down_total) / (1024.0 * 1024.0),
                    slots);
            g_expert_pool_gate = nil;
            g_expert_pool_up = nil;
            g_expert_pool_down = nil;
            return 0;
        }
        g_expert_pool_gate.label = @"ds4_expert_pool_gate";
        g_expert_pool_up.label = @"ds4_expert_pool_up";
        g_expert_pool_down.label = @"ds4_expert_pool_down";
    }

    g_expert_pool_entries = calloc((size_t)slots, sizeof(g_expert_pool_entries[0]));
    if (!g_expert_pool_entries) {
        g_expert_pool_gate = nil;
        g_expert_pool_up = nil;
        g_expert_pool_down = nil;
        return 0;
    }
    for (uint32_t i = 0; i < slots; i++) {
        g_expert_pool_entries[i].prev = -1;
        g_expert_pool_entries[i].next = -1;
    }
    g_expert_pool_gate_expert_bytes = gate_expert_bytes;
    g_expert_pool_down_expert_bytes = down_expert_bytes;
    g_expert_pool_slot_bytes = slot_bytes;
    g_expert_pool_slots = slots;
    if ((g_expert_pool_lookahead != 0 || g_expert_pool_pinned_total != 0 || g_expert_pool_hit_only) && !g_expert_pool_prefetch_queue) {
        uint64_t qcap64 = ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PREFETCH_QUEUE", (uint64_t)slots * 8ull);
        if (qcap64 < 64u) qcap64 = 64u;
        if (qcap64 > 65536u) qcap64 = 65536u;
        g_expert_pool_prefetch_qcap = (uint32_t)qcap64;
        g_expert_pool_prefetch_queue = calloc(g_expert_pool_prefetch_qcap,
                                              sizeof(g_expert_pool_prefetch_queue[0]));
        if (g_expert_pool_prefetch_queue) {
            pthread_t th;
            if (pthread_create(&th, NULL, ds4_gpu_expert_pool_prefetch_main, NULL) == 0) {
                pthread_detach(th);
                g_expert_pool_prefetch_thread_started = true;
                atexit(ds4_gpu_expert_pool_stop_prefetch);
            } else {
                free(g_expert_pool_prefetch_queue);
                g_expert_pool_prefetch_queue = NULL;
                g_expert_pool_prefetch_qcap = 0;
                fprintf(stderr, "ds4: expert-pool async prefetch thread failed to start; continuing sync-only\n");
            }
        }
    }
    if (!g_expert_pool_summary_registered) {
        atexit(ds4_gpu_expert_pool_summary);
        g_expert_pool_summary_registered = true;
    }
    fprintf(stderr,
            "ds4: expert-pool enabled: %.2f MiB, %u slots, one expert %.3f MiB "
            "(gate=%.3f up=%.3f down=%.3f), configured layers %u:%u, serving tail %u/%u layers with >=%u slots/layer. "
            "LRU resident pool with async predictor prefetch lookahead=%u top=%u q=%u "
            "self=%s adjacent=%s admit_after=%u pinned=%u static=%u dynamic=%u auto_pin_top=%u min_req=%u reserve=%u hotlist=%u/%u hit_only=%s wait_inflight=%s foreground_fill=%s warm_batch=%s pf_evict=%s%s; hits bypass A3 scratch copy.\n",
            (double)(2ull * gate_total + down_total) / (1024.0 * 1024.0),
            slots,
            (double)slot_bytes / (1024.0 * 1024.0),
            (double)gate_expert_bytes / (1024.0 * 1024.0),
            (double)gate_expert_bytes / (1024.0 * 1024.0),
            (double)down_expert_bytes / (1024.0 * 1024.0),
            g_expert_pool_layer_start,
            g_expert_pool_layer_end,
            ds4_gpu_expert_pool_served_layers(),
            ds4_gpu_expert_pool_requested_layers(),
            ds4_gpu_expert_pool_min_layer_slots(),
            g_expert_pool_lookahead,
            g_expert_pool_prefetch_top,
            g_expert_pool_prefetch_qcap,
            g_expert_pool_prefetch_self ? "on" : "off",
            g_expert_pool_prefetch_adjacent ? "on" : "off",
            g_expert_pool_admit_after,
            g_expert_pool_pinned_total,
            g_expert_pool_static_pinned_total,
            g_expert_pool_dynamic_pinned_total,
            g_expert_pool_auto_pin_top,
            g_expert_pool_auto_pin_min_req,
            g_expert_pool_pin_reserve,
            g_expert_pool_hotlist_top,
            g_expert_pool_hotlist_interval,
            g_expert_pool_hit_only ? "on" : "off",
            g_expert_pool_wait_inflight ? "on" : "off",
            g_expert_pool_foreground_fill ? "on" : "off",
            g_expert_pool_warm_batch ? "on" : "off",
            g_expert_pool_prefetch_evict ? "on" : "off",
            g_expert_pool_hotlock_enabled ? " +hotlock" : "");
    return 1;
}
