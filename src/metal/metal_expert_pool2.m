/* metal_expert_pool2.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

void ds4_gpu_expert_pool_print(const char *tag) {
    if (g_expert_pool_calls == 0 && g_expert_pool_requests == 0 &&
        g_expert_pool_prefetches == 0 && g_expert_pool_fallbacks == 0 &&
        g_expert_pool_admission_skips == 0) return;
    const double hit_pct = g_expert_pool_requests ?
        100.0 * (double)g_expert_pool_hits / (double)g_expert_pool_requests : 0.0;
    const double pinned_hit_pct = g_expert_pool_pinned_requests ?
        100.0 * (double)g_expert_pool_pinned_hits / (double)g_expert_pool_pinned_requests : 0.0;
    fprintf(stderr,
            "ds4: expert-pool %s: calls=%llu requests=%llu hit=%.2f%% hits=%llu "
            "misses=%llu inflight=%llu fallbacks=%llu admit_skip=%llu slots=%u slot=%.3f MiB budget=%.2f MiB "
            "layers=%u:%u pinned=%u static=%u dynamic=%u auto_pin=%u/%u/%u auto_updates=%llu "
            "pinned_req=%llu pinned_hit=%.2f%% pinned_hits=%llu pinned_miss=%llu pinned_pf=%llu "
            "prefetch=%llu pf_hit=%llu pf_miss=%llu pf_drop=%llu pf_copy=%llu "
            "miss_copy=%.2f GiB sync_copy=%.3f ms pf_copy_ms=%.3f wait=%llu/%.3f ms\n",
            tag ? tag : "live",
            (unsigned long long)g_expert_pool_calls,
            (unsigned long long)g_expert_pool_requests,
            hit_pct,
            (unsigned long long)g_expert_pool_hits,
            (unsigned long long)g_expert_pool_misses,
            (unsigned long long)g_expert_pool_inflight_hits,
            (unsigned long long)g_expert_pool_fallbacks,
            (unsigned long long)g_expert_pool_admission_skips,
            g_expert_pool_slots,
            (double)g_expert_pool_slot_bytes / (1024.0 * 1024.0),
            (double)g_expert_pool_budget_bytes / (1024.0 * 1024.0),
            g_expert_pool_layer_start,
            g_expert_pool_layer_end,
            g_expert_pool_pinned_total,
            g_expert_pool_static_pinned_total,
            g_expert_pool_dynamic_pinned_total,
            g_expert_pool_auto_pin_top,
            g_expert_pool_auto_pin_min_req,
            g_expert_pool_pin_reserve,
            (unsigned long long)g_expert_pool_auto_pin_updates,
            (unsigned long long)g_expert_pool_pinned_requests,
            pinned_hit_pct,
            (unsigned long long)g_expert_pool_pinned_hits,
            (unsigned long long)g_expert_pool_pinned_misses,
            (unsigned long long)g_expert_pool_pinned_prefetches,
            (unsigned long long)g_expert_pool_prefetches,
            (unsigned long long)g_expert_pool_prefetch_hits,
            (unsigned long long)g_expert_pool_prefetch_misses,
            (unsigned long long)g_expert_pool_prefetch_drops,
            (unsigned long long)g_expert_pool_prefetch_copied,
            (double)g_expert_pool_miss_copy_bytes / (1024.0 * 1024.0 * 1024.0),
            g_expert_pool_miss_copy_ms,
            g_expert_pool_prefetch_copy_ms,
            (unsigned long long)g_expert_pool_sync_waits,
            g_expert_pool_wait_ms);
}

void ds4_gpu_expert_pool_summary(void) {
    ds4_gpu_expert_pool_print("summary");
    uint32_t top = g_expert_pool_hotlist_top != 0 ? g_expert_pool_hotlist_top : 16u;
    ds4_gpu_expert_pool_print_hotlist("summary", top, true);
}

void ds4_gpu_expert_pool_lru_unlink(int32_t idx) {
    ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[idx];
    if (e->prev >= 0) g_expert_pool_entries[e->prev].next = e->next;
    if (e->next >= 0) g_expert_pool_entries[e->next].prev = e->prev;
    if (g_expert_pool_head == idx) g_expert_pool_head = e->next;
    if (g_expert_pool_tail == idx) g_expert_pool_tail = e->prev;
    e->prev = -1;
    e->next = -1;
}

void ds4_gpu_expert_pool_lru_link_head(int32_t idx) {
    ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[idx];
    e->prev = -1;
    e->next = g_expert_pool_head;
    if (g_expert_pool_head >= 0) g_expert_pool_entries[g_expert_pool_head].prev = idx;
    g_expert_pool_head = idx;
    if (g_expert_pool_tail < 0) g_expert_pool_tail = idx;
}

void ds4_gpu_expert_pool_lru_touch(int32_t idx) {
    if (idx == g_expert_pool_head) return;
    ds4_gpu_expert_pool_lru_unlink(idx);
    ds4_gpu_expert_pool_lru_link_head(idx);
}

int32_t ds4_gpu_expert_pool_find(const void *model_map, uint32_t layer, uint32_t expert) {
    if (!g_expert_pool_entries) return -1;
    for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
        ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[i];
        if (e->used && e->model_map == model_map && e->layer == layer && e->expert == expert) {
            return (int32_t)i;
        }
    }
    return -1;
}

void ds4_gpu_expert_pool_clear_busy(void) {
    if (!g_expert_pool_entries) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
        g_expert_pool_entries[i].busy = false;
    }
    pthread_cond_broadcast(&g_expert_pool_cv);
    pthread_mutex_unlock(&g_expert_pool_mu);
}

int ds4_gpu_expert_pool_copy_slot(
        int32_t slot,
        const ds4_metal_expert_pool_meta *meta,
        uint32_t expert) {
    if (slot < 0 || !meta || !meta->used || expert >= meta->n_expert_total) return 0;
    uint8_t *gate_dst_base = (uint8_t *)g_expert_pool_gate.contents;
    uint8_t *up_dst_base = (uint8_t *)g_expert_pool_up.contents;
    uint8_t *down_dst_base = (uint8_t *)g_expert_pool_down.contents;
    const uint8_t *map = (const uint8_t *)meta->model_map;
    if (!gate_dst_base || !up_dst_base || !down_dst_base || !map) return 0;

    const uint64_t gate_src = (uint64_t)expert * meta->gate_expert_bytes;
    const uint64_t down_src = (uint64_t)expert * meta->down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)(uint32_t)slot * meta->gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)(uint32_t)slot * meta->down_expert_bytes;
    memcpy(gate_dst_base + gate_dst,
           map + meta->gate_offset + gate_src,
           (size_t)meta->gate_expert_bytes);
    memcpy(up_dst_base + gate_dst,
           map + meta->up_offset + gate_src,
           (size_t)meta->gate_expert_bytes);
    memcpy(down_dst_base + down_dst,
           map + meta->down_offset + down_src,
           (size_t)meta->down_expert_bytes);
    return 1;
}

void ds4_gpu_expert_pool_register_layer_meta(
        const void *model_map,
        uint32_t    layer,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS || !model_map || n_expert_total == 0) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    ds4_metal_expert_pool_meta *m = &g_expert_pool_meta[layer];
    if (!m->used) {
        m->used = true;
        m->model_map = model_map;
        m->layer = layer;
        m->gate_offset = gate_offset;
        m->up_offset = up_offset;
        m->down_offset = down_offset;
        m->gate_expert_bytes = gate_expert_bytes;
        m->down_expert_bytes = down_expert_bytes;
        m->n_expert_total = n_expert_total;
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

int ds4_gpu_expert_pool_enqueue_prefetch_unlocked(
        const void *model_map,
        uint32_t layer,
        uint32_t expert) {
    if (!g_expert_pool_prefetch_queue || g_expert_pool_prefetch_qcap == 0) return 0;
    if (!ds4_gpu_expert_pool_layer_allowed(layer)) return 0;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return 0;
    const ds4_metal_expert_pool_meta *m = &g_expert_pool_meta[layer];
    if (!m->used || m->model_map != model_map || expert >= m->n_expert_total) return 0;

    int32_t slot = ds4_gpu_expert_pool_find(model_map, layer, expert);
    if (slot >= 0) return 1;
    for (uint32_t i = 0, q = g_expert_pool_prefetch_qhead;
         i < g_expert_pool_prefetch_qcount;
         i++, q = (q + 1u) % g_expert_pool_prefetch_qcap) {
        ds4_metal_expert_prefetch_req *r = &g_expert_pool_prefetch_queue[q];
        if (r->used && r->model_map == model_map && r->layer == layer && r->expert == expert) {
            return 1;
        }
    }
    if (g_expert_pool_prefetch_qcount >= g_expert_pool_prefetch_qcap) {
        g_expert_pool_prefetch_drops++;
        return 0;
    }
    ds4_metal_expert_prefetch_req *r = NULL;
    if (ds4_gpu_expert_pool_is_pinned(layer, expert)) {
        g_expert_pool_prefetch_qhead = (g_expert_pool_prefetch_qhead + g_expert_pool_prefetch_qcap - 1u) %
                                      g_expert_pool_prefetch_qcap;
        r = &g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead];
    } else {
        r = &g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qtail];
        g_expert_pool_prefetch_qtail = (g_expert_pool_prefetch_qtail + 1u) % g_expert_pool_prefetch_qcap;
    }
    r->used = true;
    r->model_map = model_map;
    r->layer = layer;
    r->expert = expert;
    g_expert_pool_prefetch_qcount++;
    pthread_cond_signal(&g_expert_pool_cv);
    return 1;
}

void ds4_gpu_expert_pool_predict_enqueue(
        const void *model_map,
        uint32_t layer,
        const uint32_t *active_ids,
        uint32_t n_active) {
    if (!model_map || !active_ids || n_active == 0 || g_expert_pool_lookahead == 0) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
        uint32_t keep = n_active;
        if (keep > DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX) keep = DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX;
        for (uint32_t i = 0; i < keep; i++) {
            g_expert_pool_last_active[layer][i] = active_ids[i];
        }
        g_expert_pool_last_active_n[layer] = keep;
    }
    if (g_expert_pool_prefetch_self && g_expert_pool_prefetch_top != 0 && layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
        uint32_t self_n = n_active;
        if (self_n > g_expert_pool_prefetch_top) self_n = g_expert_pool_prefetch_top;
        for (uint32_t i = 0; i < self_n; i++) {
            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, active_ids[i]);
        }
    }
    for (uint32_t d = 1; d <= g_expert_pool_lookahead; d++) {
        const uint32_t next_layer = layer + d;
        if (next_layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) break;
        const ds4_metal_expert_pool_meta *m = &g_expert_pool_meta[next_layer];
        if (!m->used || m->model_map != model_map) continue;
        /* Predictor 1 (opt-in): adjacent-layer reuse is weak for this GGUF.  Keep
         * it available for experiments, but do not let it flood the queue by
         * default; the decode-critical predictor is same-layer temporal reuse
         * below. */
        if (g_expert_pool_prefetch_adjacent) {
            const uint32_t active_prefetch = g_expert_pool_prefetch_top != 0 &&
                                            n_active > g_expert_pool_prefetch_top ?
                                            g_expert_pool_prefetch_top : n_active;
            for (uint32_t i = 0; i < active_prefetch; i++) {
                (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, next_layer, active_ids[i]);
            }
        }
        /* Predictor 2: once a layer has fired, repeat its previous active set on
         * the next token. This is the useful decode case; it preserves the LRU
         * resident pool while moving likely misses to the background thread. */
        uint32_t prev_n = g_expert_pool_last_active_n[next_layer];
        if (g_expert_pool_prefetch_top != 0 && prev_n > g_expert_pool_prefetch_top) {
            prev_n = g_expert_pool_prefetch_top;
        }
        for (uint32_t i = 0; i < prev_n; i++) {
            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map,
                                                                next_layer,
                                                                g_expert_pool_last_active[next_layer][i]);
        }
        /* Predictor 3: measured hot experts per layer (pool request counters).
         * Opt-in via --expert-pool-prefetch-top; helps when a few code/text
         * experts dominate. */
        if (g_expert_pool_prefetch_top != 0) {
            bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
            for (uint32_t k = 0; k < g_expert_pool_prefetch_top; k++) {
                int best = -1;
                uint64_t best_req = 0;
                for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
                    if (printed[e]) continue;
                    const uint64_t req = g_expert_pool_hot_count[next_layer][e];
                    if (req > best_req) {
                        best_req = req;
                        best = (int)e;
                    }
                }
                if (best < 0 || best_req == 0) break;
                printed[(uint32_t)best] = true;
                (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, next_layer, (uint32_t)best);
            }
        }
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static bool ds4_gpu_expert_pool_protected(
        const ds4_metal_expert_pool_entry *e,
        const void *model_map,
        uint32_t layer,
        const uint32_t *active_ids,
        uint32_t n_active) {
    if (!e || !e->used) return false;
    if (e->busy || e->loading) return true;
    if (e->model_map != model_map) return false;
    if (e->pinned) return true;
    if (e->layer != layer) {
        /* Effective LRU is per served layer: layer A must never evict layer B,
         * otherwise a small global pool cycles through layers and can hit 0% even
         * when each individual layer has strong temporal locality. */
        return true;
    }
    for (uint32_t i = 0; i < n_active; i++) {
        if (active_ids[i] == e->expert) return true;
    }
    return false;
}

int32_t ds4_gpu_expert_pool_victim(
        const void *model_map,
        uint32_t layer,
        uint32_t layer_cap,
        const uint32_t *active_ids,
        uint32_t n_active) {
    uint32_t layer_used = 0;
    int32_t free_slot = -1;
    for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
        ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[i];
        if (!e->used) {
            if (free_slot < 0) free_slot = (int32_t)i;
            continue;
        }
        if (e->model_map == model_map && e->layer == layer) layer_used++;
    }
    if (free_slot >= 0 && layer_used < layer_cap) return free_slot;

    for (int32_t idx = g_expert_pool_tail; idx >= 0; idx = g_expert_pool_entries[idx].prev) {
        if (!ds4_gpu_expert_pool_protected(&g_expert_pool_entries[idx], model_map, layer,
                                           active_ids, n_active)) {
            ds4_gpu_expert_pool_lru_unlink(idx);
            memset(&g_expert_pool_entries[idx], 0, sizeof(g_expert_pool_entries[idx]));
            g_expert_pool_entries[idx].prev = -1;
            g_expert_pool_entries[idx].next = -1;
            return idx;
        }
    }
    return -1;
}

void ds4_gpu_expert_pool_queue_pinned_layer(
        const void *model_map,
        uint32_t layer) {
    if (!model_map || g_expert_pool_pinned_total == 0) return;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    if (g_expert_pool_pinned_layer_queued[layer]) return;

    pthread_mutex_lock(&g_expert_pool_mu);
    if (!g_expert_pool_pinned_layer_queued[layer]) {
        uint32_t queued = 0;
        uint32_t wanted = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            if (!g_expert_pool_pinned[layer][expert]) continue;
            wanted++;
            if (ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, expert)) {
                queued++;
            }
        }
        if (queued != 0) {
            g_expert_pool_pinned_prefetches += queued;
        }
        if (wanted != 0 && queued == wanted) {
            g_expert_pool_pinned_layer_queued[layer] = true;
        }
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

void ds4_gpu_expert_pool_auto_pin_layer(
        const void *model_map,
        uint32_t layer,
        uint32_t layer_cap) {
    if (!model_map || g_expert_pool_auto_pin_top == 0) return;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS || layer_cap <= DS4_METAL_ROUTED_TOPK) return;

    uint64_t layer_req = 0;
    for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
        layer_req += g_expert_pool_hot_count[layer][expert];
    }
    if (layer_req < g_expert_pool_auto_pin_min_req) return;
    if (g_expert_pool_auto_pin_interval != 0 &&
        layer_req < g_expert_pool_auto_pin_last_req[layer] + g_expert_pool_auto_pin_interval) {
        return;
    }
    g_expert_pool_auto_pin_last_req[layer] = layer_req;

    uint32_t max_pin = g_expert_pool_auto_pin_top;
    uint32_t reserve = g_expert_pool_pin_reserve;
    if (reserve < DS4_METAL_ROUTED_TOPK) reserve = DS4_METAL_ROUTED_TOPK;
    if (layer_cap > reserve) {
        const uint32_t cap_limit = layer_cap - reserve;
        if (max_pin > cap_limit) max_pin = cap_limit;
    } else {
        max_pin = 0;
    }
    if (max_pin == 0 || g_expert_pool_pinned_per_layer[layer] >= max_pin) return;

    bool chosen[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
    uint32_t added = 0;
    while (g_expert_pool_pinned_per_layer[layer] < max_pin) {
        int best = -1;
        uint64_t best_req = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            if (chosen[expert] || g_expert_pool_pinned[layer][expert]) continue;
            const uint64_t req = g_expert_pool_hot_count[layer][expert];
            if (req >= (uint64_t)g_expert_pool_auto_pin_min_req && req > best_req) {
                best_req = req;
                best = (int)expert;
            }
        }
        if (best < 0) break;
        chosen[(uint32_t)best] = true;
        if (ds4_gpu_expert_pool_add_pin(layer, (uint32_t)best, true)) {
            added++;
        }
    }
    if (added != 0) {
        uint32_t queued = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            if (!g_expert_pool_pinned[layer][expert]) continue;
            if (ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, expert)) queued++;
        }
        if (queued != 0) g_expert_pool_pinned_prefetches += queued;
        g_expert_pool_auto_pin_updates++;
    }
}

static bool ds4_gpu_expert_pool_choose_hotlock(
        uint32_t layer,
        uint32_t *experts_out,
        uint32_t *n_out) {
    if (!g_expert_pool_hotlock_enabled || !experts_out || !n_out) return false;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return false;
    if (g_expert_pool_slot_bytes == 0 || g_expert_pool_slots == 0) return false;
    uint32_t max_keep = g_expert_pool_hotlock_top;
    if (max_keep > g_expert_pool_slots) max_keep = g_expert_pool_slots;
    if (max_keep > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) max_keep = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
    bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
    uint32_t count = 0;
    while (count < max_keep) {
        int best = -1;
        uint64_t best_req = 0;
        for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
            if (printed[e]) continue;
            const uint64_t req = g_expert_pool_hot_count[layer][e];
            if (req > best_req) {
                best_req = req;
                best = (int)e;
            }
        }
        if (best < 0 || best_req == 0) break;
        printed[(uint32_t)best] = true;
        experts_out[count++] = (uint32_t)best;
    }
    *n_out = count;
    return count != 0;
}

void ds4_gpu_expert_pool_maybe_hotlock_layer(
        const void *model_map,
        uint32_t layer) {
    if (!g_expert_pool_hotlock_enabled || !model_map) return;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    uint32_t hot[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
    uint32_t n_hot = 0;
    if (!ds4_gpu_expert_pool_choose_hotlock(layer, hot, &n_hot)) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    for (uint32_t i = 0; i < n_hot; i++) {
        (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, hot[i]);
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}
