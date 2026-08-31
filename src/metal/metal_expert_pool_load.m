/* metal_expert_pool_load.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_try_load_layer_experts_to_pool(
        const void *model_map,
        uint32_t    layer_index,
        id<MTLBuffer> selectedbuf,
        NSUInteger  selected_off,
        uint32_t    n_picks,
        uint32_t    n_active,
        const uint32_t *active_ids,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total,
        id<MTLBuffer> *gate_buf,
        id<MTLBuffer> *up_buf,
        id<MTLBuffer> *down_buf,
        uint32_t    *source_n_total_expert) {
    if (!ds4_gpu_expert_pool_is_enabled() || !ds4_gpu_expert_pool_layer_allowed(layer_index) ||
        !model_map || !selectedbuf || !active_ids ||
        !gate_buf || !up_buf || !down_buf || !source_n_total_expert || n_active == 0) {
        return 0;
    }
    if (!ds4_gpu_expert_pool_init(gate_expert_bytes, down_expert_bytes)) return 0;
    ds4_gpu_expert_pool_clear_busy();
    uint32_t layer_cap = 0;
    if (!ds4_gpu_expert_pool_layer_served(layer_index, &layer_cap)) {
        if (!g_expert_pool_cycle_guard_reported) {
            const uint32_t requested = ds4_gpu_expert_pool_requested_layers();
            const uint32_t served = ds4_gpu_expert_pool_served_layers();
            fprintf(stderr,
                    "ds4: expert-pool per-layer LRU: %u slots cannot serve layer %u in configured range %u:%u "
                    "(serving tail %u/%u layers, min_slots/layer=%u); bypassing this layer.\n",
                    g_expert_pool_slots,
                    layer_index,
                    g_expert_pool_layer_start,
                    g_expert_pool_layer_end,
                    served,
                    requested,
                    ds4_gpu_expert_pool_min_layer_slots());
            g_expert_pool_cycle_guard_reported = true;
        }
        g_expert_pool_fallbacks++;
        return 0;
    }
    if (n_active > layer_cap) {
        g_expert_pool_fallbacks++;
        return 0;
    }
    if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        g_expert_pool_pinned_per_layer[layer_index] > layer_cap) {
        static bool warned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
        if (!warned[layer_index]) {
            fprintf(stderr,
                    "ds4: expert-pool pinned whitelist layer %u has %u experts but layer cap is %u; "
                    "LRU misses may fall back until --expert-pool-mb or min_layer_slots is increased.\n",
                    layer_index,
                    g_expert_pool_pinned_per_layer[layer_index],
                    layer_cap);
            warned[layer_index] = true;
        }
    }
    if (selectedbuf.storageMode != MTLStorageModeShared) return 0;

    int32_t pool_slots[DS4_METAL_ACTIVE_EXPERTS_MAX];
    if (n_active > DS4_METAL_ACTIVE_EXPERTS_MAX) return 0;
    if (!g_expert_pool_gate.contents || !g_expert_pool_up.contents || !g_expert_pool_down.contents) return 0;

    ds4_metal_expert_pool_meta meta = {
        .used = true,
        .model_map = model_map,
        .layer = layer_index,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .n_expert_total = n_expert_total,
    };
    ds4_gpu_expert_pool_register_layer_meta(model_map,
                                            layer_index,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes,
                                            n_expert_total);
    ds4_gpu_expert_pool_queue_pinned_layer(model_map, layer_index);

    if (g_expert_pool_hit_only) {
        bool all_ready = true;
        bool ready_flags[DS4_METAL_ACTIVE_EXPERTS_MAX] = { false };
        pthread_mutex_lock(&g_expert_pool_mu);
        for (uint32_t i = 0; i < n_active; i++) {
            const uint32_t expert = active_ids[i];
            if (expert >= n_expert_total) {
                all_ready = false;
                continue;
            }
            const int32_t slot = ds4_gpu_expert_pool_find(model_map, layer_index, expert);
            ds4_metal_expert_pool_entry *entry = slot >= 0 ? &g_expert_pool_entries[slot] : NULL;
            ready_flags[i] = entry && entry->ready;
            if (!ready_flags[i]) {
                all_ready = false;
                if (g_expert_pool_prefetch_queue) {
                    (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer_index, expert);
                }
            }
        }
        if (!all_ready) {
            for (uint32_t i = 0; i < n_active; i++) {
                const uint32_t expert = active_ids[i];
                if (expert >= n_expert_total) continue;
                const int request_pinned = ds4_gpu_expert_pool_is_pinned(layer_index, expert);
                g_expert_pool_requests++;
                if (request_pinned) g_expert_pool_pinned_requests++;
                if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                    expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                    g_expert_pool_hot_count[layer_index][expert]++;
                    ds4_gpu_expert_pool_auto_pin_layer(model_map, layer_index, layer_cap);
                }
                if (ready_flags[i]) {
                    const int32_t slot = ds4_gpu_expert_pool_find(model_map, layer_index, expert);
                    if (slot >= 0) {
                        g_expert_pool_hits++;
                        if (request_pinned) g_expert_pool_pinned_hits++;
                        g_expert_pool_entries[slot].last_used = ++g_expert_pool_clock;
                        ds4_gpu_expert_pool_lru_touch(slot);
                    }
                } else {
                    g_expert_pool_misses++;
                    if (request_pinned) g_expert_pool_pinned_misses++;
                }
            }
            g_expert_pool_admission_skips++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            return 0;
        }
        pthread_mutex_unlock(&g_expert_pool_mu);
    }

    double copy_t0 = 0.0;
    bool copied_any = false;
    int32_t marked_busy[DS4_METAL_ACTIVE_EXPERTS_MAX];
    uint32_t n_marked_busy = 0;

    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t expert = active_ids[i];
        if (expert >= n_expert_total) {
            ds4_gpu_expert_pool_clear_busy();
            return 0;
        }

        bool counted_request = false;
        bool request_pinned = false;
        for (;;) {
            int32_t slot = -1;
            bool need_sync_copy = false;
            bool wait_for_prefetch = false;

            pthread_mutex_lock(&g_expert_pool_mu);
            if (!counted_request) {
                g_expert_pool_requests++;
                const int was_pinned = ds4_gpu_expert_pool_is_pinned(layer_index, expert);
                if (was_pinned) {
                    g_expert_pool_pinned_requests++;
                    request_pinned = true;
                }
                if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                    expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                    g_expert_pool_hot_count[layer_index][expert]++;
                    ds4_gpu_expert_pool_auto_pin_layer(model_map, layer_index, layer_cap);
                }
                counted_request = true;
            }

            slot = ds4_gpu_expert_pool_find(model_map, layer_index, expert);
            if (slot >= 0) {
                ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                if (e->ready) {
                    g_expert_pool_hits++;
                    if (request_pinned) g_expert_pool_pinned_hits++;
                    e->busy = true;
                    e->last_used = ++g_expert_pool_clock;
                    ds4_gpu_expert_pool_lru_touch(slot);
                    pool_slots[i] = slot;
                    if (n_marked_busy < DS4_METAL_ACTIVE_EXPERTS_MAX) marked_busy[n_marked_busy++] = slot;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    break;
                }
                if (e->loading) {
                    g_expert_pool_inflight_hits++;
                    if (g_expert_pool_wait_inflight) {
                        wait_for_prefetch = true;
                    } else {
                        pthread_mutex_unlock(&g_expert_pool_mu);
                        ds4_gpu_expert_pool_clear_busy();
                        return 0;
                    }
                } else {
                    /* Stale half-entry: evict and re-copy synchronously below. */
                    ds4_gpu_expert_pool_lru_unlink(slot);
                    memset(e, 0, sizeof(*e));
                    e->prev = -1;
                    e->next = -1;
                    slot = -1;
                }
            }

            if (wait_for_prefetch) {
                const double wait_t0 = ds4_gpu_now_ms();
                g_expert_pool_sync_waits++;
                while (slot >= 0) {
                    ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                    if (!(e->used && e->model_map == model_map && e->layer == layer_index &&
                          e->expert == expert && e->loading && !e->ready)) {
                        break;
                    }
                    pthread_cond_wait(&g_expert_pool_cv, &g_expert_pool_mu);
                }
                g_expert_pool_wait_ms += ds4_gpu_now_ms() - wait_t0;
                pthread_mutex_unlock(&g_expert_pool_mu);
                continue;
            }

            if (slot < 0) {
                if (!g_expert_pool_foreground_fill) {
                    if (g_expert_pool_prefetch_self && g_expert_pool_prefetch_top != 0) {
                        uint32_t self_n = n_active;
                        if (self_n > g_expert_pool_prefetch_top) self_n = g_expert_pool_prefetch_top;
                        for (uint32_t j = 0; j < self_n; j++) {
                            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map,
                                                                                layer_index,
                                                                                active_ids[j]);
                        }
                    }
                    g_expert_pool_admission_skips++;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    ds4_gpu_expert_pool_clear_busy();
                    return 0;
                }
                if (g_expert_pool_admit_after > 1u && layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                    expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS &&
                    g_expert_pool_hot_count[layer_index][expert] < g_expert_pool_admit_after) {
                    if (g_expert_pool_prefetch_self && g_expert_pool_prefetch_top != 0) {
                        uint32_t self_n = n_active;
                        if (self_n > g_expert_pool_prefetch_top) self_n = g_expert_pool_prefetch_top;
                        for (uint32_t j = 0; j < self_n; j++) {
                            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map,
                                                                                layer_index,
                                                                                active_ids[j]);
                        }
                    }
                    for (uint32_t j = 0; j < n_active; j++) {
                        if (j == i) continue;
                        const uint32_t other = active_ids[j];
                        if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                            other < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                            g_expert_pool_requests++;
                            g_expert_pool_hot_count[layer_index][other]++;
                        }
                    }
                    g_expert_pool_admission_skips++;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    ds4_gpu_expert_pool_clear_busy();
                    return 0;
                }
                slot = ds4_gpu_expert_pool_victim(model_map, layer_index, layer_cap, active_ids, n_active);
                if (slot < 0) {
                    g_expert_pool_fallbacks++;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    ds4_gpu_expert_pool_clear_busy();
                    return 0;
                }
                ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                e->used = true;
                e->ready = false;
                e->loading = true;
                e->busy = false;
                e->model_map = model_map;
                e->layer = layer_index;
                e->expert = expert;
                e->pinned = ds4_gpu_expert_pool_is_pinned(layer_index, expert);
                e->last_used = ++g_expert_pool_clock;
                ds4_gpu_expert_pool_lru_link_head(slot);
                need_sync_copy = true;
                g_expert_pool_misses++;
                if (request_pinned) g_expert_pool_pinned_misses++;
                pthread_mutex_unlock(&g_expert_pool_mu);
            } else {
                pthread_mutex_unlock(&g_expert_pool_mu);
            }

            if (need_sync_copy) {
                if (!copied_any) {
                    copy_t0 = ds4_gpu_now_ms();
                    copied_any = true;
                }
                const double one_t0 = ds4_gpu_now_ms();
                const int ok = ds4_gpu_expert_pool_copy_slot(slot, &meta, expert);
                const double one_dt = ds4_gpu_now_ms() - one_t0;
                (void)one_dt;

                pthread_mutex_lock(&g_expert_pool_mu);
                ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                if (e->used && e->model_map == model_map && e->layer == layer_index &&
                    e->expert == expert && e->loading) {
                    if (ok) {
                        e->ready = true;
                        e->loading = false;
                        e->busy = true;
                        e->last_used = ++g_expert_pool_clock;
                        ds4_gpu_expert_pool_lru_touch(slot);
                        pool_slots[i] = slot;
                        if (n_marked_busy < DS4_METAL_ACTIVE_EXPERTS_MAX) marked_busy[n_marked_busy++] = slot;
                        g_expert_pool_miss_copy_bytes += g_expert_pool_slot_bytes;
                        pthread_cond_broadcast(&g_expert_pool_cv);
                        pthread_mutex_unlock(&g_expert_pool_mu);
                        break;
                    } else {
                        ds4_gpu_expert_pool_lru_unlink(slot);
                        memset(e, 0, sizeof(*e));
                        e->prev = -1;
                        e->next = -1;
                        pthread_cond_broadcast(&g_expert_pool_cv);
                        pthread_mutex_unlock(&g_expert_pool_mu);
                        ds4_gpu_expert_pool_clear_busy();
                        return 0;
                    }
                }
                pthread_mutex_unlock(&g_expert_pool_mu);
                continue;
            }
        }
    }
    if (copied_any) g_expert_pool_miss_copy_ms += ds4_gpu_now_ms() - copy_t0;

    int32_t *sel_cpu = (int32_t *)((uint8_t *)selectedbuf.contents + (size_t)selected_off);
    for (uint32_t i = 0; i < n_picks; i++) {
        int32_t compact = sel_cpu[i];
        if (compact < 0 || (uint32_t)compact >= n_active) {
            ds4_gpu_expert_pool_clear_busy();
            return 0;
        }
        sel_cpu[i] = pool_slots[(uint32_t)compact];
    }

    *gate_buf = g_expert_pool_gate;
    *up_buf = g_expert_pool_up;
    *down_buf = g_expert_pool_down;
    *source_n_total_expert = g_expert_pool_slots;
    g_expert_pool_calls++;
    if (g_expert_pool_hotlist_interval != 0 &&
        g_expert_pool_hotlist_top != 0 &&
        (g_expert_pool_calls % g_expert_pool_hotlist_interval) == 0) {
        ds4_gpu_expert_pool_print_hotlist("live", g_expert_pool_hotlist_top, false);
    }

    /* Keep the LRU pool mandatory, but overlap future misses: current-layer active
     * IDs are a strong predictor for the next layer in this hash-routed q2 GGUF,
     * and hotlock (when enabled) keeps the measured top experts resident. */
    ds4_gpu_expert_pool_predict_enqueue(model_map, layer_index, active_ids, n_active);
    ds4_gpu_expert_pool_maybe_hotlock_layer(model_map, layer_index);
    (void)marked_busy;
    (void)n_marked_busy;
    return 1;
}
