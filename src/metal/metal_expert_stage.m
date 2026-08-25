/* metal_expert_stage.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* Non-blocking token note from the graph thread: the exact hash-layer arming
 * spins on slot busy gates, so it runs on the prediction thread instead. */
static void ds4_gpu_expert_prefetch_enqueue_hash(int token) {
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_seq != g_pf_picked_seq) g_pf_superseded++;
    g_pf_slot.layer = 0;
    g_pf_slot.kind = 1;
    g_pf_slot.token = token;
    g_pf_seq++;
    g_pf_enqueued++;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mu);
}

ds4_metal_expert_gather_ctx * volatile g_rf_ctx;

pthread_mutex_t g_rf_mu = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t g_rf_cv = PTHREAD_COND_INITIALIZER;

volatile int g_rf_link_down;

ds4_metal_stage_slot g_stage[2];

volatile int g_stage_recent;

/* parity of the most recently armed slot */
uint64_t g_stage_hit_bytes_total, g_stage_armed, g_stage_fetched, g_stage_dropped;

static uint64_t g_stage_slots_total;

/* sum of armed expert counts (completion denominator) */

int ds4_gpu_expert_stage_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_STAGE") > 0 ? 1 : 0;
        /* Wave 33: staging needs working fetch connections to the peer, not
         * specifically the forward-dial client.  Accept mode (worker side:
         * peer dials in, DS4_DIST_EXPERT_FETCH_ACCEPT_PORT) provides the same
         * connections -- and the measured asymmetry was exactly this gate:
         * coordinator decode layers hit 50-100% staged RAM (hit_mib 20-40 of
         * 40.5) while worker decode ran 100% cold (hit_mib=0.0 all run). */
        if (cached && !getenv("DS4_DIST_EXPERT_FETCH_HOST") &&
            !getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT")) cached = 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: predicted experts staged from peer SSD into RAM one layer ahead\n");
        }
    }
    return cached;
}

/* Arm a staging slot for a freshly predicted layer.  Callable from the
 * prediction thread (score routing) and the graph thread (token-hash routing
 * hook), so the whole re-arm is serialized by a mutex. */
static pthread_mutex_t g_stage_arm_mu = PTHREAD_MUTEX_INITIALIZER;

void ds4_gpu_expert_stage_arm(uint32_t layer,
                                     const int *top_idx,
                                     uint32_t n_top,
                                     int token) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const ds4_metal_layer_router *r = &g_layer_router[layer];
    if (!r->valid) return;
    pthread_mutex_lock(&g_stage_arm_mu);
    ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    const uint64_t stride = 2u * r->gate_expert_bytes + r->down_expert_bytes;
    const size_t need = (size_t)stride * DS4_METAL_STAGE_MAX_EXPERTS;
    if (!s->buf || s->buf_cap < need) {
        free(s->buf);
        s->buf = (uint8_t *)malloc(need);
        s->buf_cap = s->buf ? need : 0;
        if (!s->buf) {
            pthread_mutex_unlock(&g_stage_arm_mu);
            return;
        }
    }
    s->gen++;                       /* invalidates all ready flags */
    __sync_synchronize();
    /* Wait out fetchers still writing this slot's buffer under the old gen:
     * a new-gen fetcher must never interleave writes with an in-flight recv
     * draining into the same region.  Off the gather critical path;
     * in-flight units finish in ~2ms. */
    while (s->busy) usleep(50);
    s->layer = layer;
    s->token = token;
    s->gate_off = r->gate_exps_off;
    s->up_off = r->up_exps_off;
    s->down_off = r->down_exps_off;
    s->gate_eb = r->gate_expert_bytes;
    s->down_eb = r->down_expert_bytes;
    s->stride = stride;
    uint32_t n = n_top;
    if (n > DS4_METAL_STAGE_MAX_EXPERTS) n = DS4_METAL_STAGE_MAX_EXPERTS;
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (top_idx[i] < 0) break;
        s->ids[m++] = (uint32_t)top_idx[i];
    }
    s->n = m;
    __sync_synchronize();
    s->next = 0;
    g_stage_recent = (int)(layer & 1u);
    g_stage_armed++;
    g_stage_slots_total += m;
    /* Wake the remote fetch threads parked on the shared condvar. */
    pthread_mutex_lock(&g_rf_mu);
    pthread_cond_broadcast(&g_rf_cv);
    pthread_mutex_unlock(&g_rf_mu);
    if ((g_stage_armed & 63u) == 0) {
        fprintf(stderr,
                "ds4-stage: armed=%llu fetched=%llu dropped=%llu hit_gib=%.2f "
                "(completion %.0f%%)\n",
                (unsigned long long)g_stage_armed,
                (unsigned long long)g_stage_fetched,
                (unsigned long long)g_stage_dropped,
                (double)g_stage_hit_bytes_total / 1073741824.0,
                g_stage_slots_total ? 100.0 * (double)g_stage_fetched /
                    (double)g_stage_slots_total : 0.0);
    }
    pthread_mutex_unlock(&g_stage_arm_mu);
}

/* Skip the re-arm when the slot already holds this (layer, token): the
 * token-hash hook and the prediction chain may both try to arm hash layers. */
void ds4_gpu_expert_stage_arm_if_new(uint32_t layer,
                                            const int *top_idx,
                                            uint32_t n_top,
                                            int token) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    if (s->gen != 0 && s->layer == layer && s->token == token) return;
    ds4_gpu_expert_stage_arm(layer, top_idx, n_top, token);
}

/* Prediction-thread side of the token note: arm exact staging for the
 * earliest hash layer of each slot parity; the prediction chain (also exact
 * for hash layers) covers the rest. */
void ds4_gpu_expert_hash_note_run(int token) {
    int armed_parity[2] = { 0, 0 };
    for (uint32_t l = 0; l < 8u && l < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; l++) {
        const ds4_metal_layer_router *r = &g_layer_router[l];
        if (!r->valid || r->hash_off == UINT64_MAX) continue;
        const int p = (int)(l & 1u);
        if (armed_parity[p]) continue;
        armed_parity[p] = 1;
        if (token < 0 || (uint32_t)token >= r->hash_rows) continue;
        const int32_t *row = (const int32_t *)((const uint8_t *)r->model_map + r->hash_off) +
                             (size_t)token * r->hash_k;
        int hidx[16];
        uint32_t m = 0;
        for (uint32_t i = 0; i < r->hash_k && i < 16u; i++) {
            if (row[i] >= 0 && (uint32_t)row[i] < r->n_expert) hidx[m++] = row[i];
        }
        if (m) {
            const uint32_t hgen = ++g_pf_pred_gen[l];
            for (uint32_t i = 0; i < m; i++) g_pf_pred_mark[l][(uint32_t)hidx[i]] = hgen;
            ds4_gpu_expert_stage_arm_if_new(l, hidx, m, token);
        }
        if (armed_parity[0] && armed_parity[1]) break;
    }
}

/* Graph-thread hook from ds4_gpu_router_select_tensor: the first hash-mode
 * router select of each decode forward reveals the token id, which exactly
 * determines the hash-routed layers' experts.  MUST NOT BLOCK: arming spins
 * on slot busy gates, so it is queued to the prediction thread (latest-wins
 * is correct here -- at a forward boundary any pending older job is stale). */
void ds4_gpu_expert_router_note(int token, int hash_mode) {
    if (!hash_mode) {
        g_pf_saw_nonhash = 1;
        return;
    }
    if (!g_pf_saw_nonhash) {
        g_pf_token = token;
        return;   /* later hash layer of the same forward */
    }
    g_pf_saw_nonhash = 0;
    g_pf_token = token;
    if (!ds4_gpu_expert_stage_enabled()) return;
    ds4_gpu_expert_prefetch_enqueue_hash(token);
}

/* Fetch-thread side: claim and stage one expert of slot s.  Returns 0 when
 * the slot has no more work.  The three tensor segments are pipelined on the
 * connection (send all, then recv all) so the server pread, the wire
 * transfer and the request RTT overlap: ~2.5ms per expert instead of ~5
 * (measured gap between 79.6% prediction accuracy and only 54% staged hits
 * was fetch completion, not prediction). */
int ds4_gpu_expert_stage_fetch_one(ds4_metal_stage_slot *s, int conn_slot) {
    const uint32_t gen = s->gen;
    const uint32_t i = __sync_fetch_and_add(&s->next, 1u);
    if (i >= s->n) return 0;
    __sync_fetch_and_add(&s->busy, 1u);
    if (s->gen != gen) {
        __sync_fetch_and_sub(&s->busy, 1u);
        return 0;
    }
    const uint64_t e = (uint64_t)s->ids[i];
    uint8_t *dst = s->buf + (size_t)i * s->stride;
    const uint64_t seg_off[3] = {
        s->gate_off + e * s->gate_eb,
        s->up_off + e * s->gate_eb,
        s->down_off + e * s->down_eb,
    };
    const uint64_t seg_len[3] = { s->gate_eb, s->gate_eb, s->down_eb };
    uint64_t dst_off = 0;
    int ok = 1;

    /* Locally cached experts never touch the link. */
    int all_cached = g_model_map_ptr != NULL;
    for (uint32_t k = 0; k < 3u && all_cached; k++) {
        all_cached = ds4_gpu_expert_range_mostly_cached(g_model_map_ptr, seg_off[k], seg_len[k]);
    }
    if (all_cached || g_rf_link_down) {
        for (uint32_t k = 0; k < 3u && ok; k++) {
            ok = ds4_gpu_pread_full(g_model_fd, dst + dst_off, seg_off[k], (size_t)seg_len[k]);
            dst_off += seg_len[k];
        }
    } else {
        /* Pipeline: send all three requests, then drain the responses. */
        uint32_t sent = 0;
        while (sent < 3u && ds4_dist_expert_fetch_send(conn_slot, seg_off[sent],
                                                       (uint32_t)seg_len[sent])) {
            sent++;
        }
        for (uint32_t k = 0; k < 3u; k++) {
            if (k < sent) {
                /* Must drain in order even if superseded: the connection's
                 * response stream stays aligned, and a stale write to this
                 * region is safe because re-arming waits on s->busy. */
                if (!ds4_dist_expert_fetch_recv(conn_slot, dst + dst_off, (uint32_t)seg_len[k])) {
                    if (!g_rf_link_down) {
                        g_rf_link_down = 1;
                        fprintf(stderr,
                                "ds4: expert remote fetch link down; gather continues local-only\n");
                    }
                    ok = ds4_gpu_pread_full(g_model_fd, dst + dst_off, seg_off[k],
                                            (size_t)seg_len[k]) && ok;
                }
            } else {
                ok = ds4_gpu_pread_full(g_model_fd, dst + dst_off, seg_off[k],
                                        (size_t)seg_len[k]) && ok;
            }
            dst_off += seg_len[k];
        }
    }
    if (ok && s->gen == gen) {
        __sync_synchronize();
        s->ready_gen[i] = gen;
        __sync_fetch_and_add(&g_stage_fetched, 1u);
    } else if (!ok || s->gen != gen) {
        __sync_fetch_and_add(&g_stage_dropped, 1u);
    }
    __sync_fetch_and_sub(&s->busy, 1u);
    return 1;
}

int ds4_gpu_expert_stage_has_work(void) {
    for (int k = 0; k < 2; k++) {
        if (g_stage[k].gen != 0 && g_stage[k].next < g_stage[k].n) return 1;
    }
    return 0;
}

/* Is (layer, expert) part of the armed prediction set (ready or not)?  Used
 * by the gather to decide whether a short wait for in-flight staged bytes
 * beats an immediate slow local read. */
int ds4_gpu_expert_stage_pending(uint32_t layer, uint32_t id) {
    const ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    if (s->gen == 0 || s->layer != layer) return 0;
    const uint32_t n = s->n;
    for (uint32_t i = 0; i < n && i < DS4_METAL_STAGE_MAX_EXPERTS; i++) {
        if (s->ids[i] == id) return 1;
    }
    return 0;
}

/* A staged-but-late expert finishes within ~2.5ms (pipelined fetch); local
 * cold pread costs ~2.7ms+ of the slow disk *and* its bandwidth.  Waiting a
 * bounded moment for in-flight bytes converts late completions into hits.
 * 0 disables. */
uint32_t ds4_gpu_expert_stage_wait_us(void) {
    static uint32_t cached = UINT32_MAX;
    if (cached == UINT32_MAX) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_STAGE_WAIT_US", 2500u);
        if (v > 20000u) v = 20000u;
        cached = (uint32_t)v;
    }
    return cached;
}

/* Gather side: return the staged bytes for (layer, expert, part) or NULL.
 * Safe by construction: the slot for layer L is only re-armed by the
 * prediction made during layer L+1's gather, after L's gather finished. */
const uint8_t *ds4_gpu_expert_stage_find(uint32_t layer,
                                                uint32_t id,
                                                uint32_t part,
                                                uint64_t *len_out) {
    const ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    const uint32_t gen = s->gen;
    if (gen == 0 || s->layer != layer) return NULL;
    for (uint32_t i = 0; i < s->n; i++) {
        if (s->ids[i] != id) continue;
        if (s->ready_gen[i] != gen) return NULL;
        const uint8_t *base = s->buf + (size_t)i * s->stride;
        switch (part) {
        case 0: *len_out = s->gate_eb; return base;
        case 1: *len_out = s->gate_eb; return base + s->gate_eb;
        default: *len_out = s->down_eb; return base + 2u * s->gate_eb;
        }
    }
    return NULL;
}

/* Accuracy accounting: compare this layer's actual active set against the
 * latest prediction for it (stats only; harmless races). */
void ds4_gpu_expert_prefetch_note_actual(uint32_t layer, const uint32_t *ids, uint32_t n) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const uint32_t gen = g_pf_pred_gen[layer];
    if (gen == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) continue;
        g_pf_pred_total++;
        if (g_pf_pred_mark[layer][ids[i]] == gen) g_pf_pred_hits++;
    }
}

static ds4_metal_expert_gather_pool g_expert_gather_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
};
