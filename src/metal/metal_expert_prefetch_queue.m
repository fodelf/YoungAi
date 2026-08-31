/* metal_expert_prefetch_queue.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* Wave 36: batch-union prediction.  Per sampled row, score all experts with
 * the same selection rule as predict_one, keep each row's top-k, and union
 * them (per-expert max score orders the advisories).  Local advisories only:
 * a 50-70 expert union (~400MiB) belongs to the local fast path -- the
 * 16-slot peer staging machinery cannot hold it and TB cannot deliver it in
 * one drain window.  The advisories flow while g_gather_active is clear,
 * i.e. exactly inside the next layer's GPU drain (~50-60ms, disk idle), so
 * the following gather's cached-fd preads hit warm pages.  Wrong predictions
 * cost only wasted read-ahead; correctness is untouched. */
static int ds4_gpu_expert_prefetch_predict_union(uint32_t layer,
                                                 const float *rows,
                                                 uint32_t n_rows,
                                                 uint32_t my_seq) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return 1;
    const ds4_metal_layer_router *r = &g_layer_router[layer];
    if (!r->valid || g_model_fd < 0) return 1;
    if (r->hash_off != UINT64_MAX) return 1;  /* hash routing is per token id */
    if (r->n_expert == 0 || r->n_expert > 1024u) return 1;
    const uint8_t *map = (const uint8_t *)r->model_map;
    const __fp16 *w16 = (const __fp16 *)(map + r->gate_inp_off);
    const float *w32 = (const float *)(map + r->gate_inp_off);
    const float *bias = r->probs_bias_off != UINT64_MAX ?
        (const float *)(map + r->probs_bias_off) : NULL;
    const uint32_t row_top = (uint32_t)ds4_gpu_expert_prefetch_top();

    float best[1024];
    for (uint32_t e = 0; e < r->n_expert; e++) best[e] = -1e30f;

    for (uint32_t ri = 0; ri < n_rows; ri++) {
        if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
        const float *x = rows + (size_t)ri * DS4_METAL_PF_MAX_EMBD;
        int top_idx[DS4_METAL_PF_MAX_TOP];
        float top_score[DS4_METAL_PF_MAX_TOP];
        uint32_t k = row_top;
        if (k > DS4_METAL_PF_MAX_TOP) k = DS4_METAL_PF_MAX_TOP;
        for (uint32_t i = 0; i < k; i++) { top_idx[i] = -1; top_score[i] = -1e30f; }
        for (uint32_t e = 0; e < r->n_expert; e++) {
            float acc = 0.0f;
            if (r->gate_inp_is_f32) {
                const float *row = w32 + (size_t)e * r->n_embd;
                for (uint32_t dd = 0; dd < r->n_embd; dd++) acc += row[dd] * x[dd];
            } else {
                const __fp16 *row = w16 + (size_t)e * r->n_embd;
                for (uint32_t dd = 0; dd < r->n_embd; dd++) acc += (float)row[dd] * x[dd];
            }
            float s = sqrtf(ds4_gpu_pf_softplus(acc));
            if (bias) s += bias[e];
            if (s <= top_score[k - 1u]) continue;
            uint32_t j = k - 1u;
            while (j > 0u && s > top_score[j - 1u]) {
                top_score[j] = top_score[j - 1u];
                top_idx[j] = top_idx[j - 1u];
                j--;
            }
            top_score[j] = s;
            top_idx[j] = (int)e;
        }
        for (uint32_t i = 0; i < k; i++) {
            if (top_idx[i] < 0) break;
            if (top_score[i] > best[(uint32_t)top_idx[i]]) {
                best[(uint32_t)top_idx[i]] = top_score[i];
            }
        }
    }

    /* Advise the union in descending score order: the idle window is spent on
     * the most confident experts first. */
    const int fd = g_model_fd;
    for (;;) {
        if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
        float bs = -1e29f;
        int be = -1;
        for (uint32_t e = 0; e < r->n_expert; e++) {
            if (best[e] > bs) { bs = best[e]; be = (int)e; }
        }
        if (be < 0 || bs <= -1e29f) break;
        best[be] = -1e30f;
        const uint64_t eo = (uint64_t)be;
        const uint64_t seg_off[3] = {
            r->gate_exps_off + eo * r->gate_expert_bytes,
            r->up_exps_off + eo * r->gate_expert_bytes,
            r->down_exps_off + eo * r->down_expert_bytes,
        };
        const uint64_t seg_len[3] = { r->gate_expert_bytes, r->gate_expert_bytes,
                                      r->down_expert_bytes };
        for (uint32_t s = 0; s < 3u; s++) {
            if (ds4_gpu_expert_range_mostly_cached(r->model_map, seg_off[s], seg_len[s])) {
                g_pf_skip_cached++;
                continue;
            }
            if (!ds4_gpu_expert_prefetch_advise_paced(fd, seg_off[s], seg_len[s], my_seq)) {
                return 0;
            }
        }
    }
    return 1;
}

/* job->layer is the first predicted layer (L+1); predict L+1..L+depth from
 * the same hidden snapshot, closest deadline first.  Only d==0 is staged:
 * its slot is never re-armed while its layer's gather can still read it.
 * kind==1 jobs carry only a token id and arm the exact hash-routed layers
 * (moved off the graph thread: arming spins on the slot's busy gate). */
static void ds4_gpu_expert_prefetch_run_job(const ds4_metal_pf_job *job, uint32_t my_seq) {
    if (job->kind == 1) {
        ds4_gpu_expert_hash_note_run(job->token);
        return;
    }
    if (job->n_rows > 1u) {
        /* Batch union job: depth 1 only (deeper layers from the same snapshot
         * degrade, and the window barely fits one union). */
        (void)ds4_gpu_expert_prefetch_predict_union(job->layer, job->xrows,
                                                    job->n_rows, my_seq);
        return;
    }
    const uint32_t depth = ds4_gpu_expert_prefetch_depth();
    for (uint32_t d = 0; d < depth; d++) {
        if (g_pf_seq != my_seq || g_pf_shutdown) return;
        if (!ds4_gpu_expert_prefetch_predict_one(job->layer + d, job->x, my_seq, d == 0u)) return;
    }
}

static void *ds4_gpu_expert_prefetch_thread(void *arg) {
    (void)arg;
    static ds4_metal_pf_job local;
    uint32_t my_seq = 0;
    for (;;) {
        pthread_mutex_lock(&g_pf_mu);
        while (!g_pf_shutdown && g_pf_seq == my_seq) {
            pthread_cond_wait(&g_pf_cv, &g_pf_mu);
        }
        if (g_pf_shutdown) {
            pthread_mutex_unlock(&g_pf_mu);
            break;
        }
        my_seq = g_pf_seq;
        g_pf_picked_seq = my_seq;
        memcpy(&local, &g_pf_slot, sizeof(local));
        pthread_mutex_unlock(&g_pf_mu);
        ds4_gpu_expert_prefetch_run_job(&local, my_seq);
    }
    return NULL;
}

/* Called from the serial graph thread only (no init race). */
int ds4_gpu_expert_prefetch_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        /* Auto-adapts to the offload verdict (resident models never need
         * read-ahead) — EXCEPT while capture/eval instrumentation is armed:
         * 捕获铁律(2026-06-24)=取料必须可复现, 预取线程引入非决定性与 OOM 风
         * 险, 所以 --cap-dir/--eval-ids 在场时强制关。 */
        const char *cap = ds4_tool_cap_dir();
        const char *ids = ds4_tool_eval_ids();
        if ((cap && cap[0]) || (ids && ids[0])) {
            cached = 0;
        } else {
            cached = ds4_gpu_expert_offload_enabled() ? 1 : 0;
        }
        if (cached && g_model_fd < 0) {
            fprintf(stderr,
                    "ds4: expert prefetch needs a model fd; disabled\n");
            cached = 0;
        }
        if (cached) {
            pthread_t th;
            if (pthread_create(&th, NULL, ds4_gpu_expert_prefetch_thread, NULL) == 0) {
                fprintf(stderr,
                        "ds4: cross-layer expert prefetch enabled (top %d predicted experts "
                        "read ahead per next layer)\n",
                        ds4_gpu_expert_prefetch_top());
            } else {
                cached = 0;
            }
        }
    }
    return cached;
}

void ds4_gpu_expert_prefetch_enqueue(uint32_t next_layer, const float *x, uint32_t n_embd) {
    if (next_layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const ds4_metal_layer_router *r = &g_layer_router[next_layer];
    if (!r->valid || r->n_embd != n_embd) return;
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_seq != g_pf_picked_seq) g_pf_superseded++;   /* old job replaced unstarted/aborted */
    g_pf_slot.layer = next_layer;
    g_pf_slot.kind = 0;
    g_pf_slot.token = g_pf_token;
    g_pf_slot.n_rows = 1;
    memcpy(g_pf_slot.x, x, (size_t)n_embd * sizeof(float));
    g_pf_seq++;
    g_pf_enqueued++;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mu);
}

/* Wave 36: batch snapshot.  x is the row-major [n_tokens x n_embd] FFN input
 * of a verify batch; stride-sample up to DS4_METAL_PF_MAX_ROWS rows so the
 * union prediction sees the whole token range.  Row sampling keeps the
 * per-job score cost ~16 router matvecs (~15-25ms on the prefetch thread,
 * inside the gather+drain window). */
void ds4_gpu_expert_prefetch_enqueue_batch(uint32_t next_layer,
                                                  const float *x,
                                                  uint32_t n_embd,
                                                  uint32_t n_tokens) {
    if (next_layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    if (!x || n_tokens < 2u) return;
    const ds4_metal_layer_router *r = &g_layer_router[next_layer];
    if (!r->valid || r->n_embd != n_embd) return;
    uint32_t n_rows = n_tokens;
    if (n_rows > DS4_METAL_PF_MAX_ROWS) n_rows = DS4_METAL_PF_MAX_ROWS;
    uint32_t stride = n_tokens / n_rows;
    if (stride == 0) stride = 1;
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_seq != g_pf_picked_seq) g_pf_superseded++;
    g_pf_slot.layer = next_layer;
    g_pf_slot.kind = 0;
    g_pf_slot.token = g_pf_token;
    uint32_t m = 0;
    for (uint32_t t = 0; t < n_tokens && m < n_rows; t += stride, m++) {
        memcpy(g_pf_slot.xrows + (size_t)m * DS4_METAL_PF_MAX_EMBD,
               x + (size_t)t * n_embd,
               (size_t)n_embd * sizeof(float));
    }
    g_pf_slot.n_rows = m;
    g_pf_seq++;
    g_pf_enqueued++;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mu);
}
