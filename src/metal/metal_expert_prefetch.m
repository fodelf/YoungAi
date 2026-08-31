/* metal_expert_prefetch.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* Prediction margin: top-8 predicted experts are read ahead per next layer. */
int ds4_gpu_expert_prefetch_top(void) {
    return 8;
}

/* Predict a RANGE of upcoming layers (L+1 .. L+depth) from the same hidden
 * snapshot.  The paced read-ahead can only use SSD-idle windows (~drain time
 * per layer); depth D gives each layer's prediction up to D windows to finish
 * before its gather arrives.  Closest deadline (L+1) is advised first, and
 * mincore skips ranges an earlier job already pulled in, so deeper lookahead
 * degrades gracefully.  Accuracy cost is measurable via the ds4-io pf= field
 * (predictions for L+d use a hidden state that is d-1 layers stale). */
uint32_t ds4_gpu_expert_prefetch_depth(void) {
    return 1u;
}

/* Scheduling (v2, after the 1.56->1.49 regression): decode keeps the SSD
 * ~100% busy, so read-ahead must never compete with the foreground gather --
 * the first version did, and the extra traffic cost more than the overlap
 * gained.  v2 is "polite": the prefetch thread only issues advisories in
 * 1MiB chunks while no foreground gather is running (g_gather_active), skips
 * ranges that are already page-cache resident (mincore), works in router
 * score order so the scarce idle window is spent on the most likely experts,
 * and a newer prediction supersedes an unfinished older one (latest-wins). */
pthread_mutex_t g_pf_mu = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t g_pf_cv = PTHREAD_COND_INITIALIZER;

ds4_metal_pf_job g_pf_slot;

/* single slot, latest wins */
volatile uint32_t g_pf_seq;

/* bumped per enqueue */
uint32_t g_pf_picked_seq;

/* last seq picked by the worker */
int g_pf_shutdown;

uint64_t g_pf_enqueued, g_pf_superseded, g_pf_advise_fail, g_pf_skip_cached;

volatile int g_gather_active;

/* foreground gather/stream running */
/* Prediction sets for accuracy accounting: mark[layer][expert] == gen[layer]
 * means "expert was in the latest predicted set for that layer".  Written by
 * the prefetch thread, read by the gather entry; stats-only, races benign. */
uint32_t g_pf_pred_gen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

uint32_t g_pf_pred_mark[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];

/* Current decode token (router hook).  Hash-routed early layers' expert sets
 * are an exact function of this id. */
volatile int g_pf_token = -1;

int g_pf_saw_nonhash = 1;

float ds4_gpu_pf_softplus(float v) {
    if (v > 20.0f) return v;
    if (v < -20.0f) return expf(v);
    return log1pf(expf(v));
}

static void ds4_gpu_expert_prefetch_advise(int fd, uint64_t off, uint64_t len) {
    struct radvisory ra;
    ra.ra_offset = (off_t)off;
    ra.ra_count = (int)len;
    if (fcntl(fd, F_RDADVISE, &ra) != -1) return;
    g_pf_advise_fail++;
    /* Portable fallback: pull the range through the page cache with plain
     * preads into a throwaway buffer (single prefetch thread => static ok). */
    static uint8_t scratch[262144];
    uint64_t done = 0;
    while (done < len) {
        size_t chunk = sizeof(scratch);
        if ((uint64_t)chunk > len - done) chunk = (size_t)(len - done);
        if (!ds4_gpu_pread_full(fd, scratch, off + done, chunk)) break;
        done += chunk;
    }
}

/* Issue read-ahead in small chunks, yielding whenever a foreground gather is
 * running and aborting as soon as a newer prediction supersedes this one.
 * Returns 0 on abort. */
int ds4_gpu_expert_prefetch_advise_paced(int fd, uint64_t off, uint64_t len, uint32_t my_seq) {
    const uint64_t chunk = 1ull << 20;
    uint64_t done = 0;
    while (done < len) {
        while (g_gather_active) {
            if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
            usleep(200);
        }
        if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
        uint64_t n = chunk;
        if (len - done < n) n = len - done;
        ds4_gpu_expert_prefetch_advise(fd, off + done, n);
        done += n;
    }
    return 1;
}

/* True when (almost) the whole file range is already page-cache resident, in
 * which case read-ahead would only waste the idle window.  Ground truth via
 * mincore on the model mapping (same UBC pages the preads hit). */
int ds4_gpu_expert_range_mostly_cached(const void *model_map, uint64_t off, uint64_t len) {
    char vec[8192];          /* stack: called from prefetch + remote fetch threads */
    static size_t page;
    if (page == 0) {
        long p = sysconf(_SC_PAGESIZE);
        page = p > 0 ? (size_t)p : 16384u;
    }
    const uintptr_t addr = (uintptr_t)model_map + (uintptr_t)off;
    const uintptr_t start = addr & ~(uintptr_t)(page - 1u);
    const size_t span = (size_t)(addr + len - start);
    const size_t npages = (span + page - 1u) / page;
    if (npages == 0 || npages > sizeof(vec)) return 0;
    if (mincore((void *)start, span, vec) != 0) return 0;
    size_t resident = 0;
    for (size_t i = 0; i < npages; i++) resident += (size_t)(vec[i] & 1);
    return resident * 10u >= npages * 9u;   /* >=90% resident: skip */
}

/* Predict one layer, then either hand the predicted experts to the remote
 * staging pipeline (stage_this: pulled from the peer's SSD into RAM during
 * this layer's window) or issue local paced read-ahead advisories.  Returns 0
 * when superseded/shutting down. */
int ds4_gpu_expert_prefetch_predict_one(uint32_t layer, const float *x, uint32_t my_seq,
                                               int stage_this) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return 1;
    const ds4_metal_layer_router *r = &g_layer_router[layer];
    if (!r->valid || g_model_fd < 0) return 1;
    /* Token-id hash routing (early layers): the expert set is an exact
     * function of the current token -- 100% "prediction" accuracy. */
    if (r->hash_off != UINT64_MAX) {
        const int tok = g_pf_token;
        if (tok < 0 || (uint32_t)tok >= r->hash_rows) return 1;
        const int32_t *row = (const int32_t *)((const uint8_t *)r->model_map + r->hash_off) +
                             (size_t)tok * r->hash_k;
        int hidx[16];
        uint32_t m = 0;
        for (uint32_t i = 0; i < r->hash_k && i < 16u; i++) {
            if (row[i] >= 0 && (uint32_t)row[i] < r->n_expert) hidx[(int)m++] = row[i];
        }
        if (m == 0) return 1;
        const uint32_t hgen = ++g_pf_pred_gen[layer];
        for (uint32_t i = 0; i < m; i++) g_pf_pred_mark[layer][(uint32_t)hidx[i]] = hgen;
        if (stage_this && ds4_gpu_expert_stage_enabled()) {
            ds4_gpu_expert_stage_arm_if_new(layer, hidx, m, tok);
            return 1;
        }
        const int hfd = g_model_fd;
        for (uint32_t i = 0; i < m; i++) {
            const uint64_t e = (uint64_t)hidx[i];
            const uint64_t hsoff[3] = {
                r->gate_exps_off + e * r->gate_expert_bytes,
                r->up_exps_off + e * r->gate_expert_bytes,
                r->down_exps_off + e * r->down_expert_bytes,
            };
            const uint64_t hslen[3] = { r->gate_expert_bytes, r->gate_expert_bytes,
                                        r->down_expert_bytes };
            for (uint32_t s = 0; s < 3u; s++) {
                if (ds4_gpu_expert_range_mostly_cached(r->model_map, hsoff[s], hslen[s])) {
                    g_pf_skip_cached++;
                    continue;
                }
                if (!ds4_gpu_expert_prefetch_advise_paced(hfd, hsoff[s], hslen[s], my_seq)) {
                    return 0;
                }
            }
        }
        return 1;
    }
    const uint8_t *map = (const uint8_t *)r->model_map;
    const __fp16 *w16 = (const __fp16 *)(map + r->gate_inp_off);
    const float *w32 = (const float *)(map + r->gate_inp_off);
    const float *bias = r->probs_bias_off != UINT64_MAX ?
        (const float *)(map + r->probs_bias_off) : NULL;
    const uint32_t n_top = (uint32_t)ds4_gpu_expert_prefetch_top();
    /* Opportunistic staging depth: the top-n_top experts saturate the link
     * budget; a couple of extra candidates ride the idle tail of the window
     * (fetched last, in score order) and convert some prediction misses. */
    /* Stage 2 extras beyond the top set: cheap hedge against routing jitter. */
    uint32_t n_top_total = n_top + 2u;
    if (n_top_total > DS4_METAL_PF_MAX_TOP) n_top_total = DS4_METAL_PF_MAX_TOP;

    /* score = sqrt(softplus(gate_inp . x)) + bias, matching the selection rule
     * in layer_topk_selected_experts (ranking only; weights don't matter). */
    int top_idx[DS4_METAL_PF_MAX_TOP];
    float top_score[DS4_METAL_PF_MAX_TOP];
    for (uint32_t i = 0; i < n_top_total; i++) { top_idx[i] = -1; top_score[i] = -1e30f; }
    for (uint32_t e = 0; e < r->n_expert; e++) {
        float acc = 0.0f;
        if (r->gate_inp_is_f32) {
            const float *row = w32 + (size_t)e * r->n_embd;
            for (uint32_t d = 0; d < r->n_embd; d++) acc += row[d] * x[d];
        } else {
            const __fp16 *row = w16 + (size_t)e * r->n_embd;
            for (uint32_t d = 0; d < r->n_embd; d++) acc += (float)row[d] * x[d];
        }
        float s = sqrtf(ds4_gpu_pf_softplus(acc));
        if (bias) s += bias[e];
        if (s <= top_score[n_top_total - 1u]) continue;
        uint32_t j = n_top_total - 1u;
        while (j > 0u && s > top_score[j - 1u]) {
            top_score[j] = top_score[j - 1u];
            top_idx[j] = top_idx[j - 1u];
            j--;
        }
        top_score[j] = s;
        top_idx[j] = (int)e;
    }

    /* Mark the headline predicted set first (prediction-accuracy accounting
     * stays comparable across TOP settings), then stage/advise in descending
     * score order; abort mid-way once a newer prediction lands. */
    const uint32_t gen = ++g_pf_pred_gen[layer];
    for (uint32_t i = 0; i < n_top; i++) {
        if (top_idx[i] < 0) break;
        g_pf_pred_mark[layer][(uint32_t)top_idx[i]] = gen;
    }
    if (stage_this && ds4_gpu_expert_stage_enabled()) {
        ds4_gpu_expert_stage_arm(layer, top_idx, n_top_total, g_pf_token);
        return 1;   /* fetch threads stage from the peer; no local advisories */
    }
    const int fd = g_model_fd;   /* read-ahead must hit the page cache, never the F_NOCACHE fd */
    for (uint32_t i = 0; i < n_top; i++) {
        if (top_idx[i] < 0) break;
        const uint64_t e = (uint64_t)top_idx[i];
        const uint64_t seg_off[3] = {
            r->gate_exps_off + e * r->gate_expert_bytes,
            r->up_exps_off + e * r->gate_expert_bytes,
            r->down_exps_off + e * r->down_expert_bytes,
        };
        const uint64_t seg_len[3] = {
            r->gate_expert_bytes, r->gate_expert_bytes, r->down_expert_bytes,
        };
        for (uint32_t s = 0; s < 3u; s++) {
            if (ds4_gpu_expert_range_mostly_cached(r->model_map, seg_off[s], seg_len[s])) {
                g_pf_skip_cached++;
                continue;
            }
            if (!ds4_gpu_expert_prefetch_advise_paced(fd, seg_off[s], seg_len[s], my_seq)) {
                return 0;   /* superseded or shutting down */
            }
        }
    }
    return 1;
}
