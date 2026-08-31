/* metal_expert.h — 专家 offload 子系统共享 typedef/extern/原型(机械搬移)。仅经 metal_internal.h 包含。 */
#ifndef DS4_METAL_EXPERT_H
#define DS4_METAL_EXPERT_H

#define DS4_METAL_EXPERT_PROFILE_MAX_LAYERS 128u

#define DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS 1024u

typedef struct {
    bool        used;
    bool        ready;
    bool        loading;
    bool        busy;
    bool        pinned;
    const void *model_map;
    uint32_t    layer;
    uint32_t    expert;
    int32_t     prev;
    int32_t     next;
    uint64_t    last_used;
} ds4_metal_expert_pool_entry;

typedef struct {
    bool        used;
    const void *model_map;
    uint32_t    layer;
    uint32_t    expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
    uint32_t    n_expert_total;
} ds4_metal_expert_pool_meta;

typedef struct {
    bool        used;
    const void *model_map;
    uint32_t    layer;
    uint32_t    expert;
} ds4_metal_expert_prefetch_req;

#define DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX 64u

/* ---- project.md P2.1: cross-layer router prediction + async expert prefetch --
 *
 * Decode is a serial chain: gather(L) -> GPU(L) -> drain -> gather(L+1) ...
 * The router of layer L+1 is a tiny F16 [n_expert][n_embd] matrix, and the
 * residual stream changes slowly across one layer, so evaluating router(L+1)
 * on layer L's MoE input already ranks the true top-6 with high overlap
 * (Pre-Attention Expert Prediction, arXiv:2511.10676).  While the GPU computes
 * layer L we re-evaluate router(L+1) on a background thread and issue
 * F_RDADVISE read-ahead for the predicted experts' gate/up/down ranges, so by
 * the time gather(L+1) preads them they are (partially) page-cache hits.
 * Predictions are purely advisory: a miss only wastes read bandwidth, the
 * gather path stays bit-exact.  Follows the expert-offload verdict (resident
 * models never need it); forced off while capture/eval instrumentation is
 * armed (捕获铁律: 可复现, 预取线程引入非决定性)。 */

#define DS4_METAL_PF_MAX_EMBD 8192u

#define DS4_METAL_PF_MAX_TOP 32u

typedef struct {
    int valid;
    int gate_inp_is_f32;      /* early layers carry an F32 router matrix */
    const void *model_map;
    uint64_t gate_inp_off;    /* F16/F32 [n_expert][n_embd] rows */
    uint64_t probs_bias_off;  /* F32 [n_expert], UINT64_MAX = absent */
    uint64_t gate_exps_off;
    uint64_t up_exps_off;
    uint64_t down_exps_off;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t n_embd;
    uint32_t n_expert;
    uint64_t hash_off;        /* token-id hash routing LUT (I32 [k][n_vocab]); UINT64_MAX = score routing */
    uint32_t hash_k;
    uint32_t hash_rows;
} ds4_metal_layer_router;

/* Wave 36: batch rows.  A kc-token verify batch activates a 50-74 expert
 * union per layer, but the old job carried ONE hidden row -- the predictor
 * covered ~8 of them and batch gathers ran effectively unpredicted (all
 * cold).  Batch jobs snapshot up to this many stride-sampled rows and the
 * predictor unions their per-row top-k (neighbouring tokens share experts,
 * so 16 sampled rows recover most of the union). */
#define DS4_METAL_PF_MAX_ROWS 16u

typedef struct {
    uint32_t layer;                      /* layer being predicted (= L+1) */
    int kind;                            /* 0 = score prediction from x; 1 = token-hash note */
    int token;
    uint32_t n_rows;                     /* 0/1 = single-x decode job; >1 = batch union job */
    float x[DS4_METAL_PF_MAX_EMBD];      /* router-input snapshot from layer L (kind 0) */
    float xrows[DS4_METAL_PF_MAX_ROWS * DS4_METAL_PF_MAX_EMBD]; /* batch row snapshots */
} ds4_metal_pf_job;

/* ---- project.md P2.1 x P2.2: prediction-driven remote expert staging ----
 *
 * Racing the per-layer gather cursor against the local pread threads cannot
 * use the Thunderbolt link well: a decode layer exposes only 18 units for
 * ~10ms and 8 local threads claim them instantly, while a remote unit costs a
 * ~2ms round trip that then sits on the layer's critical path (measured:
 * rfetch stuck at ~5MiB/layer).  Staging inverts it: the router prediction
 * for layer L+1 (≈80% top-6 coverage) dispatches the predicted experts to the
 * remote fetch threads which pull them into a RAM staging slot DURING layer
 * L's entire gather+GPU window (~14ms x 4.5GB/s >> 54MiB, no tail
 * constraint).  At layer L+1's gather, staged experts are RAM memcpys; only
 * mispredicted experts touch the local SSD.  Two slots alternate by layer
 * parity; only the first predicted layer (d==0) is staged, so a slot is never
 * re-armed while the gather of its layer can still read it.  Stage bytes are
 * advisory: a missing/partial entry just falls back to the normal local read. */

#define DS4_METAL_STAGE_MAX_EXPERTS 16u

/* Shared with the remote-fetch worker section below (defined here so the
 * prediction thread can wake the fetch threads after arming a slot). */
typedef struct ds4_metal_expert_gather_ctx ds4_metal_expert_gather_ctx;

typedef struct {
    volatile uint32_t gen;                 /* bumped on (re)arm; 0 = never armed */
    volatile uint32_t busy;                /* fetchers inside this slot (re-arm gate) */
    uint32_t layer;
    int token;                             /* decode token this arm belongs to */
    uint32_t n;
    volatile uint32_t next;                /* atomic claim index for fetch threads */
    uint32_t ids[DS4_METAL_STAGE_MAX_EXPERTS];
    volatile uint32_t ready_gen[DS4_METAL_STAGE_MAX_EXPERTS];
    uint64_t gate_off, up_off, down_off;
    uint64_t gate_eb, down_eb;
    uint64_t stride;                       /* per-expert bytes in buf */
    uint8_t *buf;
    size_t buf_cap;
} ds4_metal_stage_slot;

struct ds4_metal_expert_gather_ctx {   /* typedef'd forward at the staging slots */
    const void *model_map;
    const uint8_t *map;
    uint8_t *gate_dst;
    uint8_t *up_dst;
    uint8_t *down_dst;
    const uint32_t *active_ids;
    uint32_t n_active;
    uint32_t n_expert_total;
    uint32_t layer_index;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t next_slot;          /* atomic unit cursor (unit = slot*3 + part) */
    uint32_t done;               /* atomic completed-unit counter */
    int use_pread;
    int pread_fd;
    int ok;
};

/* ---- project.md P2.2 low-cost variant: remote expert fetch workers ----
 *
 * P0.3 measured the coordinator's SSD at ~2.5GB/s (every pattern) and the
 * worker's at 5.5-6.7GB/s -- and the worker's disk idles during the
 * coordinator's pipeline half.  Both machines map the byte-identical GGUF, so
 * these workers pull gather units off the same atomic cursor as the local
 * pread threads but satisfy them via ds4_dist_expert_fetch() from the peer's
 * disk over Thunderbolt: aggregate supply = local SSD + TB link.  Failure of
 * the link permanently degrades to local-only (correctness never depends on
 * the peer).  Enabled by --expert-fetch-host on the puller side and
 * --expert-fetch-serve on the serving side.
 * (g_rf_ctx/g_rf_mu/g_rf_cv/g_rf_link_down live next to the staging slots
 * above so the prediction thread can wake the fetch threads.) */

typedef struct {
    uint64_t src_off;
    uint8_t *dst;
    uint64_t len;
    double t0;
} ds4_metal_rf_pending;

/* VQ gather 并行化(2026-07-26): 每专家 dequant 独立写不相交 scratch → pthread 分片。
 * 串行单线程 ~256ms/层是 decode 0.09t/s 的主因; 8 线程 dequant → ~数倍。 */
typedef struct {
    const void *model_map; const uint8_t *blob; const uint32_t *active_ids;
    uint16_t *gbase, *ubase, *dbase;
    uint64_t down_offset, down_expert_bytes;
    uint32_t in, mid, out_dim, lo, hi;
    volatile int *err;
} ds4_vq_gather_task;

extern bool g_expert_pool_init_attempted;
extern bool g_expert_pool_summary_registered;
extern id<MTLBuffer> g_expert_pool_gate, g_expert_pool_up, g_expert_pool_down;
extern uint64_t g_expert_pool_budget_bytes, g_expert_pool_gate_expert_bytes, g_expert_pool_down_expert_bytes, g_expert_pool_slot_bytes;
extern uint32_t g_expert_pool_slots;
extern ds4_metal_expert_pool_entry *g_expert_pool_entries;
extern ds4_metal_expert_pool_meta g_expert_pool_meta[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern int32_t g_expert_pool_head;
extern int32_t g_expert_pool_tail;
extern uint64_t g_expert_pool_calls, g_expert_pool_requests, g_expert_pool_hits, g_expert_pool_misses;
extern uint64_t g_expert_pool_inflight_hits, g_expert_pool_prefetches, g_expert_pool_prefetch_hits, g_expert_pool_prefetch_misses;
extern uint64_t g_expert_pool_prefetch_drops, g_expert_pool_prefetch_copied, g_expert_pool_sync_waits, g_expert_pool_fallbacks;
extern uint64_t g_expert_pool_miss_copy_bytes;
extern double g_expert_pool_miss_copy_ms, g_expert_pool_wait_ms, g_expert_pool_prefetch_copy_ms;
extern uint32_t g_expert_pool_layer_start, g_expert_pool_layer_end, g_expert_pool_lookahead;
extern uint32_t g_expert_pool_prefetch_top;
extern int g_expert_pool_prefetch_self, g_expert_pool_prefetch_adjacent, g_expert_pool_wait_inflight, g_expert_pool_foreground_fill;
extern int g_expert_pool_prefetch_evict, g_expert_pool_hit_only;
extern uint32_t g_expert_pool_admit_after;
extern int g_expert_pool_hotlock_enabled;
extern uint32_t g_expert_pool_hotlock_top;
extern uint64_t g_expert_pool_admission_skips, g_expert_pool_clock;
extern bool g_expert_pool_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
extern uint32_t g_expert_pool_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern uint32_t g_expert_pool_pinned_total, g_expert_pool_static_pinned_total, g_expert_pool_dynamic_pinned_total;
extern uint64_t g_expert_pool_pinned_requests, g_expert_pool_pinned_hits, g_expert_pool_pinned_misses, g_expert_pool_pinned_prefetches;
extern uint64_t g_expert_pool_auto_pin_updates;
extern bool g_expert_pool_pinned_layer_queued[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern uint32_t g_expert_pool_auto_pin_top, g_expert_pool_auto_pin_min_req, g_expert_pool_auto_pin_interval, g_expert_pool_pin_reserve;
extern uint32_t g_expert_pool_hotlist_top, g_expert_pool_hotlist_interval;
extern uint64_t g_expert_pool_auto_pin_last_req[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern uint32_t g_expert_pool_last_active_n[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern uint32_t g_expert_pool_last_active[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX];
extern uint64_t g_expert_pool_hot_count[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
extern pthread_mutex_t g_expert_pool_mu;
extern pthread_cond_t g_expert_pool_cv;
extern bool g_expert_pool_prefetch_thread_started;
extern bool g_expert_pool_prefetch_shutdown;
extern bool g_expert_pool_cycle_guard_reported;
extern ds4_metal_expert_prefetch_req *g_expert_pool_prefetch_queue;
extern uint32_t g_expert_pool_prefetch_qcap, g_expert_pool_prefetch_qhead, g_expert_pool_prefetch_qtail, g_expert_pool_prefetch_qcount;
extern uint64_t g_pf_pred_hits, g_pf_pred_total;
extern int g_expert_gather_nocache_call;
extern ds4_metal_layer_router g_layer_router[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern pthread_mutex_t g_pf_mu;
extern pthread_cond_t g_pf_cv;
extern ds4_metal_pf_job g_pf_slot;
extern volatile uint32_t g_pf_seq;
extern uint32_t g_pf_picked_seq;
extern int g_pf_shutdown;
extern uint64_t g_pf_enqueued, g_pf_superseded, g_pf_advise_fail, g_pf_skip_cached;
extern volatile int g_gather_active;
extern uint32_t g_pf_pred_gen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
extern uint32_t g_pf_pred_mark[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
extern volatile int g_pf_token;
extern int g_pf_saw_nonhash;
extern ds4_metal_expert_gather_ctx * volatile g_rf_ctx;
extern pthread_mutex_t g_rf_mu;
extern pthread_cond_t g_rf_cv;
extern volatile int g_rf_link_down;
extern ds4_metal_stage_slot g_stage[2];
extern volatile int g_stage_recent;
extern uint64_t g_stage_hit_bytes_total, g_stage_armed, g_stage_fetched, g_stage_dropped;
extern id<MTLBuffer> g_moe_vq_gate_scratch, g_moe_vq_up_scratch, g_moe_vq_down_scratch;

/* Expert-fetch client half config (ds4_gpu_set_expert_fetch_client): host[0]
 * empty = client off; accept_port != 0 = reverse-established accept mode. */
extern char g_efetch_client_host[256];
extern int g_efetch_client_port;
extern int g_efetch_client_accept_port;

void ds4_gpu_set_expert_offload(int enabled);
int ds4_gpu_expert_offload_enabled(void);
uint32_t ds4_gpu_expert_pool_min_layer_slots(void);
uint32_t ds4_gpu_expert_pool_requested_layers(void);
uint32_t ds4_gpu_expert_pool_served_layers(void);
int ds4_gpu_expert_pool_layer_served(uint32_t layer, uint32_t *cap_out);
int ds4_gpu_expert_pool_layer_allowed(uint32_t layer);
int ds4_gpu_expert_pool_add_pin(uint32_t layer, uint32_t expert, bool dynamic);
int ds4_gpu_expert_pool_is_pinned(uint32_t layer, uint32_t expert);
void ds4_gpu_expert_pool_print_hotlist(const char *tag, uint32_t top, bool include_pinned_empty);
int ds4_gpu_expert_pool_is_enabled(void);
void ds4_gpu_expert_pool_print(const char *tag);
void ds4_gpu_expert_pool_summary(void);
void ds4_gpu_expert_pool_lru_unlink(int32_t idx);
void ds4_gpu_expert_pool_lru_link_head(int32_t idx);
void ds4_gpu_expert_pool_lru_touch(int32_t idx);
int32_t ds4_gpu_expert_pool_find(const void *model_map, uint32_t layer, uint32_t expert);
void ds4_gpu_expert_pool_clear_busy(void);
int ds4_gpu_expert_pool_copy_slot( int32_t slot, const ds4_metal_expert_pool_meta *meta, uint32_t expert);
void ds4_gpu_expert_pool_register_layer_meta( const void *model_map, uint32_t layer, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t n_expert_total);
int ds4_gpu_expert_pool_enqueue_prefetch_unlocked( const void *model_map, uint32_t layer, uint32_t expert);
void ds4_gpu_expert_pool_predict_enqueue( const void *model_map, uint32_t layer, const uint32_t *active_ids, uint32_t n_active);
int32_t ds4_gpu_expert_pool_victim( const void *model_map, uint32_t layer, uint32_t layer_cap, const uint32_t *active_ids, uint32_t n_active);
void ds4_gpu_expert_pool_queue_pinned_layer( const void *model_map, uint32_t layer);
void ds4_gpu_expert_pool_auto_pin_layer( const void *model_map, uint32_t layer, uint32_t layer_cap);
void ds4_gpu_expert_pool_maybe_hotlock_layer( const void *model_map, uint32_t layer);
int ds4_gpu_expert_pool_init(uint64_t gate_expert_bytes, uint64_t down_expert_bytes);
int ds4_gpu_try_load_layer_experts_to_pool( const void *model_map, uint32_t layer_index, id<MTLBuffer> selectedbuf, NSUInteger selected_off, uint32_t n_picks, uint32_t n_active, const uint32_t *active_ids, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t n_expert_total, id<MTLBuffer> *gate_buf, id<MTLBuffer> *up_buf, id<MTLBuffer> *down_buf, uint32_t *source_n_total_expert);
uint32_t ds4_gpu_expert_gather_threads(void);
int ds4_gpu_pread_full(int fd, void *dst, uint64_t src_off, size_t len);
int ds4_gpu_expert_pread_enabled(void);
int ds4_gpu_expert_batch_nocache_enabled(void);
uint32_t ds4_gpu_expert_batch_nocache_min_tokens(void);
uint32_t ds4_gpu_moe_mm_id_min(void);
int ds4_gpu_expert_pread_fd_nocache(void);
int ds4_gpu_expert_pread_fd(void);
int ds4_gpu_expert_prefetch_top(void);
uint32_t ds4_gpu_expert_prefetch_depth(void);
float ds4_gpu_pf_softplus(float v);
int ds4_gpu_expert_prefetch_advise_paced(int fd, uint64_t off, uint64_t len, uint32_t my_seq);
int ds4_gpu_expert_range_mostly_cached(const void *model_map, uint64_t off, uint64_t len);
int ds4_gpu_expert_prefetch_predict_one(uint32_t layer, const float *x, uint32_t my_seq, int stage_this);
int ds4_gpu_expert_prefetch_enabled(void);
void ds4_gpu_expert_prefetch_enqueue(uint32_t next_layer, const float *x, uint32_t n_embd);
void ds4_gpu_expert_prefetch_enqueue_batch(uint32_t next_layer, const float *x, uint32_t n_embd, uint32_t n_tokens);
int ds4_gpu_expert_stage_enabled(void);
void ds4_gpu_expert_stage_arm(uint32_t layer, const int *top_idx, uint32_t n_top, int token);
void ds4_gpu_expert_stage_arm_if_new(uint32_t layer, const int *top_idx, uint32_t n_top, int token);
void ds4_gpu_expert_hash_note_run(int token);
void ds4_gpu_expert_router_note(int token, int hash_mode);
int ds4_gpu_expert_stage_fetch_one(ds4_metal_stage_slot *s, int conn_slot);
int ds4_gpu_expert_stage_has_work(void);
int ds4_gpu_expert_stage_pending(uint32_t layer, uint32_t id);
uint32_t ds4_gpu_expert_stage_wait_us(void);
const uint8_t *ds4_gpu_expert_stage_find(uint32_t layer, uint32_t id, uint32_t part, uint64_t *len_out);
void ds4_gpu_expert_prefetch_note_actual(uint32_t layer, const uint32_t *ids, uint32_t n);
void ds4_gpu_expert_gather_copy_unit(ds4_metal_expert_gather_ctx *ctx, uint32_t unit);
int ds4_gpu_expert_remote_fetch_slots(void);
void *ds4_gpu_expert_gather_temp_worker(void *arg);
int ds4_gpu_expert_gather_pool_run(ds4_metal_expert_gather_ctx *ctx, uint32_t nth);
int ds4_gpu_ensure_moe_scratch(uint32_t n_active, uint64_t gate_expert_bytes, uint64_t down_expert_bytes);
int ds4_gpu_expert_pin_hot(uint32_t layer_index, uint32_t expert_id);
void ds4_gpu_expert_pin_mlock_layer(uint32_t layer_index, const void *model_map, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes);
void ds4_gpu_resid_pin_mlock_layer(uint32_t layer_index, const void *res_gate_ptr, const void *res_up_ptr, const void *res_down_ptr, const float *res_lut, uint64_t gate_expert_bytes, uint64_t down_expert_bytes);
int ds4_gpu_gather_experts_run( const void *model_map, uint32_t layer_index, uint32_t n_active, const uint32_t *active_ids, uint8_t *gate_dst, uint8_t *up_dst, uint8_t *down_dst, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t n_expert_total);
int ds4_gpu_load_layer_experts_to_scratch( const void *model_map, uint32_t layer_index, uint32_t n_active, const uint32_t *active_ids, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t n_expert_total);
int ds4_gpu_collect_active_experts( id<MTLBuffer> selectedbuf, NSUInteger selected_off, uint32_t n_picks, uint32_t n_expert_total, uint32_t *active_ids, uint32_t active_cap, uint32_t *n_active_out);
int ds4_gpu_vq_unified_gather( const void *model_map, const uint8_t *blob, uint32_t n_active, const uint32_t *active_ids, uint64_t down_offset, uint64_t down_expert_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim);
int ds4_gpu_hot_unified_gather( const void *model_map, uint32_t n_active, const uint32_t *active_ids, const float *lut, const void *hot_gate, const void *hot_up, const void *hot_down, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim);
int ds4_gpu_remap_selected_to_slots( id<MTLBuffer> selectedbuf, NSUInteger selected_off, uint32_t n_picks, uint32_t n_expert_total, uint32_t *active_ids, uint32_t n_active);
int ds4_gpu_compact_selected_experts( id<MTLBuffer> selectedbuf, NSUInteger selected_off, uint32_t n_picks, uint32_t n_expert_total, uint32_t *active_ids, uint32_t active_cap, uint32_t *n_active_out);

#endif /* DS4_METAL_EXPERT_H */
