/* dist_state.h — 机械拆自 ds4_distributed.c: 运行时状态类型(Runtime State / PC.1 类型 / spec_io)。只供 dist_internal.h 包含。 */
#ifndef DS4_DIST_STATE_H
#define DS4_DIST_STATE_H

/* =========================================================================
 * Runtime State
 * =========================================================================
 *
 * The coordinator registry is shared by the accept thread and by the session
 * calls made from the main inference thread. Workers keep per-session KV state
 * keyed by the coordinator-provided session ID so independent callers do not
 * share token timelines by accident.
 */

typedef struct ds4_dist_worker_entry {
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    struct ds4_dist_worker_entry *next;
} ds4_dist_worker_entry;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t n_layers;
    uint32_t local_start;
    uint32_t local_end;
    uint32_t ctx_size;
    bool local_has_output;
    bool local_can_output_head;
    bool replay_check;
    bool debug;
    bool use_control_for_work;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    uint64_t generation;
    pthread_mutex_t mu;
    ds4_dist_worker_entry *workers;
    bool shutting_down;
    /* docs/archive/mtp.md Phase 1: request MTP drafts + cross-machine batch verification from
     * the last-layer worker (set by --mtp-role worker on the coordinator). */
    bool mtp_draft;
    /* 本机 MTP (--mtp-role coordinator): the drafter runs here, not on the worker.
     * Round 1 sends a plain frame; the worker returns hidden state, the local
     * output head turns it into logits, and ds4_session_mtp_draft rolls the K
     * candidates from that same hidden — same VERIFY batch as worker drafting. */
    bool mtp_draft_local;
    /* Wave 68+ DS4_DIST_SPEC_PIPE state-carried hooks (default NULL/0 => standard
     * synchronous path, NO signature change to eval_span and its 10 callers, so
     * the off path is byte-identical).  spec_overlap_cb is invoked by
     * dist_coordinator_eval_remote_on_fd BETWEEN the WORK send and the blocking
     * result recv -- exactly while the worker computes -- so the coordinator can
     * speculatively compute the NEXT cycle's local slice in that idle window.
     * spec_precomputed_hidden, when set, makes dist_coordinator_eval_span reuse it
     * and skip its own layer_slice (the precompute from the prev overlap window). */
    int   (*spec_overlap_cb)(void *ctx);
    void   *spec_overlap_ctx;
    const float *spec_precomputed_hidden;
} ds4_dist_coordinator_state;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_client_ctx;

typedef struct {
    ds4_dist_coordinator_state *state;
    int listen_fd;
} ds4_dist_accept_ctx;

typedef struct ds4_dist_worker_session {
    uint64_t session_id;
    uint64_t token_hash;
    bool token_hash_valid;
    ds4_session *session;
    /* docs/archive/mtp.md Phase 1: when the previous frame was a speculative VERIFY batch,
     * spec_pending is set and spec_base_len records the timeline length before
     * the batch was applied. The next frame carries accept_len; the worker rolls
     * its layer-slice KV back to spec_base_len + accept_len before proceeding. */
    bool spec_pending;
    uint32_t spec_base_len;
    struct ds4_dist_worker_session *next;
} ds4_dist_worker_session;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t layer_start;
    uint32_t layer_end;
    bool has_output;
    int ctx_size;
    int listen_fd;
    pthread_mutex_t mu;
    ds4_dist_worker_session *sessions;
} ds4_dist_worker_state;

typedef struct ds4_dist_worker_upstream ds4_dist_worker_upstream;

typedef struct ds4_dist_pending_request {
    uint64_t request_id;
    double downstream_t0;
    ds4_dist_telemetry_fixed telemetry;
    struct ds4_dist_pending_request *next;
} ds4_dist_pending_request;

typedef struct ds4_dist_worker_forwarder {
    ds4_dist_worker_upstream *upstream;
    char host[NI_MAXHOST];
    uint32_t port;
    int fd;
    pthread_t tid;
    bool thread_started;
    pthread_mutex_t send_mu;
    pthread_mutex_t queue_mu;
    pthread_cond_t queue_not_full;
    ds4_dist_pending_request *pending_head;
    ds4_dist_pending_request *pending_tail;
    uint32_t pending_count;
    uint32_t pending_depth;
    bool closing;
    struct ds4_dist_worker_forwarder *next;
} ds4_dist_worker_forwarder;

struct ds4_dist_worker_upstream {
    ds4_dist_worker_state *state;
    int fd;
    pthread_mutex_t write_mu;
    pthread_mutex_t forward_mu;
    ds4_dist_worker_forwarder *forwarders;
};

typedef struct ds4_dist_worker_job {
    void *payload;
    uint32_t bytes;
    struct ds4_dist_worker_job *next;
} ds4_dist_worker_job;

typedef struct {
    ds4_dist_worker_state *state;
    ds4_dist_worker_upstream *upstream;
    pthread_mutex_t mu;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    ds4_dist_worker_job *head;
    ds4_dist_worker_job *tail;
    uint32_t queued;
    uint32_t depth;
    bool closed;
    bool canceled;
    int rc;
} ds4_dist_worker_job_queue;

typedef struct {
    ds4_dist_worker_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_data_client_ctx;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t kind;
} ds4_dist_route_return;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    int fd;
} ds4_dist_route_entry;

typedef struct {
    ds4_dist_route_entry *entry;
    uint32_t count;
    void *blob;
    uint32_t blob_bytes;
} ds4_dist_route_plan;

typedef struct {
    uint32_t ctx;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t comp_cap;
    uint32_t token_count;
    uint32_t n_layers;
    uint32_t head_dim;
    uint32_t indexer_head_dim;
    uint32_t vocab;
    uint32_t raw_live;
} ds4_dist_kv_layout;

typedef struct {
    FILE *fp;
    uint64_t bytes;
    uint32_t layer_start;
    uint32_t layer_end;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
} ds4_dist_kv_shard_file;

/* Reverse-connect target: the coordinator dials this worker control address
 * (see dist_coordinator_reverse_connect_main). Defined here so it can live inside
 * the session for the session-driven coordinator path. */
typedef struct {
    ds4_dist_coordinator_state *state;
    const char *host;   /* worker control host (from --coordinator HOST PORT) */
    int port;
} ds4_dist_reverse_ctx;

struct ds4_dist_session {
    ds4_dist_coordinator_state state;
    int listen_fd;
    pthread_t accept_tid;
    bool accept_started;
    ds4_dist_accept_ctx accept_ctx;
    /* Reverse-connect (DS4_DIST_REVERSE_CONNECT): coordinator dials a listening
     * worker instead of accepting. reverse_ctx outlives the detached thread. */
    ds4_dist_reverse_ctx reverse_ctx;
    pthread_t reverse_tid;
    bool reverse_started;
    ds4_dist_route_plan plan;
    bool plan_ready;
    uint64_t plan_generation;
    uint64_t session_id;
    uint64_t request_id;
    /* docs/archive/mtp.md Phase 1: the previous speculative cycle ran a VERIFY batch and the
     * remote worker still has all K candidate tokens in its layer KV. The next
     * frame must carry spec_accept_len so the worker rolls back to the accepted
     * prefix before applying new work. */
    bool spec_accept_pending;
    uint32_t spec_accept_len;
    /* PC.1 copy speculation: adaptive copied-draft length (SuffixDecoding /
     * HF assisted-decoding style). Starts small so one wrong copy costs little
     * (an 8-token verify batch is ~3x a single decode forward), doubles on a
     * fully-accepted tail, resets on a (near-)total rejection. 0 = uninit. */
    uint64_t mtp_calls;
    uint64_t mtp_round1;
    uint64_t mtp_verify;
    uint64_t mtp_accept_tokens;
    uint64_t mtp_draft_tokens;
    uint64_t mtp_first_hit;
    uint64_t mtp_disabled_cycles;
    uint64_t mtp_disable_until_call;
    uint64_t mtp_window_calls;
    uint64_t mtp_window_accept_tokens;
    uint64_t mtp_window_first_hit;
    /* Probe telemetry: total cross-machine forwards (eval_span calls) issued
     * across all speculative calls. The whole point of carry-over drafting is to
     * collapse the per-cycle forward count from 2 (Round-1 + verify) to 1
     * (fused verify only), so forwards_per_call and accept_tokens/forward are the
     * two numbers that decide whether MTP is net-positive on this hardware. */
    uint64_t mtp_forwards;
    /* 本机 MTP zero-extra-forward (DS4_DIST_MTP_CARRY_DRAFT): the MTP head drafts
     * the NEXT cycle's candidates from the boundary row's hidden of THIS cycle's
     * verify batch, so no dedicated Round-1 forward is needed to produce a draft
     * hidden. carry_valid means carry_drafts[0..carry_draft_n-1] were drafted for
     * positions starting at carry_pos; carry_drafts[0] is the MTP guess for the
     * token the caller will resample as next first_token (validated by equality
     * before the fused batch trusts the tail). Greedy-only, like every spec path. */
    bool carry_valid;
    uint32_t carry_pos;
    uint32_t carry_draft_n;
    int carry_drafts[64];
    /* Wave 68+ DS4_DIST_SPEC_PIPE: speculative layer-pipeline overlap.  During the
     * worker's compute of cycle N (the coordinator-idle t_remote_blocked measured
     * at ~50% of r2), precompute cycle N+1's local-slice hidden, betting cycle N
     * fully accepts (the dominant heavy-echo case, ~73%).  On full-accept the
     * precompute is reused so the serial coord(t_local)+worker(t_remote) collapses
     * toward max(); on partial accept the speculative KV advance is rolled back
     * (ds4_session_layer_slice_rollback) and the hidden discarded.  Off => legacy
     * synchronous eval_span (byte-identical, validated via --dump-logprobs). */
    bool     spec_pipe_valid;       /* spec_pipe_hidden holds a usable precompute */
    uint32_t spec_pipe_kb;          /* batch token count the precompute was built for */
    uint32_t spec_pipe_base;        /* layer-slice KV len the precompute assumes */
    int      spec_pipe_toks[64];    /* the predicted batch tokens precomputed */
    float   *spec_pipe_hidden;      /* n_tokens*hc precomputed local hidden (owned) */
    uint32_t spec_pipe_hidden_cap;  /* bytes allocated for spec_pipe_hidden */
    /* Scratch logits row for the penalized speculative accept gates (vocab
     * floats, lazily allocated; session is process-lifetime like the fields
     * above). The gate must see the SAME penalized pick the plain sampler
     * would make, but on a copy: the caller-visible row stays raw because the
     * sampler re-applies its own penalty at the same window. */
    float   *spec_pen;
};

/* =========================================================================
 * PC.1 copy speculation (project.md §3.5): prompt-lookup drafting.
 * The drafter is not a model; it is an n-gram matcher over the session
 * transcript ("copy what the context already said"). Zero draft memory, zero
 * draft compute, no MTP weights needed; only the existing VERIFY batch +
 * accept_len rollback protocol is reused. Greedy-only like the MTP path.
 *
 * Always armed -- no enable flag. It is the natural default drafter for
 * greedy distributed decode: self-tuning (adaptive bet ladder), a no-op on
 * text with no transcript repeat (a miss never issues a verify batch), and
 * lossless (every accepted row is gated by the penalized argmax the plain
 * sampler would produce). An explicitly
 * configured MTP drafter takes the drafting job instead -- copy-spec only
 * owns the no-drafter domain. The lengths below are the calibrated values
 * (waves 19/31/36); copy-spec 整族已删除(2026-08-05)。 */
/* DIST_CS_* 常数族已删除(copy-spec 整族)。 */



typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_dist_logprob;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    ds4_session *progress_session;
    uint64_t first_request_id;
    uint64_t *expected_hashes;
    uint32_t count;
    uint32_t total_tokens;
    uint32_t chunk_cap;
    uint32_t progress_base;
    uint32_t progress_total;
    uint32_t progress_completed;
    bool progress_done;
    uint64_t hc_values;
    bool allow_hidden;
    uint32_t final_kind;
    void *final_payload;
    uint32_t final_payload_bytes;
    int rc;
    char err[256];
    pthread_mutex_t progress_mu;
    pthread_cond_t progress_cv;
} ds4_dist_prefill_result_reader;

typedef struct {
    uint32_t pos;
    uint32_t n_tokens;
    uint32_t hidden_bytes;
    uint64_t request_id;
    uint64_t prefix_hash;
    uint64_t result_hash;
    bool reset_session;
    bool ack_only;
    float *hidden;
} ds4_dist_prefill_send_slot;

typedef struct {
    ds4_dist_coordinator_state *state;
    const ds4_dist_route_plan *plan;
    const ds4_tokens *prompt;
    uint64_t session_id;
    int fd;
    ds4_dist_prefill_send_slot *slots;
    uint32_t slot_count;
    uint32_t head;
    uint32_t tail;
    uint32_t queued;
    bool producer_done;
    bool stop;
    int rc;
    double send_sec;
    uint64_t send_bytes;
    char err[256];
    pthread_mutex_t mu;
    pthread_cond_t can_enqueue;
    pthread_cond_t can_dequeue;
} ds4_dist_prefill_sender;

/* docs/archive/mtp.md Phase 1 speculative I/O threaded through the coordinator eval path.
 * NULL on every non-speculative call (the normal decode/prefill path is byte
 * identical). When set on a DRAFT frame the worker appends MTP draft ids
 * (read into drafts[0..draft_n-1]); on a VERIFY frame the worker returns
 * n_tokens logit rows copied into verify_logits. */
typedef struct {
    uint32_t draft_cap;
    uint32_t accept_len;
    uint32_t extra_flags;     /* DS4_DIST_WORK_F_DRAFT or _VERIFY */
    uint32_t draft_n;
    uint32_t drafts[64];
    float   *verify_logits;   /* n_tokens * vocab floats when VERIFY */
    /* 本机 MTP carry-over draft (DS4_DIST_MTP_CARRY_DRAFT): optional sink for the
     * worker's returned per-row hidden states on a VERIFY batch. Only fillable
     * when the worker holds no output head (returns RESULT_HIDDEN_STATE, the
     * local-MTP topology); NULL otherwise. Sized n_tokens * hc_values floats.
     * The coordinator re-runs its output head on the boundary row from here to
     * seed cur_hc for the next cycle's MTP draft, eliding the Round-1 forward. */
    float   *hidden_rows;
} ds4_dist_spec_io;

typedef struct {
    const uint8_t *p;
    uint32_t remaining;
} ds4_dist_mem_reader;

#endif /* DS4_DIST_STATE_H */
