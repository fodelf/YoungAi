#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f
/* session 默认 ctx: CLI(-c)与 server(--ctx) 同一个默认, help 文本经 DS4_STRINGIFY
 * 同源拼接 —— 改这里, 代码与 --help 显示一起变, 不会再各写一份跑飞。 */
#define DS4_DEFAULT_CTX_SIZE 32768
#define DS4_STRINGIFY_(x) #x
#define DS4_STRINGIFY(x) DS4_STRINGIFY_(x)
/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold.
 * (无 u 后缀: server --help 经 DS4_STRINGIFY 拼进文本, 比较端自行转 uint32。) */
#define DS4_THINK_MAX_MIN_CONTEXT 393216

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    /* --dist-prefill-cap: 分布式会话 prefill 批上限(0=自动), 压宽批的专家工作集。 */
    uint32_t prefill_cap;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
    /* Tensor parallelism (Stage 2 skeleton). Distinct from the layer-pipeline
     * mode above: when tp_enabled, BOTH machines load the full layer stack and
     * run in lockstep, splitting the MoE down_proj input (ff) dimension and
     * summing partial [n_embd] outputs via an all-reduce over a dedicated TP
     * socket. tp_layers caps how many leading layers use the TP split (the rest
     * stay replicated) so the path can be brought up on 2-3 layers first. The
     * existing role/listen/coordinator host:port fields select who listens. */
    bool tp_enabled;
    uint32_t tp_layers;
    /* --reverse-connect: flip who dials whom (coordinator dials a listening
     * worker; TP 模式同义)。Works around a host where one connect direction
     * fails (observed: macOS Local Network privacy refusing in-process connect
     * over the thunderbolt bridge -> EHOSTUNREACH while nc succeeds). The
     * address flags swap accordingly (see dist_cli validation). */
    bool reverse_connect;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    /* Optional sidecar GGUF holding the per-layer go1b "hidden variable z^L"
     * four-loss correction tensors (blk.{L}.corr_*). When NULL, the engine
     * auto-detects ds4-go1b-corr.gguf next to model_path. Absent => pure 1-bit. */
    const char *corr_path;
    /* Optional 1-bit residual sidecar GGUF (blk.{L}.ffn_*_exps_res, go1b). When set,
     * the routed-MoE sums a second 1-bit layer into each expert. Absent => single 1-bit. */
    const char *residual_path;
    /* Optional go-onebit DQZ2 runtime sidecar (--zchain FILE):
     * the quantizer's per-layer multiplicative correction chain -- GE per-expert
     * gains folded into the router weights + a per-token scale λ(x) on the routed
     * MoE contribution. Absent => embedded blk.L.opt_* auto-load, else bare base. */
    const char *zchain_path;
    /* --vq-dir: VQ 码本目录侧车(优先于 residual_path; 都缺则用 GGUF 内嵌 blob)。 */
    const char *vq_dir_path;
    /* --draft-gguf: 独立 DSpark drafter GGUF(仅 mtp.* 张量); 主模型自带 mtp.* 时优先。 */
    const char *draft_gguf_path;
    /* --draft-zchain: drafter 反修放大器侧车, 3 层链合并进主链槽 43..45。 */
    const char *draft_zchain_path;
    /* --mm-image-cmd: 多模态外部图像编码器命令(缺省探 ./mm-ui)。 */
    const char *mm_image_cmd;
    /* --spec: DSpark 投机解码 + 在线调度仲裁(投机/纯解码谁快用谁)。贪心 verify
     * 逐位对照 ⇒ token 序列与纯解码逐字一致, 只换速度不换输出。默认关。 */
    bool spec;
    ds4_backend backend;
    int n_threads;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    bool warm_weights;
    bool quality;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);

/* Mode P/G dynamic-routing classifier: true => programming prompt (route to the
 * resident programming model), false => everyday prompt (route to the full
 * cached model). Model-free text heuristic; safe to call before opening any
 * engine, which is exactly what the router needs to pick the model. */
bool ds4_prompt_is_programming(const char *prompt);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
/* Multi-domain z-sidecar plugin hot-swap: free the current corr sidecar and
 * load `path` (~90MB, seconds). NULL/"" unloads (pure 1-bit). Call between
 * generations only. Returns 0 on success, -1 on load failure (previous corr
 * is kept in that case). */
int ds4_engine_corr_switch(ds4_engine *e, const char *path);
/* Multimodal registry (ds4_multimodal.c): "text" built in; the "image"
 * family auto-binds the frontend-domain UI-sketch encoder at open when
 * present (--mm-image-cmd > ./mm-ui, built by `make mm-ui`). Server
 * /v1/messages image content blocks consume the registry's TEXT form so
 * rendered prompts / disk-KV prefix keys / exact replay stay byte-stable.
 * NULL before open or on OOM (consumers must fail closed, not fake). */
struct ds4_mm;
struct ds4_mm *ds4_engine_mm(ds4_engine *e);
/* ---- Sampling-lane policy (问答/编程链路的一等抽象) ----
 * The frontend that knows the request shape (server/agent/cli/dist coordinator)
 * declares which lane the session is generating for; the core sampler scopes
 * repeat/anticycle penalties by lane:
 *   FREE          - free content (chat answers, agent prose): freq penalty and
 *                   anticycle bans apply.
 *   TOOL_SYNTAX   - forced tool-call syntax emission: raw logits, no penalties
 *                   (a penalty-diverted syntax token corrupts the protocol).
 *   COPY_EMISSION - copy-constrained value emission (server primer values):
 *                   raw logits; the copy contract REQUIRES verbatim context
 *                   reuse, which the anticycle self-copy ban would forbid.
 * Anticycle bans scan only [generation start, end). Mark the boundary with
 * ds4_session_mark_generation_start (pins at the current checkpoint length;
 * call after prefill, before the first sampled token of a response) or set it
 * explicitly after a rebuild whose checkpoint already contains generated
 * tokens. Unmarked sessions fall back to 0 = whole context (legacy). The env
 * freq-penalty window intentionally IGNORES the boundary (2026-07-06 mono
 * verdict: the early-generation window must include the prompt tail or the
 * 2-bit penalty is too weak). */
enum {
    DS4_LANE_FREE = 0,
    DS4_LANE_TOOL_SYNTAX = 1,
    DS4_LANE_COPY_EMISSION = 2,
};
void ds4_session_set_lane(ds4_session *s, int lane);
int  ds4_session_lane(const ds4_session *s);
void ds4_session_mark_generation_start(ds4_session *s);
void ds4_session_set_generation_start(ds4_session *s, int pos);
/* Per-request penalties (OpenAI frequency_penalty / presence_penalty semantics,
 * counted over the generated region only), FREE lane only. Sessions are reused across requests: call on
 * EVERY request; (0,0) = none/clear. */
void ds4_session_set_request_penalties(ds4_session *s, float freq, float presence);
/* Whether greedy speculative acceptance (copy-spec / MTP argmax gating) is
 * distribution-preserving for the current request: true only at temperature 0.
 * Default true (legacy); frontends must set false for sampled requests so the
 * dist accept gate falls back to plain decode instead of silently going
 * greedy on n-gram hits. */
void ds4_session_set_spec_greedy(ds4_session *s, int greedy_ok);
int  ds4_session_spec_greedy_ok(const ds4_session *s);
/* Built-in reference-corpus (language-idiom) drafter lookup: longest suffix of
 * tail[0..len) (>= min_g tokens) occurring in the engine's idiom corpus; copies
 * up to cap continuation tokens into out[]. Returns the anchor length, 0 = no
 * match (a miss costs nothing -- callers just decode normally). Used by both
 * the single-machine and distributed copy-spec drafters. */
uint32_t ds4_engine_ref_match(ds4_engine *e, const int *tail, uint32_t len,
                              uint32_t min_g, uint32_t cap,
                              int *out, uint32_t *out_n);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);

/* 请求批处理: N 个会话各推进一个 token, 共享一次前向的行无关部分(骨干 FFN/MoE/输出头
 * 权重只读一遍)。注意力半层各回自己的图算, KV 结构不变。实测每行成本 1 行 35.9ms /
 * 4 行 18.9 / 8 行 14.9 ⇒ 8 路聚合约 1.96×。要求同引擎、GPU 后端、非分布式。 */
int ds4_session_eval_multi(ds4_session **sessions, const int *tokens, uint32_t n,
                           char *err, size_t errlen);

/* Append N KNOWN tokens in ONE layer-major batch (guided-primer structure
 * injection: the server knows the tokens, no sampling). See ds4.c. */
int ds4_session_eval_span(ds4_session *s, const int *tokens, int n,
                          char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
/* 磁盘 KV 兼容键(routed 张量类型码; 0=无 routed=不可存)。bits 口径会把不同 2-bit
 * 格式塌成同一个 2, 造成跨模型 KV 静默互认 —— 兼容判定一律用这个, bits 只作显示。 */
int ds4_engine_routed_kv_key(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);
/* docs/archive/mtp.md Phase 1 cross-machine verifier: run a K-token candidate batch through
 * the final-layer worker slice and emit per-row greedy argmax into
 * row_tops[0..n_tokens-1]. Writes layer KV for pos0..pos0+n_tokens-1 without
 * committing the timeline (caller commits accepted prefix + rolls back the rest). */
int ds4_session_verify_batch_argmax(ds4_session *s,
                                    const int *tokens,
                                    uint32_t n_tokens,
                                    uint32_t pos0,
                                    uint32_t layer_start,
                                    uint32_t layer_end,
                                    const float *input_hc,
                                    float *row_logits,
                                    char *err,
                                    size_t errlen);
/* Truncate the layer-slice timeline to new_len after a speculative batch; query
 * the current committed length. */
int ds4_session_layer_slice_rollback(ds4_session *s, uint32_t new_len,
                                     char *err, size_t errlen);
uint32_t ds4_session_layer_slice_len(const ds4_session *s);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

/* Live Mach phys_footprint (bytes) and the configured --mem-budget-mb budget
 * (bytes, 0 if unset).  Exposed for backends that size caches dynamically
 * against the headroom rather than a fixed cap. */
uint64_t ds4_runtime_phys_footprint_bytes(void);
uint64_t ds4_runtime_mem_budget_bytes(void);


/* ---- 取料入口(2026-08-22: 从 env 迁到 CLI) --------------------------------
 * 捕获/评估的入口过去是 DS4_CAP_DIR / DS4_EVAL_IDS / DS4_EVAL_HDUMP /
 * DS4_EVAL_LOGITS 四个环境变量。env 让"这次跑到底做了什么"不可见, 且漏设即静默换行为
 * (本项目已因此废掉一整轮反修)。改成命令行参数: 发车命令里一眼可见, 进程内全局存取。 */
void        ds4_tool_set_cap_dir(const char *p);
const char *ds4_tool_cap_dir(void);
void        ds4_tool_set_cap_layers(const char *p);   /* --cap-layers "lo-hi" */
const char *ds4_tool_cap_layers(void);
void        ds4_tool_set_eval_ids(const char *p);
const char *ds4_tool_eval_ids(void);
void        ds4_tool_set_eval_hdump(const char *p);
const char *ds4_tool_eval_hdump(void);
void        ds4_tool_set_eval_logits(const char *p);
void        ds4_tool_set_eval_no_bos(int v);
int         ds4_tool_eval_no_bos(void);
const char *ds4_tool_eval_logits(void);
/* --amp-anchor FILE [--amp-anchor-route]: 捕获回放钉锚(判决仪器)。 */
void        ds4_tool_set_amp_anchor(const char *p, int route_on);
const char *ds4_tool_amp_anchor(void);
int         ds4_tool_amp_anchor_route(void);
/* --multi-bench N: N 会话并发批基准仪器(跑完 exit)。 */
void        ds4_tool_set_multi_bench(int n);
int         ds4_tool_multi_bench(void);
/* --prefill-chunk N: prefill 分块 token 上限(0=整段一批; 不设=按后端默认)。 */
void        ds4_tool_set_prefill_chunk(int chunk);
int         ds4_tool_prefill_chunk(void);
/* --mem-budget-mb N: 进程内存红线; 看门狗 90% abort + L1 装载闸 85% 拒载。
 * 不设 = 护栏不武装(危险, 加载大模型的脚本必须传)。 */
void        ds4_set_mem_budget_mb(int mb);

#endif
