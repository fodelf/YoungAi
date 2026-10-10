#ifndef DS4_SERVER_TYPES2_H

#define DS4_SERVER_TYPES2_H

/* ds4_server.c 拆分共享类型/宏 (后半: 流式/服务器/任务/KV/trace) 与跨文件全局。 */

typedef enum {
    OPENAI_STREAM_THINKING,
    OPENAI_STREAM_TEXT,
    OPENAI_STREAM_TOOL,
    OPENAI_STREAM_SUPPRESS,
} openai_stream_mode;

typedef enum {
    DSML_TOOL_BETWEEN_INVOKES,
    DSML_TOOL_BETWEEN_PARAMS,
    DSML_TOOL_PARAM_VALUE,
    DSML_TOOL_DONE,
    DSML_TOOL_ERROR,
} dsml_tool_stream_state;

/* Shared states for protocol-specific DSML stream projections.  The model
 * still samples DSML; these states only translate already-sampled bytes into
 * OpenAI / Anthropic wire events while final parsing remains authoritative. */
typedef struct {
    dsml_tool_stream_state state;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} openai_tool_stream;

typedef struct {
    openai_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_reasoning;
    bool sent_content;
    openai_tool_stream tool;
} openai_stream;

typedef enum {
    DSML_DECODE_OUTSIDE,
    DSML_DECODE_STRUCTURAL,
    DSML_DECODE_STRING_BODY,
    DSML_DECODE_JSON_STRUCTURAL,
    DSML_DECODE_JSON_STRING,
} dsml_decode_state;

typedef enum {
    DSML_TRACK_SEARCH,
    DSML_TRACK_STRUCTURAL,
    DSML_TRACK_STRING_BODY,
    DSML_TRACK_JSON_PARAM,
    DSML_TRACK_DONE,
} dsml_track_mode;

typedef struct {
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
} dsml_syntax;

typedef struct {
    dsml_track_mode mode;
    dsml_decode_state decode;
    const dsml_syntax *syn;
    size_t pos;
    bool json_in_string;
    bool json_escaped;
} dsml_decode_tracker;

typedef enum {
    RESP_STREAM_THINKING,
    RESP_STREAM_TEXT,
    RESP_STREAM_SUPPRESS,
} responses_stream_mode;

typedef struct {
    responses_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool reasoning_item_opened;
    bool reasoning_item_closed;
    bool reasoning_summary_started;
    bool reasoning_closed_naturally;
    bool message_item_opened;
    bool message_text_part_open;
    bool message_item_closed;
    bool reasoning_emitted_any;
    bool message_emitted_any;
    buf reasoning_text;
    buf message_text;
    char response_id[40];
    char reasoning_id[40];
    char message_id[40];
    int reasoning_index;   /* output_index of the reasoning item (0 if present) */
    int message_index;     /* output_index of the assistant message item */
    int next_output_index; /* monotonic counter for upcoming output items */
    int sequence;          /* monotonic per-event sequence_number Codex consumes */
} responses_stream;

/* Item identity per tool call must be stable across added/done/completed. */
typedef struct {
    char fc_id[40];
    char call_id[64];
    bool is_custom;
    int output_index;
} responses_tool_item;

typedef enum {
    ANTH_STREAM_THINKING,
    ANTH_STREAM_TEXT,
    ANTH_STREAM_TOOL,
    ANTH_STREAM_SUPPRESS,
} anthropic_stream_mode;

typedef enum {
    ANTH_BLOCK_NONE,
    ANTH_BLOCK_THINKING,
    ANTH_BLOCK_TEXT,
    ANTH_BLOCK_TOOL,
} anthropic_block_type;

typedef struct {
    dsml_tool_stream_state state;
    const dsml_syntax *syn;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} anthropic_tool_stream;

/* Anthropic streaming uses the same sampled DSML bytes that will later be
 * parsed and remembered for exact continuation.  This state is only a wire
 * projection: it turns an in-progress DSML block into content_block/tool_use
 * SSE events, and never rewrites the model-visible transcript or cache key. */
typedef struct {
    anthropic_stream_mode mode;
    anthropic_block_type open_block;
    int next_index;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_thinking;
    bool sent_text;
    anthropic_tool_stream tool;
} anthropic_stream;

typedef struct job job;

typedef ds4_kvstore_entry kv_entry;

typedef ds4_kvstore_options kv_cache_options;

typedef ds4_kvstore kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

typedef struct {
    bool valid;
    /* Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /* Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;
    /* Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    /* Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;
    size_t visible_len;
} visible_live_state;

struct server {
    ds4_engine *engine;
    ds4_session *session;
    int ctx_size;              /* 批处理快路建临时会话用 */
    int batch_max;             /* 并发批上限(0=关), 见 generate_jobs_batched */
    int default_tokens;
    /* Server-side hard cap on any request's output tokens (0 = uncapped).
     * Resource protection for small local models with weak EOS discipline:
     * a client asking for max_tokens=32000 (e.g. Claude Code utility calls)
     * must not pin the single graph worker for hours. */
    int max_output_tokens;
    /* --dry-multiplier/--dry-base/--dry-allowed-length(2026-09-21, 113-1.md §4): V4.1 生成路的 DRY 序列复读惩罚, 服务级开关
     * (客户端协议里没有这个字段; qtf 这种调用方也不会传)。0 = 关 = 裸模型真值; 开了对每条请求生效, 温 0 也生效。 */
    float dry_multiplier, dry_base;
    int dry_allowed_length;
    /* --nothink: force non-thinking mode for every request. For served base
     * models with no think training, client-side thinking configs (Claude Code
     * sends them explicitly) would otherwise burn the whole output budget on
     * garbage reasoning. */
    bool force_nothink;
    /* --tool-primer: seed each tool-enabled assistant turn with the DSML
     * tool-call opener so a base/continuation model lands inside the format
     * and only has to continue it. */
    bool tool_primer;
    /* 同调用禁重契约(2026-07-22): 上一次发出的工具调用 (工具名, 参数名, 值字节)。
     * 多轮回路实证失败形态 = 消化 tool_result 后逐字节重发同一调用(复读吸引子);
     * 值区闭合时与上一调用同名同参数同值 → 拒闭合走分歧(new≠old 契约同族)。
     * 跟 session 生命周期走(跨请求保持)。 */
    char *prev_call_tool;
    char *prev_call_param;
    char *prev_call_val;
    size_t prev_call_len;
    kv_disk_cache kv;
    tool_memory tool_mem;
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state thinking_live;
    bool disable_exact_dsml_tool_replay;
    bool enable_cors;
    pthread_mutex_t tool_mu;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    int clients;
    uint64_t seq;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
    /* 监控(server_monitor.c, 2026-10-07): GET /monitor 页与 GET /metrics 的数据面; 起服时 mon_open, 停服 mon_close。 */
    struct server_monitor *mon;
    const char *backend_name;   /* ds4_backend_name(cfg.engine.backend), /metrics 的 engine.backend */
    int port;                   /* 监听端口: 训练页切到训练时交给 train_cycle.sh, 让 ds4-train 用同一端口接管页面(src/train/) */
};

/* Jobs are stack-owned by the client thread.  The worker signals completion
 * after the response has been written, so request data and the socket remain
 * valid without heap-allocating per-request job objects. */
struct job {
    int fd;
    request req;
    bool done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
    uint64_t mon;   /* 监控记录 id(mon_begin; 0 = 没记): 生成路的 mon_prefill/mon_token/mon_end 都靠它找到这条请求 */
    /* 非空 = 这不是请求, 是"热切侧车/后训练件"任务(server_plugins.c): 堆上分配, 没有客户端在等, worker 做完自己释放。
     * 走同一个队列是为了让它排在已到的请求后面、只在两条请求之间生效 —— 插件表是进程级的。 */
    struct server_plugin_switch *sw;
};

/* 客户端没给 max_tokens 时用它 = 不设上限; 真正的界是 ctx − 提示, 两条生成路各自 clamp。
 * 为什么不是某个具体数字: 见 server_config.c 的长注释(2026-09-22 删掉写死的 393216)。 */
#define SERVER_NO_OUTPUT_CAP INT_MAX

#define DS4_TOOL_MEMORY_DEFAULT_MAX_IDS 100000

#define DS4_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)

#define KV_CACHE_FIXED_HEADER DS4_KVSTORE_FIXED_HEADER

#define KV_CACHE_HIT_HALF_LIFE_SECONDS DS4_KVSTORE_HIT_HALF_LIFE_SECONDS

#define KV_EXT_TOOL_MAP DS4_KVSTORE_EXT_TOOL_MAP

#define KV_EXT_RESPONSES_VISIBLE DS4_KVSTORE_EXT_RESPONSES_VISIBLE

#define KV_EXT_THINKING_VISIBLE DS4_KVSTORE_EXT_THINKING_VISIBLE

#define KV_TOOL_MAP_MAGIC0 'K'

#define KV_TOOL_MAP_MAGIC1 'T'

#define KV_TOOL_MAP_MAGIC2 'M'

#define KV_TOOL_MAP_VERSION 1u

#define KV_TOOL_MAP_HEADER 8u

typedef enum {
    KV_REASON_UNKNOWN   = DS4_KVSTORE_REASON_UNKNOWN,
    KV_REASON_COLD      = DS4_KVSTORE_REASON_COLD,
    KV_REASON_CONTINUED = DS4_KVSTORE_REASON_CONTINUED,
    KV_REASON_EVICT     = DS4_KVSTORE_REASON_EVICT,
    KV_REASON_SHUTDOWN  = DS4_KVSTORE_REASON_SHUTDOWN,
} kv_cache_reason;

#define TRACE_CACHE_BEFORE 8

#define TRACE_CACHE_AFTER  8

#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

typedef struct {
    server *srv;
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    const char *phase;
    bool has_tools;
    bool responses_protocol;
    double t0;
    double last_t;
    int last_current;
    bool seen;
    /* SSE keepalive during long prefill: send HTTP/SSE headers ahead of
     * generation and emit a `:` comment line every few seconds so HTTP/TCP
     * idle timeouts on the client side don't close the connection while the
     * server is busy doing prefill. */
    int fd;
    bool stream;
    bool enable_cors;
    bool headers_sent;
    bool stream_failed;
    double last_keepalive;
    /* 监控的预填进度(V4 会话路用; V4.1 路在 v41_progress_cb 里直接报, 这两项留 0 免得报两遍) */
    server *mon_srv;
    uint64_t mon;
} server_prefill_progress;

typedef struct {
    bool inside;
    char tail[8]; /* Long enough for "</think>". */
    int tail_len;
} thinking_state;

/* ---- guided copy-constrained decode (tool-primer v1) ----------------------
 * Feasibility oracle for the primer's free-content regions: a candidate token
 * may be emitted iff its text bytes extend at least one active span of the
 * copy source (byte-level, so it is tokenizer-boundary agnostic). Before the
 * first emitted token any source offset may start a span (anchored mode
 * restricts starts to positions right after a \x01 separator — used for the
 * declared-tool-name list); afterwards only continuations of live spans
 * survive. Rationale: coding-agent tool argument values are almost always
 * verbatim spans of the prompt, while a 2-bit base model's free greedy
 * continuation of `parameter name="file_path">` is a documentation-style
 * placeholder ($FILE_PATH) — constraining decode turns "copy the value"
 * from a probability hope into a structural guarantee (fable5 判决链). */
/* 候选位置容量: 超过即停止搜集(server_primer.c 会打一次性 warning)。截断的后果
 * 不是崩, 而是位置集不全 → 可行续写被误判 infeasible → 走 best-logit 另选 token。 */
#define PRIMER_COPY_MAX_POS 64

typedef struct {
    const char *src;
    size_t      len;
    size_t      pos[PRIMER_COPY_MAX_POS];   /* offsets AFTER the matched prefix */
    int         n_pos;
    bool        anchored;    /* first token must start right after a \x01 */
    bool        started;
    bool        dead;        /* escape hatch tripped: constraint disabled */
} primer_copy;

typedef struct {
    char method[8];
    char path[256];
    char query[256];      /* '?' 后面的原文(不含 '?'); 没有就是空串。/metrics 看 requests=all / format=prometheus */
    bool accept_text;     /* 请求头 Accept 带 text/plain 或 application/openmetrics-text: Prometheus 抓取 /metrics 的问法 */
    char *body;
    size_t body_len;
} http_request;

typedef struct {
    server *srv;
    int fd;
} client_arg;

typedef struct {
    ds4_engine_options engine;
    const char *host;
    int port;
    int ctx_size;
    int batch_max;             /* --batch: 并发合批上限(0=关, 钳到 DS4_SERVER_BATCH_LANES) */
    bool tool_primer;
    bool force_nothink;
    int max_output_tokens;
    float dry_multiplier, dry_base; int dry_allowed_length;   /* 见 server_config 同名字段 */
    int default_tokens;
    const char *chdir_path;
    const char *trace_path;
    const char *kv_disk_dir;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    bool kv_cache_reject_different_quant;
    bool disable_exact_dsml_tool_replay;
    int tool_memory_max_ids;
    bool enable_cors;
} server_config;

/* 跨文件全局 (定义在各自 .c 里) */

extern volatile sig_atomic_t g_stop_requested;

extern bool g_force_nothink;

/* --base-native: base 底模用母语骨架渲染(# User:/# Assistant:)替代 chat 角色帧。 */
extern int g_base_native;

/* --primer-compact: 引导注入只把语义锚点送进 KV(对外 text 仍是完整合法 DSML)。 */
extern bool g_primer_compact;

extern volatile sig_atomic_t g_listen_fd;

extern char *g_soul_text;

extern int g_knowledge_n;

extern int g_req_mode;

extern const dsml_syntax dsml_syntaxes[3];  /* 条目数与定义处一致(长/短/裸三种 DSML 语法), 供跨文件 sizeof */

#endif /* DS4_SERVER_TYPES2_H */
