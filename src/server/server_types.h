#ifndef DS4_SERVER_TYPES_H

#define DS4_SERVER_TYPES_H

/* ds4_server.c 拆分共享类型/宏 (前半: 包含/JSON/消息/请求/DSML 标签)。
 * 与原文件同序搬运; 后半在 server_types2.h, 两者只经 server_internal.h 进入。 */

/* OpenAI/Anthropic compatible local server.
 *
 * HTTP is intentionally simple: each client connection is handled by a small
 * blocking thread that parses one request, then queues a job to the single
 * Metal worker.  The worker owns the ds4_session and therefore owns all live KV
 * cache state.  That keeps session reuse, disk checkpointing, and future
 * batching decisions in one place instead of spreading graph mutations across
 * client threads. */

#include "ds4.h"

#include "ds4_distributed.h"

#include "ds4_kvstore.h"

#include "ds4_multimodal.h"

#include "rax.h"

#include <arpa/inet.h>

#include <ctype.h>

#include <dirent.h>

#include <errno.h>

#include <float.h>

#include <fcntl.h>

#include <limits.h>

#include <math.h>

#include <netinet/in.h>

#include <poll.h>

#include <pthread.h>

#include <signal.h>

#include <stdbool.h>

#include <stdint.h>

#include <stdarg.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <strings.h>

#include <sys/socket.h>

#include <sys/stat.h>

#include <sys/time.h>

#include <sys/types.h>

#include <time.h>

#include <unistd.h>

#define DS4_SERVER_IO_TIMEOUT_SEC 10

#define DS4_SERVER_SEND_STALL_TIMEOUT_MS 2000

/* 并发合批的最大路数: batch worker 的会话/状态数组全按它定长, --batch 钳到它。
 * 8 = 单 graph worker 上稀疏 MoE tile 欠填的聚合甜点(见 request-batching 设计)。 */
#define DS4_SERVER_BATCH_LANES 8

/* 工具 id 前缀: 它是 DSML 精确重放表的键(id → 采样原字节), 生成与重渲染两侧
 * 必须同前缀 —— 原来三处手写字符串, 改一处漏一处 = 重放查不到, 重渲染 prompt
 * 与活 KV 不再逐字节匹配。 */
#define DS4_TOOL_ID_PREFIX_ANTHROPIC "toolu_"
#define DS4_TOOL_ID_PREFIX_OPENAI    "call_"

/* 合批出队前的聚集窗口: 首请求到达后再等这么久收拢同型请求, 换 tile 填充率。 */
#define DS4_SERVER_BATCH_WAIT_MS 60

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

typedef struct server server;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_dsml;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char *wire_name;
    char *namespace;
    /* Distinguish the Responses hosted tool from a normal function that
     * happens to be named "tool_search". */
    bool responses_tool_search;
    char **prop;
    int len;
    int cap;
    /* input_schema "required" names — the guided primer emits exactly these
     * (multi-param tools like Edit die CC-side when only prop[0] is sent). */
    char **req;
    int req_len;
    int req_cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    char *role;
    char *content;
    char *reasoning;
    char *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len;
    int tool_call_ids_cap;
    tool_calls calls;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

typedef struct {
    req_kind kind;
    api_style api;
    ds4_tokens prompt;
    char *model;
    bool model_from_request;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    /* 会话区起点(字节偏移, 渲染器亲手记录): 值拷贝源边界。0 = 未知(退回 strstr
     * 启发式)。根因(2026-07-15 行为门实证): header 内容(--soul 示例/客户端 system)
     * 可以合法包含 "# User:" 字样, strstr 找首个标记会把拷贝边界劫持进 header,
     * 示例值全部变成可抄 → 无关问题也抄 soul 命令。 */
    size_t prompt_conv_off;
    tool_schema_orders tool_orders;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    /* OpenAI frequency_penalty / presence_penalty (chat completions + responses;
     * /v1/messages 无此字段)。0 = 未指定; 会话跨请求复用, 必须每请求落到会话
     * ((0,0) 即清除), 语义在核心采样器且只数生成区。 */
    float frequency_penalty;
    float presence_penalty;
    /* DRY 序列复读惩罚(2026-09-22 起可按请求给; 以前只有服务启动参数 --dry-*)。
     * dry_set = 请求里出现过 dry_multiplier ⇒ 这一条请求用请求里的三个值, 否则用服务默认。
     * 为什么要按请求: 复读惩罚该由调用方按活儿定(长报告要, 抄数不要), 而且扫参数不必每次重启服务(一次 100 s 装载)。 */
    float dry_multiplier, dry_base;
    int dry_allowed_length;
    bool dry_set;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    int cache_read_tokens;
    int cache_write_tokens;
    ds4_think_mode think_mode;
    bool has_tools;
    bool prompt_preserves_reasoning;
    /* For /v1/responses: emit reasoning_summary_* events / fields only when the
     * client opted in via reasoning.summary. Other APIs leave this false; the
     * field is ignored on those code paths. */
    bool reasoning_summary_emit;
    /* Responses continuation contract:
     *
     * A live Responses tool loop is not a normal "new prompt with a long
     * prefix" request.  The protocol gives tool outputs a call_id that binds
     * them to a prior assistant tool call.  If that call_id is still known in
     * memory, the live KV is the authoritative prefix, including any hidden
     * thinking that the client did not replay.  These fields carry the parsed
     * evidence needed by generate_job() to append only the new suffix.
     *
     * A tool-output-only request has no stateless prefix to match.  If the live
     * call_id binding is gone by the time the worker executes it, DS4 must ask
     * for a full replay rather than cold-prefilling a prompt that starts with a
     * naked tool result.  Similarly, if live state is gone, a reasoning-mode
     * tool replay must contain the prior reasoning item (or an equivalent
     * opaque reasoning state from a future implementation). */
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    stop_list responses_live_call_ids;
    char *responses_live_suffix_text;
    bool anthropic_requires_live_tool_state;
    stop_list anthropic_live_call_ids;
    char *anthropic_live_suffix_text;
    /* tool_choice forcing (Anthropic {"type":"tool","name":X} / {"type":"any"},
     * OpenAI "required"): the primer skips the free region and injects the frame
     * directly; a named force also pins the tool-name region to that literal. */
    char *tool_force_name;   /* named tool to force, or NULL */
    bool tool_force_any;     /* force some tool call (name still model-chosen) */
    tool_replay_stats tool_replay;
    int ds4_mode;            /* 接口显式意图(2026-07-25): 0=auto(问句启发式兜底) 1=code 2=qa */
} request;

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

#define DS4_DSML "｜DSML｜"

#define DS4_DSML_SHORT "DSML｜"

#define DS4_TOOL_CALLS_START "<" DS4_DSML "tool_calls>"

#define DS4_TOOL_CALLS_END "</" DS4_DSML "tool_calls>"

#define DS4_INVOKE_START "<" DS4_DSML "invoke"

#define DS4_INVOKE_END "</" DS4_DSML "invoke>"

#define DS4_PARAM_START "<" DS4_DSML "parameter"

#define DS4_PARAM_END "</" DS4_DSML "parameter>"

#define DS4_TOOL_CALLS_START_SHORT "<" DS4_DSML_SHORT "tool_calls>"

#define DS4_TOOL_CALLS_END_SHORT "</" DS4_DSML_SHORT "tool_calls>"

#define DS4_INVOKE_START_SHORT "<" DS4_DSML_SHORT "invoke"

#define DS4_INVOKE_END_SHORT "</" DS4_DSML_SHORT "invoke>"

#define DS4_PARAM_START_SHORT "<" DS4_DSML_SHORT "parameter"

#define DS4_PARAM_END_SHORT "</" DS4_DSML_SHORT "parameter>"

/* Streaming is a translation state machine over the raw DS4 text.  The model
 * may produce <think> and DSML tool blocks; clients should receive those as
 * protocol-native reasoning/tool deltas, never as visible assistant text. */
/* ---- Built-in browser chat page ----------------------------------------
 * GET /, /chat and /index.html serve web/chat.html verbatim.  The path is
 * resolved against the working directory — the same cwd contract as the
 * Metal shader sources under metal/, so --chdir fixes both.  Same-origin
 * serving keeps the
 * page's fetch()es CORS-free; --cors stays only for pages hosted elsewhere.
 * The file is re-read per request: it is tens of KB, and editing the page
 * then refreshing the browser must not require a server restart. */
#define DS4_CHAT_PAGE_FILE "web/chat.html"
/* GET /monitor 的监控页(2026-10-07): 同一套 cwd 契约, 零依赖单文件, 每秒拉一次同源 GET /metrics。 */
#define DS4_MONITOR_PAGE_FILE "web/monitor.html"

#endif /* DS4_SERVER_TYPES_H */
