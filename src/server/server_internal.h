#ifndef DS4_SERVER_INTERNAL_H
#define DS4_SERVER_INTERNAL_H
/* ds4_server.c 拆分内部接口: 原 static 函数按跨文件使用提升, 命名不变。
 * 测试也从这里取内部符号 (tests/server_tests_*.c)。 */
#include "server_types.h"
#include "server_types2.h"

void stop_signal_handler(int sig);
void die(const char *msg);
static inline void *xrealloc(void *p, size_t n) {   /* 同 xmalloc: 与 core 撞名, 保内部链接 */
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}
char *xstrdup(const char *s);
bool random_bytes(void *dst, size_t len);
char *xstrndup(const char *s, size_t n);
char *json_minify_raw_value(const char *json);
bool json_content(const char **p, char **out);
void random_tool_id(char *dst, size_t dstlen, api_style api);
void tool_call_free(tool_call *tc);
void tool_calls_free(tool_calls *calls);
void tool_calls_push(tool_calls *calls, tool_call tc);
void chat_msg_add_tool_call_id(chat_msg *m, const char *id);
void chat_msg_free(chat_msg *m);
void chat_msgs_free(chat_msgs *msgs);
void chat_msgs_push(chat_msgs *msgs, chat_msg msg);
void tool_schema_order_free(tool_schema_order *o);
void tool_schema_orders_free(tool_schema_orders *orders);
void tool_schema_order_prop_push(tool_schema_order *o, char *prop);
void tool_schema_order_req_push(tool_schema_order *o, char *req);
void tool_schema_orders_push(tool_schema_orders *orders, tool_schema_order order);
const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name);
void request_init(request *r, req_kind kind, int max_tokens);
void request_free(request *r);
ds4_think_mode think_mode_from_enabled(bool enabled, ds4_think_mode effort);
bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out);
bool parse_reasoning_effort_value(const char **p, ds4_think_mode *out);
bool parse_thinking_control_value(const char **p, bool *thinking_enabled);
bool parse_output_config_effort(const char **p, ds4_think_mode *effort);
bool model_alias_disables_thinking(const char *model);
bool model_alias_enables_thinking(const char *model);
const char *server_model_id_from_engine(ds4_engine *engine);
bool server_model_alias_known(const char *id);
void stop_list_clear(stop_list *stops);
void stop_list_push(stop_list *stops, char *s);
bool parse_stop(const char **p, stop_list *out);
bool stop_list_find_from(const stop_list *stops, const char *text, size_t from, size_t *pos, size_t *len);
size_t stop_list_stream_safe_len(const stop_list *stops, size_t text_len);
size_t utf8_stream_safe_len(const char *s, size_t start, size_t limit, bool final);
bool parse_stream_options(const char **p, bool *include_usage);
bool parse_tool_calls_value(const char **p, tool_calls *calls);
void append_raw_json_line(buf *b, const char *json);
char *openai_function_schema_from_tool(const char *raw);
char *responses_special_schema_from_tool(const char *raw);
char *responses_namespace_function_schema_from_tool(const char *raw, const char *namespace, char **wire_name);
void tool_schema_orders_add_json_wire(tool_schema_orders *orders, const char *json, const char *namespace, const char *wire_name, bool responses_tool_search);
void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json);
bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders);
bool parse_messages(const char **p, chat_msgs *msgs);
bool append_anthropic_block_content(buf *dst, const char *text);
bool json_parse_image_source(const char **p, char **src_type, char **src_media, char **src_data);
bool mm_image_source_to_text(ds4_mm *mm, const char *src_type, const char *src_media, const char *src_data, buf *dst);
bool json_tool_result_content(const char **p, ds4_mm *mm, char **out);
bool parse_anthropic_messages(const char **p, chat_msgs *msgs, ds4_mm *mm);
bool parse_anthropic_system(const char **p, char **out);
void knowledge_load(const char *path);
const char *knowledge_retrieve(const char *query);
void append_tools_prompt_text(buf *b, const char *tool_schemas);
void append_tool_result_text(buf *b, const char *s);
bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order);
void append_json_object_or_empty(buf *b, const char *json);
void append_dsml_tool_calls_text(buf *b, const tool_calls *calls);
bool role_is_system(const char *role);
bool chat_history_uses_tool_context(const chat_msgs *msgs, const char *tool_schemas);
void base_native_default_stops(stop_list *stops);
char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas, const tool_schema_orders *tool_orders, ds4_think_mode think_mode, size_t *conv_off);
bool responses_validate_tool_outputs(server *s, const chat_msgs *msgs, ds4_think_mode think_mode, bool *requires_live_tool_state, bool *requires_live_reasoning, char *err, size_t errlen);
void responses_prepare_live_continuation(request *r, const chat_msgs *msgs);
bool anthropic_validate_tool_results(server *s, const chat_msgs *msgs, bool *requires_live_tool_state, char *err, size_t errlen);
void anthropic_prepare_live_continuation(request *r, const chat_msgs *msgs);
bool parse_chat_request(ds4_engine *e, server *s, const char *body, int def_tokens, int ctx_size, request *r, char *err, size_t errlen);
bool parse_anthropic_request(ds4_engine *e, server *s, const char *body, int def_tokens, int ctx_size, request *r, char *err, size_t errlen);
bool parse_responses_content_array(const char **p, char **out);
bool parse_responses_input(const char **p, chat_msgs *msgs, buf *loaded_tool_schemas, tool_schema_orders *orders);
bool parse_responses_request(ds4_engine *e, server *s, const char *body, int def_tokens, int ctx_size, request *r, char *err, size_t errlen);
bool parse_completion_request(ds4_engine *e, const char *body, int def_tokens, int ctx_size, request *r, char *err, size_t errlen);
bool send_all(int fd, const void *p, size_t n);
void json_escape(buf *b, const char *s);
void json_escape_n(buf *b, const char *s, size_t n);
void json_escape_fragment_n(buf *b, const char *s, size_t n);
const char *find_any_tool_start(const char *s);
void observe_tool_markers(const char *scan, bool *saw_start, bool *saw_end, bool *orphan_end);
size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit);
const char *find_last_substr(const char *s, const char *needle);
char *dsml_unescape_text(const char *s);
char *dsml_attr(const char *tag, const char *name);
bool parse_generated_message_ex(const char *text, bool require_thinking_closed, char **content_out, char **reasoning_out, tool_calls *calls);
bool parse_generated_message_for_response(const char *text, bool has_tools, bool saw_tool_start, bool require_thinking_closed, const char **finish_io, char *err, size_t errlen, char **content_out, char **reasoning_out, tool_calls *calls, bool *recovered_out);
void append_json_object_string(buf *b, const char *json);
void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix, const tool_schema_orders *orders);
void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix, const tool_schema_orders *orders);
bool http_response(int fd, bool enable_cors, int code, const char *type, const char *body);
bool http_response_n(int fd, bool enable_cors, int code, const char *type, const char *body, size_t body_len);
bool http_error(int fd, bool enable_cors, int code, const char *msg);
bool request_exceeds_context(const request *r, int ctx_size);
bool http_error_context_length_exceeded(int fd, bool enable_cors, const request *r, int n_prompt_tokens, int ctx_size);
bool serve_chat_page(int fd, bool enable_cors, const char *path);
bool serve_page_file(int fd, bool enable_cors, const char *path);   /* 同 serve_chat_page, 404 文案报传入的路径(监控页用) */
bool path_route_is(const char *path, const char *route);
/* 监控(server_monitor*.c, 2026-10-07): GET /monitor 页 + GET /metrics(JSON / Prometheus 文本)。
 * 生命周期钩子: 客户端线程 mon_begin(入队前) → worker mon_prefill(开始预填) → mon_prefill_progress(每块) → mon_first_token(预填完)
 * → mon_token(每个 token) → mon_end(收尾; 幂等, 记录不在就不动)。id = 0 时全部钩子是空操作。 */
struct server_monitor *mon_open(server *s);
void mon_close(struct server_monitor *m);
uint64_t mon_begin(server *s, const request *r, const char *path);
void mon_prefill(server *s, uint64_t id, int prompt_tokens, int cached, int max_tokens);
void mon_prefill_progress(server *s, uint64_t id, int current, int total);
void mon_first_token(server *s, uint64_t id);
void mon_token(server *s, uint64_t id, int generated);
void mon_end(server *s, uint64_t id, const char *finish, int generated, int drafts_offered, int drafts_accepted);
/* 磁盘 KV 缓存事件(event = "parked" 存盘 / "restored" 命中装回); parked/bytes = 事后的条目数与总字节; evicted = 这次存盘挤掉几条 */
void mon_kv_event(server *s, const char *event, int tokens, int parked, uint64_t bytes, int evicted);
void mon_metrics_json(server *s, buf *b, bool all_requests);
void mon_prometheus_text(server *s, buf *b);
bool http_accepts_text(const char *headers, size_t n);   /* Accept 带 text/plain 或 openmetrics(server_httpd.c; /metrics 给 Prometheus 文本的判据) */
bool sse_headers(int fd, bool enable_cors);
bool sse_error_event(int fd, const request *r, const char *msg);
bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish);
int clamp_usage_tokens(int value, int max);
void append_openai_usage_json(buf *b, const request *r, int prompt_tokens, int completion_tokens);
bool sse_done(int fd, const request *r, const char *id, int prompt_tokens, int completion_tokens);
bool sse_chat_finish(int fd, const request *r, const char *id, const char *content, const char *reasoning, const tool_calls *calls, const char *finish, int prompt_tokens, int completion_tokens);
void openai_stream_start(const request *r, openai_stream *st);
void openai_stream_free(openai_stream *st);
bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit);
bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit);
bool raw_partial_any(const char *raw, size_t raw_len, size_t pos, const char *a, const char *b);
const char *find_lit_bounded(const char *s, size_t n, const char *lit);
bool dsml_decode_state_is_tool(dsml_decode_state state);
void dsml_decode_tracker_init(dsml_decode_tracker *dt);
void dsml_decode_tracker_update(dsml_decode_tracker *dt, const char *raw, size_t raw_len);
size_t tool_param_value_stream_safe_len(const char *raw, size_t start, size_t raw_len, const char *param_end, bool is_string);
bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id, openai_stream *st, const char *raw, size_t raw_len, bool final);
bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id, openai_stream *st, const char *raw, size_t raw_len, const tool_calls *calls, const char *finish, int prompt_tokens, int completion_tokens);
bool request_uses_openai_live_stream(const request *r);
bool request_uses_responses_live_stream(const request *r);
bool request_uses_structured_stream(const request *r);
void responses_random_id(char *dst, size_t dstlen, const char *prefix);
void responses_stream_init(const request *r, responses_stream *st);
void responses_stream_free(responses_stream *st);
bool responses_sse_emit_event(int fd, responses_stream *st, const char *body);
bool responses_sse_created(int fd, const request *r, responses_stream *st, long created_at);
bool responses_sse_reasoning_added(int fd, responses_stream *st);
bool responses_sse_reasoning_summary_part_added(int fd, responses_stream *st);
bool responses_sse_reasoning_delta(int fd, responses_stream *st, const char *text, size_t len);
const char *responses_item_status_for_finish(const char *finish);
bool responses_sse_reasoning_done(int fd, responses_stream *st, const char *finish);
bool responses_sse_message_added(int fd, responses_stream *st);
bool responses_sse_message_text_part_added(int fd, responses_stream *st);
bool responses_sse_output_text_delta(int fd, responses_stream *st, const char *text, size_t len);
bool responses_sse_message_done(int fd, responses_stream *st, const char *finish);
void responses_tool_items_build(responses_tool_item **out, const tool_calls *calls, int starting_output_index);
void responses_append_function_call_item(buf *b, const tool_call *tc, const responses_tool_item *item, const char *item_status, bool with_args, const tool_schema_orders *orders);
bool responses_sse_function_call_event(int fd, responses_stream *st, const tool_call *tc, const responses_tool_item *item, const tool_schema_orders *orders, const char *finish, bool done);
bool responses_sse_function_call_arguments_done(int fd, responses_stream *st, const tool_call *tc, const responses_tool_item *item, const tool_schema_orders *orders);
const char *responses_status_for_finish(const char *finish);
void append_responses_usage_json(buf *b, const request *r, int input_tokens, int output_tokens);
bool responses_sse_completed(int fd, const request *r, responses_stream *st, const tool_calls *calls, const responses_tool_item *tool_items, const char *finish, int prompt_tokens, int completion_tokens, long created_at);
bool responses_sse_stream_update(int fd, const request *r, responses_stream *st, const char *raw, size_t raw_len, bool final);
bool responses_sse_finish_live(int fd, const request *r, responses_stream *st, const char *raw, size_t raw_len, const char *recovered_content, const tool_calls *calls, const char *finish, int prompt_tokens, int completion_tokens, long created_at);
bool responses_final_response(int fd, bool enable_cors, const request *r, const char *id, const char *text, const char *reasoning, const tool_calls *calls, const char *finish, int prompt_tokens, int completion_tokens);
bool final_response(int fd, bool enable_cors, const request *r, const char *id, const char *text, const char *reasoning, const tool_calls *calls, const char *finish, int prompt_tokens, int completion_tokens);
const char *anthropic_stop_reason(const char *finish);
void append_anthropic_content(buf *b, const char *text, const char *reasoning, const tool_calls *calls, const char *id_prefix, const tool_schema_orders *orders);
bool anthropic_final_response(int fd, bool enable_cors, const request *r, const char *id, const char *text, const char *reasoning, const tool_calls *calls, const char *finish, int prompt_tokens, int completion_tokens);
bool sse_event(int fd, const char *event, const char *data);
bool anthropic_sse_start_live(int fd, const request *r, const char *id, int prompt_tokens, anthropic_stream *st);
void anthropic_stream_free(anthropic_stream *st);
bool anthropic_sse_open_block(int fd, anthropic_stream *st, anthropic_block_type type);
bool anthropic_sse_delta_live(int fd, const anthropic_stream *st, anthropic_block_type type, const char *text, size_t len);
bool anthropic_sse_close_block_live(int fd, const char *id, anthropic_stream *st);
bool anthropic_tool_emit_args_fragment(int fd, anthropic_stream *st, const char *text, size_t len);
bool anthropic_tool_emit_string_value(int fd, anthropic_stream *st, const char *text, size_t len);
bool anthropic_tool_stream_init(anthropic_tool_stream *ts, const char *raw, size_t raw_len, size_t pos);
bool anthropic_tool_stream_fail(anthropic_tool_stream *ts);
bool anthropic_tool_start_invoke(int fd, server *s, anthropic_stream *st, const char *raw, size_t raw_len);
bool anthropic_tool_start_param(int fd, anthropic_stream *st, const char *raw, size_t raw_len);
bool anthropic_tool_finish_param(int fd, anthropic_stream *st, const char *raw, size_t value_end);
size_t text_stream_safe_limit(const char *raw, size_t start, size_t raw_len, bool has_tools, bool final);
bool anthropic_sse_stream_update(int fd, server *s, const request *r, const char *id, anthropic_stream *st, const char *raw, size_t raw_len, bool final);
bool anthropic_sse_finish_live(int fd, server *s, const request *r, const char *id, anthropic_stream *st, const char *raw, size_t raw_len, const tool_calls *calls, const char *finish, int completion_tokens);
double now_sec(void);
void server_log(ds4_log_type type, const char *fmt, ...);
int tool_memory_max_entries(const tool_memory *m);
tool_memory_block *tool_memory_find_block_locked(tool_memory *m, const char *dsml, size_t len);
void tool_memory_free(tool_memory *m);
void live_tool_state_free(live_tool_state *st);
void visible_live_free(visible_live_state *st);
void thinking_live_clear(server *s);
void thinking_live_remember(server *s, const char *visible_text);
void responses_live_remember(server *s, const char *visible_text, const tool_calls *calls);
void anthropic_live_remember(server *s, const tool_calls *calls);
void responses_live_clear(server *s);
void anthropic_live_clear(server *s);
bool responses_live_has_call_id(server *s, const char *id);
bool anthropic_live_has_call_id(server *s, const char *id);
bool responses_live_matches_request(server *s, const stop_list *ids, int live_tokens);
bool anthropic_live_matches_request(server *s, const stop_list *ids, int live_tokens);
bool tool_memory_has_id(server *s, const char *id);
const char *tool_memory_lookup_locked(tool_memory *m, const char *id, tool_memory_source *source, tool_memory_block **block);
void tool_memory_remember(server *s, const tool_calls *calls);
void tool_memory_put_source(server *s, const char *id, const char *dsml, tool_memory_source source);
void tool_memory_attach_to_messages(server *s, chat_msgs *msgs, tool_replay_stats *stats);
void assign_tool_call_ids(server *s, tool_calls *calls, api_style api);
void apply_openai_stream_tool_ids(tool_calls *calls, const openai_stream *st);
void apply_anthropic_stream_tool_ids(tool_calls *calls, const anthropic_stream *st);
kv_cache_options kv_cache_default_options(void);
void le_put32(uint8_t *p, uint32_t v);
bool id_list_contains(const stop_list *ids, const char *id);
void id_list_push_unique(stop_list *ids, const char *id);
void id_list_free(stop_list *ids);
void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids);
char *path_join(const char *dir, const char *name);
bool kv_tool_map_serialized_size(server *s, const char *text, uint64_t *bytes_out);
bool kv_tool_map_write(server *s, FILE *fp, const char *text, uint64_t *written_bytes);
int kv_tool_map_load_from_pos(server *s, FILE *fp, const stop_list *wanted);
void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs);
bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb, bool reject_different_quant, kv_cache_options opt);
void kv_cache_close(kv_disk_cache *kc);
char *render_tokens_text(ds4_engine *engine, const ds4_tokens *tokens, size_t *out_len);
void tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n);
void build_prompt_from_exact_prefix_and_text_suffix( ds4_engine *engine, const ds4_tokens *exact_prefix, const char *suffix_text, ds4_tokens *out);
int kv_cache_store_len(const kv_disk_cache *kc, int tokens);
int kv_cache_chat_anchor_pos(const kv_disk_cache *kc, const ds4_tokens *prompt, int user_token_id, int assistant_token_id);
int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens);
bool kv_cache_store_live_prefix(server *s, const ds4_tokens *tokens, int store_len, const char *reason);
void kv_cache_store_current(server *s, const char *reason);
void kv_cache_note_store(kv_disk_cache *kc, int tokens);
int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens);
void kv_cache_restore_suppressed_continued(kv_disk_cache *kc, int old_tokens, int suppressed_tokens);
void kv_cache_discard_failed_disk_entry(server *s, const char *path);
void kv_cache_maybe_store_continued(server *s);
int kv_cache_try_load_text(server *s, const char *prompt_text, ds4_tokens *effective_prompt, char **loaded_path_out, uint8_t *loaded_ext_flags_out, bool responses_protocol);
int kv_cache_try_load(server *s, const request *req, ds4_tokens *effective_prompt, char **loaded_path_out, uint8_t *loaded_ext_flags_out);
int live_text_prefix_prompt(server *s, const request *req, ds4_tokens *effective_prompt);
int responses_live_continuation_prompt(server *s, const request *req, int live_pos, ds4_tokens *effective_prompt, int *matched_ids);
int anthropic_live_continuation_prompt(server *s, const request *req, int live_pos, ds4_tokens *effective_prompt, int *matched_ids);
int responses_live_visible_prefix_prompt(server *s, const request *req, int live_pos, ds4_tokens *effective_prompt);
int thinking_live_visible_prefix_prompt(server *s, const request *req, int live_pos, ds4_tokens *effective_prompt);
void trace_cache_capture( trace_cache_diag *d, const ds4_tokens *live, const ds4_tokens *prompt, int old_pos, int common);
const char *trace_cache_miss_reason(const trace_cache_diag *d);
uint64_t trace_begin( server *s, const job *j, int cached, int effective_prompt_tokens, const trace_cache_diag *cache_diag, const char *cache_source, int disk_cached, const char *disk_path);
void trace_piece(server *s, uint64_t id, const char *piece, size_t len);
void trace_token_ids(server *s, uint64_t id, const int *prompt, int n_prompt, const int32_t *gen, int n_gen);
void trace_event(server *s, uint64_t id, const char *fmt, ...);
void trace_finish( server *s, uint64_t id, const request *r, const char *final_finish, int completion, bool saw_tool_start, bool saw_tool_end, const char *parsed_content, const char *parsed_reasoning, const tool_calls *parsed_calls, double elapsed);
void request_ctx_span(char *buf, size_t len, int cached, int prompt);
void log_flags(char *buf, size_t len, bool responses_protocol, bool tools, bool thinking, bool dsml_start, bool dsml_end);
void log_decode_progress(req_kind kind, int prompt_tokens, int completion, bool responses_protocol, bool tools, bool thinking, bool dsml_start, bool dsml_end, double decode_t0, double *last_t, int *last_completion);
void thinking_state_feed(thinking_state *st, const char *p, size_t len);
thinking_state thinking_state_from_prompt(const request *r);
char *build_invalid_dsml_tool_error_suffix(const request *r, const thinking_state *thinking, const char *detail);
bool continue_after_invalid_dsml(server *s, const request *r, const thinking_state *thinking, const char *detail, int *tokens_appended, char *err, size_t errlen);
bool should_remember_thinking_checkpoint(const request *r, const thinking_state *thinking, const char *finish);
void log_tool_calls_summary(const char *ctx, const tool_calls *calls, bool responses_protocol);
void server_progress_cb(void *ud, const char *event, int current, int total);
/* V4.1 服务生成路(server_generate_v41.c): 没有会话, token 从引擎回调来; generate_job 入口按模型分流 */
void generate_job_v41(server *s, job *j);
/* 一条 V4.1 请求的生成期状态(2026-09-30 从 server_generate_v41.c 搬到头文件: 并发调度器 server_sched_v41.c 也要用)。
 * 三段: v41_gen_begin(校验/id/trace/采样面; 失败已回响应) → 引擎每出一个 token 调 v41_emit(文本累积/思考段/DSML/停止串/流式/挂断探测)
 * → v41_gen_end(收尾响应/trace/释放)。单请求路(generate_job_v41)与并发路走同一套, 客户端看到的字节一样。 */
typedef struct {
    server *s;
    job *j;
    char id[96];
    uint64_t trace_id;
    char ctx_span[48];
    char req_flags[64];
    server_prefill_progress progress;
    int prompt_tokens, max_tokens, completion, eos;
    buf text;
    size_t plain_stream_pos, stop_scan_from, tool_scan_from;
    const char *finish;
    char err[256];
    bool started;             /* 第一个 token 到了 = 预填结束; 流式头/角色块/live 流在这一刻起 */
    bool stream_dead;         /* 客户端流写失败: 后面的 token 只累积不再写 */
    thinking_state thinking;
    bool thinking_gates_tool_markers, tool_scan_waiting_for_think_close;
    dsml_decode_tracker dsml_tracker;
    bool saw_tool_start, saw_tool_end, saw_orphan_tool_end;
    int next_tool_progress, next_decode_log;
    double t0, decode_t0, last_decode_log_t;
    int last_decode_log_completion;
    int next_alive_check;     /* 下一次探"客户端还在不在"的生成位(见 V41_ALIVE_CHECK_TOKENS) */
    bool client_gone;         /* 探到对端已挂断: 预填/解码都立刻停, 收尾不再当成引擎故障 */
    bool responses_protocol, structured_stream, openai_live_chat, responses_live_chat;
    anthropic_stream anthropic_live;
    openai_stream openai_live;
    responses_stream responses_live;
    long responses_created_at;
    int32_t *ids; int n_ids, cap_ids;   /* 引擎真吐的 token id(含 EOS): 收尾写进 --trace, 真实请求才能逐位重放对拍(bug.md §5.4 E0) */
    ds4_decode_sampling sp;   /* 本请求的采样面(请求没带的落到 ds4.h 的官方默认); 单 worker 路覆写全局, 并发路按请求传给引擎 */
    int spec_rounds, spec_offered, spec_accepted;   /* 投机账(调用方在 v41_gen_end 前从引擎取): 监控页的草稿接受率; rounds=0 = 没投机 */
} v41_gen;
/* 每多少个 token 探一次对端: 一次 poll 约 1 µs, 一步解码 40 ms ⇒ 开销在噪声里; 最坏多算 16 个 token。 */
enum { V41_ALIVE_CHECK_TOKENS = 16 };
bool v41_gen_begin(server *s, job *j, v41_gen *g);   /* false = 请求不合法, 错误响应已发 */
int v41_emit(int token, void *ud);                    /* 引擎逐 token 回调; 非 0 = 停 */
int v41_progress_cb(void *ud, const char *event, int current, int total);   /* 预填块间: 心跳 + 探客户端; 非 0 = 中止 */
void v41_gen_end(v41_gen *g, int rc);                 /* rc = 引擎返回码(非 0 且没出过 token = 预填失败) */
/* V4.1 并发调度器(server_sched_v41.c, batch.md §3.2): --batch N ≥ 2 时 worker 线程整个交给它 */
void v41_sched_run(server *s);
/* 热切侧车/后训练件(server_plugins.c): init 起服时挂上训练页模块的钩子; apply 由 worker 在没有活着的请求时调, 做完释放 j */
void server_plugins_init(server *s, const char *gguf);
void server_plugins_apply(server *s, job *j);
/* 任务队列(server_batch.c): dequeue 阻塞到有 job 或服务在停(NULL); dequeue_try 不阻塞; job_finish 唤醒等着的客户端线程 */
job *dequeue(server *s);
job *dequeue_try(server *s);
void job_finish(job *j);
/* 上下文大小: V4 从会话读, V4.1 没有会话(s->session == NULL)就用起服时定下的 s->ctx_size */
static inline int server_ctx_size(const server *s) { return s->session ? ds4_session_ctx(s->session) : s->ctx_size; }
void send_prefill_failure_response(server *s, const job *j, const server_prefill_progress *progress, const char *ctx, const char *flags, const char *err);
char *build_tool_checkpoint_suffix(const request *r, const char *content, const char *reasoning, const tool_calls *calls);
char *build_responses_visible_assistant_suffix(const request *r, const char *content, const char *reasoning, const tool_calls *calls);
char *build_toolless_thinking_visible_text(const request *r, const char *content);
void remember_thinking_checkpoint(server *s, const job *j, const char *ctx, uint64_t trace_id, const char *content);
void canonicalize_tool_checkpoint(server *s, const job *j, const char *ctx, uint64_t trace_id, const char *content, const char *reasoning, const tool_calls *calls);
bool should_canonicalize_tool_checkpoint(const server *s, const tool_calls *calls);
void primer_copy_init(primer_copy *cm, const char *src, size_t len, bool anchored);
int primer_copy_step(server *s, primer_copy *cm, const char *stopchars, int stop_tok, bool hard, bool *fell_back);
int primer_divergence_token(server *s, int exclude_tok);
void generate_job(server *s, job *j);
bool enqueue(server *s, job *j);
void *worker_main(void *arg);
void append_model_json_values(buf *b, const char *id, const char *name, int ctx, int default_tokens);
void *client_main(void *arg);
int listen_on(const char *host, int port);
void configure_client_socket(int fd);
void set_client_socket_nonblocking(int fd);
/* 生成期间每隔几个 token 探一次对端(server_http.c): true = 客户端已经走了, 立刻停生成 */
bool client_disconnected(int fd);
void log_context_memory(ds4_backend backend, int ctx_size);
void server_close_resources(server *s);
void usage(FILE *fp);
server_config parse_options(int argc, char **argv);

#ifdef DS4_SERVER_TEST
/* 测试构建专用符号 (定义在各 .c 的 #ifdef DS4_SERVER_TEST 块里) */
dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len);
void tool_memory_put(server *s, const char *id, const char *dsml);
void sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits, uint8_t reason, uint8_t ext_flags, uint32_t tokens, uint32_t hits, uint32_t ctx_size, uint64_t created_at, uint64_t last_used, uint64_t payload_bytes);
double kv_entry_eviction_score(const kv_entry *e, const ds4_tokens *live, uint64_t now, const ds4_kvstore_eviction_context *incoming);
void kv_cache_evict(kv_disk_cache *kc, const ds4_tokens *live, uint64_t extra_bytes, const ds4_kvstore_eviction_context *incoming);
bool kv_cache_file_size_fits(const kv_disk_cache *kc, uint64_t text_bytes, uint64_t payload_bytes, uint64_t tool_map_bytes, uint64_t *file_bytes_out, uint64_t *required_bytes_out);
int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text, int quant_bits, int ctx_size);
#endif /* DS4_SERVER_TEST */

/* xmalloc: ds4.c 已有同名外部符号(ds4_internal.h), server 侧同名实现为避免
 * 链接冲突保持内部链接, 以 static inline 形式共享给全部 server 编译单元。 */
static inline void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}
#endif /* DS4_SERVER_INTERNAL_H */
