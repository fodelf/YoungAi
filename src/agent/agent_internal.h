/* agent_internal.h — ds4_agent.c 机械拆分后的内部接口。
 * 原单文件里跨拆分文件使用的 static 函数/变量在这里声明(去 static, 符号名不变);
 * 只在单个 .c 内用的仍是该文件的 static。除 main 外这些符号仍是 agent 私有约定,
 * 其它程序不 include 本头。 */
#ifndef DS4_AGENT_INTERNAL_H
#define DS4_AGENT_INTERNAL_H

#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_kvstore.h"
#include "ds4_web.h"
#include "linenoise.h"

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* This is intentionally not in linenoise.h, but it is part of the existing
 * multiplexed editor implementation.  The agent uses it only to restore text
 * after Enter is pressed while the model is still busy. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

#include "agent_types.h"

/* ds4.c 也导出全局 xmalloc, 同名不能去 static 导出; 原样保持内部链接,
 * 以 static inline 落在头里让每个拆分文件都拿到同一份实现。 */
static inline void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4-agent: malloc");
        exit(1);
    }
    return p;
}

/* ---- 跨文件函数/变量声明 (按原 ds4_agent.c 中的定义顺序) ---- */
extern volatile sig_atomic_t agent_sigint;
extern agent_worker *agent_completion_worker;
void agent_sigint_handler(int sig);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);
void *xrealloc(void *ptr, size_t n);
void write_all(int fd, const char *p, size_t n);
void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n);
char *agent_input_buf_take(agent_input_buf *b);
void agent_input_buf_free(agent_input_buf *b);
bool parse_power_percent(const char *arg, int *out);
bool agent_slash_command_known(const char *cmd);
double now_sec(void);
agent_config parse_options(int argc, char **argv);
void log_context_memory(ds4_backend backend, int ctx_size);
ds4_think_mode effective_think_mode(const agent_config *cfg);
extern const char agent_dsml_syntax_reminder[];
void agent_append_system_prompt(ds4_engine *engine, ds4_tokens *tokens, const char *extra);
void agent_worker_note_system_prompt_seen(agent_worker *w);
void agent_worker_maybe_append_datetime_context(agent_worker *w);
void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w);
void agent_wake_locked(agent_worker *w);
void agent_publish(agent_worker *w, const char *s, size_t n);
void agent_publishf(agent_worker *w, const char *fmt, ...);
void agent_set_status(agent_worker *w, agent_worker_state state);
void agent_set_error(agent_worker *w, const char *msg);
void agent_trace(agent_worker *w, const char *fmt, ...);
void agent_trace_token(agent_worker *w, int token, const char *text, size_t text_len, int index);
void agent_trace_tokens(agent_worker *w, const char *label, const ds4_tokens *tokens, int start);
void agent_trace_text(agent_worker *w, const char *label, const char *text, size_t len);
bool bytes_has_prefix(const char *p, size_t n, const char *prefix);
bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix);
const char *agent_tool_arg_value(const agent_tool_call *call, const char *name);
void agent_dsml_parser_free(agent_dsml_parser *p);
void agent_dsml_parser_reset(agent_dsml_parser *p);
void agent_dsml_start(agent_dsml_parser *p);
void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n);
char *agent_tail_capture_take(agent_tail_capture *t, size_t *len);
void renderer_write(agent_token_renderer *r, const char *s, size_t n);
void renderer_reset_color(agent_token_renderer *r);
void renderer_restore_text_attrs(agent_token_renderer *r);
void renderer_flush_utf8(agent_token_renderer *r);
void renderer_write_char_raw(agent_token_renderer *r, char c);
void renderer_write_plain_byte(agent_token_renderer *r, char c);
extern const agent_syntax agent_syntaxes[];
const agent_syntax *agent_syntax_for_lang(const char *lang);
const agent_syntax *agent_syntax_for_path(const char *path);
int renderer_terminal_cols(void);
void renderer_code_emit_buffered_line(agent_token_renderer *r, bool with_newline);
void renderer_code_byte(agent_token_renderer *r, char c);
void renderer_code_emit_backtick_literals(agent_token_renderer *r, size_t count);
void renderer_code_begin(agent_token_renderer *r);
void renderer_code_stream_begin(agent_token_renderer *r, const agent_syntax *syntax);
void renderer_code_stream_set_prefix(agent_token_renderer *r, const char *prefix, const char *color);
void renderer_code_stream_set_upto_marker(agent_token_renderer *r, bool enabled);
void renderer_code_end(agent_token_renderer *r);
void renderer_write_char(agent_token_renderer *r, char c);
void renderer_finish(agent_token_renderer *r);
void renderer_color(agent_token_renderer *r, const char *seq);
void renderer_plain(agent_token_renderer *r, const char *s, size_t n);
void agent_tool_viz_write(agent_stream_renderer *sr, const char *s, size_t n);
void agent_tool_viz_puts(agent_stream_renderer *sr, const char *s);
void agent_tool_viz_start(agent_stream_renderer *sr);
void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name);
void agent_tool_viz_render_read(agent_stream_renderer *sr);
void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name);
void agent_tool_viz_param_end(agent_stream_renderer *sr);
void agent_tool_viz_restore_param_color(agent_stream_renderer *sr);
void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c);
void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status);
void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish);
void worker_progress_cb(void *ud, const char *event, int current, int total);
bool worker_should_interrupt(agent_worker *w);
void agent_buf_append(agent_buf *b, const char *s, size_t n);
void agent_buf_puts(agent_buf *b, const char *s);
char *agent_buf_take(agent_buf *b);
bool agent_tokens_equal(const ds4_tokens *a, const ds4_tokens *b);
bool agent_mkdir_p(const char *path);
char *agent_default_cache_dir(void);
char *agent_kv_path_for_sha(const char *dir, const char sha[41]);
void agent_session_identity_sha(const char *title, uint64_t created_at, char sha_out[41]);
void agent_worker_clear_session_identity(agent_worker *w);
void agent_kv_session_meta_free(agent_kv_session_meta *m);
bool agent_kv_read_text(FILE *fp, uint32_t text_bytes, char **text_out, char *err, size_t err_len);
bool agent_kv_write_title_trailer(FILE *fp, const char *title, char *err, size_t err_len);
bool agent_kv_read_title_trailer(FILE *fp, const ds4_kvstore_entry *hdr, char **title_out, char *err, size_t err_len);
void agent_kv_identity_sha(const ds4_kvstore_entry *hdr, const char *text, uint32_t text_bytes, const char *title, char sha_out[41]);
bool agent_kv_load_path(agent_worker *w, const char *path, const char *expected_sha, const char *expected_text, size_t expected_text_len, ds4_tokens *loaded_tokens, agent_kv_session_meta *meta_out, char *err, size_t err_len);
bool agent_kv_save_path(agent_worker *w, const char *path, const ds4_tokens *tokens, const char *reason, char sha_out[41], const char *session_title, uint64_t session_created_at, char *err, size_t err_len);
void agent_worker_build_system_tokens(agent_worker *w, ds4_tokens *out);
void agent_publish_system_status(agent_worker *w, const char *msg);
int agent_web_confirm(void *privdata, const char *message, char *err, size_t err_len);
void agent_web_log(void *privdata, const char *message);
bool worker_take_web_approval_request(agent_worker *w, char *message, size_t message_len);
void worker_answer_web_approval(agent_worker *w, bool allow, const char *deny_error);
char *worker_request_queued_user_drain(agent_worker *w);
bool worker_take_queued_user_drain_request(agent_worker *w);
void worker_answer_queued_user_drain(agent_worker *w, char *text);
int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens, bool publish_progress, char *err, size_t err_len);
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len);
bool agent_worker_wait_distributed_route(agent_worker *w, char *err, size_t err_len);
bool agent_worker_has_user_session(agent_worker *w);
bool agent_worker_needs_save(agent_worker *w);
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41], int *tokens_out, char *err, size_t err_len);
bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len);
void agent_format_age(uint64_t when, char *buf, size_t len);
char *agent_session_title_from_prompt(const char *prompt, size_t max_bytes);
char *agent_session_title_from_text(const char *text, size_t text_len, size_t max_bytes);
char *agent_session_title_from_file(const char *path, size_t max_bytes);
const char *agent_history_next_marker(const char *p, const char *end, agent_history_mark *mark, size_t *mark_len);
void agent_history_trim(const char **p, const char **end);
bool agent_history_is_tool_user(const char *p, const char *end);
const char *agent_history_start_for_turns(const char *text, size_t len, int user_turns, bool *tool_only);
void agent_history_render_compaction_summary(agent_worker *w, const char *text, size_t len);
void agent_history_publish_limited(agent_worker *w, const char *p, const char *end, int max_lines, size_t max_bytes);
void agent_history_render_assistant(agent_worker *w, const char *p, const char *end);
bool agent_worker_show_history(agent_worker *w, int user_turns, char *err, size_t err_len);
void agent_worker_list_sessions(agent_worker *w);
void agent_switch_completion_callback(const char *buf, linenoiseCompletions *lc);
bool agent_worker_find_session(agent_worker *w, const char *prefix, char sha_out[41], char **path_out, char *err, size_t err_len);
bool agent_worker_delete_session(agent_worker *w, const char *prefix, char sha_out[41], char *err, size_t err_len);
bool agent_worker_strip_session(agent_worker *w, const char *prefix, char sha_out[41], uint32_t *tokens_out, char *err, size_t err_len);
bool agent_worker_switch_session(agent_worker *w, const char *prefix, int history_turns, char *err, size_t err_len);
int agent_parse_timeout(const char *s);
int agent_parse_int_default(const char *s, int def, int min, int max);
bool agent_parse_bool_default(const char *s, bool def);
void agent_line_spans_free(agent_line_spans *spans);
void agent_split_lines(const char *data, size_t len, agent_line_spans *spans);
int agent_read_file_bytes(const char *path, char **data, size_t *len, char *err, size_t errlen);
bool agent_old_new_line_effect(const char *old_data, size_t old_len, const char *new_data, size_t new_len, size_t edit_offset, size_t replaced_len, int *start_line, int *end_line, int *delta);
char *agent_edit_result(const char *path, int start_line, int end_line, int delta, const char *new_data, size_t new_len, const char *kind);
bool agent_tool_result_fits_context(agent_worker *w, const char *result, int reserve_tokens, int *tokens_out);
char *agent_tool_read(agent_worker *w, const agent_tool_call *call);
char *agent_tool_more(agent_worker *w, const agent_tool_call *call);
char *agent_tool_write(agent_worker *w, const agent_tool_call *call);
char *agent_tool_list(const agent_tool_call *call);
void agent_edit_result_append_context(agent_buf *b, const char *path, const char *data, size_t len, int anchor_start, int anchor_end);
bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer, agent_dsml_parser *p, const char *next_text, size_t next_len);
bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call, char *err, size_t err_len);
char *agent_tool_edit(agent_worker *w, const agent_tool_call *call);
char *agent_tool_search(agent_worker *w, const agent_tool_call *call);
char *agent_tool_google_search(agent_worker *w, const agent_tool_call *call);
char *agent_tool_visit_page(agent_worker *w, const agent_tool_call *call);
void agent_bash_jobs_free(agent_worker *w);
agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid);
void agent_bash_remove_job(agent_worker *w, agent_bash_job *target);
agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd, int timeout_sec, char *err, size_t err_len);
char *agent_bash_observation(agent_bash_job *job, bool mark_observed);
char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job, bool wait, int refresh_sec, bool stop, bool remove_if_done);
int agent_tool_job_id(const agent_tool_call *call);
pid_t agent_tool_pid(const agent_tool_call *call);
char *agent_execute_tool_calls(agent_worker *w, const agent_tool_calls *calls);
char *agent_bash_jobs_compaction_observation(agent_worker *w);
bool agent_worker_compact(agent_worker *w, const char *reason, char *err, size_t err_len);
bool agent_worker_compact_if_needed(agent_worker *w, const char *reason, char *err, size_t err_len);
int worker_accept_generated_token(agent_worker *w, int token, int *generated, double t0, agent_stream_renderer *stream, char *err, size_t err_len);
int worker_force_generated_text(agent_worker *w, const char *text, int max_tokens, int *generated, double t0, agent_stream_renderer *stream, char *err, size_t err_len);
void worker_request_save(agent_worker *w);
void worker_request_compact(agent_worker *w);
void worker_request_power(agent_worker *w, int power);
void worker_apply_pending_power(agent_worker *w);
void *worker_main(void *arg);
int set_nonblock(int fd, bool on, int *old_flags);
void drain_wake_fd(int fd);
bool worker_submit(agent_worker *w, const char *text);
void worker_interrupt(agent_worker *w);
void worker_stop(agent_worker *w);
void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status);
void worker_get_status(agent_worker *w, agent_status *status);
bool worker_is_idle(agent_worker *w);
bool worker_is_initialized(agent_worker *w, agent_status *status);
bool stdout_is_tty(void);
char *agent_format_user_prompt_echo(const char *text);
void agent_echo_user_prompt(const char *text);
void build_prompt_text(const agent_status *st, char *buf, size_t len);
unsigned agent_next_prefill_label(void);
void agent_prompt_queue_push(agent_prompt_queue *q, const char *text);
char *agent_prompt_queue_pop(agent_prompt_queue *q);
void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text);
char *agent_prompt_queue_take_all(agent_prompt_queue *q);
char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q);
void agent_prompt_queue_free(agent_prompt_queue *q);
bool agent_footer_is_multiline(const char *status);
void build_footer_text(const agent_status *st, const agent_prompt_queue *queue, int cols, char *buf, size_t len);
void editor_read_stdin(agent_editor *ed);
bool editor_take_queued_byte(agent_editor *ed, unsigned char byte);
bool editor_take_bare_escape(agent_editor *ed);
void editor_replace_input(agent_editor *ed, const char *text);
void editor_note_output(agent_editor *ed, const char *text, size_t len);
void editor_write_terminal_text(const char *text, size_t len);
bool editor_query_cursor(agent_editor *ed, int *col_out);
void editor_move_to_output_cursor(agent_editor *ed);
bool editor_get_terminal_size(int *rows, int *cols);
void editor_csi_cursor(int row, int col);
void editor_save_output_cursor(agent_editor *ed);
void editor_restore_output_cursor(agent_editor *ed);
void editor_move_to_prompt_row(agent_editor *ed);
void editor_move_to_prompt_cursor(agent_editor *ed);
void editor_clear_prompt_region(agent_editor *ed);
void editor_scroll_output_up(int bottom, int lines);
bool editor_set_scroll_layout(agent_editor *ed, int reserved_rows, bool allow_shrink, bool scroll_on_grow);
void editor_restore_terminal_layout(agent_editor *ed);
int editor_start(agent_editor *ed, const char *prompt, const char *status, const char *initial);
void editor_stop(agent_editor *ed);
void editor_hide(agent_editor *ed);
void editor_show(agent_editor *ed);
void editor_set_prompt_status(agent_editor *ed, const char *prompt, const char *status);
void editor_write_async(agent_editor *ed, const char *text, size_t len, const char *prompt, const char *status, bool force_show);
void editor_cancel_input_with_hint(agent_editor *ed, const char *prompt, const char *status);
void runtime_help(void);
void agent_format_ctx_size(int ctx_size, char *buf, size_t len);
void editor_write_welcome_banner(agent_editor *editor, const agent_config *cfg, const char *prompt, const char *statusline);
int agent_worker_init(agent_worker *w, ds4_engine *engine, agent_config *cfg);
void agent_worker_free(agent_worker *w);
bool agent_prompt_yes_no_ex(const char *prompt, const agent_yes_no_options *opts, bool *timed_out);
bool agent_maybe_save_before_leaving_session(agent_worker *w);
agent_exit_save_result agent_maybe_save_before_exiting(agent_worker *w);
int run_agent_non_interactive(ds4_engine *engine, agent_config *cfg);

#endif /* DS4_AGENT_INTERNAL_H */
