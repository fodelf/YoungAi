/* server_gen_log.c — 机械拆分自 ds4_server.c (9931-10173 行): 生成日志/思考态跟踪/无效 DSML 恢复。 */

#include "server_internal.h"

void request_ctx_span(char *buf, size_t len, int cached, int prompt) {
    int suffix = prompt - cached;
    if (suffix < 0) suffix = 0;
    snprintf(buf, len, "%d..%d:%d", cached, prompt, suffix);
}

void log_flags(char *buf, size_t len, bool responses_protocol,
                      bool tools, bool thinking,
                      bool dsml_start, bool dsml_end) {
    size_t used = 0;
    buf[0] = '\0';
#define ADD_FLAG(name) do { \
    int n = snprintf(buf + used, used < len ? len - used : 0, "%s%s", used ? " " : "", name); \
    if (n > 0) used += (size_t)n; \
} while (0)
    if (responses_protocol) ADD_FLAG("RESPPROTO");
    if (tools) ADD_FLAG("TOOLS");
    if (thinking) ADD_FLAG("THINKING");
    if (dsml_start) ADD_FLAG("DSML_START");
    if (dsml_end) ADD_FLAG("DSML_END");
#undef ADD_FLAG
}

void log_decode_progress(req_kind kind, int prompt_tokens, int completion,
                                bool responses_protocol,
                                bool tools, bool thinking,
                                bool dsml_start, bool dsml_end,
                                double decode_t0,
                                double *last_t, int *last_completion) {
    const double now = now_sec();
    const double elapsed = now - decode_t0;
    const double interval_s = now - *last_t;
    const int interval_tokens = completion - *last_completion;
    const double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    const double avg_tps = elapsed > 0.0 ? (double)completion / elapsed : 0.0;
    char ctx[48];
    request_ctx_span(ctx, sizeof(ctx),
                     prompt_tokens + *last_completion,
                     prompt_tokens + completion);
    char flags[80];
    log_flags(flags, sizeof(flags), responses_protocol,
              tools, thinking, dsml_start, dsml_end);
    server_log(DS4_LOG_GENERATION,
               "ds4-server: %s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs",
               kind == REQ_CHAT ? "chat" : "completion",
               ctx,
               completion,
               flags[0] ? " " : "",
               flags,
               chunk_tps,
               avg_tps,
               elapsed);
    *last_t = now;
    *last_completion = completion;
}

static bool thinking_tail_ends_with(const thinking_state *st, const char *s) {
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}

void thinking_state_feed(thinking_state *st, const char *p, size_t len) {
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (thinking_tail_ends_with(st, "<think>")) st->inside = true;
        else if (thinking_tail_ends_with(st, "</think>")) st->inside = false;
    }
}

thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        thinking_state_feed(&st, r->prompt_text, strlen(r->prompt_text));
    } else if (r && ds4_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}

static char *rendered_chat_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = "<｜begin▁of▁sentence｜>";
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    /* V4.1: BOS 后可能先是 <｜System｜>, 再是 thinking 的 effort 前缀(75 或 100 那行); 都不是 system 正文, 跳过 */
    const char *sys_tok = ds4_chat_system_token();
    if (sys_tok[0] && !strncmp(p, sys_tok, strlen(sys_tok))) p += strlen(sys_tok);
    const char *effort_prefixes[] = { ds4_think_effort_prefix(DS4_THINK_HIGH), ds4_think_effort_prefix(DS4_THINK_MAX) };
    for (size_t i = 0; i < sizeof(effort_prefixes) / sizeof(effort_prefixes[0]); i++) {
        const size_t n = strlen(effort_prefixes[i]);
        if (n && !strncmp(p, effort_prefixes[i], n)) { p += n; break; }
    }
    while (*p && isspace((unsigned char)*p)) p++;

    const char *user = strstr(p, "<｜User｜>");
    const char *assistant = strstr(p, "<｜Assistant｜>");
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}

char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    char *system = rendered_chat_system_region(r ? r->prompt_text : NULL);
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
             "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }

    buf suffix = {0};
    if (r && ds4_think_mode_enabled(r->think_mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, "<｜end▁of▁sentence｜><｜User｜><tool_result>");
    append_tool_result_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
    buf_puts(&suffix, "</tool_result><｜Assistant｜>");
    buf_puts(&suffix, r && ds4_think_mode_enabled(r->think_mode) ? "<think>" : "</think>");

    free(system);
    buf_free(&tool_error);
    return buf_take(&suffix);
}

static bool append_rendered_suffix_to_live_session(server *s, const char *suffix,
                                                   int *tokens_appended,
                                                   char *err, size_t errlen) {
    if (tokens_appended) *tokens_appended = 0;
    if (!s || !suffix || !suffix[0]) return true;
    const ds4_tokens *live = ds4_session_tokens(s->session);
    if (!live) {
        if (err && errlen) snprintf(err, errlen, "live session is unavailable");
        return false;
    }

    ds4_tokens target = {0};
    build_prompt_from_exact_prefix_and_text_suffix(s->engine, live, suffix, &target);
    const int before = ds4_session_pos(s->session);
    bool ok = ds4_session_sync(s->session, &target, err, errlen) == 0;
    if (ok && tokens_appended) {
        int delta = ds4_session_pos(s->session) - before;
        *tokens_appended = delta > 0 ? delta : 0;
    }
    ds4_tokens_free(&target);
    return ok;
}

bool continue_after_invalid_dsml(server *s, const request *r,
                                        const thinking_state *thinking,
                                        const char *detail,
                                        int *tokens_appended,
                                        char *err, size_t errlen) {
    char *suffix = build_invalid_dsml_tool_error_suffix(r, thinking, detail);
    bool ok = append_rendered_suffix_to_live_session(s, suffix,
                                                     tokens_appended,
                                                     err, errlen);
    free(suffix);
    return ok;
}

bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish) {
    if (!r || r->kind != REQ_CHAT || r->has_tools) return false;
    if (r->prompt_preserves_reasoning) return false;
    if (!ds4_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}

void log_tool_calls_summary(const char *ctx, const tool_calls *calls,
                                   bool responses_protocol) {
    if (!calls || calls->len == 0) return;
    buf names = {0};
    buf ids = {0};
    for (int i = 0; i < calls->len; i++) {
        if (i) buf_putc(&names, ',');
        if (i) buf_putc(&ids, ',');
        buf_puts(&names, calls->v[i].name ? calls->v[i].name : "?");
        buf_puts(&ids, calls->v[i].id ? calls->v[i].id : "?");
    }
    char flags[32];
    log_flags(flags, sizeof(flags), responses_protocol, false, false, false, false);
    server_log(DS4_LOG_TOOL,
               "ds4-server: tool calls ctx=%s%s%s n=%d raw_dsml=%d ids=[%s] names=[%s]",
               ctx,
               flags[0] ? " " : "",
               flags,
               calls->len,
               calls->raw_dsml && calls->raw_dsml[0] ? 1 : 0,
               ids.ptr ? ids.ptr : "",
               names.ptr ? names.ptr : "");
    buf_free(&ids);
    buf_free(&names);
}
