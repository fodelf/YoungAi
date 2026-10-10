/* server_openai_stream.c — 机械拆分自 ds4_server.c (5629-5814, 6233-6579 行): OpenAI 流式输出。 */

#include "server_internal.h"

void openai_stream_start(const request *r, openai_stream *st) {
    memset(st, 0, sizeof(*st));
    st->active = true;
    st->mode = ds4_think_mode_enabled(r->think_mode) ? OPENAI_STREAM_THINKING : OPENAI_STREAM_TEXT;
}

static void openai_tool_stream_free(openai_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    ts->ids = NULL;
    ts->ids_cap = 0;
}

void openai_stream_free(openai_stream *st) {
    if (!st) return;
    openai_tool_stream_free(&st->tool);
}

static bool openai_tool_stream_has_id(const openai_tool_stream *ts,
                                      const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}

static const char *openai_tool_stream_id(server *s, openai_tool_stream *ts,
                                         int index) {
    if (!ts || index < 0) return "";
    if (index >= ts->ids_cap) {
        int old = ts->ids_cap;
        int cap = old ? old : 4;
        while (cap <= index) cap *= 2;
        ts->ids = xrealloc(ts->ids, (size_t)cap * sizeof(ts->ids[0]));
        memset(ts->ids + old, 0, (size_t)(cap - old) * sizeof(ts->ids[0]));
        ts->ids_cap = cap;
    }
    if (!ts->ids[index]) {
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), API_OPENAI);
            if (!openai_tool_stream_has_id(ts, id, index) &&
                !tool_memory_has_id(s, id)) break;
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}

static bool sse_chat_delta_n(int fd, const request *r, const char *id,
                             const char *field, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    ds4_buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    ds4_buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
    json_escape(&b, field);
    ds4_buf_putc(&b, ':');
    json_escape_n(&b, text, len);
    ds4_buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    ds4_buf_free(&b);
    return ok;
}

/* OpenAI clients can consume function.arguments as a stream of JSON text
 * fragments.  DS4 generates XML-ish DSML instead, so this parser switches to a
 * hidden tool mode at <...tool_calls>, emits the tool header once the invoke tag
 * is complete, then translates each parameter body into argument deltas while
 * holding only tiny tails for partial closing tags, UTF-8, and DSML entities. */
static bool sse_chat_tool_call_start_delta(int fd, const request *r, const char *id,
                                           int index, const char *tool_id,
                                           const char *name) {
    buf b = {0};
    long now = (long)time(NULL);
    ds4_buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    ds4_buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    ds4_buf_printf(&b, "%d", index);
    ds4_buf_puts(&b, ",\"id\":");
    json_escape(&b, tool_id ? tool_id : "");
    ds4_buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, name ? name : "");
    ds4_buf_puts(&b, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    ds4_buf_free(&b);
    return ok;
}

static bool sse_chat_tool_call_args_delta_n(int fd, const request *r, const char *id,
                                            int index, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    ds4_buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    ds4_buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    ds4_buf_printf(&b, "%d", index);
    ds4_buf_puts(&b, ",\"function\":{\"arguments\":");
    json_escape_n(&b, text, len);
    ds4_buf_puts(&b, "}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    ds4_buf_free(&b);
    return ok;
}

bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}

bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}

bool raw_partial_any(const char *raw, size_t raw_len, size_t pos,
                            const char *a, const char *b) {
    return raw_partial_lit(raw, raw_len, pos, a) || raw_partial_lit(raw, raw_len, pos, b);
}

const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++) {
        if (!memcmp(s + i, lit, m)) return s + i;
    }
    return NULL;
}

static bool openai_tool_emit_args_fragment(int fd, const request *r, const char *id,
                                           openai_tool_stream *ts,
                                           const char *text, size_t len) {
    return sse_chat_tool_call_args_delta_n(fd, r, id, ts->index, text, len);
}

static bool openai_tool_emit_string_value(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = dsml_unescape_text(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    ds4_buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}

static bool openai_tool_emit_param_prefix(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *name, bool is_string) {
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else ds4_buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    ds4_buf_putc(&frag, ':');
    if (is_string) ds4_buf_putc(&frag, '"');
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    ds4_buf_free(&frag);
    return ok;
}

static bool openai_tool_stream_init(openai_tool_stream *ts, const char *raw,
                                    size_t raw_len, size_t pos) {
    openai_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = DSML_TOOL_BETWEEN_INVOKES;
    ts->parse_pos = pos;
    if (raw_full_lit(raw, raw_len, pos, DS4_TOOL_CALLS_START)) {
        ts->parse_pos += strlen(DS4_TOOL_CALLS_START);
        ts->tool_calls_end = DS4_TOOL_CALLS_END;
        ts->invoke_start = DS4_INVOKE_START;
        ts->invoke_end = DS4_INVOKE_END;
        ts->param_start = DS4_PARAM_START;
        ts->param_end = DS4_PARAM_END;
    } else if (raw_full_lit(raw, raw_len, pos, DS4_TOOL_CALLS_START_SHORT)) {
        ts->parse_pos += strlen(DS4_TOOL_CALLS_START_SHORT);
        ts->tool_calls_end = DS4_TOOL_CALLS_END_SHORT;
        ts->invoke_start = DS4_INVOKE_START_SHORT;
        ts->invoke_end = DS4_INVOKE_END_SHORT;
        ts->param_start = DS4_PARAM_START_SHORT;
        ts->param_end = DS4_PARAM_END_SHORT;
    } else if (raw_full_lit(raw, raw_len, pos, "<tool_calls>")) {
        ts->parse_pos += strlen("<tool_calls>");
        ts->tool_calls_end = "</tool_calls>";
        ts->invoke_start = "<invoke";
        ts->invoke_end = "</invoke>";
        ts->param_start = "<parameter";
        ts->param_end = "</parameter>";
    } else {
        ts->active = false;
        ts->state = DSML_TOOL_ERROR;
        return false;
    }
    return true;
}

static bool openai_tool_stream_fail(openai_tool_stream *ts) {
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return true;
}

static bool openai_tool_start_invoke(int fd, server *s, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t raw_len) {
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    free(tag);
    if (!name) return openai_tool_stream_fail(ts);

    const char *tool_id = openai_tool_stream_id(s, ts, ts->index);
    bool ok = sse_chat_tool_call_start_delta(fd, r, id, ts->index, tool_id, name) &&
              openai_tool_emit_args_fragment(fd, r, id, ts, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool openai_tool_start_param(int fd, const request *r, const char *id,
                                    openai_tool_stream *ts,
                                    const char *raw, size_t raw_len) {
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return openai_tool_stream_fail(ts);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = openai_tool_emit_param_prefix(fd, r, id, ts, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}

static bool openai_tool_finish_param(int fd, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t value_end) {
    if (value_end > ts->parse_pos) {
        bool ok = ts->param_is_string ?
            openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                          value_end - ts->parse_pos) :
            openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                           value_end - ts->parse_pos);
        if (!ok) return false;
    }
    if (ts->param_is_string &&
        !openai_tool_emit_args_fragment(fd, r, id, ts, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool openai_tool_stream_update(int fd, server *s, const request *r, const char *id,
                                      openai_tool_stream *ts,
                                      const char *raw, size_t raw_len) {
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == DSML_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->tool_calls_end)) {
                ts->parse_pos += strlen(ts->tool_calls_end);
                ts->active = false;
                ts->state = DSML_TOOL_DONE;
                return true;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->tool_calls_end, ts->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_invoke(fd, s, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_end)) {
                if (ts->args_open &&
                    !openai_tool_emit_args_fragment(fd, r, id, ts, "}", 1)) return false;
                ts->args_open = false;
                ts->parse_pos += strlen(ts->invoke_end);
                ts->index++;
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->invoke_end, ts->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->param_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_param(fd, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos,
                                               raw_len - ts->parse_pos,
                                               ts->param_end);
            if (end) {
                if (!openai_tool_finish_param(fd, r, id, ts, raw,
                                              (size_t)(end - raw))) return false;
                continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos,
                                                            raw_len, ts->param_end,
                                                            ts->param_is_string);
            if (limit > ts->parse_pos) {
                bool ok = ts->param_is_string ?
                    openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                                  limit - ts->parse_pos) :
                    openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                                   limit - ts->parse_pos);
                if (!ok) return false;
                ts->parse_pos = limit;
            }
            return true;
        }

        return true;
    }
    return true;
}

bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                     openai_stream *st,
                                     const char *raw, size_t raw_len,
                                     bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == OPENAI_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) {
            limit = (size_t)(close - raw);
        } else if (final) {
            limit = raw_len;
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "reasoning_content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos)) return false;
            st->sent_reasoning = true;
            st->emit_pos = limit;
        }

        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen("</think>");
            st->mode = OPENAI_STREAM_TEXT;
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
            return true;
        } else {
            return true;
        }
    }

    if (st->mode == OPENAI_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos)) return false;
            st->sent_content = true;
            st->emit_pos = limit;
        }

        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            if (openai_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = OPENAI_STREAM_TOOL;
            } else {
                st->mode = OPENAI_STREAM_SUPPRESS;
            }
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
        }
    }

    if (st->mode == OPENAI_STREAM_TOOL) {
        if (!openai_tool_stream_update(fd, s, r, id, &st->tool, raw, raw_len)) return false;
        if (!st->tool.active) st->mode = OPENAI_STREAM_SUPPRESS;
    }
    return true;
}

bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                   openai_stream *st, const char *raw,
                                   size_t raw_len, const tool_calls *calls,
                                   const char *finish, int prompt_tokens,
                                   int completion_tokens) {
    if (!openai_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (calls && calls->len && !st->tool.emitted_any) {
        ds4_buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        ds4_buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        ds4_buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    ds4_buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    ds4_buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    ds4_buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    ds4_buf_free(&b);
    return ok;
}

bool request_uses_openai_live_stream(const request *r) {
    return r->stream && r->api == API_OPENAI && r->kind == REQ_CHAT;
}

bool request_uses_responses_live_stream(const request *r) {
    return r->stream && r->api == API_RESPONSES && r->kind == REQ_CHAT;
}

bool request_uses_structured_stream(const request *r) {
    return r->stream && (r->api == API_ANTHROPIC ||
                         r->api == API_RESPONSES ||
                         request_uses_openai_live_stream(r));
}
