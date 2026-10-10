/* server_anthropic.c — 机械拆分自 ds4_server.c (7425-7905 行): Anthropic 应答与流基元。 */

#include "server_internal.h"

const char *anthropic_stop_reason(const char *finish) {
    if (finish && !strcmp(finish, "tool_calls")) return "tool_use";
    if (finish && !strcmp(finish, "length")) return "max_tokens";
    return "end_turn";
}

static void append_anthropic_tool_use(buf *b, const tool_call *tc, const char *id_prefix, int i,
                                      const tool_schema_orders *orders) {
    (void)orders;
    char idbuf[128];
    snprintf(idbuf, sizeof(idbuf), DS4_TOOL_ID_PREFIX_ANTHROPIC "%s_%d", id_prefix, i);
    ds4_buf_puts(b, "{\"type\":\"tool_use\",\"id\":");
    json_escape(b, tc->id && tc->id[0] ? tc->id : idbuf);
    ds4_buf_puts(b, ",\"name\":");
    json_escape(b, tc->name ? tc->name : "");
    ds4_buf_puts(b, ",\"input\":");
    append_json_object_or_empty(b, tc->arguments);
    ds4_buf_putc(b, '}');
}

static void append_anthropic_thinking(buf *b, const char *reasoning, const char *signature) {
    ds4_buf_puts(b, "{\"type\":\"thinking\",\"thinking\":");
    json_escape(b, reasoning ? reasoning : "");
    ds4_buf_puts(b, ",\"signature\":");
    json_escape(b, signature ? signature : "");
    ds4_buf_putc(b, '}');
}

void append_anthropic_content(buf *b, const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *id_prefix,
                                     const tool_schema_orders *orders) {
    ds4_buf_putc(b, '[');
    bool wrote = false;
    bool wrote_after_thinking = false;
    if (reasoning && reasoning[0]) {
        append_anthropic_thinking(b, reasoning, id_prefix);
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) ds4_buf_putc(b, ',');
        ds4_buf_puts(b, "{\"type\":\"text\",\"text\":");
        json_escape(b, text);
        ds4_buf_putc(b, '}');
        wrote = true;
        wrote_after_thinking = true;
    }
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) ds4_buf_putc(b, ',');
            append_anthropic_tool_use(b, &calls->v[i], id_prefix, i, orders);
            wrote = true;
            wrote_after_thinking = true;
        }
    }
    if (!wrote || ((reasoning && reasoning[0]) && !wrote_after_thinking)) {
        if (wrote) ds4_buf_putc(b, ',');
        ds4_buf_puts(b, "{\"type\":\"text\",\"text\":\"\"}");
    }
    ds4_buf_putc(b, ']');
}

static void append_anthropic_usage_json(buf *b, const request *r,
                                        int prompt_tokens, int completion_tokens) {
    int cache_read_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    cache_read_tokens = clamp_usage_tokens(cache_read_tokens, prompt_tokens);
    cache_write_tokens = clamp_usage_tokens(cache_write_tokens, prompt_tokens - cache_read_tokens);
    int input_tokens = prompt_tokens - cache_read_tokens - cache_write_tokens;
    if (input_tokens < 0) input_tokens = 0;
    ds4_buf_printf(b,
               "{\"input_tokens\":%d,\"output_tokens\":%d,"
               "\"cache_read_input_tokens\":%d,\"cache_creation_input_tokens\":%d}",
               input_tokens, completion_tokens, cache_read_tokens, cache_write_tokens);
}

bool anthropic_final_response(int fd, bool enable_cors,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    buf b = {0};
    ds4_buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, r->model);
    ds4_buf_puts(&b, ",\"content\":");
    append_anthropic_content(&b, text, reasoning, calls, id, &r->tool_orders);
    ds4_buf_puts(&b, ",\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    ds4_buf_puts(&b, ",\"stop_sequence\":null,\"usage\":");
    append_anthropic_usage_json(&b, r, prompt_tokens, completion_tokens);
    ds4_buf_puts(&b, "}\n");
    bool ok = http_response(fd, enable_cors, 200, "application/json", b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool sse_event(int fd, const char *event, const char *data) {
    buf b = {0};
    ds4_buf_puts(&b, "event: ");
    ds4_buf_puts(&b, event);
    ds4_buf_puts(&b, "\ndata: ");
    ds4_buf_puts(&b, data);
    ds4_buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    ds4_buf_free(&b);
    return ok;
}

bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st) {
    buf b = {0};
    json_escape(&b, r->model);
    char *model_json = ds4_buf_take(&b);

    ds4_buf_printf(&b,
        "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\","
        "\"role\":\"assistant\",\"model\":%s,\"content\":[],\"stop_reason\":null,"
        "\"stop_sequence\":null,\"usage\":",
        id, model_json);
    append_anthropic_usage_json(&b, r, prompt_tokens, 0);
    ds4_buf_puts(&b, "}}");
    bool ok = sse_event(fd, "message_start", b.ptr);
    ds4_buf_free(&b);
    free(model_json);

    memset(st, 0, sizeof(*st));
    st->active = ok;
    st->mode = ds4_think_mode_enabled(r->think_mode) ? ANTH_STREAM_THINKING : ANTH_STREAM_TEXT;
    return ok;
}

static void anthropic_tool_stream_free(anthropic_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    ts->ids = NULL;
    ts->ids_cap = 0;
}

void anthropic_stream_free(anthropic_stream *st) {
    if (!st) return;
    anthropic_tool_stream_free(&st->tool);
}

static bool anthropic_tool_stream_has_id(const anthropic_tool_stream *ts,
                                         const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}

static const char *anthropic_tool_stream_id(server *s, anthropic_tool_stream *ts,
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
            random_tool_id(id, sizeof(id), API_ANTHROPIC);
            if (!anthropic_tool_stream_has_id(ts, id, index) &&
                !tool_memory_has_id(s, id)) break;
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}

/* Text and thinking blocks have fixed JSON shapes.  Tool blocks are opened by
 * name later, after the DSML invoke tag is complete, so they use a dedicated
 * opener instead of this helper. */
bool anthropic_sse_open_block(int fd, anthropic_stream *st,
                                     anthropic_block_type type) {
    if (st->open_block == type) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\","
                   "\"signature\":\"\"}}",
                   st->next_index);
    } else {
        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
                   st->next_index);
    }
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    ds4_buf_free(&b);
    if (ok) st->open_block = type;
    return ok;
}

static bool anthropic_sse_open_tool_block(int fd, anthropic_stream *st,
                                          const char *tool_id,
                                          const char *name) {
    if (st->open_block == ANTH_BLOCK_TOOL) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    ds4_buf_printf(&b,
               "{\"type\":\"content_block_start\",\"index\":%d,"
               "\"content_block\":{\"type\":\"tool_use\",\"id\":",
               st->next_index);
    json_escape(&b, tool_id ? tool_id : "");
    ds4_buf_puts(&b, ",\"name\":");
    json_escape(&b, name ? name : "");
    ds4_buf_puts(&b, ",\"input\":{}}}");
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    ds4_buf_free(&b);
    if (ok) st->open_block = ANTH_BLOCK_TOOL;
    return ok;
}

bool anthropic_sse_delta_live(int fd, const anthropic_stream *st,
                                     anthropic_block_type type,
                                     const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        ds4_buf_puts(&b, "}}");
    } else {
        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"text_delta\",\"text\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        ds4_buf_puts(&b, "}}");
    }
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    ds4_buf_free(&b);
    return ok;
}

/* Anthropic's input_json_delta carries a fragment of a JSON object, encoded as
 * a JSON string.  We stream exactly the same object that the final DSML parser
 * will build: an opening "{", quoted keys, raw JSON values or escaped string
 * contents, and the closing "}". */
static bool anthropic_sse_tool_delta_live(int fd, const anthropic_stream *st,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    ds4_buf_printf(&b,
               "{\"type\":\"content_block_delta\",\"index\":%d,"
               "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
               st->next_index);
    json_escape_n(&b, text, len);
    ds4_buf_puts(&b, "}}");
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool anthropic_sse_close_block_live(int fd, const char *id,
                                           anthropic_stream *st) {
    if (st->open_block == ANTH_BLOCK_NONE) return true;

    buf b = {0};
    bool ok = true;
    if (st->open_block == ANTH_BLOCK_THINKING) {
        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"signature_delta\",\"signature\":",
                   st->next_index);
        json_escape(&b, id);
        ds4_buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        ds4_buf_free(&b);
    }
    if (ok) {
        ds4_buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        ds4_buf_free(&b);
    }
    if (ok) {
        st->open_block = ANTH_BLOCK_NONE;
        st->next_index++;
    }
    return ok;
}

bool anthropic_tool_emit_args_fragment(int fd, anthropic_stream *st,
                                              const char *text, size_t len) {
    return anthropic_sse_tool_delta_live(fd, st, text, len);
}

bool anthropic_tool_emit_string_value(int fd, anthropic_stream *st,
                                             const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = dsml_unescape_text(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = anthropic_tool_emit_args_fragment(fd, st,
                                                frag.ptr ? frag.ptr : "",
                                                frag.len);
    ds4_buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}

static bool anthropic_tool_emit_param_prefix(int fd, anthropic_stream *st,
                                             const char *name, bool is_string) {
    anthropic_tool_stream *ts = &st->tool;
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else ds4_buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    ds4_buf_putc(&frag, ':');
    if (is_string) ds4_buf_putc(&frag, '"');
    bool ok = anthropic_tool_emit_args_fragment(fd, st,
                                                frag.ptr ? frag.ptr : "",
                                                frag.len);
    ds4_buf_free(&frag);
    return ok;
}

/* The parser below mirrors the OpenAI tool-delta parser but keeps Anthropic's
 * content-block lifecycle local.  A callback abstraction would save lines, but
 * it would hide the different block/stop semantics that make this code easy to
 * audit when a client reports a streaming regression. */
bool anthropic_tool_stream_init(anthropic_tool_stream *ts,
                                       const char *raw, size_t raw_len,
                                       size_t pos) {
    anthropic_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = DSML_TOOL_BETWEEN_INVOKES;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        const dsml_syntax *syn = &dsml_syntaxes[i];
        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_start)) {
            ts->syn = syn;
            ts->parse_pos = pos + strlen(syn->tool_calls_start);
            return true;
        }
    }
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return false;
}

bool anthropic_tool_stream_fail(anthropic_tool_stream *ts) {
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return true;
}

bool anthropic_tool_start_invoke(int fd, server *s, anthropic_stream *st,
                                        const char *raw, size_t raw_len) {
    anthropic_tool_stream *ts = &st->tool;
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos,
                         (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    free(tag);
    if (!name) return anthropic_tool_stream_fail(ts);

    /* This id is already visible to the client.  After final parsing,
     * apply_anthropic_stream_tool_ids() copies it into the parsed tool_call
     * before tool_memory_remember(), so the next tool_result can continue from
     * the live KV state instead of re-rendering canonical JSON. */
    const char *tool_id = anthropic_tool_stream_id(s, ts, ts->index);
    bool ok = anthropic_sse_open_tool_block(fd, st, tool_id, name) &&
              anthropic_tool_emit_args_fragment(fd, st, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

bool anthropic_tool_start_param(int fd, anthropic_stream *st,
                                       const char *raw, size_t raw_len) {
    anthropic_tool_stream *ts = &st->tool;
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos,
                         (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return anthropic_tool_stream_fail(ts);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = anthropic_tool_emit_param_prefix(fd, st, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}

bool anthropic_tool_finish_param(int fd, anthropic_stream *st,
                                        const char *raw, size_t value_end) {
    anthropic_tool_stream *ts = &st->tool;
    if (value_end > ts->parse_pos) {
        bool ok = ts->param_is_string ?
            anthropic_tool_emit_string_value(fd, st, raw + ts->parse_pos,
                                             value_end - ts->parse_pos) :
            anthropic_tool_emit_args_fragment(fd, st, raw + ts->parse_pos,
                                              value_end - ts->parse_pos);
        if (!ok) return false;
    }
    if (ts->param_is_string &&
        !anthropic_tool_emit_args_fragment(fd, st, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->syn->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}
