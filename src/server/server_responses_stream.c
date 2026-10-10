/* server_responses_stream.c — 机械拆分自 ds4_server.c (6580-7056 行): Responses 流式输出(事件)。 */

#include "server_internal.h"

/* Codex' Responses API uses 24-hex suffixes for response/item ids. Prefix
 * controls the variant (resp_, rs_, msg_, fc_) so each event references a
 * stable identifier across output_item.added / .done. */
void responses_random_id(char *dst, size_t dstlen, const char *prefix) {
    unsigned char bytes[12];
    size_t pos = snprintf(dst, dstlen, "%s", prefix);
    if (pos >= dstlen) return;
    static uint64_t fallback_ctr;
    if (!random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint64_t b = ++fallback_ctr ^ (uint64_t)(uintptr_t)dst;
        memcpy(bytes, &a, sizeof(a));
        memcpy(bytes + sizeof(a), &b, sizeof(uint32_t));
    }
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 2 < dstlen; i++) {
        dst[pos++] = hex[bytes[i] >> 4];
        dst[pos++] = hex[bytes[i] & 15];
    }
    dst[pos] = '\0';
}

void responses_stream_init(const request *r, responses_stream *st) {
    memset(st, 0, sizeof(*st));
    st->mode = ds4_think_mode_enabled(r->think_mode) ? RESP_STREAM_THINKING : RESP_STREAM_TEXT;
    responses_random_id(st->response_id, sizeof(st->response_id), "resp_");
    responses_random_id(st->reasoning_id, sizeof(st->reasoning_id), "rs_");
    responses_random_id(st->message_id, sizeof(st->message_id), "msg_");
    st->reasoning_index = -1;
    st->message_index = -1;
}

void responses_stream_free(responses_stream *st) {
    if (!st) return;
    ds4_buf_free(&st->reasoning_text);
    ds4_buf_free(&st->message_text);
}

/* Codex parses an explicit sequence_number on every Responses event for
 * ordering and reconnect resilience. We inject it after the `{"type":...` head
 * so emitters can stay readable while still producing the wire shape Codex
 * expects. */
bool responses_sse_emit_event(int fd, responses_stream *st, const char *body) {
    buf b = {0};
    ds4_buf_puts(&b, "data: ");
    /* body always starts with `{"type":"..."`. We splice in sequence_number
     * after the closing quote of that string so every event has it as the
     * second field. */
    const char *type_close = NULL;
    if (body[0] == '{') {
        const char *p = body + 1;
        /* Skip the literal `"type":` then the value string. */
        if (!strncmp(p, "\"type\":\"", 8)) {
            const char *q = p + 8;
            while (*q && *q != '"') {
                if (*q == '\\' && q[1]) q += 2;
                else q++;
            }
            if (*q == '"') type_close = q + 1;
        }
    }
    if (type_close) {
        size_t head_len = (size_t)(type_close - body);
        ds4_buf_append(&b, body, head_len);
        ds4_buf_printf(&b, ",\"sequence_number\":%d", st->sequence++);
        ds4_buf_puts(&b, type_close);
    } else {
        ds4_buf_puts(&b, body);
    }
    ds4_buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_created(int fd, const request *r, responses_stream *st,
                                  long created_at) {
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.created\",\"response\":{\"id\":\"%s\","
        "\"object\":\"response\",\"created_at\":%ld,\"status\":\"in_progress\","
        "\"model\":", st->response_id, created_at);
    json_escape(&b, r->model);
    ds4_buf_puts(&b, ",\"output\":[]}}");
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_reasoning_added(int fd, responses_stream *st) {
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"in_progress\","
        "\"summary\":[]}}",
        st->reasoning_index, st->reasoning_id);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_reasoning_summary_part_added(int fd, responses_stream *st) {
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_part.added\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
        "\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}",
        st->reasoning_id, st->reasoning_index);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_reasoning_delta(int fd, responses_stream *st,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_text.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"delta\":",
        st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, text, len);
    ds4_buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

const char *responses_item_status_for_finish(const char *finish) {
    if (finish && (!strcmp(finish, "length") || !strcmp(finish, "error"))) return "incomplete";
    return "completed";
}

bool responses_sse_reasoning_done(int fd, responses_stream *st,
                                         const char *finish) {
    /* If the stream terminates before `</think>` was actually observed the
     * reasoning item is partial — regardless of why generation stopped (EOS,
     * stop sequence, tool_calls, length, error). Force the item to incomplete
     * so a client replay rejects it instead of feeding unfinished hidden state
     * back as completed history. */
    (void)finish;
    const char *item_status =
        st->reasoning_closed_naturally ? "completed" : "incomplete";
    /* Mirror the message-item close sequence: emit summary_text.done +
     * summary_part.done before the output_item.done so clients that key off
     * part lifecycle don't see a dangling open summary part. */
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_text.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"text\":",
        st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, st->reasoning_text.ptr ? st->reasoning_text.ptr : "",
                  st->reasoning_text.len);
    ds4_buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        ds4_buf_free(&b);
        return false;
    }

    if (st->reasoning_summary_started) {
        ds4_buf_free(&b);
        ds4_buf_printf(&b,
            "{\"type\":\"response.reasoning_summary_part.done\","
            "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
            "\"part\":{\"type\":\"summary_text\",\"text\":",
            st->reasoning_id, st->reasoning_index);
        json_escape_n(&b, st->reasoning_text.ptr ? st->reasoning_text.ptr : "",
                      st->reasoning_text.len);
        ds4_buf_puts(&b, "}}");
        ok = responses_sse_emit_event(fd, st, b.ptr);
        if (!ok) {
            ds4_buf_free(&b);
            return false;
        }
    }

    ds4_buf_free(&b);
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
        st->reasoning_index, st->reasoning_id, item_status);
    if (st->reasoning_text.len) {
        ds4_buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
        json_escape_n(&b, st->reasoning_text.ptr, st->reasoning_text.len);
        ds4_buf_putc(&b, '}');
    }
    ds4_buf_puts(&b, "]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_message_added(int fd, responses_stream *st) {
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"in_progress\","
        "\"role\":\"assistant\",\"content\":[]}}",
        st->message_index, st->message_id);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_message_text_part_added(int fd, responses_stream *st) {
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.content_part.added\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
        st->message_id, st->message_index);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_output_text_delta(int fd, responses_stream *st,
                                            const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_text.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"delta\":",
        st->message_id, st->message_index);
    json_escape_n(&b, text, len);
    ds4_buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

bool responses_sse_message_done(int fd, responses_stream *st,
                                       const char *finish) {
    const char *item_status = responses_item_status_for_finish(finish);
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_text.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"text\":",
        st->message_id, st->message_index);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    ds4_buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        ds4_buf_free(&b);
        return false;
    }

    ds4_buf_free(&b);
    ds4_buf_printf(&b,
        "{\"type\":\"response.content_part.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":",
        st->message_id, st->message_index);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    ds4_buf_puts(&b, ",\"annotations\":[]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        ds4_buf_free(&b);
        return false;
    }

    ds4_buf_free(&b);
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
        "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
        st->message_index, st->message_id, item_status);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    ds4_buf_puts(&b, ",\"annotations\":[]}]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

static bool responses_tool_call_is_tool_search(const tool_call *tc,
                                               const tool_schema_order *order) {
    return tc && tc->name && !strcmp(tc->name, "tool_search") &&
           (!order || order->responses_tool_search);
}

/* The internal tool_call doesn't track whether it came from a function_call or
 * a custom_tool_call (or what tool kind is registered). For round-trip
 * correctness with the rare custom_tool_call clients, we preserve any provided
 * call_id verbatim and pre-assign a stable fc_id; the discriminator currently
 * defaults to function_call because Codex CLI registers all its tools as
 * function tools. */
void responses_tool_items_build(responses_tool_item **out,
                                       const tool_calls *calls,
                                       int starting_output_index) {
    *out = NULL;
    if (!calls || calls->len == 0) return;
    responses_tool_item *items = xmalloc((size_t)calls->len * sizeof(*items));
    for (int i = 0; i < calls->len; i++) {
        memset(&items[i], 0, sizeof(items[i]));
        responses_random_id(items[i].fc_id, sizeof(items[i].fc_id), "fc_");
        if (calls->v[i].id && calls->v[i].id[0]) {
            snprintf(items[i].call_id, sizeof(items[i].call_id), "%s", calls->v[i].id);
        } else {
            responses_random_id(items[i].call_id, sizeof(items[i].call_id), "call_");
        }
        items[i].is_custom = false;
        items[i].output_index = starting_output_index + i;
    }
    *out = items;
}

void responses_append_function_call_item(buf *b, const tool_call *tc,
                                                const responses_tool_item *item,
                                                const char *item_status,
                                                bool with_args,
                                                const tool_schema_orders *orders) {
    const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
    if (responses_tool_call_is_tool_search(tc, order)) {
        ds4_buf_printf(b,
            "{\"id\":\"%s\",\"type\":\"tool_search_call\",\"status\":\"%s\","
            "\"call_id\":\"%s\",\"execution\":\"client\",\"arguments\":",
            item->fc_id, item_status, item->call_id);
        if (with_args) append_json_object_or_empty(b, tc->arguments);
        else ds4_buf_puts(b, "{}");
        ds4_buf_putc(b, '}');
        return;
    }

    const char *item_type = item->is_custom ? "custom_tool_call" : "function_call";
    const char *body_field = item->is_custom ? "input" : "arguments";
    ds4_buf_printf(b,
        "{\"id\":\"%s\",\"type\":\"%s\",\"status\":\"%s\",\"name\":",
        item->fc_id, item_type, item_status);
    json_escape(b, order && order->wire_name ? order->wire_name :
                   (tc->name ? tc->name : ""));
    if (order && order->namespace) {
        ds4_buf_puts(b, ",\"namespace\":");
        json_escape(b, order->namespace);
    }
    ds4_buf_puts(b, ",\"call_id\":");
    json_escape(b, item->call_id);
    ds4_buf_printf(b, ",\"%s\":", body_field);
    if (!with_args) {
        ds4_buf_puts(b, "\"\"");
    } else if (item->is_custom) {
        json_escape(b, tc->arguments ? tc->arguments : "");
    } else {
        append_json_object_string(b, tc->arguments);
    }
    ds4_buf_putc(b, '}');
}

bool responses_sse_function_call_event(int fd, responses_stream *st,
                                              const tool_call *tc,
                                              const responses_tool_item *item,
                                              const tool_schema_orders *orders,
                                              const char *finish,
                                              bool done) {
    /* The added event marks a tool call as in_progress per the Responses
     * lifecycle; only output_item.done (and the terminal response output)
     * carry the final completed / incomplete status. The added item ships with
     * an empty arguments string so clients that accumulate via
     * function_call_arguments.delta + .done don't end up with doubled JSON. */
    const char *item_status = done ? responses_item_status_for_finish(finish) : "in_progress";
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.output_item.%s\",\"output_index\":%d,\"item\":",
        done ? "done" : "added", item->output_index);
    responses_append_function_call_item(&b, tc, item, item_status, done, orders);
    ds4_buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    return ok;
}

/* Stream function-call arguments as a single delta + done, since DS4 generates
 * the whole DSML invoke as one unit before the worker decides which tool was
 * called. Clients that follow the OpenAI Responses lifecycle expect both
 * events between output_item.added (in_progress) and output_item.done. */
bool responses_sse_function_call_arguments_done(int fd, responses_stream *st,
                                                       const tool_call *tc,
                                                       const responses_tool_item *item,
                                                       const tool_schema_orders *orders) {
    const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
    if (item->is_custom || responses_tool_call_is_tool_search(tc, order)) return true;
    buf args = {0};
    append_json_object_string(&args, tc->arguments);
    buf b = {0};
    ds4_buf_printf(&b,
        "{\"type\":\"response.function_call_arguments.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"delta\":",
        item->fc_id, item->output_index);
    ds4_buf_append(&b, args.ptr ? args.ptr : "\"\"", args.ptr ? args.len : 2);
    ds4_buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        ds4_buf_free(&b);
        ds4_buf_free(&args);
        return false;
    }

    ds4_buf_free(&b);
    ds4_buf_printf(&b,
        "{\"type\":\"response.function_call_arguments.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"name\":",
        item->fc_id, item->output_index);
    json_escape(&b, order && order->wire_name ? order->wire_name :
                    (tc->name ? tc->name : ""));
    if (order && order->namespace) {
        ds4_buf_puts(&b, ",\"namespace\":");
        json_escape(&b, order->namespace);
    }
    ds4_buf_puts(&b, ",\"arguments\":");
    ds4_buf_append(&b, args.ptr ? args.ptr : "\"\"", args.ptr ? args.len : 2);
    ds4_buf_putc(&b, '}');
    ok = responses_sse_emit_event(fd, st, b.ptr);
    ds4_buf_free(&b);
    ds4_buf_free(&args);
    return ok;
}

const char *responses_status_for_finish(const char *finish) {
    if (finish && !strcmp(finish, "length")) return "incomplete";
    if (finish && !strcmp(finish, "error")) return "failed";
    return "completed";
}

void append_responses_usage_json(buf *b, const request *r,
                                        int input_tokens, int output_tokens) {
    int cached_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    cached_tokens = clamp_usage_tokens(cached_tokens, input_tokens);
    cache_write_tokens = clamp_usage_tokens(cache_write_tokens, input_tokens - cached_tokens);
    ds4_buf_printf(b,
        "{\"input_tokens\":%d,\"input_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d},"
        "\"output_tokens\":%d,\"output_tokens_details\":{\"reasoning_tokens\":0},"
        "\"total_tokens\":%d}",
        input_tokens, cached_tokens, cache_write_tokens,
        output_tokens, input_tokens + output_tokens);
}
