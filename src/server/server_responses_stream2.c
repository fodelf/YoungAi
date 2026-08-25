/* server_responses_stream2.c — 机械拆分自 ds4_server.c (7057-7424 行): Responses 流式输出(收尾/终响应)。 */

#include "server_internal.h"

bool responses_sse_completed(int fd, const request *r,
                                    responses_stream *st,
                                    const tool_calls *calls,
                                    const responses_tool_item *tool_items,
                                    const char *finish,
                                    int prompt_tokens, int completion_tokens,
                                    long created_at) {
    /* Codex routes terminal behaviour off the event type, not response.status.
     * Decide here so clients see response.failed / response.incomplete instead
     * of a "completed" wrapper marked failed in a sub-field. */
    const char *event_type = "response.completed";
    if (finish && !strcmp(finish, "error")) event_type = "response.failed";
    else if (finish && !strcmp(finish, "length")) event_type = "response.incomplete";
    const char *status = responses_status_for_finish(finish);

    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"%s\",\"response\":{\"id\":\"%s\","
        "\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\",\"model\":",
        event_type, st->response_id, created_at, status);
    json_escape(&b, r->model);
    if (!strcmp(event_type, "response.failed")) {
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\","
                     "\"message\":\"generation failed\"}");
    } else if (!strcmp(event_type, "response.incomplete")) {
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_tokens\"}");
    }
    const char *item_status = responses_item_status_for_finish(finish);
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (st->reasoning_emitted_any) {
        /* Match responses_sse_reasoning_done: if the stream stopped before
         * </think>, the reasoning item is partial regardless of the
         * response-level finish status, so replay must reject it. */
        const char *reasoning_status =
            st->reasoning_closed_naturally ? "completed" : "incomplete";
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
            st->reasoning_id, reasoning_status);
        if (st->reasoning_text.len) {
            buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
            json_escape_n(&b, st->reasoning_text.ptr, st->reasoning_text.len);
            buf_putc(&b, '}');
        }
        buf_puts(&b, "]}");
        wrote = true;
    }
    if (st->message_emitted_any) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
            "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
            st->message_id, item_status);
        json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                      st->message_text.len);
        buf_puts(&b, ",\"annotations\":[]}]}");
        wrote = true;
    }
    if (calls && tool_items) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(&b, ',');
            responses_append_function_call_item(&b, &calls->v[i], &tool_items[i],
                                                item_status, true,
                                                &r->tool_orders);
            wrote = true;
        }
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}}");
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

/* Responses streaming consumes the same raw token text the OpenAI live stream
 * consumes: <think>...</think> is reasoning, anything before the tool-call
 * marker is output text. Tool-call argument deltas are not surfaced because
 * Codex' SSE parser only ingests function_call items via output_item.done. */
bool responses_sse_stream_update(int fd, const request *r,
                                        responses_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final) {
    if (!st->active || !raw) return true;

    /* The client only sees reasoning if it explicitly opted in via
     * reasoning.summary. Otherwise we still need to walk past <think>...</think>
     * to find the user-visible text, but we suppress the per-chunk emission. */
    const bool emit_reasoning = r->reasoning_summary_emit;

    if (st->mode == RESP_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            /* The chat template ends the prompt with the literal `<think>` (or
             * `</think>` when thinking is off), so generation usually starts
             * mid-reasoning rather than with the open tag. If the model does
             * happen to repeat `<think>` we skip it; otherwise start from
             * position 0. The earlier "no-think-prefix => switch to TEXT"
             * shortcut here was incorrect: it leaked reasoning to clients as
             * regular output_text because the model was already inside the
             * think block when it produced its first token. The actual
             * mode change to TEXT happens only when `</think>` is observed. */
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
            if (emit_reasoning) {
                if (!st->reasoning_item_opened) {
                    st->reasoning_index = st->next_output_index++;
                    if (!responses_sse_reasoning_added(fd, st)) return false;
                    st->reasoning_item_opened = true;
                }
                if (!st->reasoning_summary_started) {
                    if (!responses_sse_reasoning_summary_part_added(fd, st)) return false;
                    st->reasoning_summary_started = true;
                }
                if (!responses_sse_reasoning_delta(fd, st,
                                                   raw + st->emit_pos,
                                                   limit - st->emit_pos)) return false;
                buf_append(&st->reasoning_text, raw + st->emit_pos, limit - st->emit_pos);
                st->reasoning_emitted_any = true;
            }
            st->emit_pos = limit;
        }

        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen("</think>");
            st->mode = RESP_STREAM_TEXT;
            st->reasoning_closed_naturally = true;
        } else if (final) {
            st->mode = RESP_STREAM_SUPPRESS;
            return true;
        } else {
            return true;
        }
    }

    if (st->mode == RESP_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!st->message_item_opened) {
                st->message_index = st->next_output_index++;
                if (!responses_sse_message_added(fd, st)) return false;
                st->message_item_opened = true;
            }
            if (!st->message_text_part_open) {
                if (!responses_sse_message_text_part_added(fd, st)) return false;
                st->message_text_part_open = true;
            }
            if (!responses_sse_output_text_delta(fd, st,
                                                 raw + st->emit_pos,
                                                 limit - st->emit_pos)) return false;
            buf_append(&st->message_text, raw + st->emit_pos, limit - st->emit_pos);
            st->message_emitted_any = true;
            st->emit_pos = limit;
        }

        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            st->mode = RESP_STREAM_SUPPRESS;
        } else if (final) {
            st->mode = RESP_STREAM_SUPPRESS;
        }
    }
    return true;
}

bool responses_sse_finish_live(int fd, const request *r,
                                      responses_stream *st,
                                      const char *raw, size_t raw_len,
                                      const char *recovered_content,
                                      const tool_calls *calls,
                                      const char *finish,
                                      int prompt_tokens, int completion_tokens,
                                      long created_at) {
    if (!responses_sse_stream_update(fd, r, st, raw, raw_len, true)) return false;

    /* Close any half-open reasoning summary so the TUI knows the part ended
     * before we slot in any tool calls or completion. */
    if (st->reasoning_item_opened && !st->reasoning_item_closed) {
        if (!responses_sse_reasoning_done(fd, st, finish)) return false;
        st->reasoning_item_closed = true;
    }
    /* Recovery path: when DSML tool parsing fails the worker promotes the entire
     * generation to assistant text. Streaming had already entered suppress mode
     * at the tool marker, so anything in raw[st->emit_pos..raw_len] never made
     * it to the client. Emit those bytes as additional output_text deltas so
     * what the client accumulates matches output_item.done and the terminal
     * response. We use the stream cursor instead of comparing against
     * recovered_content because the raw text can begin with `<think>...</think>`
     * which the streaming side consumed as reasoning, not message text. */
    if (recovered_content && raw && st->emit_pos < raw_len) {
        const char *tail = raw + st->emit_pos;
        size_t tail_len = raw_len - st->emit_pos;
        if (!st->message_item_opened) {
            st->message_index = st->next_output_index++;
            if (!responses_sse_message_added(fd, st)) return false;
            st->message_item_opened = true;
        }
        if (!st->message_text_part_open) {
            if (!responses_sse_message_text_part_added(fd, st)) return false;
            st->message_text_part_open = true;
        }
        if (!responses_sse_output_text_delta(fd, st, tail, tail_len)) return false;
        buf_append(&st->message_text, tail, tail_len);
        st->message_emitted_any = true;
        st->emit_pos = raw_len;
    }
    if (st->message_item_opened && !st->message_item_closed) {
        if (!responses_sse_message_done(fd, st, finish)) return false;
        st->message_item_closed = true;
    }
    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, calls, st->next_output_index);
    if (items && calls) st->next_output_index += calls->len;
    bool ok = true;
    if (items && calls) {
        for (int i = 0; i < calls->len && ok; i++) {
            ok = responses_sse_function_call_event(fd, st, &calls->v[i], &items[i],
                                                   &r->tool_orders, finish, false);
            if (ok) ok = responses_sse_function_call_arguments_done(fd, st, &calls->v[i],
                                                                    &items[i],
                                                                    &r->tool_orders);
            if (ok) ok = responses_sse_function_call_event(fd, st, &calls->v[i], &items[i],
                                                           &r->tool_orders, finish, true);
        }
    }
    if (ok) ok = responses_sse_completed(fd, r, st, calls, items, finish,
                                         prompt_tokens, completion_tokens, created_at);
    free(items);
    return ok;
}

bool responses_final_response(int fd, bool enable_cors,
                                     const request *r, const char *id,
                                     const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    (void)id;
    char response_id[40], reasoning_id[40], message_id[40];
    responses_random_id(response_id, sizeof(response_id), "resp_");
    responses_random_id(reasoning_id, sizeof(reasoning_id), "rs_");
    responses_random_id(message_id, sizeof(message_id), "msg_");

    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, calls, 0);

    long now = (long)time(NULL);
    const char *status = responses_status_for_finish(finish);
    const char *item_status = responses_item_status_for_finish(finish);
    buf b = {0};
    buf_printf(&b,
        "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\","
        "\"model\":",
        response_id, now, status);
    json_escape(&b, r->model);
    if (finish && !strcmp(finish, "error")) {
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\","
                     "\"message\":\"generation failed\"}");
    } else if (finish && !strcmp(finish, "length")) {
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_tokens\"}");
    }
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (reasoning && reasoning[0] && r->reasoning_summary_emit) {
        /* Non-streaming path runs after the worker has post-processed the
         * generation, so any reasoning here came from a parsed assistant turn
         * where </think> was observed (otherwise the reasoning text would be
         * empty). Tag it with the response-level item_status which still flips
         * to incomplete/failed when finish is length/error. */
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\","
            "\"summary\":[{\"type\":\"summary_text\",\"text\":",
            reasoning_id, item_status);
        json_escape(&b, reasoning);
        buf_puts(&b, "}]}");
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
            "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
            message_id, item_status);
        json_escape(&b, text);
        buf_puts(&b, ",\"annotations\":[]}]}");
        wrote = true;
    }
    if (calls && items) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(&b, ',');
            responses_append_function_call_item(&b, &calls->v[i], &items[i],
                                                item_status, true,
                                                &r->tool_orders);
            wrote = true;
        }
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_putc(&b, '}');
    bool ok = http_response(fd, enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    free(items);
    return ok;
}

bool final_response(int fd, bool enable_cors,
                           const request *r, const char *id, const char *text,
                           const char *reasoning, const tool_calls *calls, const char *finish,
                           int prompt_tokens, int completion_tokens) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        json_escape(&b, text ? text : "");
        if (reasoning && reasoning[0]) {
            buf_puts(&b, ",\"reasoning_content\":");
            json_escape(&b, reasoning);
        }
        if (calls && calls->len) {
            buf_puts(&b, ",\"tool_calls\":");
            append_tool_calls_json(&b, calls, id, &r->tool_orders);
        }
        buf_puts(&b, "},\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    } else {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text);
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    }
    append_openai_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n");
    bool ok = http_response(fd, enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}
