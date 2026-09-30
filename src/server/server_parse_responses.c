/* server_parse_responses.c — 机械拆分自 ds4_server.c (3585-3714, 4191-4550 行): Responses 请求解析。 */

#include "server_internal.h"

/* Responses API: convert a content-array item (input_text/output_text/text) into a
 * concatenated string. Strict shape check: bare string, null, or an array of
 * recognized text blocks. Numbers / objects / arrays-of-primitives at the top
 * level all reject so the client sees a 400 instead of an answer built on
 * silently dropped context. */
bool parse_responses_content_array(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        return false;
    }
    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            char *type = NULL;
            char *text = NULL;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) {
                    free(type);
                    free(text);
                    goto fail;
                }
                json_ws(p);
                if (**p != ':') {
                    free(key);
                    free(type);
                    free(text);
                    goto fail;
                }
                (*p)++;
                if (!strcmp(key, "type")) {
                    free(type);
                    if (!json_string(p, &type)) {
                        free(key);
                        free(text);
                        goto fail;
                    }
                } else if (!strcmp(key, "text")) {
                    free(text);
                    /* The text field of a typed content block is a plain JSON
                     * string. Accept null as the empty string for parity with
                     * upstream serializers that emit null for empty blocks. */
                    json_ws(p);
                    if (json_lit(p, "null")) {
                        text = xstrdup("");
                    } else if (!json_string(p, &text)) {
                        free(key);
                        free(type);
                        goto fail;
                    }
                } else if (!json_skip_value(p)) {
                    free(key);
                    free(type);
                    free(text);
                    goto fail;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') {
                free(type);
                free(text);
                goto fail;
            }
            (*p)++;
            /* Fail closed: a content object must carry a known text-like type
             * AND a text field. Anything else — missing type, missing text,
             * image/file/audio types, future schema-drift — is rejected so the
             * client gets a 400 instead of an answer built on context the
             * server discarded silently. */
            bool is_text_block = type && (
                !strcmp(type, "input_text") ||
                !strcmp(type, "output_text") ||
                !strcmp(type, "text") ||
                !strcmp(type, "summary_text") ||
                !strcmp(type, "reasoning_text"));
            if (!is_text_block || !text) {
                free(type);
                free(text);
                goto fail;
            }
            buf_puts(&b, text);
            free(type);
            free(text);
        } else {
            /* Reject primitives, arrays-of-arrays, nulls: a content array
             * element must be either a string or a typed text object. */
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

/* Responses API has `reasoning: {"effort": "...", "summary": "..."}`. effort
 * controls thinking depth; summary mode (auto/concise/detailed) controls
 * whether the wire emits summary deltas at all — per the spec, no reasoning
 * summary is surfaced unless the client opts in. */
static bool parse_responses_reasoning(const char **p, ds4_think_mode *effort,
                                      bool *summary_opted_in,
                                      bool *effort_seen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "effort")) {
            json_ws(p);
            /* A `null` effort doesn't change thinking_enabled — it's the same
             * as omitting the field. Only treat the field as a control if it
             * carried an actual value. */
            if (json_lit(p, "null")) {
                /* nothing */
            } else {
                if (!parse_reasoning_effort_value(p, effort)) {
                    free(key);
                    return false;
                }
                if (effort_seen) *effort_seen = true;
            }
        } else if (!strcmp(key, "summary")) {
            json_ws(p);
            if (json_lit(p, "null")) {
                /* explicit null disables summary */
            } else if (**p == '"') {
                char *mode = NULL;
                if (!json_string(p, &mode)) {
                    free(key);
                    return false;
                }
                if (summary_opted_in &&
                    (!strcmp(mode, "auto") ||
                     !strcmp(mode, "concise") ||
                     !strcmp(mode, "detailed")))
                {
                    *summary_opted_in = true;
                }
                free(mode);
            } else if (!json_skip_value(p)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

bool parse_responses_request(ds4_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    r->api = API_RESPONSES;
    const char *p = body;
    bool got_input = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = false;   /* 默认不思考(2026-09-30, 见 request_init 的注释); reasoning.effort 显式给了才开 */
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;
    chat_msgs msgs = {0};
    buf loaded_tool_schemas = {0};
    char *instructions = NULL;
    char *tool_schemas = NULL;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "input")) {
            chat_msgs_free(&msgs);
            json_ws(&p);
            /* Codex CLI always sends `input` as an array; tolerate bare strings
             * for parity with other Responses-API callers. */
            if (*p == '"') {
                char *plain = NULL;
                if (!json_string(&p, &plain)) {
                    free(key);
                    goto bad;
                }
                chat_msg msg = {0};
                msg.role = xstrdup("user");
                msg.content = plain;
                chat_msgs_push(&msgs, msg);
            } else if (!parse_responses_input(&p, &msgs, &loaded_tool_schemas,
                                              &r->tool_orders)) {
                free(key);
                goto bad;
            }
            got_input = true;
        } else if (!strcmp(key, "instructions")) {
            free(instructions);
            instructions = NULL;
            json_ws(&p);
            if (json_lit(&p, "null")) {
                instructions = xstrdup("");
            } else if (!json_string(&p, &instructions)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '"') {
                char *choice = NULL;
                if (!json_string(&p, &choice)) {
                    free(key);
                    goto bad;
                }
                /* DS4 honours "none" (disable tools) and "auto" (model decides).
                 * "required" and explicit function targets need constrained
                 * decoding we don't implement — reject so clients see the
                 * limitation instead of silently downgrading to auto. */
                if (!strcmp(choice, "none")) {
                    tool_choice_none = true;
                } else if (strcmp(choice, "auto") != 0) {
                    snprintf(err, errlen, "tool_choice=%s not supported", choice);
                    free(choice);
                    free(key);
                    chat_msgs_free(&msgs);
                    buf_free(&loaded_tool_schemas);
                    free(instructions);
                    free(tool_schemas);
                    request_free(r);
                    return false;
                }
                free(choice);
            } else if (*p == '{') {
                snprintf(err, errlen, "forced tool_choice not supported");
                free(key);
                chat_msgs_free(&msgs);
                buf_free(&loaded_tool_schemas);
                free(instructions);
                free(tool_schemas);
                request_free(r);
                return false;
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            if (!json_string(&p, &r->model)) {
                free(key);
                goto bad;
            }
            r->model_from_request = true;
        } else if (!strcmp(key, "max_output_tokens") || !strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "mode")) {
            /* 接口显式意图(2026-07-25 用户裁决: 写码/问答在调用处区分, 不靠服务端猜):
             * "code"=零知识注入+语言感知代码锚; "qa"=允许检索+答问锚; 缺省=问句启发式兜底 */
            char *mv = NULL;
            if (!json_string(&p, &mv)) { free(key); goto bad; }
            if (mv) {
                if (!strcmp(mv, "code")) r->ds4_mode = 1;
                else if (!strcmp(mv, "qa")) r->ds4_mode = 2;
                free(mv);
            }
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
        } else if (!strcmp(key, "frequency_penalty")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->frequency_penalty = (float)v;
        } else if (!strcmp(key, "presence_penalty")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->presence_penalty = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning")) {
            bool effort_seen = false;
            if (!parse_responses_reasoning(&p, &reasoning_effort,
                                           &r->reasoning_summary_emit,
                                           &effort_seen)) {
                free(key);
                goto bad;
            }
            /* Only an explicit effort value counts as the client opting into
             * thinking control. summary alone, or `reasoning: null`, leaves the
             * default behaviour (and the model_alias_* fallbacks below) intact. */
            if (effort_seen) {
                got_thinking = true;
                /* Responses-API effort of "minimal" / "none" maps to disabled
                 * thinking. Other effort values turn thinking on (default is off) and choose HIGH or MAX. */
                thinking_enabled = reasoning_effort != DS4_THINK_NONE;
            }
        } else if (!strcmp(key, "previous_response_id") ||
                   !strcmp(key, "conversation"))
        {
            /* Official Responses state can be durable:
             *   previous_response_id chains to a stored prior response, and
             *   conversation points at a persistent Conversations object.
             *
             * DS4 does not yet implement that durable store.  The supported
             * modes are either (a) a live in-memory continuation checked by
             * visible transcript / tool call ids, or (b) stateless replay of
             * the full input items.  Accepting a non-null durable reference
             * without loading the referenced items would silently truncate the
             * prompt, so reject it explicitly. */
            json_ws(&p);
            if (!json_lit(&p, "null")) {
                snprintf(err, errlen,
                         "%s is not supported; replay full input instead",
                         key);
                free(key);
                chat_msgs_free(&msgs);
                buf_free(&loaded_tool_schemas);
                free(instructions);
                free(tool_schemas);
                request_free(r);
                return false;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_input) {
        snprintf(err, errlen, "missing input");
        chat_msgs_free(&msgs);
        buf_free(&loaded_tool_schemas);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    /* instructions in the Responses API replaces any system message — for Codex
     * it carries the full agent system prompt. Prepend it so render produces a
     * standard system+chat layout. */
    if (instructions && instructions[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = instructions;
        instructions = NULL;
        /* Insert at the head so it precedes the conversation. */
        chat_msgs_push(&msgs, msg);
        if (msgs.len > 1) {
            chat_msg tmp = msgs.v[msgs.len - 1];
            for (int i = msgs.len - 1; i > 0; i--) msgs.v[i] = msgs.v[i - 1];
            msgs.v[0] = tmp;
        }
    }
    buf combined_tool_schemas = {0};
    if (tool_schemas && tool_schemas[0]) buf_puts(&combined_tool_schemas, tool_schemas);
    if (loaded_tool_schemas.len) {
        if (combined_tool_schemas.len) buf_putc(&combined_tool_schemas, '\n');
        buf_append(&combined_tool_schemas, loaded_tool_schemas.ptr,
                   loaded_tool_schemas.len);
    }
    const char *active_tool_schemas =
        (!tool_choice_none && combined_tool_schemas.len) ?
        combined_tool_schemas.ptr : NULL;
    r->has_tools = active_tool_schemas && active_tool_schemas[0];
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    if (g_force_nothink) r->think_mode = DS4_THINK_NONE;
    if (!responses_validate_tool_outputs(s, &msgs, r->think_mode,
                                         &r->responses_requires_live_tool_state,
                                         &r->responses_requires_live_reasoning,
                                         err, errlen)) {
        chat_msgs_free(&msgs);
        buf_free(&combined_tool_schemas);
        buf_free(&loaded_tool_schemas);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    responses_prepare_live_continuation(r, &msgs);
    g_req_mode = r->ds4_mode;   /* 渲染前置位(单图worker串行, 静态安全同 argmax scratch 纪律) */
    r->prompt_text = render_chat_prompt_text(&msgs, active_tool_schemas,
                                             &r->tool_orders, r->think_mode,
                                             &r->prompt_conv_off);
    base_native_default_stops(&r->stops);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    chat_msgs_free(&msgs);
    buf_free(&combined_tool_schemas);
    buf_free(&loaded_tool_schemas);
    free(instructions);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    buf_free(&loaded_tool_schemas);
    free(instructions);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}
