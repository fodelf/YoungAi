/* server_parse_responses_input.c — 机械拆分自 ds4_server.c (3715-4190 行): Responses input 数组解析。 */

#include "server_internal.h"

/* Codex /v1/responses input items have a `type` discriminator (message,
 * function_call, function_call_output, reasoning, custom_tool_call,
 * custom_tool_call_output, ...). We collapse them into chat_msgs the same way
 * the chat completion / Anthropic parsers do, so the rest of the engine sees a
 * single conversation history shape.
 *
 * Protocol contract for stateless replay:
 *   - The client must replay response.output items before tool outputs.
 *   - For reasoning models, the replay must also include reasoning state.  DS4
 *     can render plain reasoning summaries/content, but it cannot decrypt
 *     reasoning.encrypted_content.  If live state is unavailable and the replay
 *     only contains visible messages/tool calls, later validation marks it as a
 *     lower-fidelity replay; generate_job() logs that and continues from the
 *     visible transcript rather than killing a recoverable agent session.
 *
 * Reasoning items are merged into the next assistant message so
 * render_chat_prompt_text can wrap them in <think>. */
bool parse_responses_input(const char **p, chat_msgs *msgs,
                                  buf *loaded_tool_schemas,
                                  tool_schema_orders *orders) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    buf pending_reasoning = {0};

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') goto fail;
        (*p)++;
        char *type = NULL;
        char *role = NULL;
        char *content = NULL;
        char *name = NULL;
        char *namespace = NULL;
        char *call_id = NULL;
        char *item_id = NULL;
        char *arguments = NULL;
        char *output = NULL;
        char *input_str = NULL;
        char *summary = NULL;
        char *action = NULL;
        char *result = NULL;
        char *tools_json = NULL;
        char *status_str = NULL;
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto item_fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto item_fail;
            }
            (*p)++;
            if (!strcmp(key, "type")) {
                free(type);
                if (!json_string(p, &type)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "role")) {
                free(role);
                if (!json_string(p, &role)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "content")) {
                free(content);
                if (!parse_responses_content_array(p, &content)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "name")) {
                free(name);
                if (!json_string(p, &name)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "namespace")) {
                free(namespace);
                if (!json_string(p, &namespace)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "call_id")) {
                free(call_id);
                if (!json_string(p, &call_id)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "id")) {
                free(item_id);
                if (!json_string(p, &item_id)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "arguments")) {
                free(arguments);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &arguments)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &arguments)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "output")) {
                free(output);
                json_ws(p);
                if (**p == '[') {
                    if (!parse_responses_content_array(p, &output)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (**p == '"') {
                    if (!json_string(p, &output)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &output)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "input")) {
                free(input_str);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &input_str)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &input_str)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "summary")) {
                free(summary);
                if (!parse_responses_content_array(p, &summary)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "action")) {
                free(action);
                if (!json_raw_value(p, &action)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "result")) {
                free(result);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &result)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &result)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "status")) {
                free(status_str);
                if (!json_string(p, &status_str)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "tools")) {
                /* tool_search_output items carry their discovered tool list
                 * here instead of in `output` / `result`. Keep it separate
                 * from the human-visible result body so malformed tool lists
                 * never get mistaken for normal tool output. */
                free(tools_json);
                if (!json_raw_value(p, &tools_json)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto item_fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
            continue;
item_fail:
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            buf_free(&pending_reasoning);
            return false;
        }
        if (**p != '}') {
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            goto fail;
        }
        (*p)++;

        const char *t = type ? type : "message";
        /* Replayed items must be in a terminal "completed" state. in_progress,
         * incomplete, and failed all represent partial model state the client
         * never confirmed — feeding them back as history would let DS4 continue
         * from a tool action that never finished. Reject explicitly. */
        if (status_str && status_str[0] &&
            strcmp(status_str, "completed") != 0)
        {
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            buf_free(&pending_reasoning);
            return false;
        }
        /* Three classes of items:
         *   1. consumes_reasoning: assistant message / function_call / hosted-tool
         *      call. Attaches pending reasoning to its own assistant message.
         *   2. is_bookkeeping: compaction / context_compaction etc. Semantically
         *      transparent — passes through without touching pending_reasoning.
         *   3. everything else (user message, tool output): forces pending
         *      reasoning to flush in-position as an empty assistant message so it
         *      stays before this item in the rendered history. */
        bool consumes_reasoning =
            (!strcmp(t, "message") && role && !strcmp(role, "assistant")) ||
            !strcmp(t, "function_call") || !strcmp(t, "custom_tool_call") ||
            !strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
            !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call");
        bool is_bookkeeping =
            !strcmp(t, "compaction") || !strcmp(t, "context_compaction");
        if (!consumes_reasoning && !is_bookkeeping && pending_reasoning.len) {
            chat_msg flush_msg = {0};
            flush_msg.role = xstrdup("assistant");
            flush_msg.content = xstrdup("");
            flush_msg.reasoning = buf_take(&pending_reasoning);
            chat_msgs_push(msgs, flush_msg);
        }
        if (!strcmp(t, "message")) {
            chat_msg msg = {0};
            msg.role = xstrdup(role ? role : "user");
            msg.content = content ? content : xstrdup("");
            content = NULL;
            if (!strcmp(msg.role, "assistant") && pending_reasoning.len) {
                msg.reasoning = buf_take(&pending_reasoning);
            }
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "function_call") || !strcmp(t, "custom_tool_call")) {
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            /* function_call uses `arguments` (JSON string); custom_tool_call uses
             * `input` (free text). Treat both as the same on-wire argument blob —
             * append_dsml_arguments_from_json will fall back to a single text param
             * if the value isn't a JSON object. */
            const char *args_src = arguments ? arguments :
                                   input_str ? input_str : "{}";
            tc.arguments = xstrdup(args_src);
            if (strcmp(t, "custom_tool_call") && namespace && namespace[0] &&
                name && name[0])
            {
                buf qualified = {0};
                buf_puts(&qualified, namespace);
                buf_puts(&qualified, name);
                tc.name = buf_take(&qualified);
            } else {
                tc.name = xstrdup(name ? name : "");
            }
            /* A Responses turn that has both message text and tool calls splits
             * them across separate output items; the chat template renders the
             * second assistant record without an `<|Assistant|>` prefix, leaving
             * the tool call bare. Merge into the previous assistant message
             * when nothing user-like / tool-output-like came between them. */
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending_reasoning.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending_reasoning);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending_reasoning.len) msg.reasoning = buf_take(&pending_reasoning);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "function_call_output") || !strcmp(t, "custom_tool_call_output")) {
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            msg.content = output ? output : xstrdup("");
            output = NULL;
            if (call_id || item_id) {
                chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            }
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "reasoning")) {
            /* Stash so it merges into the next assistant message. summary is the
             * short-form list, content is the verbose chain. Either can be empty. */
            if (summary && summary[0]) {
                if (pending_reasoning.len) buf_putc(&pending_reasoning, '\n');
                buf_puts(&pending_reasoning, summary);
            }
            if (content && content[0]) {
                if (pending_reasoning.len) buf_putc(&pending_reasoning, '\n');
                buf_puts(&pending_reasoning, content);
            }
        } else if (!strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
                   !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call"))
        {
            /* Hosted-tool history isn't natively supported (DS4 doesn't register
             * these tools), but a Codex client may still replay them when the
             * model used them in a prior turn. Surface them as function_call
             * shaped history so the next prompt retains the action that ran. */
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            if (!strcmp(t, "tool_search_call")) {
                tc.name = xstrdup("tool_search");
            } else if (!strcmp(t, "local_shell_call")) {
                tc.name = xstrdup("local_shell");
            } else {
                tc.name = xstrdup(t);
            }
            const char *args_src = action ? action :
                                   arguments ? arguments :
                                   input_str ? input_str : "{}";
            tc.arguments = xstrdup(args_src);
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending_reasoning.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending_reasoning);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending_reasoning.len) msg.reasoning = buf_take(&pending_reasoning);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "local_shell_call_output") ||
                   !strcmp(t, "web_search_call_output") ||
                   !strcmp(t, "tool_search_output") ||
                   !strcmp(t, "tool_search_call_output") ||
                   !strcmp(t, "image_generation_call_output"))
        {
            if (!strcmp(t, "tool_search_output") && tools_json &&
                loaded_tool_schemas && orders)
            {
                const char *tools_p = tools_json;
                char *schemas = NULL;
                if (!parse_tools_value(&tools_p, &schemas, orders)) {
                    free(schemas);
                    free(type);
                    free(role);
                    free(content);
                    free(name);
                    free(namespace);
                    free(call_id);
                    free(item_id);
                    free(arguments);
                    free(output);
                    free(input_str);
                    free(summary);
                    free(action);
                    free(result);
                    free(tools_json);
                    free(status_str);
                    buf_free(&pending_reasoning);
                    return false;
                }
                if (schemas && schemas[0]) {
                    if (loaded_tool_schemas->len) buf_putc(loaded_tool_schemas, '\n');
                    buf_puts(loaded_tool_schemas, schemas);
                }
                free(schemas);
            }
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            const char *body = output ? output :
                               result ? result :
                               tools_json ? tools_json : "";
            msg.content = xstrdup(body);
            if (call_id || item_id) {
                chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            }
            chat_msgs_push(msgs, msg);
        } else if (!is_bookkeeping) {
            /* Anything we don't have an explicit branch for would silently
             * drop replay context. Fail the parse instead so the client sees
             * the limitation rather than ending up with stale generation
             * built on an incomplete history. Only compaction/context_compaction
             * (true Codex bookkeeping) are allowed to pass through silently. */
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            buf_free(&pending_reasoning);
            return false;
        }

        free(type);
        free(role);
        free(content);
        free(name);
        free(namespace);
        free(call_id);
        free(item_id);
        free(arguments);
        free(output);
        free(input_str);
        free(summary);
        free(action);
        free(result);
        free(tools_json);
        free(status_str);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    /* Trailing reasoning with no following message/tool item: attach it to an
     * empty assistant message so the next turn still renders a <think>...</think>
     * block. Dropping it loses model state when a previous response ended with
     * a reasoning-only incomplete turn and the client replays the history. */
    if (pending_reasoning.len) {
        chat_msg msg = {0};
        msg.role = xstrdup("assistant");
        msg.content = xstrdup("");
        msg.reasoning = buf_take(&pending_reasoning);
        chat_msgs_push(msgs, msg);
    }
    buf_free(&pending_reasoning);
    return true;
fail:
    buf_free(&pending_reasoning);
    return false;
}
