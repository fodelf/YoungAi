/* server_parse_anthropic_req.c — 机械拆分自 ds4_server.c (3332-3584 行): Anthropic 请求解析。 */

#include "server_internal.h"

bool parse_anthropic_request(ds4_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    r->api = API_ANTHROPIC;
    const char *p = body;
    bool got_messages = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;
    chat_msgs msgs = {0};
    char *system = NULL;
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
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&msgs);
            if (!parse_anthropic_messages(&p, &msgs, ds4_engine_mm(e))) {
                free(key);
                goto bad;
            }
            got_messages = true;
        } else if (!strcmp(key, "system")) {
            free(system);
            if (!parse_anthropic_system(&p, &system)) {
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
            if (*p == '{') {
                char *tc_type = NULL, *tc_name = NULL;
                p++;
                json_ws(&p);
                while (*p && *p != '}') {
                    char *ckey = NULL;
                    if (!json_string(&p, &ckey)) {
                        free(tc_type); free(tc_name);
                        free(key);
                        goto bad;
                    }
                    json_ws(&p);
                    if (*p != ':') {
                        free(tc_type); free(tc_name);
                        free(ckey);
                        free(key);
                        goto bad;
                    }
                    p++;
                    if (!strcmp(ckey, "type")) {
                        free(tc_type); tc_type = NULL;
                        if (!json_string(&p, &tc_type)) {
                            free(tc_name);
                            free(ckey);
                            free(key);
                            goto bad;
                        }
                        tool_choice_none = !strcmp(tc_type, "none");
                    } else if (!strcmp(ckey, "name")) {
                        free(tc_name); tc_name = NULL;
                        if (!json_string(&p, &tc_name)) {
                            free(tc_type);
                            free(ckey);
                            free(key);
                            goto bad;
                        }
                    } else if (!json_skip_value(&p)) {
                        free(tc_type); free(tc_name);
                        free(ckey);
                        free(key);
                        goto bad;
                    }
                    free(ckey);
                    json_ws(&p);
                    if (*p == ',') p++;
                    json_ws(&p);
                }
                /* {"type":"tool","name":X} pins the tool-name region to X;
                 * {"type":"any"} forces a tool call with a model-chosen name. */
                if (tc_type && !strcmp(tc_type, "tool") && tc_name) {
                    free(r->tool_force_name);
                    r->tool_force_name = tc_name;
                    tc_name = NULL;
                } else if (tc_type && !strcmp(tc_type, "any")) {
                    r->tool_force_any = true;
                }
                free(tc_type); free(tc_name);
                if (*p != '}') {
                    free(key);
                    goto bad;
                }
                p++;
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
        } else if (!strcmp(key, "max_tokens")) {
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
            r->temperature_set = true;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &r->top_k)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stop_sequences")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "output_config")) {
            if (!parse_output_config_effort(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
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
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    if (system && system[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = system;
        system = NULL;
        chat_msgs_push(&msgs, msg);
    }
    r->has_tools = tool_schemas && tool_schemas[0] && !tool_choice_none;
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    if (g_force_nothink) r->think_mode = DS4_THINK_NONE;
    if (!anthropic_validate_tool_results(s, &msgs,
                                         &r->anthropic_requires_live_tool_state,
                                         err, errlen))
    {
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    anthropic_prepare_live_continuation(r, &msgs);
    const char *active_tool_schemas = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    g_req_mode = r->ds4_mode;   /* 渲染前置位(单图worker串行, 静态安全同 argmax scratch 纪律) */
    r->prompt_text = render_chat_prompt_text(&msgs, active_tool_schemas,
                                             &r->tool_orders, r->think_mode,
                                             &r->prompt_conv_off);
    base_native_default_stops(&r->stops);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}
