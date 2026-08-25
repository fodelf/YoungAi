/* server_parse_completion.c — 机械拆分自 ds4_server.c (4551-4748 行): completions 请求解析。 */

#include "server_internal.h"

static bool parse_prompt(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    if (**p == '"') {
        if (!json_string(p, out)) return false;
    } else {
        *out = xstrdup("");
        if (**p && **p != ']' && !json_skip_value(p)) return false;
    }
    while (**p && **p != ']') {
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            if (!json_skip_value(p)) return false;
        } else {
            break;
        }
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

bool parse_completion_request(ds4_engine *e, const char *body, int def_tokens,
                                     int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_COMPLETION, def_tokens);
    const char *p = body;
    char *prompt = NULL;
    bool got_thinking = false;
    bool thinking_enabled = true;
    bool raw = false; /* "raw":true => bare continuation: prompt is the exact rendered
                         text (caller supplies BOS); no chat frame. Base-model probe path. */
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;

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
        if (!strcmp(key, "prompt")) {
            free(prompt);
            if (!parse_prompt(&p, &prompt)) {
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
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "min_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->min_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &r->top_k)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "seed")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->seed = v > 0.0 ? (uint64_t)v : 0;
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream_options")) {
            if (!parse_stream_options(&p, &r->stream_include_usage)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "raw")) {
            if (!json_bool(&p, &raw)) {
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
    if (!prompt) {
        snprintf(err, errlen, "missing prompt");
        request_free(r);
        return false;
    }
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    if (g_force_nothink) r->think_mode = DS4_THINK_NONE;
    if (raw) {
        /* Bare continuation: no frame, no injected BOS/think tokens. The prompt text is
         * tokenized as rendered chat so DS4 special-token literals resolve; a base-model
         * probe passes its own BOS explicitly. */
        r->prompt_text = prompt;
        ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
        return true;
    }
    buf rendered = {0};
    buf_puts(&rendered, "<｜begin▁of▁sentence｜>");
    if (r->think_mode == DS4_THINK_MAX) buf_puts(&rendered, ds4_think_max_prefix());
    buf_puts(&rendered, "You are a helpful assistant<｜User｜>");
    buf_puts(&rendered, prompt);
    buf_puts(&rendered, "<｜Assistant｜>");
    buf_puts(&rendered, ds4_think_mode_enabled(r->think_mode) ? "<think>" : "</think>");
    r->prompt_text = buf_take(&rendered);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    free(prompt);
    return true;
bad:
    free(prompt);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}
