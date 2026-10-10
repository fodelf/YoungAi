/* server_dsml_parse.c — 机械拆分自 ds4_server.c (4749-5242 行): 生成文本的 DSML 解析。 */

#include "server_internal.h"

static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

bool send_all(int fd, const void *p, size_t n) {
    const char *s = p;
    long long deadline = wall_ms() + DS4_SERVER_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
        ssize_t w = send(fd, s, n, 0);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + DS4_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}

void json_escape(buf *b, const char *s) { ds4_json_escape(b, s); }   /* 实现在 src/common/ds4_json.c(训练页 daemon 共用) */

void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}

void json_escape_fragment_n(buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            ds4_buf_putc(b, '\\');
            ds4_buf_putc(b, (char)c);
        } else if (c == '\n') {
            ds4_buf_puts(b, "\\n");
        } else if (c == '\r') {
            ds4_buf_puts(b, "\\r");
        } else if (c == '\t') {
            ds4_buf_puts(b, "\\t");
        } else if (c < 0x20) {
            ds4_buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            ds4_buf_putc(b, (char)c);
        }
    }
}

/* 所有 DSML 写法里最早出现的块起点。need_sep: 只认前面紧跟 "\n\n" 的(模板在正文与工具块之间写的分隔),
 * 命中时返回指向 "\n\n" —— raw_dsml 要带着它, 重放时才与采样出的字节一致。 */
static const char *find_tool_block(const char *s, bool need_sep, const dsml_syntax **syn_out) {
    const char *best = NULL;
    for (int i = 0; i < DSML_SYN_COUNT; i++) {
        const char *p = strstr(s, dsml_syntaxes[i].tool_calls_start);
        while (p && need_sep && !(p - s >= 2 && p[-1] == '\n' && p[-2] == '\n'))
            p = strstr(p + 1, dsml_syntaxes[i].tool_calls_start);
        if (p && need_sep) p -= 2;
        if (p && (!best || p < best)) {
            best = p;
            if (syn_out) *syn_out = &dsml_syntaxes[i];
        }
    }
    return best;
}

const char *find_any_tool_start(const char *s) {
    return find_tool_block(s, false, NULL);
}

static const char *find_any_tool_end(const char *s) {
    const char *best = NULL;
    for (int i = 0; i < DSML_SYN_COUNT; i++) {
        const char *p = strstr(s, dsml_syntaxes[i].tool_calls_end);
        if (p && (!best || p < best)) best = p;
    }
    return best;
}

void observe_tool_markers(const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end) {
    if (!scan) return;
    bool had_start = *saw_start;
    const char *start = find_any_tool_start(scan);
    if (start) *saw_start = true;

    const char *end_scan = had_start ? scan : (start ? start : NULL);
    const char *end = end_scan ? find_any_tool_end(end_scan) : NULL;
    if (end) {
        *saw_end = true;
    } else if (!had_start && !start && find_any_tool_end(scan)) {
        if (orphan_end) *orphan_end = true;
    }
}

size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit) {
    while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
    return limit;
}

static const char *skip_ascii_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

const char *find_last_substr(const char *s, const char *needle) {
    if (!s || !needle || !needle[0]) return NULL;
    const char *last = NULL;
    const char *p = s;
    while ((p = strstr(p, needle)) != NULL) {
        last = p;
        p++;
    }
    return last;
}

/* The prompt renderer escapes DSML text so a tool argument can safely contain
 * shell operators or closing tags.  The generated-DSML parser must undo exactly
 * those entities before it turns parameters back into JSON; otherwise
 * parse->render is not a stable cache key. */
char *dsml_unescape_text(const char *s) {
    buf b = {0};
    for (s = s ? s : ""; *s; s++) {
        if (*s != '&') {
            ds4_buf_putc(&b, *s);
        } else if (!strncmp(s, "&amp;", 5)) {
            ds4_buf_putc(&b, '&');
            s += 4;
        } else if (!strncmp(s, "&lt;", 4)) {
            ds4_buf_putc(&b, '<');
            s += 3;
        } else if (!strncmp(s, "&gt;", 4)) {
            ds4_buf_putc(&b, '>');
            s += 3;
        } else if (!strncmp(s, "&quot;", 6)) {
            ds4_buf_putc(&b, '"');
            s += 5;
        } else if (!strncmp(s, "&apos;", 6)) {
            ds4_buf_putc(&b, '\'');
            s += 5;
        } else {
            ds4_buf_putc(&b, '&');
        }
    }
    return ds4_buf_take(&b);
}

char *dsml_attr(const char *tag, const char *name) {
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    const char *q = strchr(p, '"');
    if (!q) return NULL;
    char *raw = xstrndup(p, (size_t)(q - p));
    char *decoded = dsml_unescape_text(raw);
    free(raw);
    return decoded;
}

static void tool_call_json_args_add(buf *args, const char *name, const char *value, const char *is_string) {
    if (args->len) ds4_buf_puts(args, ", ");
    json_escape(args, name ? name : "");
    ds4_buf_puts(args, ": ");
    if (is_string && !strcmp(is_string, "true")) {
        json_escape(args, value ? value : "");
    } else {
        char *min = json_minify_raw_value(value ? value : "null");
        ds4_buf_puts(args, min && min[0] ? min : "null");
        free(min);
    }
}

/* DSML produced by the model is usually a flat list of typed parameters:
 *
 *   <parameter name="path" string="true">/tmp/x</parameter>
 *   <parameter name="timeout" string="false">10</parameter>
 *
 * Long generations sometimes drift into a looser XML-ish shape, omitting the
 * outer string attribute and putting child parameters inside it.  The server
 * does not know client tool schemas, so it cannot make that semantically
 * perfect.  Still, returning a structured JSON value lets the client/tool layer
 * reject or repair the call, which is much better than aborting the assistant
 * turn and losing the whole sampled continuation.
 */
static bool dsml_parse_leaf_param_json(const char **p_in, const dsml_syntax *syn, buf *out) {
    const char *param_start = syn->param_start, *param_end = syn->param_end;
    const char *p = *p_in;
    if (strncmp(p, param_start, strlen(param_start)) != 0) return false;
    const char *tag_end = strchr(p, '>');
    if (!tag_end) return false;

    char *tag = xstrndup(p, (size_t)(tag_end - p + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name) {
        free(is_string);
        return false;
    }

    const char *value_start = tag_end + 1;
    const char *value_end = strstr(value_start, param_end);
    if (!value_end) {
        free(name);
        free(is_string);
        return false;
    }

    char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
    const char *type = is_string ? is_string : "true";
    char *value = !strcmp(type, "true") ?
        dsml_value_unescape(syn, raw_value) : xstrdup(raw_value);
    tool_call_json_args_add(out, name, value, type);

    free(name);
    free(is_string);
    free(raw_value);
    free(value);
    *p_in = value_end + strlen(param_end);
    return true;
}

static bool dsml_parse_nested_params_object(const char **p_in, const dsml_syntax *syn, buf *out) {
    const char *p = *p_in;
    buf members = {0};
    bool any = false;

    for (;;) {
        p = skip_ascii_ws(p);
        if (strncmp(p, syn->param_start, strlen(syn->param_start)) != 0) break;
        if (!dsml_parse_leaf_param_json(&p, syn, &members)) {
            ds4_buf_free(&members);
            return false;
        }
        any = true;
    }

    if (!any) {
        ds4_buf_free(&members);
        return false;
    }
    ds4_buf_putc(out, '{');
    ds4_buf_puts(out, members.ptr ? members.ptr : "");
    ds4_buf_putc(out, '}');
    ds4_buf_free(&members);
    *p_in = p;
    return true;
}

static void split_reasoning_content(const char *text, size_t n, char **content_out, char **reasoning_out) {
    char *s = xstrndup(text ? text : "", n);
    char *body = s;
    if (!strncmp(body, "<think>", 7)) body += 7;

    char *think_end = strstr(body, "</think>");
    if (think_end) {
        *think_end = '\0';
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup(think_end + 8);
    } else {
        *reasoning_out = NULL;
        *content_out = xstrdup(s);
    }
    free(s);
}

bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                       char **content_out, char **reasoning_out,
                                       tool_calls *calls) {
    text = text ? text : "";
    const char *tool_search = text;

    /* When thinking mode is enabled the model is expected to close
     * </think> before it enters the executable assistant surface.  DSML inside
     * reasoning is just model text: it may be a mistaken attempt, a quotation,
     * or an explanation of the protocol.  Treating it as a real tool call
     * duplicates it into both reasoning and structured tool_calls, and can make
     * clients execute something the assistant had not actually emitted as its
     * post-thinking action. */
    if (require_thinking_closed) {
        const char *think_end = find_last_substr(text, "</think>");
        if (!think_end) {
            /* ★思考没闭合 = 这一轮没有正文★(2026-09-22 改, 按官方 deepseek-reasoner 语义: 思考被上限截断时
             * reasoning_content 放已想的部分, content 空, finish_reason=length)。
             * 以前这里把整段思考当 content 返回, 于是同一台服务两种协议给相反的答案 —— 流式路
             * (server_openai_stream.c OPENAI_STREAM_THINKING)看不到 </think> 时全部走 reasoning_content,
             * 非流式路却把它当正文。实撞代价(2026-09-22 早盘): 大盘趋势那条 16384 token 全是英文思考,
             * 调用方当成最终答案收下, 再用子串匹配从复述的提示模板里"解析"出涨跌结论, 据此启动了下游选股。
             * DSML 照旧不认(思考里的工具块不可执行, 这条没变)。 */
            fprintf(stderr, "ds4-server: thinking not closed, returning it as reasoning with empty content\n");
            const char *body = text;
            if (!strncmp(body, "<think>", 7)) body += 7;
            *reasoning_out = xstrdup(body);
            *content_out = xstrdup("");
            return true;
        }
        tool_search = think_end + 8;
    }

    /* 先找模板里那种带 "\n\n" 分隔的块, 找不到再退到裸标记: 正文中途引用一个标签不该截走后面真正的工具块 */
    const dsml_syntax *syn = NULL;
    const char *start = find_tool_block(tool_search, true, &syn);
    if (!start) start = find_tool_block(tool_search, false, &syn);
    if (!start) {
        split_reasoning_content(text, strlen(text), content_out, reasoning_out);
        return true;
    }

    size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(start - text));
    const char *raw_block_start = start;
    const char *tool_calls_start = syn->tool_calls_start;
    const char *tool_calls_end = syn->tool_calls_end;
    const char *invoke_start = syn->invoke_start;
    const char *invoke_end = syn->invoke_end;
    const char *param_start = syn->param_start;
    const char *param_end = syn->param_end;

    const char *p = strstr(start, tool_calls_start);
    if (!p) return false;
    p += strlen(tool_calls_start);

    for (;;) {
        p = skip_ascii_ws(p);
        if (!strncmp(p, tool_calls_end, strlen(tool_calls_end))) {
            const char *raw_block_end = p + strlen(tool_calls_end);
            free(calls->raw_dsml);
            calls->raw_dsml = xstrndup(raw_block_start, (size_t)(raw_block_end - raw_block_start));
            split_reasoning_content(text, content_len, content_out, reasoning_out);
            return true;
        }
        if (strncmp(p, invoke_start, strlen(invoke_start)) != 0) return false;
        const char *tag_end = strchr(p, '>');
        if (!tag_end) return false;
        char *tag = xstrndup(p, (size_t)(tag_end - p + 1));
        char *name = dsml_attr(tag, "name");
        free(tag);
        if (!name) return false;
        p = tag_end + 1;

        buf args = {0};
        while (true) {
            p = skip_ascii_ws(p);
            if (!strncmp(p, invoke_end, strlen(invoke_end))) {
                p += strlen(invoke_end);
                break;
            }
            if (strncmp(p, param_start, strlen(param_start)) != 0) {
                free(name);
                ds4_buf_free(&args);
                return false;
            }
            tag_end = strchr(p, '>');
            if (!tag_end) {
                free(name);
                ds4_buf_free(&args);
                return false;
            }
            tag = xstrndup(p, (size_t)(tag_end - p + 1));
            char *param_name = dsml_attr(tag, "name");
            char *param_is_string = dsml_attr(tag, "string");
            free(tag);
            if (!param_name) {
                free(name);
                free(param_name);
                free(param_is_string);
                ds4_buf_free(&args);
                return false;
            }
            const char *value_start = tag_end + 1;
            if (!param_is_string &&
                !strncmp(skip_ascii_ws(value_start), param_start, strlen(param_start)))
            {
                buf nested = {0};
                const char *nested_p = value_start;
                if (!dsml_parse_nested_params_object(&nested_p, syn, &nested)) {
                    free(name);
                    free(param_name);
                    ds4_buf_free(&nested);
                    ds4_buf_free(&args);
                    return false;
                }
                tool_call_json_args_add(&args, param_name,
                                        nested.ptr ? nested.ptr : "{}",
                                        "false");
                ds4_buf_free(&nested);
                p = skip_ascii_ws(nested_p);
                if (!strncmp(p, param_end, strlen(param_end))) {
                    p += strlen(param_end);
                }
                free(param_name);
                continue;
            }
            const char *value_end = strstr(value_start, param_end);
            if (!value_end) {
                free(name);
                free(param_name);
                free(param_is_string);
                ds4_buf_free(&args);
                return false;
            }
            char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
            const char *type = param_is_string ? param_is_string : "true";
            char *value = !strcmp(type, "true") ?
                dsml_value_unescape(syn, raw_value) : xstrdup(raw_value);
            tool_call_json_args_add(&args, param_name, value, type);
            free(param_name);
            free(param_is_string);
            free(raw_value);
            free(value);
            p = value_end + strlen(param_end);
        }

        tool_call tc = {0};
        tc.name = name;
        buf wrapped = {0};
        ds4_buf_putc(&wrapped, '{');
        ds4_buf_puts(&wrapped, args.ptr ? args.ptr : "");
        ds4_buf_putc(&wrapped, '}');
        tc.arguments = ds4_buf_take(&wrapped);
        tool_calls_push(calls, tc);
        ds4_buf_free(&args);
    }
}
