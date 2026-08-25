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

void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}

void json_escape_fragment_n(buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
}

const char *find_any_tool_start(const char *s) {
    const char *best = NULL;
    const char *candidates[] = {
        strstr(s, DS4_TOOL_CALLS_START),
        strstr(s, DS4_TOOL_CALLS_START_SHORT),
        strstr(s, "<tool_calls>"),
    };
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        if (candidates[i] && (!best || candidates[i] < best)) best = candidates[i];
    }
    return best;
}

static const char *find_any_tool_end(const char *s) {
    const char *best = NULL;
    const char *candidates[] = {
        strstr(s, DS4_TOOL_CALLS_END),
        strstr(s, DS4_TOOL_CALLS_END_SHORT),
        strstr(s, "</tool_calls>"),
    };
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        if (candidates[i] && (!best || candidates[i] < best)) best = candidates[i];
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
            buf_putc(&b, *s);
        } else if (!strncmp(s, "&amp;", 5)) {
            buf_putc(&b, '&');
            s += 4;
        } else if (!strncmp(s, "&lt;", 4)) {
            buf_putc(&b, '<');
            s += 3;
        } else if (!strncmp(s, "&gt;", 4)) {
            buf_putc(&b, '>');
            s += 3;
        } else if (!strncmp(s, "&quot;", 6)) {
            buf_putc(&b, '"');
            s += 5;
        } else if (!strncmp(s, "&apos;", 6)) {
            buf_putc(&b, '\'');
            s += 5;
        } else {
            buf_putc(&b, '&');
        }
    }
    return buf_take(&b);
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
    if (args->len) buf_puts(args, ", ");
    json_escape(args, name ? name : "");
    buf_puts(args, ": ");
    if (is_string && !strcmp(is_string, "true")) {
        json_escape(args, value ? value : "");
    } else {
        char *min = json_minify_raw_value(value ? value : "null");
        buf_puts(args, min && min[0] ? min : "null");
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
static bool dsml_parse_leaf_param_json(const char **p_in, const char *param_start,
                                       const char *param_end, buf *out) {
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
        dsml_unescape_text(raw_value) : xstrdup(raw_value);
    tool_call_json_args_add(out, name, value, type);

    free(name);
    free(is_string);
    free(raw_value);
    free(value);
    *p_in = value_end + strlen(param_end);
    return true;
}

static bool dsml_parse_nested_params_object(const char **p_in,
                                            const char *param_start,
                                            const char *param_end,
                                            buf *out) {
    const char *p = *p_in;
    buf members = {0};
    bool any = false;

    for (;;) {
        p = skip_ascii_ws(p);
        if (strncmp(p, param_start, strlen(param_start)) != 0) break;
        if (!dsml_parse_leaf_param_json(&p, param_start, param_end, &members)) {
            buf_free(&members);
            return false;
        }
        any = true;
    }

    if (!any) {
        buf_free(&members);
        return false;
    }
    buf_putc(out, '{');
    buf_puts(out, members.ptr ? members.ptr : "");
    buf_putc(out, '}');
    buf_free(&members);
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
            /* Model did not close thinking, ignore any DSML in reasoning */
            fprintf(stderr, "ds4-server: thinking not closed, ignoring DSML in reasoning\n");
            split_reasoning_content(text, strlen(text), content_out, reasoning_out);
            return true;
        }
        tool_search = think_end + 8;
    }

    const char *start = strstr(tool_search, "\n\n" DS4_TOOL_CALLS_START);
    int style = 0; /* 0: DSML, 1: plain XML, 2: DSML with the first vertical bar omitted. */
    if (!start) start = strstr(tool_search, DS4_TOOL_CALLS_START);
    if (!start) {
        start = strstr(tool_search, "\n\n" DS4_TOOL_CALLS_START_SHORT);
        style = start ? 2 : style;
    }
    if (!start) {
        start = strstr(tool_search, DS4_TOOL_CALLS_START_SHORT);
        style = start ? 2 : style;
    }
    if (!start) {
        start = strstr(tool_search, "\n\n<tool_calls>");
        style = start ? 1 : style;
    }
    if (!start) {
        start = strstr(tool_search, "<tool_calls>");
        style = start ? 1 : style;
    }
    if (!start) {
        split_reasoning_content(text, strlen(text), content_out, reasoning_out);
        return true;
    }

    size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(start - text));
    const char *raw_block_start = start;
    const char *tool_calls_start = DS4_TOOL_CALLS_START;
    const char *tool_calls_end = DS4_TOOL_CALLS_END;
    const char *invoke_start = DS4_INVOKE_START;
    const char *invoke_end = DS4_INVOKE_END;
    const char *param_start = DS4_PARAM_START;
    const char *param_end = DS4_PARAM_END;
    if (style == 1) {
        tool_calls_start = "<tool_calls>";
        tool_calls_end = "</tool_calls>";
        invoke_start = "<invoke";
        invoke_end = "</invoke>";
        param_start = "<parameter";
        param_end = "</parameter>";
    } else if (style == 2) {
        tool_calls_start = DS4_TOOL_CALLS_START_SHORT;
        tool_calls_end = DS4_TOOL_CALLS_END_SHORT;
        invoke_start = DS4_INVOKE_START_SHORT;
        invoke_end = DS4_INVOKE_END_SHORT;
        param_start = DS4_PARAM_START_SHORT;
        param_end = DS4_PARAM_END_SHORT;
    }

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
                buf_free(&args);
                return false;
            }
            tag_end = strchr(p, '>');
            if (!tag_end) {
                free(name);
                buf_free(&args);
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
                buf_free(&args);
                return false;
            }
            const char *value_start = tag_end + 1;
            if (!param_is_string &&
                !strncmp(skip_ascii_ws(value_start), param_start, strlen(param_start)))
            {
                buf nested = {0};
                const char *nested_p = value_start;
                if (!dsml_parse_nested_params_object(&nested_p, param_start,
                                                     param_end, &nested)) {
                    free(name);
                    free(param_name);
                    buf_free(&nested);
                    buf_free(&args);
                    return false;
                }
                tool_call_json_args_add(&args, param_name,
                                        nested.ptr ? nested.ptr : "{}",
                                        "false");
                buf_free(&nested);
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
                buf_free(&args);
                return false;
            }
            char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
            const char *type = param_is_string ? param_is_string : "true";
            char *value = !strcmp(type, "true") ?
                dsml_unescape_text(raw_value) : xstrdup(raw_value);
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
        buf_putc(&wrapped, '{');
        buf_puts(&wrapped, args.ptr ? args.ptr : "");
        buf_putc(&wrapped, '}');
        tc.arguments = buf_take(&wrapped);
        tool_calls_push(calls, tc);
        buf_free(&args);
    }
}
