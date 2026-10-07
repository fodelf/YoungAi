/* server_http.c — 机械拆分自 ds4_server.c (5243-5628 行): HTTP/CORS/SSE 应答基元与 chat 页。 */

#include "server_internal.h"

static const char *tool_parse_failure_recovery_finish(const char *finish) {
    /* Once DSML failed to parse there is no executable tool call to report.
     * Preserve a true length stop, because callers can distinguish truncation
     * from a completed turn.  Every other non-error tool-parse failure becomes
     * a normal assistant stop with the raw model text returned as content. */
    if (finish && !strcmp(finish, "length")) return "length";
    return "stop";
}

bool parse_generated_message_for_response(const char *text,
                                                 bool has_tools,
                                                 bool saw_tool_start,
                                                 bool require_thinking_closed,
                                                 const char **finish_io,
                                                 char *err,
                                                 size_t errlen,
                                                 char **content_out,
                                                 char **reasoning_out,
                                                 tool_calls *calls,
                                                 bool *recovered_out) {
    if (recovered_out) *recovered_out = false;

    bool parsed_ok = parse_generated_message_ex(text ? text : "",
                                                require_thinking_closed,
                                                content_out, reasoning_out,
                                                calls);
    if (parsed_ok) return true;

    free(*content_out);
    free(*reasoning_out);
    *content_out = xstrdup(text ? text : "");
    *reasoning_out = NULL;
    tool_calls_free(calls);

    /* A malformed tool block is model output, not a server failure.  The
     * generation worker may hide this turn from the client, append a tool error
     * plus protocol reminder to the live session, and let the model try again.
     * If that continuation is unavailable, parsed_content keeps the raw text as
     * a last-resort assistant fallback instead of crashing the request. */
    const char *finish = finish_io && *finish_io ? *finish_io : "stop";
    if (has_tools && saw_tool_start && strcmp(finish, "error") != 0) {
        if (finish_io) *finish_io = tool_parse_failure_recovery_finish(finish);
        if (err && errlen) snprintf(err, errlen, "invalid tool call");
        if (recovered_out) *recovered_out = true;
    }
    return false;
}

void append_json_object_string(buf *b, const char *json) {
    buf tmp = {0};
    append_json_object_or_empty(&tmp, json);
    json_escape(b, tmp.ptr ? tmp.ptr : "{}");
    buf_free(&tmp);
}

void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}

void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"index\":");
        buf_printf(b, "%d", i);
        buf_puts(b, ",\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}

static void append_cors_headers(buf *h) {
    buf_puts(h,
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n");
}

bool http_response(int fd, bool enable_cors, int code, const char *type, const char *body) {
    const char *reason = code == 200 ? "OK" :
                         code == 204 ? "No Content" :
                         code == 400 ? "Bad Request" :
                         code == 404 ? "Not Found" :
                         code == 409 ? "Conflict" :
                         code == 500 ? "Internal Server Error" : "Error";
    const size_t body_len = body ? strlen(body) : 0;
    buf h = {0};
    buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Length: %zu\r\n",
        code, reason, body_len);
    if (type && type[0]) {
        buf_puts(&h, "Content-Type: ");
        buf_puts(&h, type);
        buf_puts(&h, "\r\n");
    }
    if (enable_cors) append_cors_headers(&h);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    if (ok && body_len) ok = send_all(fd, body, body_len);
    buf_free(&h);
    return ok;
}

bool http_error(int fd, bool enable_cors, int code, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    json_escape(&b, msg);
    buf_puts(&b, ",\"type\":\"invalid_request_error\"}}\n");
    bool ok = http_response(fd, enable_cors, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* 对端还在不在: 非阻塞探一下, 不读走任何数据。
 *
 * ★为什么要它★(2026-09-22 早盘实撞): 非流式请求在生成期间一个字节都不写, 客户端超时放弃/断开时
 * 服务端毫不知情, 会把一条 16384 token 的请求一直算到底。那天一条 132k token 的请求被客户端
 * 重发了 4 遍(贪心 ⇒ 四次输出逐字节相同), 102 分钟 GPU 全算给了已经挂断的连接; 而本服务串行跑图,
 * 这些僵尸请求把队列堵成"每一次重试都必然超时"的死循环。流式路本来就能靠写失败发现(见 sse_chunk),
 * 非流式路只能自己探。
 *
 * 判据: POLLHUP/POLLERR/POLLNVAL, 或者可读且 recv(MSG_PEEK) 返回 0(对端发过 FIN)。
 * ★可读且真有数据不算走★: 那是客户端在发下一个请求的字节, 本服务一条连接只服务一条请求, 不拿它当断开。
 * 不用 POLLRDHUP: 它要 _GNU_SOURCE 且 macOS 没有, 而半关闭在两边都表现为"可读 + peek 到 0 字节"。 */
bool client_disconnected(int fd) {
    if (fd < 0) return false;
    struct pollfd pfd = {.fd = fd, .events = POLLIN, .revents = 0};
    int rc;
    do {
        rc = poll(&pfd, 1, 0);
    } while (rc < 0 && errno == EINTR);
    if (rc <= 0) return false;   /* 出错也当"还在": 探测本身绝不能成为掐生成的理由 */
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return true;
    if (pfd.revents & POLLIN) {
        char probe;
        ssize_t n;
        do {
            n = recv(fd, &probe, 1, MSG_PEEK);
        } while (n < 0 && errno == EINTR);
        if (n == 0) return true;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) return true;
    }
    return false;
}

static const char *context_length_error_param(const request *r) {
    if (!r) return "prompt";
    if (r->api == API_RESPONSES) return "input";
    return r->kind == REQ_COMPLETION ? "prompt" : "messages";
}

bool request_exceeds_context(const request *r, int ctx_size) {
    /* ds4_session_sync() rejects prompt->len >= ctx_size because generation
     * needs at least one free context slot.  Catch the same boundary here so
     * clients get a normal protocol error instead of a later backend failure. */
    return r && r->prompt.len >= ctx_size;
}

bool http_error_context_length_exceeded(int fd, bool enable_cors,
                                               const request *r,
                                               int n_prompt_tokens,
                                               int ctx_size) {
    buf b = {0};
    char msg[160];
    snprintf(msg, sizeof(msg),
             "Prompt has %d tokens, but the configured context size is %d tokens",
             n_prompt_tokens, ctx_size);

    if (r && r->api == API_ANTHROPIC) {
        buf_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"n_prompt_tokens\":");
        buf_printf(&b, "%d", n_prompt_tokens);
        buf_puts(&b, ",\"n_ctx\":");
        buf_printf(&b, "%d", ctx_size);
        buf_puts(&b, "}}\n");
    } else {
        buf_puts(&b, "{\"error\":{\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"type\":\"invalid_request_error\",\"param\":");
        json_escape(&b, context_length_error_param(r));
        buf_puts(&b, ",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":");
        buf_printf(&b, "%d", n_prompt_tokens);
        buf_puts(&b, ",\"n_ctx\":");
        buf_printf(&b, "%d", ctx_size);
        buf_puts(&b, "}}\n");
    }
    bool ok = http_response(fd, enable_cors, 400, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static char *read_text_file_dup(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long n = ftell(fp);
    if (n < 0) { fclose(fp); return NULL; }
    rewind(fp);
    char *text = malloc((size_t)n + 1);
    if (!text) { fclose(fp); return NULL; }
    size_t got = fread(text, 1, (size_t)n, fp);
    fclose(fp);
    text[got] = '\0';
    return text;
}

bool serve_chat_page(int fd, bool enable_cors, const char *path) {
    char *html = read_text_file_dup(path);
    if (!html) {
        return http_error(fd, enable_cors, 404,
                          "chat page file " DS4_CHAT_PAGE_FILE " not found; "
                          "start ds4-server from the repo root or pass --chdir");
    }
    bool ok = http_response(fd, enable_cors, 200, "text/html; charset=utf-8", html);
    free(html);
    return ok;
}

/* 监控页等其它静态页: 与 serve_chat_page 同一套 cwd 契约, 404 文案报的是传入的路径(chat 页那条为了老测试文案不动)。 */
bool serve_page_file(int fd, bool enable_cors, const char *path) {
    char *html = read_text_file_dup(path);
    if (!html) {
        char msg[400];
        snprintf(msg, sizeof msg, "page file %s not found; start ds4-server from the repo root or pass --chdir", path);
        return http_error(fd, enable_cors, 404, msg);
    }
    bool ok = http_response(fd, enable_cors, 200, "text/html; charset=utf-8", html);
    free(html);
    return ok;
}

/* Route match that tolerates a query string ("/?from=x" is still "/"). */
bool path_route_is(const char *path, const char *route) {
    size_t n = strlen(route);
    return strncmp(path, route, n) == 0 && (path[n] == '\0' || path[n] == '?');
}

bool sse_headers(int fd, bool enable_cors) {
    buf h = {0};
    buf_puts(&h,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n");
    if (enable_cors) append_cors_headers(&h);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    buf_free(&h);
    return ok;
}

bool sse_error_event(int fd, const request *r, const char *msg) {
    const char *message = msg && msg[0] ? msg : "internal server error";
    buf b = {0};
    if (r && r->api == API_ANTHROPIC) {
        buf_puts(&b, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":");
        json_escape(&b, message);
        buf_puts(&b, "}}\n\n");
    } else {
        buf_puts(&b, "event: error\ndata: {\"error\":{\"message\":");
        json_escape(&b, message);
        buf_puts(&b, ",\"type\":\"server_error\"}}\n\n");
    }
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":");
        if (text) {
            buf_puts(&b, "{\"content\":");
            json_escape(&b, text);
            buf_putc(&b, '}');
        } else {
            buf_puts(&b, finish ? "{}" : "{\"role\":\"assistant\"}");
        }
        buf_puts(&b, ",\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text ? text : "");
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    }
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

int clamp_usage_tokens(int value, int max) {
    if (value < 0) return 0;
    if (max >= 0 && value > max) return max;
    return value;
}

void append_openai_usage_json(buf *b, const request *r,
                                     int prompt_tokens, int completion_tokens) {
    int cached_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    cached_tokens = clamp_usage_tokens(cached_tokens, prompt_tokens);
    cache_write_tokens = clamp_usage_tokens(cache_write_tokens, prompt_tokens - cached_tokens);
    /* OpenAI defines cached_tokens as prompt tokens retrieved from cache.
     * Newly-prefilled tokens are useful to expose, but they are a DS4 extension
     * and must stay separate so OpenAI-compatible clients do not over-count
     * cache hits. */
    buf_printf(b,
               "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
               "\"prompt_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d}}",
               prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
               cached_tokens, cache_write_tokens);
}

static bool sse_usage_chunk(int fd, const request *r, const char *id,
                            int prompt_tokens, int completion_tokens) {
    if (!r->stream_include_usage) return true;

    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    }
    append_openai_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n\n");

    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

bool sse_done(int fd, const request *r, const char *id,
                     int prompt_tokens, int completion_tokens) {
    return sse_usage_chunk(fd, r, id, prompt_tokens, completion_tokens) &&
           send_all(fd, "data: [DONE]\n\n", 14);
}

bool sse_chat_finish(int fd, const request *r, const char *id, const char *content,
                            const char *reasoning, const tool_calls *calls, const char *finish,
                            int prompt_tokens, int completion_tokens) {
    if (!sse_chunk(fd, r, id, NULL, NULL)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (reasoning && reasoning[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":");
        json_escape(&b, reasoning);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (content && content[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        json_escape(&b, content);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (calls && calls->len) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(&b);
    return ok;
}
