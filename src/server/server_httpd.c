/* server_httpd.c — 机械拆分自 ds4_server.c (12449-12786 行): HTTP 读写/模型列表/客户端线程。 */

#include "server_internal.h"
#include "../train/train_internal.h"

static void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            return strtol(v, NULL, 10);
        }
        if (p < end) p++;
    }
    return 0;
}

/* Accept 头里有没有 text/plain 或 openmetrics: Prometheus 抓 /metrics 就是这么问的(照 Strata wants_prometheus 的判法) */
bool http_accepts_text(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 7 && strncasecmp(line, "Accept:", 7) == 0) {
            char v[512];
            snprintf(v, sizeof v, "%.*s", (int)(len - 7 < sizeof v - 1 ? len - 7 : sizeof v - 1), line + 7);
            for (char *c = v; *c; c++) *c = (char)tolower((unsigned char)*c);
            return strstr(v, "text/plain") != NULL || strstr(v, "openmetrics") != NULL;
        }
        if (p < end) p++;
    }
    return false;
}

static bool read_http_request(int fd, http_request *r) {
    buf b = {0};
    ssize_t hend = -1;
    const size_t max_header = 64 * 1024;
    const size_t max_body = 64 * 1024 * 1024;

    while (hend < 0 && b.len < max_header) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        ds4_buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q = strchr(r->path, '?');
    r->query[0] = '\0';
    if (q) { *q = '\0'; snprintf(r->query, sizeof r->query, "%s", q + 1); }
    r->accept_text = http_accepts_text(b.ptr, (size_t)hend);

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        ds4_buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    ds4_buf_free(&b);
    return true;
fail:
    ds4_buf_free(&b);
    return false;
}

void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    ds4_buf_printf(b,
        "{\"id\":");
    json_escape(b, id);
    ds4_buf_puts(b,
        ",\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"ds4.c\","
        "\"name\":");
    json_escape(b, name);
    ds4_buf_printf(b,
        ","
        "\"context_length\":%d,"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"frequency_penalty\","
            "\"presence_penalty\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"reasoning_effort\"]}",
        ctx,
        ctx,
        max_completion);
}

static void append_model_json(buf *b, const server *s, const char *id) {
    append_model_json_values(b,
                             id,
                             ds4_engine_model_name(s->engine),
                             server_ctx_size(s),
                             s->default_tokens);
}

static bool send_model(server *s, int fd, const char *id) {
    buf b = {0};
    append_model_json(&b, s, id);
    ds4_buf_putc(&b, '\n');
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    ds4_buf_free(&b);
    return ok;
}

static bool send_models(server *s, int fd) {
    buf b = {0};
    ds4_buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    append_model_json(&b, s, "deepseek-v4-flash");
    ds4_buf_putc(&b, ',');
    append_model_json(&b, s, "deepseek-v4-pro");
    ds4_buf_puts(&b, "]}\n");
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    ds4_buf_free(&b);
    return ok;
}

static void client_done(server *s) {
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}

void *client_main(void *arg) {
    client_arg *ca = arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, s->enable_cors, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, s->enable_cors, 204, NULL, "");
        http_request_free(&hr);
        goto done;
    }

    /* 首页 = 工作台(web/studio.html: 聊天/训练/语料/记录, 10-10); 老的单页聊天留在 /chat */
    if (!strcmp(hr.method, "GET") && path_route_is(hr.path, "/chat")) {
        serve_chat_page(fd, s->enable_cors, DS4_CHAT_PAGE_FILE);
        http_request_free(&hr);
        goto done;
    }

    /* 工作台页与它的接口(src/train/train_api.c, 与主进程 ds4-train 共用一份): 这里是模型子进程, 页面能聊、能热切侧车/③(server_plugins.c);
     * 发车/换模型/起服这些管子进程的操作 tr_api 在 serving=1 时拒 —— 它们归主进程(主进程把 /api/models/plugins|list 转到这里) */
    if (!strcmp(hr.path, "/") || !strcmp(hr.path, "/index.html") || !strcmp(hr.path, "/studio") || !strcmp(hr.path, "/train") || !strcmp(hr.path, "/logo.jpg") || !strcmp(hr.path, "/favicon.ico") || !strncmp(hr.path, "/api/train/", 11) || !strncmp(hr.path, "/api/models/", 12)) {
        tr_req rq; memset(&rq, 0, sizeof rq);
        snprintf(rq.method, sizeof rq.method, "%s", hr.method); snprintf(rq.path, sizeof rq.path, "%s", hr.path); snprintf(rq.query, sizeof rq.query, "%s", hr.query);
        rq.body = hr.body; rq.body_len = hr.body_len;
        buf b = {0}; const char *ctype = "application/json; charset=utf-8";
        const int code = tr_api(&rq, &b, &ctype, 1, s->port);
        http_response_n(fd, s->enable_cors, code, ctype, b.ptr ? b.ptr : "", b.len);
        ds4_buf_free(&b);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && (!strcmp(hr.path, "/monitor") || !strcmp(hr.path, "/monitor.html"))) {
        serve_page_file(fd, s->enable_cors, DS4_MONITOR_PAGE_FILE);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/metrics")) {
        /* 监控数据面(server_monitor.c): 默认 JSON(监控页每秒拉一次; ?requests=all 给全部保留的请求), Prometheus 问法给文本 */
        buf b = {0};
        if (hr.accept_text || strstr(hr.query, "format=prometheus")) {
            mon_prometheus_text(s, &b);
            http_response(fd, s->enable_cors, 200, "text/plain; version=0.0.4; charset=utf-8", b.ptr ? b.ptr : "");
        } else {
            mon_metrics_json(s, &b, strstr(hr.query, "requests=all") != NULL);
            http_response(fd, s->enable_cors, 200, "application/json", b.ptr ? b.ptr : "{}");
        }
        ds4_buf_free(&b);
        http_request_free(&hr);
        goto done;
    }

    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        send_models(s, fd);
        http_request_free(&hr);
        goto done;
    }
    const char *model_path_prefix = "/v1/models/";
    const size_t model_path_prefix_len = strlen(model_path_prefix);
    if (!strcmp(hr.method, "GET") &&
        !strncmp(hr.path, model_path_prefix, model_path_prefix_len) &&
        server_model_alias_known(hr.path + model_path_prefix_len))
    {
        send_model(s, fd, hr.path + model_path_prefix_len);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok = false;
    const int ctx_size = server_ctx_size(s);
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") &&
               !strcmp(hr.path, "/v1/messages/count_tokens")) {
        /* Anthropic token counting: parse + render + tokenize exactly like a
         * real /v1/messages request (same chat template, same replay attach),
         * answer with the true prompt token count, run no inference.  Clients
         * use this for context budgeting; real tokenizer numbers beat any
         * client-side estimate.  Parse errors fall through to the shared 400. */
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
        if (ok) {
            buf b = {0};
            ds4_buf_printf(&b, "{\"input_tokens\":%d}\n", req.prompt.len);
            http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
            ds4_buf_free(&b);
            request_free(&req);
            http_request_free(&hr);
            goto done;
        }
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, hr.body, s->default_tokens,
                                      ctx_size, &req, err, sizeof(err));
    } else {
        http_error(fd, s->enable_cors, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    char route_path[256];
    snprintf(route_path, sizeof route_path, "%s", hr.path);
    http_request_free(&hr);
    if (!ok) {
        http_error(fd, s->enable_cors, 400, err);
        goto done;
    }
    if (s->force_nothink) req.think_mode = DS4_THINK_NONE;
    if (!req.model_from_request) {
        free(req.model);
        req.model = xstrdup(server_model_id_from_engine(s->engine));
    }
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, s->enable_cors, &req, req.prompt.len, ctx_size);
        request_free(&req);
        goto done;
    }

    set_client_socket_nonblocking(fd);
    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.req = req;
    j.mon = mon_begin(s, &j.req, route_path);   /* 监控: 从这一刻起算排队 */
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    pthread_mutex_lock(&j.mu);
    if (!enqueue(s, &j)) {
        pthread_mutex_unlock(&j.mu);
        http_error(fd, s->enable_cors, 503, "server shutting down");
        mon_end(s, j.mon, "error", 0, -1, -1);
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        request_free(&j.req);
        goto done;
    }
    while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
    pthread_mutex_unlock(&j.mu);
    /* worker 没走到收尾的失败路(请求不合法 / 预填准入拒绝)在这里补记 error; 正常路 worker 已收, 这次是空操作 */
    mon_end(s, j.mon, "error", 0, -1, -1);

    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    close(fd);
    client_done(s);
    return NULL;
}

int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = DS4_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}
