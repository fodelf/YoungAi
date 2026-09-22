/* server_httpd.c — 机械拆分自 ds4_server.c (12449-12786 行): HTTP 读写/模型列表/客户端线程。 */

#include "server_internal.h"

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
        buf_append(&b, tmp, (size_t)n);
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
    if (q) *q = '\0';

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_printf(b,
        "{\"id\":");
    json_escape(b, id);
    buf_puts(b,
        ",\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"ds4.c\","
        "\"name\":");
    json_escape(b, name);
    buf_printf(b,
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
    buf_putc(&b, '\n');
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool send_models(server *s, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    append_model_json(&b, s, "deepseek-v4-flash");
    buf_putc(&b, ',');
    append_model_json(&b, s, "deepseek-v4-pro");
    buf_puts(&b, "]}\n");
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
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

    if (!strcmp(hr.method, "GET") &&
        (path_route_is(hr.path, "/") || path_route_is(hr.path, "/chat") ||
         path_route_is(hr.path, "/index.html"))) {
        serve_chat_page(fd, s->enable_cors, DS4_CHAT_PAGE_FILE);
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
            buf_printf(&b, "{\"input_tokens\":%d}\n", req.prompt.len);
            http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
            buf_free(&b);
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
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    pthread_mutex_lock(&j.mu);
    if (!enqueue(s, &j)) {
        pthread_mutex_unlock(&j.mu);
        http_error(fd, s->enable_cors, 503, "server shutting down");
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        request_free(&j.req);
        goto done;
    }
    while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
    pthread_mutex_unlock(&j.mu);

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
