/* train_http.c — ds4-train(主进程)的 HTTP 层: 自己答的请求单线程阻塞、一次一条(本机工具, 不求并发); 归模型子进程的请求交 train_proxy.c
 * 开线程转发(一条回答几分钟, 不能堵住页面的轮询)。总述见 train_internal.h。
 * 只认 HTTP/1.1 的 请求行 + 头 + Content-Length 体; 不做 keep-alive(每条 Connection: close), 浏览器照样好用。
 * 出错会怎样: 请求头超 64 KB 或体超 1 MB 直接断开(没有任何正当请求会这么大)。 */
#include "train_internal.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#define TR_HDR_MAX (64u * 1024u)
#define TR_BODY_MAX TR_UPLOAD_MAX   /* 上传料走原始体 */

static bool send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, 0);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += (size_t)w; n -= (size_t)w;
    }
    return true;
}

static const char *status_text(int code) {
    switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    default: return "Internal Server Error";
    }
}

static void respond(int fd, int code, const char *ctype, const ds4_buf *body) {
    ds4_buf h = {0};
    ds4_buf_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                   code, status_text(code), ctype, body->len);
    if (send_all(fd, h.ptr, h.len) && body->len) send_all(fd, body->ptr, body->len);
    ds4_buf_free(&h);
}

/* 读到头结束(\r\n\r\n), 再按 Content-Length 读体; 返回 false = 请求不合法/太大/对端断开。*raw = 整条请求原字节(转发用), *raw_len 其长度 */
static bool read_request(int fd, tr_req *rq, char **raw, size_t *raw_len) {
    ds4_buf in = {0};
    char tmp[4096];
    size_t hend = 0;
    for (;;) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n < 0) { if (errno == EINTR) continue; ds4_buf_free(&in); return false; }
        if (n == 0) break;
        ds4_buf_append(&in, tmp, (size_t)n);
        const char *e = strstr(in.ptr, "\r\n\r\n");
        if (e) { hend = (size_t)(e - in.ptr) + 4; break; }
        if (in.len > TR_HDR_MAX) { ds4_buf_free(&in); return false; }
    }
    if (!hend) { ds4_buf_free(&in); return false; }
    size_t clen = 0;
    for (const char *p = in.ptr; p < in.ptr + hend; ) {
        const char *nl = strstr(p, "\r\n");
        if (!nl) break;
        if (!strncasecmp(p, "Content-Length:", 15)) clen = (size_t)strtoul(p + 15, NULL, 10);
        p = nl + 2;
    }
    if (clen > TR_BODY_MAX) { ds4_buf_free(&in); return false; }
    while (in.len < hend + clen) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n < 0) { if (errno == EINTR) continue; ds4_buf_free(&in); return false; }
        if (n == 0) break;
        ds4_buf_append(&in, tmp, (size_t)n);
    }
    if (in.len < hend + clen) { ds4_buf_free(&in); return false; }
    memset(rq, 0, sizeof *rq);
    /* 请求行: METHOD SP target SP version */
    char *line_end = strstr(in.ptr, "\r\n");
    *line_end = 0;
    char target[1024] = "";
    if (sscanf(in.ptr, "%7s %1023s", rq->method, target) != 2) { ds4_buf_free(&in); return false; }
    char *q = strchr(target, '?');
    if (q) { *q = 0; snprintf(rq->query, sizeof rq->query, "%s", q + 1); }
    snprintf(rq->path, sizeof rq->path, "%s", target);
    in.ptr[hend + clen] = 0;   /* 缓冲总有 +1 的 NUL 位 */
    rq->body = in.ptr + hend; rq->body_len = clen;
    *line_end = '\r';     /* 请求行的 \r 刚才被 sscanf 用 NUL 顶掉了, 转发要原样 */
    *raw = in.ptr; *raw_len = hend + clen;   /* 体指着它, 处理完由调用方 free */
    return true;
}

int tr_http_serve(const char *host, int port, tr_handler h) {
    signal(SIGPIPE, SIG_IGN);
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) { fprintf(stderr, "ds4-train: 监听地址不合法 %s\n", host); return 1; }
    if (bind(ls, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); return 1; }
    if (listen(ls, 16) < 0) { perror("listen"); return 1; }
    fprintf(stderr, "ds4-train: http://%s:%d/  (根 %s)\n", host, port, tr_root);
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; perror("accept"); continue; }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        tr_req rq; char *raw = NULL; size_t raw_len = 0;
        ds4_buf out = {0};
        if (read_request(fd, &rq, &raw, &raw_len)) {
            /* 归模型子进程的路, 子进程装着就转过去(线程接管 fd 与 raw); 没装着就落到自己的路由(/api/models/… 会答"没有装着的模型", /v1 404) */
            if (tr_proxy_path(rq.path) && tr_model_live()) { tr_proxy_start(fd, raw, raw_len, TR_MODEL_PORT(port)); continue; }
            const char *ctype = "application/json; charset=utf-8";
            const int code = h(&rq, &out, &ctype);
            respond(fd, code, ctype, &out);
        }
        ds4_buf_free(&out);
        free(raw);
        close(fd);
    }
}
