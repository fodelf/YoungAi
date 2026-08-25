#include "ds4_web.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DS4_WEB_DEFAULT_PORT 9333
#include "web_internal.h"

void *web_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4_web: malloc");
        exit(1);
    }
    return p;
}

char *web_xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = web_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

void web_buf_append(web_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        char *p = realloc(b->ptr, cap);
        if (!p) {
            perror("ds4_web: realloc");
            exit(1);
        }
        b->ptr = p;
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

void web_buf_puts(web_buf *b, const char *s) {
    web_buf_append(b, s, strlen(s));
}

char *web_buf_take(web_buf *b) {
    if (!b->ptr) return web_xstrdup("");
    char *p = b->ptr;
    b->ptr = NULL;
    b->len = b->cap = 0;
    return p;
}

void web_set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

void web_log(ds4_web *web, const char *msg) {
    if (web && web->log) web->log(web->log_privdata, msg);
}

bool web_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) return false;
        *p = '/';
    }
    return mkdir(tmp, 0700) == 0 || errno == EEXIST;
}

int web_tcp_connect(const char *host, int port, int timeout_ms,
                           char *err, size_t err_len) {
    char service[32];
    snprintf(service, sizeof(service), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, service, &hints, &res);
    if (gai != 0) {
        web_set_err(err, err_len, "getaddrinfo %s: %s", host, gai_strerror(gai));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc == 0) {
            if (flags >= 0) fcntl(fd, F_SETFL, flags);
            break;
        }
        if (errno == EINPROGRESS) {
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            rc = poll(&pfd, 1, timeout_ms);
            if (rc > 0) {
                int soerr = 0;
                socklen_t slen = sizeof(soerr);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen);
                if (soerr == 0) {
                    if (flags >= 0) fcntl(fd, F_SETFL, flags);
                    break;
                }
                errno = soerr;
            }
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) web_set_err(err, err_len, "connect %s:%d failed: %s",
                            host, port, strerror(errno));
    return fd;
}

int web_write_all(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = write(fd, p, len);
#endif
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

ssize_t web_read_some(int fd, char *buf, size_t len, int timeout_ms) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int rc = poll(&pfd, 1, timeout_ms);
    if (rc <= 0) return rc == 0 ? 0 : -1;
    for (;;) {
        ssize_t n = read(fd, buf, len);
        if (n < 0 && errno == EINTR) continue;
        return n;
    }
}

char *web_http_request(const char *method, int port, const char *path,
                              char *err, size_t err_len) {
    int fd = web_tcp_connect("127.0.0.1", port, DS4_WEB_CONNECT_TIMEOUT_MS,
                             err, err_len);
    if (fd < 0) return NULL;
    web_buf req = {0};
    char line[512];
    snprintf(line, sizeof(line),
             "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n",
             method, path, port);
    web_buf_puts(&req, line);
    if (web_write_all(fd, req.ptr, req.len) != 0) {
        web_set_err(err, err_len, "write HTTP request failed: %s", strerror(errno));
        close(fd);
        free(req.ptr);
        return NULL;
    }
    free(req.ptr);

    web_buf resp = {0};
    char tmp[4096];
    for (;;) {
        ssize_t n = web_read_some(fd, tmp, sizeof(tmp), DS4_WEB_CONNECT_TIMEOUT_MS);
        if (n < 0) {
            web_set_err(err, err_len, "read HTTP response failed: %s", strerror(errno));
            close(fd);
            free(resp.ptr);
            return NULL;
        }
        if (n == 0) break;
        web_buf_append(&resp, tmp, (size_t)n);
    }
    close(fd);
    if (!resp.ptr) {
        web_set_err(err, err_len, "empty HTTP response");
        return NULL;
    }
    char *body = strstr(resp.ptr, "\r\n\r\n");
    if (!body) {
        web_set_err(err, err_len, "malformed HTTP response");
        free(resp.ptr);
        return NULL;
    }
    body += 4;
    char *out = web_xstrdup(body);
    free(resp.ptr);
    return out;
}

bool web_cdp_alive(ds4_web *web) {
    char err[160] = {0};
    char *body = web_http_request("GET", web->port, "/json/version", err, sizeof(err));
    if (!body) return false;
    bool ok = strstr(body, "webSocketDebuggerUrl") != NULL;
    free(body);
    return ok;
}

