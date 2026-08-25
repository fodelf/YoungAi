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

char *web_json_get_string(const char *json, const char *key);

char *web_url_encode(const char *s) {
    static const char hex[] = "0123456789ABCDEF";
    web_buf b = {0};
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            web_buf_append(&b, (const char *)&c, 1);
        } else {
            char e[3] = {'%', hex[c >> 4], hex[c & 15]};
            web_buf_append(&b, e, 3);
        }
    }
    return web_buf_take(&b);
}

void web_random_bytes(unsigned char *buf, size_t len) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t off = 0;
        while (off < len) {
            ssize_t n = read(fd, buf + off, len - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            off += (size_t)n;
        }
        close(fd);
        if (off == len) return;
    }
    uint64_t x = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
    for (size_t i = 0; i < len; i++) {
        x = x * 6364136223846793005ULL + 1;
        buf[i] = (unsigned char)(x >> 32);
    }
}

char *web_base64(const unsigned char *data, size_t len) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t outlen = ((len + 2) / 3) * 4;
    char *out = web_xmalloc(outlen + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out[j++] = tab[(v >> 18) & 63];
        out[j++] = tab[(v >> 12) & 63];
        out[j++] = (i + 1 < len) ? tab[(v >> 6) & 63] : '=';
        out[j++] = (i + 2 < len) ? tab[v & 63] : '=';
    }
    out[j] = '\0';
    return out;
}

char *web_json_quote(const char *s) {
    web_buf b = {0};
    web_buf_puts(&b, "\"");
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '\\': web_buf_puts(&b, "\\\\"); break;
        case '"': web_buf_puts(&b, "\\\""); break;
        case '\n': web_buf_puts(&b, "\\n"); break;
        case '\r': web_buf_puts(&b, "\\r"); break;
        case '\t': web_buf_puts(&b, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                web_buf_puts(&b, tmp);
            } else {
                web_buf_append(&b, (const char *)&c, 1);
            }
            break;
        }
    }
    web_buf_puts(&b, "\"");
    return web_buf_take(&b);
}

int web_ws_connect(const char *ws_url, cdp_ws *ws,
                          char *err, size_t err_len) {
    const char *p = ws_url;
    if (strncmp(p, "ws://", 5) != 0) {
        web_set_err(err, err_len, "unsupported websocket URL: %s", ws_url);
        return -1;
    }
    p += 5;
    const char *slash = strchr(p, '/');
    if (!slash) {
        web_set_err(err, err_len, "malformed websocket URL");
        return -1;
    }
    char hostport[256];
    size_t hp_len = (size_t)(slash - p);
    if (hp_len >= sizeof(hostport)) hp_len = sizeof(hostport) - 1;
    memcpy(hostport, p, hp_len);
    hostport[hp_len] = '\0';
    char *colon = strrchr(hostport, ':');
    int port = 80;
    if (colon) {
        *colon = '\0';
        port = atoi(colon + 1);
    }
    const char *host = hostport;
    int fd = web_tcp_connect(host, port, DS4_WEB_CONNECT_TIMEOUT_MS, err, err_len);
    if (fd < 0) return -1;

    unsigned char rnd[16];
    web_random_bytes(rnd, sizeof(rnd));
    char *key = web_base64(rnd, sizeof(rnd));
    web_buf req = {0};
    char line[512];
    snprintf(line, sizeof(line),
             "GET %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Key: %s\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n",
             slash, host, port, key);
    web_buf_puts(&req, line);
    free(key);
    if (web_write_all(fd, req.ptr, req.len) != 0) {
        web_set_err(err, err_len, "websocket handshake write failed");
        close(fd);
        free(req.ptr);
        return -1;
    }
    free(req.ptr);

    web_buf resp = {0};
    char tmp[1024];
    while (!strstr(resp.ptr ? resp.ptr : "", "\r\n\r\n")) {
        ssize_t n = web_read_some(fd, tmp, sizeof(tmp), DS4_WEB_CONNECT_TIMEOUT_MS);
        if (n <= 0) {
            web_set_err(err, err_len, "websocket handshake read failed");
            close(fd);
            free(resp.ptr);
            return -1;
        }
        web_buf_append(&resp, tmp, (size_t)n);
        if (resp.len > 8192) break;
    }
    bool ok = resp.ptr && strstr(resp.ptr, " 101 ") != NULL;
    free(resp.ptr);
    if (!ok) {
        web_set_err(err, err_len, "websocket handshake rejected");
        close(fd);
        return -1;
    }
    ws->fd = fd;
    ws->next_id = 1;
    return 0;
}

void web_ws_close(cdp_ws *ws) {
    if (ws && ws->fd >= 0) {
        close(ws->fd);
        ws->fd = -1;
    }
}

int web_read_exact(int fd, unsigned char *buf, size_t len, int timeout_ms) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = web_read_some(fd, (char *)buf + off, len - off, timeout_ms);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

int web_ws_send_text(cdp_ws *ws, const char *text,
                            char *err, size_t err_len) {
    size_t len = strlen(text);
    web_buf frame = {0};
    unsigned char hdr[14];
    size_t h = 0;
    hdr[h++] = 0x81;
    if (len < 126) {
        hdr[h++] = 0x80 | (unsigned char)len;
    } else if (len <= 0xffff) {
        hdr[h++] = 0x80 | 126;
        hdr[h++] = (unsigned char)(len >> 8);
        hdr[h++] = (unsigned char)len;
    } else {
        hdr[h++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) hdr[h++] = (unsigned char)((uint64_t)len >> (i * 8));
    }
    unsigned char mask[4];
    web_random_bytes(mask, sizeof(mask));
    for (int i = 0; i < 4; i++) hdr[h++] = mask[i];
    web_buf_append(&frame, (const char *)hdr, h);
    for (size_t i = 0; i < len; i++) {
        char c = text[i] ^ mask[i & 3];
        web_buf_append(&frame, &c, 1);
    }
    int rc = web_write_all(ws->fd, frame.ptr, frame.len);
    free(frame.ptr);
    if (rc != 0) {
        web_set_err(err, err_len, "websocket write failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

int web_ws_send_pong(cdp_ws *ws, const unsigned char *payload, size_t len) {
    if (len > 125) len = 125;
    unsigned char hdr[2 + 4 + 125];
    hdr[0] = 0x8a;
    hdr[1] = 0x80 | (unsigned char)len;
    unsigned char mask[4];
    web_random_bytes(mask, sizeof(mask));
    memcpy(hdr + 2, mask, 4);
    for (size_t i = 0; i < len; i++) hdr[6 + i] = payload[i] ^ mask[i & 3];
    return web_write_all(ws->fd, hdr, 6 + len);
}

char *web_ws_read_message(cdp_ws *ws, char *err, size_t err_len) {
    web_buf msg = {0};
    for (;;) {
        unsigned char h[2];
        if (web_read_exact(ws->fd, h, 2, DS4_WEB_CDP_TIMEOUT_MS) != 0) {
            web_set_err(err, err_len, "websocket read timeout");
            free(msg.ptr);
            return NULL;
        }
        bool fin = (h[0] & 0x80) != 0;
        int opcode = h[0] & 0x0f;
        bool masked = (h[1] & 0x80) != 0;
        uint64_t len = h[1] & 0x7f;
        if (len == 126) {
            unsigned char x[2];
            if (web_read_exact(ws->fd, x, 2, DS4_WEB_CDP_TIMEOUT_MS) != 0) goto fail;
            len = ((uint64_t)x[0] << 8) | x[1];
        } else if (len == 127) {
            unsigned char x[8];
            if (web_read_exact(ws->fd, x, 8, DS4_WEB_CDP_TIMEOUT_MS) != 0) goto fail;
            len = 0;
            for (int i = 0; i < 8; i++) len = (len << 8) | x[i];
        }
        unsigned char mask[4] = {0};
        if (masked && web_read_exact(ws->fd, mask, 4, DS4_WEB_CDP_TIMEOUT_MS) != 0)
            goto fail;
        if (len > DS4_WEB_MAX_RESULT_BYTES * 4ULL) {
            web_set_err(err, err_len, "websocket message too large");
            free(msg.ptr);
            return NULL;
        }
        unsigned char *payload = web_xmalloc((size_t)len + 1);
        if (len && web_read_exact(ws->fd, payload, (size_t)len,
                                  DS4_WEB_CDP_TIMEOUT_MS) != 0) {
            free(payload);
            goto fail;
        }
        for (uint64_t i = 0; masked && i < len; i++) payload[i] ^= mask[i & 3];
        payload[len] = '\0';
        if (opcode == 0x8) {
            free(payload);
            web_set_err(err, err_len, "websocket closed");
            free(msg.ptr);
            return NULL;
        } else if (opcode == 0x9) {
            web_ws_send_pong(ws, payload, (size_t)len);
            free(payload);
            continue;
        } else if (opcode == 0x1 || opcode == 0x0) {
            web_buf_append(&msg, (const char *)payload, (size_t)len);
            free(payload);
            if (fin) return web_buf_take(&msg);
        } else {
            free(payload);
        }
    }
fail:
    web_set_err(err, err_len, "websocket frame read failed");
    free(msg.ptr);
    return NULL;
}

bool web_json_id_matches(const char *json, int id) {
    const char *p = strstr(json, "\"id\"");
    if (!p) return false;
    p = strchr(p, ':');
    if (!p) return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p) == id;
}

char *web_cdp_call(cdp_ws *ws, const char *method, const char *params,
                          char *err, size_t err_len) {
    int id = ws->next_id++;
    web_buf req = {0};
    char head[256];
    snprintf(head, sizeof(head), "{\"id\":%d,\"method\":", id);
    web_buf_puts(&req, head);
    char *qmethod = web_json_quote(method);
    web_buf_puts(&req, qmethod);
    free(qmethod);
    if (params && params[0]) {
        web_buf_puts(&req, ",\"params\":");
        web_buf_puts(&req, params);
    }
    web_buf_puts(&req, "}");
    char *wire = web_buf_take(&req);
    if (web_ws_send_text(ws, wire, err, err_len) != 0) {
        free(wire);
        return NULL;
    }
    free(wire);
    for (;;) {
        char *msg = web_ws_read_message(ws, err, err_len);
        if (!msg) return NULL;
        if (web_json_id_matches(msg, id)) return msg;
        free(msg);
    }
}

void web_cdp_call_optional(cdp_ws *ws, const char *method, const char *params) {
    char err[160] = {0};
    char *resp = web_cdp_call(ws, method, params, err, sizeof(err));
    free(resp);
}

int web_hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int x;
        if (c >= '0' && c <= '9') x = c - '0';
        else if (c >= 'a' && c <= 'f') x = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') x = c - 'A' + 10;
        else return -1;
        v = (v << 4) | x;
    }
    return v;
}

void web_utf8_append(web_buf *b, unsigned code) {
    char out[4];
    if (code <= 0x7f) {
        out[0] = (char)code;
        web_buf_append(b, out, 1);
    } else if (code <= 0x7ff) {
        out[0] = (char)(0xc0 | (code >> 6));
        out[1] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 2);
    } else if (code <= 0xffff) {
        out[0] = (char)(0xe0 | (code >> 12));
        out[1] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[2] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 3);
    } else {
        out[0] = (char)(0xf0 | (code >> 18));
        out[1] = (char)(0x80 | ((code >> 12) & 0x3f));
        out[2] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[3] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 4);
    }
}

char *web_json_parse_string_at(const char *q, const char **endp) {
    if (*q != '"') return NULL;
    q++;
    web_buf b = {0};
    while (*q && *q != '"') {
        if (*q != '\\') {
            web_buf_append(&b, q++, 1);
            continue;
        }
        q++;
        switch (*q) {
        case '"': web_buf_append(&b, "\"", 1); q++; break;
        case '\\': web_buf_append(&b, "\\", 1); q++; break;
        case '/': web_buf_append(&b, "/", 1); q++; break;
        case 'b': web_buf_append(&b, "\b", 1); q++; break;
        case 'f': web_buf_append(&b, "\f", 1); q++; break;
        case 'n': web_buf_append(&b, "\n", 1); q++; break;
        case 'r': web_buf_append(&b, "\r", 1); q++; break;
        case 't': web_buf_append(&b, "\t", 1); q++; break;
        case 'u': {
            int v = web_hex4(q + 1);
            if (v < 0) { free(b.ptr); return NULL; }
            q += 5;
            if (v >= 0xd800 && v <= 0xdbff && q[0] == '\\' && q[1] == 'u') {
                int lo = web_hex4(q + 2);
                if (lo >= 0xdc00 && lo <= 0xdfff) {
                    unsigned code = 0x10000 + (((unsigned)v - 0xd800) << 10) +
                                    ((unsigned)lo - 0xdc00);
                    web_utf8_append(&b, code);
                    q += 6;
                    break;
                }
            }
            web_utf8_append(&b, (unsigned)v);
            break;
        }
        default:
            if (*q) web_buf_append(&b, q++, 1);
            break;
        }
    }
    if (*q != '"') {
        free(b.ptr);
        return NULL;
    }
    if (endp) *endp = q + 1;
    return web_buf_take(&b);
}

char *web_json_get_string(const char *json, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        p += strlen(pat);
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p++ != ':') continue;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '"') return web_json_parse_string_at(p, NULL);
    }
    return NULL;
}

