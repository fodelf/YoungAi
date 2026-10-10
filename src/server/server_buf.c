/* server_buf.c — 机械拆分自 ds4_server.c (42-478 行): 内存 + 服务端专用的 JSON 工具。
 * buf 与 JSON 词法基元(json_string/json_skip_value 等)2026-10-10 挪到 src/common/ds4_json.c(训练器读 jsonl 也用), 这里只剩服务端独有的两件。 */

#include "server_internal.h"

volatile sig_atomic_t g_stop_requested = 0;

/* --nothink: force non-thinking mode BEFORE prompt rendering (parsers consult
 * this when deciding think_mode; flipping the field after parse is too late
 * because the think template is already rendered into the prompt). */
bool g_force_nothink = false;

volatile sig_atomic_t g_listen_fd = -1;

void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

void die(const char *msg) {
    fprintf(stderr, "ds4-server: %s\n", msg);
    exit(1);
}

char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

bool random_bytes(void *dst, size_t len) {
    unsigned char *p = dst;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return false;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            close(fd);
            return false;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    close(fd);
    return true;
}

char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return xstrdup(json ? json : "null");
    const char *end = p;

    buf b = {0};
    bool in_string = false;
    bool escape = false;
    for (const char *s = start; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (in_string) {
            ds4_buf_putc(&b, (char)c);
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
            ds4_buf_putc(&b, (char)c);
        } else if (!isspace(c)) {
            ds4_buf_putc(&b, (char)c);
        }
    }
    return ds4_buf_take(&b);
}

bool json_content(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }

    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            ds4_buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) goto fail;
                json_ws(p);
                if (**p != ':') {
                    free(key);
                    goto fail;
                }
                (*p)++;
                if (!strcmp(key, "text")) {
                    char *s = NULL;
                    if (!json_string(p, &s)) {
                        free(key);
                        goto fail;
                    }
                    ds4_buf_puts(&b, s);
                    free(s);
                } else if (!json_skip_value(p)) {
                    free(key);
                    goto fail;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') goto fail;
            (*p)++;
        } else if (!json_skip_value(p)) {
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = ds4_buf_take(&b);
    return true;
fail:
    ds4_buf_free(&b);
    return false;
}
