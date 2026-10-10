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

/* Python json.dumps(ensure_ascii=False) 的字节形状: 分隔符 ", " 与 ": ", 字符串里非 ASCII 原样、只转义
 * 引号/反斜杠/控制字符。带空格 DSML 那一代的官方 encoding.py 用它序列化工具 schema 和非字符串参数, 模型只见过
 * 这种形状; 客户端的紧凑 JSON 原样塞进提示词 = 每条 schema 都是没训练过的写法。数字照客户端原样
 * (Python 会把 1e5 改写成 100000.0, 这种少见形状不追)。 */
static void pyjson_string(buf *b, const char *s) {
    ds4_buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { ds4_buf_putc(b, '\\'); ds4_buf_putc(b, (char)c); }
        else if (c == '\n') ds4_buf_puts(b, "\\n");
        else if (c == '\r') ds4_buf_puts(b, "\\r");
        else if (c == '\t') ds4_buf_puts(b, "\\t");
        else if (c == '\b') ds4_buf_puts(b, "\\b");
        else if (c == '\f') ds4_buf_puts(b, "\\f");
        else if (c < 0x20) ds4_buf_printf(b, "\\u%04x", c);
        else ds4_buf_putc(b, (char)c);
    }
    ds4_buf_putc(b, '"');
}

static bool pyjson_value(const char **p, buf *b) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        pyjson_string(b, s);
        free(s);
        return true;
    }
    if (**p != '{' && **p != '[') {
        const char *start = *p;
        if (!json_skip_value(p)) return false;
        ds4_buf_append(b, start, (size_t)(*p - start));
        return true;
    }
    const bool obj = **p == '{';
    const char close = obj ? '}' : ']';
    ds4_buf_putc(b, **p);
    (*p)++;
    json_ws(p);
    for (bool first = true; **p && **p != close; first = false) {
        if (!first) ds4_buf_puts(b, ", ");
        if (obj) {
            char *k = NULL;
            if (!json_string(p, &k)) return false;
            pyjson_string(b, k);
            free(k);
            json_ws(p);
            if (**p != ':') return false;
            (*p)++;
            ds4_buf_puts(b, ": ");
        }
        if (!pyjson_value(p, b)) return false;
        json_ws(p);
        if (**p == ',') { (*p)++; json_ws(p); }
        else if (**p != close) return false;
    }
    if (**p != close) return false;
    ds4_buf_putc(b, close);
    (*p)++;
    return true;
}

/* 解析失败(客户端发了坏 JSON)退回压缩形状: 至少与旧行为一致, 不吞掉内容 */
char *json_pyfmt_raw_value(const char *json) {
    const char *p = json ? json : "null";
    buf b = {0};
    if (!pyjson_value(&p, &b)) {
        ds4_buf_free(&b);
        return json_minify_raw_value(json);
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
