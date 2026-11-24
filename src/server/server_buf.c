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

/* ★单 worker 内存准入(2026-11-12)★: 与 --batch 调度器同一规则 —— 开一条请求的预填要 MemAvailable ≥ 2 × 预填峰值字节。
 * 单 worker 以前没有这条: 588715/406879 token 的提示直接把请求态与持久检查点推进 2500 MB 看门狗线。
 * 不够就回 503 + 标准 error, 不进 ds4_engine_v41_generate_argmax 分配状态 —— 拒绝请求好过把服务杀掉。 */
long server_mem_available_mb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;   /* 读不到 = 不拦(Mac 编译路没有 /proc) */
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}
bool server_v41_mem_admit(server *s, job *j, char *err, size_t errn) {
    const long avail = server_mem_available_mb();
    const uint64_t peak_mb = (ds4_v41_req_prefill_bytes(j->req.prompt.len) + ds4_v41_images_bytes(j->req.images)) >> 20;
    if (avail >= 0 && (uint64_t)avail < 2u * peak_mb) {
        server_mem_diag();   /* ★先落归属读数★: 拒绝的原因不能只剩一个公式, 必须指出哪个进程占着内存 */
        snprintf(err, errn, "server memory insufficient: prefill peak %llu MB, MemAvailable %ld MB (admission = 2x peak)",
                 (unsigned long long)peak_mb, avail);
        return false;
    }
    (void)s;   /* 准入是纯内存判据, 不依赖 server 状态 */
    return true;
}

/* ★内存准入失败时的归属诊断(2026-11-24)★: 只看公式看不出谁占着内存。
 * 读什么、为什么: 主机 /proc/meminfo 全字段(含 cached/slab/swap) + 主机进程 VmHWM/VmRSS 前 8 名。
 * VmHWM 是峰值驻留, 不是现 RSS —— ds4-server 启动时模型映射能到 94+ GiB, 之后 RSS 只剩几百 MB,
 * 只看 RSS 会把最大的占用者报成小角色。GB10 统一内存里 GPU 占用就是整机内存的真值, 且不在 ps RSS,
 * 所以再读 NVML compute-app usedGpuMemory; 读不到(Mac/无驱动)就明说, 不猜。
 * 只在准入拒绝那一次调用, 不进每秒监控路。 */
typedef struct { unsigned pid; double hwm_mb, rss_mb; char name[32]; } mem_diag_proc;
typedef struct {
    double total, avail, free, cached, buffers, active_anon, inactive_anon, active_file, inactive_file,
           slab_reclaim, slab_unreclaim, swap_total, swap_free;
} mem_diag_info;

static bool mem_diag_kb(const char *line, const char *key, double *out) {
    const size_t k = strlen(key);
    if (strncmp(line, key, k) != 0) return false;
    double kb = 0;
    if (sscanf(line + k, " %lf kB", &kb) != 1 || kb < 0) return false;
    *out = kb / 1024.0;
    return true;
}

static void mem_diag_name(const char *line, char *name, size_t cap) {
    const char *c = strchr(line, ':');
    if (!c) return;
    c++; while (*c == ' ' || *c == '\t') c++;
    snprintf(name, cap, "%s", c);
    name[strcspn(name, "\r\n")] = 0;
}

static void mem_diag_keep_top(mem_diag_proc *top, int *n, int cap, const mem_diag_proc *c) {
    int pos;
    if (*n < cap) { pos = (*n)++; }
    else {
        if (c->hwm_mb <= top[cap - 1].hwm_mb) return;
        pos = cap - 1; top[pos] = *c;   /* 已按 hwm 降序, 末位就是最小 */
    }
    while (pos > 0 && top[pos - 1].hwm_mb < top[pos].hwm_mb) {
        mem_diag_proc t = top[pos - 1]; top[pos - 1] = top[pos]; top[pos] = t; pos--;
    }
}

static void mem_diag_scan_procs(mem_diag_proc *top, int *n, int cap) {
    DIR *d = opendir("/proc");
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d))) {
        const char *s = de->d_name; unsigned pid = 0; bool numeric = true;
        for (; *s; s++) {
            if (*s < '0' || *s > '9') { numeric = false; break; }
            pid = pid * 10u + (unsigned)(*s - '0');
        }
        if (!numeric || !pid || pid > 0x7fffffff) continue;
        char path[64]; snprintf(path, sizeof path, "/proc/%u/status", pid);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        mem_diag_proc c = {pid, 0, 0, ""}; char line[256];
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "Name:", 5)) mem_diag_name(line, c.name, sizeof c.name);
            else if (!strncmp(line, "VmHWM:", 6)) (void)mem_diag_kb(line, "VmHWM:", &c.hwm_mb);
            else if (!strncmp(line, "VmRSS:", 6)) (void)mem_diag_kb(line, "VmRSS:", &c.rss_mb);
        }
        fclose(f);
        if (c.hwm_mb > 0) mem_diag_keep_top(top, n, cap, &c);
    }
    closedir(d);
}

void server_mem_diag(void) {
    mem_diag_info mi = {0}; bool have_meminfo = false;
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            (void)mem_diag_kb(line, "MemTotal:", &mi.total);
            (void)mem_diag_kb(line, "MemAvailable:", &mi.avail);
            (void)mem_diag_kb(line, "MemFree:", &mi.free);
            (void)mem_diag_kb(line, "Buffers:", &mi.buffers);
            (void)mem_diag_kb(line, "Cached:", &mi.cached);
            (void)mem_diag_kb(line, "Active(anon):", &mi.active_anon);
            (void)mem_diag_kb(line, "Inactive(anon):", &mi.inactive_anon);
            (void)mem_diag_kb(line, "Active(file):", &mi.active_file);
            (void)mem_diag_kb(line, "Inactive(file):", &mi.inactive_file);
            (void)mem_diag_kb(line, "SReclaimable:", &mi.slab_reclaim);
            (void)mem_diag_kb(line, "SUnreclaim:", &mi.slab_unreclaim);
            (void)mem_diag_kb(line, "SwapTotal:", &mi.swap_total);
            (void)mem_diag_kb(line, "SwapFree:", &mi.swap_free);
        }
        fclose(f); have_meminfo = true;
    }
    if (have_meminfo) server_log(DS4_LOG_DEFAULT,
        "ds4-server: mem attribution: total=%.0f MB avail=%.0f MB free=%.0f MB buffers=%.0f MB cached=%.0f MB"
        " active(anon)=%.0f inactive(anon)=%.0f active(file)=%.0f inactive(file)=%.0f"
        " slab=%.0f+%.0f MB swap=%.0f MB used of %.0f MB",
        mi.total, mi.avail, mi.free, mi.buffers, mi.cached, mi.active_anon, mi.inactive_anon, mi.active_file,
        mi.inactive_file, mi.slab_reclaim, mi.slab_unreclaim, mi.swap_total - mi.swap_free, mi.swap_total);
    else server_log(DS4_LOG_DEFAULT, "ds4-server: mem attribution: /proc/meminfo unreadable; host attribution unavailable");
    mem_diag_proc top[8]; int ntop = 0;
    mem_diag_scan_procs(top, &ntop, 8);
    for (int i = 0; i < ntop; i++) server_log(DS4_LOG_DEFAULT,
        "ds4-server: mem occupant host: pid %u %s VmHWM=%.0f MB VmRSS=%.0f MB",
        top[i].pid, top[i].name[0] ? top[i].name : "?", top[i].hwm_mb, top[i].rss_mb);
    unsigned gp[8]; unsigned long long gb[8]; const int ng = mon_hw_compute_apps(gp, gb, 8);
    if (ng <= 0) server_log(DS4_LOG_DEFAULT, "ds4-server: mem occupant gpu: NVML compute-app memory unavailable");
    else for (int i = 0; i < ng; i++) server_log(DS4_LOG_DEFAULT,
        "ds4-server: mem occupant gpu: pid %u usedGpuMemory=%.0f MiB", gp[i], (double)gb[i] / (1024.0 * 1024.0));
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

bool json_content(const char **p, char **out) { return json_content_sink(p, out, NULL); }

/* OpenAI 消息 content: 字串 / null / 分块数组。分块 {"type":"text"} 取文本; {"type":"image_url"} 进 sink(2026-10-13 之前这里把图片块
 * 悄悄跳过 —— 用户拍照上传"识别不了"的直接原因: 图片根本没进模型); sk == NULL 时图片块 = 解析失败(400), 不再静默丢。 */
bool json_content_sink(const char **p, char **out, img_sink *sk) {
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
            char *type = NULL, *url = NULL;
            bool ok = true;
            while (ok && **p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) { ok = false; break; }
                json_ws(p);
                if (**p != ':') { free(key); ok = false; break; }
                (*p)++;
                if (!strcmp(key, "text")) {
                    char *s = NULL;
                    ok = json_string(p, &s);
                    if (ok) ds4_buf_puts(&b, s);
                    free(s);
                } else if (!strcmp(key, "type")) {
                    free(type); type = NULL;
                    ok = json_string(p, &type);
                } else if (!strcmp(key, "image_url")) {   /* "image_url": "data:…" 或 {"url": "data:…", "detail": …} */
                    json_ws(p);
                    free(url); url = NULL;
                    if (**p == '"') ok = json_string(p, &url);
                    else if (**p == '{') {
                        (*p)++; json_ws(p);
                        while (ok && **p && **p != '}') {
                            char *k2 = NULL;
                            ok = json_string(p, &k2);
                            if (ok) { json_ws(p); ok = **p == ':'; if (ok) (*p)++; }
                            if (ok) { if (!strcmp(k2, "url")) { free(url); url = NULL; ok = json_string(p, &url); } else ok = json_skip_value(p); }
                            free(k2);
                            if (ok) { json_ws(p); if (**p == ',') (*p)++; json_ws(p); }
                        }
                        if (ok) ok = **p == '}';
                        if (ok) (*p)++;
                    } else ok = json_skip_value(p);
                } else ok = json_skip_value(p);
                free(key);
                if (ok) { json_ws(p); if (**p == ',') (*p)++; json_ws(p); }
            }
            if (ok) ok = **p == '}';
            if (ok) (*p)++;
            if (ok && type && !strcmp(type, "image_url")) ok = url && mm_image_data_url(sk, url, &b);
            free(type); free(url);
            if (!ok) goto fail;
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
