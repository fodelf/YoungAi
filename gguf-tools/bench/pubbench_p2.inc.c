/* ============================ HTTP(libcurl) ============================ */

static double now_s(void);

/* urllib 的 timeout 是 **socket 空闲超时**(connect 与每次 recv 各算一次), 不是总时长。
 * curl 的 LOW_SPEED_TIME 在"还没收到第一个字节"的等待期不触发(实测: 服务端 sleep 8s、
 * timeout=2 时 .py 判 timed out 而 curl 一直等) —— 所以这里用进度回调自己算空闲计时:
 * 每收到一段数据就刷新 last, 空闲超过 timeout 即中止(CURLE_ABORTED_BY_CALLBACK)。 */
typedef struct { char *p; size_t n, cap; double last, idle_max; } buf_t;
static size_t curl_sink(void *ptr, size_t sz, size_t nm, void *ud) {
    buf_t *b = ud; size_t add = sz * nm;
    if (b->n + add + 1 > b->cap) { size_t c = b->cap ? b->cap : 4096; while (c < b->n + add + 1) c *= 2; b->p = xrealloc(b->p, c); b->cap = c; }
    memcpy(b->p + b->n, ptr, add); b->n += add; b->p[b->n] = 0;
    b->last = now_s();
    return add;
}
static int curl_idle_abort(void *ud, curl_off_t dt, curl_off_t dn, curl_off_t ut, curl_off_t un) {
    buf_t *b = ud;
    (void)dt; (void)dn; (void)ut; (void)un;
    return (now_s() - b->last > b->idle_max) ? 1 : 0;
}
/* 状态行的 reason phrase(HTTPError 的消息里带) */
typedef struct { char reason[128]; buf_t *b; } hdr_ctx;
static size_t curl_hdr(void *ptr, size_t sz, size_t nm, void *ud) {
    hdr_ctx *h = ud; size_t n = sz * nm;
    const char *s = ptr;
    if (h->b) h->b->last = now_s();   /* 收到响应头也算"有数据": http.client 读完头才读体 */
    if (n > 9 && !strncmp(s, "HTTP/", 5)) {
        const char *sp1 = memchr(s, ' ', n);
        if (sp1) {
            const char *sp2 = memchr(sp1 + 1, ' ', n - (size_t)(sp1 + 1 - s));
            if (sp2) {
                size_t rl = n - (size_t)(sp2 + 1 - s);
                while (rl && (sp2[rl] == '\n' || sp2[rl] == '\r' || sp2[rl] == 0)) rl--;
                if (rl >= sizeof h->reason) rl = sizeof h->reason - 1;
                memcpy(h->reason, sp2 + 1, rl); h->reason[rl] = 0;
            }
        }
    }
    return n;
}

/* http_post_json: 复刻 _OPENER.open(Request(url, data, {"Content-Type": ...}), timeout)
 *   - _OPENER = build_opener(ProxyHandler({})) → 绕全局代理(07-14 502 教训)
 *   - urllib 的 timeout 是 socket 超时(每次 recv), 不是总时长 → 用 LOW_SPEED_TIME 近似
 *   - urllib 发 Connection: close, 不复用连接 → FORBID_REUSE 保持同一条线上行为
 * 失败时 errmsg 填 Python 侧 str(e) 的形状(见转录清单 B 组: 非逐字符保证)。 */
static int http_post_json(const char *url, const char *body, long timeout_s,
                          char **out, size_t *out_n, char *errmsg, size_t errcap) {
    CURL *h = curl_easy_init();
    if (!h) { snprintf(errmsg, errcap, "<urlopen error curl init failed>"); return -1; }
    buf_t b = {0};
    b.last = now_s(); b.idle_max = (double)timeout_s;
    hdr_ctx hc; hc.reason[0] = 0; hc.b = &b;
    struct curl_slist *hdr = NULL;
    hdr = curl_slist_append(hdr, "Content-Type: application/json");
    hdr = curl_slist_append(hdr, "Connection: close");
    curl_easy_setopt(h, CURLOPT_URL, url);
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_sink);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &b);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, curl_hdr);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &hc);
    curl_easy_setopt(h, CURLOPT_PROXY, "");
    curl_easy_setopt(h, CURLOPT_NOPROXY, "*");
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_FORBID_REUSE, 1L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, timeout_s);
    curl_easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(h, CURLOPT_XFERINFOFUNCTION, curl_idle_abort);
    curl_easy_setopt(h, CURLOPT_XFERINFODATA, &b);
    CURLcode rc = curl_easy_perform(h);
    long code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
    int ret = 0;
    if (rc != CURLE_OK) {
        long oserr = 0;
        curl_easy_getinfo(h, CURLINFO_OS_ERRNO, &oserr);
        if (rc == CURLE_OPERATION_TIMEDOUT || rc == CURLE_ABORTED_BY_CALLBACK) snprintf(errmsg, errcap, "timed out");
        else if (oserr) snprintf(errmsg, errcap, "<urlopen error [Errno %ld] %s>", oserr, strerror((int)oserr));
        else snprintf(errmsg, errcap, "<urlopen error %s>", curl_easy_strerror(rc));
        ret = -1;
    } else if (code >= 400) {
        snprintf(errmsg, errcap, "HTTP Error %ld: %s", code, hc.reason[0] ? hc.reason : "");
        ret = -1;
    }
    curl_slist_free_all(hdr);
    curl_easy_cleanup(h);
    if (ret == 0) { *out = b.p ? b.p : xstrdup(""); *out_n = b.n; }
    else free(b.p);
    return ret;
}

/* fetch 用: 裸 urlopen(走全局代理 env, 带 User-Agent), 120s */
static int http_get(const char *url, char **out, size_t *out_n, char *errmsg, size_t errcap) {
    CURL *h = curl_easy_init();
    if (!h) { snprintf(errmsg, errcap, "curl init failed"); return -1; }
    buf_t b = {0};
    b.last = now_s(); b.idle_max = 120.0;
    hdr_ctx hc; hc.reason[0] = 0; hc.b = &b;
    struct curl_slist *hdr = curl_slist_append(NULL, "User-Agent: pubbench/1.0");
    curl_easy_setopt(h, CURLOPT_URL, url);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_sink);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &b);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, curl_hdr);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &hc);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 120L);
    curl_easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(h, CURLOPT_XFERINFOFUNCTION, curl_idle_abort);
    curl_easy_setopt(h, CURLOPT_XFERINFODATA, &b);
    CURLcode rc = curl_easy_perform(h);
    long code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
    int ret = 0;
    if (rc != CURLE_OK) { snprintf(errmsg, errcap, "<urlopen error %s>", curl_easy_strerror(rc)); ret = -1; }
    else if (code >= 400) { snprintf(errmsg, errcap, "HTTP Error %ld: %s", code, hc.reason[0] ? hc.reason : ""); ret = -1; }
    curl_slist_free_all(hdr);
    curl_easy_cleanup(h);
    if (ret == 0) { *out = b.p ? b.p : xstrdup(""); *out_n = b.n; }
    else free(b.p);
    return ret;
}

/* gzip.GzipFile(...).read() */
static int gunzip(const char *in, size_t n, char **out, size_t *out_n) {
    z_stream zs; memset(&zs, 0, sizeof zs);
    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) return -1;
    size_t cap = n * 4 + 65536, len = 0;
    char *o = xmalloc(cap);
    zs.next_in = (Bytef *)in; zs.avail_in = (uInt)n;
    int r;
    do {
        if (len == cap) { cap *= 2; o = xrealloc(o, cap); }
        zs.next_out = (Bytef *)(o + len); zs.avail_out = (uInt)(cap - len);
        r = inflate(&zs, Z_NO_FLUSH);
        len = cap - zs.avail_out;
        if (r != Z_OK && r != Z_STREAM_END && r != Z_BUF_ERROR) { inflateEnd(&zs); free(o); return -1; }
    } while (r != Z_STREAM_END && zs.avail_in);
    inflateEnd(&zs);
    *out = o; *out_n = len;
    return 0;
}

/* ============================ 常量与数据集 ============================ */

static const char *HUMANEVAL_URLS[] = {
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    NULL
};
/* THUDM/humaneval-x 的 go 分片路径有过变动, 依次尝试 */
static const char *HUMANEVALX_GO_URLS[] = {
    "https://huggingface.co/datasets/THUDM/humaneval-x/resolve/main/data/go/data/humaneval.jsonl",
    "https://huggingface.co/datasets/THUDM/humaneval-x/resolve/main/go/data/humaneval.jsonl",
    NULL
};
static const char *PY_STOP[] = { "\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```", NULL };
static const char *GO_STOP[] = { "\nfunc main(", "\n// Test", "\npackage ", "\n```", NULL };
/* BOS = "<｜begin▁of▁sentence｜>" (U+FF5C / U+2581, 拆写防 \x 逃逸吃掉后随十六进制字符) */
static const char *BOS = "<\xef\xbd\x9c" "begin\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c" ">";
#define FULLMARK "\x00" "FULL" "\x00"   /* 6 字节, 含 NUL */
static const char FULL_MARK[6] = { 0, 'F', 'U', 'L', 'L', 0 };

/* 默认仍是 /tmp(旧行为不变)。--cache-dir 指向 gguf-tools/bench/data/
 * 可跑在联不上外网的机器上(Spark 直连 GitHub/HF 超时) —— 那两份 164 题 jsonl 已入库。*/
static const char *CACHE;

static void makedirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s", path);
    size_t n = strlen(tmp);
    for (size_t i = 1; i <= n; i++) {
        if (tmp[i] == '/' || i == n) {
            char c = tmp[i]; tmp[i] = 0;
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) { tmp[i] = c; if (i == n) die("OSError: [Errno %d] %s: '%s'", errno, strerror(errno), path); }
            tmp[i] = c;
        }
    }
}

static char *fetch(const char *const *urls, const char *name) {
    makedirs(CACHE);
    char *path = xmalloc(strlen(CACHE) + strlen(name) + 2);
    sprintf(path, "%s/%s", CACHE, name);
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) return path;
    char last[512]; snprintf(last, sizeof last, "None");
    for (int i = 0; urls[i]; i++) {
        logln("[fetch] %s", urls[i]);
        char *data; size_t dn; char err[512];
        if (http_get(urls[i], &data, &dn, err, sizeof err) != 0) {
            snprintf(last, sizeof last, "%s", err);
            logln("[fetch] fail: %s", err);
            continue;
        }
        size_t L = strlen(urls[i]);
        if (L > 3 && !strcmp(urls[i] + L - 3, ".gz")) {
            char *un; size_t un_n;
            if (gunzip(data, dn, &un, &un_n) != 0) {
                free(data);
                snprintf(last, sizeof last, "Not a gzipped file");
                logln("[fetch] fail: %s", last);
                continue;
            }
            free(data); data = un; dn = un_n;
        }
        FILE *f = fopen(path, "wb");
        if (!f) die("OSError: [Errno %d] %s: '%s'", errno, strerror(errno), path);
        fwrite(data, 1, dn, f); fclose(f); free(data);
        return path;
    }
    die("RuntimeError: dataset %s fetch failed: %s", name, last);
    return NULL;
}

typedef struct { jv **row; size_t n; } rows_t;

static rows_t load_jsonl(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("FileNotFoundError: [Errno %d] %s: '%s'", errno, strerror(errno), path);
    rows_t r = {0}; size_t cap = 0;
    char *line = NULL; size_t lcap = 0; ssize_t k;
    while ((k = getline(&line, &lcap, f)) != -1) {
        char *st = py_strip(line, (size_t)k);
        if (*st) {
            jv *v = json_loads(st, strlen(st));
            if (!v) die("json.decoder.JSONDecodeError: Expecting value: %s", path);
            if (r.n == cap) { cap = cap ? cap * 2 : 64; r.row = xrealloc(r.row, cap * sizeof *r.row); }
            r.row[r.n++] = v;
        }
        free(st);
    }
    free(line); fclose(f);
    return r;
}

/* 数据集缓存: .py 每次 rejudge 行都重新 load, 结果等价; C 侧 once 化(线程安全) */
static pthread_mutex_t g_ds_mu = PTHREAD_MUTEX_INITIALIZER;
static rows_t g_ds_py, g_ds_go;
static int g_have_py, g_have_go;
static rows_t dataset(int is_go) {
    pthread_mutex_lock(&g_ds_mu);
    if (is_go && !g_have_go) { g_ds_go = load_jsonl(fetch(HUMANEVALX_GO_URLS, "humaneval_x_go.jsonl")); g_have_go = 1; }
    if (!is_go && !g_have_py) { g_ds_py = load_jsonl(fetch(HUMANEVAL_URLS, "HumanEval.jsonl")); g_have_py = 1; }
    rows_t r = is_go ? g_ds_go : g_ds_py;
    pthread_mutex_unlock(&g_ds_mu);
    return r;
}

/* ============================ 正则等价件 ============================ */
/* .py 用了 6 处 re, 全部手写等价匹配器。逐条注明对应的 pattern, 语义(贪婪/惰性/回溯/
 * ^ 锚点/非重叠推进)照抄 —— 抽取器是判分口径的核心, 任何"简化"都会改分数。*/

typedef struct { char **v; size_t n, cap; } slist;
static void sl_push(slist *l, char *s) {
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 8; l->v = xrealloc(l->v, l->cap * sizeof *l->v); }
    l->v[l->n++] = s;
}
static void sl_free(slist *l) { for (size_t i = 0; i < l->n; i++) free(l->v[i]); free(l->v); l->v = NULL; l->n = l->cap = 0; }

static int fence_lang_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-';
}
/* re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)(?:```|\Z)", text, re.S)
 * 惰性体 = 到下一个 ``` 之前(没有则到串尾); 起点匹配失败则起点 +1 重试(````` 边界) */
static slist find_fences(const char *t) {
    slist out = {0};
    size_t n = strlen(t), pos = 0;
    while (pos < n) {
        const char *p = strstr(t + pos, "```");
        if (!p) break;
        size_t i = (size_t)(p - t), j = i + 3;
        while (j < n && fence_lang_char(t[j])) j++;
        if (j < n && t[j] == '\n') {
            size_t bs = j + 1;
            const char *c = strstr(t + bs, "```");
            size_t be = c ? (size_t)(c - t) : n;
            sl_push(&out, xstrndup(t + bs, be - bs));
            pos = c ? be + 3 : n;
        } else {
            pos = i + 1;
        }
    }
    return out;
}

/* re.findall(r"(?:def|func)\s+(\w+)\s*\(", s) / re.finditer(r"func\s+(\w+)\s*\(", s)
 * 注意 .py 没有词边界: "undef foo(" 也会命中 "def foo("。照抄。*/
static slist find_decl_names(const char *s, int allow_def) {
    slist out = {0};
    size_t n = strlen(s), i = 0;
    while (i < n) {
        size_t kl = 0;
        if (allow_def && i + 3 <= n && !strncmp(s + i, "def", 3)) kl = 3;
        else if (i + 4 <= n && !strncmp(s + i, "func", 4)) kl = 4;
        if (!kl) { i++; continue; }
        size_t j = i + kl, j0 = j;
        while (j < n && py_space((unsigned char)s[j])) j++;
        if (j == j0) { i++; continue; }                 /* \s+ 至少一个 */
        size_t w0 = j;
        while (j < n && py_word((unsigned char)s[j])) j++;
        if (j == w0) { i++; continue; }                 /* \w+ 至少一个 */
        size_t we = j;
        while (j < n && py_space((unsigned char)s[j])) j++;
        if (j < n && s[j] == '(') { sl_push(&out, xstrndup(s + w0, we - w0)); i = j + 1; }
        else i++;
    }
    return out;
}

/* 起点扫描: 命中 "(?:def|func)\s+<ent>\s*\(" 时返回 '(' 之后的下标, 否则 -1 */
static long decl_of_at(const char *s, size_t n, const char *ent, size_t el, size_t from) {
    for (size_t i = from; i < n; i++) {
        size_t kl = 0;
        if (i + 3 <= n && !strncmp(s + i, "def", 3)) kl = 3;
        else if (i + 4 <= n && !strncmp(s + i, "func", 4)) kl = 4;
        if (!kl) continue;
        size_t j = i + kl, j0 = j;
        while (j < n && py_space((unsigned char)s[j])) j++;
        if (j == j0) continue;
        if (j + el > n || strncmp(s + j, ent, el)) continue;
        j += el;
        while (j < n && py_space((unsigned char)s[j])) j++;
        if (j < n && s[j] == '(') return (long)(j + 1);
    }
    return -1;
}
/* re.search(r"(?:def|func)\s+<ent>\s*\(", f) */
static int has_decl(const char *f, const char *ent) {
    return decl_of_at(f, strlen(f), ent, strlen(ent), 0) >= 0;
}
/* re.search(r"(?:def|func)\s+<ent>\s*\([^\n]*\)[^\n]*(?:\{|:)", f)
 * 两个 [^\n]* 是贪婪+回溯: 先取到行尾再往回找 ')' , 再往回找 '{' 或 ':'。只判存在性,
 * 所以回溯顺序不影响结果; 起点失败要继续往后找下一个声明(re.search 会推进起点)。*/
static int has_full_decl(const char *f, const char *ent) {
    size_t n = strlen(f), el = strlen(ent);
    size_t from = 0;
    while (1) {
        long q = decl_of_at(f, n, ent, el, from);
        if (q < 0) return 0;
        size_t le = (size_t)q;
        while (le < n && f[le] != '\n') le++;
        for (size_t r = le; r > (size_t)q; r--) {
            if (f[r - 1] != ')') continue;
            for (size_t u = le; u >= r + 1; u--) {
                if (f[u - 1] == '{' || f[u - 1] == ':') return 1;
                if (u - 1 == r) break;
            }
        }
        from = (size_t)q;   /* 下一个候选起点 */
    }
}

/* '^' 位置: 串首或换行之后(re.M) */
static int at_line_start(const char *s, size_t i) { return i == 0 || s[i - 1] == '\n'; }

/* re.findall(r'"([^"]+)"', blk) */
static void quoted_strings(const char *b, slist *out) {
    size_t n = strlen(b), i = 0;
    while (i < n) {
        if (b[i] != '"') { i++; continue; }
        size_t j = i + 1;
        while (j < n && b[j] != '"') j++;
        if (j > i + 1 && j < n) { sl_push(out, xstrndup(b + i + 1, j - i - 1)); i = j + 1; }
        else i++;
    }
}

/* _go_import_paths(s) 逐式:
 *   for blk in re.findall(r"(?ms)^import\s*\((.*?)\)", s): out += re.findall(r'"([^"]+)"', blk)
 *   out += re.findall(r'(?m)^import\s+"([^"]+)"', s)  */
static slist go_import_paths(const char *s) {
    slist out = {0};
    size_t n = strlen(s);
    for (size_t i = 0; i < n; ) {
        if (!at_line_start(s, i) || i + 6 > n || strncmp(s + i, "import", 6)) { i++; continue; }
        size_t j = i + 6;
        while (j < n && py_space((unsigned char)s[j])) j++;   /* \s* 可跨行 */
        if (j >= n || s[j] != '(') { i++; continue; }
        const char *cp = memchr(s + j + 1, ')', n - j - 1);
        if (!cp) { i++; continue; }
        size_t k = (size_t)(cp - s);
        char *blk = xstrndup(s + j + 1, k - j - 1);
        quoted_strings(blk, &out);
        free(blk);
        i = k + 1;
    }
    slist single = {0};
    for (size_t i = 0; i < n; ) {
        if (!at_line_start(s, i) || i + 6 > n || strncmp(s + i, "import", 6)) { i++; continue; }
        size_t j = i + 6, j0 = j;
        while (j < n && py_space((unsigned char)s[j])) j++;
        if (j == j0 || j >= n || s[j] != '"') { i++; continue; }
        size_t k = j + 1;
        while (k < n && s[k] != '"') k++;
        if (k == j + 1 || k >= n) { i++; continue; }
        sl_push(&single, xstrndup(s + j + 1, k - j - 1));
        i = k + 1;
    }
    for (size_t t = 0; t < single.n; t++) sl_push(&out, single.v[t]);
    free(single.v);
    return out;
}

/* `\s*$` 尾: 贪婪吃空白后需落在 '$'(串尾或 '\n' 前)。返回匹配终点, -1=不匹配 */
static long ws_to_dollar(const char *s, size_t n, size_t from) {
    size_t w = from;
    while (w < n && py_space((unsigned char)s[w])) w++;
    if (w == n) return (long)n;
    for (size_t p = w; p > from; p--) if (s[p - 1] == '\n') return (long)(p - 1);
    if (from == n) return (long)n;
    if (s[from] == '\n') return (long)from;
    return -1;
}

/* _go_strip_headers(s) 逐式三连 re.sub */
static char *go_strip_headers(const char *s0) {
    /* ① re.sub(r"(?ms)^import\s*\(.*?\)\s*", "", s) */
    sb_t a = {0};
    {
        size_t n = strlen(s0);
        for (size_t i = 0; i < n; ) {
            if (at_line_start(s0, i) && i + 6 <= n && !strncmp(s0 + i, "import", 6)) {
                size_t j = i + 6;
                while (j < n && py_space((unsigned char)s0[j])) j++;
                if (j < n && s0[j] == '(') {
                    const char *cp = memchr(s0 + j + 1, ')', n - j - 1);
                    if (cp) {
                        size_t e = (size_t)(cp - s0) + 1;
                        while (e < n && py_space((unsigned char)s0[e])) e++;   /* 尾部 \s* 贪婪 */
                        i = e; continue;
                    }
                }
            }
            sb_ch(&a, s0[i]); i++;
        }
    }
    /* ② re.sub(r'(?m)^import\s+"[^"]+"\s*$', "", s) */
    sb_t b = {0};
    {
        const char *s = a.p ? a.p : ""; size_t n = a.n;
        for (size_t i = 0; i < n; ) {
            int matched = 0;
            if (at_line_start(s, i) && i + 6 <= n && !strncmp(s + i, "import", 6)) {
                size_t j = i + 6, j0 = j;
                while (j < n && py_space((unsigned char)s[j])) j++;
                if (j > j0 && j < n && s[j] == '"') {
                    size_t k = j + 1;
                    while (k < n && s[k] != '"') k++;
                    if (k > j + 1 && k < n) {
                        long e = ws_to_dollar(s, n, k + 1);
                        if (e >= 0) { i = (size_t)e; matched = 1; }
                    }
                }
            }
            if (!matched) { sb_ch(&b, s[i]); i++; }
        }
    }
    /* ③ re.sub(r"(?m)^package\s+\w+\s*$", "", s) */
    sb_t c = {0};
    {
        const char *s = b.p ? b.p : ""; size_t n = b.n;
        for (size_t i = 0; i < n; ) {
            int matched = 0;
            if (at_line_start(s, i) && i + 7 <= n && !strncmp(s + i, "package", 7)) {
                size_t j = i + 7, j0 = j;
                while (j < n && py_space((unsigned char)s[j])) j++;
                if (j > j0) {
                    size_t w0 = j;
                    while (j < n && py_word((unsigned char)s[j])) j++;
                    if (j > w0) {
                        long e = ws_to_dollar(s, n, j);
                        if (e >= 0) { i = (size_t)e; matched = 1; }
                    }
                }
            }
            if (!matched) { sb_ch(&c, s[i]); i++; }
        }
    }
    sb_free(&a); sb_free(&b);
    return c.p ? c.p : xstrdup("");
}

