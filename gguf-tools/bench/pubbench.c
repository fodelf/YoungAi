/* pubbench.c — 公共标尺 harness(C, 2026-08-25 Python→C 迁移): HumanEval(Python) /
 * HumanEval-X(Go) 子集, greedy, 原始输出全落盘。逐式转录 scripts/pubbench.py(438 行),
 * 判分口径一个字符都不许漂 —— 这是"真实编码可用性"铁律的判官工具。
 *
 * 量化口径: 同一套题对 基线 与 量化 各跑一遍(--tag 区分), --compare 出 per-task delta 表。
 * 判决器: Python=官方同构(拼 prompt+completion+test 交 python3 子进程执行, 超时即败);
 *         Go=拼包后 go test(拼装规则首火可能需拧一扣, 结果标 experimental 直到人工抽查确认)。
 * 注意: 会执行模型生成的代码(与官方 human-eval harness 同风险面), 只在本机跑。
 *
 * ★关于 python3★: 判 Python 题时本工具 fork/exec `python3 -c <程序>`。这里的 python3 是
 * "被判代码的目标运行时"(HumanEval 题目本身就是 Python 函数), 不是仓库里的 Python 代码 ——
 * 与"链上不留 Python"的迁移铁律不冲突。.py 版走的是 sys.executable, C 版走 PATH 上的 python3。
 *
 * 金标(转录验收):
 *   ① --selftest(两个 suite)输出与 .py 逐字符相同(164/164 标准答案全过);
 *   ② 同一份已落盘结果 JSONL 上, --compare / --rejudge / verdict 表与 .py 逐字符相同;
 *   ③ 抽取器夹具: 围栏/完整重写/裸续写/嵌套 brace 等构造样例上, extract_completion 输出
 *      与 .py 逐字节相同(migrate/pubbench_fixtures/)。
 * 歧义与已知偏差清单: migrate/pubbench_transcription_notes.md。
 *
 * 构建: gcc -O3 -o pubbench pubbench.c -lcurl -lz -lpthread -lm
 * 用法: pubbench [--suite humaneval|humaneval-x-go] [--url URL] [--tag TAG] [--limit N]
 *               [--offset N] [--max-tokens N] [--mode code] [--api chat|completions]
 *               [--http-timeout S] [--jobs N] [--eval-timeout S] [--out-dir DIR]
 *               [--compare A.jsonl B.jsonl] [--rejudge FILE.jsonl] [--selftest]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <curl/curl.h>
#include <zlib.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

/* ============================ 基础设施 ============================ */

static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

/* log(msg): .py 的 sys.stderr.write(msg+"\n")+flush。多线程下加锁, 与 GIL 下的整行原子等价 */
static void logln(const char *fmt, ...) {
    char buf[8192];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    pthread_mutex_lock(&g_log_mu);
    fputs(buf, stderr); fputc('\n', stderr); fflush(stderr);
    pthread_mutex_unlock(&g_log_mu);
}

/* die(): .py 的未捕获异常 —— 不复刻 traceback, 只保异常类名+文本与退出码 1 */
static void die(const char *fmt, ...) {
    char buf[4096];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%s\n", buf); fflush(stderr);
    exit(1);
}

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("MemoryError"); return p; }
static void *xrealloc(void *p, size_t n) { void *q = realloc(p, n ? n : 1); if (!q) die("MemoryError"); return q; }
static char *xstrdup(const char *s) { size_t n = strlen(s) + 1; char *p = xmalloc(n); memcpy(p, s, n); return p; }
static char *xstrndup(const char *s, size_t n) { char *p = xmalloc(n + 1); memcpy(p, s, n); p[n] = 0; return p; }

/* 变长字节缓冲(所有字符串都带显式长度: completion 里的 "\x00FULL\x00" 标记含 NUL) */
typedef struct { char *p; size_t n, cap; } sb_t;
static void sb_reserve(sb_t *b, size_t add) {
    if (b->n + add + 1 <= b->cap) return;
    size_t c = b->cap ? b->cap : 64;
    while (c < b->n + add + 1) c *= 2;
    b->p = xrealloc(b->p, c); b->cap = c;
}
static void sb_add(sb_t *b, const char *s, size_t n) { sb_reserve(b, n); memcpy(b->p + b->n, s, n); b->n += n; b->p[b->n] = 0; }
static void sb_puts(sb_t *b, const char *s) { sb_add(b, s, strlen(s)); }
static void sb_ch(sb_t *b, char c) { sb_add(b, &c, 1); }
static void sb_fmt(sb_t *b, const char *fmt, ...) {
    char tmp[1024];
    va_list ap; va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k < 0) return;
    if ((size_t)k < sizeof tmp) { sb_add(b, tmp, (size_t)k); return; }
    char *big = xmalloc((size_t)k + 1);
    va_start(ap, fmt); vsnprintf(big, (size_t)k + 1, fmt, ap); va_end(ap);
    sb_add(b, big, (size_t)k); free(big);
}
static void sb_free(sb_t *b) { free(b->p); b->p = NULL; b->n = b->cap = 0; }

/* 带长度的字符串(completion 用) */
typedef struct { char *p; size_t n; } str_t;
static str_t str_dup(const char *s, size_t n) { str_t r; r.p = xstrndup(s, n); r.n = n; return r; }
static str_t str_cstr(const char *s) { return str_dup(s, strlen(s)); }

/* ---- Python 语义的字符串小工具 ---- */
/* Python str.strip()/rstrip() 还吃 \x1c-\x1f/\x85/Unicode 空白, 这里只覆盖 ASCII 集
 * (题面/模型输出里不会出现那些字符, 见转录清单 B 组) */
static int py_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static int py_word(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

static size_t rstrip_len(const char *s, size_t n) { while (n && py_space((unsigned char)s[n - 1])) n--; return n; }
static char *py_rstrip(const char *s) { return xstrndup(s, rstrip_len(s, strlen(s))); }
static char *py_strip(const char *s, size_t n) {
    size_t a = 0; while (a < n && py_space((unsigned char)s[a])) a++;
    while (n > a && py_space((unsigned char)s[n - 1])) n--;
    return xstrndup(s + a, n - a);
}
/* splitlines()[-1]: 末尾若是换行则不产生空行(Python 语义)。只按 \n/\r/\r\n 切 */
static char *py_last_line(const char *s) {
    size_t n = strlen(s);
    if (!n) return xstrdup("");           /* splitlines() == [] → 调用方另行处理 */
    size_t e = n;
    if (s[e - 1] == '\n') { e--; if (e && s[e - 1] == '\r') e--; }
    else if (s[e - 1] == '\r') e--;
    size_t a = 0;
    for (size_t i = e; i > 0; i--) if (s[i - 1] == '\n' || s[i - 1] == '\r') { a = i; break; }
    return xstrndup(s + a, e - a);
}
/* 取前/后 k 个"字符"(码点)对应的字节数 —— .py 的 [:80] / [-800:] 是按字符切 */
static size_t head_chars(const char *s, size_t n, int maxc) {
    size_t i = 0; int c = 0;
    while (i < n && c < maxc) {
        unsigned char ch = (unsigned char)s[i];
        size_t adv = ch < 0x80 ? 1 : (ch >> 5) == 6 ? 2 : (ch >> 4) == 14 ? 3 : (ch >> 3) == 30 ? 4 : 1;
        if (i + adv > n) adv = n - i;
        i += adv; c++;
    }
    return i;
}
static size_t tail_chars(const char *s, size_t n, int maxc) {   /* 返回起始字节偏移 */
    size_t i = n; int c = 0;
    while (i > 0 && c < maxc) {
        i--;
        while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
        c++;
    }
    return i;
}

/* repr(float): Python 的最短往返表示(vq_merge_v4.c 同源实现) */
static const char *py_float_repr(double v, char *out) {
    static char e[64];
    if (isnan(v)) { strcpy(out, "nan"); return out; }
    if (isinf(v)) { strcpy(out, v > 0 ? "inf" : "-inf"); return out; }
    int p;
    for (p = 0; p <= 17; p++) { snprintf(e, sizeof e, "%.*e", p, v); if (strtod(e, NULL) == v) break; }
    const char *s = e; int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    char dig[40]; int nd = 0;
    dig[nd++] = *s++;
    if (*s == '.') { s++; while (*s != 'e' && *s != 'E') dig[nd++] = *s++; }
    while (*s != 'e' && *s != 'E') s++;
    int decpt = atoi(s + 1) + 1;
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    dig[nd] = 0;
    char *o = out;
    if (neg) *o++ = '-';
    if (decpt <= -4 || decpt > 16) {
        *o++ = dig[0];
        if (nd > 1) { *o++ = '.'; memcpy(o, dig + 1, nd - 1); o += nd - 1; }
        o += sprintf(o, "e%+03d", decpt - 1);
    } else if (decpt <= 0) {
        *o++ = '0'; *o++ = '.';
        for (int i = 0; i < -decpt; i++) *o++ = '0';
        memcpy(o, dig, nd); o += nd;
    } else if (decpt >= nd) {
        memcpy(o, dig, nd); o += nd;
        for (int i = 0; i < decpt - nd; i++) *o++ = '0';
        *o++ = '.'; *o++ = '0';
    } else {
        memcpy(o, dig, decpt); o += decpt; *o++ = '.';
        memcpy(o, dig + decpt, nd - decpt); o += nd - decpt;
    }
    *o = 0; return out;
}
/* round(x, 1): 先正确舍入到 1 位十进制(libc %.1f 与 CPython 同为最近偶数), 再回到最近 double */
static double py_round1(double x) { char b[64]; snprintf(b, sizeof b, "%.1f", x); return strtod(b, NULL); }

/* bytes.decode("utf-8","replace"): 每个"最大子部分"换一个 U+FFFD(CPython 语义) */
static char *utf8_replace(const char *s, size_t n, size_t *out_n) {
    sb_t o = {0};
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int need = c < 0x80 ? 0 : (c >= 0xC2 && c <= 0xDF) ? 1 : (c >= 0xE0 && c <= 0xEF) ? 2 : (c >= 0xF0 && c <= 0xF4) ? 3 : -1;
        if (need < 0) { sb_puts(&o, "\xEF\xBF\xBD"); i++; continue; }
        if (need == 0) { sb_ch(&o, (char)c); i++; continue; }
        size_t k = 1;
        int ok = 1;
        while (k <= (size_t)need) {
            if (i + k >= n) { ok = 0; break; }
            unsigned char d = (unsigned char)s[i + k];
            unsigned char lo = 0x80, hi = 0xBF;
            if (k == 1) {
                if (c == 0xE0) lo = 0xA0;
                else if (c == 0xED) hi = 0x9F;
                else if (c == 0xF0) lo = 0x90;
                else if (c == 0xF4) hi = 0x8F;
            }
            if (d < lo || d > hi) { ok = 0; break; }
            k++;
        }
        if (!ok) { sb_puts(&o, "\xEF\xBF\xBD"); i += k; continue; }   /* k = 已吃掉的最大子部分 */
        sb_add(&o, s + i, (size_t)need + 1); i += (size_t)need + 1;
    }
    if (out_n) *out_n = o.n;
    if (!o.p) { o.p = xstrdup(""); }
    return o.p;
}

/* ============================ 最小 JSON ============================ */

enum { JNULL, JBOOL, JINT, JFLOAT, JSTR, JARR, JOBJ };
typedef struct jv jv;
struct jv {
    int type;
    int b;                 /* JBOOL */
    long long i;           /* JINT */
    double f;              /* JFLOAT */
    char *s; size_t slen;  /* JSTR */
    jv **kid; char **key; size_t n, cap;   /* JARR(key=NULL) / JOBJ */
};

static jv *jnew(int t) { jv *v = xmalloc(sizeof *v); memset(v, 0, sizeof *v); v->type = t; return v; }
static void jpush(jv *o, char *k, jv *val) {
    if (o->n == o->cap) { o->cap = o->cap ? o->cap * 2 : 8; o->kid = xrealloc(o->kid, o->cap * sizeof *o->kid); o->key = xrealloc(o->key, o->cap * sizeof *o->key); }
    o->key[o->n] = k; o->kid[o->n] = val; o->n++;
}
static jv *jget(const jv *o, const char *k) {
    if (!o || o->type != JOBJ) return NULL;
    for (size_t i = 0; i < o->n; i++) if (o->key[i] && !strcmp(o->key[i], k)) return o->kid[i];
    return NULL;
}
/* dict 赋值: 键已存在则原位替换(保序), 否则追加 —— 与 Python dict 一致 */
static void jset(jv *o, const char *k, jv *val) {
    for (size_t i = 0; i < o->n; i++) if (o->key[i] && !strcmp(o->key[i], k)) { o->kid[i] = val; return; }
    jpush(o, xstrdup(k), val);
}
static const char *jstr(const jv *o, const char *k, const char *def) {
    jv *v = jget(o, k);
    return (v && v->type == JSTR) ? v->s : def;
}
static const char *jstr_req(const jv *o, const char *k) {
    jv *v = jget(o, k);
    if (!v || v->type != JSTR) die("KeyError: '%s'", k);
    return v->s;
}

static void jfree(jv *v) {
    if (!v) return;
    for (size_t i = 0; i < v->n; i++) { free(v->key ? v->key[i] : NULL); jfree(v->kid[i]); }
    free(v->kid); free(v->key); free(v->s); free(v);
}

static void utf8_emit(sb_t *o, unsigned cp) {
    if (cp < 0x80) sb_ch(o, (char)cp);
    else if (cp < 0x800) { sb_ch(o, (char)(0xC0 | (cp >> 6))); sb_ch(o, (char)(0x80 | (cp & 63))); }
    else if (cp < 0x10000) { sb_ch(o, (char)(0xE0 | (cp >> 12))); sb_ch(o, (char)(0x80 | ((cp >> 6) & 63))); sb_ch(o, (char)(0x80 | (cp & 63))); }
    else { sb_ch(o, (char)(0xF0 | (cp >> 18))); sb_ch(o, (char)(0x80 | ((cp >> 12) & 63))); sb_ch(o, (char)(0x80 | ((cp >> 6) & 63))); sb_ch(o, (char)(0x80 | (cp & 63))); }
}

typedef struct { const char *s; size_t n, i; int bad; } jp_t;
static void jp_ws(jp_t *p) { while (p->i < p->n && (p->s[p->i] == ' ' || p->s[p->i] == '\t' || p->s[p->i] == '\n' || p->s[p->i] == '\r')) p->i++; }
static jv *jp_val(jp_t *p);

static int jp_hex4(jp_t *p, unsigned *out) {
    unsigned v = 0;
    for (int k = 0; k < 4; k++) {
        if (p->i >= p->n) return 0;
        char c = p->s[p->i++];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v; return 1;
}
static jv *jp_str(jp_t *p) {
    if (p->i >= p->n || p->s[p->i] != '"') { p->bad = 1; return NULL; }
    p->i++;
    sb_t o = {0};
    while (1) {
        if (p->i >= p->n) { p->bad = 1; sb_free(&o); return NULL; }
        unsigned char c = (unsigned char)p->s[p->i++];
        if (c == '"') break;
        if (c != '\\') { sb_ch(&o, (char)c); continue; }
        if (p->i >= p->n) { p->bad = 1; sb_free(&o); return NULL; }
        char e = p->s[p->i++];
        switch (e) {
            case '"': sb_ch(&o, '"'); break;
            case '\\': sb_ch(&o, '\\'); break;
            case '/': sb_ch(&o, '/'); break;
            case 'b': sb_ch(&o, '\b'); break;
            case 'f': sb_ch(&o, '\f'); break;
            case 'n': sb_ch(&o, '\n'); break;
            case 'r': sb_ch(&o, '\r'); break;
            case 't': sb_ch(&o, '\t'); break;
            case 'u': {
                unsigned cp;
                if (!jp_hex4(p, &cp)) { p->bad = 1; sb_free(&o); return NULL; }
                if (cp >= 0xD800 && cp <= 0xDBFF && p->i + 1 < p->n && p->s[p->i] == '\\' && p->s[p->i + 1] == 'u') {
                    size_t save = p->i;
                    p->i += 2;
                    unsigned lo;
                    if (jp_hex4(p, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    else p->i = save;
                }
                utf8_emit(&o, cp);
                break;
            }
            default: p->bad = 1; sb_free(&o); return NULL;
        }
    }
    jv *v = jnew(JSTR);
    v->s = o.p ? o.p : xstrdup(""); v->slen = o.n;
    return v;
}
static jv *jp_val(jp_t *p) {
    jp_ws(p);
    if (p->i >= p->n) { p->bad = 1; return NULL; }
    char c = p->s[p->i];
    if (c == '"') return jp_str(p);
    if (c == '{') {
        p->i++;
        jv *o = jnew(JOBJ);
        jp_ws(p);
        if (p->i < p->n && p->s[p->i] == '}') { p->i++; return o; }
        while (1) {
            jp_ws(p);
            jv *k = jp_str(p);
            if (!k) { p->bad = 1; jfree(o); return NULL; }
            jp_ws(p);
            if (p->i >= p->n || p->s[p->i] != ':') { p->bad = 1; jfree(k); jfree(o); return NULL; }
            p->i++;
            jv *v = jp_val(p);
            if (!v) { p->bad = 1; jfree(k); jfree(o); return NULL; }
            char *ks = xstrdup(k->s); jfree(k);
            /* 重复键: Python 后者覆盖前者且保留首次位置 */
            int found = 0;
            for (size_t t = 0; t < o->n; t++) if (!strcmp(o->key[t], ks)) { jfree(o->kid[t]); o->kid[t] = v; found = 1; free(ks); break; }
            if (!found) jpush(o, ks, v);
            jp_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') { p->i++; continue; }
            if (p->i < p->n && p->s[p->i] == '}') { p->i++; return o; }
            p->bad = 1; jfree(o); return NULL;
        }
    }
    if (c == '[') {
        p->i++;
        jv *a = jnew(JARR);
        jp_ws(p);
        if (p->i < p->n && p->s[p->i] == ']') { p->i++; return a; }
        while (1) {
            jv *v = jp_val(p);
            if (!v) { p->bad = 1; jfree(a); return NULL; }
            jpush(a, NULL, v);
            jp_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') { p->i++; continue; }
            if (p->i < p->n && p->s[p->i] == ']') { p->i++; return a; }
            p->bad = 1; jfree(a); return NULL;
        }
    }
    if (!strncmp(p->s + p->i, "true", 4) && p->i + 4 <= p->n) { p->i += 4; jv *v = jnew(JBOOL); v->b = 1; return v; }
    if (!strncmp(p->s + p->i, "false", 5) && p->i + 5 <= p->n) { p->i += 5; jv *v = jnew(JBOOL); v->b = 0; return v; }
    if (!strncmp(p->s + p->i, "null", 4) && p->i + 4 <= p->n) { p->i += 4; return jnew(JNULL); }
    /* 数字: 带 . / e 的落 float, 否则落 int(与 json.loads 的 int/float 分流一致) */
    size_t st = p->i;
    if (p->i < p->n && (p->s[p->i] == '-' || p->s[p->i] == '+')) p->i++;
    int isf = 0, any = 0;
    while (p->i < p->n) {
        char d = p->s[p->i];
        if (d >= '0' && d <= '9') { any = 1; p->i++; }
        else if (d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') { isf = 1; p->i++; }
        else break;
    }
    if (!any) { p->bad = 1; return NULL; }
    char *tmp = xstrndup(p->s + st, p->i - st);
    jv *v;
    if (isf) { v = jnew(JFLOAT); v->f = strtod(tmp, NULL); }
    else { v = jnew(JINT); v->i = strtoll(tmp, NULL, 10); }
    free(tmp);
    return v;
}
/* json.loads: 成功返回 jv, 失败返回 NULL(调用方按 .py 的 except 分支处理) */
static jv *json_loads(const char *s, size_t n) {
    jp_t p = { s, n, 0, 0 };
    jv *v = jp_val(&p);
    if (!v || p.bad) { jfree(v); return NULL; }
    jp_ws(&p);
    if (p.i != p.n) { jfree(v); return NULL; }
    return v;
}

/* json.dumps 的字符串转义: ensure_ascii=False 只转 \\ " 和 <0x20; True 时非 ASCII 走 \uXXXX */
static void json_esc(sb_t *o, const char *s, size_t n, int ensure_ascii) {
    sb_ch(o, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"': sb_puts(o, "\\\""); continue;
            case '\\': sb_puts(o, "\\\\"); continue;
            case '\n': sb_puts(o, "\\n"); continue;
            case '\r': sb_puts(o, "\\r"); continue;
            case '\t': sb_puts(o, "\\t"); continue;
            case '\b': sb_puts(o, "\\b"); continue;
            case '\f': sb_puts(o, "\\f"); continue;
            default: break;
        }
        if (c < 0x20) { sb_fmt(o, "\\u%04x", c); continue; }
        if (c < 0x80 || !ensure_ascii) { sb_ch(o, (char)c); continue; }
        /* ensure_ascii: 解回码点再按 \uXXXX(BMP 外走代理对), 与 CPython 一致 */
        unsigned cp = c; int extra = 0;
        if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
        for (int k = 0; k < extra && i + 1 < n; k++) { cp = (cp << 6) | ((unsigned char)s[++i] & 0x3F); }
        if (cp < 0x10000) sb_fmt(o, "\\u%04x", cp);
        else { cp -= 0x10000; sb_fmt(o, "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)); }
    }
    sb_ch(o, '"');
}
static void json_dump(sb_t *o, const jv *v, int ensure_ascii) {
    char fb[64];
    switch (v->type) {
        case JNULL: sb_puts(o, "null"); break;
        case JBOOL: sb_puts(o, v->b ? "true" : "false"); break;
        case JINT: sb_fmt(o, "%lld", v->i); break;
        case JFLOAT: sb_puts(o, py_float_repr(v->f, fb)); break;
        case JSTR: json_esc(o, v->s, v->slen, ensure_ascii); break;
        case JARR:
            sb_ch(o, '[');
            for (size_t i = 0; i < v->n; i++) { if (i) sb_puts(o, ", "); json_dump(o, v->kid[i], ensure_ascii); }
            sb_ch(o, ']');
            break;
        case JOBJ:
            sb_ch(o, '{');
            for (size_t i = 0; i < v->n; i++) {
                if (i) sb_puts(o, ", ");
                json_esc(o, v->key[i], strlen(v->key[i]), ensure_ascii);
                sb_puts(o, ": ");
                json_dump(o, v->kid[i], ensure_ascii);
            }
            sb_ch(o, '}');
            break;
    }
}
static jv *jstrv(const char *s, size_t n) { jv *v = jnew(JSTR); v->s = xstrndup(s, n); v->slen = n; return v; }
static jv *jboolv(int b) { jv *v = jnew(JBOOL); v->b = b; return v; }
static jv *jfloatv(double f) { jv *v = jnew(JFLOAT); v->f = f; return v; }
static jv *jintv(long long i) { jv *v = jnew(JINT); v->i = i; return v; }

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

/* 默认仍是 /tmp(旧行为不变)。PUBBENCH_CACHE 指向 gguf-tools/go-onebit/pubbench_data/
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

/* ============================ extract_completion ============================ */

/* 逐分支照抄 .py 的 extract_completion(text, prompt, stops, strip_prompt, close_brace)。
 * 返回带长度的 str_t —— "完整重写"形态的前缀标记 "\x00FULL\x00" 含 NUL 字节。*/
static str_t extract_completion(const char *text, const char *prompt, const char *const *stops,
                                int strip_prompt, int close_brace) {
    /* 服务端 mode:code 常给围栏; 取第一个围栏体, 否则用原文 */
    slist fences = find_fences(text);
    char *body = xstrdup(fences.n ? fences.v[0] : text);

    /* chat 完整重写形态(2026-08-18): 围栏体自带入口函数完整定义 —— 直接原样返回,
     * eval 侧当独立完整程序拼 test(续写式剥重叠对它必然错位)。*/
    if (strip_prompt && fences.n) {
        slist dm = find_decl_names(prompt, 1);
        if (dm.n) {
            const char *ent = dm.v[dm.n - 1];
            const char *best = NULL; size_t bl = 0;
            for (size_t i = 0; i < fences.n; i++) {
                const char *f = fences.v[i];
                if (!has_decl(f, ent) || !has_full_decl(f, ent)) continue;
                size_t L = strlen(f);
                if (!best || L > bl) { best = f; bl = L; }   /* max(key=len): 并列取先出现者 */
            }
            if (best) {
                size_t rl = rstrip_len(best, bl);
                sb_t o = {0};
                sb_add(&o, FULL_MARK, 6);
                sb_add(&o, best, rl);
                sb_ch(&o, '\n');
                free(body); sl_free(&dm); sl_free(&fences);
                str_t r = { o.p, o.n };
                return r;
            }
        }
        sl_free(&dm);
    }

    /* 若模型把题面(签名)复述了, 剥掉与 prompt 重叠的头部。
     * completions 裸续写(strip_prompt=0)禁用: response 本就是纯续写, 而"最后一行锚"
     * 对 HumanEval 恒为 \"\"\" — 会命中续写里任意 docstring 把正确答案整段扔掉(t4 实证)。*/
    char *p_tail = py_rstrip(prompt);
    if (!strip_prompt) {
        /* pass */
    } else if (*p_tail && strstr(body, p_tail)) {
        const char *hit = strstr(body, p_tail);
        char *nb = xstrdup(hit + strlen(p_tail));
        free(body); body = nb;
    } else {
        char *sig = NULL;
        if (*p_tail) { char *ll = py_last_line(p_tail); sig = py_strip(ll, strlen(ll)); free(ll); }
        else sig = xstrdup("");
        if (*sig && strstr(body, sig)) {
            const char *hit = strstr(body, sig);
            char *nb = xstrdup(hit + strlen(sig));
            free(body); body = nb;
            if (body[0] == ':') { char *t = xstrdup(body + 1); free(body); body = t; }
        }
        free(sig);
    }
    free(p_tail);

    /* 客户端截断(不依赖服务端 stop 参数) */
    size_t cut = strlen(body);
    for (int i = 0; stops[i]; i++) {
        const char *h = strstr(body, stops[i]);
        if (h) { size_t k = (size_t)(h - body); if (k < cut) cut = k; }
    }
    body[cut] = 0;

    /* Go 漂移截断(2026-07-27 v2): ①入口函数被重复声明处=确定的漂移起点, 截掉;
     * ②截到最后一个第 0 列 "}"(gofmt 顶层函数闭合必在列 0) —— 不用花括号深度配平,
     * Go 类型语法的内联括号(interface{}/struct{})会骗计数器。*/
    if (close_brace) {
        slist ms = find_decl_names(prompt, 0);
        if (ms.n) {
            char pat[256];
            snprintf(pat, sizeof pat, "func %s(", ms.v[ms.n - 1]);
            const char *h = strstr(body, pat);
            if (h) body[(size_t)(h - body)] = 0;
        }
        sl_free(&ms);
        /* rfind("\n}") */
        size_t bl = strlen(body);
        if (bl >= 2) for (size_t i = bl - 1; i > 0; i--) {
            if (body[i - 1] == '\n' && body[i] == '}') { body[i + 1] = 0; break; }
        }
    }
    size_t rl = rstrip_len(body, strlen(body));
    sb_t o = {0};
    sb_add(&o, body, rl);
    sb_ch(&o, '\n');
    free(body); sl_free(&fences);
    str_t r = { o.p, o.n };
    return r;
}

/* ============================ 子进程执行 ============================ */

typedef struct { int rc, timed_out, spawn_errno; char *out; size_t out_n; char *err; size_t err_n; } proc_t;

static pthread_mutex_t g_fork_mu = PTHREAD_MUTEX_INITIALIZER;
static double now_s(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

static void sink_append(char **buf, size_t *n, size_t *cap, const char *d, size_t dn) {
    if (*n + dn + 1 > *cap) {
        size_t c = *cap ? *cap : 65536;
        while (c < *n + dn + 1) c *= 2;
        *buf = xrealloc(*buf, c); *cap = c;
    }
    memcpy(*buf + *n, d, dn); *n += dn; (*buf)[*n] = 0;
    /* 只有末尾 400-800 字符会被读; 超过 2 MiB 时丢头保尾, 防生成代码狂打日志把内存吃干
     * (.py 无上限, 这是转录清单里显式记录的偏差, 不影响任何判分读数)。*/
    if (*n > (2u << 20)) { size_t keep = 1u << 20; memmove(*buf, *buf + (*n - keep), keep); *n = keep; (*buf)[*n] = 0; }
}

/* subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=T) */
static void run_proc(char *const argv[], const char *cwd, double timeout_s, proc_t *r) {
    memset(r, 0, sizeof *r);
    int po[2], pe[2], px[2];
    pthread_mutex_lock(&g_fork_mu);
    if (pipe(po) || pipe(pe) || pipe(px)) { pthread_mutex_unlock(&g_fork_mu); r->spawn_errno = errno; return; }
    for (int i = 0; i < 2; i++) { fcntl(po[i], F_SETFD, FD_CLOEXEC); fcntl(pe[i], F_SETFD, FD_CLOEXEC); fcntl(px[i], F_SETFD, FD_CLOEXEC); }
    pid_t pid = fork();
    if (pid == 0) {
        dup2(po[1], 1); dup2(pe[1], 2);
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) { dup2(devnull, 0); close(devnull); }
        if (cwd && chdir(cwd) != 0) { int e = errno; ssize_t w = write(px[1], &e, sizeof e); (void)w; _exit(127); }
        execvp(argv[0], argv);
        int e = errno; ssize_t w = write(px[1], &e, sizeof e); (void)w;
        _exit(127);
    }
    close(po[1]); close(pe[1]); close(px[1]);
    pthread_mutex_unlock(&g_fork_mu);
    if (pid < 0) { close(po[0]); close(pe[0]); close(px[0]); r->spawn_errno = errno; return; }

    int se = 0;
    if (read(px[0], &se, sizeof se) == (ssize_t)sizeof se) r->spawn_errno = se;
    close(px[0]);

    size_t ocap = 0, ecap = 0;
    double deadline = now_s() + timeout_s;
    int of = po[0], ef = pe[0];
    fcntl(of, F_SETFL, O_NONBLOCK); fcntl(ef, F_SETFL, O_NONBLOCK);
    char tmp[65536];
    while (of >= 0 || ef >= 0) {
        double left = deadline - now_s();
        if (left <= 0) { r->timed_out = 1; break; }
        struct pollfd pf[2]; int np = 0;
        int io = -1, ie = -1;
        if (of >= 0) { pf[np].fd = of; pf[np].events = POLLIN; io = np++; }
        if (ef >= 0) { pf[np].fd = ef; pf[np].events = POLLIN; ie = np++; }
        int pr = poll(pf, (nfds_t)np, (int)(left * 1000) + 1);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) continue;
        if (io >= 0 && (pf[io].revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t k = read(of, tmp, sizeof tmp);
            if (k > 0) sink_append(&r->out, &r->out_n, &ocap, tmp, (size_t)k);
            else if (k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR)) { close(of); of = -1; }
        }
        if (ie >= 0 && (pf[ie].revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t k = read(ef, tmp, sizeof tmp);
            if (k > 0) sink_append(&r->err, &r->err_n, &ecap, tmp, (size_t)k);
            else if (k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR)) { close(ef); ef = -1; }
        }
    }
    if (r->timed_out) {
        kill(pid, SIGKILL);
        if (of >= 0) close(of);
        if (ef >= 0) close(ef);
    } else {
        if (of >= 0) close(of);
        if (ef >= 0) close(ef);
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    r->rc = WIFEXITED(st) ? WEXITSTATUS(st) : -(WTERMSIG(st));
    if (!r->out) r->out = xstrdup("");
    if (!r->err) r->err = xstrdup("");
}
static void proc_free(proc_t *r) { free(r->out); free(r->err); r->out = r->err = NULL; }

/* text=True 的 universal newlines: \r\n / 孤立 \r → \n */
static char *unix_newlines(const char *s, size_t n, size_t *out_n) {
    char *o = xmalloc(n + 1); size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\r') { if (i + 1 < n && s[i + 1] == '\n') i++; o[k++] = '\n'; }
        else o[k++] = s[i];
    }
    o[k] = 0; if (out_n) *out_n = k;
    return o;
}
/* r.stderr[-N:] / r.stdout[-N:] —— 按字符切 */
static char *tail_of(const char *s, size_t n, int chars) {
    size_t nn; char *u = unix_newlines(s, n, &nn);
    size_t off = tail_chars(u, nn, chars);
    char *r = xstrndup(u + off, nn - off);
    free(u);
    return r;
}

/* ============================ 判定器 ============================ */

typedef struct { const char *prompt, *test, *entry_point, *test_setup, *import_; } row_t;
typedef struct { int ok; char *err; } judge_t;

static int is_full(const str_t *c) { return c->n >= 6 && !memcmp(c->p, FULL_MARK, 6); }

static judge_t eval_python(const row_t *row, const str_t *completion, double timeout_s) {
    judge_t J = { 0, NULL };
    sb_t prog = {0};
    if (is_full(completion)) {
        sb_add(&prog, completion->p + 6, completion->n - 6);
    } else {
        sb_puts(&prog, row->prompt);
        sb_add(&prog, completion->p, completion->n);
    }
    sb_ch(&prog, '\n');
    sb_puts(&prog, row->test);
    sb_fmt(&prog, "\ncheck(%s)\n", row->entry_point);
    /* sys.executable → C 侧走 PATH 上的 python3(被判代码的目标运行时) */
    char *argv[] = { (char *)"python3", (char *)"-c", prog.p, NULL };
    proc_t r;
    run_proc(argv, NULL, timeout_s, &r);
    if (r.spawn_errno) {
        char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s: 'python3'", r.spawn_errno, strerror(r.spawn_errno));
        J.ok = 0; J.err = xstrdup(b);
    } else if (r.timed_out) {
        J.ok = 0; J.err = xstrdup("timeout");
    } else {
        J.ok = (r.rc == 0);
        J.err = J.ok ? xstrdup("") : tail_of(r.err, r.err_n, 800);
    }
    proc_free(&r); sb_free(&prog);
    return J;
}

static judge_t eval_go(const row_t *row, const str_t *completion, double timeout_s) {
    judge_t J = { 0, NULL };
    /* 单文件拼装(2026-07-27 修): humaneval-x go 的 test 与解答共享同文件 import。
     * 合并全部 import 路径去重 → 一个 _test.go; goimports -w 删未用 import。*/
    char tmpl[4096];
    const char *td = getenv("TMPDIR");
    if (!td || !*td) td = "/tmp";
    size_t tl = strlen(td);
    snprintf(tmpl, sizeof tmpl, "%s%spubbench_go_XXXXXX", td, (tl && td[tl - 1] == '/') ? "" : "/");
    char *d = mkdtemp(tmpl);
    if (!d) { J.ok = 0; char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s", errno, strerror(errno)); J.err = xstrdup(b); return J; }

    const char *setup = row->test_setup ? row->test_setup : "";
    sb_t sol = {0};
    if (is_full(completion)) sb_add(&sol, completion->p + 6, completion->n - 6);
    else { sb_puts(&sol, row->prompt); sb_add(&sol, completion->p, completion->n); }

    sb_t setup_test = {0};
    sb_puts(&setup_test, setup); sb_ch(&setup_test, '\n'); sb_puts(&setup_test, row->test);

    slist paths = {0};
    const char *srcs[4] = { row->import_ ? row->import_ : "", sol.p ? sol.p : "", setup, row->test };
    for (int i = 0; i < 4; i++) {
        slist got = go_import_paths(srcs[i]);
        for (size_t t = 0; t < got.n; t++) {
            int dup = 0;
            for (size_t u = 0; u < paths.n; u++) if (!strcmp(paths.v[u], got.v[t])) { dup = 1; break; }
            if (!dup) sl_push(&paths, xstrdup(got.v[t]));
        }
        sl_free(&got);
    }
    sb_t src = {0};
    sb_puts(&src, "package main\n\n");
    if (paths.n) {
        sb_puts(&src, "import (\n");
        for (size_t t = 0; t < paths.n; t++) sb_fmt(&src, "    \"%s\"\n", paths.v[t]);
        sb_puts(&src, ")\n");
    }
    sb_ch(&src, '\n');
    char *h1 = go_strip_headers(sol.p ? sol.p : "");
    char *h2 = go_strip_headers(setup_test.p ? setup_test.p : "");
    sb_puts(&src, h1); sb_ch(&src, '\n'); sb_puts(&src, h2);
    free(h1); free(h2);
    sl_free(&paths); sb_free(&sol); sb_free(&setup_test);

    char p[4200];
    snprintf(p, sizeof p, "%s/sol_test.go", d);
    FILE *f = fopen(p, "w");
    if (!f) { char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s: '%s'", errno, strerror(errno), p); J.ok = 0; J.err = xstrdup(b); sb_free(&src); return J; }
    fwrite(src.p, 1, src.n, f); fclose(f); sb_free(&src);

    proc_t r;
    char gi[4096];
    const char *home = getenv("HOME");
    snprintf(gi, sizeof gi, "%s/go/bin/goimports", home ? home : "");
    struct stat st;
    if (stat(gi, &st) == 0) {
        char *av[] = { gi, (char *)"-w", p, NULL };
        run_proc(av, NULL, 60, &r);
        if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
        proc_free(&r);
    }
    { char *av[] = { (char *)"go", (char *)"mod", (char *)"init", (char *)"pubbench", NULL };
      run_proc(av, d, 60, &r);
      if (r.spawn_errno) { char b[256]; snprintf(b, sizeof b, "harness: [Errno %d] %s: 'go'", r.spawn_errno, strerror(r.spawn_errno)); proc_free(&r); J.ok = 0; J.err = xstrdup(b); return J; }
      if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
      proc_free(&r); }
    { char *av[] = { (char *)"go", (char *)"mod", (char *)"tidy", NULL };
      run_proc(av, d, 120, &r);
      if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
      proc_free(&r); }
    { char *av[] = { (char *)"go", (char *)"test", (char *)"./...", NULL };
      run_proc(av, d, timeout_s, &r);
      if (r.timed_out) { proc_free(&r); J.ok = 0; J.err = xstrdup("timeout"); return J; }
      J.ok = (r.rc == 0);
      if (J.ok) J.err = xstrdup("");
      else {
          char *a = tail_of(r.out, r.out_n, 400), *b = tail_of(r.err, r.err_n, 400);
          sb_t e = {0}; sb_puts(&e, a); sb_puts(&e, b);
          free(a); free(b);
          J.err = e.p ? e.p : xstrdup("");
      }
      proc_free(&r); }
    return J;
}

/* ============================ call_server ============================ */

typedef struct { char *txt; size_t txt_n; char *raw; size_t raw_n; double dt; int failed; char err[512]; } srv_t;

/* api=completions: BASE 模型口径 = /v1/completions raw 裸续写(BOS+题面), 返回纯续写文本。
 * api=chat: 原 /v1/chat/completions mode:code(instruct/chat 部署用)。*/
static srv_t call_server(const char *url, const char *prompt, int max_tokens, int timeout, const char *mode, const char *api) {
    srv_t R; memset(&R, 0, sizeof R);
    int is_comp = !strcmp(api, "completions");
    sb_t body = {0};
    if (is_comp) {
        sb_puts(&body, "{\"model\": \"ds4\", \"prompt\": ");
        sb_t pp = {0}; sb_puts(&pp, BOS); sb_puts(&pp, prompt);
        json_esc(&body, pp.p, pp.n, 1);
        sb_free(&pp);
        sb_fmt(&body, ", \"temperature\": 0, \"max_tokens\": %d, \"raw\": true}", max_tokens);
    } else {
        sb_puts(&body, "{\"model\": \"ds4\", \"messages\": [{\"role\": \"user\", \"content\": ");
        json_esc(&body, prompt, strlen(prompt), 1);
        sb_fmt(&body, "}], \"temperature\": 0, \"max_tokens\": %d, \"stream\": false, \"mode\": ", max_tokens);
        json_esc(&body, mode, strlen(mode), 1);
        sb_ch(&body, '}');
    }
    char full[2048];
    size_t ul = strlen(url);
    while (ul && url[ul - 1] == '/') ul--;              /* url.rstrip("/") */
    snprintf(full, sizeof full, "%.*s%s", (int)ul, url, is_comp ? "/v1/completions" : "/v1/chat/completions");

    double t0 = now_s();
    char *raw = NULL; size_t raw_n = 0;
    if (http_post_json(full, body.p, timeout, &raw, &raw_n, R.err, sizeof R.err) != 0) {
        R.failed = 1; sb_free(&body);
        return R;
    }
    R.dt = now_s() - t0;
    sb_free(&body);
    size_t dec_n;
    char *dec = utf8_replace(raw, raw_n, &dec_n);       /* .decode("utf-8","replace") */
    free(raw);
    R.raw = dec; R.raw_n = dec_n;
    R.txt = xstrdup(""); R.txt_n = 0;                   /* txt = "" 起手, 解析失败即回落 raw */
    jv *j = json_loads(dec, dec_n);
    if (j) {
        jv *ch = jget(j, "choices");
        if (ch && ch->type == JARR && ch->n > 0) {
            jv *c = ch->kid[0];
            jv *t = NULL;
            if (is_comp) t = jget(c, "text");
            else { jv *m = jget(c, "message"); t = m ? jget(m, "content") : NULL; }
            if (t && t->type == JSTR) { free(R.txt); R.txt = xstrndup(t->s, t->slen); R.txt_n = t->slen; }
            else { free(R.txt); R.txt = xstrndup(dec, dec_n); R.txt_n = dec_n; }
        } else { free(R.txt); R.txt = xstrndup(dec, dec_n); R.txt_n = dec_n; }
        jfree(j);
    } else { free(R.txt); R.txt = xstrndup(dec, dec_n); R.txt_n = dec_n; }
    return R;
}

/* ============================ 参数 ============================ */

typedef struct {
    const char *suite, *url, *tag, *mode, *api, *out_dir;
    int limit, offset, max_tokens, http_timeout, jobs, eval_timeout;
    const char *cmp_a, *cmp_b, *rejudge_path;
    int selftest;
} args_t;

/* ============================ run_suite ============================ */

typedef struct {
    const args_t *a;
    rows_t rows;
    const char *const *stops;
    int is_go;
    const char *lang;
    jv **results;
    int done_n;
    size_t next;
    pthread_mutex_t mu;
} job_t;

static void one_task(job_t *J, size_t k) {
    const args_t *a = J->a;
    jv *row = J->rows.row[k];
    char tidbuf[128];
    const char *tid = jstr(row, "task_id", NULL);
    if (!tid) { snprintf(tidbuf, sizeof tidbuf, "%s/%d", a->suite, a->offset + (int)k); tid = tidbuf; }

    /* ★Go 生成端 prompt 补全(2026-08-05)★: 判定端一直补 package+import 编译, 生成端却发
     * 裸函数 — BASE 模型的 Go 语料函数永远在文件头之后, 裸函数=分布外 ⇒ TODO 弃权/风格漂移。*/
    const char *prompt = jstr_req(row, "prompt");
    sb_t gp = {0};
    if (J->is_go) {
        const char *imp0 = jstr(row, "import", "");
        char *imp = py_strip(imp0, strlen(imp0));
        sb_puts(&gp, "package main\n\n");
        if (*imp) { sb_puts(&gp, imp); sb_puts(&gp, "\n\n"); }
        const char *pl = prompt;
        while (*pl == '\n') pl++;                        /* .lstrip("\n") */
        sb_puts(&gp, pl);
        free(imp);
    } else {
        sb_puts(&gp, prompt);
    }
    const char *gen_prompt = gp.p ? gp.p : "";

    srv_t S = call_server(a->url, gen_prompt, a->max_tokens, a->http_timeout, a->mode, a->api);
    str_t comp; char *err; int ok; double dt;
    char *text, *raw; size_t text_n, raw_n;
    if (S.failed) {
        text = xstrdup(""); text_n = 0; raw = xstrdup(""); raw_n = 0; dt = 0.0;
        comp = str_cstr(""); ok = 0;
        char b[600]; snprintf(b, sizeof b, "request: %s", S.err);
        err = xstrdup(b);
    } else {
        text = S.txt; text_n = S.txt_n; raw = S.raw; raw_n = S.raw_n; dt = S.dt;
        comp = extract_completion(text, gen_prompt, J->stops,
                                  strcmp(a->api, "completions") != 0, J->is_go);
        row_t r = { prompt, jstr(row, "test", NULL), jstr(row, "entry_point", NULL),
                    jstr(row, "test_setup", ""), jstr(row, "import", "") };
        /* .py 里 row["test"]/row["entry_point"] 在 eval_* 的 try 之外, 缺字段会被
         * one_task 的 except 兜成 err="request: 'test'"; 两份入库数据集恒有这些字段,
         * C 侧直接判死(转录清单 B 组) */
        if (!r.test || (!J->is_go && !r.entry_point)) die("KeyError: 'test'");
        judge_t j = J->is_go ? eval_go(&r, &comp, a->eval_timeout) : eval_python(&r, &comp, a->eval_timeout);
        ok = j.ok; err = j.err;
    }

    jv *rec = jnew(JOBJ);
    jpush(rec, xstrdup("task_id"), jstrv(tid, strlen(tid)));
    jpush(rec, xstrdup("lang"), jstrv(J->lang, strlen(J->lang)));
    jpush(rec, xstrdup("tag"), jstrv(a->tag, strlen(a->tag)));
    jpush(rec, xstrdup("pass"), jboolv(ok));
    jpush(rec, xstrdup("err"), jstrv(err, strlen(err)));
    jpush(rec, xstrdup("gen_seconds"), jfloatv(py_round1(dt)));
    jpush(rec, xstrdup("prompt"), jstrv(prompt, strlen(prompt)));
    jpush(rec, xstrdup("completion"), jstrv(comp.p, comp.n));
    jpush(rec, xstrdup("response_text"), jstrv(text, text_n));
    jpush(rec, xstrdup("response_raw"), jstrv(raw, raw_n));

    pthread_mutex_lock(&J->mu);
    J->results[k] = rec;
    int dn = ++J->done_n;
    pthread_mutex_unlock(&J->mu);

    char suffix[512] = "";
    if (*err) {
        char *ll = py_last_line(err);
        size_t hb = head_chars(ll, strlen(ll), 80);
        snprintf(suffix, sizeof suffix, " (%.*s)", (int)hb, ll);
        free(ll);
    }
    logln("[%d/%zu] %s gen=%.0fs %s%s", dn, J->rows.n, tid, dt, ok ? "PASS" : "FAIL", suffix);

    free(text); free(raw); free(err); free(comp.p); sb_free(&gp);
}

static void *worker_main(void *p) {
    job_t *J = p;
    while (1) {
        pthread_mutex_lock(&J->mu);
        size_t k = J->next++;
        pthread_mutex_unlock(&J->mu);
        if (k >= J->rows.n) break;
        one_task(J, k);
    }
    return NULL;
}

/* Python 切片 rows[off : off+limit] 语义(含负下标折返与钳位) */
static rows_t py_slice(rows_t r, int off, int lim) {
    long n = (long)r.n;
    long a = off, b = (long)off + (long)lim;
    if (a < 0) { a += n; if (a < 0) a = 0; }
    if (a > n) a = n;
    if (b < 0) { b += n; if (b < 0) b = 0; }
    if (b > n) b = n;
    if (b < a) b = a;
    rows_t o; o.row = r.row + a; o.n = (size_t)(b - a);
    return o;
}

static void run_suite(const args_t *a) {
    job_t J; memset(&J, 0, sizeof J);
    pthread_mutex_init(&J.mu, NULL);
    J.a = a;
    int is_go = strcmp(a->suite, "humaneval") != 0;
    J.is_go = is_go;
    if (!is_go) { J.rows = dataset(0); J.stops = PY_STOP; J.lang = "python"; }
    else { J.rows = dataset(1); J.stops = GO_STOP; J.lang = "go"; }
    J.rows = py_slice(J.rows, a->offset, a->limit);
    makedirs(a->out_dir);
    char out_path[4096];
    snprintf(out_path, sizeof out_path, "%s/pubbench_%s_%s.jsonl", a->out_dir, a->suite, a->tag);
    J.results = xmalloc((J.rows.n ? J.rows.n : 1) * sizeof *J.results);
    memset(J.results, 0, (J.rows.n ? J.rows.n : 1) * sizeof *J.results);

    /* ★并发(2026-08-18 用户令"测试并发不要串行")★: server 推理单 worker 串行, 但并发请求
     * 消掉 client 间隙+判题(go test 1-5s/题)与生成流水; 结果按题序落盘。
     * ThreadPoolExecutor.map 的语义 = 任务按序入队 FIFO, 结果按序取回。*/
    if (a->jobs <= 0) die("ValueError: max_workers must be greater than 0");
    int nth = a->jobs;
    if ((size_t)nth > J.rows.n) nth = (int)J.rows.n;
    if (nth < 1) nth = 1;
    pthread_t *th = xmalloc((size_t)nth * sizeof *th);
    for (int t = 0; t < nth; t++) pthread_create(&th[t], NULL, worker_main, &J);
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);

    FILE *out = fopen(out_path, "w");
    if (!out) die("OSError: [Errno %d] %s: '%s'", errno, strerror(errno), out_path);
    int n_pass = 0;
    for (size_t k = 0; k < J.rows.n; k++) {
        jv *rec = J.results[k];
        jv *pv = jget(rec, "pass");
        n_pass += (pv && pv->b) ? 1 : 0;
        sb_t line = {0};
        json_dump(&line, rec, 0);                       /* ensure_ascii=False */
        fwrite(line.p, 1, line.n, out); fputc('\n', out);
        sb_free(&line);
    }
    fclose(out);
    const char *note = is_go ? " (go judge=experimental, 人工抽查后作数)" : "";
    logln("[done] %s tag=%s pass@1 = %d/%zu%s", a->suite, a->tag, n_pass, J.rows.n, note);
    logln("[raw] %s", out_path);
    jv *sum = jnew(JOBJ);
    jpush(sum, xstrdup("suite"), jstrv(a->suite, strlen(a->suite)));
    jpush(sum, xstrdup("tag"), jstrv(a->tag, strlen(a->tag)));
    jpush(sum, xstrdup("pass"), jintv(n_pass));
    jpush(sum, xstrdup("total"), jintv((long long)J.rows.n));
    jpush(sum, xstrdup("raw"), jstrv(out_path, strlen(out_path)));
    sb_t line = {0};
    json_dump(&line, sum, 1);                           /* print(json.dumps(...)) = ensure_ascii 默认 True */
    fwrite(line.p, 1, line.n, stdout); fputc('\n', stdout);
    sb_free(&line); jfree(sum);
    free(th); free(J.results);
}

/* ============================ rejudge ============================ */

typedef struct {
    rows_t rows;
    const char *api;
    size_t next;
    int n_pass;
    pthread_mutex_t mu;
} rej_t;

static void rejudge_one(rej_t *R, size_t idx) {
    jv *r = R->rows.row[idx];
    const char *lang = jstr_req(r, "lang");
    int is_go = strcmp(lang, "python") != 0;
    const char *const *stops = is_go ? GO_STOP : PY_STOP;

    const char *prompt = jstr_req(r, "prompt");
    const char *test = jstr(r, "test", "");
    const char *entry = jstr(r, "entry_point", "");
    const char *setup = "", *imp = "";
    /* 数据集字段(test/entry_point)不在 jsonl 里 → 从数据集按 task_id 回填 */
    if (!*test) {
        rows_t ds = dataset(is_go);
        const char *tid = jstr_req(r, "task_id");
        jv *full = NULL;
        for (size_t i = 0; i < ds.n; i++) {
            const char *t = jstr(ds.row[i], "task_id", NULL);
            if (t && !strcmp(t, tid)) full = ds.row[i];   /* dict 语义: 后者覆盖 */
        }
        if (!full) die("KeyError: '%s'", tid);
        test = jstr_req(full, "test");
        entry = jstr(full, "entry_point", "");
        setup = jstr(full, "test_setup", "");
        imp = jstr(full, "import", "");
    }
    const char *rtext = jstr_req(r, "response_text");
    str_t comp = extract_completion(rtext, prompt, stops, strcmp(R->api, "completions") != 0, is_go);
    row_t row = { prompt, test, entry, setup, imp };
    judge_t j = is_go ? eval_go(&row, &comp, 15) : eval_python(&row, &comp, 15);

    jv *pv = jget(r, "pass");
    int was = (pv && pv->type == JBOOL) ? pv->b : 0;
    char changed[64] = "";
    if (j.ok != was) snprintf(changed, sizeof changed, "  [改判 %s→%s]", was ? "True" : "False", j.ok ? "True" : "False");
    char suffix[512] = "";
    if (*j.err) {
        char *ll = py_last_line(j.err);
        size_t hb = head_chars(ll, strlen(ll), 70);
        snprintf(suffix, sizeof suffix, " (%.*s)", (int)hb, ll);
        free(ll);
    }
    logln("%s: %s%s%s", jstr_req(r, "task_id"), j.ok ? "PASS" : "FAIL", changed, suffix);

    jset(r, "pass", jboolv(j.ok));
    jset(r, "err", jstrv(j.err, strlen(j.err)));
    jset(r, "completion", jstrv(comp.p, comp.n));

    pthread_mutex_lock(&R->mu);
    R->n_pass += j.ok ? 1 : 0;
    pthread_mutex_unlock(&R->mu);
    free(j.err); free(comp.p);
}
static void *rejudge_worker(void *p) {
    rej_t *R = p;
    while (1) {
        pthread_mutex_lock(&R->mu);
        size_t k = R->next++;
        pthread_mutex_unlock(&R->mu);
        if (k >= R->rows.n) break;
        rejudge_one(R, k);
    }
    return NULL;
}

/* 离线重判: 用已存 response_text 重新抽取+评测(修抽取器后免重生成), 原地重写 pass/completion */
static void rejudge(const char *path, const char *api) {
    rej_t R; memset(&R, 0, sizeof R);
    pthread_mutex_init(&R.mu, NULL);
    R.rows = load_jsonl(path);
    R.api = api;
    int nth = 8;                                        /* 判题并发(用户令 08-18) */
    if ((size_t)nth > R.rows.n) nth = (int)R.rows.n;
    if (nth < 1) nth = 1;
    pthread_t th[8];
    for (int t = 0; t < nth; t++) pthread_create(&th[t], NULL, rejudge_worker, &R);
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);

    FILE *f = fopen(path, "w");
    if (!f) die("OSError: [Errno %d] %s: '%s'", errno, strerror(errno), path);
    for (size_t i = 0; i < R.rows.n; i++) {
        sb_t line = {0};
        json_dump(&line, R.rows.row[i], 0);
        fwrite(line.p, 1, line.n, f); fputc('\n', f);
        sb_free(&line);
    }
    fclose(f);
    logln("[rejudge] %s: pass@1 = %d/%zu", path, R.n_pass, R.rows.n);
}

/* ============================ verdict / compare ============================ */

/* 两个互相独立的决策闸。阈值是"决策带"不是测量值: 粗到经得起 n=20 的 ±2 题噪声,
 * 编码的是 2026-07-25 对话已定的逻辑(fable5.md): 参赛的标的=可运行作品+诚实方法论;
 * 买机的标的=产品阶段(驻留/MTP/KV/日用速度)是否真实存在。不是大赛评审标准。*/
static void verdict(int B, int Q, int N, const char *base_tag, const char *quant_tag) {
    printf("\n");
    printf("== 判决闸门: 基线=%s B=%d/%d | 量化=%s Q=%d/%d | n=%d 噪声底≈±2题 ==\n",
           base_tag, B, N, quant_tag, Q, N, N);
    if ((double)B < 0.6 * N) {
        printf("[INVALID] 基线 %d/%d < 60%% — q2 基线编程域不该这么低; 先查 harness/服务层"
               "(历史教训: 服务层bug曾把好引擎打成零代码), 本轮分数不作任何决策依据\n", B, N);
        return;
    }
    if (B == 0) die("ZeroDivisionError: division by zero");
    double ret = (double)Q / (double)B;
    /* 故事口径(用户裁决 07-25): 消费级单机+量化模型+直接可用编程。双机组网/慢速运行不构成故事。
     * 阈值=Claude 的专业判断(07-25 晚定稿, 责任署名, 错误可检验形态在 fable5.md 当日条目):
     * 锚=竞争替代 — 64G Mac 上人人五分钟可跑 Qwen3-coder 30B 级免费模型, 故事必须明显打赢它;
     * 50% 保留(≈CodeLlama-13B 档)打不赢, 保留七成的 frontier 级 MoE + 1M ctx + 自研引擎才成立。*/
    const char *g1, *g2;
    if ((double)Q >= 0.6 * N && ret >= 0.7)
        g1 = "GO — 故事成立: 单机可用速度实录为 Demo; 现役双机只作开发素材, 购机为截稿(08-16)前关键路径";
    else if ((double)Q >= 0.5 * N && ret >= 0.5)
        g1 = "BORDER — 打不赢'懒人替代'(64G 免费 30B coder), quant 故事不达标; fallback=引擎+官方q2单机(96G)故事存在但评级偏弱, 默认弃本届";
    else
        g1 = "NO — 质量主张撑不住, 什么机器都救不了故事; 回算法迭代";
    if ((double)Q >= 0.6 * N && ret >= 0.7)
        g2 = "GO — 买 96G M4 Max(非 64G: 多的几千块买'量化 vs 官方q2 同机对照演示'位+日常 parity 参照; 按 9.5t/s@120GB/s 线性外推 546GB/s≈30-40t/s, 到手 24h ds4-bench 实测替换)";
    else
        g2 = "不买本周期 — 竞争锚下故事不成立, 机器无标的; 下个质量里程碑再判";
    printf("[GOAI 参赛闸] %s\n", g1);
    printf("[买机闸]      %s\n", g2);
    printf("[口径] 质量闸先行, 两闸都由质量分决定; GO 态下购机档位服务'消费级单机直接编程'故事; "
           "报名前读 goaihz.com 章程 (三赛道截止 2026-08-16, 具身 08-20)\n");
    if (!strncmp(g1, "GO", 2) || !strncmp(g2, "GO", 2))
        printf("[下单前加固] GO≠下单, 动钱前三点全过: "
               "① LIMIT=164 跑满全套收窄置信区间(n=20 delta CI≈±31%% → n=164 ≈±11%%) "
               "② 第二条腿 gen_coding_probe 真实任务可用性过(HumanEval 是短函数补全, ≠真实编辑工作流) "
               "③ 新机到手 24h 内 ds4-bench 实测替换所有速度外推, 不达可用速度按残值退/售\n");
}

/* dict{task_id: rec}(保序, 同键后者覆盖) */
typedef struct { char **k; jv **v; size_t n, cap; } dict_t;
static void dict_put(dict_t *d, const char *k, jv *v) {
    for (size_t i = 0; i < d->n; i++) if (!strcmp(d->k[i], k)) { d->v[i] = v; return; }
    if (d->n == d->cap) { d->cap = d->cap ? d->cap * 2 : 32; d->k = xrealloc(d->k, d->cap * sizeof *d->k); d->v = xrealloc(d->v, d->cap * sizeof *d->v); }
    d->k[d->n] = xstrdup(k); d->v[d->n] = v; d->n++;
}
static jv *dict_get(const dict_t *d, const char *k) {
    for (size_t i = 0; i < d->n; i++) if (!strcmp(d->k[i], k)) return d->v[i];
    return NULL;
}
static int cmp_str(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

static void load_dict(const char *path, dict_t *d) {
    rows_t r = load_jsonl(path);
    for (size_t i = 0; i < r.n; i++) dict_put(d, jstr_req(r.row[i], "task_id"), r.row[i]);
}

static void compare(const char *fa, const char *fb) {
    dict_t A = {0}, B = {0};
    load_dict(fa, &A); load_dict(fb, &B);
    if (!A.n || !B.n) die("StopIteration");
    const char *ka = jstr_req(A.v[0], "tag"), *kb = jstr_req(B.v[0], "tag");
    /* both = sorted(set(A) & set(B)) —— Python 字符串序 = UTF-8 字节序 */
    char **both = xmalloc((A.n ? A.n : 1) * sizeof *both);
    size_t nb = 0;
    for (size_t i = 0; i < A.n; i++) if (dict_get(&B, A.k[i])) both[nb++] = A.k[i];
    qsort(both, nb, sizeof *both, cmp_str);
    int pa = 0, pb = 0;
    for (size_t i = 0; i < nb; i++) {
        jv *x = jget(dict_get(&A, both[i]), "pass"), *y = jget(dict_get(&B, both[i]), "pass");
        pa += (x && x->b) ? 1 : 0;
        pb += (y && y->b) ? 1 : 0;
    }
    printf("(顺序约定: A=基线=%s, B=量化=%s)\n", ka, kb);
    printf("%-18s %8s %8s\n", "task_id", ka, kb);
    for (size_t i = 0; i < nb; i++) {
        jv *x = jget(dict_get(&A, both[i]), "pass"), *y = jget(dict_get(&B, both[i]), "pass");
        printf("%-18s %8s %8s\n", both[i], (x && x->b) ? "PASS" : ".", (y && y->b) ? "PASS" : ".");
    }
    printf("%-18s %5d/%zu %5d/%zu   delta(%s-%s) = %+d\n", "TOTAL", pa, nb, pb, nb, kb, ka, pb - pa);
    verdict(pa, pb, (int)nb, ka, kb);
    free(both);
}

/* ============================ selftest ============================ */

/* harness 自检: 拿数据集自带的 canonical_solution 冒充模型输出, 走与真跑完全同一条
 * 抽取+判定链路。**期望 164/164** —— 标准答案判不过 = harness 坏了(缺 go 工具链/
 * 抽取器误剥/拼装规则跑偏), 与被测模型的质量无关。换机器/换 Go 版本后先跑这个,
 * 免得把环境问题算到模型头上。*/
static int selftest(const char *suite, int limit, int offset, int eval_timeout) {
    int is_go = strcmp(suite, "humaneval") != 0;
    rows_t rows = py_slice(dataset(is_go), offset, limit);
    const char *lang = is_go ? "go" : "python";
    (void)lang;
    int n_pass = 0;
    slist failed = {0};
    double t0 = now_s();
    for (size_t k = 0; k < rows.n; k++) {
        jv *row = rows.row[k];
        char tidbuf[128];
        const char *tid = jstr(row, "task_id", NULL);
        if (!tid) { snprintf(tidbuf, sizeof tidbuf, "%s/%d", suite, offset + (int)k); tid = tidbuf; }
        /* canonical_solution 就是"续写部分", 与 completions 裸续写口径同形, 直接当 completion */
        const char *cs = jstr_req(row, "canonical_solution");
        str_t comp = str_cstr(cs);
        row_t r = { jstr_req(row, "prompt"), jstr(row, "test", NULL), jstr(row, "entry_point", NULL),
                    jstr(row, "test_setup", ""), jstr(row, "import", "") };
        if (!r.test) die("KeyError: 'test'");
        if (!is_go && !r.entry_point) die("KeyError: 'entry_point'");
        judge_t j = is_go ? eval_go(&r, &comp, eval_timeout) : eval_python(&r, &comp, eval_timeout);
        n_pass += j.ok ? 1 : 0;
        if (!j.ok) {
            sl_push(&failed, xstrdup(tid));
            char *ll = py_last_line(j.err);
            size_t hb = head_chars(ll, strlen(ll), 100);
            logln("%s: FAIL  (%.*s)", tid, (int)hb, ll);
            free(ll);
        } else if ((k + 1) % 20 == 0) {
            logln("[selftest] %zu/%zu ... %d pass", k + 1, rows.n, n_pass);
        }
        free(j.err); free(comp.p);
    }
    logln("[selftest] %s: %d/%zu 标准答案通过 (%.0fs)", suite, n_pass, rows.n, now_s() - t0);
    if (failed.n) {
        sb_t s = {0};
        for (size_t i = 0; i < failed.n; i++) { if (i) sb_ch(&s, ' '); sb_puts(&s, failed.v[i]); }
        logln("[selftest] 未通过: %s", s.p ? s.p : "");
        logln("[selftest] ★harness 有问题★ — 标准答案本应全过, 先修环境/判定器再跑模型");
        sb_free(&s);
    }
    sl_free(&failed);
    return n_pass == (int)rows.n;
}

/* ============================ main / argparse ============================ */

static const char *USAGE =
    "usage: pubbench [-h] [--suite {humaneval,humaneval-x-go}] [--url URL] [--tag TAG]\n"
    "                [--limit LIMIT] [--offset OFFSET] [--max-tokens MAX_TOKENS]\n"
    "                [--mode MODE] [--api {chat,completions}]\n"
    "                [--http-timeout HTTP_TIMEOUT] [--jobs JOBS]\n"
    "                [--eval-timeout EVAL_TIMEOUT] [--out-dir OUT_DIR]\n"
    "                [--compare A.jsonl B.jsonl] [--rejudge FILE.jsonl] [--selftest]\n";

static void ap_err(const char *fmt, ...) {
    char b[512];
    va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    fputs(USAGE, stderr);
    fprintf(stderr, "pubbench: error: %s\n", b);
    exit(2);
}
static int ap_int(const char *opt, const char *s) {
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || *end) ap_err("argument %s: invalid int value: '%s'", opt, s);
    return (int)v;
}
static int env_int(const char *name, const char *def) {
    const char *s = getenv(name);
    if (!s) s = def;
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || *end) die("ValueError: invalid literal for int() with base 10: '%s'", s);
    return (int)v;
}
static const char *env_str(const char *name, const char *def) { const char *s = getenv(name); return s ? s : def; }

/* .py 的 out-dir 默认 = os.path.dirname(os.path.abspath(__file__)) + "/../reports/pubbench"
 * (未规范化, 打印时带 "..")。C 版同形, 但 dirname 是可执行文件所在目录(calib/ 而非
 * scripts/) —— 落地目录同一个, 打印出来的路径少一处目录名差异, 见转录清单 B 组。*/
static const char *default_out_dir(void) {
    static char buf[4096], dir[4096];
    buf[0] = 0;
#ifdef __APPLE__
    uint32_t sz = sizeof buf;
    if (_NSGetExecutablePath(buf, &sz) != 0) buf[0] = 0;
#else
    ssize_t k = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (k > 0) buf[k] = 0; else buf[0] = 0;
#endif
    if (!buf[0]) { snprintf(dir, sizeof dir, "../reports/pubbench"); return dir; }
    char *slash = strrchr(buf, '/');
    if (slash) *slash = 0;
    snprintf(dir, sizeof dir, "%s/../reports/pubbench", buf);
    return dir;
}

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    signal(SIGPIPE, SIG_IGN);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CACHE = env_str("PUBBENCH_CACHE", "/tmp/pubbench_cache");

    args_t a;
    a.suite = "humaneval";
    a.url = env_str("DS4_URL", "http://127.0.0.1:8080");
    a.tag = "run";                                      /* 例: base_q2 / vq14 */
    a.limit = 20;
    a.offset = 0;
    a.max_tokens = 320;
    a.mode = "code";                                    /* ds4 server mode:code */
    a.api = env_str("PUBBENCH_API", "chat");            /* completions=BASE 裸续写口径 */
    a.http_timeout = 900;                               /* 慢机 2-5 t/s 留足 */
    a.jobs = env_int("PUBBENCH_JOBS", "4");             /* 并发(生成+判题流水) */
    a.eval_timeout = 15;
    a.out_dir = env_str("PUBBENCH_OUT", default_out_dir());
    a.cmp_a = a.cmp_b = a.rejudge_path = NULL;
    a.selftest = 0;

    for (int i = 1; i < argc; i++) {
        char opt[64], *eq;
        const char *arg = argv[i];
        snprintf(opt, sizeof opt, "%s", arg);
        const char *inline_val = NULL;
        if ((eq = strchr(opt, '=')) && !strncmp(opt, "--", 2)) { *eq = 0; inline_val = arg + (eq - opt) + 1; }
        #define NEXTV(name) (inline_val ? inline_val : (i + 1 < argc ? argv[++i] : (ap_err("argument %s: expected one argument", name), (char *)NULL)))
        if (!strcmp(opt, "-h") || !strcmp(opt, "--help")) { fputs(USAGE, stdout); return 0; }
        else if (!strcmp(opt, "--suite")) {
            a.suite = NEXTV("--suite");
            if (strcmp(a.suite, "humaneval") && strcmp(a.suite, "humaneval-x-go"))
                ap_err("argument --suite: invalid choice: '%s' (choose from 'humaneval', 'humaneval-x-go')", a.suite);
        }
        else if (!strcmp(opt, "--url")) a.url = NEXTV("--url");
        else if (!strcmp(opt, "--tag")) a.tag = NEXTV("--tag");
        else if (!strcmp(opt, "--limit")) a.limit = ap_int("--limit", NEXTV("--limit"));
        else if (!strcmp(opt, "--offset")) a.offset = ap_int("--offset", NEXTV("--offset"));
        else if (!strcmp(opt, "--max-tokens")) a.max_tokens = ap_int("--max-tokens", NEXTV("--max-tokens"));
        else if (!strcmp(opt, "--mode")) a.mode = NEXTV("--mode");
        else if (!strcmp(opt, "--api")) {
            a.api = NEXTV("--api");
            if (strcmp(a.api, "chat") && strcmp(a.api, "completions"))
                ap_err("argument --api: invalid choice: '%s' (choose from 'chat', 'completions')", a.api);
        }
        else if (!strcmp(opt, "--http-timeout")) a.http_timeout = ap_int("--http-timeout", NEXTV("--http-timeout"));
        else if (!strcmp(opt, "--jobs")) a.jobs = ap_int("--jobs", NEXTV("--jobs"));
        else if (!strcmp(opt, "--eval-timeout")) a.eval_timeout = ap_int("--eval-timeout", NEXTV("--eval-timeout"));
        else if (!strcmp(opt, "--out-dir")) a.out_dir = NEXTV("--out-dir");
        else if (!strcmp(opt, "--compare")) {
            if (inline_val) ap_err("argument --compare: expected 2 arguments");
            if (i + 2 >= argc) ap_err("argument --compare: expected 2 arguments");
            a.cmp_a = argv[++i]; a.cmp_b = argv[++i];
        }
        else if (!strcmp(opt, "--rejudge")) a.rejudge_path = NEXTV("--rejudge");
        else if (!strcmp(opt, "--selftest")) a.selftest = 1;
        else ap_err("unrecognized arguments: %s", arg);
        #undef NEXTV
    }

    if (a.selftest) return selftest(a.suite, a.limit, a.offset, a.eval_timeout) ? 0 : 1;
    if (a.rejudge_path) { rejudge(a.rejudge_path, a.api); return 0; }
    if (a.cmp_a) compare(a.cmp_a, a.cmp_b);
    else run_suite(&a);
    return 0;
}
