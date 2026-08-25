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

