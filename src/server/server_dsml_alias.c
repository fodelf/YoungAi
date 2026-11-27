/* server_dsml_alias.c — DSML 标签别名: 模型在某些位置把 DSML 标签写成裸 XML(<parameter …>、</parameter>、</invoke>),
 * 意图没有歧义的认下来。最终解析、KV 语法跟踪、OpenAI/Anthropic 两条流式投影都经这里判标签, 别直接拿 syn->param_* / syn->invoke_end 比 ——
 * 两处各认各的, 就是 10-16 那种"一边当调用、一边当正文"。模型输出一个字节不改, 精确重放表里存的仍是原字节。 */

#include "server_internal.h"

/* 参数开标签认两种: 本写法的, 或裸 "<parameter "。2026-10-11 实撞(tests/server_tests_bare_param.c):
 * system 正文 + 官方工具说明的上下文里, 量化模型在参数开标签位的首选是裸 "<", 写成 <parameter name="x">值</｜DSML｜ parameter>
 * (闭标签照写 DSML 的、不带 string 属性)。同一请求: 助理的中文 system 5/5 这样写, 一句英文 2/5, 不带 system 0/5 ——
 * 渲染与官方 encoding.py 逐字节同, 是模型在这一位的选择; 早上的"语法位贪心"救不了(argmax 就是 "<")。
 * 意图没有歧义(在 invoke 块里、有 name、闭标签是 DSML 的), 解析侧认下来; 模型输出一个字节不改, 精确重放表里存的仍是原字节。
 * 别名带尾空格: "<parameters>" 之类不是开标签。纯 <parameter> 写法(DSML_SYN_PLAIN)本身就是它, 不叠别名。 */
#define DSML_BARE_PARAM_START "<parameter "
static bool dsml_has_bare_alias(const dsml_syntax *syn) { return strcmp(syn->param_start, "<parameter") != 0; }

size_t dsml_param_open_len(const dsml_syntax *syn, const char *p) {
    const size_t n = strlen(syn->param_start);
    if (!strncmp(p, syn->param_start, n)) return n;
    if (dsml_has_bare_alias(syn) && !strncmp(p, DSML_BARE_PARAM_START, strlen(DSML_BARE_PARAM_START))) return strlen(DSML_BARE_PARAM_START);
    return 0;
}

bool dsml_param_open_full(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    return raw_full_lit(raw, raw_len, pos, syn->param_start) ||
           (dsml_has_bare_alias(syn) && raw_full_lit(raw, raw_len, pos, DSML_BARE_PARAM_START));
}

bool dsml_param_open_partial(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    return raw_partial_lit(raw, raw_len, pos, syn->param_start) ||
           (dsml_has_bare_alias(syn) && raw_partial_lit(raw, raw_len, pos, DSML_BARE_PARAM_START));
}

/* 闭标签同样认裸 "</parameter>"(2026-10-11 深夜实撞, 助理查 DGX Spark 价: 开标签裸了之后闭标签也裸, 一块里两个 invoke 都这样,
 * invoke/calls 标签仍是 DSML 的)。只认 DSML 闭标签的代价: 最终解析找不到闭标签 ⇒ 整块退回正文; 流式把后面整段(含第二个 invoke)
 * 当第一个参数的值推给客户端, 助理拿着带标签的 query 连撞 5 步。三条路(最终解析 / 流式 / KV 语法跟踪)都经这三个函数找闭标签。 */
#define DSML_BARE_PARAM_END "</parameter>"

size_t dsml_param_close_len(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    if (raw_full_lit(raw, raw_len, pos, syn->param_end)) return strlen(syn->param_end);
    if (dsml_has_bare_alias(syn) && raw_full_lit(raw, raw_len, pos, DSML_BARE_PARAM_END)) return strlen(DSML_BARE_PARAM_END);
    return 0;
}

/* 缓冲从 pos 起到末尾是某个闭标签的真前缀(至少 min_len 字节): 流式在这里扣住不发, 等下一块 */
bool dsml_param_close_partial(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos, size_t min_len) {
    if (!raw || pos > raw_len || raw_len - pos < min_len) return false;
    return raw_partial_lit(raw, raw_len, pos, syn->param_end) ||
           (dsml_has_bare_alias(syn) && raw_partial_lit(raw, raw_len, pos, DSML_BARE_PARAM_END)) ||
           dsml_calls_end_partial(syn, raw, raw_len, pos);   /* 块闭合也可能直接关掉值(隐式闭合, 见下), 尾巴像它的开头同样扣住 */
}

#define DSML_BARE_INVOKE_END "</invoke>"
static bool dsml_has_bare_invoke_alias(const dsml_syntax *syn) { return strcmp(syn->invoke_end, DSML_BARE_INVOKE_END) != 0; }

/* ★隐式闭合★(2026-10-18 17:22 实撞, trace serve_20261018_171753 r2, 1017 底座): 模型写完 write_file 的 content 值后直接写
 * `</html></｜DSML｜ calls>` 就 EOS —— 参数闭标签和 invoke 闭标签都没写。官方解析器同样报错, 但意图没有歧义: 块闭合把还开着的
 * 参数和 invoke 一起关掉。不认的代价: 10.6k 思考 + 2.5k 文件体整段退回正文, 用户看着调用"打印出来了却没执行"。
 * 只认 "值 → </calls>" 这一种形状: 值后面先出现 invoke 闭标签再出现块闭合(参数没关、invoke 关了)仍是写坏, 不认
 * (tests/server_tests_tool_agree.c "unclosed param" 要的就是报错)。闭合长度 0: 调用方的游标停在块闭合上, 外层循环自己收块。 */
bool dsml_calls_end_at(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    return raw_full_lit(raw, raw_len, pos, syn->tool_calls_end);
}
bool dsml_calls_end_partial(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    return raw_partial_lit(raw, raw_len, pos, syn->tool_calls_end);
}

/* ★伪 XML 标签(2026-11-26 实撞)★: 模型在 DSML 块里不写 canonical invoke/parameter,
 * 而是写工具名标签和参数名标签: <lightpanda_search">/<query>/</query>/</lightpanda_search>。
 * 意图没有歧义(块起点是 DSML 的, 工具名标签在 invoke 位, 参数名标签在 parameter 位,
 * 闭标签与开标签同名), 解析侧认下来; 模型输出一个字节不改, 精确重放表里存的仍是原字节。
 * 只认"名字后直接 >"或"名字后一个多余 " 再 >"的形状(实撞 <lightpanda_search"> 就是后者);
 * <oops/> 与 <oops name="..."> 这类带属性/自闭合的标签不认 —— 那是写坏, 不是伪 DSML。
 * 三条路(最终解析 / 流式投影 / KV 语法跟踪)都经这里判标签, 规则若再分叉, 收尾一致性检查会报 error。 */
static bool tag_name_ascii(const char *raw, size_t raw_len, size_t i, char *name, size_t cap) {
    size_t n = 0;
    while (i < raw_len && n + 1 < cap &&
           (isalnum((unsigned char)raw[i]) || raw[i] == '_' || raw[i] == '-')) {
        name[n++] = raw[i++];
    }
    if (n == 0) return false;
    name[n] = 0;
    return true;
}

bool dsml_tag_open_at(const char *raw, size_t raw_len, size_t pos,
                      char *name, size_t name_cap, size_t *end_out) {
    if (!raw || pos >= raw_len || raw[pos] != '<') return false;
    if (pos + 1 < raw_len && raw[pos + 1] == '/') return false;
    size_t i = pos + 1;
    if (!tag_name_ascii(raw, raw_len, i, name, name_cap)) return false;
    while (i < raw_len && (isalnum((unsigned char)raw[i]) || raw[i] == '_' || raw[i] == '-')) i++;
    if (i < raw_len && raw[i] == '>') { if (end_out) *end_out = i + 1; return true; }
    if (i + 1 < raw_len && raw[i] == '"' && raw[i + 1] == '>') { if (end_out) *end_out = i + 2; return true; }
    return false;
}

bool dsml_tag_close_at(const char *raw, size_t raw_len, size_t pos,
                       char *name, size_t name_cap, size_t *end_out) {
    if (!raw || pos >= raw_len || raw[pos] != '<') return false;
    if (pos + 1 >= raw_len || raw[pos + 1] != '/') return false;
    size_t i = pos + 2;
    if (!tag_name_ascii(raw, raw_len, i, name, name_cap)) return false;
    while (i < raw_len && (isalnum((unsigned char)raw[i]) || raw[i] == '_' || raw[i] == '-')) i++;
    if (i < raw_len && raw[i] == '>') { if (end_out) *end_out = i + 1; return true; }
    return false;
}

/* 伪标签还没写完(缓冲从 pos 起到末尾是 <名字 或 <名字" 或 </名字 的前半截): 流式在这里扣住不发, 等下一块。
 * 只在“没有 >”时成立: 一个完整但写坏的标签(<oops/>、带属性)由三条路各自判不认, 不归这里扣。 */
bool dsml_tag_open_partial(const char *raw, size_t raw_len, size_t pos) {
    if (!raw || pos >= raw_len || raw[pos] != '<' || raw[pos + 1] == '/') return false;
    return memchr(raw + pos, '>', raw_len - pos) == NULL;
}

bool dsml_tag_close_partial(const char *raw, size_t raw_len, size_t pos) {
    if (!raw || pos >= raw_len || raw[pos] != '<' || pos + 1 >= raw_len || raw[pos + 1] != '/') return false;
    return memchr(raw + pos, '>', raw_len - pos) == NULL;
}

const char *dsml_tag_close_find(const char *s, size_t n, const char *name, size_t *len_out) {
    if (!s || !name || !name[0]) return NULL;
    for (size_t i = 0; i < n; i++) {
        if (s[i] != '<') continue;
        char got[64]; size_t end = 0;
        if (dsml_tag_close_at(s, n, i, got, sizeof got, &end) && !strcmp(got, name)) {
            if (len_out) *len_out = end - i;
            return s + i;
        }
    }
    return NULL;
}
static const char *dsml_implicit_close_find(const dsml_syntax *syn, const char *s, size_t n) {
    const char *c = find_lit_bounded(s, n, syn->tool_calls_end);
    if (!c) return NULL;
    const size_t m = (size_t)(c - s);
    if (find_lit_bounded(s, m, syn->invoke_end)) return NULL;
    if (dsml_has_bare_invoke_alias(syn) && find_lit_bounded(s, m, DSML_BARE_INVOKE_END)) return NULL;
    return c;
}

/* invoke 闭标签同样认裸 "</invoke>"(2026-10-16 实撞, 助理"故意不带参数调 read_file": 开标签是 DSML 的, 闭标签写成裸 </invoke>,
 * calls 闭标签仍是 DSML 的)。在参数之间的位置出现, 意图没有歧义。不认的代价: 最终解析整块退回正文, 而流式投影已经把
 * 调用头和 "{" 推给了客户端 —— 客户端拿到一个参数为半截 "{" 的调用(trace request 356)。最终解析、KV 语法跟踪、两条流式投影都经这两个函数; 规则若再分叉, 收尾的一致性检查会报 error(tests/server_tests_tool_agree.c)。 */

size_t dsml_invoke_close_len(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    if (raw_full_lit(raw, raw_len, pos, syn->invoke_end)) return strlen(syn->invoke_end);
    if (dsml_has_bare_invoke_alias(syn) && raw_full_lit(raw, raw_len, pos, DSML_BARE_INVOKE_END)) return strlen(DSML_BARE_INVOKE_END);
    return 0;
}

bool dsml_invoke_close_partial(const dsml_syntax *syn, const char *raw, size_t raw_len, size_t pos) {
    return raw_partial_lit(raw, raw_len, pos, syn->invoke_end) ||
           (dsml_has_bare_invoke_alias(syn) && raw_partial_lit(raw, raw_len, pos, DSML_BARE_INVOKE_END));
}

/* [s, s+n) 内最早出现的闭标签(两种写法取靠前的), *len_out = 命中的那个的长度; 更早出现的隐式闭合(块闭合直接接值)长度 0 */
const char *dsml_param_close_find(const dsml_syntax *syn, const char *s, size_t n, size_t *len_out) {
    const char *a = find_lit_bounded(s, n, syn->param_end), *b = NULL;
    if (dsml_has_bare_alias(syn)) b = find_lit_bounded(s, a ? (size_t)(a - s) : n, DSML_BARE_PARAM_END);   /* 只在 a 之前找 */
    const char *r = b ? b : a;
    const char *c = dsml_implicit_close_find(syn, s, r ? (size_t)(r - s) : n);
    if (c) { if (len_out) *len_out = 0; return c; }
    if (len_out) *len_out = b ? strlen(DSML_BARE_PARAM_END) : (a ? strlen(syn->param_end) : 0);
    return r;
}
