/* core_ptrain_ids.c — ids 行(2026-11-02): {"ids": [t0, t1, …]} 一行 = 一条裸 token 流当原文段, 第 0 个 id 顶 BOS 的位置, 从第 1 个起每个都是目标。
 *
 * 为什么要它: 判决尺 / 反修拟合料是 token 流(gguf/go-onebit/code/fit.ids 这种, 30 种语言各几个 128 位窗拼起来, 窗与窗之间是上下文断点),
 *   FP 教师锚(v41judge/teacher_*.bin)就是对这串原样算的。V 步要拿它当"编程不许掉"的那份梯度: 回成文本再分词在窗界会合并出别的 token(11-02 实撞: 15359 位回编码 15356 位,
 *   首个不同位 3582 在 "}\n\nFROM" 这种断点), 教师表就对不上; 直接喂 ids 才能和锚逐位同。
 * 留出: 按 ids 串的十进制文本做 FNV 五取一(与 text 行同一规则; 给了 eval= 时料不留出); src 2(eval= 文件)= 泛化留出。
 * 出错会怎样: 不是只有 "ids" 一个键的对象 ⇒ 不认(返回 0, 调用方按普通料行解析报它自己的错); 元素不是非负整数 / 少于 2 个 ⇒ 认是 ids 行但解析失败, 整份料停。 */
#include "core_ptrain.h"
#include "../common/ds4_json.h"
#ifndef DS4_NO_GPU

/* 只认 {"ids":[…]}(可有空白)。返回: 1 = 是 ids 行且解析成功(ids/n 为 malloc), 0 = 不是 ids 行, 失败打印后 exit —— 料里一行坏了不许静默跳过 */
bool pt_parse_ids_line(const char *ln, int32_t **ids, uint32_t *n) {
    const char *p = ln; json_ws(&p);
    if (*p != '{') return false;
    p++; json_ws(&p);
    char *key = NULL;
    if (!json_string(&p, &key)) return false;
    const bool is_ids = !strcmp(key, "ids"); free(key);
    if (!is_ids) return false;
    json_ws(&p);
    if (*p != ':') return false;
    p++; json_ws(&p);
    if (*p != '[') { fprintf(stderr, "ds4: --ptrain ids 行: ids 要是数组\n"); exit(1); }
    p++;
    uint32_t cap = 4096, k = 0; int32_t *v = xmalloc((size_t)cap * 4);
    for (;;) {
        json_ws(&p);
        if (*p == ']') { p++; break; }
        if (*p < '0' || *p > '9') { fprintf(stderr, "ds4: --ptrain ids 行: 第 %u 个元素不是非负整数\n", k); exit(1); }
        long x = 0; while (*p >= '0' && *p <= '9') { x = x * 10 + (*p - '0'); if (x > 0x7fffffffL) { fprintf(stderr, "ds4: --ptrain ids 行: 元素过大\n"); exit(1); } p++; }
        if (k == cap) { cap *= 2u; v = realloc(v, (size_t)cap * 4); if (!v) { fprintf(stderr, "ds4: --ptrain ids 行: 内存\n"); exit(1); } }
        v[k++] = (int32_t)x;
        json_ws(&p);
        if (*p == ',') p++;
    }
    json_ws(&p);
    if (*p != '}') { fprintf(stderr, "ds4: --ptrain ids 行: 只认 ids 一个键\n"); exit(1); }
    p++; json_ws(&p);
    if (*p) { fprintf(stderr, "ds4: --ptrain ids 行: 行尾有多余内容\n"); exit(1); }
    if (k < 2) { fprintf(stderr, "ds4: --ptrain ids 行: 至少 2 个 id\n"); exit(1); }
    *ids = v; *n = k;
    return true;
}

bool pt_add_ids(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, const int32_t *ids, uint32_t n) {
    int eval = 0;
    if (src == 2) eval = 2;
    else if (!(src == 0 && c->evalf[0] && !c->data_eval)) {   /* 留出键 = ids 的十进制串(与 text 行同一个 FNV 五取一) */
        char *s = xmalloc((size_t)n * 12u + 1u); size_t o = 0;
        for (uint32_t i = 0; i < n; i++) o += (size_t)snprintf(s + o, 12u, "%d ", ids[i]);
        eval = pt_is_eval(s); free(s);
    }
    return pt_add_tokens(e, c, d, src, eval, ids[0], ids + 1, n - 1u, 0);
}
#endif /* !DS4_NO_GPU */
