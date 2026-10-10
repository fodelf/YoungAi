/* core_ptrain_data.c — 后训练 ③ 的料: 配置、jsonl 读取、分词(2026-10-10 收口版; 总述见 core_ptrain.h)。
 *
 * 料 = jsonl, 一行一题: {"messages":[{"role":"user","content":"问"},{"role":"assistant","content":"答"}], "context":"材料(可省)"}
 *   或一行一段原文: {"text":"..."}(unsloth 的 continued pretraining 口径: 不套聊天模板, BOS + 原文 + EOS, 每个 token 都是目标)。
 *   messages 前面可以多一条 {"role":"system",...}(两边渲染都带); 别的字段一律报错 —— 10-09 那版收 reward/mode/split, 用户 10-10 判为跑偏, 不留口子。
 *   带 context = 上下文蒸馏(同一份 context 的题共一块, 教师前缀只预填一次); 不带 = 答案 one-hot(交叉熵), 不要教师。
 *   留出: 问题(text 行: 整段原文)去空白后的 FNV-1a 五取一(措辞没进训练); 保持料(cfg hold=)同规则, 只是教师 = 不看材料的部署态自己。
 * 渲染一律走部署同一个函数(ds4_encode_chat_prompt: BOS <｜User｜>正文 <｜Assistant｜></think>, 不思考):
 *   学生 = 问题;  教师 = 材料 + 空行 + 问题;  两边接同一串答案 token + EOS(答案单独分词, 两边逐 id 相同, 位置一一对齐)。
 * 出错会怎样: 一行坏了整趟停(打行号 + 原因), 不跳过 —— 静默跳过会让"料少了一半"看起来像"没学会"。 */
#include "core_ptrain.h"
#include "../common/ds4_json.h"
#ifndef DS4_NO_GPU

static char *pt_slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)n + 1);
    if (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    return b;
}
static void pt_rstrip(char *s) { size_t n = strlen(s); while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = 0; }

bool pt_cfg_load(pt_cfg *c, const char *path) {
    memset(c, 0, sizeof *c);
    c->layer_lo = 0; c->layer_hi = DS4_N_LAYER - 1u;   /* 缺省全层(10-01 子集 1 轮: 全层 −32% 对 L21-39 −26%) */
    c->rank = 64; c->topk = 64; c->epochs = 3; c->batch = 4; c->maxlen = 1024; c->probe_n = 6; c->probe_tok = 96; c->seed = 1; c->dbg_layer = -1;
    c->lr = 2e-4f; c->clip = 1.0f; c->init_std = 0.f;
    char *b = pt_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了配置 %s\n", path); return false; }
    for (char *ln = strtok(b, "\n"); ln; ln = strtok(NULL, "\n")) {
        while (*ln == ' ' || *ln == '\t') ln++;
        if (!*ln || *ln == '#') continue;
        char *eq = strchr(ln, '=');
        if (!eq) { fprintf(stderr, "ds4: --ptrain 配置行没有 '=': %s\n", ln); free(b); return false; }
        *eq = 0; char *k = ln, *v = eq + 1; pt_rstrip(k); pt_rstrip(v);
        while (*v == ' ') v++;
        if (!strcmp(k, "data")) snprintf(c->data, sizeof c->data, "%s", v);
        else if (!strcmp(k, "out")) snprintf(c->out, sizeof c->out, "%s", v);
        else if (!strcmp(k, "hold")) snprintf(c->hold, sizeof c->hold, "%s", v);
        else if (!strcmp(k, "init")) snprintf(c->init, sizeof c->init, "%s", v);
        else if (!strcmp(k, "layers")) { unsigned a = 0, z = 0; if (sscanf(v, "%u-%u", &a, &z) == 2) { c->layer_lo = a; c->layer_hi = z; } else c->layer_lo = c->layer_hi = (uint32_t)atoi(v); }
        else if (!strcmp(k, "rank")) c->rank = (uint32_t)atoi(v);
        else if (!strcmp(k, "topk")) c->topk = (uint32_t)atoi(v);
        else if (!strcmp(k, "epochs")) c->epochs = (uint32_t)atoi(v);
        else if (!strcmp(k, "batch")) c->batch = (uint32_t)atoi(v);
        else if (!strcmp(k, "maxlen")) c->maxlen = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_n")) c->probe_n = (uint32_t)atoi(v);
        else if (!strcmp(k, "probe_tok")) c->probe_tok = (uint32_t)atoi(v);
        else if (!strcmp(k, "seed")) c->seed = (uint32_t)atoi(v);
        else if (!strcmp(k, "gradcheck")) c->gradcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "prof")) c->prof = (uint32_t)atoi(v);
        else if (!strcmp(k, "rccheck")) c->rccheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "diag")) c->diag = (uint32_t)atoi(v);
        else if (!strcmp(k, "max_steps")) c->max_steps = (uint32_t)atoi(v);
        else if (!strcmp(k, "epoch_tok")) c->epoch_tok = (uint32_t)atoi(v);
        else if (!strcmp(k, "packcheck")) c->packcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "nopack")) c->nopack = (uint32_t)atoi(v);
        else if (!strcmp(k, "dbglayer")) c->dbg_layer = atoi(v);
        else if (!strcmp(k, "poison")) c->poison = (uint32_t)atoi(v);
        else if (!strcmp(k, "gclayers")) {
            for (char *q = v, *nx; *q && c->ngcl < 16u; q = nx) {
                c->gcl[c->ngcl] = (uint32_t)strtoul(q, &nx, 10);
                if (nx == q) break;
                c->ngcl++;
                while (*nx == '/' || *nx == ' ') nx++;
            }
        }
        else if (!strcmp(k, "hccheck")) {   /* 层号用 / 隔开(脚本的额外配置按逗号拆行, 这里不能用逗号) */
            for (char *q = v, *nx; *q && c->nhcl < 16u; q = nx) {
                c->hcl[c->nhcl] = (uint32_t)strtoul(q, &nx, 10);
                if (nx == q) break;
                c->hcmid[c->nhcl++] = *nx == 'm' ? 1u : 0u;
                if (*nx == 'm') nx++;
                while (*nx == '/' || *nx == ' ') nx++;
            }
        }
        else if (!strcmp(k, "lr")) c->lr = (float)atof(v);
        else if (!strcmp(k, "clip")) c->clip = (float)atof(v);
        else if (!strcmp(k, "init_std")) c->init_std = (float)atof(v);
        else { fprintf(stderr, "ds4: --ptrain 不认识的配置项 %s\n", k); free(b); return false; }
    }
    free(b);
    if (!c->data[0] || !c->out[0]) { fprintf(stderr, "ds4: --ptrain 配置缺 data= 或 out=\n"); return false; }
    if (c->layer_hi >= DS4_N_LAYER || c->layer_lo > c->layer_hi) { fprintf(stderr, "ds4: --ptrain layers 越界(%u-%u, 模型 %u 层)\n", c->layer_lo, c->layer_hi, DS4_N_LAYER); return false; }
    /* 反传从出口往下逐层推, 出口梯度直接进第 hi 层: hi 不是末层时中间那几层被跳过, 梯度是错的(不报错, 只是训歪) */
    if (c->layer_hi != DS4_N_LAYER - 1u) { fprintf(stderr, "ds4: --ptrain layers 必须到末层 L%u(给的是 %u-%u)\n", DS4_N_LAYER - 1u, c->layer_lo, c->layer_hi); return false; }
    if (!c->rank || c->topk < 1 || c->topk > 128 || !c->batch || c->maxlen < 16) { fprintf(stderr, "ds4: --ptrain rank/topk(1..128)/batch/maxlen 不合法\n"); return false; }
    return true;
}

/* 块: 同一份材料一块(名字 = 材料的 FNV, 教师前缀只预填一次); 保持料共一块(hold); 不带材料的普通题共一块(sft); 原文段共一块(text) */
static uint32_t pt_chunk_of(pt_data *d, const char *ctx, int hold, int raw) {
    char name[32];
    if (hold) snprintf(name, sizeof name, "hold");
    else if (raw) snprintf(name, sizeof name, "text");
    else if (!ctx[0]) snprintf(name, sizeof name, "sft");
    else {
        uint64_t h = 1469598103934665603ull;
        for (const unsigned char *p = (const unsigned char *)ctx; *p; p++) { h ^= *p; h *= 1099511628211ull; }
        snprintf(name, sizeof name, "c_%016llx", (unsigned long long)h);
    }
    for (uint32_t i = 0; i < d->nch; i++) if (!strcmp(d->ch[i].name, name)) return i;
    d->ch = realloc(d->ch, (d->nch + 1) * sizeof *d->ch);
    pt_chunk *c = &d->ch[d->nch];
    memset(c, 0, sizeof *c);
    c->name = strdup(name); c->text = strdup(ctx); c->hold = hold; c->sft = !hold && !ctx[0];
    return d->nch++;
}

static int32_t *pt_cat(const ds4_tokens *a, const ds4_tokens *b, uint32_t *n) {
    *n = (uint32_t)(a->len + b->len);
    int32_t *v = xmalloc((size_t)*n * 4);
    for (int i = 0; i < a->len; i++) v[i] = a->v[i];
    for (int i = 0; i < b->len; i++) v[a->len + i] = b->v[i];
    return v;
}

/* 留出 = 问题去空白后 FNV-1a 五取一: 同一道题不管出现在哪份料、哪一趟, 永远在同一边 */
static int pt_is_eval(const char *q) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char *p = (const unsigned char *)q; *p; p++) { if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') continue; h ^= *p; h *= 1099511628211ull; }
    return h % 5u == 0u;
}

/* one-hot 的表(目标 = 序列里答案位的 token 自己, 余量 0): 同一个 KL 核算出来就是 −log p(目标), 梯度 = 交叉熵的; 不进教师缓存 */
static void pt_fill_onehot(const pt_cfg *c, pt_sample *s) {
    const uint32_t K = c->topk, m = s->m;
    s->sft = 1;
    s->top_id = xmalloc((size_t)m * K * 4); s->top_p = xmalloc((size_t)m * K * 4); s->top_rest = xmalloc((size_t)m * 4);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t k = 0; k < K; k++) { s->top_id[i * K + k] = k ? -1 : s->sids[s->sa0 + i]; s->top_p[i * K + k] = k ? 0.f : 1.f; }
        s->top_rest[i] = 0.f;
    }
}

/* text 行: BOS + 原文 + EOS, 从第 1 个 token 起每个都是目标。超 maxlen 的按 token 切成连续几段(每段 BOS + ≤ maxlen−1 个 token, EOS 只跟末段),
 * 各段自成样本、同一边留出(按整段原文算 FNV) —— unsloth 是截到 max_seq_length 丢尾, 这里不丢: 料就是一篇文档一行, 丢尾 = 后半篇白传。 */
static bool pt_add_text(ds4_engine *e, const pt_cfg *c, pt_data *d, const char *txt) {
    const uint32_t ci = pt_chunk_of(d, "", 0, 1);
    const int eval = pt_is_eval(txt);
    ds4_tokens at = {0};
    ds4_tokenize_text(e, txt, &at);
    ds4_tokens_push(&at, ds4_token_eos(e));
    const uint32_t W = c->maxlen - 1u;
    for (uint32_t off = 0; off < (uint32_t)at.len; off += W) {
        const uint32_t m = (uint32_t)at.len - off < W ? (uint32_t)at.len - off : W;
        d->s = realloc(d->s, (d->ns + 1) * sizeof *d->s);
        pt_sample *s = &d->s[d->ns++];
        memset(s, 0, sizeof *s);
        s->chunk = ci; s->eval = eval; s->raw = 1; s->q = strdup(""); s->a = strdup(""); s->src = 0;
        s->sn = m + 1u; s->sa0 = 1u; s->m = m;
        s->sids = xmalloc((size_t)s->sn * 4);
        s->sids[0] = e->vocab.bos_id;
        memcpy(s->sids + 1, at.v + off, (size_t)m * 4);
        s->tids = xmalloc((size_t)s->sn * 4); memcpy(s->tids, s->sids, (size_t)s->sn * 4); s->tn = s->sn; s->ta0 = 1u;
        pt_fill_onehot(c, s);
    }
    ds4_tokens_free(&at);
    return true;
}

static bool pt_add_sample(ds4_engine *e, const pt_cfg *c, pt_data *d, int src, const char *sys, const char *ctx, const char *q, const char *a, uint32_t *skipped) {
    const int hold = src == 1;
    const uint32_t ci = pt_chunk_of(d, ctx, hold, 0);
    ds4_tokens sp = {0}, tp = {0}, at = {0};
    ds4_encode_chat_prompt(e, sys, q, DS4_THINK_NONE, &sp);
    if (ctx[0]) {
        const size_t tl = strlen(ctx) + strlen(q) + 3;
        char *tt = xmalloc(tl);
        snprintf(tt, tl, "%s\n\n%s", ctx, q);
        ds4_encode_chat_prompt(e, sys, tt, DS4_THINK_NONE, &tp);
        free(tt);
    } else ds4_encode_chat_prompt(e, sys, q, DS4_THINK_NONE, &tp);   /* 保持料 / 普通题: 教师提示 = 学生提示 */
    ds4_tokenize_text(e, a, &at);
    ds4_tokens_push(&at, ds4_token_eos(e));
    if ((uint32_t)(sp.len + at.len) > c->maxlen) { (*skipped)++; ds4_tokens_free(&sp); ds4_tokens_free(&tp); ds4_tokens_free(&at); return true; }
    d->s = realloc(d->s, (d->ns + 1) * sizeof *d->s);
    pt_sample *s = &d->s[d->ns++];
    memset(s, 0, sizeof *s);
    s->chunk = ci; s->eval = pt_is_eval(q); s->q = strdup(q); s->a = strdup(a); s->src = src;
    s->sids = pt_cat(&sp, &at, &s->sn); s->sa0 = (uint32_t)sp.len;
    s->tids = pt_cat(&tp, &at, &s->tn); s->ta0 = (uint32_t)tp.len;
    s->m = (uint32_t)at.len;
    if (d->ch[ci].sft) pt_fill_onehot(c, s);   /* 不带材料: 目标 = 答案 token 本身 */
    ds4_tokens_free(&sp); ds4_tokens_free(&tp); ds4_tokens_free(&at);
    return true;
}

/* 一行 jsonl → system/context/user/assistant 四串(malloc; 没有的给 ""), 或 text 一串(*txt 非空时另四串都是 ""); 失败时 *err 指向原因(静态串) */
static bool pt_parse_line(const char *ln, char **sys, char **ctx, char **q, char **a, char **txt, const char **err) {
    const char *p = ln; char *key = NULL, *role = NULL, *content = NULL;
    *sys = *ctx = *q = *a = *txt = NULL; *err = "不是 JSON 对象";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    for (;;) {
        json_ws(&p);
        if (*p == '}') { p++; break; }
        if (!json_string(&p, &key)) { *err = "键不是字符串"; return false; }
        json_ws(&p);
        if (*p != ':') { free(key); *err = "键后面没有冒号"; return false; }
        p++;
        if (!strcmp(key, "context")) {
            if (!json_string(&p, ctx)) { free(key); *err = "context 要是字符串"; return false; }
        } else if (!strcmp(key, "text")) {
            if (!json_string(&p, txt)) { free(key); *err = "text 要是字符串"; return false; }
        } else if (!strcmp(key, "messages")) {
            json_ws(&p);
            if (*p != '[') { free(key); *err = "messages 要是数组"; return false; }
            p++;
            for (;;) {
                json_ws(&p);
                if (*p == ']') { p++; break; }
                if (*p != '{') { free(key); *err = "messages 的元素要是对象"; return false; }
                p++; role = content = NULL;
                for (;;) {
                    json_ws(&p);
                    if (*p == '}') { p++; break; }
                    char *mk = NULL;
                    if (!json_string(&p, &mk)) { free(key); *err = "message 的键不是字符串"; return false; }
                    json_ws(&p);
                    if (*p != ':') { free(key); free(mk); *err = "message 键后面没有冒号"; return false; }
                    p++;
                    bool ok = !strcmp(mk, "role") ? json_string(&p, &role) : !strcmp(mk, "content") ? json_string(&p, &content) : false;
                    if (!ok) { *err = !strcmp(mk, "role") || !strcmp(mk, "content") ? "role/content 要是字符串" : "message 只认 role / content"; free(key); free(mk); return false; }
                    free(mk);
                    json_ws(&p);
                    if (*p == ',') p++;
                }
                if (!role || !content) { free(key); *err = "message 缺 role 或 content"; return false; }
                char **slot = !strcmp(role, "system") ? sys : !strcmp(role, "user") ? q : !strcmp(role, "assistant") ? a : NULL;
                if (!slot) { free(key); *err = "role 只认 system / user / assistant"; return false; }
                if (*slot) { free(key); *err = "只收单轮(system 可选 + user + assistant 各一条)"; return false; }
                *slot = content; free(role); role = content = NULL;
                json_ws(&p);
                if (*p == ',') p++;
            }
        } else { free(key); *err = "只认 messages / context / text 三个字段(reward/mode/split 10-10 起不收)"; return false; }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
    }
    json_ws(&p);
    if (*p) { *err = "行尾有多余内容"; return false; }
    if (!*sys) *sys = strdup("");
    if (!*ctx) *ctx = strdup("");
    if (*txt) {   /* 原文行: 不许再带 messages/context(一行只能是一种料, 混着写说不清目标是什么) */
        if (*q || *a || (*ctx)[0]) { *err = "text 行不能再带 messages / context"; return false; }
        pt_rstrip(*txt);
        if (!(*txt)[0]) { *err = "text 是空的"; return false; }
        if (!*q) *q = strdup(""); if (!*a) *a = strdup("");
        return true;
    }
    *txt = strdup("");
    if (!*q || !*a) { *err = "messages 要有 user 和 assistant"; return false; }
    pt_rstrip(*q); pt_rstrip(*a); pt_rstrip(*ctx);
    if (!(*q)[0] || !(*a)[0]) { *err = "问或答是空的"; return false; }
    return true;
}

/* 读一份 jsonl(src 0 = data, 1 = hold) */
static bool pt_load_file(ds4_engine *e, const pt_cfg *c, pt_data *d, const char *path, int src, uint32_t *skipped) {
    char *b = pt_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --ptrain 读不了料 %s\n", path); return false; }
    uint32_t lineno = 0, n0 = d->ns;
    bool ok = true;
    for (char *p = b; ok && p && *p; ) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        lineno++;
        const char *q0 = p; while (*q0 == ' ' || *q0 == '\t' || *q0 == '\r') q0++;
        if (*q0) {
            char *sys, *ctx, *q, *a, *txt; const char *err;
            if (!pt_parse_line(p, &sys, &ctx, &q, &a, &txt, &err)) { fprintf(stderr, "ds4: --ptrain %s 第 %u 行: %s\n", path, lineno, err); ok = false; }
            else if (src == 1 && (ctx[0] || txt[0])) { fprintf(stderr, "ds4: --ptrain %s 第 %u 行: 保持料只收不带 context 的问答(教师 = 部署态自己)\n", path, lineno); ok = false; }
            else if (txt[0]) ok = pt_add_text(e, c, d, txt);
            else ok = pt_add_sample(e, c, d, src, sys[0] ? sys : NULL, ctx, q, a, skipped);
            free(sys); free(ctx); free(q); free(a); free(txt);
        }
        p = nl ? nl + 1 : NULL;
    }
    free(b);
    if (ok) fprintf(stderr, "ds4: [ptrain] %s: %u 行, 收 %u 题\n", path, lineno, d->ns - n0);
    return ok;
}

bool pt_data_load(ds4_engine *e, const pt_cfg *c, pt_data *d) {
    memset(d, 0, sizeof *d);
    uint32_t skipped = 0;
    if (!pt_load_file(e, c, d, c->data, 0, &skipped)) return false;
    if (c->hold[0]) { if (!pt_load_file(e, c, d, c->hold, 1, &skipped)) return false; }
    else fprintf(stderr, "ds4: ★[ptrain] 没给 hold=(保持料): ③ 在通用问题上没有东西钉着, 10-01 实撞局部性崩 —— 只在故意对照时这么跑★\n");
    /* 每块的公共前缀: 同块各题教师序列逐 id 相同的那一段(封顶到最短提示 −1: 前缀里不许含任何一题自己的问题 token) */
    for (uint32_t ci = 0; ci < d->nch; ci++) {
        const pt_sample *f = NULL; uint32_t lcp = 0;
        for (uint32_t i = 0; i < d->ns; i++) {
            const pt_sample *s = &d->s[i];
            if (s->chunk != ci) continue;
            if (!f) { f = s; lcp = s->ta0 - 1u; continue; }
            uint32_t k = 0; while (k < lcp && k < s->ta0 - 1u && f->tids[k] == s->tids[k]) k++;
            lcp = k;
        }
        d->ch[ci].lcp = lcp;
    }
    uint32_t ne = 0, nsft = 0, nraw = 0, nh = 0, mx = 0, h[4] = { 0, 0, 0, 0 };   /* 学生序列长度分布: 训练缓冲按 maxlen 行预分配, 定 maxlen 要看尾巴有多长 */
    for (uint32_t i = 0; i < d->ns; i++) {
        ne += d->s[i].eval ? 1u : 0u; nsft += d->s[i].sft && !d->s[i].raw ? 1u : 0u; nraw += d->s[i].raw ? 1u : 0u; nh += d->s[i].src == 1 ? 1u : 0u;
        const uint32_t L = d->s[i].sn;
        if (L > mx) mx = L;
        h[L <= 256u ? 0 : L <= 512u ? 1 : L <= 768u ? 2 : 3]++;
    }
    fprintf(stderr, "ds4: [ptrain] 料: %u 块, %u 题(训练 %u / 留出 %u; 带材料 %u, 不带材料 %u, 原文段 %u, 保持料 %u), 超 maxlen %u 跳过 %u 题; 长度 ≤256/≤512/≤768/更长 = %u/%u/%u/%u, 最长 %u\n",
            d->nch, d->ns, d->ns - ne, ne, d->ns - nsft - nraw - nh, nsft, nraw, nh, c->maxlen, skipped, h[0], h[1], h[2], h[3], mx);
    return d->ns > 0;
}

void pt_data_free(pt_data *d) {
    for (uint32_t i = 0; i < d->nch; i++) { free(d->ch[i].name); free(d->ch[i].text); }
    for (uint32_t i = 0; i < d->ns; i++) {
        pt_sample *s = &d->s[i];
        free(s->q); free(s->a); free(s->sids); free(s->tids); free(s->top_id); free(s->top_p); free(s->top_rest);
    }
    free(d->ch); free(d->s); memset(d, 0, sizeof *d);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_data_nonempty_tu;
