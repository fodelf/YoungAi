/* core_ptrain_probe.c — 训练器的生成侧: 贪心探针(留出题 / 自定义题, 部署态与挂 ③ 并排)与奖励回路的采样器。2026-10-06 从 core_ptrain.c 拆出, 同时改成合批。
 *
 * ★为什么合批★(10-06 用户 "原始引擎 43 t/s, 这才多少语料"): 第 0010 次奖励回路一轮 40 分钟里 839 s 是采样 —— 160 份一份一份跑, 每份先预填 550 token
 * 的提示再单流解码(投机关 30 t/s); 贪心探针 40 份同样单流。解码是字节墙(一步要流一遍骨架 + 专家权重), 8 行一步与 1 行一步几乎同价, 所以走服务端
 * 并发那条出口(core_v41_req.c: 请求态各自预填 → 收缩 → 每步 N 行一次前向, 反修表只在批态里一份)。门 = 温 0 下与单请求路逐字节同(服务端合批
 * 09-30 的门同一句; probe 文件 diff); 采样下 u 由 (seed, 位置, i) 哈希(cuda_v41_sample.inc.cu), 种子按份给 ⇒ 同样逐字节同(sample 文件 diff)。
 * 预填仍逐路(请求态之间没有 KV 复用): 同一提示的 8 份要预填 8 遍, 这是下一刀。
 * 部署态(①+②)的探针答案各轮不变: 自定义题的那份算一次缓存进 probe_base= 文件(奖励回路各轮共用), 留出题的那份在 base[] 里(一趟内各轮共用)。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define PT_LINE_MAX (1u << 20)   /* probe_base 文件一行(一道题的答案, ≤ probe_tok 个 token)的上限 */

typedef struct { ds4_engine *e; char *buf; size_t n, cap; int eos; } pt_gen;
static int pt_emit(int tok, void *ud) {
    pt_gen *g = (pt_gen *)ud;
    if (tok == ds4_token_eos(g->e)) { g->eos = 1; return 1; }
    size_t len = 0;
    char *t = ds4_token_text(g->e, tok, &len);
    if (t) {
        if (g->n + len + 1 > g->cap) { g->cap = (g->n + len + 1) * 2; g->buf = realloc(g->buf, g->cap); }
        memcpy(g->buf + g->n, t, len); g->n += len; g->buf[g->n] = 0;
        free(t);
    }
    return 0;
}

/* 合批生成: n 条提示在同一份 ③(pt_dir; NULL = 只挂 ②)下一起解码, 每条各自的采样面(sp[i]; sp = NULL 全贪心), 每条最多 probe_tok 个 token
 * (含 EOS 那一位, 与 generate_argmax 的 n_predict 同口径)。一次最多 DS4_V41_GEMV_MAX_TOK 路, 多了分批; 批内各路先各自预填完(一次一路: 预填态带
 * 整份反修表, 收缩后只剩 KV), 再每步把还活着的路合成一次前向。out[i] = 答案文本(malloc; 失败 "(生成失败)"), eos[i](可 NULL) = 1 以 EOS 收口 / 0 被截断。 */
static bool pt_gen_batch(ds4_engine *e, const pt_cfg *c, const char *pt_dir, uint32_t bcap, uint32_t n, const int *const *ids, const uint32_t *len,
                         const ds4_decode_sampling *sp, char **out, int *eos) {
    const char *keep = g_ds4_v41_pt_dir;
    g_ds4_v41_pt_dir = pt_dir;
    if (bcap < 1u) bcap = 1u;
    if (bcap > DS4_V41_GEMV_MAX_TOK) bcap = DS4_V41_GEMV_MAX_TOK;
    const uint32_t cap = n < bcap ? n : bcap;
    struct ds4_v41_batch *b = cap ? ds4_v41_batch_open(e, (int)cap) : NULL;
    bool ok = b != NULL;
    if (!ok) fprintf(stderr, "ds4: [ptrain 生成] 批态开不出来(%u 行)\n", cap);
    for (uint32_t i = 0; i < n; i++) { out[i] = NULL; if (eos) eos[i] = 0; }
    for (uint32_t g0 = 0; ok && g0 < n; g0 += cap) {
        const uint32_t ng = n - g0 < cap ? n - g0 : cap;
        struct ds4_v41_req *r[DS4_V41_GEMV_MAX_TOK] = {0};
        pt_gen g[DS4_V41_GEMV_MAX_TOK];
        uint32_t made[DS4_V41_GEMV_MAX_TOK]; int live[DS4_V41_GEMV_MAX_TOK];
        for (uint32_t i = 0; i < ng; i++) { g[i] = (pt_gen){ e, NULL, 0, 0, 0 }; made[i] = 0; live[i] = 0; }
        for (uint32_t i = 0; ok && i < ng; i++) {
            const ds4_decode_sampling greedy = {0};   /* 温 0 = 设备 argmax(core_v41_req.c 三条取 token 的路里的默认那条) */
            r[i] = ds4_v41_req_open(e, ids[g0 + i], (int)len[g0 + i], (int)c->probe_tok, sp ? &sp[g0 + i] : &greedy);
            if (!r[i]) { fprintf(stderr, "ds4: [ptrain 生成] 第 %u 路请求态开不出来\n", g0 + i); ok = false; break; }
            int prc;
            while ((prc = ds4_v41_req_prefill_step(r[i])) == 0) {}
            if (prc < 0) { fprintf(stderr, "ds4: [ptrain 生成] 第 %u 路预填失败\n", g0 + i); ok = false; break; }
            made[i] = 1;   /* 首个 token 预填就出了 */
            live[i] = pt_emit(ds4_v41_req_next(r[i]), &g[i]) == 0 && made[i] < c->probe_tok;
        }
        while (ok) {
            struct ds4_v41_req *act[DS4_V41_GEMV_MAX_TOK]; uint32_t who[DS4_V41_GEMV_MAX_TOK], na = 0;
            for (uint32_t i = 0; i < ng; i++) if (live[i] && ds4_v41_req_room(r[i]) > 0) { act[na] = r[i]; who[na++] = i; }
            if (!na) break;
            if (ds4_v41_multi_step(b, act, (int)na) != 0) { fprintf(stderr, "ds4: [ptrain 生成] 合批一步失败(%u 路)\n", na); ok = false; break; }
            for (uint32_t k = 0; k < na; k++) {
                int o[DS4_MTP_MAX_BLOCK + 2u];
                const int m = ds4_v41_req_take(act[k], o, (int)(DS4_MTP_MAX_BLOCK + 2u));
                const uint32_t i = who[k];
                for (int j = 0; j < m && live[i]; j++) { made[i]++; if (pt_emit(o[j], &g[i]) != 0 || made[i] >= c->probe_tok) live[i] = 0; }
            }
        }
        for (uint32_t i = 0; i < ng; i++) {
            if (ok) { out[g0 + i] = g[i].buf ? g[i].buf : strdup(""); if (eos) eos[g0 + i] = g[i].eos; }
            else free(g[i].buf);
            ds4_v41_req_close(r[i]);
        }
    }
    for (uint32_t i = 0; i < n; i++) if (!out[i]) out[i] = strdup("(生成失败)");
    if (b) ds4_v41_batch_close(b);
    g_ds4_v41_pt_dir = keep;
    return ok;
}

/* 文本的 \n \t \ 写成 \n \t \\(sample 文件与 probe_base 文件同一种转义), 与反转义 */
static void pt_esc_puts(FILE *f, const char *t) {
    for (; *t; t++) { if (*t == '\n') fputs("\\n", f); else if (*t == '\t') fputs("\\t", f); else if (*t == '\\') fputs("\\\\", f); else fputc(*t, f); }
}
static char *pt_unesc(const char *s) {
    char *o = xmalloc(strlen(s) + 1); size_t n = 0;
    for (; *s; s++) {
        if (*s == '\\' && s[1]) { s++; o[n++] = *s == 'n' ? '\n' : *s == 't' ? '\t' : *s; }
        else o[n++] = *s;
    }
    o[n] = 0;
    return o;
}

/* 自定义探针的部署态答案缓存(probe_base=<文件>): 部署态 = ①+②, 各轮不变, 奖励回路每轮重新装载再算 20 道是白算。一行一道 "题号\t答案"(转义同上);
 * 行数对不上题数(题目换了)就当没有, 重算重写。 */
static void pt_probe_base_load(const char *path, pt_userq *uq) {
    if (!path[0] || !uq->n) return;
    FILE *f = fopen(path, "r");
    if (!f) return;
    char **tmp = xmalloc_zeroed(uq->n + 1, sizeof(char *));
    char *ln = xmalloc(PT_LINE_MAX); uint32_t got = 0;
    while (fgets(ln, (int)PT_LINE_MAX, f)) {
        size_t L = strlen(ln); while (L && (ln[L - 1] == '\n' || ln[L - 1] == '\r')) ln[--L] = 0;
        char *tab = strchr(ln, '\t'); if (!tab) continue;
        const uint32_t k = (uint32_t)strtoul(ln, NULL, 10);
        if (k >= 1 && k <= uq->n && !tmp[k - 1]) { tmp[k - 1] = pt_unesc(tab + 1); got++; }
    }
    free(ln); fclose(f);
    if (got == uq->n) {
        for (uint32_t k = 0; k < uq->n; k++) { if (!uq->ub[k]) uq->ub[k] = tmp[k]; else free(tmp[k]); }
        fprintf(stderr, "ds4: [ptrain 探针] 部署态答案 %u 道 ← %s\n", got, path);
    } else {
        for (uint32_t k = 0; k < uq->n; k++) free(tmp[k]);
        fprintf(stderr, "ds4: [ptrain 探针] %s 里 %u 道对不上 %u 题, 部署态重算\n", path, got, uq->n);
    }
    free(tmp);
}
static void pt_probe_base_save(const char *path, const pt_userq *uq) {
    if (!path[0] || !uq->n) return;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ds4: [ptrain 探针] 写不了 %s\n", path); return; }
    for (uint32_t k = 0; k < uq->n; k++) { fprintf(f, "%u\t", k + 1u); pt_esc_puts(f, uq->ub[k] ? uq->ub[k] : ""); fputc('\n', f); }
    fclose(f);
    fprintf(stderr, "ds4: [ptrain 探针] 部署态答案 %u 道 → %s\n", uq->n, path);
}

/* 探针题: 复盘留出题里按块均匀抽 probe_n 道(每块至多一道, 块按清单序等距取) —— 全量料里大盘块排在前面, 顺序取会全是大盘题;
 * 末尾再加至多 PT_PROBE_HOLD 道保持料留出题(通用题): 原样输出直接看 ③ 有没有把通用回答带偏, 不只看 KL */
static uint32_t pt_probe_pick(const pt_data *d, uint32_t want, uint32_t *out) {
    uint32_t nc = 0, k = 0;
    uint32_t *first = xmalloc((size_t)(d->nch + 1) * 4);
    for (uint32_t ci = 0; ci < d->nch; ci++) {
        first[ci] = UINT32_MAX;
        if (d->ch[ci].hold) continue;
        for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].chunk == ci && d->s[i].eval && d->s[i].top_id) { first[ci] = i; break; }
        if (first[ci] != UINT32_MAX) nc++;
    }
    for (uint32_t q = 0, seen = 0; q < d->nch && k < want; q++) {
        if (first[q] == UINT32_MAX) continue;
        if ((uint64_t)seen * want / (nc ? nc : 1) >= k) out[k++] = first[q];   /* 第 seen 个有题的块越过第 k 个等分点就取(块比题少时每块都取) */
        seen++;
    }
    free(first);
    for (uint32_t i = 0, h = 0; i < d->ns && h < PT_PROBE_HOLD; i++)
        if (d->s[i].eval && d->s[i].top_id && d->ch[d->s[i].chunk].hold) { out[k++] = i; h++; }
    return k;
}

/* 还没有部署态答案的那些题, 一批算出来填进 slot[](部署态 = 不挂 ③) */
static bool pt_fill_base(ds4_engine *e, const pt_cfg *c, uint32_t n, const int *const *ids, const uint32_t *len, char **slot) {
    const int **mi = xmalloc((size_t)(n + 1) * sizeof *mi); uint32_t *ml = xmalloc((size_t)(n + 1) * 4), *who = xmalloc((size_t)(n + 1) * 4), miss = 0;
    for (uint32_t k = 0; k < n; k++) if (!slot[k]) { mi[miss] = ids[k]; ml[miss] = len[k]; who[miss++] = k; }
    bool ok = true;
    if (miss) {
        char **tmp = xmalloc_zeroed(miss + 1, sizeof(char *));
        ok = pt_gen_batch(e, c, NULL, c->probe_batch, miss, mi, ml, NULL, tmp, NULL);
        for (uint32_t j = 0; j < miss; j++) slot[who[j]] = tmp[j];
        free(tmp);
    }
    free(mi); free(ml); free(who);
    return ok;
}

bool pt_probes(ds4_engine *e, const pt_cfg *c, const pt_data *d, char **base, pt_userq *uq, const char *pt_dir, const char *tag) {
    char p[1200]; snprintf(p, sizeof p, "%s/probe_%s.txt", c->out, tag);
    FILE *f = fopen(p, "w");
    const double t0 = now_sec();
    bool ok = true;
    uint32_t *pick = xmalloc((size_t)(c->probe_n + PT_PROBE_HOLD + 1) * 4);
    const uint32_t np = pt_probe_pick(d, c->probe_n, pick);
    {   /* ---- 留出题探针: 部署态只算一次(base[k] 跨轮), 挂 ③ 每轮一批 ---- */
        const int **ids = xmalloc((size_t)(np + 1) * sizeof *ids); uint32_t *len = xmalloc((size_t)(np + 1) * 4);
        char **ans = xmalloc_zeroed(np + 1, sizeof(char *));
        for (uint32_t k = 0; k < np; k++) { const pt_sample *s = &d->s[pick[k]]; ids[k] = s->sids; len[k] = s->sa0; }
        ok = pt_fill_base(e, c, np, ids, len, base);
        if (ok && pt_dir && np) ok = pt_gen_batch(e, c, pt_dir, c->probe_batch, np, ids, len, NULL, ans, NULL);
        for (uint32_t k = 0; k < np; k++) {
            const pt_sample *s = &d->s[pick[k]];
            fprintf(stderr, "\n[ptrain 探针 %s #%u] 问: %s\n  部署态(①+②): %s\n  挂 ③: %s\n  教师参考: %.300s\n", tag, k, s->q, base[k], ans[k] ? ans[k] : "(第 0 步与部署态同)", s->a);
            if (f) fprintf(f, "#%u 问: %s\n部署态: %s\n挂③: %s\n教师参考: %s\n\n", k, s->q, base[k], ans[k] ? ans[k] : "(同部署态)", s->a);
            free(ans[k]);
        }
        free(ids); free(len); free(ans);
    }
    free(pick);
    /* ---- 自定义题(probe_q=): 部署态(probe_base= 缓存) + 挂 ③ 各一批; 之后奖励回路的采样 ---- */
    ds4_tokens *tk = xmalloc_zeroed(uq->n + 1, sizeof *tk);
    const int **qid = xmalloc((size_t)(uq->n + 1) * sizeof *qid); uint32_t *qlen = xmalloc((size_t)(uq->n + 1) * 4);
    for (uint32_t k = 0; k < uq->n; k++) { ds4_encode_chat_prompt(e, NULL, uq->q[k], DS4_THINK_NONE, &tk[k]); qid[k] = tk[k].v; qlen[k] = (uint32_t)tk[k].len; }
    if (ok && uq->n) {
        uint32_t miss = 0;
        for (uint32_t k = 0; k < uq->n; k++) if (!uq->ub[k]) miss++;
        if (miss) {
            pt_probe_base_load(c->probe_base, uq);
            miss = 0; for (uint32_t k = 0; k < uq->n; k++) if (!uq->ub[k]) miss++;
            if (miss) { ok = pt_fill_base(e, c, uq->n, qid, qlen, uq->ub); if (ok) pt_probe_base_save(c->probe_base, uq); }
        }
        char **ans = xmalloc_zeroed(uq->n + 1, sizeof(char *));
        if (ok && pt_dir) ok = pt_gen_batch(e, c, pt_dir, c->probe_batch, uq->n, qid, qlen, NULL, ans, NULL);
        for (uint32_t k = 0; k < uq->n; k++) {
            fprintf(stderr, "\n[ptrain 自定义探针 %s #%u] 问: %s\n  部署态(①+②): %s\n  挂 ③: %s\n", tag, k, uq->q[k], uq->ub[k], ans[k] ? ans[k] : "(第 0 步与部署态同)");
            if (f) fprintf(f, "自定义#%u 问: %s\n部署态: %s\n挂③: %s\n\n", k, uq->q[k], uq->ub[k], ans[k] ? ans[k] : "(同部署态)");
            free(ans[k]);
        }
        free(ans);
    }
    if (f) fclose(f);
    if (ok) fprintf(stderr, "ds4: [ptrain 探针 %s] 留出题 %u 道 + 自定义 %u 道(部署态 / 挂 ③ 各 %u 路一批), %.0f s\n", tag, np, uq->n, c->probe_batch, now_sec() - t0);
    /* ★奖励回路的采样器★(10-04): 自定义探针题每道再按模型卡配方(温 1 / top_p 1 / min_p 0, 种子由 seed/题号/份号定, 可复现)抽 sample_n 份
     * → out/sample_<tag>.txt, 一行一份 "题号\t份号\t收口\t答案"; 结算归脚本(kd_domain/<域>.sh <域>_reward), 这里只抽不评。
     * 只在"落盘的 ③"上抽: epochs=0 时第 0 步的 init 就是产物(回路第 0 轮); 否则每轮末的 ckpt(第 0 步那份 = 上一轮末同一份 ③ 同种子, 抽了也是重复)。
     * 同一道题的 G 份一批(同一提示, 各自种子): 以前一份一份跑, 160 份 839 s(10-06)。 */
    if (ok && c->sample_n && uq->n && ((c->epochs == 0) == !strcmp(tag, "e00"))) {
        snprintf(p, sizeof p, "%s/sample_%s.txt", c->out, tag);
        FILE *sf = fopen(p, "w");
        if (!sf) { fprintf(stderr, "ds4: [ptrain] 写不了 %s\n", p); ok = false; }
        const double t1 = now_sec(); uint32_t neos = 0;
        const uint32_t G = c->sample_n;
        ds4_decode_sampling *smp = xmalloc((size_t)G * sizeof *smp); const int **sid = xmalloc((size_t)G * sizeof *sid); uint32_t *slen = xmalloc((size_t)G * 4);
        char **sans = xmalloc_zeroed(G + 1, sizeof(char *)); int *seos = xmalloc_zeroed(G + 1, sizeof(int));
        for (uint32_t k = 0; ok && k < uq->n; k++) {
            for (uint32_t j = 0; j < G; j++) {
                uint64_t sd = 0x9E3779B97F4A7C15ull ^ ((uint64_t)c->seed * 0x100000001B3ull) ^ ((uint64_t)(k + 1u) << 20) ^ (uint64_t)(j + 1u);
                if (!sd) sd = 1;   /* 0 = 按时钟 */
                smp[j] = (ds4_decode_sampling){ .temperature = DS4_DEFAULT_TEMPERATURE, .top_p = DS4_DEFAULT_TOP_P, .min_p = DS4_DEFAULT_MIN_P, .top_k = 0, .seed = sd };
                sid[j] = qid[k]; slen[j] = qlen[k];
            }
            ok = pt_gen_batch(e, c, pt_dir, c->sample_batch, G, sid, slen, smp, sans, seos);
            for (uint32_t j = 0; j < G; j++) {
                fprintf(sf, "%u\t%u\t%d\t", k + 1u, j + 1u, seos[j]); pt_esc_puts(sf, sans[j]); fputc('\n', sf);
                neos += seos[j] ? 1u : 0u; free(sans[j]); sans[j] = NULL;
            }
        }
        if (sf) fclose(sf);
        free(smp); free(sid); free(slen); free(sans); free(seos);
        if (ok) fprintf(stderr, "ds4: [ptrain 采样 %s] %u 题 × %u 份(挂 %s, %u 路一批), EOS 收口 %u 份, %.0f s → %s\n", tag, uq->n, G, pt_dir ? pt_dir : "部署态", c->sample_batch, neos, now_sec() - t1, p);
    }
    for (uint32_t k = 0; k < uq->n; k++) ds4_tokens_free(&tk[k]);
    free(tk); free(qid); free(qlen);
    return ok;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_probe_nonempty_tu;
