/* core_ptrain_probe.c — 训练器的生成侧: 贪心探针(留出题, 部署态与挂 ③ 并排原样输出)。2026-10-06 从 core_ptrain.c 拆出并改走合批驱动;
 * 10-10 收口: 自定义探针 / 部署态答案缓存 / 奖励回路采样器一并砍掉, 只剩留出题探针。
 *
 * 走公共驱动 ds4_v41_gen_run(core_v41_gen.c)的整组模式, 1 路一批: 行数变 ⇒ 稠密段末位舍入变 ⇒ 近平局的那一位翻面(10-06 门: 8 路下 20 道里 3 道数字不同),
 * 1 路与单请求路逐字节同, 贪心读数才与历史可比。探针只看行为, 判决归留出损失与脚本的 wt2 门。
 * 部署态(①+②)的探针答案各轮不变: 算一次放 base[](一趟内各轮共用)。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

typedef struct { char **out; } pt_gen_sink;
static void pt_gen_done(void *ud, uint32_t job, const char *text, size_t len, int eos, uint32_t ntok) {
    pt_gen_sink *k = (pt_gen_sink *)ud;
    (void)eos; (void)ntok;
    k->out[job] = xmalloc(len + 1); memcpy(k->out[job], text, len); k->out[job][len] = 0;
}
/* n 条提示在同一份 ③(pt_dir; NULL = 只挂 ②)下贪心解码, 每条最多 probe_tok 个 token(含 EOS 那一位, 与 generate_argmax 的 n_predict 同口径)。
 * out[i] = 答案文本(malloc; 失败 "(生成失败)") */
static bool pt_gen_batch(ds4_engine *e, const pt_cfg *c, const char *pt_dir, uint32_t n, const int *const *ids, const uint32_t *len, char **out) {
    const char *keep = g_ds4_v41_pt_dir;
    g_ds4_v41_pt_dir = pt_dir;
    const ds4_decode_sampling greedy = {0};   /* 温 0 = 设备 argmax(core_v41_req.c 三条取 token 的路里的默认那条) */
    ds4_v41_gen_job *jobs = xmalloc((size_t)(n + 1) * sizeof *jobs);
    for (uint32_t i = 0; i < n; i++) { jobs[i] = (ds4_v41_gen_job){ .ids = ids[i], .len = len[i], .max_tok = c->probe_tok, .sp = greedy }; out[i] = NULL; }
    pt_gen_sink sink = { out };
    const bool ok = ds4_v41_gen_run(e, jobs, n, 1, 0, pt_gen_done, &sink);
    if (!ok) fprintf(stderr, "ds4: [ptrain 探针] 合批生成失败\n");
    for (uint32_t i = 0; i < n; i++) if (!out[i]) out[i] = strdup("(生成失败)");
    free(jobs);
    g_ds4_v41_pt_dir = keep;
    return ok;
}

/* 探针题: 料的留出题里按块均匀抽 probe_n 道(每块至多一道, 块按出现序等距取) —— 大块排前面时顺序取会全是它的题;
 * 末尾再加至多 PT_PROBE_HOLD 道保持料留出题(通用题): 原样输出直接看 ③ 有没有把通用回答带偏, 不只看 KL */
static uint32_t pt_probe_pick(const pt_data *d, uint32_t want, uint32_t *out) {
    uint32_t nc = 0, k = 0;
    uint32_t *first = xmalloc((size_t)(d->nch + 1) * 4);
    for (uint32_t ci = 0; ci < d->nch; ci++) {
        first[ci] = UINT32_MAX;
        if (d->ch[ci].hold) continue;
        for (uint32_t i = 0; i < d->ns; i++) if (d->s[i].chunk == ci && d->s[i].eval && d->s[i].top_id && !d->s[i].raw) { first[ci] = i; break; }   /* 原文段没有问句, 不探 */
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

bool pt_probes(ds4_engine *e, const pt_cfg *c, const pt_data *d, char **base, const char *pt_dir, const char *tag) {
    char p[1200]; snprintf(p, sizeof p, "%s/probe_%s.txt", c->out, tag);
    FILE *f = fopen(p, "w");
    const double t0 = now_sec();
    uint32_t *pick = xmalloc((size_t)(c->probe_n + PT_PROBE_HOLD + 1) * 4);
    const uint32_t np = pt_probe_pick(d, c->probe_n, pick);
    const int **ids = xmalloc((size_t)(np + 1) * sizeof *ids); uint32_t *len = xmalloc((size_t)(np + 1) * 4);
    char **ans = xmalloc_zeroed(np + 1, sizeof(char *));
    for (uint32_t k = 0; k < np; k++) { const pt_sample *s = &d->s[pick[k]]; ids[k] = s->sids; len[k] = s->sa0; }
    /* 部署态只算一次(base[k] 跨轮): 还没有的那些题一批算出来 */
    const int **mi = xmalloc((size_t)(np + 1) * sizeof *mi); uint32_t *ml = xmalloc((size_t)(np + 1) * 4), *who = xmalloc((size_t)(np + 1) * 4), miss = 0;
    for (uint32_t k = 0; k < np; k++) if (!base[k]) { mi[miss] = ids[k]; ml[miss] = len[k]; who[miss++] = k; }
    bool ok = true;
    if (miss) {
        char **tmp = xmalloc_zeroed(miss + 1, sizeof(char *));
        ok = pt_gen_batch(e, c, NULL, miss, mi, ml, tmp);
        for (uint32_t j = 0; j < miss; j++) base[who[j]] = tmp[j];
        free(tmp);
    }
    free(mi); free(ml); free(who);
    if (ok && pt_dir && np) ok = pt_gen_batch(e, c, pt_dir, np, ids, len, ans);
    for (uint32_t k = 0; k < np; k++) {
        const pt_sample *s = &d->s[pick[k]];
        fprintf(stderr, "\n[ptrain 探针 %s #%u] 问: %s\n  部署态(①+②): %s\n  挂 ③: %s\n  参考答案: %.300s\n", tag, k, s->q, base[k], ans[k] ? ans[k] : "(第 0 步与部署态同)", s->a);
        if (f) fprintf(f, "#%u 问: %s\n部署态: %s\n挂③: %s\n参考答案: %s\n\n", k, s->q, base[k], ans[k] ? ans[k] : "(同部署态)", s->a);
        free(ans[k]);
    }
    free(ids); free(len); free(ans); free(pick);
    if (f) fclose(f);
    if (ok) fprintf(stderr, "ds4: [ptrain 探针 %s] 留出题 %u 道(部署态 / 挂 ③ 各 1 路一批), %.0f s\n", tag, np, now_sec() - t0);
    return ok;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_probe_nonempty_tu;
