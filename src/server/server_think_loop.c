/* server_think_loop.c — 逐 token 死环检测: 思考段触发就强制吐 </think>(2026-10-17, 用户定), 正文触发就强制 EOS(2026-10-18)。
 *
 * 判据: 一段(思考段 / 正文各自从零起算)里出现周期 P ≤ THINK_LOOP_MAXP 的逐 token 复读, 段长(含第一圈) ≥ max(3P, THINK_LOOP_MIN_RUN),
 * 或 ≥ max(THINK_LOOP_MIN_CYC·P, THINK_LOOP_MIN_RUN_SMALL)(短周期的环转够 8 圈就停, 不用等 256 个 token; 依据见 server_internal.h 那条枚举的注释)。
 * 为什么只抓"逐 token 精确周期": 饿了么那条(fable5 10-16 夜)进环后 46,571 位逐 id 周期 43, 模型给第一候选的概率中位 1.000,
 * 全精度底座在同一位置也是 1.000 —— 再想下去一个新 token 都不会有, 截断不丢任何信息; 换 seed 续写、开 DRY 都出不来(10-17 实测)。
 * 措辞会变的环(鹈鹕/r331)故意不抓: 它跟正常的反复推敲在 token 上分不开, 抓了就会误伤。
 * 门槛的依据: spark 全部 2323 个历史请求离线扫(bugmd_ids_tools.py cyclescan 256)触发 6 个, 逐个看原文全是真死环, 误报 0;
 * 不触发的请求里最长的精确复读只有 20~50 token。
 * 改了会怎样: MIN_RUN 调低 = 表格/分隔线这种合法重复开始误触发, 正常思考被腰斩; MAXP 调低 = 周期长的环(整段几百 token 一圈)抓不到。
 *
 * 状态 = 最近 MAXP 个思考 token 的环形缓冲 + 每个周期 P 的"逐 token 与 P 位前相同"连续计数; 每个 token O(MAXP), 约 2 KB, 可按值拷贝
 * (引擎问"再接上这几个草稿 token 之后呢"时在副本上往前推, 不动真状态)。 */
#include "server_internal.h"

void think_loop_push(think_loop *d, int32_t tok) {
    const uint32_t lim = d->n < THINK_LOOP_MAXP ? d->n : THINK_LOOP_MAXP;
    for (uint32_t p = 1; p <= lim; p++) {
        const int32_t prev = d->ring[(d->n - p) % THINK_LOOP_MAXP];
        d->run[p] = prev == tok ? d->run[p] + 1u : 0u;
    }
    d->ring[d->n % THINK_LOOP_MAXP] = tok;
    d->n++;
}

bool think_loop_hit(const think_loop *d, int *period, int *span) {
    const uint32_t lim = d->n < THINK_LOOP_MAXP ? d->n : THINK_LOOP_MAXP;
    for (uint32_t p = 1; p <= lim; p++) {
        const uint32_t r = d->run[p];   /* 连续 r 位与 p 位前相同 ⇒ 这一段长 r + p, 其中完整重复了 r / p 圈 */
        const uint32_t len = r + p;
        if ((r >= 2u * p && len >= THINK_LOOP_MIN_RUN) ||
            (len >= THINK_LOOP_MIN_CYC * p && len >= THINK_LOOP_MIN_RUN_SMALL)) {
            if (period) *period = (int)p;
            if (span) *span = (int)(r + p);
            return true;
        }
    }
    return false;
}

/* 引擎取下一个 token 之前问(ds4_force_fn): 已收到的本段 token + ahead 这几个草稿之后, 是不是已经陷进死环。
 * 是 ⇒ 思考段这一位强制 </think>, 正文这一位强制 EOS。ahead 里出现 </think> 或 EOS 就是本段已经结束, 不再管。
 * 命中时把周期/段长记在 g 上, v41_emit 收到那个强制 token 时落日志。
 * 正文也抓(2026-10-18): 10-13 r38 正文周期 4 刷了 80,891 位、10-17 不思考的鹈鹕正文三行一圈 99,280 token 到 curl 断开才停 ——
 * 思考段那套只喂 think token, 正文一直没人管。门槛同一套: 10-17 的离线账(cyclescan)本来就把正文段算在内, 正文里周期 ~250 的
 * 合法重复(工具调用模板)段长 ≤ 312, 被 3P 这条挡住, 误报 0。思考段以 think_loop_on 为开关(非思考请求没有思考段), 正文一律开。 */
int32_t v41_think_loop_force(void *ud, const int32_t *ahead, uint32_t n) {
    v41_gen *g = ud;
    const bool inside = g->thinking.inside;
    if (inside && !g->think_loop_on) return -1;
    if (!inside && g->budget_closed) {   /* 预算收口之后: 正文前 budget_prefix_n 位逐位强制起手串, 之后照常查正文死环 */
        for (uint32_t i = 0; i < n; i++) if (ahead[i] == g->eos) return -1;
        const uint32_t k = (uint32_t)g->post_n + n;
        if (k < (uint32_t)g->budget_prefix_n) return g->budget_prefix[k];
    }
    think_loop d = g->loop;
    for (uint32_t i = 0; i < n; i++) {
        if (ahead[i] == g->eos) return -1;
        if (ahead[i] == g->think_end) {   /* 草稿已越过 </think>: 只有预算收口(思考位 ≥ 预算)才接着强制起手串 */
            if (!inside || g->budget_tail_n == 0 || d.n < (uint32_t)g->think_budget) return -1;
            const uint32_t k = n - 1u - i;
            return k < (uint32_t)g->budget_prefix_n ? g->budget_prefix[k] : -1;
        }
        think_loop_push(&d, ahead[i]);
    }
    if (inside && g->budget_tail_n > 0 && d.n >= (uint32_t)g->think_budget) {   /* 思考预算: 量化权重把 </think> 压低 4~22 倍, 长推演自己收不了口 */
        /* 无状态: 预算之后的第 k 个思考位就是收尾序列第 k 个(前面的位都是被强制的, 草稿只有等于它才会被接受) */
        const uint32_t k = d.n - (uint32_t)g->think_budget;
        if (k >= (uint32_t)g->budget_tail_n) return g->think_end;
        if (k == (uint32_t)g->budget_tail_n - 1) g->budget_hit = true;
        return g->budget_tail[k];
    }
    int p = 0, span = 0;
    if (!think_loop_hit(&d, &p, &span)) return -1;
    g->loop_period = p; g->loop_span = span;
    return inside ? g->think_end : g->eos;
}

/* 思考预算(--think-budget / thinking.budget_tokens): 收尾序列 = 交接句 + </think>, v41_think_loop_force 到预算后逐位强制。
 * 光强制 </think> 不够: 1.6-bit 会在正文里接着推(收口后 2700 token 没写代码), 所以先强制一句交接(Qwen3 思考预算同法);
 * --budget-answer-prefix 再逼正文直接起手(LCB 用 ```python: 不加时 20 题里 7 题收口后在正文接着推 8k token)。 */
void v41_think_budget_setup(server *s, job *j, v41_gen *g) {
    g->think_budget = j->req.think_budget > 0 ? j->req.think_budget : s->think_budget;
    if (g->think_budget <= 0 || !g->think_loop_on) return;
    ds4_tokens tt = {0};
    ds4_tokenize_text(s->engine, "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n", &tt);
    g->budget_tail = xmalloc((size_t)(tt.len + 1) * sizeof(int32_t));
    for (int i = 0; i < tt.len; i++) g->budget_tail[i] = tt.v[i];
    g->budget_tail[tt.len] = g->think_end;
    g->budget_tail_n = tt.len + 1;
    ds4_tokens_free(&tt);
    if (!s->budget_answer_prefix || !*s->budget_answer_prefix) return;
    ds4_tokens pt = {0};
    ds4_tokenize_text(s->engine, s->budget_answer_prefix, &pt);
    g->budget_prefix = xmalloc((size_t)(pt.len + 1) * sizeof(int32_t));
    for (int i = 0; i < pt.len; i++) g->budget_prefix[i] = pt.v[i];
    g->budget_prefix_n = pt.len;
    ds4_tokens_free(&pt);
}
void v41_think_budget_free(v41_gen *g) { free(g->budget_tail); free(g->budget_prefix); g->budget_tail = g->budget_prefix = NULL; }
