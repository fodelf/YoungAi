/* server_tests_think_loop.c — 思考段死环检测器(server_think_loop.c)的边界: 门槛正好卡在 max(3P, 256) 或 max(8P, 64), 周期超上限不抓, 不重复的不抓;
 * 强制钩子在副本上推草稿、不动真状态, 草稿里已经出现 </think>/EOS 就不再管。周期 43 照饿了么那条(fable5 10-16 夜)。 */
#include "server_tests_internal.h"

/* 不重复的序列: 第 i 个 token = 1000 + i, 任何周期都对不上 */
static void push_fresh(think_loop *d, int n, int base) { for (int i = 0; i < n; i++) think_loop_push(d, base + i); }

/* 周期 p 的环推 total 个 token, 返回第一次命中时已推了几个(没命中 = -1) */
static int first_hit(think_loop *d, int p, int total) {
    for (int i = 0; i < total; i++) {
        think_loop_push(d, 50000 + i % p);
        if (think_loop_hit(d, NULL, NULL)) return i + 1;
    }
    return -1;
}

void test_think_loop_detector(void) {
    think_loop d;
    memset(&d, 0, sizeof d); push_fresh(&d, 5000, 1000);
    TEST_ASSERT(!think_loop_hit(&d, NULL, NULL));

    /* 周期 43: 段长 = 已推的环 token 数, 第 256 个时 run = 213 ≥ 86 且段长 256 ⇒ 正好此刻命中, 前一个不命中 */
    memset(&d, 0, sizeof d); push_fresh(&d, 1000, 1000);
    int p = 0, span = 0;
    TEST_ASSERT(first_hit(&d, 43, 2000) == 256);
    TEST_ASSERT(think_loop_hit(&d, &p, &span) && p == 43 && span == 256);

    /* 周期 1(同一个 token 一直刷, 10-11 那条 "." 2320 位): 第二条判据(8 圈且 ≥64)在第 64 个就抓 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 1, 300) == 64);

    /* 周期 20(10-18 16:57 "身体覆盖座垫…" 一圈 20 token): 8 圈 = 160 个, 不再等 256 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 20, 400) == 160);
    TEST_ASSERT(think_loop_hit(&d, &p, &span) && p == 20 && span == 160);

    /* 周期 22 转 7 圈(154 个)不抓 —— 历史合法段最多 5.8 圈(P=22, 段长 128); 第 176 个(8 圈)才抓 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 22, 154) == -1);
    TEST_ASSERT(first_hit(&d, 22, 400) == 176 - 154);

    /* 周期 8 刷 63 个(历史合法段 7.9 圈那条)不抓: 第二条判据有 64 的地板 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 8, 63) == -1);

    /* 周期 200: 3P = 600 > 256, 要满三圈才算 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 200, 2000) == 600);

    /* 周期 300 超过上限: 刷多少圈都不抓 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 300, 3000) == -1);

    /* 重复被打断就清零: 周期 43 刷 200 个, 插一个新 token, 再刷 200 个 —— 两段都不到 256 */
    memset(&d, 0, sizeof d); push_fresh(&d, 10, 1000);
    TEST_ASSERT(first_hit(&d, 43, 200) == -1);
    think_loop_push(&d, 7);
    TEST_ASSERT(first_hit(&d, 43, 200) == -1);
}

void test_think_loop_force_hook(void) {
    v41_gen g;
    memset(&g, 0, sizeof g);
    g.think_loop_on = true; g.thinking.inside = true; g.think_end = 128822; g.eos = 1;
    for (int i = 0; i < 250; i++) think_loop_push(&g.loop, 50000 + i % 43);   /* 环里已真吐 250 个, 差 6 个到门槛 */
    TEST_ASSERT(v41_think_loop_force(&g, NULL, 0) == -1);
    int32_t ahead[8];
    for (int i = 0; i < 8; i++) ahead[i] = 50000 + (250 + i) % 43;
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 5) == -1);          /* 255 个: 还差一个 */
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 6) == 128822);      /* 256 个: 强制 </think> */
    TEST_ASSERT(g.loop_period == 43 && g.loop_span == 256);
    TEST_ASSERT(g.loop.n == 250);                                   /* 草稿只在副本上推, 真状态不动 */
    ahead[2] = 128822;
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 6) == -1);          /* 草稿里已经收口了 */
    g.thinking.inside = true; g.think_loop_on = false;
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 8) == -1);          /* 非思考请求的思考段不开检测 */

    /* 正文(2026-10-18): 同一门槛, 命中强制 EOS 而不是 </think>; 不看 think_loop_on */
    g.thinking.inside = false; g.loop_period = 0;
    TEST_ASSERT(v41_think_loop_force(&g, NULL, 0) == -1);           /* 250 个还差 6 个 */
    ahead[2] = 50000 + (250 + 2) % 43;
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 6) == 1);           /* 第 256 个: 强制 EOS */
    TEST_ASSERT(g.loop_period == 43 && g.loop_span == 256);
    ahead[4] = 1;
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 6) == -1);          /* 草稿里已经 EOS */
}

/* 思考预算(--think-budget): 思考位到预算后逐位强制 收尾句 → </think> → 正文起手串(--budget-answer-prefix)。
 * 无状态, 只看位置: 草稿越过的位上返回的就是那一位该强制的 token, 引擎只接受等于它的草稿。 */
void test_think_budget_force_hook(void) {
    v41_gen g;
    memset(&g, 0, sizeof g);
    int32_t tail[3] = { 700, 701, 702 }, prefix[2] = { 900, 901 };   /* 收尾句 3 个 token, 起手串 2 个(真实里如 ```python) */
    g.think_loop_on = true; g.thinking.inside = true; g.think_end = 128822; g.eos = 1;
    g.think_budget = 100; g.budget_tail = tail; g.budget_tail_n = 3; g.budget_prefix = prefix; g.budget_prefix_n = 2;
    push_fresh(&g.loop, 98, 1000);                                  /* 已真吐 98 个思考 token */
    TEST_ASSERT(v41_think_loop_force(&g, NULL, 0) == -1);
    int32_t ahead[8] = { 2000, 2001, 700, 701, 702, 128822, 900, 901 };
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 1) == -1);          /* 第 100 位还是自由的 */
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 2) == 700);         /* 第 101 位: 收尾句第一个 */
    TEST_ASSERT(!g.budget_hit);
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 4) == 702);         /* 收尾句最后一个: 记 budget_hit, v41_emit 落日志 */
    TEST_ASSERT(g.budget_hit);
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 5) == 128822);      /* 收尾句之后: </think> */
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 6) == 900);         /* 草稿已越过 </think>: 起手串 */
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 7) == 901);
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 8) == -1);          /* 起手串用完: 不再管 */
    TEST_ASSERT(g.loop.n == 98);                                    /* 草稿只在副本上推 */

    /* 收口之后的正文: 按已出正文 token 数(post_n)接着强制起手串, 用完回到正文死环检测 */
    g.thinking.inside = false; g.budget_closed = true; g.post_n = 0;
    memset(&g.loop, 0, sizeof g.loop);
    TEST_ASSERT(v41_think_loop_force(&g, NULL, 0) == 900);
    TEST_ASSERT(v41_think_loop_force(&g, prefix, 1) == 901);
    TEST_ASSERT(v41_think_loop_force(&g, prefix, 2) == -1);
    g.post_n = 1;
    TEST_ASSERT(v41_think_loop_force(&g, NULL, 0) == 901);
    ahead[0] = 1;
    TEST_ASSERT(v41_think_loop_force(&g, ahead, 1) == -1);          /* 草稿里已经 EOS */

    /* 预算前草稿自己收口: 不强制起手串; 预算关(budget_tail_n = 0): 只剩死环检测 */
    memset(&g.loop, 0, sizeof g.loop);
    g.thinking.inside = true; g.budget_closed = false; g.budget_hit = false; g.post_n = 0;
    push_fresh(&g.loop, 50, 1000);
    int32_t early[2] = { 128822, 900 };
    TEST_ASSERT(v41_think_loop_force(&g, early, 2) == -1);
    g.budget_tail_n = 0;
    push_fresh(&g.loop, 200, 3000);
    TEST_ASSERT(v41_think_loop_force(&g, NULL, 0) == -1);
}
