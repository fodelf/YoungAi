/* t_units.c — 无模型离线单测 (重构阶段8 新增, 登记 needs_model=0)。
 *
 * --engine-units: 现行采样/惩罚 API 的行为契约。ds4_session 直接栈上构造
 *   (core_internal.h 给完整定义), 只填 repeat_penalize_buf /
 *   session_penalties_active 真正读的字段; 词表形状走全局 g_ds4_shape,
 *   测前保存测后恢复 — 同一进程随后还要跑别的 suite, 污染全局 = 其余
 *   suite 用错 n_vocab 静默算错。
 * --rax: rax 基数树(tool-id replay map 的底层结构)基本操作对拍 rax.h 注释。 */
/* 这里不 include test_internal.h: 那条链进 server_internal.h, 与
 * core_internal.h 在同一 TU 撞名(xmalloc/xrealloc/now_sec 一边 static inline
 * 定义、一边外部声明, 两种 include 顺序各撞一半, 都编不过)。断言基元按
 * server_tests_internal.h 的契约手声明 — 链接的仍是同一个 test_failures,
 * 失败照常计入总账; 两处签名若分叉, 链接器立刻报错兜底。 */
#include "../src/core/core_internal.h"
#include "rax.h"

extern int test_failures;
void test_assert(bool cond, const char *file, int line, const char *expr);
#define TEST_ASSERT(expr) test_assert((expr), __FILE__, __LINE__, #expr)

/* ---- 采样/惩罚: 会话与词表脚手架 ------------------------------------- */

#define UNITS_VOCAB 64

/* 惩罚核按 g_ds4_shape.n_vocab 做 token 界检; 缩到 64 让 64..128k 的
 * 越界分支可测。整个 shape 存还原, 不只 n_vocab — 防未来惩罚核多读字段。 */
static ds4_shape units_saved_shape;

static void units_shape_shrink(void) {
    units_saved_shape = g_ds4_shape;
    g_ds4_shape.n_vocab = UNITS_VOCAB;
}

static void units_shape_restore(void) {
    g_ds4_shape = units_saved_shape;
}

/* 栈上最小会话: memset 零 = FREE lane(0)/无惩罚/空 checkpoint, 每个用例
 * 只再填自己要的字段。用完必须 units_session_free(checkpoint 是堆向量)。 */
static void units_session_init(ds4_session *s, float freq, float presence,
                               int gen_start, const int *toks, int n) {
    memset(s, 0, sizeof(*s));
    s->lane = DS4_LANE_FREE;
    s->req_freq = freq;
    s->req_presence = presence;
    s->repeat_gen_start = gen_start;
    for (int i = 0; i < n; i++) ds4_tokens_push(&s->checkpoint, toks[i]);
}

static void units_session_free(ds4_session *s) {
    ds4_tokens_free(&s->checkpoint);
}

/* ---- repeat_penalize_buf / session_penalties_active ------------------- */

static void units_test_freq_accumulates_presence_once(void) {
    /* token3 出现 3 次 / token5 两次 / token7 一次。freq 按出现次数累计,
     * presence 只在首次出现扣一回: 3 -> -0.5*3-0.25, 5 -> -0.5*2-0.25,
     * 7 -> -0.5-0.25。0.5/0.25 二进制精确, 断言可用 ==。 */
    const int toks[] = {3, 5, 3, 7, 3, 5};
    ds4_session s;
    units_session_init(&s, 0.5f, 0.25f, 0, toks, 6);
    float logits[UNITS_VOCAB] = {0};
    TEST_ASSERT(session_penalties_active(&s) == 1);
    repeat_penalize_buf(&s, logits, 6);
    TEST_ASSERT(logits[3] == -1.75f);
    TEST_ASSERT(logits[5] == -1.25f);
    TEST_ASSERT(logits[7] == -0.75f);
    TEST_ASSERT(logits[0] == 0.0f && logits[4] == 0.0f && logits[63] == 0.0f);
    units_session_free(&s);
}

static void units_test_presence_only_ignores_count(void) {
    /* 纯 presence: 次数不参与, 三次出现也只扣一次。 */
    const int toks[] = {9, 9, 9};
    ds4_session s;
    units_session_init(&s, 0.0f, 0.25f, 0, toks, 3);
    float logits[UNITS_VOCAB] = {0};
    repeat_penalize_buf(&s, logits, 3);
    TEST_ASSERT(logits[9] == -0.25f);
    units_session_free(&s);
}

static void units_test_negative_penalty_adds_probability(void) {
    /* OpenAI 允许 [-2,2]: 负 freq 反向加 logit(鼓励重复)。 */
    const int toks[] = {9, 9, 9};
    ds4_session s;
    units_session_init(&s, -1.0f, 0.0f, 0, toks, 3);
    float logits[UNITS_VOCAB] = {0};
    repeat_penalize_buf(&s, logits, 3);
    TEST_ASSERT(logits[9] == 3.0f);
    units_session_free(&s);
}

static void units_test_gen_start_scopes_window(void) {
    /* gen_start=4: 位置 0..3 是 prompt 区不罚, 只数位置 4(=3)/5(=5)。 */
    const int toks[] = {3, 5, 3, 7, 3, 5};
    ds4_session s;
    units_session_init(&s, 0.5f, 0.25f, 4, toks, 6);
    float logits[UNITS_VOCAB] = {0};
    repeat_penalize_buf(&s, logits, 6);
    TEST_ASSERT(logits[3] == -0.75f);   /* 只有位置4那一次 */
    TEST_ASSERT(logits[5] == -0.75f);
    TEST_ASSERT(logits[7] == 0.0f);     /* 只在 prompt 区出现 */
    units_session_free(&s);
}

static void units_test_gen_start_unmarked_resets_to_zero(void) {
    /* 未标记(-1)与越界(>end)都回落 0 = 整个上下文进窗(单机/分布式一致的
     * 遗留默认), 且回写进会话。 */
    const int toks[] = {3, 3};
    for (int gs = -1; gs <= 10; gs += 11) {   /* -1 与 10(>end=2) 两个分支 */
        ds4_session s;
        units_session_init(&s, 0.5f, 0.0f, gs, toks, 2);
        float logits[UNITS_VOCAB] = {0};
        repeat_penalize_buf(&s, logits, 2);
        TEST_ASSERT(s.repeat_gen_start == 0);
        TEST_ASSERT(logits[3] == -1.0f);
        units_session_free(&s);
    }
}

static void units_test_non_free_lane_bypasses_everything(void) {
    /* 非 FREE lane 合同(ds4.h): 工具语法/拷贝发射要裸 logits, 惩罚全旁路,
     * penalties_active 也报 0 让调用方跳过 scratch。 */
    const int toks[] = {3, 3, 3};
    const int lanes[] = {DS4_LANE_TOOL_SYNTAX, DS4_LANE_COPY_EMISSION};
    for (size_t i = 0; i < sizeof(lanes) / sizeof(lanes[0]); i++) {
        ds4_session s;
        units_session_init(&s, 0.5f, 0.25f, 0, toks, 3);
        s.lane = lanes[i];
        float logits[UNITS_VOCAB] = {0};
        TEST_ASSERT(session_penalties_active(&s) == 0);
        repeat_penalize_buf(&s, logits, 3);
        for (int t = 0; t < UNITS_VOCAB; t++) TEST_ASSERT(logits[t] == 0.0f);
        units_session_free(&s);
    }
}

static void units_test_zero_request_penalties_inactive(void) {
    /* req 全零 = 无事可做; 非零 freq 或非零 presence 任一都算激活。 */
    const int toks[] = {3};
    ds4_session s;
    units_session_init(&s, 0.0f, 0.0f, 0, toks, 1);
    TEST_ASSERT(session_penalties_active(&s) == 0);
    TEST_ASSERT(session_penalties_active(NULL) == 0);
    s.req_freq = 0.5f;
    TEST_ASSERT(session_penalties_active(&s) == 1);
    s.req_freq = 0.0f;
    s.req_presence = -0.5f;
    TEST_ASSERT(session_penalties_active(&s) == 1);
    float logits[UNITS_VOCAB] = {0};
    s.req_presence = 0.0f;
    repeat_penalize_buf(&s, logits, 1);
    TEST_ASSERT(logits[3] == 0.0f);
    units_session_free(&s);
}

static void units_test_out_of_vocab_tokens_ignored(void) {
    /* checkpoint 里越界 id(≥n_vocab)与负 id 直接跳过 — 词表缩到 64 后
     * 这些分支才真被走到。 */
    const int toks[] = {UNITS_VOCAB, -1, 3};
    ds4_session s;
    units_session_init(&s, 0.5f, 0.25f, 0, toks, 3);
    float logits[UNITS_VOCAB] = {0};
    repeat_penalize_buf(&s, logits, 3);
    TEST_ASSERT(logits[3] == -0.75f);
    for (int t = 0; t < UNITS_VOCAB; t++)
        if (t != 3) TEST_ASSERT(logits[t] == 0.0f);
    units_session_free(&s);
}

/* ---- ds4_sample_logits(公共 API) -------------------------------------- */

static void units_fill_ramp_logits(float *logits, int peak) {
    /* 唯一峰在 peak, 其余按下标线性递减且互不相等。 */
    for (int i = 0; i < UNITS_VOCAB; i++)
        logits[i] = -0.05f * (float)i;
    logits[peak] = 1.0f;
}

static void units_test_sample_temp0_is_argmax(void) {
    float logits[UNITS_VOCAB];
    units_fill_ramp_logits(logits, 42);
    uint64_t rng = 7;
    TEST_ASSERT(ds4_sample_logits(logits, UNITS_VOCAB, 0.0f, 0, 1.0f, 0.0f, &rng) == 42);
    /* 负温同样走贪心分支 */
    TEST_ASSERT(ds4_sample_logits(logits, UNITS_VOCAB, -1.0f, 40, 0.9f, 0.05f, &rng) == 42);
}

static void units_test_sample_seed_reproducible(void) {
    /* 同 seed 同参数 => 序列逐位一致; 采出的 id 必须在词表内。
     * top_k=0 与 top_k=40 两条路径都验(全词表堆采样 / top-k 筛)。 */
    float logits[UNITS_VOCAB];
    units_fill_ramp_logits(logits, 42);
    const int topks[] = {0, 40};
    for (size_t k = 0; k < sizeof(topks) / sizeof(topks[0]); k++) {
        uint64_t rng_a = 12345, rng_b = 12345;
        for (int i = 0; i < 16; i++) {
            const int a = ds4_sample_logits(logits, UNITS_VOCAB, 1.0f, topks[k], 0.95f, 0.0f, &rng_a);
            const int b = ds4_sample_logits(logits, UNITS_VOCAB, 1.0f, topks[k], 0.95f, 0.0f, &rng_b);
            TEST_ASSERT(a == b);
            TEST_ASSERT(a >= 0 && a < UNITS_VOCAB);
        }
        TEST_ASSERT(rng_a == rng_b);   /* rng 状态推进也一致 */
    }
}

static void units_test_sample_tiny_top_p_is_argmax(void) {
    /* top_p 极小: 概率核第一个候选(=argmax)进核即越过阈值, 采样恒 argmax。 */
    float logits[UNITS_VOCAB];
    units_fill_ramp_logits(logits, 42);
    uint64_t rng = 99;
    for (int i = 0; i < 32; i++)
        TEST_ASSERT(ds4_sample_logits(logits, UNITS_VOCAB, 1.0f, 0, 1e-6f, 0.0f, &rng) == 42);
}

void test_engine_units(void) {
    units_shape_shrink();
    units_test_freq_accumulates_presence_once();
    units_test_presence_only_ignores_count();
    units_test_negative_penalty_adds_probability();
    units_test_gen_start_scopes_window();
    units_test_gen_start_unmarked_resets_to_zero();
    units_test_non_free_lane_bypasses_everything();
    units_test_zero_request_penalties_inactive();
    units_test_out_of_vocab_tokens_ignored();
    units_shape_restore();
    units_test_sample_temp0_is_argmax();
    units_test_sample_seed_reproducible();
    units_test_sample_tiny_top_p_is_argmax();
}

/* ---- rax 基数树 -------------------------------------------------------- */

void test_rax_units(void) {
    rax *rt = raxNew();
    TEST_ASSERT(rt != NULL);
    if (!rt) return;

    /* 插入: 新 key 返回 1; 重复 key 返回 0 且交回旧值(值更新)。 */
    const char *keys[] = {"foo", "foobar", "bar", "baz", "a"};
    const size_t nkeys = sizeof(keys) / sizeof(keys[0]);
    for (size_t i = 0; i < nkeys; i++) {
        TEST_ASSERT(raxInsert(rt, (unsigned char *)keys[i], strlen(keys[i]),
                              (void *)(uintptr_t)(i + 1), NULL) == 1);
    }
    TEST_ASSERT(raxSize(rt) == nkeys);
    void *old = NULL;
    TEST_ASSERT(raxInsert(rt, (unsigned char *)"foo", 3,
                          (void *)(uintptr_t)100, &old) == 0);
    TEST_ASSERT(old == (void *)(uintptr_t)1);
    TEST_ASSERT(raxSize(rt) == nkeys);

    /* 查找: 命中回值(含更新后的), 未命中回 raxNotFound 哨兵。 */
    TEST_ASSERT(raxFind(rt, (unsigned char *)"foo", 3) == (void *)(uintptr_t)100);
    TEST_ASSERT(raxFind(rt, (unsigned char *)"foobar", 6) == (void *)(uintptr_t)2);
    TEST_ASSERT(raxFind(rt, (unsigned char *)"fo", 2) == raxNotFound);      /* 前缀非成员 */
    TEST_ASSERT(raxFind(rt, (unsigned char *)"foob", 4) == raxNotFound);
    TEST_ASSERT(raxFind(rt, (unsigned char *)"nope", 4) == raxNotFound);

    /* 删除: 命中返回 1 并交回值, 未命中返回 0 不动树。 */
    old = NULL;
    TEST_ASSERT(raxRemove(rt, (unsigned char *)"bar", 3, &old) == 1);
    TEST_ASSERT(old == (void *)(uintptr_t)3);
    TEST_ASSERT(raxFind(rt, (unsigned char *)"bar", 3) == raxNotFound);
    TEST_ASSERT(raxRemove(rt, (unsigned char *)"bar", 3, NULL) == 0);
    TEST_ASSERT(raxSize(rt) == nkeys - 1);

    /* 迭代: "^" 定位最小 key, raxNext 按字典序吐全量 a/baz/foo/foobar。 */
    const char *want[] = {"a", "baz", "foo", "foobar"};
    raxIterator it;
    raxStart(&it, rt);
    TEST_ASSERT(raxSeek(&it, "^", NULL, 0) == 1);
    size_t seen = 0;
    while (raxNext(&it)) {
        TEST_ASSERT(seen < sizeof(want) / sizeof(want[0]));
        if (seen < sizeof(want) / sizeof(want[0])) {
            TEST_ASSERT(it.key_len == strlen(want[seen]));
            TEST_ASSERT(memcmp(it.key, want[seen], it.key_len) == 0);
        }
        seen++;
    }
    TEST_ASSERT(seen == sizeof(want) / sizeof(want[0]));
    TEST_ASSERT(raxEOF(&it));
    raxStop(&it);

    raxFree(rt);
}
