/* server_tests_render_v41.c — 聊天提示渲染对 DeepSeek V4.1 官方 encoding.py 的金标(2026-09-21)。
 *
 * 金标 = 官方 encoding/encoding.py 实跑(spark hf/DeepSeek-V4.1-Flash, `encode_messages(..., thinking_mode=...)`),
 * 四种对话的渲染字节原样抄进来。这里再漂一个字节 = 模板又跟官方错开 —— 09-21 实撞: 漏了 V4.1 新增的
 * <｜System｜>(id 128799), 产品路每条带 system 的请求都跑在模型没训练过的格式上(bug.md §1)。
 * 会话中途的 system 消息(官方原位 <｜System｜>mid, 并触发 assistant 头)还没照官方做, 不在这份金标里。
 * 离线单测没有引擎: 用 ds4_chat_set_system_token 模拟"tokenizer 有/没有 <｜System｜>"两种装载态。 */

#include "server_tests_internal.h"

static chat_msgs two_msgs(const char *system, const char *user) {
    chat_msgs msgs = {0};
    if (system) {
        chat_msg sys = {0};
        sys.role = xstrdup("system");
        sys.content = xstrdup(system);
        chat_msgs_push(&msgs, sys);
    }
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup(user);
    chat_msgs_push(&msgs, u);
    return msgs;
}

static void expect_render(chat_msgs *msgs, ds4_think_mode mode, const char *want) {
    char *p = render_chat_prompt_text(msgs, NULL, NULL, mode, NULL);
    TEST_ASSERT(p != NULL);
    if (p && strcmp(p, want) != 0) {
        fprintf(stderr, "render mismatch\n  want: %s\n  got:  %s\n", want, p);
        TEST_ASSERT(!"rendered prompt differs from official encoding.py");
    }
    free(p);
}

void test_render_matches_official_v41_encoding(void) {
    ds4_chat_set_system_token(true);   /* V4.1 tokenizer: 有 <｜System｜> */
    chat_msgs with_sys = two_msgs("S", "U");
    expect_render(&with_sys, DS4_THINK_NONE,
                  "<｜begin▁of▁sentence｜><｜System｜>S<｜User｜>U<｜Assistant｜></think>");
    expect_render(&with_sys, DS4_THINK_HIGH,
                  "<｜begin▁of▁sentence｜><｜System｜>"
                  "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                  "S<｜User｜>U<｜Assistant｜><think>");
    chat_msgs_free(&with_sys);

    chat_msgs only_user = two_msgs(NULL, "U");
    expect_render(&only_user, DS4_THINK_NONE,
                  "<｜begin▁of▁sentence｜><｜User｜>U<｜Assistant｜></think>");
    expect_render(&only_user, DS4_THINK_HIGH,
                  "<｜begin▁of▁sentence｜><｜System｜>"
                  "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                  "<｜User｜>U<｜Assistant｜><think>");
    chat_msgs_free(&only_user);

    /* V4 的 tokenizer 没有 <｜System｜>: 渲染退回 BOS 后直接 system 正文(V4 的模板本来就这样) */
    ds4_chat_set_system_token(false);
    chat_msgs v4 = two_msgs("S", "U");
    expect_render(&v4, DS4_THINK_NONE, "<｜begin▁of▁sentence｜>S<｜User｜>U<｜Assistant｜></think>");
    chat_msgs_free(&v4);
}
