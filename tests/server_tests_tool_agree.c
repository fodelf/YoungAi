/* server_tests_tool_agree.c — 流式工具调用与最终解析必须是同一个结论(2026-10-16)。
 * 起因: spark 真模型写了 <｜DSML｜ invoke name="read_file"> + 裸 </invoke>。流式投影已经发出调用头和 "{",
 * 最终解析判整块无效退回正文, 收尾却报正常结束 —— 客户端拿到一个参数为半截 "{" 的调用, 服务端自己当它是正文(trace request 356)。
 * 修法两条: 标签规则两边同一套(server_dsml_alias.c, 认裸 </invoke>); 流上已露出的调用与最终解析对不上就发 error 事件作废这一轮。
 * 工具块仍然边生成边投影: Claude Code 的流看门狗只认真事件(ping 连续 30 个后不再续命), 写长文件的块不能扣到收尾。
 * 每个用例逐字节喂(= 标签切在任意两个 token 之间), 核:
 *   认下  → 流上拼出的调用与参数 = 最终解析(压成线上紧凑写法)逐字节相同, 没有 error;
 *   没认下 → 流上已露出调用: 收尾是 error 事件, 不许报调用成功; 没露出: 原文补成正文, 没有调用也没有 error。 */
#include "server_tests_internal.h"

#define S_CS DS4_TOOL_CALLS_START_SPACED
#define S_CE DS4_TOOL_CALLS_END_SPACED
#define S_IS DS4_INVOKE_START_SPACED
#define S_IE DS4_INVOKE_END_SPACED
#define S_PS DS4_PARAM_START_SPACED
#define S_PE DS4_PARAM_END_SPACED

enum { WANT_CALLS, WANT_ERROR, WANT_TEXT };   /* 最终该是: 调用 / 流上已露出调用所以报错 / 整块退回正文 */

typedef struct {
    const char *name, *raw;
    int want, ncalls;
    const char *want_args;   /* WANT_CALLS: 所有调用参数按序拼起来(线上紧凑写法) */
} agree_case;

static const agree_case CASES[] = {
    { "one call", "Listing.\n\n" S_CS "\n" S_IS " name=\"list_dir\">\n" S_PS " name=\"path\" string=\"true\">docs" S_PE "\n" S_IE "\n" S_CE,
      WANT_CALLS, 1, "{\"path\":\"docs\"}" },
    { "two calls", S_CS "\n" S_IS " name=\"read\">\n" S_PS " name=\"path\" string=\"true\">a.c" S_PE "\n" S_IE "\n"
      S_IS " name=\"bash\">\n" S_PS " name=\"command\" string=\"true\">wc -l a.c" S_PE "\n" S_IE "\n" S_CE,
      WANT_CALLS, 2, "{\"path\":\"a.c\"}{\"command\":\"wc -l a.c\"}" },
    /* 10-16 原样字节: 不带参数的 invoke, 闭标签写成裸 </invoke> —— 两边都认下, 参数 {} */
    { "bare invoke end", "我按要求做一次这个调用，然后把原样结果给你。\n\n" S_CS "\n" S_IS " name=\"read_file\">\n\n</invoke>\n" S_CE,
      WANT_CALLS, 1, "{}" },
    /* 10-11 的裸参数标签 + 10-16 的裸 invoke 闭标签一起出现 */
    { "bare param and invoke", S_CS "\n" S_IS " name=\"web_search\">\n<parameter name=\"query\" string=\"true\">DGX Spark</parameter>\n</invoke>\n" S_CE,
      WANT_CALLS, 1, "{\"query\":\"DGX Spark\"}" },
    /* 11-26 原样形状: 模型在 DSML 块里写伪 XML 标签 <lightpanda_search">/<query>/</query>/</lightpanda_search>
     * (invoke 名字标签, 名字后一个多余 " 再 >; 参数名标签无属性) —— 两边都认下 */
    { "pseudo xml tags", "查一下这个。\n\n" S_CS "\n<lightpanda_search\">\n<query>best laptop price</query>\n</lightpanda_search>\n" S_CE,
      WANT_CALLS, 1, "{\"query\":\"best laptop price\"}" },
    { "utf8 and json-special chars", S_CS "\n" S_IS " name=\"bash\">\n" S_PS " name=\"command\" string=\"true\">echo \"你好\" & ls\\n" S_PE "\n"
      S_PS " name=\"timeout\" string=\"false\">30" S_PE "\n" S_IE "\n" S_CE,
      WANT_CALLS, 1, "{\"command\":\"echo \\\"你好\\\" & ls\\\\n\",\"timeout\":30}" },
    /* 10-18 17:22 原样形状(1017 底座): content 值写完直接块闭合, 参数/invoke 闭标签都没写 —— 隐式闭合, 两边都认下 */
    { "calls end closes param", "我把动画写成一份 HTML。\n\n" S_CS "\n" S_IS " name=\"write_file\">\n"
      S_PS " name=\"path\" string=\"true\">pelican-bicycle.html" S_PE "\n"
      S_PS " name=\"content\" string=\"true\"><!DOCTYPE html>\n<p>鹈鹕</p>\n</html>" S_CE,
      WANT_CALLS, 1, "{\"path\":\"pelican-bicycle.html\",\"content\":\"<!DOCTYPE html>\\n<p>鹈鹕</p>\\n</html>\"}" },
    /* 写坏且流上已露出调用头: 参数没闭合 / 参数之间夹了别的标签 / 第一个调用好、第二个坏 */
    { "unclosed param", "Trying.\n\n" S_CS "\n" S_IS " name=\"read_file\">\n" S_PS " name=\"path\" string=\"true\">a.txt\n" S_IE "\n" S_CE,
      WANT_ERROR, 0, NULL },
    { "junk between params", S_CS "\n" S_IS " name=\"read_file\">\n<oops/>\n" S_IE "\n" S_CE,
      WANT_ERROR, 0, NULL },
    { "good then bad", S_CS "\n" S_IS " name=\"read\">\n" S_PS " name=\"path\" string=\"true\">a.c" S_PE "\n" S_IE "\n"
      S_IS " name=\"read_file\">\n<oops/>\n" S_IE "\n" S_CE,
      WANT_ERROR, 0, NULL },
    /* 写坏但还没露出任何调用(块头之后直接是别的标签): 整块退回正文 */
    { "junk before any invoke", "Hmm.\n\n" S_CS "\n<oops name=\"read_file\"/>\n" S_CE,
      WANT_TEXT, 0, NULL },
};
#define NCASES ((int)(sizeof CASES / sizeof CASES[0]))

static int pair_big(int sv[2]) {   /* 逐字节喂上百帧, 到最后才读: 缓冲放大到 4 MB, 不然 macOS 8 KB 写满就卡死(同 server_tests_stray_think.c) */
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) return -1;
    const int big = 4 << 20;
    for (int i = 0; i < 2; i++) { setsockopt(sv[i], SOL_SOCKET, SO_SNDBUF, &big, sizeof big); setsockopt(sv[i], SOL_SOCKET, SO_RCVBUF, &big, sizeof big); }
    return 0;
}

/* 最终解析, 照 server_generate_v41.c 收尾的口径(has_tools + 见过工具块起点); 返回 recover(非 NULL = 整块退回正文) */
static const char *final_verdict(const char *raw, tool_calls *calls, char **content, char **reasoning, const char **finish) {
    char err[64] = ""; bool recovered = false;
    *finish = "stop";
    parse_generated_message_for_response(raw, true, true, false, finish, err, sizeof err, content, reasoning, calls, &recovered);
    if (calls->len) *finish = "tool_calls";
    return recovered ? *content : NULL;
}

/* 最终解析的参数按序拼起来, 压成线上的紧凑写法(内部记的是 {"path": "docs"}, 发出去一律紧凑, 同 append_json_object_string) */
static char *calls_args(const tool_calls *calls) {
    ds4_buf b = {0};
    for (int i = 0; i < calls->len; i++) { char *m = json_minify_raw_value(calls->v[i].arguments ? calls->v[i].arguments : "{}"); ds4_buf_puts(&b, m ? m : ""); free(m); }
    ds4_buf_putc(&b, 0);
    return b.ptr;
}

static char *concat2(const char *a, const char *b) { ds4_buf o = {0}; ds4_buf_puts(&o, a); ds4_buf_puts(&o, b); ds4_buf_putc(&o, 0); return o.ptr; }

static void check(const agree_case *c, const char *api, const char *live, const char *tail, const tool_calls *calls,
                  const char *args_key, const char *text_key, const char *tool_done) {
    char *all = concat2(live, tail);
    const bool error = strstr(tail, "event: error") != NULL;
    if (c->want == WANT_CALLS) {
        char *got = sse_join_field(all, args_key), *want = calls_args(calls);
        if (calls->len != c->ncalls || strcmp(got, want) || strcmp(got, c->want_args) || error)
            fprintf(stderr, "tool_agree %s [%s]: final %d calls %s, streamed %s, want %s, error=%d\n", api, c->name, calls->len, want, got, c->want_args, error);
        TEST_ASSERT(calls->len == c->ncalls);
        TEST_ASSERT(!strcmp(got, want));            /* 流上的 = 最终解析的 */
        TEST_ASSERT(!strcmp(got, c->want_args));
        TEST_ASSERT(!error);
        TEST_ASSERT(strstr(all, tool_done) != NULL);
        free(got); free(want);
    } else if (c->want == WANT_ERROR) {
        if (!error || strstr(tail, tool_done)) fprintf(stderr, "tool_agree %s [%s]: want error event, got tail:\n%s\n", api, c->name, tail);
        TEST_ASSERT(calls->len == 0);
        TEST_ASSERT(error);
        TEST_ASSERT(strstr(tail, tool_done) == NULL);   /* 不许报调用成功 */
    } else {
        char *text = sse_join_field(all, text_key);
        if (error || strstr(all, tool_done) || !strstr(text, S_CS)) fprintf(stderr, "tool_agree %s [%s]: want text fallback, got:\n%s\n", api, c->name, all);
        TEST_ASSERT(calls->len == 0);
        TEST_ASSERT(!error);
        TEST_ASSERT(strstr(all, tool_done) == NULL);
        TEST_ASSERT(strstr(text, S_CS) != NULL);   /* 原文补成了正文, 与非流式 content 同 */
        free(text);
    }
    free(all);
}

static void run_openai(const agree_case *c) {
    int a[2], b[2];
    TEST_ASSERT(pair_big(a) == 0 && pair_big(b) == 0);
    request r; request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI; r.stream = true; r.think_mode = DS4_THINK_NONE; r.has_tools = true;
    openai_stream st; openai_stream_start(&r, &st);
    const size_t n = strlen(c->raw); char *buf = malloc(n + 1); bool ok = true;
    for (size_t k = 1; k <= n && ok; k++) { memcpy(buf, c->raw, k); buf[k] = 0; ok = openai_sse_stream_update(a[0], NULL, &r, "c_agree", &st, buf, k, false); }
    free(buf);
    TEST_ASSERT(ok);
    shutdown(a[0], SHUT_WR);
    char *live = read_socket_text(a[1]);
    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL; const char *finish;
    const char *recover = final_verdict(c->raw, &calls, &content, &reasoning, &finish);
    apply_openai_stream_tool_ids(&calls, &st);   /* 照生产: 先继承流上已发的 id, 再补没有的 */
    assign_tool_call_ids(NULL, &calls, API_OPENAI);
    TEST_ASSERT(openai_sse_finish_live(b[0], NULL, &r, "c_agree", &st, c->raw, n, recover, &calls, finish, 10, 4));
    shutdown(b[0], SHUT_WR);
    char *tail = read_socket_text(b[1]);
    check(c, "openai", live, tail, &calls, "arguments", "content", "\"finish_reason\":\"tool_calls\"");
    for (int i = 0; c->want == WANT_CALLS && i < calls.len; i++) {   /* 流上发的 id 与收尾记住的同一个(客户端回 tool_result 用它) */
        char *all = concat2(live, tail); TEST_ASSERT(strstr(all, calls.v[i].id) != NULL); free(all);
    }
    free(live); free(tail); free(content); free(reasoning); tool_calls_free(&calls);
    openai_stream_free(&st); request_free(&r);
    close(a[0]); close(a[1]); close(b[0]); close(b[1]);
}

static void run_anthropic(const agree_case *c) {
    int a[2], b[2];
    TEST_ASSERT(pair_big(a) == 0 && pair_big(b) == 0);
    request r; request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC; r.stream = true; r.think_mode = DS4_THINK_NONE; r.has_tools = true;
    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(a[0], &r, "msg_agree", 7, &st));
    const size_t n = strlen(c->raw); char *buf = malloc(n + 1); bool ok = true;
    for (size_t k = 1; k <= n && ok; k++) { memcpy(buf, c->raw, k); buf[k] = 0; ok = anthropic_sse_stream_update(a[0], NULL, &r, "msg_agree", &st, buf, k, false); }
    free(buf);
    TEST_ASSERT(ok);
    shutdown(a[0], SHUT_WR);
    char *live = read_socket_text(a[1]);
    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL; const char *finish;
    const char *recover = final_verdict(c->raw, &calls, &content, &reasoning, &finish);
    apply_anthropic_stream_tool_ids(&calls, &st);
    assign_tool_call_ids(NULL, &calls, API_ANTHROPIC);
    TEST_ASSERT(anthropic_sse_finish_live(b[0], NULL, &r, "msg_agree", &st, c->raw, n, recover, &calls, finish, 5));
    shutdown(b[0], SHUT_WR);
    char *tail = read_socket_text(b[1]);
    check(c, "anthropic", live, tail, &calls, "partial_json", "text", "\"stop_reason\":\"tool_use\"");
    if (c->want == WANT_CALLS) {
        char *all = concat2(live, tail); int uses = 0;
        for (const char *p = all; (p = strstr(p, "\"type\":\"tool_use\"")); p++) uses++;
        TEST_ASSERT(uses == calls.len);   /* 一个调用一个 tool_use 块, 不重发 */
        for (int i = 0; i < calls.len; i++) TEST_ASSERT(strstr(all, calls.v[i].id) != NULL);
        free(all);
    }
    free(live); free(tail); free(content); free(reasoning); tool_calls_free(&calls);
    anthropic_stream_free(&st); request_free(&r);
    close(a[0]); close(a[1]); close(b[0]); close(b[1]);
}

void test_tool_stream_agrees_with_final_parse(void) {
    for (int i = 0; i < NCASES; i++) { run_openai(&CASES[i]); run_anthropic(&CASES[i]); }
}
