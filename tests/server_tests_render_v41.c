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

/* ---- 带工具的对话: 金标 = 官方 encoding.py encode_messages 实跑(2026-10-11) ----
 * case1 就是官方仓库 encoding/tests/test_input_1.json, 期望字节与它自带的 test_output_1.txt 逐字节相同。
 * case2 自造, 覆盖 case1 没有的: 并行调用、零参数调用、非字符串参数(json.dumps 形状)、值里的 < &、
 * 两条连续工具结果 + 后续用户文本并进同一个 <｜User｜>、客户端 schema 带 \u 转义和换行缩进。
 * 起因 issue fodelf/YoungAi#1: 服务端拿旧写法 <｜DSML｜tool_calls> 教模型, 模型写到一半漂成 <parameter>,
 * 工具调用全部退回成正文。 */

static void push_msg(chat_msgs *msgs, const char *role, const char *content,
                     const char *reasoning, const char *const *calls) {
    chat_msg m = {0};
    m.role = xstrdup(role);
    m.content = xstrdup(content);
    if (reasoning) m.reasoning = xstrdup(reasoning);
    for (int i = 0; calls && calls[i]; i += 2) {
        tool_call tc = {0};
        tc.name = xstrdup(calls[i]);
        tc.arguments = xstrdup(calls[i + 1]);
        tool_calls_push(&m.calls, tc);
    }
    chat_msgs_push(msgs, m);
}

static void expect_render_tools(chat_msgs *msgs, const char *schemas, ds4_think_mode mode, const char *want) {
    char *p = render_chat_prompt_text(msgs, schemas, NULL, mode, NULL);
    TEST_ASSERT(p != NULL);
    if (p && strcmp(p, want) != 0) {
        size_t i = 0;
        while (p[i] && p[i] == want[i]) i++;
        fprintf(stderr, "render mismatch at byte %zu\n  want: %.120s\n  got:  %.120s\n", i, want + i, p + i);
        TEST_ASSERT(!"tool prompt differs from official encoding.py");
    }
    free(p);
}

#define SPACED_TOOLS_HEAD \
    "## Tools\n\n" \
    "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n" \
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n" \
    "<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n" \
    "...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\n" \
    "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n" \
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n" \
    "Otherwise, output directly after </think> with tool calls or final response.\n\n" \
    "### Available Tool Schemas\n\n"
#define SPACED_TOOLS_TAIL "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"

void test_render_matches_official_spaced_dsml_tools(void) {
    ds4_chat_set_system_token(true);

    chat_msgs c1 = {0};
    static const char *const call1[] = {"get_weather", "{\"location\": \"Beijing\", \"unit\": \"celsius\"}", NULL};
    push_msg(&c1, "system", "You are a helpful assistant.", NULL, NULL);
    push_msg(&c1, "user", "What's the weather like in Beijing?", NULL, NULL);
    push_msg(&c1, "assistant", "", "The user wants the weather in Beijing. I should call get_weather.", call1);
    push_msg(&c1, "tool", "{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}", NULL, NULL);
    push_msg(&c1, "assistant", "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.",
             "Got the weather data. Let me format a nice response.", NULL);
    /* 客户端常见的紧凑 JSON(parse_tools_value 拆出的 function 对象, 一行一个) */
    const char *schemas1 =
        "{\"name\":\"get_weather\",\"description\":\"Get the weather for a specific location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The city name\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"],\"description\":\"Temperature unit\"}},\"required\":[\"location\"]}}\n"
        "{\"name\":\"search\",\"description\":\"Search the web for information\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"Search query\"},\"num_results\":{\"type\":\"integer\",\"description\":\"Number of results to return\"}},\"required\":[\"query\"]}}";
    expect_render_tools(&c1, schemas1, DS4_THINK_HIGH,
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        "You are a helpful assistant.\n\n"
        SPACED_TOOLS_HEAD
        "{\"name\": \"get_weather\", \"description\": \"Get the weather for a specific location\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The city name\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"], \"description\": \"Temperature unit\"}}, \"required\": [\"location\"]}}\n"
        "{\"name\": \"search\", \"description\": \"Search the web for information\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"Search query\"}, \"num_results\": {\"type\": \"integer\", \"description\": \"Number of results to return\"}}, \"required\": [\"query\"]}}"
        SPACED_TOOLS_TAIL
        "<｜User｜>What's the weather like in Beijing?<｜Assistant｜><think>The user wants the weather in Beijing. I should call get_weather.</think>\n\n"
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
        "<｜DSML｜ parameter name=\"location\" string=\"true\">Beijing</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"unit\" string=\"true\">celsius</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>"
        "<｜User｜><tool_result>{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}</tool_result>"
        "<｜Assistant｜><think>Got the weather data. Let me format a nice response.</think>"
        "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.<｜end▁of▁sentence｜>");
    chat_msgs_free(&c1);

    chat_msgs c2 = {0};
    static const char *const call2[] = {
        "get_weather", "{\"location\":\"a<b & c\",\"days\":3,\"opts\":{\"a\":[1,2],\"b\":true}}",
        "now", "{}", NULL};
    push_msg(&c2, "system", "S", NULL, NULL);
    push_msg(&c2, "user", "U", NULL, NULL);
    push_msg(&c2, "assistant", "", NULL, call2);
    push_msg(&c2, "tool", "R1", NULL, NULL);
    push_msg(&c2, "tool", "R2", NULL, NULL);
    push_msg(&c2, "user", "U2", NULL, NULL);
    const char *schemas2 =
        "{\n  \"name\": \"get_weather\",\n  \"description\": \"Weather \\u00b0C\",\n"
        "  \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"location\"]}\n}\n"
        "{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}";
    expect_render_tools(&c2, schemas2, DS4_THINK_NONE,
        "<｜begin▁of▁sentence｜><｜System｜>S\n\n"
        SPACED_TOOLS_HEAD
        "{\"name\": \"get_weather\", \"description\": \"Weather °C\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"location\"]}}\n"
        "{\"name\": \"now\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}"
        SPACED_TOOLS_TAIL
        "<｜User｜>U<｜Assistant｜></think>\n\n"
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
        "<｜DSML｜ parameter name=\"location\" string=\"true\">a<b & c</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"opts\" string=\"false\">{\"a\": [1, 2], \"b\": true}</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"now\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>"
        "<｜User｜><tool_result>R1</tool_result>\n\n<tool_result>R2</tool_result>\n\nU2<｜Assistant｜></think>");
    chat_msgs_free(&c2);

    ds4_chat_set_system_token(false);
}

/* 模型按官方写法吐出的工具块必须解析成结构化调用; 值原样(官方不做实体转义, "&lt;" 是值本身),
 * 只有渲染侧为闭合标签加的那个 "&lt;" 要还原 —— 渲染 → 解析往返必须拿回原值。 */
void test_parse_spaced_dsml_tool_calls(void) {
    const char *text =
        "Sure.\n\n" DS4_TOOL_CALLS_START_SPACED "\n"
        DS4_INVOKE_START_SPACED " name=\"bash\">\n"
        DS4_PARAM_START_SPACED " name=\"command\" string=\"true\">echo '&lt;b&gt;' && ls</｜DSML｜ parameter>\n"
        DS4_PARAM_START_SPACED " name=\"n\" string=\"false\">3" DS4_PARAM_END_SPACED "\n"
        DS4_INVOKE_END_SPACED "\n" DS4_TOOL_CALLS_END_SPACED;
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(text, false, &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, "Sure."));
    TEST_ASSERT(calls.len == 1);
    if (calls.len == 1) {
        TEST_ASSERT(!strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(!strcmp(calls.v[0].arguments, "{\"command\": \"echo '&lt;b&gt;' && ls\", \"n\": 3}"));
        TEST_ASSERT(calls.raw_dsml && !strncmp(calls.raw_dsml, "\n\n" DS4_TOOL_CALLS_START_SPACED, 2 + strlen(DS4_TOOL_CALLS_START_SPACED)));
    }
    free(content); free(reasoning); tool_calls_free(&calls);

    ds4_chat_set_system_token(true);
    tool_calls orig = {0};
    tool_call tc = {0};
    tc.name = xstrdup("write");
    tc.arguments = xstrdup("{\"body\":\"x</｜DSML｜ parameter>y &amp; z\"}");
    tool_calls_push(&orig, tc);
    buf b = {0};
    append_dsml_tool_calls_text(&b, &orig);
    content = reasoning = NULL;
    tool_calls back = {0};
    TEST_ASSERT(parse_generated_message_ex(b.ptr, false, &content, &reasoning, &back));
    TEST_ASSERT(back.len == 1);
    if (back.len == 1) TEST_ASSERT(!strcmp(back.v[0].arguments, "{\"body\": \"x</｜DSML｜ parameter>y &amp; z\"}"));
    free(content); free(reasoning); tool_calls_free(&back); tool_calls_free(&orig); ds4_buf_free(&b);
    ds4_chat_set_system_token(false);
}

/* 工具语法位贪心的判定(v41_syntax_greedy): 标签/参数头处 = 贪心, string 值体里 = 照采样, 思考段里 = 不管, 块外正文 = 不管 */
static int syntax_greedy_after(const char *text, bool thinking_inside) {
    v41_gen g;
    memset(&g, 0, sizeof g);
    static job j;   /* 判定只读 req.kind / req.has_tools */
    j.req.kind = REQ_CHAT;
    j.req.has_tools = true;
    g.j = &j;
    dsml_decode_tracker_init(&g.dsml_tracker);
    g.thinking.inside = thinking_inside;
    ds4_buf_puts(&g.text, text);
    dsml_decode_tracker_update(&g.dsml_tracker, g.text.ptr, g.text.len);
    const int r = v41_syntax_greedy(&g, NULL, 0);
    TEST_ASSERT(g.text.len == strlen(text));   /* 往前看不能改动已吐文本 */
    ds4_buf_free(&g.text);
    return r;
}

void test_syntax_greedy_spaced_dsml(void) {
    const char *head = "Listing.\n\n" DS4_TOOL_CALLS_START_SPACED "\n" DS4_INVOKE_START_SPACED " name=\"bash\">\n";
    TEST_ASSERT(syntax_greedy_after("Listing.", false) == 0);
    TEST_ASSERT(syntax_greedy_after(head, false) == 1);
    TEST_ASSERT(syntax_greedy_after(head, true) == 0);
    char buf[512];
    snprintf(buf, sizeof buf, "%s" DS4_PARAM_START_SPACED " name=\"command\" string=\"true\">ls -la", head);
    TEST_ASSERT(syntax_greedy_after(buf, false) == 0);   /* 参数值: 照请求采样 */
    snprintf(buf, sizeof buf, "%s" DS4_PARAM_START_SPACED " name=\"n\" string=\"false\">", head);
    TEST_ASSERT(syntax_greedy_after(buf, false) == 1);   /* JSON 值的结构位(数字/括号): 贪心 */
    snprintf(buf, sizeof buf, "%s" DS4_PARAM_START_SPACED " name=\"command\" string=\"true\">ls" DS4_PARAM_END_SPACED "\n", head);
    TEST_ASSERT(syntax_greedy_after(buf, false) == 1);   /* 值关了, 回到标签位 */
}

/* OpenAI 流式: 带空格写法也要逐段吐 tool_calls 增量, 原始 DSML 一个字节都不能漏进 content */
void test_openai_stream_spaced_dsml_tool_call(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;
    openai_stream st;
    openai_stream_start(&r, &st);
    const char *part =
        "Listing.\n\n" DS4_TOOL_CALLS_START_SPACED "\n" DS4_INVOKE_START_SPACED " name=\"bash\">\n"
        DS4_PARAM_START_SPACED " name=\"command\" string=\"true\">ls -la";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_spaced", &st, part, strlen(part), false));
    const char *full =
        "Listing.\n\n" DS4_TOOL_CALLS_START_SPACED "\n" DS4_INVOKE_START_SPACED " name=\"bash\">\n"
        DS4_PARAM_START_SPACED " name=\"command\" string=\"true\">ls -la /tmp" DS4_PARAM_END_SPACED "\n"
        DS4_INVOKE_END_SPACED "\n" DS4_TOOL_CALLS_END_SPACED;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_spaced", &st, full, strlen(full), false));
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(full, false, &content, &reasoning, &calls));
    apply_openai_stream_tool_ids(&calls, &st);
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_spaced", &st, full, strlen(full), &calls, "tool_calls", 10, 4));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"content\":\"Listing.\"") != NULL);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"ls -la\"") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\" /tmp\"") != NULL);
    TEST_ASSERT(strstr(out, "\"finish_reason\":\"tool_calls\"") != NULL);
    TEST_ASSERT(strstr(out, "DSML") == NULL);
    free(out); free(content); free(reasoning);
    tool_calls_free(&calls);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}
