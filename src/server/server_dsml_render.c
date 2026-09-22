/* server_dsml_render.c — 机械拆分自 ds4_server.c (2421-2879 行): DSML 参数渲染与 chat 提示词渲染。 */

#include "server_internal.h"

static void json_args_free(json_args *args) {
    for (int i = 0; i < args->len; i++) {
        free(args->v[i].key);
        free(args->v[i].value);
    }
    free(args->v);
    memset(args, 0, sizeof(*args));
}

static void json_args_push(json_args *args, json_arg arg) {
    if (args->len == args->cap) {
        args->cap = args->cap ? args->cap * 2 : 8;
        args->v = xrealloc(args->v, (size_t)args->cap * sizeof(args->v[0]));
    }
    args->v[args->len++] = arg;
}

static int json_args_find_unused(json_args *args, const char *key) {
    if (!key) return -1;
    for (int i = 0; i < args->len; i++) {
        if (!args->v[i].used && args->v[i].key && !strcmp(args->v[i].key, key)) return i;
    }
    return -1;
}

static bool json_args_parse(const char *json, json_args *args) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        bool is_string = false;
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') goto bad;
        p++;
        json_ws(&p);
        if (*p == '"') {
            is_string = true;
            if (!json_string(&p, &value)) goto bad;
        } else {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) goto bad;
            value = json_minify_raw_value(raw);
            free(raw);
        }

        json_arg arg = {.key = key, .value = value, .is_string = is_string};
        json_args_push(args, arg);
        key = value = NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
        continue;
bad:
        free(key);
        free(value);
        json_args_free(args);
        return false;
    }
    if (*p != '}') {
        json_args_free(args);
        return false;
    }
    return true;
}

static void append_dsml_attr_escaped(buf *b, const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (*s == '&') buf_puts(b, "&amp;");
        else if (*s == '<') buf_puts(b, "&lt;");
        else if (*s == '>') buf_puts(b, "&gt;");
        else if (*s == '"') buf_puts(b, "&quot;");
        else buf_putc(b, *s);
    }
}

static void append_dsml_parameter_text(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

void append_tool_result_text(buf *b, const char *s) {
    /* Tool output is data.  DeepSeek's renderer keeps it as ordinary text inside
     * <tool_result>...</tool_result>, so preserving literal '<', '>' and '&' is
     * important for read-file tools and shell output.  The only delimiter we must
     * protect is the wrapper's own closing tag; otherwise a file containing that
     * exact sentinel would terminate the result early. */
    const char *end = "</tool_result>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static void append_dsml_json_literal(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "\\u003c");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static void append_dsml_arg(buf *b, const json_arg *arg) {
    buf_puts(b, "<｜DSML｜parameter name=\"");
    append_dsml_attr_escaped(b, arg->key);
    buf_puts(b, "\" string=\"");
    buf_puts(b, arg->is_string ? "true" : "false");
    buf_puts(b, "\">");
    if (arg->is_string) append_dsml_parameter_text(b, arg->value);
    else append_dsml_json_literal(b, arg->value);
    buf_puts(b, "</｜DSML｜parameter>\n");
}

bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) return false;
    if (order) {
        for (int i = 0; i < order->len; i++) {
            int idx = json_args_find_unused(&args, order->prop[i]);
            if (idx < 0) continue;
            append_dsml_arg(b, &args.v[idx]);
            args.v[idx].used = true;
        }
    }
    for (int i = 0; i < args.len; i++) {
        if (args.v[i].used) continue;
        append_dsml_arg(b, &args.v[i]);
    }
    json_args_free(&args);
    return true;
}

static void append_json_arg_pair(buf *b, const json_arg *arg) {
    json_escape(b, arg->key);
    buf_puts(b, ":");
    if (arg->is_string) json_escape(b, arg->value);
    else buf_puts(b, arg->value);
}

void append_json_object_or_empty(buf *b, const char *json) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) {
        buf_puts(b, "{}");
        return;
    }
    buf_putc(b, '{');
    bool wrote = false;
    for (int i = 0; i < args.len; i++) {
        if (wrote) buf_putc(b, ',');
        append_json_arg_pair(b, &args.v[i]);
        wrote = true;
    }
    buf_putc(b, '}');
    json_args_free(&args);
}

void append_dsml_tool_calls_text(buf *b, const tool_calls *calls) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_dsml && calls->raw_dsml[0]) {
        buf_puts(b, calls->raw_dsml);
        return;
    }
    buf_puts(b, "\n\n<｜DSML｜tool_calls>\n");
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        buf_puts(b, "<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(b, tc->name);
        buf_puts(b, "\">\n");
        if (!append_dsml_arguments_from_json(b, tc->arguments, NULL)) {
            buf_puts(b, "<｜DSML｜parameter name=\"arguments\" string=\"true\">");
            append_dsml_parameter_text(b, tc->arguments);
            buf_puts(b, "</｜DSML｜parameter>\n");
        }
        buf_puts(b, "</｜DSML｜invoke>\n");
    }
    buf_puts(b, "</｜DSML｜tool_calls>");
}

bool role_is_system(const char *role) {
    return !strcmp(role, "system") || !strcmp(role, "developer");
}

static bool role_is_user_like(const char *role) {
    return !strcmp(role, "user") || !strcmp(role, "tool") || !strcmp(role, "function");
}

bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas) {
    if (tool_schemas && tool_schemas[0]) return true;
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if ((!strcmp(m->role, "assistant") && m->calls.len > 0) ||
            !strcmp(m->role, "tool") || !strcmp(m->role, "function"))
        {
            return true;
        }
    }
    return false;
}

/* ★base-native 渲染(2026-07-14, DS4_BASE_NATIVE=1)★
 * go-onebit 的底模是 BASE 模型: 从没见过 <｜User｜>/<｜Assistant｜>/<think> 这套 chat 角色帧,
 * 一套上就出符号汤(实测: 同一"写个 Go 加法"任务, chat 帧=汤 / 裸续写=正确代码)。
 * 本模式把 agent 对话渲染成 base 的母语——带注释的源文件式纯文本, 结尾停在"待续写"处:
 *   [system/tools]      → # 顶部注释块
 *   user / tool_result  → # User: / # Tool result: 注释
 *   assistant           → 正文(含其 DSML 工具调用原样)
 *   收尾                → "# Assistant:\n" 让模型接着写(工具调用帧仍由既有引导采样强制)
 * DSML 工具调用语法本身保留(引导采样已证可强制); 只换对话骨架。 */
static char *render_chat_prompt_base_native(const chat_msgs *msgs, const char *tool_schemas,
                                             size_t *conv_off) {
    buf out = {0};
    buf_puts(&out, "<｜begin▁of▁sentence｜>");
    if (tool_schemas && tool_schemas[0]) {
        buf sys = {0};
        append_tools_prompt_text(&sys, tool_schemas);
        buf_puts(&out, sys.ptr ? sys.ptr : "");
        buf_puts(&out, "\n\n");
        buf_free(&sys);
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        buf_puts(&out, m->content ? m->content : "");
        buf_puts(&out, "\n\n");
    }
    /* knowledge-primer: 用末条 user 消息检索最相关参考块, 注入 header (system 后、
     * 对话前 → KV 前缀友好且落在续写锚上游)。命中零重叠不注入(避免噪声)。 */
    bool knowledge_hit = false;
    /* 知识注入只在纯问答(无工具)时开: agent 任务有自身上下文不需百科参考, 且
     * tools+soul+knowledge 三层叠大 prompt 会触内存压力安全中止(2026-07-23 g2 实证)。 */
    if (g_knowledge_n > 0 && !(tool_schemas && tool_schemas[0])) {
        const char *last_user = NULL;
        for (int i = msgs->len - 1; i >= 0; i--)
            if (!strcmp(msgs->v[i].role, "user")) { last_user = msgs->v[i].content; break; }
        const char *ref = knowledge_retrieve(last_user);
        if (ref) {
            buf_puts(&out, "# Reference (use this to answer accurately):\n");
            buf_puts(&out, ref);
            buf_puts(&out, "\n\n");
            knowledge_hit = true;
        }
    }
    if (conv_off) *conv_off = out.len;   /* header(tools+soul+system+knowledge)止于此 */
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) continue;
        if (!strcmp(m->role, "user")) {
            buf_puts(&out, "# User:\n");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "\n\n");
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            buf_puts(&out, "# Tool result:\n");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "\n\n");
        } else if (!strcmp(m->role, "assistant")) {
            buf_puts(&out, "# Assistant:\n");
            buf_puts(&out, m->content ? m->content : "");
            append_dsml_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "\n\n");
        }
    }
    buf_puts(&out, "# Assistant:\n");
    /* 续写锚: base 惯性会"评论任务"而非动手 → 锚把续写钉进干活分布。 */
    {   /* 有 tools 时不设锚: 工具调用帧归 --tool-primer 引导采样接管(server 注入全部
         * 结构 token, 模型只填值) —— 1-bit 下 DSML 特殊 token 会被采成汉字, 结构必须
         * 由 server 强制。无 tools 时默认代码块锚(把 base 的"评论惯性"钉进干活分布)。 */
        const char *anchor = NULL;
        /* knowledge 命中 → 散文答问锚(不是代码块): 否则 ```go 锚把知识问答顶进
         * `func xxx(){ // 抄参考 }` 代码框(2026-07-23 g1 实证)。参考已在 header,
         * 这里只给"用参考直接答"的散文起手, 让续写落到答案分布而非代码分布。 */
        {
            /* knowledge 命中→答问脚手架(不是空锚: 空锚下 base 会回显问题,
             * singleflight/read 实证; 引导词把续写钉进"陈述答案"分布)。 */
            if (knowledge_hit) anchor = "Based on the reference: ";
            else if (g_req_mode == 2) anchor = "Answer: ";   /* qa 声明无命中: 散文答问锚(禁代码框) */
            else if (tool_schemas && tool_schemas[0]) anchor = "";
            else {
                /* 语言感知代码锚(2026-07-25): 硬编码 ```go 会把 Python/Rust 任务带偏
                 * (真实编码探针实证链的收尾修)。按用户文本关键词选围栏, 未识别默认 go。 */
                anchor = "```go\n";
                const char *last_user = NULL;
                for (int i = msgs->len - 1; i >= 0; i--)
                    if (!strcmp(msgs->v[i].role, "user")) { last_user = msgs->v[i].content; break; }
                if (last_user) {
                    static const struct { const char *kw, *fence; } LK[] = {
                        { "python", "```python\n" }, { "rust", "```rust\n" },
                        { "typescript", "```typescript\n" }, { "javascript", "```javascript\n" },
                        { " java", "```java\n" }, { "sql", "```sql\n" },
                        { "shell", "```bash\n" }, { "bash", "```bash\n" },
                        { " c++", "```cpp\n" }, { " c ", "```c\n" },
                    };
                    for (size_t li = 0; li < sizeof(LK) / sizeof(LK[0]); li++) {
                        const char *p = last_user; bool hit = false;
                        for (; *p; p++) {
                            size_t n = strlen(LK[li].kw); size_t m = 0;
                            while (LK[li].kw[m] && tolower((unsigned char)p[m]) == LK[li].kw[m]) m++;
                            if (m == n) { hit = true; break; }
                        }
                        if (hit) { anchor = LK[li].fence; break; }
                    }
                }
            }
        }
        buf_puts(&out, anchor);
    }
    return buf_take(&out);
}

/* base-native 的边界: 母语骨架里没有 EOS 角色帧, base 会继续自问自答("# User:" 再来一轮)
 * → 服务端注入默认 stop, 任何客户端都拿到干净单轮(客户端自带的 stop_sequences 一并生效)。 */
void base_native_default_stops(stop_list *stops) {
    if (!g_base_native) return;
    stop_list_push(stops, xstrdup("\n# User:"));
    stop_list_push(stops, xstrdup("\n# Tool result:"));
    stop_list_push(stops, xstrdup("\n# Assistant:"));
}

char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     ds4_think_mode think_mode, size_t *conv_off) {
    (void)tool_orders;
    if (conv_off) *conv_off = 0;   /* 0=未知 → 消费端退回 strstr 启发式(chat 帧路径不变) */
    /* base 底模 (--base-native): 换母语骨架; 默认=既有 chat 帧, 字节不变 */
    if (g_base_native)
        return render_chat_prompt_base_native(msgs, tool_schemas, conv_off);
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    int last_user_idx = -1;
    buf system = {0};
    /* Render tool schemas before the client system content so
     * --kv-cache-boundary-trim-tokens chops a dynamic tail from the client
     * message instead of the much larger tool-schema region. */
    if (tool_schemas && tool_schemas[0]) {
        append_tools_prompt_text(&system, tool_schemas);
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        if (system.len) buf_puts(&system, "\n\n");
        buf_puts(&system, m->content ? m->content : "");
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_user_like(m->role)) last_user_idx = i;
    }

    buf out = {0};
    buf_puts(&out, "<｜begin▁of▁sentence｜>");
    /* ★V4.1 官方 encoding.py★: 有 system 正文或 thinking 的 effort 前缀 ⇒ 先写 <｜System｜>(一次), 再前缀, 再 system 正文。
     * 2026-09-21 实撞: 这里原来是 V4 写法(BOS 后直接 system 正文, 前缀只在 max 档且是一段英文长段落), V4.1 的 tokenizer
     * 新增了 <｜System｜>(id 128799), 漏掉它 = 每条带 system 的产品请求都跑在模型没训练过的格式上(bug.md §1)。
     * V4 的 tokenizer 没这个 token, 那时 ds4_chat_system_token() 是空串, 渲染退回原样。金标单测: test_render_matches_official_v41_encoding。 */
    const char *effort = ds4_think_effort_prefix(think_mode);
    if (effort[0] || system.len) buf_puts(&out, ds4_chat_system_token());
    buf_puts(&out, effort);
    buf_puts(&out, system.ptr ? system.ptr : "");

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<｜User｜>");
            buf_puts(&out, m->content ? m->content : "");
            pending_assistant = true;
            pending_tool_result = false;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            if (!pending_tool_result) buf_puts(&out, "<｜User｜>");
            buf_puts(&out, "<tool_result>");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                buf_puts(&out, "<｜Assistant｜>");
                if (think) {
                    if (tool_context || i > last_user_idx) {
                        buf_puts(&out, "<think>");
                        buf_puts(&out, m->reasoning ? m->reasoning : "");
                        buf_puts(&out, "</think>");
                    } else {
                        buf_puts(&out, "</think>");
                    }
                } else {
                    buf_puts(&out, "</think>");
                }
            }
            buf_puts(&out, m->content ? m->content : "");
            append_dsml_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (pending_assistant) {
        buf_puts(&out, "<｜Assistant｜>");
        buf_puts(&out, think ? "<think>" : "</think>");
    }

    buf_free(&system);
    return buf_take(&out);
}
