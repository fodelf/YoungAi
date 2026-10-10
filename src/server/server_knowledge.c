/* server_knowledge.c — 机械拆分自 ds4_server.c (2281-2420 行): 知识块加载/检索与工具提示词。 */

#include "server_internal.h"

/* --soul FILE: P3 行为示例文本, 注入 tools header 尾部 (每会话静态 → KV 前缀
 * 友好)。值污染史 (gate v2: 模型抄示例占位符) 已被 copyfix 结构性解除 —— 值
 * 拷贝源从首个 <｜User｜> 起, header 里的示例值抄不到; 示例只示范行为模式
 * (panic → 定位行 → 最小 Edit)。灵魂由数据定义, 引擎只供本注入机制。 */
char *g_soul_text;

/* ★knowledge-primer (2026-07-23, DS4_KNOWLEDGE_FILE / --knowledge)★
 * 知识环修复: 1-bit base 无据知识问答退化成复读/幻觉(g1/g4 面板实证); 前提探针证实
 * "参考塞进上下文→正确内容浮现"(g1 带参考逐字复现负缓存+布隆), 但 base 续写把
 * Reference/Question 当续写→抄或漂。解: 检索最相关参考块, 作 # Reference: 注入 header,
 * 让 base-native 的 # Assistant: 续写锚落到"用参考答问"分布。纯 prompt 层, 零模型/零体积。
 * 文件格式: 参考块以 "---" 单行分隔; 检索=末条 user 消息的词重叠打分(浅 BM25 类)。 */
static char **g_knowledge_blocks;

int g_knowledge_n;

void knowledge_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "ds4-server: --knowledge: 打不开 %s\n", path); return; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return; }
    char *all = xmalloc((size_t)sz + 1);
    if (fread(all, 1, (size_t)sz, fp) != (size_t)sz) { free(all); fclose(fp); return; }
    all[sz] = '\0'; fclose(fp);
    /* split on lines that are exactly "---" */
    char *p = all;
    while (p && *p) {
        char *sep = strstr(p, "\n---\n");
        char *blk;
        if (sep) { *sep = '\0'; blk = p; p = sep + 5; }
        else { blk = p; p = NULL; }
        while (*blk == '\n' || *blk == ' ') blk++;
        if (*blk) {
            g_knowledge_blocks = xrealloc(g_knowledge_blocks,
                                          (size_t)(g_knowledge_n + 1) * sizeof(char *));
            g_knowledge_blocks[g_knowledge_n++] = xstrdup(blk);
        }
    }
    free(all);
    fprintf(stderr, "ds4-server: knowledge-primer 载入 %d 参考块 (%s)\n", g_knowledge_n, path);
}

/* word-overlap 打分: query 的每个 ≥4 字符词在 block 里出现即 +1 (大小写不敏感的粗匹配)。
 * 返回最佳块指针 (NULL=无库或零重叠, 不注入以免噪声)。 */
/* 问句形判定(2026-07-25): knowledge-primer 只该救"无据知识问答", 祈使式编码任务
 * ("Write a.../Implement...")注入参考=劫持成答题框架 → 三个真实编码任务零代码实证
 * (trace: Python 任务被塞 go-vet 参考 + "Based on the reference:" 散文锚)。
 * 根因叠加: score≥1 + ≥3字符 让 and/the/each 等停用词单命中即注入 = 滥命中。
 * 修 = 问句形才允许检索(含 ? / 疑问词开头), 祈使式天然出局; g1/g4 知识针都是问句, 保住。 */
int g_req_mode = 0;   /* 当前请求的显式意图(request.ds4_mode 渲染前置位) */

static bool query_is_question(const char *q) {
    if (!q) return false;
    if (strchr(q, '?') || strstr(q, "？")) return true;
    while (*q && isspace((unsigned char)*q)) q++;
    static const char *qw[] = { "what", "how", "why", "when", "which", "where", "who",
                                "is ", "are ", "does ", "do ", "can ", "should ",
                                "什么", "如何", "为什么", "怎么", "是否", "哪" };
    for (size_t i = 0; i < sizeof(qw) / sizeof(qw[0]); i++) {
        size_t n = strlen(qw[i]);
        if (strncasecmp(q, qw[i], n) == 0) return true;
    }
    return false;
}

const char *knowledge_retrieve(const char *query) {
    if (g_knowledge_n == 0 || !query || !query[0]) return NULL;
    if (g_req_mode == 1) return NULL;             /* 接口声明 code → 零注入(硬保证) */
    if (g_req_mode != 2 && !query_is_question(query)) return NULL;   /* auto 兜底: 祈使式零注入; qa 声明则放行 */
    int best = -1, best_score = 0;
    size_t qn = strlen(query);
    for (int k = 0; k < g_knowledge_n; k++) {
        const char *blk = g_knowledge_blocks[k];
        int score = 0;
        size_t i = 0;
        while (i < qn) {
            while (i < qn && !isalnum((unsigned char)query[i])) i++;
            size_t j = i;
            while (j < qn && isalnum((unsigned char)query[j])) j++;
            if (j - i >= 3) {   /* ≥3: 让 "vet"/"sql" 等短判别词命中(2026-07-23 g4 miss 修复) */
                char word[64];
                size_t wl = j - i < 63 ? j - i : 63;
                for (size_t w = 0; w < wl; w++) word[w] = (char)tolower((unsigned char)query[i + w]);
                word[wl] = '\0';
                /* 大小写不敏感子串搜 (block 通常短, 线性可接受) */
                for (const char *s = blk; *s; s++) {
                    if (tolower((unsigned char)*s) == word[0]) {
                        size_t m = 0;
                        while (word[m] && tolower((unsigned char)s[m]) == word[m]) m++;
                        if (!word[m]) { score++; break; }
                    }
                }
            }
            i = j;
        }
        if (score > best_score) { best_score = score; best = k; }
    }
    return (best >= 0 && best_score >= 1) ? g_knowledge_blocks[best] : NULL;  /* score≥1: ≥3字符判别词一命中即注入(g4 "vet" 修复); 误注入参考害<代码框失败 */
}

void append_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    ds4_buf_puts(b,
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as raw text and set `string=\"true\"`. "
        "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
        "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        /* NOTE (2026-07-07): a concrete worked example was tried here and
         * REMOVED: with --tool-primer the server injects the structure anyway,
         * and a fragile long-context model copied the example's placeholder
         * value verbatim instead of the user's actual path (gate v2 evidence:
         * 8x Read("/path/to/main.go")). Examples with plausible values are a
         * contamination source for continuation models. */
        "");
    /* --nothink 时剥掉 <think> 指令段(2026-07-16 针4 实证): 这两行把 think 特殊
     * token 明晃晃写进上下文, 对 1-bit base 是校准盲区陷阱 — 无 soul 模板可跟的
     * 任务(写新代码)会"听 header 的话"试图发 <think>, 劣化成 <思> 循环。 */
    if (!g_force_nothink)
        ds4_buf_puts(b,
            "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
            "Otherwise, output directly after </think> with tool calls or final response.\n\n");
    ds4_buf_puts(b, "### Available Tool Schemas\n\n");
    ds4_buf_puts(b, tool_schemas);
    ds4_buf_puts(b, "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
                "Use the exact parameter names from the schemas.");
    if (g_soul_text && g_soul_text[0]) {
        ds4_buf_puts(b, "\n\n");
        ds4_buf_puts(b, g_soul_text);
    }
}
