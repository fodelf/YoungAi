/* t_quality.c — 工具调用质量(要真模型) + 前端域多模态插件 + server 单测组 + TP all-reduce 回环 (机械拆分自 tests/ds4_test.c, 重构阶段8)。 */
#include "test_internal.h"
#include "../ds4_spatial.h"
#include "../ds4_css.h"
#ifndef DS4_NO_GPU

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        ds4_buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    ds4_buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

#endif /* !DS4_NO_GPU */

/* ---- 前端域多模态插件 (物理方位 / CSS 理解 / enricher 链) ---- */

/* mm-ui selftest 场景的手写草图副本: 数字取整, "Continue" 精确居中。
 * 格式契约与 tools/mm_ui.swift 输出一致 -- 这里断言的每个空间/CSS 事实
 * 都能手算复核。 */
static const char *TEST_UI_SKETCH =
    "[img 1280x800]\n"
    "[palette #f5f6f8 84% #101828 8%]\n"
    "[rect 0,65 1280x735 #f5f6f8]\n"
    "[rect 0,0 1280x64 #101828]\n"
    "[rect 480,280 320x240 #ffffff]\n"
    "[rect 512,436 256x44 #3b82f6]\n"
    "[text 40,22 146x22 #ffffff on #101828 \"Acme Console\"]\n"
    "[text 512,318 74x28 #101828 on #ffffff \"Sign in\"]\n"
    "[text 596,446 88x24 #ffffff on #3b82f6 \"Continue\"]\n";

static void test_spatial_annotate_facts(void) {
    char *sec = ds4_spatial_annotate(TEST_UI_SKETCH);
    TEST_ASSERT(sec != NULL);
    TEST_ASSERT(strstr(sec, "t2\"Sign in\"") != NULL);           /* 标签系统 */
    TEST_ASSERT(strstr(sec, "r1 page") != NULL);                 /* 九宫格方位 */
    TEST_ASSERT(strstr(sec, "r2 top") != NULL);
    TEST_ASSERT(strstr(sec, "r3 center") != NULL);
    TEST_ASSERT(strstr(sec, "[in r3(480,280): t2 r4]") != NULL); /* 包含树 */
    TEST_ASSERT(strstr(sec, "[stack r3 column gap=90px: t2 r4]") != NULL);
    TEST_ASSERT(strstr(sec, "[align left: t2 r4 (in r3)]") != NULL);
    TEST_ASSERT(strstr(sec, "[align centered-x in r3: r4]") != NULL);
    TEST_ASSERT(strstr(sec, "[align centered-x in r4: t3]") != NULL);
    free(sec);
    /* 非草图输入诚实缺席 */
    TEST_ASSERT(ds4_spatial_annotate("hello, not a sketch") == NULL);
}

static void test_css_annotate_facts(void) {
    char *sec = ds4_css_annotate(TEST_UI_SKETCH);
    TEST_ASSERT(sec != NULL);
    TEST_ASSERT(strstr(sec, "[css page: background:#f5f6f8]") != NULL);
    TEST_ASSERT(strstr(sec, "[css r3: background:#ffffff; "
                            "padding:38px 32px 40px 32px; display:flex; "
                            "flex-direction:column; gap:90px; "
                            "align-items:flex-start]") != NULL);
    TEST_ASSERT(strstr(sec, "[css r4: background:#3b82f6; "
                            "padding:10px 84px 10px 84px]") != NULL);
    TEST_ASSERT(strstr(sec, "[css r2: background:#101828; "
                            "padding:22px 1094px 20px 40px]") != NULL);
    free(sec);
    TEST_ASSERT(ds4_css_annotate("plain text") == NULL);
}

static void test_mm_stub_tokenize(void *ctx, const char *text, int **toks, int *n) {
    (void)ctx;
    const size_t len = strlen(text);
    *toks = malloc(len ? len * sizeof(int) : sizeof(int));
    *n = 0;
    if (!*toks) return;
    for (size_t i = 0; i < len; i++) (*toks)[i] = (int)(unsigned char)text[i];
    *n = (int)len;
}

static char *test_mm_spatial_adapter(void *ctx, const char *modality,
                                     const char *text) {
    (void)ctx;
    (void)modality;
    return ds4_spatial_annotate(text);
}

static char *test_mm_css_adapter(void *ctx, const char *modality,
                                 const char *text) {
    (void)ctx;
    (void)modality;
    return ds4_css_annotate(text);
}

static void test_mm_enrichers_compose_both_forms(void) {
    ds4_mm *mm = ds4_mm_create(test_mm_stub_tokenize, NULL);
    TEST_ASSERT(mm != NULL);
    TEST_ASSERT(ds4_mm_register_command(mm, "image", "/bin/cat") == 0);
    TEST_ASSERT(ds4_mm_register_enricher(mm, "image", test_mm_spatial_adapter,
                                         NULL) == 0);
    TEST_ASSERT(ds4_mm_register_enricher(mm, "image", test_mm_css_adapter,
                                         NULL) == 0);
    char *text = NULL;
    TEST_ASSERT(ds4_mm_encode_as_text(mm, "image/png",
                                      (const uint8_t *)TEST_UI_SKETCH,
                                      strlen(TEST_UI_SKETCH), &text) == 0);
    TEST_ASSERT(text != NULL);
    const char *sketch_tail = strstr(text, "\"Continue\"]");
    const char *spatial = strstr(text, "[in r3");
    const char *css = strstr(text, "[css r3");
    TEST_ASSERT(sketch_tail && spatial && css);
    TEST_ASSERT(sketch_tail < spatial && spatial < css);  /* 节序 = 注册序 */
    const size_t text_len = strlen(text);
    /* token 形必须携带同一份增强文本 (字节 stub tokenizer 逐字节出 token) */
    int *toks = NULL;
    int n_toks = 0;
    TEST_ASSERT(ds4_mm_encode(mm, "image/png",
                              (const uint8_t *)TEST_UI_SKETCH,
                              strlen(TEST_UI_SKETCH), &toks, &n_toks) == 0);
    TEST_ASSERT((size_t)n_toks == text_len);
    free(toks);
    free(text);
    /* text 模态没挂 enricher: identity 原样 */
    TEST_ASSERT(ds4_mm_encode_as_text(mm, "text", (const uint8_t *)"hi", 2,
                                      &text) == 0);
    TEST_ASSERT(text && !strcmp(text, "hi"));
    free(text);
    ds4_mm_free(mm);
}

void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
    test_spatial_annotate_facts();
    test_css_annotate_facts();
    test_mm_enrichers_compose_both_forms();
}

/* TP Stage 2: in-process loopback check of the all-reduce transport (frame
 * format + sum correctness + no deadlock). No model, no network. */
void test_tp_allreduce(void) {
    TEST_ASSERT(ds4_dist_tp_selftest() == 0);
}
