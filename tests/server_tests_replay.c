/* server_tests_replay.c — 机械拆分自 ds4_server.c 内嵌测试块 (15398-15842 行): 多模态块/Responses 重放。 */

#include "server_tests_internal.h"

void test_anthropic_image_block_renders_encoder_text(void) {
    /* The Claude Code paste-a-screenshot path: an image content block turns
     * into the encoder's UI-sketch text inside the user message, in client
     * order (sketch before the instruction that references it). */
    ds4_mm *mm = ds4_mm_create(NULL, NULL);
    TEST_ASSERT(mm && ds4_mm_register_command(mm, "image", "/bin/cat") == 0);

    const char *json =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image\",\"source\":{\"type\":\"base64\","
        "\"media_type\":\"image/png\","
        "\"data\":\"QlVUVE9OIDUxMiw0MzYgMjU2eDQ0ICMzYjgyZjY=\"}},"
        "{\"type\":\"text\",\"text\":\"make the button red\"}"
        "]}]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs, mm));
    TEST_ASSERT(msgs.len == 1);
    const char *sketch = strstr(msgs.v[0].content, "<image>");
    const char *instruction = strstr(msgs.v[0].content, "make the button red");
    TEST_ASSERT(sketch != NULL);
    TEST_ASSERT(strstr(msgs.v[0].content, "BUTTON 512,436 256x44 #3b82f6") != NULL);
    TEST_ASSERT(strstr(msgs.v[0].content, "</image>") != NULL);
    TEST_ASSERT(instruction != NULL && sketch < instruction);
    chat_msgs_free(&msgs);

    /* fail closed, never silently drop: no registry (encoder tool absent) */
    p = json;
    chat_msgs msgs_noreg = {0};
    TEST_ASSERT(!parse_anthropic_messages(&p, &msgs_noreg, NULL));
    chat_msgs_free(&msgs_noreg);

    /* fail closed: unsupported media type */
    const char *json_wav =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image\",\"source\":{\"type\":\"base64\","
        "\"media_type\":\"audio/wav\",\"data\":\"aGVsbG8=\"}}"
        "]}]";
    p = json_wav;
    chat_msgs msgs_wav = {0};
    TEST_ASSERT(!parse_anthropic_messages(&p, &msgs_wav, mm));
    chat_msgs_free(&msgs_wav);

    /* fail closed: url source (the server does not fetch) */
    const char *json_url =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image\",\"source\":{\"type\":\"url\","
        "\"url\":\"https://example.com/a.png\"}}"
        "]}]";
    p = json_url;
    chat_msgs msgs_url = {0};
    TEST_ASSERT(!parse_anthropic_messages(&p, &msgs_url, mm));
    chat_msgs_free(&msgs_url);
    ds4_mm_free(mm);
}

void test_anthropic_tool_result_mcp_blocks(void) {
    /* The real MCP loop: the client replays an assistant tool_use plus a user
     * tool_result whose content is a block array.  Screenshot-style MCP tools
     * return pixels inside that array; is_error carries tool failure. */
    ds4_mm *mm = ds4_mm_create(NULL, NULL);
    TEST_ASSERT(mm && ds4_mm_register_command(mm, "image", "/bin/cat") == 0);

    /* Legacy string content renders byte-identically (disk-KV prefix compat),
     * including is_error:false. */
    const char *json_str =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_01\","
        "\"is_error\":false,\"content\":\"ok\"}"
        "]}]";
    const char *p = json_str;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs, mm));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].content, "<tool_result>ok</tool_result>"));
    chat_msgs_free(&msgs);

    /* Block array: text + image + text in client order; the image resolves
     * through the modality registry exactly like a user-message image block. */
    const char *json_mixed =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_02\",\"content\":["
        "{\"type\":\"text\",\"text\":\"shot:\"},"
        "{\"type\":\"image\",\"source\":{\"type\":\"base64\","
        "\"media_type\":\"image/png\","
        "\"data\":\"QlVUVE9OIDUxMiw0MzYgMjU2eDQ0ICMzYjgyZjY=\"}},"
        "{\"type\":\"text\",\"text\":\"done\"}"
        "]}]}]";
    p = json_mixed;
    chat_msgs msgs_mixed = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs_mixed, mm));
    TEST_ASSERT(msgs_mixed.len == 1);
    TEST_ASSERT(!strcmp(msgs_mixed.v[0].content,
                        "<tool_result>shot:<image>\n"
                        "BUTTON 512,436 256x44 #3b82f6\n"
                        "</image>done</tool_result>"));
    TEST_ASSERT(msgs_mixed.v[0].tool_call_ids_len == 1 &&
                !strcmp(msgs_mixed.v[0].tool_call_ids[0], "toolu_02"));
    chat_msgs_free(&msgs_mixed);

    /* is_error:true surfaces as an explicit marker inside the envelope. */
    const char *json_err =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_03\","
        "\"content\":\"exit status 1\",\"is_error\":true}"
        "]}]";
    p = json_err;
    chat_msgs msgs_err = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs_err, mm));
    TEST_ASSERT(msgs_err.len == 1);
    TEST_ASSERT(!strcmp(msgs_err.v[0].content,
                        "<tool_result>[tool_error] exit status 1</tool_result>"));
    chat_msgs_free(&msgs_err);

    /* Fail closed: an image inside tool_result the server cannot deliver
     * (url source / no registry) must 400, never silently vanish. */
    const char *json_img_url =
        "[{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_04\",\"content\":["
        "{\"type\":\"image\",\"source\":{\"type\":\"url\","
        "\"url\":\"https://example.com/a.png\"}}"
        "]}]}]";
    p = json_img_url;
    chat_msgs msgs_img_url = {0};
    TEST_ASSERT(!parse_anthropic_messages(&p, &msgs_img_url, mm));
    chat_msgs_free(&msgs_img_url);

    p = json_mixed;
    chat_msgs msgs_noreg = {0};
    TEST_ASSERT(!parse_anthropic_messages(&p, &msgs_noreg, NULL));
    chat_msgs_free(&msgs_noreg);

    /* Text-only tool_result never needs the registry (headless servers
     * without an encoder keep working). */
    p = json_str;
    chat_msgs msgs_txt_noreg = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs_txt_noreg, NULL));
    TEST_ASSERT(msgs_txt_noreg.len == 1 &&
                !strcmp(msgs_txt_noreg.v[0].content,
                        "<tool_result>ok</tool_result>"));
    chat_msgs_free(&msgs_txt_noreg);
    ds4_mm_free(mm);
}

void test_mcp_tool_schema_names_roundtrip(void) {
    /* MCP tool names (mcp__server__tool) plus real-world schema noise
     * ($schema, additionalProperties, cache_control) must flow verbatim into
     * the DSML schema block, the guided-primer order table, and the rendered
     * invoke -- no mangling anywhere, or the client cannot match the call. */
    const char *name = "mcp__playwright__browser_take_screenshot";
    const char *tools_json =
        "[{\"name\":\"mcp__playwright__browser_take_screenshot\","
        "\"description\":\"Take a screenshot of the current page\","
        "\"cache_control\":{\"type\":\"ephemeral\"},"
        "\"input_schema\":{\"$schema\":\"http://json-schema.org/draft-07/schema#\","
        "\"type\":\"object\",\"additionalProperties\":false,"
        "\"properties\":{\"filename\":{\"type\":\"string\"},"
        "\"fullPage\":{\"type\":\"boolean\"}},"
        "\"required\":[\"filename\"]}}]";
    const char *p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, name));
    TEST_ASSERT(orders.len == 1);
    TEST_ASSERT(!strcmp(orders.v[0].name, name));
    TEST_ASSERT(orders.v[0].len == 2 && !strcmp(orders.v[0].prop[0], "filename"));
    TEST_ASSERT(orders.v[0].req_len == 1 && !strcmp(orders.v[0].req[0], "filename"));

    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("screenshot the page");
    chat_msgs_push(&msgs, user);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_mcp01");
    tc.name = xstrdup(name);
    tc.arguments = xstrdup("{\"filename\":\"page.png\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    char *prompt = render_chat_prompt_text(&msgs, schemas, &orders, DS4_THINK_NONE, NULL);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, name) != NULL);
    buf invoke = {0};
    buf_puts(&invoke, "invoke name=\"");
    buf_puts(&invoke, name);
    buf_puts(&invoke, "\"");
    TEST_ASSERT(strstr(prompt, invoke.ptr) != NULL);
    buf_free(&invoke);

    free(prompt);
    chat_msgs_free(&msgs);
    free(schemas);
    tool_schema_orders_free(&orders);
}

void test_tool_checkpoint_canonicalization_gate_exact_replay(void) {
    server s;
    memset(&s, 0, sizeof(s));

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_exact");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{}");
    tool_calls_push(&calls, tc);
    calls.raw_dsml = xstrdup(
        "\n\n" DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "</｜DSML｜invoke>\n"
        DS4_TOOL_CALLS_END);

    TEST_ASSERT(!should_canonicalize_tool_checkpoint(&s, &calls));

    s.disable_exact_dsml_tool_replay = true;
    TEST_ASSERT(should_canonicalize_tool_checkpoint(&s, &calls));

    s.disable_exact_dsml_tool_replay = false;
    free(calls.raw_dsml);
    calls.raw_dsml = NULL;
    TEST_ASSERT(should_canonicalize_tool_checkpoint(&s, &calls));

    tool_calls_free(&calls);
}

void test_responses_live_tail_renders_tool_outputs_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = DS4_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_live");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    responses_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.responses_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.responses_live_call_ids.v[0], "call_live"));
    TEST_ASSERT(r.responses_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.responses_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "exec_command") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}

void test_responses_tool_output_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_missing");
    tool.content = xstrdup("out");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    TEST_ASSERT(!responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH, NULL, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Responses continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.responses_live.valid = true;
    s.responses_live.live_tokens = 10;
    id_list_push_unique(&s.responses_live.call_ids, "call_missing");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    bool needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state, NULL,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}

void test_responses_stateless_tool_replay_requires_reasoning(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_replay");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_replay");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    bool needs_live_reasoning = false;
    bool needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    pthread_mutex_lock(&s.tool_mu);
    s.responses_live.valid = true;
    s.responses_live.live_tokens = 123;
    id_list_push_unique(&s.responses_live.call_ids, "call_replay");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = xstrdup("replayed hidden reasoning");
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = NULL;
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_NONE,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}

void test_responses_visible_suffix_matches_client_replay(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = DS4_THINK_HIGH;
    r.reasoning_summary_emit = true;

    char *suffix = build_responses_visible_assistant_suffix(&r, "5",
                                                            "hidden summary",
                                                            NULL);
    TEST_ASSERT(strstr(suffix, "hidden summary") == NULL);
    TEST_ASSERT(strstr(suffix, "</think>5") != NULL);
    free(suffix);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&calls, tc);

    suffix = build_responses_visible_assistant_suffix(&r, "",
                                                      "tool summary",
                                                      &calls);
    TEST_ASSERT(strstr(suffix, "tool summary</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜DSML｜tool_calls>") != NULL);
    free(suffix);

    tool_calls_free(&calls);
    request_free(&r);
}

void test_exact_dsml_tool_replay_can_be_disabled(void) {
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_memory_put(&s, "call_disabled", dsml);
    s.disable_exact_dsml_tool_replay = true;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disabled");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"canonical\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t bytes = 123;
    TEST_ASSERT(kv_tool_map_write(&s, fp, dsml, &bytes));
    TEST_ASSERT(bytes == 0);

    if (fp) fclose(fp);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}
