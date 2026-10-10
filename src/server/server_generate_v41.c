/* server_generate_v41.c — DeepSeek V4.1 的服务生成路(2026-09-19)。
 *
 * V4.1 前向没有 ds4_session(会话/采样/KV 复用是 P5, 还没做), 只有 core_v41_api.c 的
 * ds4_engine_v41_generate_argmax(prompt → 分块预填 → 逐 token 贪心, 每个 token 回调一次)。
 * 所以这条路与 generate_job(V4)的区别只在"token 从哪来": 这里 token 从回调来, 其余 ——
 * 文本累积 / 思考段跟踪 / DSML 工具标记 / 停止串 / 三种流式 / 收尾响应 —— 与 V4 用同一批函数,
 * 客户端看到的协议一个字节不差。
 * 没有的东西(如实, 别猜): ①KV 复用与磁盘 KV —— 每条请求整段预填(15k token 的提示约 40 s @334 t/s);
 * ②工具调用出错后的续写修复(continue_after_invalid_dsml 要会话); ③并发批处理; ④Responses/Anthropic 的活绑定
 * (它们指向会话 KV 位置, 这里没有 KV 可续)。
 * 有的: 采样与复读惩罚(见下面 ds4_engine_set_decode_sampling 那一段 —— 请求没带的采样参数落到 ds4.h 的官方默认);
 * 上下文 ds4_engine_v41_ctx()(模型元数据 deepseek4.context_length; 2026-09-22 用户定"不要任何写死的上下文", 没有 --ctx)。
 * 为什么不改 generate_job 本体: 它 1500 行、全是会话位置/回滚/primer 的控制流, 往里塞第二种 token
 * 来源只会把两条路的 bug 搅在一起; 这里独立一份, V4 一行不动。 */
#include "server_internal.h"


static const char *v41_kind(const v41_gen *g) { return g->j->req.kind == REQ_CHAT ? "chat" : "completion"; }

/* 预填块间: 照旧发 SSE 心跳与进度日志, 再探一次客户端还在不在。返回非 0 ⇒ 引擎停止预填(ds4_v41_api.h)。
 * 13 万 token 的提示预填 400 秒, 期间客户端挂断的话这 400 秒纯浪费, 后面排队的请求还要跟着一起超时。
 * (v41_gen 与 V41_ALIVE_CHECK_TOKENS 在 server_internal.h: 并发调度器 server_sched_v41.c 同用这三段) */
int v41_progress_cb(void *ud, const char *event, int current, int total) {
    v41_gen *g = ud;
    server_progress_cb(&g->progress, event, current, total);
    if (!strcmp(event, "prefill_chunk")) mon_prefill_progress(g->s, g->j->mon, current, total);   /* 监控页的读提示进度条 */
    if (!client_disconnected(g->j->fd)) return 0;
    g->client_gone = true;
    server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s client disconnected during prefill %d/%d, aborting",
               v41_kind(g), g->ctx_span, current, total);
    trace_event(g->s, g->trace_id, "client disconnected during prefill at %d/%d", current, total);
    return 1;
}

/* 预填刚结束: 与 V4 路在 ds4_session_sync 之后做的事一样 —— 发流式头(预填心跳可能已经发过)、OpenAI 角色块、
 * Anthropic/Responses 的起始事件。失败 = 客户端没了, 返回 false, 调用方停止生成。 */
static bool v41_stream_begin(v41_gen *g) {
    server *s = g->s; job *j = g->j;
    if (!j->req.stream) return true;
    if (g->progress.stream_failed) {
        server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s stream closed during prefill",
                   v41_kind(g), g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
        return false;
    }
    if (!g->progress.headers_sent && !sse_headers(j->fd, s->enable_cors)) {
        server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s sse headers failed",
                   v41_kind(g), g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
        return false;
    }
    g->progress.headers_sent = true;
    if (j->req.api == API_ANTHROPIC &&
        !anthropic_sse_start_live(j->fd, &j->req, g->id, g->prompt_tokens, &g->anthropic_live)) {
        server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s anthropic stream start failed", g->ctx_span);
        return false;
    }
    if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT && !sse_chunk(j->fd, &j->req, g->id, NULL, NULL)) {
        server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s openai role chunk failed", g->ctx_span);
        return false;
    }
    if (g->openai_live_chat) openai_stream_start(&j->req, &g->openai_live);
    if (g->responses_live_chat) {
        responses_stream_init(&j->req, &g->responses_live);
        g->responses_live.active = true;
        if (!responses_sse_created(j->fd, &j->req, &g->responses_live, g->responses_created_at)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s%s%s responses created event failed",
                       g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
            return false;
        }
    }
    return true;
}

static void v41_prefill_done(v41_gen *g) {
    g->started = true;
    g->decode_t0 = g->last_decode_log_t = now_sec();
    mon_first_token(g->s, g->j->mon);
    server_log(DS4_LOG_PREFILL, "ds4-server: %s ctx=%s%s%s prompt done %.3fs",
               v41_kind(g), g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, now_sec() - g->t0);
    trace_event(g->s, g->trace_id, "prefill done; decode_max=%d", g->max_tokens);
}

/* 引擎每出一个 token 回调一次。返回非 0 = 停止生成。逐 token 的处理与 server_generate_body3.inc 的
 * 解码圈同一套(文本累积 → 思考段 → DSML 跟踪 → 停止串 → 流式增量 → 工具标记), 只去掉了会话相关的
 * 三处(kv 续存、ds4_session_invalidate、primer)。 */
int v41_emit(int token, void *ud) {
    v41_gen *g = ud; server *s = g->s; job *j = g->j;
    if (!g->started) {
        v41_prefill_done(g);
        if (!v41_stream_begin(g)) { g->finish = "error"; snprintf(g->err, sizeof(g->err), "client stream write failed"); return 1; }
    }
    if (g_stop_requested) { g->finish = "error"; snprintf(g->err, sizeof(g->err), "shutdown requested"); return 1; }
    /* ★客户端走了就别再算★(2026-09-22): 非流式请求整个生成期间不写一个字节, 不主动探就发现不了对端挂断。
     * 实撞代价: 同一条 132k token 的请求被超时重发 4 遍, 102 分钟 GPU 全算给已经断开的连接。 */
    if (g->completion >= g->next_alive_check) {
        g->next_alive_check = g->completion + V41_ALIVE_CHECK_TOKENS;
        if (client_disconnected(j->fd)) {
            g->client_gone = true;
            g->stream_dead = true;
            g->finish = "error";
            snprintf(g->err, sizeof(g->err), "client disconnected after %d generated tokens", g->completion);
            server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s client disconnected after %d generated tokens, stopping",
                       v41_kind(g), g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, g->completion);
            trace_event(s, g->trace_id, "client disconnected after %d generated tokens", g->completion);
            return 1;
        }
    }
    if (s->trace) {   /* 记 id 要在 EOS 判断之前: EOS 没有文本, 但重放对拍要知道模型在哪一位停的 */
        if (g->n_ids == g->cap_ids) {
            g->cap_ids = g->cap_ids ? g->cap_ids * 2 : 1024;
            g->ids = xrealloc(g->ids, (size_t)g->cap_ids * sizeof(int32_t));
        }
        g->ids[g->n_ids++] = token;
    }
    if (token == g->eos) { g->finish = "stop"; return 1; }
    if (g->completion >= g->max_tokens) return 1;   /* finish 保持 "length" */

    size_t piece_len = 0;
    char *piece = ds4_token_text(s->engine, token, &piece_len);
    g->completion++;
    mon_token(s, j->mon, g->completion);
    trace_piece(s, g->trace_id, piece, piece_len);
    ds4_buf_append(&g->text, piece, piece_len);
    thinking_state_feed(&g->thinking, piece, piece_len);
    if (j->req.kind == REQ_CHAT && j->req.has_tools) dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);

    size_t stop_pos = 0, stop_len = 0;
    const bool hit_stop = stop_list_find_from(&j->req.stops, g->text.ptr, g->stop_scan_from, &stop_pos, &stop_len);
    size_t stream_len = hit_stop ? stop_pos : stop_list_stream_safe_len(&j->req.stops, g->text.len);
    if (stream_len > g->text.len) stream_len = g->text.len;
    stream_len = utf8_stream_safe_len(g->text.ptr, g->plain_stream_pos, stream_len, hit_stop);
    if (!hit_stop && j->req.stops.max_len > 1) {
        const size_t hold = j->req.stops.max_len - 1;
        g->stop_scan_from = g->text.len > hold ? g->text.len - hold : 0;
    }
    bool stream_ok = true;
    if (j->req.stream && !g->stream_dead) {
        if (!g->structured_stream && stream_len > g->plain_stream_pos) {
            char *delta = xstrndup(g->text.ptr + g->plain_stream_pos, stream_len - g->plain_stream_pos);
            stream_ok = sse_chunk(j->fd, &j->req, g->id, delta, NULL);
            free(delta);
            if (stream_ok) g->plain_stream_pos = stream_len;
        }
        if (stream_ok && j->req.api == API_ANTHROPIC)
            stream_ok = anthropic_sse_stream_update(j->fd, s, &j->req, g->id, &g->anthropic_live, g->text.ptr, stream_len, false);
        if (stream_ok && g->openai_live_chat)
            stream_ok = openai_sse_stream_update(j->fd, s, &j->req, g->id, &g->openai_live, g->text.ptr, stream_len, false);
        if (stream_ok && g->responses_live_chat)
            stream_ok = responses_sse_stream_update(j->fd, &j->req, &g->responses_live, g->text.ptr, stream_len, false);
    }
    free(piece);
    if (!stream_ok) {
        g->finish = "error"; snprintf(g->err, sizeof(g->err), "client stream write failed");
        g->stream_dead = true;
        return 1;
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        if (g->thinking_gates_tool_markers && g->thinking.inside) {
            /* 思考段里的 DSML 块不可执行(与 V4 路同一条守卫): 别让引号里的标记当成真工具调用把生成停掉 */
            g->tool_scan_waiting_for_think_close = true;
            g->tool_scan_from = g->text.len;
        } else {
            if (g->tool_scan_waiting_for_think_close) {
                const char *think_end = find_last_substr(g->text.ptr, "</think>");
                g->tool_scan_from = think_end ? (size_t)((think_end + 8) - g->text.ptr) : g->text.len;
                if (g->tool_scan_from > g->text.len) g->tool_scan_from = g->text.len;
                g->tool_scan_waiting_for_think_close = false;
            }
            if (g->tool_scan_from > g->text.len) g->tool_scan_from = g->text.len;
            const char *tool_scan = g->text.ptr ? g->text.ptr + g->tool_scan_from : "";
            bool orphan_end = false;
            const bool old_start = g->saw_tool_start, old_end = g->saw_tool_end;
            observe_tool_markers(tool_scan, &g->saw_tool_start, &g->saw_tool_end, &orphan_end);
            if (orphan_end && !g->saw_orphan_tool_end) {
                g->saw_orphan_tool_end = true;
                server_log(DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                           g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, g->completion);
                trace_event(s, g->trace_id, "ignored orphan tool-call end marker after %d generated tokens", g->completion);
            }
            if (g->saw_tool_start && !old_start) trace_event(s, g->trace_id, "entered tool-call block after %d generated tokens", g->completion);
            if (g->saw_tool_end && !old_end) trace_event(s, g->trace_id, "closed tool-call block after %d generated tokens", g->completion);
            const size_t marker_hold = 80;
            const size_t hold_from = g->text.len > marker_hold ? g->text.len - marker_hold : 0;
            if (hold_from > g->tool_scan_from) g->tool_scan_from = hold_from;
            if (s->trace && g->completion >= g->next_tool_progress) {
                trace_event(s, g->trace_id, "progress gen=%d dsml_start=%d dsml_end=%d",
                            g->completion, g->saw_tool_start ? 1 : 0, g->saw_tool_end ? 1 : 0);
                g->next_tool_progress += 128;
            }
        }
    }
    if (g->completion >= g->next_decode_log) {
        log_decode_progress(j->req.kind, g->prompt_tokens, g->completion, g->responses_protocol, j->req.has_tools,
                            g->thinking.inside, g->saw_tool_start, g->saw_tool_end, g->decode_t0,
                            &g->last_decode_log_t, &g->last_decode_log_completion);
        g->next_decode_log += 50;
    }
    if (hit_stop) {
        (void)stop_len;
        g->finish = "stop";
        g->text.len = stop_pos;
        g->text.ptr[g->text.len] = '\0';
        return 1;
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools && g->saw_tool_end) { g->finish = "tool_calls"; return 1; }
    return 0;
}

/* 生成结束后的收尾: 与 server_generate_body3.inc 尾段 + body4.inc 同一套, 去掉会话相关的检查点/活绑定/续写修复。
 * 返回对外报的 finish(静态串), 监控记录用。 */
static const char *v41_finish(v41_gen *g) {
    server *s = g->s; job *j = g->j;
    const char *finish = g->finish;
    char *err = g->err;
    buf *text = &g->text;
    const int completion = g->completion, prompt_tokens = g->prompt_tokens;
    if (text->ptr && text->len > 0) {   /* 硬截断可能切在多字节 UTF-8 中间 */
        text->len = utf8_stream_safe_len(text->ptr, 0, text->len, false);
        text->ptr[text->len] = '\0';
    }
    /* ★服务端上限截断时怎么报★(2026-09-22 改): 只对 Anthropic 路仍报 stop —— Claude Code 把 stop_reason=max_tokens 当成
     * "超出我配置的上限"直接中断整个会话(2026-07-07 实测, 见 server_generate_body3.inc 同处注释)。
     * OpenAI/Responses 路报 length(OpenAI 语义: 被 token 上限截断就是 length)。为什么必须报真话: qtf 的 CFO 流程拿到
     * finish_reason=stop 就当"模型正常写完", 把一份中途被上限剪断的报告存进库(实撞 09-21: 截断处落在复读区, 尾部 JSON
     * 解出来的是提示词里的模板值 symbol=000001, 后端还报 INVALID cross-contaminated)。报 length 才能让调用方判"这份不能用"。 */
    if (!strcmp(finish, "length") && j->req.api == API_ANTHROPIC && s->max_output_tokens > 0 &&
        completion >= s->max_output_tokens && j->req.max_tokens > s->max_output_tokens)
        finish = "stop";
    if (g_stop_requested && strcmp(finish, "error") != 0) { finish = "error"; snprintf(err, sizeof(g->err), "shutdown requested"); }
    if (j->req.kind == REQ_CHAT && j->req.has_tools && g->saw_tool_start && !g->saw_tool_end && strcmp(finish, "error") != 0) {
        /* V4 路会给模型喂一条工具错误让它重发(要会话续写); 这里没有会话, 如实报错 */
        server_log(DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s unterminated tool call (V4.1: 无会话续写修复)",
                   g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
        finish = "error"; snprintf(err, sizeof(g->err), "unterminated tool call");
    }
    if (completion > g->last_decode_log_completion)
        log_decode_progress(j->req.kind, prompt_tokens, completion, g->responses_protocol, j->req.has_tools,
                            g->thinking.inside, g->saw_tool_start, g->saw_tool_end, g->decode_t0,
                            &g->last_decode_log_t, &g->last_decode_log_completion);
    if (j->req.stream && !g->stream_dead && !g->structured_stream && text->len > g->plain_stream_pos) {
        char *tail = xstrndup(text->ptr + g->plain_stream_pos, text->len - g->plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, g->id, tail, NULL)) finish = "error";
        free(tail);
    }
    tool_calls parsed_calls = {0};
    char *parsed_content = NULL, *parsed_reasoning = NULL;
    const char *final_finish = finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        const bool parsed_ok = parse_generated_message_for_response(
            text->ptr ? text->ptr : "", j->req.has_tools, g->saw_tool_start, ds4_think_mode_enabled(j->req.think_mode),
            &final_finish, err, sizeof(g->err), &parsed_content, &parsed_reasoning, &parsed_calls, &recovered_tool_parse_failure);
        if (!parsed_ok) {
            const size_t snippet = text->len > 300 ? 300 : text->len;
            server_log(DS4_LOG_WARNING,
                       "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]",
                       g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags, final_finish, text->len,
                       g->saw_tool_start, g->saw_tool_end, (int)snippet, text->ptr ? text->ptr : "(null)");
            trace_event(s, g->trace_id, "invalid tool call returned as assistant text finish=%s", final_finish);
        }
        if (parsed_calls.len) {
            if (g->openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &g->openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream) apply_anthropic_stream_tool_ids(&parsed_calls, &g->anthropic_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        }
    }
    log_tool_calls_summary(g->ctx_span, &parsed_calls, g->responses_protocol);
    trace_finish(s, g->trace_id, &j->req, final_finish, completion, g->saw_tool_start, g->saw_tool_end,
                 parsed_content ? parsed_content : (text->ptr ? text->ptr : ""), parsed_reasoning, &parsed_calls, now_sec() - g->t0);
    /* 活绑定(Responses/Anthropic/thinking 检查点)全指向会话 KV 位置, 这条路没有 KV 可续 —— 一律清掉, 不留假绑定 */
    responses_live_clear(s); anthropic_live_clear(s); thinking_live_clear(s);

    const char *content = parsed_content ? parsed_content : (text->ptr ? text->ptr : "");
    if (j->req.stream) {
        bool ok = true;
        if (g->stream_dead) ok = false;
        else if (j->req.api == API_ANTHROPIC)
            ok = anthropic_sse_finish_live(j->fd, s, &j->req, g->id, &g->anthropic_live, text->ptr ? text->ptr : "", text->len,
                                           &parsed_calls, final_finish, completion);
        else if (g->openai_live_chat)
            ok = openai_sse_finish_live(j->fd, s, &j->req, g->id, &g->openai_live, text->ptr ? text->ptr : "", text->len,
                                        &parsed_calls, final_finish, prompt_tokens, completion);
        else if (g->responses_live_chat)
            ok = responses_sse_finish_live(j->fd, &j->req, &g->responses_live, text->ptr ? text->ptr : "", text->len,
                                           recovered_tool_parse_failure ? parsed_content : NULL, &parsed_calls, final_finish,
                                           prompt_tokens, completion, g->responses_created_at);
        else if (g->structured_stream)
            ok = sse_chat_finish(j->fd, &j->req, g->id, content, parsed_reasoning, &parsed_calls, final_finish, prompt_tokens, completion);
        else
            ok = sse_chunk(j->fd, &j->req, g->id, NULL, final_finish) && sse_done(j->fd, &j->req, g->id, prompt_tokens, completion);
        if (!ok) server_log(DS4_LOG_DEFAULT, "ds4-server: %s ctx=%s%s%s final stream failed",
                            v41_kind(g), g->ctx_span, g->req_flags[0] ? " " : "", g->req_flags);
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, s->enable_cors, &j->req, g->id, content, parsed_reasoning, &parsed_calls, final_finish, prompt_tokens, completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, s->enable_cors, &j->req, g->id, content, parsed_reasoning, &parsed_calls, final_finish, prompt_tokens, completion);
    } else {
        final_response(j->fd, s->enable_cors, &j->req, g->id, content, parsed_reasoning, &parsed_calls, final_finish, prompt_tokens, completion);
    }
    char flags[80];
    log_flags(flags, sizeof(flags), g->responses_protocol, j->req.has_tools, g->thinking.inside,
              j->req.kind == REQ_CHAT && j->req.has_tools ? g->saw_tool_start : false,
              j->req.kind == REQ_CHAT && j->req.has_tools ? g->saw_tool_end : false);
    if (!strcmp(final_finish, "error") && err[0])
        server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                   v41_kind(g), g->ctx_span, completion, flags[0] ? " " : "", flags, final_finish, err, now_sec() - g->t0);
    else
        server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                   v41_kind(g), g->ctx_span, completion, flags[0] ? " " : "", flags, final_finish, now_sec() - g->t0);
    free(parsed_content); free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    return final_finish;
}

bool v41_gen_begin(server *s, job *j, v41_gen *g) {
    memset(g, 0, sizeof *g);
    g->s = s; g->j = j;
    g->finish = "length";
    g->eos = ds4_token_eos(s->engine);
    const ds4_tokens *prompt = &j->req.prompt;
    const int ctx = ds4_engine_v41_ctx();   /* 与 s->ctx_size 同一个数(server_main.c 起服时从这里取), 这里直接取源头 */
    g->prompt_tokens = prompt->len;
    if (g->prompt_tokens < 1) { http_error(j->fd, s->enable_cors, 400, "empty prompt"); return false; }
    if (g->prompt_tokens >= ctx) { http_error_context_length_exceeded(j->fd, s->enable_cors, &j->req, g->prompt_tokens, ctx); return false; }
    /* 没有前缀缓存: 全部算写入(usage 里 cache_read=0), 与事实一致 */
    j->req.cache_read_tokens = 0;
    j->req.cache_write_tokens = g->prompt_tokens;
    g->responses_protocol = j->req.api == API_RESPONSES;
    g->t0 = now_sec();
    trace_cache_diag cache_diag = {0};
    g->trace_id = trace_begin(s, j, 0, g->prompt_tokens, &cache_diag, "v41-cold", 0, NULL);
    request_ctx_span(g->ctx_span, sizeof(g->ctx_span), 0, g->prompt_tokens);
    log_flags(g->req_flags, sizeof(g->req_flags), g->responses_protocol, j->req.has_tools, false, false, false);
    g->progress = (server_prefill_progress){
        .srv = NULL,   /* 没有会话 KV 可续存, 进度回调只发心跳与日志 */
        .kind = j->req.kind, .prompt_tokens = g->prompt_tokens, .cached_tokens = 0,
        .has_tools = j->req.has_tools, .responses_protocol = g->responses_protocol,
        .t0 = g->t0, .fd = j->fd, .stream = j->req.stream, .enable_cors = s->enable_cors,
    };
    snprintf(g->progress.ctx, sizeof(g->progress.ctx), "%s", g->ctx_span);
    g->max_tokens = j->req.max_tokens < 0 ? 0 : j->req.max_tokens;
    if (g->max_tokens > ctx - g->prompt_tokens) g->max_tokens = ctx - g->prompt_tokens;
    if (s->max_output_tokens > 0 && g->max_tokens > s->max_output_tokens) g->max_tokens = s->max_output_tokens;
    g->thinking = thinking_state_from_prompt(&j->req);
    g->thinking_gates_tool_markers = ds4_think_mode_enabled(j->req.think_mode);
    g->tool_scan_waiting_for_think_close = g->thinking_gates_tool_markers && g->thinking.inside;
    dsml_decode_tracker_init(&g->dsml_tracker);
    g->next_tool_progress = 128; g->next_decode_log = 50; g->next_alive_check = V41_ALIVE_CHECK_TOKENS;
    snprintf(g->id, sizeof(g->id), "%s-%llu", j->req.kind == REQ_CHAT ? "chatcmpl" : "cmpl", (unsigned long long)++s->seq);
    g->structured_stream = request_uses_structured_stream(&j->req);
    g->openai_live_chat = request_uses_openai_live_stream(&j->req);
    g->responses_live_chat = request_uses_responses_live_stream(&j->req);
    g->responses_created_at = (long)time(NULL);
    /* 解码采样: 请求带什么就用什么, 没带的落到 ds4.h 的官方默认(温 1.0), 与官方 API 同。
     * 以前"没带 temperature = 裸 argmax": 聊天前端大多不发 temperature ⇒ 产品请求全走贪心, 09-28 "每句以五结尾"
     * 类请求在思考段逐字死循环(FP 老师在复读位也有 86~96% 选抄, 贪心出不来)。尺脚本要贪心就显式发 temperature:0。
     * 频率/出现惩罚按请求; DRY 按服务启动参数(客户端协议没有它), 请求里带了 dry_multiplier 就按请求(base/allowed 没给用 llama.cpp 默认)。 */
    g->sp = (ds4_decode_sampling){
        .temperature = j->req.temperature, .top_p = j->req.top_p, .min_p = j->req.min_p,
        .top_k = j->req.top_k, .seed = j->req.seed, .freq_penalty = j->req.frequency_penalty, .presence_penalty = j->req.presence_penalty,
        .dry_multiplier = j->req.dry_set ? j->req.dry_multiplier : s->dry_multiplier,
        .dry_base = j->req.dry_set ? (j->req.dry_base > 1.f ? j->req.dry_base : 1.75f) : s->dry_base,
        .dry_allowed_length = j->req.dry_set ? (j->req.dry_allowed_length > 0 ? j->req.dry_allowed_length : 2) : s->dry_allowed_length,
    };
    const ds4_decode_sampling *sp = &g->sp;
    if (sp->temperature > 0.f || sp->dry_multiplier > 0.f || sp->freq_penalty != 0.f || sp->presence_penalty != 0.f)
        server_log(DS4_LOG_GENERATION, "ds4-server: %s ctx=%s V4.1 解码采样 temp=%.2f top_p=%.2f min_p=%.2f top_k=%d seed=%llu dry=%.2f/%.2f/%d freq=%.2f presence=%.2f",
                   v41_kind(g), g->ctx_span, (double)sp->temperature, (double)sp->top_p, (double)sp->min_p, sp->top_k, (unsigned long long)sp->seed,
                   (double)sp->dry_multiplier, (double)sp->dry_base, sp->dry_allowed_length, (double)sp->freq_penalty, (double)sp->presence_penalty);
    return true;
}

void v41_gen_end(v41_gen *g, int rc) {
    server *s = g->s; job *j = g->j;
    if (rc != 0 && !g->started) {   /* 预填没走完 */
        if (g->client_gone) {   /* 我们自己叫停的, 不是故障: 对端已经没了, 连错误响应都不用写 */
            trace_event(s, g->trace_id, "prefill aborted: client disconnected");
        } else {
            trace_event(s, g->trace_id, "prefill failed: V4.1 forward failed");
            send_prefill_failure_response(s, j, &g->progress, g->ctx_span, g->req_flags, "V4.1 prefill failed");
        }
        mon_end(s, j->mon, g->client_gone ? "disconnect" : "error", 0, -1, -1);
        free(g->ids); ds4_buf_free(&g->text);
        return;
    }
    if (!g->started) {   /* max_tokens == 0: 预填成了但一个 token 都不要 */
        v41_prefill_done(g);
        if (!v41_stream_begin(g)) { mon_end(s, j->mon, "disconnect", 0, -1, -1); free(g->ids); ds4_buf_free(&g->text); return; }
    }
    if (rc != 0 && strcmp(g->finish, "error") != 0) { g->finish = "error"; snprintf(g->err, sizeof(g->err), "V4.1 decode failed"); }
    const char *final_finish = v41_finish(g);
    /* 监控: 客户端挂断单独算一类(Strata 的 disconnect), 别混进 error; 草稿两项只在真投机过时报 */
    mon_end(s, j->mon, g->client_gone ? "disconnect" : final_finish, g->completion,
            g->spec_rounds > 0 ? g->spec_offered : -1, g->spec_rounds > 0 ? g->spec_accepted : -1);
    trace_token_ids(s, g->trace_id, j->req.prompt.v, j->req.prompt.len, g->ids, g->n_ids);   /* 提示 + 生成的 id 全落 trace(只在 --trace 开时) */
    free(g->ids);
    anthropic_stream_free(&g->anthropic_live);
    openai_stream_free(&g->openai_live);
    responses_stream_free(&g->responses_live);
    ds4_buf_free(&g->text);
}

/* 单 worker 路(不带 --batch): 引擎整段跑(预填 → 逐 token, 带投机), token 从回调来 */
void generate_job_v41(server *s, job *j) {
    v41_gen g;
    if (!v41_gen_begin(s, j, &g)) return;
    ds4_engine_set_decode_sampling(&g.sp);   /* 单 worker ⇒ 全局设置面按请求覆写即可 */
    server_log(DS4_LOG_PREFILL, "ds4-server: %s ctx=%s%s%s prompt start (V4.1 整段预填, decode_max=%d)",
               v41_kind(&g), g.ctx_span, g.req_flags[0] ? " " : "", g.req_flags, g.max_tokens);
    ds4_engine_v41_set_progress(v41_progress_cb, &g);
    mon_prefill(s, j->mon, g.prompt_tokens, 0, g.max_tokens);   /* 监控: 排队结束, 开始读提示 */
    const int rc = ds4_engine_v41_generate_argmax(s->engine, j->req.prompt.v, j->req.prompt.len, g.max_tokens, v41_emit, &g);
    ds4_engine_v41_set_progress(NULL, NULL);
    ds4_engine_v41_last_spec_stats(&g.spec_rounds, &g.spec_offered, &g.spec_accepted);
    v41_gen_end(&g, rc);
}

