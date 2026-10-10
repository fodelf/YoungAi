    while (!g_stop_requested && completion < max_tokens &&
           ds4_session_pos(s->session) < ds4_session_ctx(s->session)) {
        /* 自由区预算用尽仍未收束(无 stop/EOS) → 模型在干活不是在回答, 回跳注工具帧。
         * 已生成文本留在 text buf 作 content 前缀。 */
        if (primer_free_first && !primer_free_retry && completion >= primer_free_budget) {
            primer_free_retry = true;
            trace_event(s, trace_id, "primer free region unconcluded after %d tokens -> guided frame",
                        completion);
            goto guided_primer;
        }
        dsml_decode_state dsml_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            dsml_tracker.decode : DSML_DECODE_OUTSIDE;
        const bool in_tool_call = dsml_decode_state_is_tool(dsml_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (saw_tool_start || in_tool_call))) {
            kv_cache_maybe_store_continued(s);
        }
        float temperature = j->req.temperature;
        int top_k = j->req.top_k;
        float top_p = j->req.top_p;
        float min_p = j->req.min_p;
        if (ds4_think_mode_enabled(j->req.think_mode)) {
            temperature = DS4_DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DS4_DEFAULT_TOP_P;
            min_p = DS4_DEFAULT_MIN_P;
        }
        int token = ds4_session_sample(s->session, temperature, top_k, top_p, min_p, &rng);
        if (token == ds4_token_eos(s->engine)) {
            finish = "stop";
            break;
        }

        int toks[513];  /* wave 69: chained spec-pipe, up to 1 + DEPTH(<=7) verify cycles
                         * (8*64 rows + first_token); was [65]/[129]. */
        int ntok = 0;
        /* Greedy decode always takes the speculative entry: it self-gates to
         * plain flat decode unless a drafter is armed (--spec), so this path
         * is lossless either way. */
        if (temperature <= 0.0f)
        {
            ntok = ds4_session_eval_speculative_argmax(s->session,
                                                       token,
                                                       max_tokens - completion,
                                                       ds4_token_eos(s->engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                finish = "error";
                break;
            }
        } else {
            if (ds4_session_eval(s->session, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop_decode = false;
        for (int ti = 0; ti < ntok && completion < max_tokens; ti++) {
            token = toks[ti];
            if (token == ds4_token_eos(s->engine)) {
                finish = "stop";
                stop_decode = true;
                break;
            }

            size_t piece_len = 0;
            char *piece = ds4_token_text(s->engine, token, &piece_len);
            completion++;
            mon_token(s, j->mon, completion);

            trace_piece(s, trace_id, piece, piece_len);
            ds4_buf_append(&text, piece, piece_len);
            thinking_state_feed(&thinking, piece, piece_len);
            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
            }

            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = stop_list_find_from(&j->req.stops, text.ptr,
                                                stop_scan_from,
                                                &stop_pos, &stop_len);
            size_t stream_len = hit_stop ?
                stop_pos : stop_list_stream_safe_len(&j->req.stops, text.len);
            if (stream_len > text.len) stream_len = text.len;
            stream_len = utf8_stream_safe_len(text.ptr, plain_stream_pos,
                                              stream_len, hit_stop);
            if (!hit_stop && j->req.stops.max_len > 1) {
                const size_t hold = j->req.stops.max_len - 1;
                stop_scan_from = text.len > hold ? text.len - hold : 0;
            }

            if (j->req.stream && !structured_stream && stream_len > plain_stream_pos) {
                char *delta = xstrndup(text.ptr + plain_stream_pos, stream_len - plain_stream_pos);
                bool ok = sse_chunk(j->fd, &j->req, id, delta, NULL);
                free(delta);
                if (!ok) {
                    finish = "error";
                    snprintf(err, sizeof(err), "client stream write failed");
                    free(piece);
                    stop_decode = true;
                    break;
                }
                plain_stream_pos = stream_len;
            }
            if (j->req.stream && j->req.api == API_ANTHROPIC &&
                !anthropic_sse_stream_update(j->fd, s, &j->req, id,
                                             &anthropic_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (openai_live_chat &&
                !openai_sse_stream_update(j->fd, s, &j->req, id,
                                          &openai_live, text.ptr, stream_len,
                                          false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (responses_live_chat &&
                !responses_sse_stream_update(j->fd, &j->req,
                                             &responses_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            free(piece);

            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (thinking_gates_tool_markers && thinking.inside) {
                    /* A DSML block inside reasoning is not executable.  This is
                     * the live guard: do not let a quoted or mistaken marker in
                     * <think> stop decoding as a real tool call. */
                    tool_scan_waiting_for_think_close = true;
                    tool_scan_from = text.len;
                } else {
                    if (tool_scan_waiting_for_think_close) {
                        const char *think_end = find_last_substr(text.ptr, "</think>");
                        tool_scan_from = think_end ? (size_t)((think_end + 8) - text.ptr) : text.len;
                        if (tool_scan_from > text.len) tool_scan_from = text.len;
                        tool_scan_waiting_for_think_close = false;
                    }
                    if (tool_scan_from > text.len) tool_scan_from = text.len;
                    const char *tool_scan = text.ptr ? text.ptr + tool_scan_from : "";
                    bool orphan_end = false;
                    bool old_start = saw_tool_start;
                    bool old_end = saw_tool_end;
                    observe_tool_markers(tool_scan, &saw_tool_start, &saw_tool_end, &orphan_end);
                    if (orphan_end && !saw_orphan_tool_end) {
                        saw_orphan_tool_end = true;
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                                   ctx_span,
                                   req_flags[0] ? " " : "",
                                   req_flags,
                                   completion);
                        trace_event(s, trace_id,
                                    "ignored orphan tool-call end marker after %d generated tokens",
                                    completion);
                    }
                    if (saw_tool_start && !old_start) {
                        trace_event(s, trace_id, "entered tool-call block after %d generated tokens", completion);
                    }
                    if (saw_tool_end && !old_end) {
                        trace_event(s, trace_id, "closed tool-call block after %d generated tokens", completion);
                    }
                    const size_t marker_hold = 80;
                    size_t hold_from = text.len > marker_hold ? text.len - marker_hold : 0;
                    if (hold_from > tool_scan_from) tool_scan_from = hold_from;
                    if (s->trace && completion >= next_tool_progress) {
                        trace_event(s, trace_id,
                                    "progress gen=%d dsml_start=%d dsml_end=%d",
                                    completion, saw_tool_start ? 1 : 0, saw_tool_end ? 1 : 0);
                        next_tool_progress += 128;
                    }
                }
            }

            /* 客户端走了就别再算(2026-09-22, 与 V4.1 路同一条判据): 非流式请求整个生成期间不写字节,
             * 不主动探就发现不了对端挂断, 而本服务串行跑图 —— 僵尸请求会把后面排队的也拖超时。 */
            if (completion >= next_alive_check) {
                next_alive_check = completion + 16;
                if (client_disconnected(j->fd)) {
                    finish = "error";
                    snprintf(err, sizeof(err), "client disconnected after %d generated tokens", completion);
                    trace_event(s, trace_id, "client disconnected after %d generated tokens", completion);
                    stop_decode = true;
                    break;
                }
            }

            if (completion >= next_decode_log) {
                log_decode_progress(j->req.kind, prompt_tokens, completion,
                                    responses_protocol,
                                    j->req.has_tools,
                                    thinking.inside,
                                    saw_tool_start,
                                    saw_tool_end,
                                    decode_t0,
                                    &last_decode_log_t,
                                    &last_decode_log_completion);
                next_decode_log += 50;
            }

            if (hit_stop) {
                (void)stop_len;
                finish = "stop";
                text.len = stop_pos;
                text.ptr[text.len] = '\0';
                ds4_session_invalidate(s->session);
                stop_decode = true;
                break;
            }

            if (j->req.kind == REQ_CHAT && j->req.has_tools && saw_tool_end) {
                finish = "tool_calls";
                stop_decode = true;
                break;
            }
        }
        if (stop_decode) break;
    }

    /* ★transcript echo is not an answer (2026-07-17 Edit 决胜针实证)★
     * The free region can reproduce the user message verbatim and then hit a
     * stop/EOS inside the budget window -- which the free-first logic reads as
     * "concluded naturally = plain-text answer" and returns without ever
     * injecting a tool frame.  An echo of the transcript is not an answer: if
     * the generated text's opening bytes appear verbatim in the conversation
     * region of the prompt, discard the echo and fall through to the guided
     * frame (same contract as the budget-exhausted path). */
    if (primer_free_first && !primer_free_retry && !strcmp(finish, "stop") &&
        text.len >= 48 && j->req.prompt_text) {
        const char *cs = j->req.prompt_text;
        size_t cl = strlen(cs);
        if (j->req.prompt_conv_off > 0 && j->req.prompt_conv_off < cl) {
            cs += j->req.prompt_conv_off;
        } else {
            const char *u = strstr(cs, "<｜User｜>");
            if (!u) u = strstr(cs, "\n# User:");
            if (u) cs = u;
        }
        char echo_probe[65];
        size_t pl = text.len < 64 ? text.len : 64;
        memcpy(echo_probe, text.ptr, pl);
        echo_probe[pl] = '\0';
        if (strstr(cs, echo_probe)) {
            trace_event(s, trace_id,
                        "primer free region echoed the transcript (%zu bytes) -> guided frame",
                        text.len);
            /* the echo is a dead-end KV state, not a draft: roll back to the
             * prompt so the guided frame decodes from the same state as a
             * forced tool_choice turn (measured clean), and drop the text. */
            text.len = 0;
            if (text.ptr) text.ptr[0] = '\0';
            ds4_session_rewind(s->session, primer_prompt_pos);
            primer_free_retry = true;
            goto guided_primer;
        }
    }

    /* A hard truncation can cut a multi-byte token mid-UTF-8-sequence, making
     * the NON-STREAMING JSON body invalid UTF-8 (measured 2026-07-07 with the
     * output cap; the streaming path already guards via utf8_stream_safe_len).
     * Trim the partial tail -- a clean stop/EOS tail is a no-op. */
    if (text.ptr && text.len > 0) {
        text.len = utf8_stream_safe_len(text.ptr, 0, text.len, false);
        text.ptr[text.len] = '\0';
    }

    /* --max-output-tokens truncation is SERVER policy, not the client's own
     * limit.  Anthropic path only: report a normal stop ("end_turn") instead of
     * "length", because Claude Code treats stop_reason=max_tokens as "response
     * exceeded my configured maximum" and aborts the whole session (measured
     * 2026-07-07).  OpenAI/Responses keep the truthful "length" (2026-09-22):
     * a caller that stores the answer must be able to tell a finished report
     * from one the server cut off mid-sentence. */
    if (!strcmp(finish, "length") &&
        j->req.api == API_ANTHROPIC &&
        s->max_output_tokens > 0 &&
        completion >= s->max_output_tokens &&
        j->req.max_tokens > s->max_output_tokens)
        finish = "stop";

    if (g_stop_requested && strcmp(finish, "error") != 0) {
        finish = "error";
        snprintf(err, sizeof(err), "shutdown requested");
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        saw_tool_start && !saw_tool_end && strcmp(finish, "error") != 0)
    {
        /* Deterministically complete a simple truncation.  Anything more than
         * missing closing tags stays model-owned: for non-streaming requests,
         * append a tool error plus prompt reminder to the live session and let
         * the model issue a fresh call. */
        const bool completed_truncation = false;
        if (!completed_truncation) {
            if (!j->req.stream && !dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s unterminated tool call; continuing with model-visible tool error",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                trace_event(s, trace_id,
                            "unterminated tool call; continuing with model-visible tool error");
                if (continue_after_invalid_dsml(s, &j->req, &thinking,
                                                "unterminated tool call",
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    dsml_recovery_attempted = true;
                    server_log(DS4_LOG_GENERATION,
                               "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               ctx_span,
                               req_flags[0] ? " " : "",
                               req_flags,
                               recovery_tokens);
                    trace_event(s, trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    ds4_buf_free(&text);
                    goto decode_again;
                }
                finish = "error";
                snprintf(err, sizeof(err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            } else {
                finish = "error";
                snprintf(err, sizeof(err), "unterminated tool call");
            }
        }
    }

    if (completion > last_decode_log_completion) {
        log_decode_progress(j->req.kind, prompt_tokens, completion,
                            responses_protocol,
                            j->req.has_tools,
                            thinking.inside,
                            saw_tool_start,
                            saw_tool_end,
                            decode_t0,
                            &last_decode_log_t,
                            &last_decode_log_completion);
    }

    if (j->req.stream && !structured_stream && text.len > plain_stream_pos) {
        char *tail = xstrndup(text.ptr + plain_stream_pos, text.len - plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, id, tail, NULL)) finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            text.ptr ? text.ptr : "",
            j->req.has_tools,
            saw_tool_start,
            ds4_think_mode_enabled(j->req.think_mode),
            &final_finish,
            err,
            sizeof(err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure);
        if (!parsed_ok && recovered_tool_parse_failure && j->req.has_tools && saw_tool_start) {
            /* parse_generated_message failed even though DSML was present.
             * Semantic repair is intentionally avoided: if the parser cannot
             * execute the block, feed the model a tool error and the protocol
             * reminder so it owns the corrected next action. */
            if (!j->req.stream && !dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                const char *detail = err[0] ? err : "invalid tool call";
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call; continuing with model-visible tool error",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                trace_event(s, trace_id,
                            "invalid tool call; continuing with model-visible tool error");
                if (continue_after_invalid_dsml(s, &j->req, &thinking,
                                                detail,
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    dsml_recovery_attempted = true;
                    server_log(DS4_LOG_GENERATION,
                               "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               ctx_span,
                               req_flags[0] ? " " : "",
                               req_flags,
                               recovery_tokens);
                    trace_event(s, trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    free(parsed_content);
                    free(parsed_reasoning);
                    tool_calls_free(&parsed_calls);
                    ds4_buf_free(&text);
                    goto decode_again;
                }
                final_finish = "error";
                snprintf(err, sizeof(err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            }
            if (!parsed_ok) {
                /* Print raw DSML snippet for debugging */
                size_t dsml_snippet_len = 0;
                const char *dsml_start = NULL;
                const char *p;
                for (p = text.ptr; p && (size_t)(p - text.ptr) < text.len - 20; p++) {
                    if ((strncmp(p, DS4_TOOL_CALLS_START, strlen(DS4_TOOL_CALLS_START)) == 0) ||
                        (strncmp(p, DS4_TOOL_CALLS_START_SHORT, strlen(DS4_TOOL_CALLS_START_SHORT)) == 0) ||
                        (strncmp(p, "<tool_calls>", 12) == 0)) {
                        dsml_start = p;
                        break;
                    }
                }
                if (dsml_start) {
                    dsml_snippet_len = text.len - (dsml_start - text.ptr);
                    if (dsml_snippet_len > 500) dsml_snippet_len = 500;
                }
                /* Also log a snippet of the full text to see what the model output */
                size_t text_snippet_len = text.len > 300 ? 300 : text.len;
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags,
                           final_finish,
                           text.len,
                           saw_tool_start,
                           saw_tool_end,
                           (int)text_snippet_len,
                           text.ptr ? text.ptr : "(null)");
                server_log(DS4_LOG_WARNING,
                           "ds4-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags,
                           (int)dsml_snippet_len,
                           dsml_start ? dsml_start : "(none)");
                trace_event(s, trace_id,
                            "invalid tool call returned as assistant text finish=%s",
                            final_finish);
            }
        }
        if (parsed_calls.len) {
            if (openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream)
                apply_anthropic_stream_tool_ids(&parsed_calls, &anthropic_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        } else if (j->req.api == API_RESPONSES) {
            responses_live_clear(s);
        }
    }
    log_tool_calls_summary(ctx_span, &parsed_calls,
                           responses_protocol);

