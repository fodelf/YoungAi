/* server_anthropic_stream.c — 机械拆分自 ds4_server.c (7906-8197 行): Anthropic 流式输出更新。 */

#include "server_internal.h"

static bool anthropic_tool_stream_update(int fd, server *s, const char *id,
                                         anthropic_stream *st,
                                         const char *raw, size_t raw_len) {
    anthropic_tool_stream *ts = &st->tool;
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == DSML_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->syn->tool_calls_end)) {
                ts->parse_pos += strlen(ts->syn->tool_calls_end);
                ts->active = false;
                ts->state = DSML_TOOL_DONE;
                return true;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos,
                                ts->syn->tool_calls_end, ts->syn->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->syn->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!anthropic_tool_start_invoke(fd, s, st, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            /* 伪 XML invoke 开标签 <lightpanda_search">: 与最终解析同一套标签规则 */
            if (dsml_tag_open_at(raw, raw_len, ts->parse_pos, ts->invoke_name, sizeof ts->invoke_name, NULL)) {
                ts->pseudo_invoke = true;
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!anthropic_tool_start_invoke(fd, s, st, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            if (dsml_tag_open_partial(raw, raw_len, ts->parse_pos)) return true;
            return anthropic_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            const size_t invoke_close = dsml_invoke_close_len(ts->syn, raw, raw_len, ts->parse_pos);   /* 与最终解析同一套标签规则(server_dsml_alias.c) */
            char pseudo_close[64]; size_t pseudo_close_len = 0;
            const bool pseudo_invoke_close =
                dsml_tag_close_at(raw, raw_len, ts->parse_pos, pseudo_close, sizeof pseudo_close, &pseudo_close_len) &&
                ts->pseudo_invoke && !strcmp(pseudo_close, ts->invoke_name);
            if (invoke_close || pseudo_invoke_close || dsml_calls_end_at(ts->syn, raw, raw_len, ts->parse_pos)) {   /* 块闭合直接到了 = 隐式 invoke 闭合, 游标不动 */
                if (ts->args_open &&
                    !anthropic_tool_emit_args_fragment(fd, st, "}", 1)) return false;
                ts->args_open = false;
                if (!anthropic_sse_close_block_live(fd, id, st)) return false;
                if (invoke_close) ts->parse_pos += invoke_close;   /* 相对长度 */
                else if (pseudo_invoke_close) ts->parse_pos = pseudo_close_len;   /* dsml_tag_close_at 的 end_out 是绝对下标 */
                /* 块闭合(隐式 invoke 闭合)不前进, 由外层 BETWEEN_INVOKES 收块 */
                ts->index++;
                ts->pseudo_invoke = false;
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (dsml_invoke_close_partial(ts->syn, raw, raw_len, ts->parse_pos) ||
                dsml_param_open_partial(ts->syn, raw, raw_len, ts->parse_pos) ||
                dsml_calls_end_partial(ts->syn, raw, raw_len, ts->parse_pos) ||
                dsml_tag_close_partial(raw, raw_len, ts->parse_pos) ||
                dsml_tag_open_partial(raw, raw_len, ts->parse_pos)) return true;
            if (dsml_param_open_full(ts->syn, raw, raw_len, ts->parse_pos)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!anthropic_tool_start_param(fd, st, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            /* 伪 XML 参数开标签 <query>: 参数名就是标签名, string=true(与最终解析同一套标签规则) */
            if (dsml_tag_open_at(raw, raw_len, ts->parse_pos, ts->param_name, sizeof ts->param_name, NULL)) {
                ts->pseudo_param = true;
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!anthropic_tool_start_param(fd, st, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return anthropic_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            size_t end_len = 0;
            const char *end = ts->pseudo_param ?
                dsml_tag_close_find(raw + ts->parse_pos, raw_len - ts->parse_pos,
                                    ts->param_name, &end_len) :
                dsml_param_close_find(ts->syn, raw + ts->parse_pos,
                                      raw_len - ts->parse_pos, &end_len);
            if (end) {
                if (!anthropic_tool_finish_param(fd, st, raw,
                                                 (size_t)(end - raw), end_len)) return false;
                ts->pseudo_param = false;
                continue;
            }
            size_t limit = ts->pseudo_param ?
                dsml_pseudo_value_stream_safe_len(raw, ts->parse_pos, raw_len,
                                                  ts->syn, ts->param_name, ts->param_is_string) :
                tool_param_value_stream_safe_len(raw, ts->parse_pos,
                                                 raw_len,
                                                 ts->syn,
                                                 ts->param_is_string);
            if (limit > ts->parse_pos) {
                bool ok = ts->param_is_string ?
                    anthropic_tool_emit_string_value(fd, st, raw + ts->parse_pos,
                                                     limit - ts->parse_pos) :
                    anthropic_tool_emit_args_fragment(fd, st, raw + ts->parse_pos,
                                                      limit - ts->parse_pos);
                if (!ok) return false;
                ts->parse_pos = limit;
            }
            return true;
        }

        return true;
    }
    return true;
}

/* 正文里游离的 </think>(2026-10-11 实撞): 非思考模式下提示已经以 </think> 收了思考, 模型偶尔还会自己再采一个(前面是它的自言自语),
 * 正文模式原样把这个标记推给客户端, 页面上就是一行字面的 "</think>"。最终解析(split_reasoning_content)会把它前面那段切进 reasoning,
 * 流式已经推出去的收不回来, 能做的是不推标记本身。在工具块之前出现的才算(工具块里的字节归工具投影管)。
 * 返回 true = pos 之后有一个完整的游离标记, *cut = 它的起点; 调用方推完 [pos, cut) 再从 cut + 8 接着走。 */
bool text_stream_stray_think(const char *raw, size_t pos, size_t raw_len, bool has_tools, size_t *cut) {
    if (raw_len <= pos) return false;
    const char *th = find_lit_bounded(raw + pos, raw_len - pos, "</think>");
    if (!th) return false;
    const char *tool = has_tools ? find_any_tool_start(raw + pos) : NULL;
    if (tool && tool < th) return false;
    *cut = (size_t)(th - raw);
    return true;
}

size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final) {
    if (raw_len <= start) return raw_len;
    if (!final) {   /* 尾巴可能是 "</think>" 的前半截(标记被切在两个 token 之间): 先压着, 等下一个 token 证明它是不是 */
        const char *lit = "</think>";
        for (size_t k = 7; k >= 1; k--) {
            if (raw_len - start >= k && !memcmp(raw + raw_len - k, lit, k)) { raw_len -= k; break; }
        }
        if (raw_len <= start) return start;
    }

    size_t limit = raw_len;
    if (has_tools) {
        const char *tool = find_any_tool_start(raw + start);
        if (tool) {
            limit = trim_tool_separator_ws(raw, start, (size_t)(tool - raw));
            return utf8_stream_safe_len(raw, start, limit, true);
        }

        if (!final) {
            /* Tool calls are hidden from the API client and returned as
             * structured tool_use/tool_calls blocks.  The whitespace just
             * before the DSML marker is syntax too: if we stream it as
             * assistant text, the next client request sends it back and our
             * renderer adds the canonical "\n\n" separator again.  Hold
             * trailing whitespace until a following non-whitespace byte proves
             * it is ordinary text, or until a tool marker proves it should be
             * dropped. */
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;

            /* Also hold a partial '<...tool_calls...' marker that may be split
             * across generated tokens. */
            const size_t max_marker = 80;
            size_t scan = raw_len - start > max_marker ? raw_len - max_marker : start;
            for (size_t i = raw_len; i > scan; i--) {
                if (raw[i - 1] == '<') {
                    size_t marker = i - 1;
                    if (marker < limit) limit = marker;
                    break;
                }
            }
            limit = trim_tool_separator_ws(raw, start, limit);
        }
    }
    return utf8_stream_safe_len(raw, start, limit, final);
}

bool anthropic_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                        anthropic_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == ANTH_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) {
            limit = (size_t)(close - raw);
        } else if (final) {
            limit = raw_len;
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_THINKING)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_THINKING,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_thinking = true;
            st->emit_pos = limit;
        }

        if (close || final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            if (close) {
                st->emit_pos = (size_t)(close - raw) + strlen("</think>");
                st->mode = ANTH_STREAM_TEXT;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
                return true;
            }
        } else {
            return true;
        }
    }

    if (st->mode == ANTH_STREAM_TEXT) {
        for (size_t cut; text_stream_stray_think(raw, st->emit_pos, raw_len, r->has_tools, &cut); st->emit_pos = cut + 8) {
            if (cut <= st->emit_pos) continue;
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_TEXT, raw + st->emit_pos, cut - st->emit_pos)) return false;
            st->sent_text = true;
        }
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_TEXT,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_text = true;
            st->emit_pos = limit;
        }

        if (tool) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->emit_pos = (size_t)(tool - raw);
            /* On normal token-by-token updates, switch from hidden text to a
             * live tool_use projection as soon as the DSML block starts.  On
             * final catch-up from plain text, leave the block for the existing
             * final emitter so old non-incremental behavior stays unchanged. */
            if (!final &&
                anthropic_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = ANTH_STREAM_TOOL;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
            }
        } else if (final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->mode = ANTH_STREAM_SUPPRESS;
        }
    }

    if (st->mode == ANTH_STREAM_TOOL) {
        if (!anthropic_tool_stream_update(fd, s, id, st, raw, raw_len)) return false;
        if (!st->tool.active) st->mode = ANTH_STREAM_SUPPRESS;
    }
    return true;
}

static bool anthropic_sse_tool_blocks_live(int fd, const request *r, const char *id,
                                           anthropic_stream *st,
                                           const tool_calls *calls) {
    (void)r;
    if (!calls) return true;

    buf b = {0};
    /* Tool calls completed by anthropic_tool_stream_update() have already
     * produced start/delta/stop events.  Only emit the tail calls that were not
     * seen by the live projection, for example if the first DSML bytes only
     * become available during final flush. */
    int already_streamed = st->tool.emitted_any ? st->tool.index : 0;
    if (already_streamed > calls->len) already_streamed = calls->len;
    for (int i = already_streamed; i < calls->len; i++, st->next_index++) {
        const tool_call *tc = &calls->v[i];
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), DS4_TOOL_ID_PREFIX_ANTHROPIC "%s_%d", id, i);
        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"tool_use\",\"id\":",
                   st->next_index);
        json_escape(&b, tc->id && tc->id[0] ? tc->id : idbuf);
        ds4_buf_puts(&b, ",\"name\":");
        json_escape(&b, tc->name ? tc->name : "");
        ds4_buf_puts(&b, ",\"input\":{}}}");
        bool ok = sse_event(fd, "content_block_start", b.ptr);
        ds4_buf_free(&b);
        if (!ok) return false;

        ds4_buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                   st->next_index);
        append_json_object_string(&b, tc->arguments);
        ds4_buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        ds4_buf_free(&b);
        if (!ok) return false;

        ds4_buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        ds4_buf_free(&b);
        if (!ok) return false;
    }
    return true;
}

static bool anthropic_sse_stop_live(int fd, const char *finish,
                                    int completion_tokens) {
    buf b = {0};
    ds4_buf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    ds4_buf_puts(&b, ",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    ds4_buf_printf(&b, "%d}}", completion_tokens);
    bool ok = sse_event(fd, "message_delta", b.ptr);
    ds4_buf_free(&b);
    if (ok) ok = sse_event(fd, "message_stop", "{\"type\":\"message_stop\"}");
    return ok;
}

bool anthropic_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                      anthropic_stream *st, const char *raw,
                                      size_t raw_len, const char *recovered_content,
                                      const tool_calls *calls,
                                      const char *finish, int completion_tokens) {
    if (!anthropic_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;
    /* 流上已经露出的 tool_use 必须与最终解析逐个对上, 对不上就发 error 事件作废这一轮(Claude Code 见 error 自动重试), 不报 stop_reason=tool_use;
     * 原因与做法同 openai_sse_finish_live。工具块投影保持边生成边发: Claude Code 的流看门狗只认真事件, ping 连续 30 个以后不再续命(2.1.301 实读),
     * 写长文件的块扣到收尾才发会被它判断流超时 */
    const int ncalls = calls ? calls->len : 0;
    if (st->tool.emitted_any && (st->tool.state != DSML_TOOL_DONE || st->tool.index != ncalls))
        return sse_error_event(fd, r, "model produced a malformed tool call; the partial tool_use already streamed is void");
    /* 还没露出任何 tool_use 就被最终解析否决: 扣住的原文补成一个 text 块, 与非流式回的 text 一致(做法同 responses_sse_finish_live) */
    if (recovered_content && raw && !st->tool.emitted_any && st->emit_pos < raw_len) {
        if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
        if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_TEXT, raw + st->emit_pos, raw_len - st->emit_pos)) return false;
        if (!anthropic_sse_close_block_live(fd, id, st)) return false;
        st->sent_text = true;
        st->emit_pos = raw_len;
    }

    if (st->sent_thinking && !st->sent_text && (!calls || calls->len == 0)) {
        if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
        if (!anthropic_sse_close_block_live(fd, id, st)) return false;
    }

    if (!anthropic_sse_tool_blocks_live(fd, r, id, st, calls)) return false;
    return anthropic_sse_stop_live(fd, finish, completion_tokens);
}
