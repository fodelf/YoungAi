/* agent_toolviz_stream.c — 机械拆分自 ds4_agent.c: Streaming Tool Visualization (2/2: DSML 流状态机)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

static void agent_tool_viz_dump_invalid_dsml(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;

    /* The normal path hides DSML and paints a friendly semantic projection.  If
     * parsing fails, show the exact bytes we rejected so the next fix is based
     * on evidence instead of guessing from the projection. */
    if (v->param_active) {
        v->param_active = false;
        v->param_end_len = 0;
        v->param_name[0] = '\0';
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    renderer_color(sr->renderer, "\x1b[1;31m");
    if (sr->parser->raw && sr->parser->raw_len) {
        agent_tool_viz_write(sr, sr->parser->raw, sr->parser->raw_len);
    } else {
        agent_tool_viz_puts(sr, "<empty DSML>");
    }
    renderer_color(sr->renderer, "\x1b[0m");
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
}

static void agent_stream_finish_ignored_dsml(agent_stream_renderer *sr, const char *detail) {
    const char *msg =
        detail && detail[0] ? detail :
        "tool calling is not allowed inside <think></think>";
    sr->dsml_in_think = true;
    sr->dsml_in_think_reported = true;
    agent_trace(sr->renderer->worker, "dsml ignored inside thinking: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[tool call ignored: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
    agent_dsml_parser_reset(sr->parser);
    sr->dsml_active = false;
    sr->dsml_ignored = false;
}

/* Mirror parser progress into the terminal visualizer.  Parser state is the
 * source of truth; this function only decides what the user should see. */
static void agent_stream_tool_events(agent_stream_renderer *sr) {
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && p->current.name)
        agent_tool_viz_tool(sr, p->current.name);
    if (v->tool_announced && !p->current.name && !v->param_active) {
        agent_tool_viz_render_read(sr);
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->read_style = false;
        v->read_prefix_rendered = false;
        v->read_line_rendered = false;
        v->read_path[0] = '\0';
        v->read_start[0] = '\0';
        v->read_max[0] = '\0';
        v->read_whole[0] = '\0';
        v->tool_announced = false;
    }
    if (!v->param_active && p->state == AGENT_DSML_PARAM_VALUE && p->param_name)
        agent_tool_viz_param_begin(sr, p->param_name);
}

static void agent_stream_preflight_closed_param(agent_stream_renderer *sr) {
    if (!sr || sr->replay || sr->dsml_ignored || sr->tool_preflight_error)
        return;
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!p || !v->param_active || strcmp(v->param_name, "old") != 0)
        return;
    if (!p->current.name || strcmp(p->current.name, "edit") != 0)
        return;

    char err[256] = {0};
    if (agent_preflight_edit_old(sr->renderer->worker, &p->current,
                                 err, sizeof(err)))
        return;

    sr->tool_preflight_error = true;
    snprintf(sr->tool_preflight_error_msg, sizeof(sr->tool_preflight_error_msg),
             "edit old selector failed before new was generated: %s",
             err[0] ? err : "old text is not a unique match");
    agent_trace(sr->renderer->worker, "edit old preflight failed: %s",
                sr->tool_preflight_error_msg);
}

static void agent_stream_feed_dsml_byte(agent_stream_renderer *sr, char c) {
    bool was_param = !sr->dsml_ignored && sr->viz.param_active;
    agent_dsml_feed(sr->parser, &c, 1);
    if (!sr->dsml_ignored) {
        agent_stream_tool_events(sr);
        if (was_param) agent_tool_viz_param_value_byte(sr, c);
        if (was_param && sr->parser->state != AGENT_DSML_PARAM_VALUE &&
            sr->viz.param_active)
        {
            agent_stream_preflight_closed_param(sr);
            agent_tool_viz_param_end(sr);
        }
    }
    if (sr->parser->state == AGENT_DSML_DONE) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        } else {
            agent_trace(sr->renderer->worker, "dsml done calls=%d",
                        sr->parser->calls.len);
            agent_tool_viz_finish(sr, NULL);
            sr->dsml_active = false;
        }
    } else if (sr->parser->state == AGENT_DSML_ERROR) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "malformed tool call inside <think></think>");
        } else {
            char status[220];
            snprintf(status, sizeof(status), "[invalid tool call: %s]\n",
                     sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_trace(sr->renderer->worker, "dsml error %s",
                        sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_tool_viz_dump_invalid_dsml(sr);
            agent_tool_viz_finish(sr, status);
            sr->dsml_active = false;
        }
    }
}

/* Start a DSML block from the streaming detector.  The detector may accept a
 * known malformed opening form for robustness, but the parser is seeded with
 * canonical bytes so all later parsing remains strict. */
static void agent_stream_start_dsml(agent_stream_renderer *sr, bool ignored) {
    sr->dsml_active = true;
    sr->dsml_ignored = ignored;
    if (ignored) sr->dsml_in_think = true;
    sr->dsml_start_len = 0;
    sr->post_think_gap = false;
    agent_trace(sr->renderer->worker, "dsml start detected%s",
                ignored ? " inside thinking" : "");
    agent_dsml_start(sr->parser);
    if (!ignored) {
        agent_tool_viz_start(sr);
        agent_stream_tool_events(sr);
    }
}

static void agent_stream_flush_start_tail(agent_stream_renderer *sr) {
    if (!sr->dsml_start_len) return;
    sr->post_think_gap = false;
    for (size_t i = 0; i < sr->dsml_start_len; i++)
        renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
    sr->dsml_start_len = 0;
}

static bool agent_stream_dsml_start_match(const char *tail, size_t len,
                                          bool *complete) {
    static const char canonical[] = "<｜DSML｜tool_calls>";
    static const char missing_bar[] = "<DSML｜tool_calls>";
    const char *forms[] = {canonical, missing_bar};
    *complete = false;
    for (size_t i = 0; i < sizeof(forms)/sizeof(forms[0]); i++) {
        size_t form_len = strlen(forms[i]);
        if (len <= form_len && memcmp(forms[i], tail, len) == 0) {
            *complete = len == form_len;
            return true;
        }
    }
    return false;
}

static bool agent_tail_matches(const char *tail, size_t len,
                               const char *needle, size_t needle_len) {
    return len >= needle_len &&
           memcmp(tail + len - needle_len, needle, needle_len) == 0;
}

static void agent_stream_note_thinking_byte(agent_stream_renderer *sr, char c) {
    if (!sr->in_think || sr->dsml_in_think) return;
    if (sr->think_dsml_len == sizeof(sr->think_dsml_tail)) {
        memmove(sr->think_dsml_tail, sr->think_dsml_tail + 1,
                sizeof(sr->think_dsml_tail) - 1);
        sr->think_dsml_len--;
    }
    sr->think_dsml_tail[sr->think_dsml_len++] = c;

    static const char fullwidth_marker[] = "｜DSML｜";
    static const char ascii_marker[] = "|DSML|";
    if (agent_tail_matches(sr->think_dsml_tail, sr->think_dsml_len,
                           fullwidth_marker, sizeof(fullwidth_marker) - 1) ||
        agent_tail_matches(sr->think_dsml_tail, sr->think_dsml_len,
                           ascii_marker, sizeof(ascii_marker) - 1))
    {
        sr->dsml_in_think = true;
    }
}

/* Route ordinary assistant bytes either to normal markdown rendering or into
 * the DSML detector.  The detector must hold short prefixes because the model
 * can split "<｜DSML｜tool_calls>" across arbitrary tokens. */
static void agent_stream_normal_byte(agent_stream_renderer *sr, char c) {
    static const char start[] = "<｜DSML｜tool_calls>";
    agent_stream_note_thinking_byte(sr, c);

    /* DeepSeek usually emits one or more blank lines after </think> before
     * either prose or a DSML tool stanza.  At that point the bytes are just a
     * visual gap between the hidden thinking phase and the real answer, and
     * printing them makes tool calls appear after odd empty lines.  We only
     * suppress whitespace in this very narrow post-thinking window; once the
     * first non-space byte arrives, normal rendering resumes. */
    if (sr->post_think_gap &&
        (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
    {
        return;
    }

    if (sr->dsml_start_len || c == start[0]) {
        if (sr->dsml_start_len < sizeof(sr->dsml_start_tail))
            sr->dsml_start_tail[sr->dsml_start_len++] = c;
        bool complete = false;
        if (agent_stream_dsml_start_match(sr->dsml_start_tail, sr->dsml_start_len,
                                          &complete))
        {
            if (complete) {
                /* Accept the common missing-leading-bar typo
                 * "<DSML｜tool_calls>" here, but seed the parser with the
                 * canonical marker so the rest of the DSML parser stays
                 * strict and simple. */
                agent_stream_start_dsml(sr, sr->in_think);
            }
            return;
        }
        if (sr->dsml_start_len > 1 &&
            sr->dsml_start_tail[sr->dsml_start_len - 1] == start[0])
        {
            sr->post_think_gap = false;
            size_t flush = sr->dsml_start_len - 1;
            for (size_t i = 0; i < flush; i++)
                renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
            sr->dsml_start_tail[0] = start[0];
            sr->dsml_start_len = 1;
            return;
        }
        agent_stream_flush_start_tail(sr);
        return;
    }

    sr->post_think_gap = false;
    renderer_write_char(sr->renderer, c);
}

/* This is the single streaming display state machine for assistant output.  It
 * hides raw DSML as soon as the tool_calls marker is complete, lets the DSML
 * parser continue building executable calls, and paints semantic tool output
 * from parser state changes.  The sampled transcript remains unchanged: only
 * the terminal projection is rewritten. */
void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = sr->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (sr->pending_len) memcpy(buf, sr->pending, sr->pending_len);
    if (len) memcpy(buf + sr->pending_len, text, len);
    sr->pending_len = 0;

    /* The UI may reset terminal attributes while redrawing the editable prompt
     * between generated chunks.  If a DSML parameter is still streaming, make
     * each new token fragment self-contained by restoring the active parameter
     * color before visible bytes are projected.  This keeps the prompt normal
     * without sacrificing long write/edit content coloring. */
    if (len) agent_tool_viz_restore_param_color(sr);
    if (len && !sr->dsml_active) renderer_restore_text_attrs(sr->renderer);

    size_t i = 0;
    while (i < total) {
        char *cur = buf + i;
        size_t rem = total - i;
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_open)) {
            agent_stream_flush_start_tail(sr);
            sr->post_think_gap = false;
            sr->in_think = true;
            sr->renderer->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_close)) {
            agent_stream_flush_start_tail(sr);
            sr->in_think = false;
            sr->renderer->in_think = false;
            renderer_reset_color(sr->renderer);
            if (!sr->renderer->last_output_newline)
                renderer_write(sr->renderer, "\n", 1);
            renderer_write(sr->renderer, "\n", 1);
            sr->renderer->last_output_newline = true;
            sr->post_think_gap = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && !sr->dsml_active && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(sr->pending)) {
                memcpy(sr->pending, cur, rem);
                sr->pending_len = rem;
            }
            break;
        }

        if (sr->dsml_active) {
            agent_stream_feed_dsml_byte(sr, cur[0]);
        } else if (sr->in_think) {
            /* Tool calls are executable only after thinking has closed.  Still
             * route thinking bytes through the DSML start detector so an
             * accidental in-think tool stanza can be suppressed cleanly instead
             * of being shown as raw markup or, worse, executed. */
            agent_stream_normal_byte(sr, cur[0]);
        } else {
            agent_stream_normal_byte(sr, cur[0]);
        }
        i++;
    }
    free(buf);

    if (finish) {
        agent_stream_flush_start_tail(sr);
        sr->post_think_gap = false;
        if (sr->dsml_active) {
            if (sr->dsml_ignored) {
                agent_stream_finish_ignored_dsml(
                    sr, "tool calling is not allowed inside <think></think>");
            } else {
                agent_tool_viz_finish(sr, sr->tool_preflight_error ?
                                      "[tool call stopped: edit old selector failed]\n" :
                                      "[tool call interrupted]\n");
                sr->dsml_active = false;
            }
        }
        if (sr->dsml_in_think && !sr->dsml_in_think_reported) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        }
    }
}
