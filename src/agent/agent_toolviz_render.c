/* agent_toolviz_render.c — 机械拆分自 ds4_agent.c: Streaming Tool Visualization (1/2: 工具参数可视化渲染)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Streaming Tool Visualization
 * ============================================================================
 *
 * Tool calls are parsed for execution later, but they are also visualized while
 * the model is still sampling.  This state machine suppresses raw DSML and
 * prints compact, tool-specific progress such as "$ command" or
 * "Reading file 1:500...".
 */

static bool streq_any(const char *s, const char *a, const char *b,
                      const char *c, const char *d) {
    return (a && !strcmp(s, a)) || (b && !strcmp(s, b)) ||
           (c && !strcmp(s, c)) || (d && !strcmp(s, d));
}

static agent_tool_param_kind agent_tool_param_kind_for(const char *tool, const char *param) {
    if (!tool) tool = "";
    if (!param) param = "";
    if (!strcmp(tool, "bash") && !strcmp(param, "command"))
        return AGENT_TOOL_PARAM_BASH_COMMAND;
    if (!strcmp(tool, "edit") && !strcmp(param, "old"))
        return AGENT_TOOL_PARAM_DIFF_OLD;
    if (!strcmp(tool, "edit") && !strcmp(param, "new"))
        return AGENT_TOOL_PARAM_DIFF_NEW;
    if (streq_any(param, "path", "file", "filename", NULL))
        return AGENT_TOOL_PARAM_PATH;
    if (streq_any(param, "line", "start_line", "end_line", "offset") ||
        streq_any(param, "start", "end", "count", "max_lines") ||
        streq_any(param, "timeout_sec", "refresh_sec", NULL, NULL))
        return AGENT_TOOL_PARAM_OFFSET;
    if (streq_any(param, "content", "text", NULL, NULL))
        return AGENT_TOOL_PARAM_CONTENT;
    return AGENT_TOOL_PARAM_NORMAL;
}

static const char *agent_tool_param_color(agent_tool_param_kind kind) {
    switch (kind) {
    case AGENT_TOOL_PARAM_PATH: return "\x1b[32m";
    case AGENT_TOOL_PARAM_OFFSET: return "\x1b[33m";
    case AGENT_TOOL_PARAM_CONTENT: return "\x1b[34m";
    case AGENT_TOOL_PARAM_DIFF_OLD: return "\x1b[31m";
    case AGENT_TOOL_PARAM_DIFF_NEW: return "\x1b[32m";
    case AGENT_TOOL_PARAM_BASH_COMMAND: return "\x1b[1;36m";
    default: return "\x1b[37m";
    }
}

void agent_tool_viz_write(agent_stream_renderer *sr, const char *s, size_t n) {
    renderer_plain(sr->renderer, s, n);
    for (size_t i = 0; i < n; i++) sr->viz.last_output_newline = s[i] == '\n';
}

void agent_tool_viz_puts(agent_stream_renderer *sr, const char *s) {
    agent_tool_viz_write(sr, s, strlen(s));
}

void agent_tool_viz_start(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    bool line_open = !sr->renderer->last_output_newline;
    memset(v, 0, sizeof(*v));
    v->active = true;
    v->at_line_start = true;
    v->last_output_newline = true;
    if (sr->replay) {
        if (line_open) agent_tool_viz_puts(sr, "\n");
    } else if (sr->renderer->use_color) {
        /* The raw DSML start marker may arrive after ordinary text on the
         * current row.  Clear that row only for the live terminal UI; plain
         * stdout mode must never leak cursor-control escapes into pipes. */
        agent_tool_viz_puts(sr, "\r\x1b[2K");
    } else if (line_open) {
        agent_tool_viz_puts(sr, "\n");
    }
    v->last_output_newline = true;
}

static void agent_tool_viz_line_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    agent_tool_viz_puts(sr, "🛠️ ");
    v->at_line_start = false;
}

static const char *agent_tool_viz_prefix(const char *name) {
    if (!strcmp(name, "bash")) return "$ ";
    if (!strcmp(name, "read")) return "read ";
    if (!strcmp(name, "write")) return "write ";
    if (!strcmp(name, "edit")) return "edit ";
    if (!strcmp(name, "search")) return "search ";
    if (!strcmp(name, "google_search")) return "google ";
    if (!strcmp(name, "visit_page")) return "visit ";
    return NULL;
}

void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    agent_tool_viz_line_prefix(sr);
    if (v->read_style) {
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        v->read_prefix_rendered = true;
        return;
    }
    renderer_color(sr->renderer, !strcmp(v->tool_name, "bash") ?
                                "\x1b[1;36m" : "\x1b[1;37m");
    const char *prefix = agent_tool_viz_prefix(v->tool_name);
    if (prefix) {
        agent_tool_viz_puts(sr, prefix);
    } else {
        agent_tool_viz_puts(sr, v->tool_name);
        agent_tool_viz_puts(sr, " ");
    }
    renderer_color(sr->renderer, "\x1b[0m");
}

static void agent_tool_viz_append(char *dst, size_t cap, char c) {
    size_t len = strlen(dst);
    if (len + 1 >= cap) return;
    dst[len] = c;
    dst[len + 1] = '\0';
}

static void agent_tool_viz_read_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (!strcmp(v->param_name, "path")) {
        agent_tool_viz_append(v->read_path, sizeof(v->read_path), c);
        if (v->read_prefix_rendered) agent_tool_viz_write(sr, &c, 1);
    } else if (!strcmp(v->param_name, "start_line")) {
        agent_tool_viz_append(v->read_start, sizeof(v->read_start), c);
    } else if (!strcmp(v->param_name, "max_lines")) {
        agent_tool_viz_append(v->read_max, sizeof(v->read_max), c);
    } else if (!strcmp(v->param_name, "whole")) {
        agent_tool_viz_append(v->read_whole, sizeof(v->read_whole), c);
    }
}

void agent_tool_viz_render_read(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->read_style || v->read_line_rendered) return;

    if (!v->read_prefix_rendered) {
        agent_tool_viz_line_prefix(sr);
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, v->read_path[0] ? v->read_path : "<unknown>");
    } else if (!v->read_path[0]) {
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, "<unknown>");
    }
    renderer_color(sr->renderer, "\x1b[33m");
    bool whole = agent_parse_bool_default(v->read_whole, false);
    if (whole && (!v->read_start[0] || !strcmp(v->read_start, "1"))) {
        agent_tool_viz_puts(sr, " (whole file)");
    } else if (whole) {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start);
        agent_tool_viz_puts(sr, ":EOF");
    } else {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, ":");
        agent_tool_viz_puts(sr, v->read_max[0] ? v->read_max : "500");
    }
    renderer_color(sr->renderer, "\x1b[1;37m");
    agent_tool_viz_puts(sr, "...");
    renderer_color(sr->renderer, "\x1b[0m");
    agent_tool_viz_puts(sr, "\n");
    v->read_line_rendered = true;
}

static bool agent_tool_viz_param_is_code_body(agent_tool_visualizer *v) {
    if (!strcmp(v->tool_name, "write") &&
        v->param_kind == AGENT_TOOL_PARAM_CONTENT)
        return true;
    if (!strcmp(v->tool_name, "edit") &&
        (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
         v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW ||
         v->param_kind == AGENT_TOOL_PARAM_CONTENT))
        return true;
    return false;
}

static const char *agent_tool_viz_diff_prefix(agent_tool_param_kind kind,
                                              const char **color) {
    if (color) *color = NULL;
    const char *prefix = NULL;
    if (kind == AGENT_TOOL_PARAM_DIFF_OLD) {
        prefix = "- ";
        if (color) *color = "\x1b[31m";
    } else if (kind == AGENT_TOOL_PARAM_DIFF_NEW) {
        prefix = "+ ";
        if (color) *color = "\x1b[32m";
    }
    return prefix;
}

static void agent_tool_viz_code_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->at_line_start) return;
    const char *color = NULL;
    const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
    if (!prefix) return;
    renderer_color(sr->renderer, color);
    renderer_write(sr->renderer, prefix, strlen(prefix));
    renderer_color(sr->renderer, "\x1b[0m");
    sr->renderer->wrote_visible_output = true;
    sr->renderer->last_output_newline = false;
    v->last_output_newline = false;
    v->at_line_start = false;
}

static void agent_tool_viz_code_begin(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    const agent_syntax *syntax = agent_syntax_for_path(v->tool_path);
    renderer_code_stream_begin(sr->renderer, syntax);
    renderer_code_stream_set_upto_marker(sr->renderer,
        !strcmp(v->tool_name, "edit") &&
        v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD);
    v->code_param_active = true;
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        const char *color = NULL;
        const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
        /* Diff prefixes are terminal UI, not code.  Keep them outside the
         * syntax buffer so a later row repaint preserves their red/green color
         * while highlighting only the actual edited line. */
        renderer_code_stream_set_prefix(sr->renderer, prefix, color);
        agent_tool_viz_code_prefix(sr);
    }
}

static void agent_tool_viz_code_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->code_param_active) return;
    renderer_code_end(sr->renderer);
    v->code_param_active = false;
    v->at_line_start = true;
    v->last_output_newline = sr->renderer->last_output_newline;
}

static void agent_tool_viz_code_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    agent_tool_viz_code_prefix(sr);
    renderer_code_byte(sr->renderer, c);
    v->last_output_newline = c == '\n';
    v->at_line_start = c == '\n';
}

void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && sr->parser->current.name)
        agent_tool_viz_tool(sr, sr->parser->current.name);
    snprintf(v->param_name, sizeof(v->param_name), "%s", name ? name : "");
    v->param_kind = agent_tool_param_kind_for(v->tool_name, v->param_name);
    v->param_active = true;
    v->param_end_len = 0;

    if (v->read_style) return;

    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->at_line_start = true;
        agent_tool_viz_code_begin(sr);
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        if (strcmp(v->tool_name, "write")) {
            renderer_color(sr->renderer, "\x1b[1;37m");
            agent_tool_viz_puts(sr, v->param_name);
            agent_tool_viz_puts(sr, ":\n");
        }
        v->at_line_start = true;
        if (agent_tool_viz_param_is_code_body(v)) {
            agent_tool_viz_code_begin(sr);
        } else {
            renderer_color(sr->renderer, "\x1b[34m");
        }
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, " ");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, "=");
    } else {
        renderer_color(sr->renderer, agent_tool_param_color(AGENT_TOOL_PARAM_BASH_COMMAND));
        return;
    }
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
    if (v->code_param_active) agent_tool_viz_code_end(sr);
    if (!v->read_style) renderer_color(sr->renderer, "\x1b[0m");
    v->param_active = false;
    v->param_name[0] = '\0';
}

static void agent_tool_viz_param_raw_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->read_style) {
        agent_tool_viz_read_value_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_PATH) {
        agent_tool_viz_append(v->tool_path, sizeof(v->tool_path), c);
    }
    if (v->code_param_active) {
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_BASH_COMMAND) {
        agent_tool_viz_write(sr, &c, 1);
        v->at_line_start = c == '\n';
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        agent_tool_viz_code_begin(sr);
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    agent_tool_viz_write(sr, &c, 1);
    v->at_line_start = c == '\n';
}

void agent_tool_viz_restore_param_color(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active || !v->param_active || v->read_style) return;
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

static bool agent_tool_viz_param_end_tail(const char *tail, size_t len, bool *complete) {
    static const char prefix[] = "</｜DSML｜parameter";
    static const char dsml_bar[] = "｜";
    const size_t prefix_len = sizeof(prefix) - 1;
    const size_t bar_len = sizeof(dsml_bar) - 1;
    *complete = false;
    if (len <= prefix_len) return memcmp(prefix, tail, len) == 0;
    if (memcmp(prefix, tail, prefix_len) != 0) return false;
    size_t i = prefix_len;
    while (i < len && (tail[i] == ' ' || tail[i] == '\t' ||
                       tail[i] == '\r' || tail[i] == '\n')) i++;
    if (i < len && len - i <= bar_len) {
        if (memcmp(dsml_bar, tail + i, len - i) == 0) return true;
    }
    if (i + bar_len <= len && memcmp(tail + i, dsml_bar, bar_len) == 0)
        i += bar_len;
    for (; i < len; i++) {
        if (tail[i] == '>') {
            *complete = i == len - 1;
            return *complete;
        }
        if (tail[i] != ' ' && tail[i] != '\t' && tail[i] != '\r' && tail[i] != '\n')
            return false;
    }
    return true;
}

/* Stream one DSML parameter byte into the visualizer.  The visualizer must not
 * wait for the whole parameter: large write/edit contents should show progress
 * as the model emits them, while still detecting the closing parameter tag. */
void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;

    if (v->param_end_len || c == '<') {
        if (v->param_end_len == sizeof(v->param_end_tail)) {
            size_t keep = v->param_end_len;
            v->param_end_len = 0;
            for (size_t i = 0; i < keep; i++)
                agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
            if (c != '<') {
                agent_tool_viz_param_raw_byte(sr, c);
                return;
            }
        }
        if (v->param_end_len < sizeof(v->param_end_tail))
            v->param_end_tail[v->param_end_len++] = c;
        bool complete = false;
        if (agent_tool_viz_param_end_tail(v->param_end_tail, v->param_end_len, &complete)) {
            if (complete) agent_tool_viz_param_end(sr);
            return;
        }
        size_t keep = v->param_end_len;
        v->param_end_len = 0;
        for (size_t i = 0; i < keep; i++)
            agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
        return;
    }
    agent_tool_viz_param_raw_byte(sr, c);
}

void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;
    if (v->param_active) agent_tool_viz_param_end(sr);
    if (!status || !status[0]) agent_tool_viz_render_read(sr);
    if (status && status[0]) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[90m");
        agent_tool_viz_puts(sr, status);
        renderer_color(sr->renderer, "\x1b[0m");
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    v->active = false;
}
