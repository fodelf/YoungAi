/* agent_web.c — 机械拆分自 ds4_agent.c: Browser Web Tools。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Browser Web Tools
 * ============================================================================
 *
 * The browser subsystem lives in ds4_web.c: it owns visible Chrome and CDP.  The
 * agent side only asks for permission, dispatches tools, and caps visit_page
 * output using the same "head plus temp file" shape as bash.
 */

#define AGENT_WEB_HEAD_BYTES (8*1024)
#define AGENT_WEB_HEAD_LINES 100

static int agent_count_lines(const char *s) {
    if (!s || !s[0]) return 0;
    int lines = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '\n') lines++;
    }
    if (s[strlen(s) - 1] != '\n') lines++;
    return lines;
}

static char *agent_string_head(const char *s, int max_lines, size_t max_bytes,
                               int *lines_read, bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!s) return xstrdup("");
    size_t used = 0;
    int lines = 0;
    while (s[used] && used < max_bytes && lines < max_lines) {
        if (s[used++] == '\n') lines++;
    }
    if (s[used] && used >= max_bytes && byte_limited) *byte_limited = true;
    if (used && s[used - 1] != '\n' && lines < max_lines) lines++;
    if (lines_read) *lines_read = lines;
    return xstrndup(s, used);
}

static bool agent_write_temp_text(const char *prefix, const char *text,
                                  char *path, size_t path_len,
                                  char *err, size_t err_len) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s_XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snprintf(err, err_len, "failed to create temporary file: %s", strerror(errno));
        return false;
    }
    size_t len = text ? strlen(text) : 0;
    const char *p = text ? text : "";
    size_t left = len;
    while (left) {
        ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            snprintf(err, err_len, "failed to write temporary file: %s", strerror(errno));
            close(fd);
            unlink(tmpl);
            return false;
        }
        p += n;
        left -= (size_t)n;
    }
    if (close(fd) != 0) {
        snprintf(err, err_len, "failed to close temporary file: %s", strerror(errno));
        unlink(tmpl);
        return false;
    }
    snprintf(path, path_len, "%s", tmpl);
    return true;
}

char *agent_tool_google_search(agent_worker *w, const agent_tool_call *call) {
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: google_search requires query\n");
    char err[256] = {0};
    char *md = ds4_web_google_search(w->web, query, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: google_search failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    return md;
}

char *agent_tool_visit_page(agent_worker *w, const agent_tool_call *call) {
    const char *url = agent_tool_arg_value(call, "url");
    if (!url || !url[0]) return xstrdup("Tool error: visit_page requires url\n");
    char err[256] = {0};
    char *md = ds4_web_visit_page(w->web, url, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char path[PATH_MAX];
    if (!agent_write_temp_text("ds4_agent_web", md, path, sizeof(path),
                               err, sizeof(err)))
    {
        free(md);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "could not store rendered page");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int total_lines = agent_count_lines(md);
    int shown_lines = 0;
    bool byte_limited = false;
    char *head = agent_string_head(md, AGENT_WEB_HEAD_LINES, AGENT_WEB_HEAD_BYTES,
                                   &shown_lines, &byte_limited);
    bool truncated = byte_limited || shown_lines < total_lines;
    agent_buf out = {0};
    char line[PATH_MAX + 256];
    snprintf(line, sizeof(line),
             "visit_page url=%s\noutput_path=%s (%zu bytes, %d lines)\n",
             url, path, strlen(md), total_lines);
    agent_buf_puts(&out, line);
    if (truncated) {
        snprintf(line, sizeof(line), "<head -%d %s>\n",
                 AGENT_WEB_HEAD_LINES, path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</head>\n");
        agent_buf_puts(&out,
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n");
    } else {
        agent_buf_puts(&out, "<markdown>\n");
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</markdown>\n");
    }
    free(head);
    free(md);
    return agent_buf_take(&out);
}
