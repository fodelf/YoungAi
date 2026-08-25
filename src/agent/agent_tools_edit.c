/* agent_tools_edit.c — 机械拆分自 ds4_agent.c: Edit And Search Tools (1/2: edit 工具)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Edit And Search Tools
 * ============================================================================
 */

static int agent_write_file_bytes(const char *path, const char *data, size_t len,
                                  char *err, size_t errlen) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    size_t wr = fwrite(data, 1, len, fp);
    if (wr != len) {
        snprintf(err, errlen, "write %s: %s", path, strerror(errno));
        fclose(fp);
        return -1;
    }
    if (fclose(fp) != 0) {
        snprintf(err, errlen, "close %s: %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

static void agent_edit_result_append_line(agent_buf *b, const char *data,
                                          const agent_line_span *sp,
                                          int line) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%d ", line);
    agent_buf_puts(b, prefix);
    agent_buf_append(b, data + sp->start, sp->content_end - sp->start);
    agent_buf_puts(b, "\n");
}

/* Successful edits return the nearby post-edit file shape.  This spends cheap
 * prefill tokens to save expensive model retries: the model immediately sees
 * shifted line numbers, braces, semicolons, and accidental duplication. */
void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end) {
    enum {
        CONTEXT_BEFORE = 5,
        CONTEXT_AFTER = 8,
        EDITED_CONTEXT_HEAD = 18,
        EDITED_CONTEXT_TAIL = 18
    };

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (spans.len <= 0) {
        agent_line_spans_free(&spans);
        return;
    }

    if (anchor_start < 1) anchor_start = 1;
    if (anchor_start > spans.len) anchor_start = spans.len;
    if (anchor_end < anchor_start) anchor_end = anchor_start;
    if (anchor_end > spans.len) anchor_end = spans.len;

    int ctx_start = anchor_start - CONTEXT_BEFORE;
    if (ctx_start < 1) ctx_start = 1;
    int ctx_end = anchor_end + CONTEXT_AFTER;
    if (ctx_end > spans.len) ctx_end = spans.len;

    char hdr[PATH_MAX + 160];
    snprintf(hdr, sizeof(hdr),
             "Current file around edit: %s lines %d-%d of %d\n",
             path, ctx_start, ctx_end, spans.len);
    agent_buf_puts(b, hdr);

    int edited_lines = anchor_end - anchor_start + 1;
    if (edited_lines <= EDITED_CONTEXT_HEAD + EDITED_CONTEXT_TAIL) {
        for (int line = ctx_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    } else {
        int head_end = anchor_start + EDITED_CONTEXT_HEAD - 1;
        int tail_start = anchor_end - EDITED_CONTEXT_TAIL + 1;
        for (int line = ctx_start; line <= head_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
        snprintf(hdr, sizeof(hdr),
                 "... %d edited lines omitted ...\n",
                 tail_start - head_end - 1);
        agent_buf_puts(b, hdr);
        for (int line = tail_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    }

    agent_line_spans_free(&spans);
}

static const char *agent_memmem_simple(const char *hay, size_t hay_len,
                                       const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    size_t last = hay_len - needle_len;
    for (size_t i = 0; i <= last; i++) {
        if (hay[i] == needle[0] && !memcmp(hay + i, needle, needle_len))
            return hay + i;
    }
    return NULL;
}

static bool agent_find_unique(const char *data, size_t len,
                              const char *needle, size_t needle_len,
                              const char **match, const char *label,
                              char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    const char *first = agent_memmem_simple(data, len, needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique", label);
        return false;
    }
    *match = first;
    return true;
}

/* Find an anchor only in the suffix after start.
 *
 * Anchored edits use "head [upto] tail": the head fixes the edit start, and
 * the tail should delimit the first unique end point after that start. A tail
 * may legitimately appear earlier in the file, so checking global uniqueness
 * would reject valid edits.
 */
static bool agent_find_unique_after(const char *data, size_t len,
                                    const char *start,
                                    const char *needle, size_t needle_len,
                                    const char **match, const char *label,
                                    char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    if (start < data || start > data + len) {
        snprintf(err, err_len, "%s search starts outside file", label);
        return false;
    }
    size_t off = (size_t)(start - data);
    const char *first = agent_memmem_simple(data + off, len - off,
                                            needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found after old head", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique after old head", label);
        return false;
    }
    *match = first;
    return true;
}

static bool agent_edit_old_may_be_closing_tag(const char *text, size_t len) {
    size_t i = 0;
    while (i < len && (text[i] == ' ' || text[i] == '\t' ||
                       text[i] == '\r' || text[i] == '\n'))
        i++;
    return i < len && text[i] == '<';
}

static bool agent_edit_old_is_only_space(const char *text, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (text[i] != ' ' && text[i] != '\t' &&
            text[i] != '\r' && text[i] != '\n')
            return false;
    }
    return true;
}

static bool agent_span_has_nonspace(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (!isspace((unsigned char)s[i])) return true;
    }
    return false;
}

static bool agent_edit_old_prefix_mature_for_upto(const char *old, size_t old_len) {
    if (old_len < AGENT_EDIT_UPTO_MIN_PREFIX_BYTES) return false;
    int nonempty_lines = 0;
    bool line_has_text = false;
    for (size_t i = 0; i < old_len; i++) {
        if (old[i] == '\n') {
            if (line_has_text) nonempty_lines++;
            line_has_text = false;
        } else if (!isspace((unsigned char)old[i])) {
            line_has_text = true;
        }
    }
    return nonempty_lines >= AGENT_EDIT_UPTO_MIN_PREFIX_LINES;
}

static bool agent_edit_old_ready_for_upto(const char *old, size_t old_len) {
    if (!old_len || strstr(old, "[upto]")) return false;
    if (!agent_edit_old_prefix_mature_for_upto(old, old_len)) return false;
    size_t end = old_len;
    while (end > 0 && (old[end - 1] == ' ' || old[end - 1] == '\t' ||
                       old[end - 1] == '\r'))
        end--;
    return end > 0 && old[end - 1] == '\n';
}

/* While the model streams an edit old=... argument, stop it from retyping a
 * large exact old block once the emitted prefix is already a unique file
 * anchor.  The next sampled token is inspected before eval: if it would keep
 * writing old text rather than close the parameter, the caller evaluates a
 * complete "[upto]" marker line instead. */
bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
                                                  agent_dsml_parser *p,
                                                  const char *next_text,
                                                  size_t next_len) {
    if (!forcer || !p) return false;
    bool in_edit_old = p->state == AGENT_DSML_PARAM_VALUE &&
        p->current.name && strcmp(p->current.name, "edit") == 0 &&
        p->param_name && strcmp(p->param_name, "old") == 0;
    if (!in_edit_old) {
        forcer->active = false;
        forcer->done = false;
        return false;
    }
    if (!forcer->active) {
        forcer->active = true;
        forcer->done = false;
    }
    if (forcer->done)
        return false;
    if (agent_edit_old_may_be_closing_tag(next_text, next_len) ||
        agent_edit_old_is_only_space(next_text, next_len))
        return false;

    const char *path = agent_tool_arg_value(&p->current, "path");
    if (!path || !path[0] || p->param_value_start > p->raw_len) return false;

    const char *old = p->raw + p->param_value_start;
    size_t old_len = p->raw_len - p->param_value_start;
    if (!agent_edit_old_ready_for_upto(old, old_len)) return false;

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0)
        return false;

    const char *match = NULL;
    bool unique = agent_find_unique(data, len, old, old_len, &match,
                                    "old prefix", err, sizeof(err));
    free(data);
    if (unique) {
        forcer->done = true;
        return true;
    }
    return false;
}

static bool agent_edit_find_old_span(const char *data, size_t len,
                                     const char *old, const char **match,
                                     size_t *match_len, bool *anchored,
                                     char *err, size_t err_len) {
    static const char marker[] = "[upto]";
    size_t old_len = strlen(old);
    const char *upto = strstr(old, marker);
    if (!upto) {
        *anchored = false;
        if (!agent_find_unique(data, len, old, old_len, match, "old text",
                               err, err_len))
            return false;
        *match_len = old_len;
        return true;
    }
    if (strstr(upto + strlen(marker), marker)) {
        snprintf(err, err_len, "old text contains more than one [upto] marker");
        return false;
    }
    size_t head_len = (size_t)(upto - old);
    const char *tail = upto + strlen(marker);
    size_t tail_len = old_len - head_len - strlen(marker);
    if (!agent_span_has_nonspace(tail, tail_len)) {
        snprintf(err, err_len,
                 "old text after [upto] must include a unique tail anchor");
        return false;
    }
    const char *head_pos = NULL;
    const char *tail_pos = NULL;
    if (!agent_find_unique(data, len, old, head_len, &head_pos, "old head",
                           err, err_len))
        return false;
    if (!agent_find_unique_after(data, len, head_pos + head_len,
                                 tail, tail_len, &tail_pos, "old tail",
                                 err, err_len))
        return false;
    *anchored = true;
    *match = head_pos;
    *match_len = (size_t)(tail_pos - head_pos) + tail_len;
    return true;
}

bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return true; /* Cannot preflight until path is known. */

    const char *old = agent_tool_arg_value(call, "old");
    if (!old || !old[0]) {
        snprintf(err, err_len, "edit requires non-empty old text");
        return false;
    }

    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, err_len) != 0)
        return false;

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    bool ok = agent_edit_find_old_span(data, len, old, &match, &match_len,
                                       &anchored, err, err_len);
    free(data);
    return ok;
}

static char *agent_apply_file_splice(const char *path,
                                     const char *data, size_t len,
                                     size_t offset, size_t remove_len,
                                     const char *insert, const char *kind) {
    char err[256];
    if (!insert) insert = "";
    size_t insert_len = strlen(insert);
    size_t out_len = offset + insert_len + (len - offset - remove_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, offset);
    memcpy(out + offset, insert, insert_len);
    memcpy(out + offset + insert_len, data + offset + remove_len,
           len - offset - remove_len);
    out[out_len] = '\0';

    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    if (rc != 0) {
        free(out);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int start_line = 0, end_line = 0, delta = 0;
    agent_old_new_line_effect(data, len, out, out_len, offset, remove_len,
                              &start_line, &end_line, &delta);
    char *result = agent_edit_result(path, start_line, end_line, delta,
                                     out, out_len, kind);
    free(out);
    return result;
}

/* Old/new editing is intentionally conservative: exact old text must be unique.
 * For large replacements, old may contain one [upto] marker: the head must be
 * unique, and the tail must be unique after that head before the whole span is
 * replaced. */
char *agent_tool_edit(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return xstrdup("Tool error: edit requires path\n");
    const char *old = agent_tool_arg_value(call, "old");
    const char *new_text = agent_tool_arg_value(call, "new");
    if (!old || !old[0]) return xstrdup("Tool error: edit requires non-empty old text\n");
    if (!new_text) return xstrdup("Tool error: edit requires new text\n");

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    if (!agent_edit_find_old_span(data, len, old, &match, &match_len,
                                  &anchored, err, sizeof(err)))
    {
        free(data);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char *result = agent_apply_file_splice(path, data, len,
                                           (size_t)(match - data), match_len,
                                           new_text,
                                           anchored ? "anchored old/new replacement"
                                                    : "old/new replacement");
    free(data);
    return result;
}
