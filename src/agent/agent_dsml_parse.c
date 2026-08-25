/* agent_dsml_parse.c — 机械拆分自 ds4_agent.c: DSML Tool-Call Parser。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * DSML Tool-Call Parser
 * ============================================================================
 *
 * The model streams raw text tokens.  This parser recognizes completed DSML
 * tool stanzas and keeps a copy of the raw stanza for diagnostics.  It is
 * deliberately strict after the opening marker: typo recovery belongs to the
 * streaming detector so the actual tool parser stays small and predictable.
 */

bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void agent_tool_call_free(agent_tool_call *c) {
    if (!c) return;
    free(c->name);
    for (int i = 0; i < c->argc; i++) {
        free(c->args[i].name);
        free(c->args[i].value);
    }
    free(c->args);
    memset(c, 0, sizeof(*c));
}

static void agent_tool_calls_free(agent_tool_calls *calls) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) agent_tool_call_free(&calls->v[i]);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static void agent_tool_call_add_arg(agent_tool_call *c, const char *name,
                                    const char *value, size_t value_len,
                                    bool is_string) {
    if (c->argc == c->argcap) {
        c->argcap = c->argcap ? c->argcap * 2 : 4;
        c->args = xrealloc(c->args, (size_t)c->argcap * sizeof(c->args[0]));
    }
    c->args[c->argc++] = (agent_tool_arg){
        .name = xstrdup(name),
        .value = xstrndup(value, value_len),
        .is_string = is_string,
    };
}

static void agent_tool_calls_push(agent_tool_calls *calls, agent_tool_call *call) {
    if (!call->name) return;
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 2;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = *call;
    memset(call, 0, sizeof(*call));
}

const char *agent_tool_arg_value(const agent_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}

void agent_dsml_parser_free(agent_dsml_parser *p) {
    if (!p) return;
    free(p->raw);
    agent_tool_call_free(&p->current);
    free(p->param_name);
    agent_tool_calls_free(&p->calls);
    memset(p, 0, sizeof(*p));
}

void agent_dsml_parser_reset(agent_dsml_parser *p) {
    agent_dsml_parser_free(p);
    p->state = AGENT_DSML_SEARCH;
}

static void agent_dsml_raw_append(agent_dsml_parser *p, const char *s, size_t n) {
    if (!n) return;
    if (p->raw_len + n + 1 > p->raw_cap) {
        size_t cap = p->raw_cap ? p->raw_cap * 2 : 512;
        while (cap < p->raw_len + n + 1) cap *= 2;
        p->raw = xrealloc(p->raw, cap);
        p->raw_cap = cap;
    }
    memcpy(p->raw + p->raw_len, s, n);
    p->raw_len += n;
    p->raw[p->raw_len] = '\0';
}

static char *agent_parse_attr(const char *tag, const char *name) {
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    const char *end = strchr(p, '"');
    if (!end) return NULL;
    return xstrndup(p, (size_t)(end - p));
}

static void agent_dsml_set_error(agent_dsml_parser *p, const char *msg) {
    p->state = AGENT_DSML_ERROR;
    snprintf(p->error, sizeof(p->error), "%s", msg);
}

static bool agent_dsml_close_tag_at(const char *s, const char *name, size_t *tag_len) {
    char prefix[64];
    static const char dsml_bar[] = "｜";
    snprintf(prefix, sizeof(prefix), "</｜DSML｜%s", name);
    size_t prefix_len = strlen(prefix);
    if (strncmp(s, prefix, prefix_len) != 0) return false;
    const char *p = s + prefix_len;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (strncmp(p, dsml_bar, strlen(dsml_bar)) == 0) p += strlen(dsml_bar);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '>') return false;
    if (tag_len) *tag_len = (size_t)(p - s) + 1;
    return true;
}

/* Find a DSML closing tag while accepting the few harmless closing-tag variants
 * the model has been observed to emit.  Opening tags stay strict so accidental
 * prose does not become a tool call. */
static char *agent_dsml_find_close_tag(const char *s, const char *name, size_t *tag_len) {
    const char *p = s;
    while ((p = strstr(p, "</｜DSML｜")) != NULL) {
        if (agent_dsml_close_tag_at(p, name, tag_len)) return (char *)p;
        p++;
    }
    return NULL;
}

/* Parse as much of the accumulated DSML buffer as possible.  The parser can be
 * called after every streamed byte: incomplete input leaves state unchanged
 * until enough bytes arrive, while malformed completed input switches to
 * AGENT_DSML_ERROR so the model gets a retryable tool error. */
static void agent_dsml_parse(agent_dsml_parser *p) {
    static const char invoke_start[] = "<｜DSML｜invoke";
    static const char param_start[] = "<｜DSML｜parameter";

    while (p->state == AGENT_DSML_STRUCTURAL || p->state == AGENT_DSML_PARAM_VALUE) {
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            size_t end_tag_len = 0;
            char *end = agent_dsml_find_close_tag(p->raw + p->param_value_start,
                                                  "parameter", &end_tag_len);
            if (!end) return;
            agent_tool_call_add_arg(&p->current, p->param_name ? p->param_name : "",
                                    p->raw + p->param_value_start,
                                    (size_t)(end - (p->raw + p->param_value_start)),
                                    p->param_is_string);
            free(p->param_name);
            p->param_name = NULL;
            p->parse_pos = (size_t)(end - p->raw) + end_tag_len;
            p->state = AGENT_DSML_STRUCTURAL;
            continue;
        }

        while (p->parse_pos < p->raw_len &&
               (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
                p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
            p->parse_pos++;
        if (p->parse_pos >= p->raw_len) return;

        size_t close_len = 0;
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "tool_calls", &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            p->state = AGENT_DSML_DONE;
            return;
        }
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "invoke", &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            continue;
        }

        char *tag_end = strchr(p->raw + p->parse_pos, '>');
        if (!tag_end) return;
        size_t tag_len = (size_t)(tag_end - (p->raw + p->parse_pos)) + 1;
        char *tag = xstrndup(p->raw + p->parse_pos, tag_len);

        if (!strncmp(tag, invoke_start, strlen(invoke_start))) {
            agent_tool_call_free(&p->current);
            p->current.name = agent_parse_attr(tag, "name");
            if (!p->current.name) {
                free(tag);
                agent_dsml_set_error(p, "tool invoke without name");
                return;
            }
            p->parse_pos += tag_len;
        } else if (!strncmp(tag, param_start, strlen(param_start))) {
            free(p->param_name);
            p->param_name = agent_parse_attr(tag, "name");
            char *is_string = agent_parse_attr(tag, "string");
            p->param_is_string = is_string && !strcmp(is_string, "true");
            free(is_string);
            if (!p->param_name) {
                free(tag);
                agent_dsml_set_error(p, "tool parameter without name");
                return;
            }
            p->parse_pos += tag_len;
            p->param_value_start = p->parse_pos;
            p->state = AGENT_DSML_PARAM_VALUE;
        } else {
            snprintf(p->error, sizeof(p->error), "unexpected DSML tag: %.*s",
                     (int)(tag_len > 80 ? 80 : tag_len), tag);
            free(tag);
            p->state = AGENT_DSML_ERROR;
            return;
        }
        free(tag);
    }
}

void agent_dsml_start(agent_dsml_parser *p) {
    static const char start[] = "<｜DSML｜tool_calls>";
    p->state = AGENT_DSML_STRUCTURAL;
    p->search_len = 0;
    agent_dsml_raw_append(p, start, strlen(start));
    p->parse_pos = strlen(start);
}

void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n) {
    static const char start[] = "<｜DSML｜tool_calls>";
    const size_t start_len = sizeof(start) - 1;
    if (p->state == AGENT_DSML_DONE || p->state == AGENT_DSML_ERROR) return;

    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (p->state == AGENT_DSML_SEARCH) {
            if (p->search_len == sizeof(p->search_tail)) {
                memmove(p->search_tail, p->search_tail + 1, --p->search_len);
            }
            p->search_tail[p->search_len++] = c;
            if (p->search_len >= start_len &&
                memcmp(p->search_tail + p->search_len - start_len, start, start_len) == 0)
                agent_dsml_start(p);
            continue;
        }

        agent_dsml_raw_append(p, &c, 1);
        agent_dsml_parse(p);
    }
}
