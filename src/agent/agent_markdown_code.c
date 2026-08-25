/* agent_markdown_code.c — 机械拆分自 ds4_agent.c: Assistant Markdown Rendering (2/3: 语法匹配 + 代码块渲染)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

static bool agent_syntax_alias_match(const char *aliases, const char *lang) {
    if (!aliases || !lang || !lang[0]) return false;
    size_t llen = strlen(lang);
    const char *p = aliases;
    while (*p) {
        while (*p == ' ') p++;
        const char *start = p;
        while (*p && *p != ' ') p++;
        if ((size_t)(p - start) == llen && !strncasecmp(start, lang, llen))
            return true;
    }
    return false;
}

const agent_syntax *agent_syntax_for_lang(const char *lang) {
    if (lang && lang[0]) {
        for (const agent_syntax *s = agent_syntaxes; s->name; s++) {
            if (!strcasecmp(s->name, lang) ||
                agent_syntax_alias_match(s->aliases, lang))
                return s;
        }
    }
    return &agent_syntaxes[0];
}

const agent_syntax *agent_syntax_for_path(const char *path) {
    if (!path || !path[0]) return agent_syntax_for_lang(NULL);
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!strcasecmp(base, "Dockerfile")) return agent_syntax_for_lang("sh");
    if (!strcasecmp(base, "Makefile") || !strcasecmp(base, "makefile"))
        return agent_syntax_for_lang("sh");
    const char *dot = strrchr(base, '.');
    if (!dot || !dot[1]) return agent_syntax_for_lang(NULL);
    return agent_syntax_for_lang(dot + 1);
}

static bool agent_syntax_separator(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || isspace(uc) || strchr(",.()+-/*=~%[]{}<>:;!&|^?", c) != NULL;
}

static const char *agent_syntax_line_comment(const agent_syntax *syn,
                                             const char *p) {
    if (!syn) return NULL;
    for (int i = 0; i < 3 && syn->singleline_comments[i]; i++) {
        const char *m = syn->singleline_comments[i];
        size_t mlen = strlen(m);
        if (mlen && !strncmp(p, m, mlen)) return m;
    }
    return NULL;
}

static int agent_syntax_color(int hl) {
    switch (hl) {
    case AGENT_HL_COMMENT: return 244;
    case AGENT_HL_KEYWORD1: return 214;
    case AGENT_HL_KEYWORD2: return 81;
    case AGENT_HL_STRING: return 150;
    case AGENT_HL_NUMBER: return 203;
    default: return 252;
    }
}

static void renderer_syntax_write(agent_token_renderer *r, int hl,
                                  const char *s, size_t n) {
    if (!n) return;
    if (hl != AGENT_HL_NORMAL) r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color && hl != AGENT_HL_NORMAL) {
        char seq[32];
        snprintf(seq, sizeof(seq), "\x1b[38;5;%dm", agent_syntax_color(hl));
        renderer_write(r, seq, strlen(seq));
    }
    renderer_write(r, s, n);
    if (r->use_color && hl != AGENT_HL_NORMAL) renderer_write(r, "\x1b[0m", 4);
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}

static void renderer_syntax_write_upto_marker(agent_token_renderer *r) {
    static const char marker[] = "[upto]";
    r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color) {
        renderer_write(r, "\x1b[38;5;244m[", strlen("\x1b[38;5;244m["));
        renderer_write(r, "\x1b[1;38;5;177mupto",
                       strlen("\x1b[1;38;5;177mupto"));
        renderer_write(r, "\x1b[38;5;244m]\x1b[0m",
                       strlen("\x1b[38;5;244m]\x1b[0m"));
    } else {
        renderer_write(r, marker, sizeof(marker) - 1);
    }
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}

static size_t agent_syntax_keyword_len(const char *kw, bool *secondary) {
    size_t len = strlen(kw);
    *secondary = len && kw[len - 1] == '|';
    return *secondary ? len - 1 : len;
}

static bool agent_syntax_match_keyword(const agent_syntax *syn,
                                       const char *p,
                                       const char *line_end,
                                       size_t *out_len,
                                       int *out_hl) {
    if (!syn || !syn->keywords) return false;
    for (int i = 0; syn->keywords[i]; i++) {
        bool secondary = false;
        size_t klen = agent_syntax_keyword_len(syn->keywords[i], &secondary);
        if ((size_t)(line_end - p) < klen) continue;
        bool match = (syn->flags & AGENT_SYNTAX_CASE_INSENSITIVE) ?
            !strncasecmp(p, syn->keywords[i], klen) :
            !strncmp(p, syn->keywords[i], klen);
        if (!match) continue;
        if (!agent_syntax_separator(p[klen])) continue;
        *out_len = klen;
        *out_hl = secondary ? AGENT_HL_KEYWORD2 : AGENT_HL_KEYWORD1;
        return true;
    }
    return false;
}

static bool agent_syntax_number_start(const char *p, const char *line,
                                      bool prev_sep, int prev_hl) {
    unsigned char c = (unsigned char)*p;
    if (isdigit(c) && (prev_sep || prev_hl == AGENT_HL_NUMBER)) return true;
    if (*p == '.' && p > line && prev_hl == AGENT_HL_NUMBER) return true;
    return false;
}

static size_t agent_syntax_number_len(const char *p, const char *line_end) {
    const char *q = p;
    while (q < line_end) {
        unsigned char c = (unsigned char)*q;
        if (isalnum(c) || *q == '_' || *q == '.' || *q == '+' || *q == '-') q++;
        else break;
    }
    return (size_t)(q - p);
}

static void renderer_syntax_emit_line(agent_token_renderer *r,
                                      const char *line, size_t len) {
    const agent_syntax *syn = r->md_syntax ? r->md_syntax : agent_syntax_for_lang(NULL);
    const char *p = line;
    const char *end = line + len;
    bool prev_sep = true;
    int prev_hl = AGENT_HL_NORMAL;
    int in_string = 0;

    while (p < end) {
        if (r->md_code_highlight_upto &&
            (size_t)(end - p) >= strlen("[upto]") &&
            !strncmp(p, "[upto]", strlen("[upto]")))
        {
            renderer_syntax_write_upto_marker(r);
            p += strlen("[upto]");
            prev_sep = true;
            prev_hl = AGENT_HL_NORMAL;
            continue;
        }

        if (r->md_code_in_ml_comment) {
            const char *mce = syn->multiline_end;
            if (mce && *mce) {
                size_t mlen = strlen(mce);
                const char *q = p;
                while (q < end && ((size_t)(end - q) < mlen ||
                       strncmp(q, mce, mlen))) q++;
                if (q < end) {
                    q += mlen;
                    renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
                    p = q;
                    r->md_code_in_ml_comment = false;
                    prev_sep = true;
                    prev_hl = AGENT_HL_COMMENT;
                    continue;
                }
            }
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        const char *scs = agent_syntax_line_comment(syn, p);
        if (!in_string && scs) {
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        if (!in_string && syn->multiline_start && syn->multiline_end &&
            !strncmp(p, syn->multiline_start, strlen(syn->multiline_start))) {
            size_t mlen = strlen(syn->multiline_start);
            const char *q = p + mlen;
            size_t elen = strlen(syn->multiline_end);
            while (q < end && ((size_t)(end - q) < elen ||
                   strncmp(q, syn->multiline_end, elen))) q++;
            if (q < end) q += elen;
            else r->md_code_in_ml_comment = true;
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_COMMENT;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) && in_string) {
            const char *q = p;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == in_string) {
                    in_string = 0;
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) &&
            (*p == '"' || *p == '\'' ||
             ((syn->flags & AGENT_SYNTAX_BACKTICK_STRINGS) && *p == '`'))) {
            int quote = *p;
            const char *q = p + 1;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == quote) {
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_NUMBERS) &&
            agent_syntax_number_start(p, line, prev_sep, prev_hl)) {
            size_t nlen = agent_syntax_number_len(p, end);
            renderer_syntax_write(r, AGENT_HL_NUMBER, p, nlen);
            p += nlen;
            prev_sep = false;
            prev_hl = AGENT_HL_NUMBER;
            continue;
        }

        if (prev_sep) {
            size_t klen = 0;
            int khl = AGENT_HL_NORMAL;
            if (agent_syntax_match_keyword(syn, p, end, &klen, &khl)) {
                renderer_syntax_write(r, khl, p, klen);
                p += klen;
                prev_sep = false;
                prev_hl = khl;
                continue;
            }
        }

        renderer_syntax_write(r, AGENT_HL_NORMAL, p, 1);
        prev_sep = agent_syntax_separator(*p);
        prev_hl = AGENT_HL_NORMAL;
        p++;
    }
}

static void renderer_code_line_append(agent_token_renderer *r,
                                      const char *s, size_t n) {
    if (!n) return;
    if (r->md_code_line_len + n + 1 > r->md_code_line_cap) {
        size_t cap = r->md_code_line_cap ? r->md_code_line_cap * 2 : 256;
        while (cap < r->md_code_line_len + n + 1) cap *= 2;
        r->md_code_line = xrealloc(r->md_code_line, cap);
        r->md_code_line_cap = cap;
    }
    memcpy(r->md_code_line + r->md_code_line_len, s, n);
    r->md_code_line_len += n;
    r->md_code_line[r->md_code_line_len] = '\0';
}

int renderer_terminal_cols(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}

static bool renderer_code_line_can_repaint(agent_token_renderer *r) {
    if (!r->use_color || r->capture || r->md_code_line_len == 0) return false;
    int cols = renderer_terminal_cols();
    size_t prefix_len = r->md_code_line_prefix ?
                        strlen(r->md_code_line_prefix) : 0;
    if (cols <= 1 || prefix_len + r->md_code_line_len >= (size_t)cols)
        return false;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        unsigned char c = (unsigned char)r->md_code_line[i];
        if (c == '\t' || c == 0x1b || c >= 0x80 || (c < 0x20 && c != '\r'))
            return false;
    }
    return true;
}

static void renderer_code_write_line_prefix(agent_token_renderer *r) {
    if (!r->md_code_line_prefix) return;
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, r->md_code_line_prefix_color,
                       strlen(r->md_code_line_prefix_color));
    renderer_write(r, r->md_code_line_prefix,
                   strlen(r->md_code_line_prefix));
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}

/* Run the syntax highlighter in silent mode to learn whether the already
 * streamed line would change if repainted, while preserving the multiline
 * comment state until the caller decides whether repaint is safe. */
static bool renderer_code_scan_line(agent_token_renderer *r,
                                    bool *final_ml_comment) {
    bool old_silent = r->md_syntax_silent;
    bool old_highlight = r->md_syntax_has_highlight;
    bool old_ml_comment = r->md_code_in_ml_comment;

    r->md_syntax_silent = true;
    r->md_syntax_has_highlight = false;
    renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    bool changed = r->md_syntax_has_highlight;
    *final_ml_comment = r->md_code_in_ml_comment;

    r->md_code_in_ml_comment = old_ml_comment;
    r->md_syntax_silent = old_silent;
    r->md_syntax_has_highlight = old_highlight;
    return changed;
}

/* Code is shown as soon as bytes arrive.  At end-of-line we can cheaply
 * replace only that terminal row with syntax-highlighted text, but only for
 * simple one-row ASCII lines; long, tabbed, escaped, or UTF-8 lines are left
 * as streamed and only advance the highlighter state. */
void renderer_code_emit_buffered_line(agent_token_renderer *r,
                                             bool with_newline) {
    bool final_ml_comment = r->md_code_in_ml_comment;
    bool changed = renderer_code_scan_line(r, &final_ml_comment);
    bool repaint = changed && renderer_code_line_can_repaint(r);
    if (repaint) {
        renderer_reset_color(r);
        renderer_write(r, "\r\x1b[0K", 5);
        renderer_code_write_line_prefix(r);
        renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    } else {
        r->md_code_in_ml_comment = final_ml_comment;
    }
    r->md_code_line_len = 0;
    if (with_newline) {
        renderer_write_plain_byte(r, '\n');
        r->wrote_visible_output = true;
        r->last_output_newline = true;
        r->md_code_line_start = true;
    }
}

void renderer_code_byte(agent_token_renderer *r, char c) {
    if (c == '\n') {
        renderer_code_emit_buffered_line(r, true);
        return;
    }
    renderer_code_line_append(r, &c, 1);
    renderer_write_plain_byte(r, c);
    if (c != ' ' && c != '\t' && c != '\r') r->md_code_line_start = false;
}

void renderer_code_emit_backtick_literals(agent_token_renderer *r,
                                                 size_t count) {
    for (size_t i = 0; i < count; i++) renderer_code_byte(r, '`');
}

void renderer_code_begin(agent_token_renderer *r) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = true;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    r->md_code_line_len = 0;
}

void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = syntax ? syntax : agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    r->md_code_line_len = 0;
}

void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color) {
    r->md_code_line_prefix = prefix;
    r->md_code_line_prefix_color = color;
}

void renderer_code_stream_set_upto_marker(agent_token_renderer *r,
                                                 bool enabled) {
    r->md_code_highlight_upto = enabled;
}

void renderer_code_end(agent_token_renderer *r) {
    bool only_space = true;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        if (r->md_code_line[i] != ' ' && r->md_code_line[i] != '\t' &&
            r->md_code_line[i] != '\r') {
            only_space = false;
            break;
        }
    }
    if (r->md_code_line_len && !only_space)
        renderer_code_emit_buffered_line(r, false);
    else
        r->md_code_line_len = 0;
    r->md_code_block = false;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
}
