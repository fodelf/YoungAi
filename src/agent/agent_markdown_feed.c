/* agent_markdown_feed.c — 机械拆分自 ds4_agent.c: Assistant Markdown Rendering (3/3: markdown 流入口)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* Tiny streaming Markdown highlighter for assistant prose.  It deliberately
 * recognizes only delimiters that the model commonly emits in short answers:
 * **bold**, *italic*, `inline code`, ``inline code`` and fenced code blocks.
 * The state machine holds only possible delimiter bytes; once a byte is known
 * to be ordinary text it is sent to the raw UTF-8 writer above.  Tool
 * visualization and redirected output bypass this layer. */
static void renderer_markdown_clear_pending(agent_token_renderer *r) {
    r->md_pending = AGENT_MD_PENDING_NONE;
    r->md_pending_len = 0;
}

static void renderer_markdown_emit_pending_literals(agent_token_renderer *r) {
    char c;
    if (r->md_pending == AGENT_MD_PENDING_STAR) {
        c = '*';
    } else if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        c = '`';
    } else {
        return;
    }
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (r->md_code_block) {
        if (c == '`') renderer_code_emit_backtick_literals(r, count);
        else for (size_t i = 0; i < count; i++) renderer_code_byte(r, c);
        return;
    }
    for (size_t i = 0; i < count; i++) renderer_write_char_raw(r, c);
}

static void renderer_markdown_commit_backticks(agent_token_renderer *r) {
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (count >= 3) {
        for (size_t i = 0; i < count; i++) renderer_write_plain_byte(r, '`');
        if (r->md_code_block) renderer_code_end(r);
        else renderer_code_begin(r);
        return;
    }
    if (r->md_code_block) {
        renderer_code_emit_backtick_literals(r, count);
        return;
    }
    /* Support both `code` and ``code``.  The latter is uncommon in model
     * replies, but accepting it costs nothing and avoids leaking delimiters. */
    r->md_inline_code = !r->md_inline_code;
}

static bool renderer_space_byte(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* Consume one byte of markdown-aware assistant output.  Backticks and stars
 * are held in r->pending until the parser knows whether they form a marker;
 * all ordinary text is emitted with the current terminal attributes. */
static void renderer_markdown_feed(agent_token_renderer *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            if (r->md_code_block) {
                r->md_fence_lang[r->md_fence_lang_len] = '\0';
                r->md_syntax = agent_syntax_for_lang(r->md_fence_lang);
            }
            renderer_write_plain_byte(r, '\n');
            r->md_fence_info = false;
        } else if (r->md_code_block) {
            unsigned char uc = (unsigned char)c;
            if (r->md_fence_lang_len + 1 < sizeof(r->md_fence_lang) &&
                (isalnum(uc) || c == '_' || c == '-' || c == '+' || c == '#'))
            {
                r->md_fence_lang[r->md_fence_lang_len++] = c;
            }
            renderer_write_plain_byte(r, c);
        }
        return;
    }

    if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        if (c == '`') {
            r->md_pending_len++;
            return;
        }
        renderer_markdown_commit_backticks(r);
        renderer_markdown_feed(r, c);
        return;
    }

    if (r->md_pending == AGENT_MD_PENDING_STAR) {
        renderer_markdown_clear_pending(r);
        if (!r->md_inline_code && !r->md_code_block && c == '*') {
            r->md_bold = !r->md_bold;
            return;
        }
        if (!r->md_inline_code && !r->md_code_block &&
            (r->md_italic || !renderer_space_byte(c)))
        {
            r->md_italic = !r->md_italic;
            renderer_markdown_feed(r, c);
            return;
        }
        renderer_write_char_raw(r, '*');
        renderer_markdown_feed(r, c);
        return;
    }

    if (c == '`' && (!r->md_code_block || r->md_code_line_start)) {
        r->md_pending = AGENT_MD_PENDING_BACKTICK;
        r->md_pending_len = 1;
        return;
    }
    if (r->md_code_block) {
        renderer_code_byte(r, c);
        return;
    }
    if (!r->md_inline_code && !r->md_code_block && c == '*') {
        r->md_pending = AGENT_MD_PENDING_STAR;
        r->md_pending_len = 1;
        return;
    }
    renderer_write_char_raw(r, c);
}

static void renderer_markdown_finish(agent_token_renderer *r) {
    /* A closing code fence can be the final bytes of the assistant reply.  In
     * that case no following character arrives to force the pending backticks
     * through the normal streaming path, so commit a full fence here instead of
     * leaking the literal ``` marker to the terminal. */
    if (r->md_pending == AGENT_MD_PENDING_BACKTICK && r->md_pending_len >= 3)
        renderer_markdown_commit_backticks(r);
    else
        renderer_markdown_emit_pending_literals(r);
    if (r->md_code_block && r->md_code_line_len)
        renderer_code_emit_buffered_line(r, false);
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    r->md_fence_info = false;
    r->md_code_line_start = false;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    free(r->md_code_line);
    r->md_code_line = NULL;
    r->md_code_line_len = 0;
    r->md_code_line_cap = 0;
}

void renderer_write_char(agent_token_renderer *r, char c) {
    if (!r->format_markdown || r->in_think) {
        renderer_markdown_emit_pending_literals(r);
        renderer_write_char_raw(r, c);
        return;
    }
    renderer_markdown_feed(r, c);
}

/* Render assistant text while hiding <think> tags and dimming thinking text.
 * The function is also responsible for not prematurely emitting a partial
 * control tag split across model tokens. */
static void renderer_process(agent_token_renderer *r, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = r->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (r->pending_len) memcpy(buf, r->pending, r->pending_len);
    if (len) memcpy(buf + r->pending_len, text, len);
    r->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            r->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            r->in_think = false;
            renderer_reset_color(r);
            if (!r->last_output_newline) renderer_write(r, "\n", 1);
            renderer_write(r, "\n", 1);
            r->last_output_newline = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(r->pending)) {
                memcpy(r->pending, cur, rem);
                r->pending_len = rem;
            }
            break;
        }
        renderer_write_char(r, cur[0]);
        i++;
    }
    free(buf);
}

void renderer_finish(agent_token_renderer *r) {
    if (r->format_thinking) {
        renderer_process(r, NULL, 0, true);
    }
    renderer_markdown_finish(r);
    renderer_flush_utf8(r);
    renderer_reset_color(r);
    if (r->wrote_visible_output) {
        if (!r->last_output_newline) renderer_write(r, "\n", 1);
        renderer_write(r, "\n", 1);
        r->last_output_newline = true;
    }
}

void renderer_color(agent_token_renderer *r, const char *seq) {
    renderer_markdown_emit_pending_literals(r);
    renderer_flush_utf8(r);
    bool reset = !seq || !seq[0] || !strcmp(seq, "\x1b[0m");
    if (r->use_color && seq && seq[0]) renderer_write(r, seq, strlen(seq));
    r->color_open = r->use_color && !reset;
}

void renderer_plain(agent_token_renderer *r, const char *s, size_t n) {
    renderer_markdown_emit_pending_literals(r);
    renderer_flush_utf8(r);
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    if (n) r->last_output_newline = s[n - 1] == '\n';
}
