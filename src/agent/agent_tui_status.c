/* agent_tui_status.c — 机械拆分自 ds4_agent.c: Terminal Prompt, Status Footer, And Async Output Rendering (1/4: 状态行/footer/队列)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Terminal Prompt, Status Footer, And Async Output Rendering
 * ============================================================================
 */

#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_PROGRESS_BAR_WIDTH 32
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

static void agent_progress_append(char *buf, size_t len, size_t *pos,
                                  const char *s) {
    if (len == 0 || *pos >= len - 1) return;
    size_t avail = len - *pos;
    int n = snprintf(buf + *pos, avail, "%s", s);
    if (n <= 0) return;
    if ((size_t)n >= avail) *pos = len - 1;
    else *pos += (size_t)n;
}

void build_prompt_text(const agent_status *st, char *buf, size_t len) {
    (void)st;
    snprintf(buf, len, "ds4-agent> ");
}

static void agent_progress_bar(int done, int total, char *buf, size_t len,
                               bool color) {
    if (len == 0) return;
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * AGENT_PROGRESS_BAR_WIDTH) / total);
    if (filled < 0) filled = 0;
    if (filled > AGENT_PROGRESS_BAR_WIDTH) filled = AGENT_PROGRESS_BAR_WIDTH;
    if (color && filled == 0 && done < total) filled = 1;
    size_t pos = 0;
    agent_progress_append(buf, len, &pos, "[");
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_BAR_FILL);
    for (int i = 0; i < AGENT_PROGRESS_BAR_WIDTH && pos + 1 < len; i++) {
        if (color && i == filled) {
            agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
        }
        agent_progress_append(buf, len, &pos, i < filled ? "▶" : "·");
    }
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
    agent_progress_append(buf, len, &pos, "]");
    buf[pos < len ? pos : len - 1] = '\0';
}

static void agent_power_status_suffix(const agent_status *st,
                                      char *buf, size_t len) {
    if (len == 0) return;
    if (st->power_percent > 0 && st->power_percent < 100)
        snprintf(buf, len, " | ⚡ %d%%", st->power_percent);
    else
        buf[0] = '\0';
}

unsigned agent_next_prefill_label(void) {
    static unsigned next;
    return next++;
}

/* Keep each prefill operation on a single playful label so the footer does not
 * visually churn while progress updates stream in. */
static const char *agent_prefill_label(const agent_status *st) {
    static const char *labels[] = {
        "reading",
        "absorbing",
        "studying",
        "gathering",
        "crunching",
        "scrutinizing",
    };
    size_t n = sizeof(labels) / sizeof(labels[0]);
    return labels[(st ? st->prefill_label : 0u) % n];
}

/* Build the one-line footer shown below the prompt.  It is intentionally compact
 * because linenoise redraws it on every progress update. */
static void build_status_text(const agent_status *st, char *buf, size_t len) {
    char used[32], total_ctx[32];
    char power[32];
    agent_format_ctx_size(st->ctx_used, used, sizeof(used));
    agent_format_ctx_size(st->ctx_size, total_ctx, sizeof(total_ctx));
    agent_power_status_suffix(st, power, sizeof(power));

    switch (st->state) {
    case AGENT_WORKER_PREFILL: {
        int done = st->prefill_done;
        int total = st->prefill_total > 0 ? st->prefill_total : 1;
        if (done > total) done = total;
        double pct = 100.0 * (double)done / (double)total;
        char bar[AGENT_PROGRESS_BAR_MAX_BYTES];
        agent_progress_bar(done, total, bar, sizeof(bar), stdout_is_tty());
        snprintf(buf, len, "ctx %s/%s | %s %s %d/%d %.1f%%%s",
                 used, total_ctx, agent_prefill_label(st), bar,
                 done, total, pct, power);
        break;
    }
    case AGENT_WORKER_GENERATING:
        snprintf(buf, len, "ctx %s/%s | generation %d tokens %.1f t/s%s",
                 used, total_ctx, st->generated, st->gen_tps, power);
        break;
    case AGENT_WORKER_COMPACTING:
        snprintf(buf, len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s%s",
                 used, total_ctx, st->generated, st->gen_tps, power);
        break;
    case AGENT_WORKER_DRAINING:
        snprintf(buf, len, "ctx %s/%s | stopping after distributed cluster drains%s",
                 used, total_ctx, power);
        break;
    case AGENT_WORKER_SAVING:
        snprintf(buf, len, "ctx %s/%s | saving session%s", used, total_ctx, power);
        break;
    case AGENT_WORKER_ERROR:
        snprintf(buf, len, "ctx %s/%s | error: %s%s", used, total_ctx,
                 st->error[0] ? st->error : "unknown error", power);
        break;
    case AGENT_WORKER_STOPPED:
        snprintf(buf, len, "ctx %s/%s | interrupted%s", used, total_ctx, power);
        break;
    default:
        snprintf(buf, len, "ctx %s/%s | idle%s", used, total_ctx, power);
        break;
    }
}


void agent_prompt_queue_push(agent_prompt_queue *q, const char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    q->v[q->len++] = xstrdup(text ? text : "");
}

char *agent_prompt_queue_pop(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    char *text = q->v[0];
    memmove(q->v, q->v + 1, (q->len - 1) * sizeof(q->v[0]));
    q->len--;
    return text;
}

void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    memmove(q->v + 1, q->v, q->len * sizeof(q->v[0]));
    q->v[0] = text;
    q->len++;
}

char *agent_prompt_queue_take_all(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    if (q->len == 1) return agent_prompt_queue_pop(q);

    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char hdr[64];
        if (i) agent_buf_puts(&b, "\n\n");
        snprintf(hdr, sizeof(hdr), "Queued user message %zu:\n", i + 1);
        agent_buf_puts(&b, hdr);
        agent_buf_puts(&b, q->v[i]);
        free(q->v[i]);
    }
    q->len = 0;
    return agent_buf_take(&b);
}

char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char *echo = agent_format_user_prompt_echo(q->v[i]);
        agent_buf_puts(&b, echo);
        free(echo);
    }
    return agent_buf_take(&b);
}

static const char *agent_prompt_queue_peek(const agent_prompt_queue *q) {
    return q->len ? q->v[0] : NULL;
}

void agent_prompt_queue_free(agent_prompt_queue *q) {
    for (size_t i = 0; i < q->len; i++) free(q->v[i]);
    free(q->v);
    memset(q, 0, sizeof(*q));
}

bool agent_footer_is_multiline(const char *status) {
    return status && strchr(status, '\n');
}

/* Build the editable footer.  With queued prompts, the footer becomes multiple
 * rows: a compact queue preview first, then the normal status row. */
void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len) {
    char status[512];
    build_status_text(st, status, sizeof(status));
    if (!queue || !queue->len) {
        snprintf(buf, len, "%s", status);
        return;
    }

    const char *queued = agent_prompt_queue_peek(queue);
    if (cols < 40) cols = 40;
    int max_rows = 3;
    size_t budget = (size_t)cols * (size_t)max_rows;
    const char *plain_suffix = " (ctrl+x to edit, ESC to send ASAP)";
    size_t queued_len = strlen(queued);
    char more_suffix[160];
    const char *suffix = plain_suffix;
    size_t take = queued_len;
    if (queued_len + strlen(plain_suffix) > budget) {
        size_t reserve = 72;
        take = budget > reserve ? budget - reserve : budget / 2;
        snprintf(more_suffix, sizeof(more_suffix),
                 "... %zu characters more ..., (ctrl+x to edit, ESC to send ASAP)",
                 queued_len - take);
        suffix = more_suffix;
    }

    agent_buf msg = {0};
    agent_buf_puts(&msg, "queued: ");
    for (size_t i = 0; i < take; i++) {
        char c = queued[i];
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        agent_buf_append(&msg, &c, 1);
    }
    agent_buf_puts(&msg, suffix);
    char *preview = agent_buf_take(&msg);

    agent_buf out = {0};
    size_t pos = 0, preview_len = strlen(preview);
    for (int row = 0; row < max_rows && pos < preview_len; row++) {
        if (row) agent_buf_puts(&out, "\n");
        if (stdout_is_tty()) agent_buf_puts(&out, AGENT_QUEUE_STYLE);
        size_t part = preview_len - pos;
        if (part > (size_t)cols) part = (size_t)cols;
        agent_buf_append(&out, preview + pos, part);
        if (stdout_is_tty()) agent_buf_puts(&out, "\x1b[0m");
        pos += part;
    }
    agent_buf_puts(&out, "\n");
    if (stdout_is_tty()) agent_buf_puts(&out, AGENT_STATUS_STYLE_START);
    agent_buf_puts(&out, status);
    snprintf(buf, len, "%s", out.ptr ? out.ptr : "");
    free(preview);
    free(out.ptr);
}
