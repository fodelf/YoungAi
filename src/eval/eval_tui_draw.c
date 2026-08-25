/* ds4-eval TUI drawing: the two-pane layout, streamed model output pane,
 * and status line (moved verbatim from ds4_eval.c). */

#include "eval_internal.h"

static void tui_draw_title(eval_ui *ui) {
    term_move(1, 1);
    tui_clear_left_line(ui, 1);
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));
    fputs("ds4-eval (" ANSI_BOLD "p" ANSI_RESET ")ause (" ANSI_BOLD "q" ANSI_RESET ")uit", stdout);
    printf(" %s", elapsed);
    if (ui->paused) {
        fputs(" " ANSI_RED ANSI_BOLD "PAUSED" ANSI_RESET, stdout);
    }
}

void tui_draw_frame(eval_ui *ui) {
    fputs("\x1b[2J", stdout);
    tui_draw_title(ui);
    term_move(1, ui->right_x);
    fputs(ANSI_BOLD "live sampled tokens" ANSI_RESET, stdout);

    for (int row = 1; row <= ui->rows; row++) {
        term_move(row, ui->left_w + 1);
        fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    }
}

static void tui_draw_left(eval_ui *ui) {
    int passed = 0;
    int failed = 0;
    for (int i = 0; i < ui->ncases; i++) {
        if (ui->status[i] == EVAL_PASSED) passed++;
        else if (ui->status[i] == EVAL_FAILED) failed++;
    }

    tui_draw_title(ui);

    term_move(2, 1);
    tui_clear_left_line(ui, 2);
    printf("score %s%d%s/%d  failed %s%d%s",
           ANSI_GREEN, passed, ANSI_RESET,
           ui->ncases,
           failed ? ANSI_RED : ANSI_DIM, failed, ANSI_RESET);

    const int first_row = 4;
    const int visible_rows = ui->rows >= first_row ? ui->rows - first_row + 1 : 0;
    const int shown = ui->ncases < visible_rows ? ui->ncases : visible_rows;
    int first = 0;
    if (shown > 0 && ui->ncases > shown) {
        /* The list follows the selection cursor, not the running test.  This
         * makes arrow navigation visible immediately.  When normal execution
         * advances, main() moves the selection to the new active case so the
         * viewport naturally follows the work again. */
        int anchor = ui->selected_case;
        if (anchor < 0) anchor = ui->active_case;
        if (anchor >= ui->ncases) anchor = ui->ncases - 1;
        first = anchor - shown / 2;
        if (first < 0) first = 0;
        if (first > ui->ncases - shown) first = ui->ncases - shown;
    }

    for (int row = 0; row < visible_rows; row++) {
        const int screen_row = first_row + row;
        const int i = first + row;
        term_move(screen_row, 1);
        tui_clear_left_line(ui, screen_row);
        if (i >= ui->ncases) continue;

        if (i == ui->active_case) fputs(ANSI_BOLD, stdout);
        term_set_color_for_status(ui->status[i]);
        printf("%c%2d ", ui->selection_active && i == ui->selected_case ? '>' : ' ', i + 1);
        printf("%-4s", status_name(ui->status[i]));
        fputs(ANSI_RESET, stdout);

        const int answer_w = 18;
        const int answer_col = ui->left_w - answer_w + 1;
        int title_w = answer_col - 11;
        if (title_w > 0) {
            fputc(' ', stdout);
            char title[512];
            snprintf(title, sizeof(title), "%s: %s",
                     eval_case_at(i)->source, eval_case_at(i)->title);
            print_trimmed(title, title_w);
        }
        if (ui->status[i] == EVAL_FAILED || ui->status[i] == EVAL_PASSED) {
            char answers[64];
            snprintf(answers, sizeof(answers), "%s/%s",
                     ui->guess[i][0] ? ui->guess[i] : "?",
                     eval_case_at(i)->answer);
            term_move(screen_row, answer_col);
            print_trimmed(answers, answer_w);
        }
        fputs(ANSI_RESET, stdout);
    }
}

void tui_reset_stream(eval_ui *ui, const eval_case *tc, bool in_think) {
    ui->stream.len = 0;
    if (ui->stream.v) ui->stream.v[0] = '\0';
    ui->styles.len = 0;
    ui->in_think = in_think;
    ui->pending_tag_len = 0;
    ui->generated = 0;
    ui->prefill_current = 0;
    ui->prefill_total = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;

    for (int row = 2; row <= ui->rows; row++) {
        term_move(row, ui->right_x);
        term_clear_to_eol();
    }
    tui_draw_question_preview(ui, tc);
}

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void stream_append_visible(eval_ui *ui, const char *p, size_t n) {
    buf_append(&ui->stream, p, n);
    style_append(&ui->styles, ui->in_think ? 1 : 0, n);
}

void stream_append_token_text(eval_ui *ui, const char *text, size_t len, bool finish) {
    const char *open = "<think>";
    const char *close = "</think>";
    size_t total = ui->pending_tag_len + len;
    char *tmp = malloc(total ? total : 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (ui->pending_tag_len) memcpy(tmp, ui->pending_tag, ui->pending_tag_len);
    if (len) memcpy(tmp + ui->pending_tag_len, text, len);
    ui->pending_tag_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = tmp + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, open)) {
            ui->in_think = true;
            i += strlen(open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, close)) {
            ui->in_think = false;
            stream_append_visible(ui, "\n", 1);
            i += strlen(close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, open) ||
             bytes_is_partial_prefix(cur, rem, close)))
        {
            if (rem < sizeof(ui->pending_tag)) {
                memcpy(ui->pending_tag, cur, rem);
                ui->pending_tag_len = rem;
            }
            break;
        }
        stream_append_visible(ui, cur, 1);
        i++;
    }
    free(tmp);
}

typedef struct {
    size_t start;
    size_t end;
} line_span;

void tui_draw_stream(eval_ui *ui) {
    if (!ui->enabled) return;
    const int width = tui_right_text_w(ui);

    line_span *lines = NULL;
    int line_len = 0;
    int line_cap = 0;
    size_t line_start = 0;
    int col = 0;

    for (size_t i = 0; i < ui->stream.len; i++) {
        char c = ui->stream.v[i];
        bool end_line = false;
        size_t end = i;
        if (c == '\n') {
            end_line = true;
            end = i;
        } else {
            col++;
            if (col >= width) {
                end_line = true;
                end = i + 1;
            }
        }
        if (end_line) {
            if (line_len == line_cap) {
                line_cap = line_cap ? line_cap * 2 : 64;
                line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
                if (!v) {
                    fprintf(stderr, "ds4-eval: out of memory\n");
                    exit(1);
                }
                lines = v;
            }
            lines[line_len++] = (line_span){line_start, end};
            line_start = i + 1;
            col = 0;
        }
    }
    if (line_start <= ui->stream.len) {
        if (line_len == line_cap) {
            line_cap = line_cap ? line_cap * 2 : 64;
            line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
            if (!v) {
                fprintf(stderr, "ds4-eval: out of memory\n");
                exit(1);
            }
            lines = v;
        }
        lines[line_len++] = (line_span){line_start, ui->stream.len};
    }

    int start = line_len > ui->body_h ? line_len - ui->body_h : 0;
    for (int row = 0; row < ui->body_h; row++) {
        term_move(ui->body_y + row, ui->right_x);
        term_clear_to_eol();
        int li = start + row;
        if (li >= line_len) continue;
        unsigned char cur_style = 255;
        int printed = 0;
        for (size_t i = lines[li].start; i < lines[li].end && printed < width; i++) {
            unsigned char st = ui->styles.v[i];
            if (st != cur_style) {
                fputs(st ? ANSI_DIM : ANSI_RESET, stdout);
                cur_style = st;
            }
            char c = ui->stream.v[i];
            if (c == '\r' || c == '\n') continue;
            if (c == '\t') c = ' ';
            fputc((unsigned char)c, stdout);
            printed++;
        }
        fputs(ANSI_RESET, stdout);
    }
    free(lines);
}

static void format_short_count(char *dst, size_t dstlen, int n) {
    if (n >= 1000000) {
        snprintf(dst, dstlen, "%dm", (n + 500000) / 1000000);
    } else if (n >= 10000) {
        snprintf(dst, dstlen, "%dk", (n + 500) / 1000);
    } else {
        snprintf(dst, dstlen, "%d", n);
    }
}

static const char *short_phase_name(const char *phase) {
    if (!strcmp(phase, "prefill")) return "FILL";
    if (!strcmp(phase, "thinking")) return "THINK";
    if (!strcmp(phase, "answer")) return "ANS";
    if (!strcmp(phase, "passed")) return "PASS";
    if (!strcmp(phase, "failed")) return "FAIL";
    return "RUN";
}

static void tui_draw_right_status(eval_ui *ui, const char *phase) {
    term_move(1, ui->right_x);
    term_clear_to_eol();
    const eval_case *tc = eval_case_at(ui->active_case);
    char id[13];
    char gen[16], max[16], cur[16], total[16];
    char line[512];
    snprintf(id, sizeof(id), "%.12s", tc->id);
    if (ui->prefill_total > 0 && ui->status[ui->active_case] == EVAL_PREFILL) {
        double pct = 100.0 * (double)ui->prefill_current / (double)ui->prefill_total;
        format_short_count(cur, sizeof(cur), ui->prefill_current);
        format_short_count(total, sizeof(total), ui->prefill_total);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s %.0f%%  %s/%s",
                 ui->speed_tps, short_phase_name(phase), cur, total, pct,
                 tc->source, id);
    } else {
        int phase_max = !strcmp(phase, "thinking") ? ui->think_max_tokens : ui->max_tokens;
        int phase_gen = ui->generated;
        if (phase_max < 0) phase_max = 0;
        if (!strcmp(phase, "thinking") && phase_gen > phase_max) phase_gen = phase_max;
        format_short_count(gen, sizeof(gen), phase_gen);
        format_short_count(max, sizeof(max), phase_max);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s  %s/%s",
                 ui->speed_tps, short_phase_name(phase), gen, max,
                 tc->source, id);
    }
    print_trimmed(line, tui_right_text_w(ui));
}

void tui_refresh(eval_ui *ui, const char *phase) {
    if (!ui->enabled) return;
    tui_draw_left(ui);
    tui_draw_right_status(ui, phase);
    tui_draw_stream(ui);
    fflush(stdout);
}
