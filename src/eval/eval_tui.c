/* ds4-eval TUI core: terminal state, raw-mode keyboard input thread, run
 * clock, and lifecycle (moved verbatim from ds4_eval.c).  Drawing lives in
 * eval_tui_draw.c. */

#include "eval_internal.h"

static eval_ui *global_ui;


void tui_run_clock_tick(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    double now = run_clock_sec();
    double delta = now - ui->run_last_sec;
    if (delta > 0.0) ui->run_elapsed_sec += delta;
    ui->run_last_sec = now;
}

void tui_run_clock_start(eval_ui *ui) {
    if (!ui || ui->run_clock_active) return;
    ui->run_last_sec = run_clock_sec();
    ui->run_clock_active = true;
}

void tui_run_clock_stop(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    tui_run_clock_tick(ui);
    ui->run_clock_active = false;
}

double tui_run_clock_visible_sec(const eval_ui *ui) {
    if (!ui) return 0.0;
    double elapsed = ui->run_elapsed_sec;
    if (ui->run_clock_active) {
        double delta = run_clock_sec() - ui->run_last_sec;
        if (delta > 0.0) elapsed += delta;
    }
    return elapsed;
}

void format_run_elapsed(char *dst, size_t dstlen, double sec) {
    if (sec < 0.0) sec = 0.0;
    unsigned long long minutes = (unsigned long long)(sec / 60.0);
    unsigned long long hours = minutes / 60ull;
    minutes %= 60ull;
    snprintf(dst, dstlen, "%02lluh:%02llum", hours, minutes);
}

static eval_input global_input = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
};

static int terminal_size(int *cols, int *rows) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0 || ws.ws_row == 0) {
        *cols = 80;
        *rows = 24;
        return -1;
    }
    *cols = ws.ws_col;
    *rows = ws.ws_row;
    return 0;
}

void term_move(int row, int col) {
    printf("\x1b[%d;%dH", row, col);
}

void term_clear_to_eol(void) {
    fputs("\x1b[K", stdout);
}

int tui_right_text_w(const eval_ui *ui) {
    /* Avoid writing the physical last column.  Many terminals defer wrapping
     * until the next byte after the last column, which would spill right-pane
     * status text into the next row's left pane. */
    return ui->right_w > 1 ? ui->right_w - 1 : ui->right_w;
}

void tui_clear_left_line(eval_ui *ui, int row) {
    term_move(row, 1);
    for (int i = 0; i < ui->left_w; i++) fputc(' ', stdout);
    term_move(row, ui->left_w + 1);
    fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    term_move(row, 1);
}

void term_set_color_for_status(eval_status st) {
    switch (st) {
    case EVAL_PENDING:  fputs(ANSI_DIM, stdout); break;
    case EVAL_PREFILL:  fputs(ANSI_CYAN, stdout); break;
    case EVAL_THINKING: fputs(ANSI_YELLOW, stdout); break;
    case EVAL_SKIPPED:  fputs(ANSI_DIM, stdout); break;
    case EVAL_STOPPED:  fputs(ANSI_YELLOW ANSI_BOLD, stdout); break;
    case EVAL_PASSED:   fputs(ANSI_GREEN ANSI_BOLD, stdout); break;
    case EVAL_FAILED:   fputs(ANSI_RED ANSI_BOLD, stdout); break;
    }
}

const char *status_name(eval_status st) {
    switch (st) {
    case EVAL_PENDING: return "PEND";
    case EVAL_PREFILL: return "FILL";
    case EVAL_THINKING: return "RUN";
    case EVAL_SKIPPED: return "SKIP";
    case EVAL_STOPPED: return "STOP";
    case EVAL_PASSED: return "PASS";
    case EVAL_FAILED: return "FAIL";
    }
    return "?";
}

static bool eval_status_running(eval_status st) {
    return st == EVAL_PREFILL || st == EVAL_THINKING;
}

void print_trimmed(const char *s, int width) {
    if (width <= 0) return;
    int len = (int)strlen(s);
    if (len <= width) {
        fputs(s, stdout);
        return;
    }
    if (width <= 3) {
        for (int i = 0; i < width; i++) fputc('.', stdout);
        return;
    }
    fwrite(s, 1, (size_t)width - 3, stdout);
    fputs("...", stdout);
}

void tui_draw_question_preview(eval_ui *ui, const eval_case *tc) {
    const char *q = tc->question;
    size_t pos = 0;
    const int width = tui_right_text_w(ui);

    for (int row = 0; row < 3; row++) {
        term_move(2 + row, ui->right_x);
        term_clear_to_eol();
        fputs(ANSI_BLUE, stdout);

        int printed = 0;
        bool last_space = true;
        while (q[pos] && printed < width) {
            unsigned char c = (unsigned char)q[pos++];
            if (isspace(c)) {
                if (last_space) continue;
                c = ' ';
                last_space = true;
            } else {
                last_space = false;
            }
            fputc(c, stdout);
            printed++;
        }

        if (row == 2) {
            while (q[pos] && isspace((unsigned char)q[pos])) pos++;
            if (q[pos] && width >= 3) {
                int ell_col = ui->right_x + (printed >= 3 ? printed - 3 : 0);
                term_move(2 + row, ell_col);
                fputs("...", stdout);
            }
        }
        fputs(ANSI_RESET, stdout);
    }

    term_move(5, ui->right_x);
    term_clear_to_eol();
}

static void input_queue_key(int key) {
    pthread_mutex_lock(&global_input.mu);
    if (key == 'A') {
        global_input.move_delta--;
    } else if (key == 'B') {
        global_input.move_delta++;
    } else if (key == '\r' || key == '\n') {
        global_input.enter_pressed = true;
    } else if (key == 'p' || key == 'P') {
        global_input.pause_pressed = true;
    } else if (key == 'q' || key == 'Q') {
        global_input.quit_pressed = true;
    }
    pthread_mutex_unlock(&global_input.mu);
}

static void *input_thread_main(void *arg) {
    (void)arg;
    int esc_state = 0;
    char buf[64];

    while (global_input.running) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) {
            usleep(5000);
            continue;
        }
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)buf[i];
            if (esc_state == 0) {
                if (c == 27) esc_state = 1;
                else if (c == 'p' || c == 'P' || c == 'q' || c == 'Q') input_queue_key(c);
                else if (c == '\r' || c == '\n') input_queue_key(c);
            } else if (esc_state == 1) {
                esc_state = (c == '[' || c == 'O') ? 2 : 0;
            } else {
                if (c == 'A' || c == 'B') input_queue_key(c);
                esc_state = 0;
            }
        }
    }
    return NULL;
}

static void tui_start_input(void) {
    if (!isatty(STDIN_FILENO)) return;
    if (tcgetattr(STDIN_FILENO, &global_input.orig_termios) != 0) return;

    /* The input thread is intentionally boring: it never writes to the terminal,
     * it only queues navigation/control state.  Rendering remains owned by the
     * main thread and follows the same full-frame redraw path as the
     * noninteractive UI. Keep ISIG set so Ctrl-C still restores the alternate
     * screen. */
    struct termios raw = global_input.orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return;

    pthread_mutex_lock(&global_input.mu);
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    global_input.raw_mode = true;
    global_input.enabled = true;
    global_input.running = 1;
    if (pthread_create(&global_input.thread, NULL, input_thread_main, NULL) == 0) {
        global_input.thread_started = true;
    } else {
        global_input.running = 0;
        global_input.enabled = false;
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
}

static void tui_stop_input(void) {
    if (global_input.thread_started) {
        global_input.running = 0;
        pthread_join(global_input.thread, NULL);
        global_input.thread_started = false;
    }
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    global_input.enabled = false;
}

void tui_consume_input(eval_ui *ui) {
    if (!ui->enabled || !global_input.enabled) return;

    pthread_mutex_lock(&global_input.mu);
    int move = global_input.move_delta;
    bool enter = global_input.enter_pressed;
    bool pause = global_input.pause_pressed;
    bool quit = global_input.quit_pressed;
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    if (quit) {
        ui->quit_requested = true;
        ui->paused = false;
        return;
    }

    if (pause) ui->paused = !ui->paused;

    if (move) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        int selected = ui->selected_case + move;
        if (selected < 0) selected = 0;
        if (selected >= ui->ncases) selected = ui->ncases - 1;
        ui->selected_case = selected;
    }

    if (enter) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        if (ui->selected_case != ui->active_case ||
            !eval_status_running(ui->status[ui->active_case]))
        {
            ui->requested_case = ui->selected_case;
        }
    }
}

void tui_restore(void) {
    eval_ui *ui = global_ui;
    tui_stop_input();
    if (!ui || !ui->active) return;
    fputs(ANSI_RESET "\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
    ui->active = false;
}

static void tui_signal_restore(int sig) {
    /* Signal handlers cannot safely run the full tui_restore() path: that path
     * joins the input thread and uses stdio.  For Ctrl-C / termination, do the
     * minimal terminal repair directly, then re-raise the signal with its
     * default action so the process status remains correct. */
    eval_ui *ui = global_ui;
    global_input.running = 0;
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    if (ui && ui->active) {
        const char restore[] = ANSI_RESET "\x1b[?25h\x1b[?1049l";
        if (write(STDOUT_FILENO, restore, sizeof(restore) - 1) == -1) {}
        ui->active = false;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

void tui_start(eval_ui *ui, int ncases, int max_tokens, bool enabled) {
    memset(ui, 0, sizeof(*ui));
    ui->enabled = enabled;
    ui->ncases = ncases;
    ui->max_tokens = max_tokens;
    ui->selected_case = 0;
    ui->requested_case = -1;
    ui->quit_requested = false;
    ui->status = calloc((size_t)ncases, sizeof(*ui->status));
    ui->guess = calloc((size_t)ncases, sizeof(*ui->guess));
    ui->prompt_tokens = calloc((size_t)ncases, sizeof(*ui->prompt_tokens));
    ui->generated_tokens = calloc((size_t)ncases, sizeof(*ui->generated_tokens));
    if (!ui->status || !ui->guess || !ui->prompt_tokens || !ui->generated_tokens) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (!enabled) return;

    terminal_size(&ui->cols, &ui->rows);
    if (ui->cols < 90 || ui->rows < 18) {
        ui->enabled = false;
        return;
    }
    ui->left_w = ui->cols / 2;
    if (ui->left_w < 42) ui->left_w = 42;
    if (ui->left_w > 72) ui->left_w = 72;
    ui->right_x = ui->left_w + 3;
    ui->right_w = ui->cols - ui->right_x + 1;
    ui->body_y = 6;
    ui->body_h = ui->rows - ui->body_y + 1;

    global_ui = ui;
    atexit(tui_restore);
    signal(SIGINT, tui_signal_restore);
    signal(SIGTERM, tui_signal_restore);
#ifdef SIGHUP
    signal(SIGHUP, tui_signal_restore);
#endif
#ifdef SIGQUIT
    signal(SIGQUIT, tui_signal_restore);
#endif

    fputs("\x1b[?1049h\x1b[?25l", stdout);
    ui->active = true;
    tui_start_input();
    tui_draw_frame(ui);
    tui_refresh(ui, "idle");
}

void tui_free(eval_ui *ui) {
    if (ui->active) tui_restore();
    free(ui->status);
    free(ui->guess);
    free(ui->prompt_tokens);
    free(ui->generated_tokens);
    buf_free(&ui->stream);
    style_free(&ui->styles);
    memset(ui, 0, sizeof(*ui));
}

void plain_set_thinking_color(bool use_color) {
    if (use_color) fputs(ANSI_DIM, stdout);
}

void plain_reset_color(bool use_color) {
    if (use_color) fputs(ANSI_RESET, stdout);
}

bool tui_has_switch_request(eval_ui *ui, int running_idx) {
    return ui->enabled &&
           ui->requested_case >= 0 &&
           ui->requested_case < ui->ncases &&
           ui->requested_case != running_idx;
}

bool tui_has_quit_request(eval_ui *ui) {
    return ui->enabled && ui->quit_requested;
}

void mark_case_pending(eval_ui *ui, int idx) {
    ui->status[idx] = EVAL_PENDING;
    ui->guess[idx][0] = '\0';
}

double tui_wait_if_paused(eval_ui *ui, const char *phase) {
    if (!ui->enabled || !ui->paused) return 0.0;
    bool was_running = ui->run_clock_active;
    if (was_running) tui_run_clock_stop(ui);
    double start = now_sec();
    tui_refresh(ui, phase);
    while (ui->paused) {
        usleep(50000);
        tui_consume_input(ui);
        tui_refresh(ui, phase);
    }
    if (was_running) tui_run_clock_start(ui);
    tui_refresh(ui, phase);
    return now_sec() - start;
}
