/* eval_internal.h: module-internal declarations for the src/eval/ sources, the
 * mechanical split of the original single-file ds4_eval.c.  Every type,
 * macro, and function keeps its original name; functions listed below lost
 * their file-local `static` only because their callers now live in a
 * different file of this module. */
#ifndef DS4_EVAL_INTERNAL_H
#define DS4_EVAL_INTERNAL_H

#include "ds4.h"
#include "ds4_distributed.h"

/* ds4-eval: small built-in benchmark integration test.
 *
 * This program is deliberately not a unit test.  It loads the real model,
 * renders chat prompts, prefills them through ds4_session_sync(), samples the
 * continuation token by token, and grades the final answer.  The terminal UI is
 * also intentionally simple: no ncurses, just ANSI cursor movement, colors, and
 * a fixed two-pane layout.
 *
 * The embedded questions are small fixed subsets of GPQA Diamond, SuperGPQA,
 * AIME 2025, and COMPSEC.  The SuperGPQA slice is intentionally audited: rows
 * with wrong keys, missing figures, or underspecified prompts are replaced
 * instead of being locally re-keyed, because ds4-eval is a regression harness
 * and a bad target is worse than a merely hard target.  COMPSEC contains a
 * small audited subset of reduced C/C++ single-function vulnerability
 * localization questions derived from public CVE writeups; the CVE anchors and
 * private rationales are not rendered to the model.
 * GPQA is released under CC BY 4.0.  SuperGPQA is released under ODC-BY and
 * includes mostly original data plus a limited amount of transformed
 * third-party data.  The AIME 2025 mirror used here is MIT licensed.  Source
 * mirrors used while preparing this file:
 * https://huggingface.co/datasets/Wanfq/gpqa
 * https://huggingface.co/datasets/m-a-p/SuperGPQA
 * https://huggingface.co/datasets/test-time-compute/aime_2025
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define ANSI_RESET "\x1b[0m"
#define ANSI_DIM "\x1b[90m"
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BLUE "\x1b[34m"
#define ANSI_CYAN "\x1b[36m"
#define ANSI_BOLD "\x1b[1m"

#define EVAL_MAX_CHOICES 10
#define EVAL_ANSWER_MAX 32
#define EVAL_MAX_CONTEXT 1000000

typedef enum {
    EVAL_PENDING,
    EVAL_PREFILL,
    EVAL_THINKING,
    EVAL_SKIPPED,
    EVAL_STOPPED,
    EVAL_PASSED,
    EVAL_FAILED,
} eval_status;

typedef enum {
    EVAL_RUN_OK,
    EVAL_RUN_ERROR,
    EVAL_RUN_SWITCH,
    EVAL_RUN_QUIT,
} eval_run_result;

typedef enum {
    EVAL_THINK_CLOSE_NONE,
    EVAL_THINK_CLOSE_NATURAL,
    EVAL_THINK_CLOSE_SOFT,
    EVAL_THINK_CLOSE_HARD,
} eval_think_close_kind;

typedef struct {
    const char *source;
    const char *id;
    const char *domain;
    const char *title;
    const char *question;
    const char *choice[EVAL_MAX_CHOICES];
    const char *answer;
} eval_case;

typedef struct {
    char *v;
    size_t len;
    size_t cap;
} byte_buf;

typedef struct {
    unsigned char *v;
    size_t len;
    size_t cap;
} style_buf;

typedef struct {
    const char *model_path;
    const char *trace_path;
    const char *regrade_trace_path;
    const char *case_sequence;
    ds4_backend backend;
    int threads;
    int ctx_size;
    int max_tokens;
    int question_limit;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    int pause_ms;
    int power_percent;
    int soft_limit_reply_budget;
    int hard_limit_reply_budget;
    int soft_limit_think_close_rank;
    ds4_think_mode think_mode;
    ds4_dist_options dist;
    bool plain;
    bool warm_weights;
    bool quality;
    bool self_test_extractors;
} eval_config;

typedef struct {
    eval_think_close_kind kind;
    int token_index;
    int remaining_budget;
    int rank;
} eval_think_close_info;

typedef struct {
    int cols;
    int rows;
    int left_w;
    int right_x;
    int right_w;
    int body_y;
    int body_h;
    bool active;
    bool enabled;
    int ncases;
    eval_status *status;
    char (*guess)[EVAL_ANSWER_MAX];
    int *prompt_tokens;
    int *generated_tokens;
    int active_case;
    int generated;
    int max_tokens;
    int think_max_tokens;
    int prefill_current;
    int prefill_total;
    double phase_start_sec;
    double speed_tps;
    double run_elapsed_sec;
    double run_last_sec;
    bool run_clock_active;
    bool selection_active;
    int selected_case;
    int requested_case;
    bool paused;
    bool quit_requested;
    byte_buf stream;
    style_buf styles;
    bool in_think;
    char pending_tag[16];
    size_t pending_tag_len;
} eval_ui;

typedef struct {
    bool enabled;
    bool raw_mode;
    bool thread_started;
    volatile sig_atomic_t running;
    pthread_t thread;
    pthread_mutex_t mu;
    struct termios orig_termios;
    int move_delta;
    bool enter_pressed;
    bool pause_pressed;
    bool quit_pressed;
} eval_input;

/* Embedded question table (eval_cases_a/b/c.c).  The original single
 * eval_cases[] array is split into three arrays only because a C array
 * literal cannot span files; eval_case_at()/eval_case_count() preserve the
 * exact original order, so case indexes and --seq numbering are unchanged. */
extern const eval_case eval_cases_a[];
extern const size_t eval_cases_a_count;
extern const eval_case eval_cases_b[];
extern const size_t eval_cases_b_count;
extern const eval_case eval_cases_c[];
extern const size_t eval_cases_c_count;
const eval_case *eval_case_at(size_t i);
size_t eval_case_count(void);

/* eval_util.c */
void buf_append(byte_buf *b, const char *p, size_t n);
void buf_appendf(byte_buf *b, const char *fmt, ...);
int eval_case_nchoices(const eval_case *tc);
bool eval_case_is_multiple_choice(const eval_case *tc);
bool eval_case_is_compsec(const eval_case *tc);
void style_append(style_buf *b, unsigned char style, size_t n);
void buf_free(byte_buf *b);
void style_free(style_buf *b);
double now_sec(void);
double run_clock_sec(void);

/* eval_opts.c */
eval_config parse_options(int argc, char **argv);
void eval_warn_think_max_downgraded(const eval_config *cfg);
void eval_warn_context_budget(const eval_config *cfg, int max_prompt_tokens, int max_prompt_case);

/* eval_tui.c */
void tui_run_clock_tick(eval_ui *ui);
void tui_run_clock_start(eval_ui *ui);
void tui_run_clock_stop(eval_ui *ui);
double tui_run_clock_visible_sec(const eval_ui *ui);
void format_run_elapsed(char *dst, size_t dstlen, double sec);
void term_move(int row, int col);
void term_clear_to_eol(void);
int tui_right_text_w(const eval_ui *ui);
void tui_clear_left_line(eval_ui *ui, int row);
void term_set_color_for_status(eval_status st);
const char *status_name(eval_status st);
void print_trimmed(const char *s, int width);
void tui_draw_question_preview(eval_ui *ui, const eval_case *tc);
void tui_consume_input(eval_ui *ui);
void tui_restore(void);
void tui_start(eval_ui *ui, int ncases, int max_tokens, bool enabled);
void tui_free(eval_ui *ui);
void plain_set_thinking_color(bool use_color);
void plain_reset_color(bool use_color);
bool tui_has_switch_request(eval_ui *ui, int running_idx);
bool tui_has_quit_request(eval_ui *ui);
void mark_case_pending(eval_ui *ui, int idx);
double tui_wait_if_paused(eval_ui *ui, const char *phase);

/* eval_tui_draw.c */
void tui_draw_frame(eval_ui *ui);
void tui_reset_stream(eval_ui *ui, const eval_case *tc, bool in_think);
void stream_append_token_text(eval_ui *ui, const char *text, size_t len, bool finish);
void tui_draw_stream(eval_ui *ui);
void tui_refresh(eval_ui *ui, const char *phase);

/* eval_run.c */
int eval_max_prompt_tokens(ds4_engine *engine, const eval_config *cfg, int ncases, int ctx_for_think_mode, int *max_case_out);
int eval_auto_context_size(ds4_engine *engine, eval_config *cfg, int ncases, int *max_prompt_out, int *max_case_out);
eval_run_result run_one_case(ds4_engine *engine, ds4_session *session, const eval_config *cfg, eval_ui *ui, FILE *trace, int idx, uint64_t *rng);

/* eval_trace.c */
int token_rank_in_top(ds4_session *session, int token, int max_rank);
void trace_write_header(FILE *trace, const eval_config *cfg, const char *model_name, int ncases, int max_prompt_tokens);
void trace_write_case(FILE *trace, const eval_config *cfg, const eval_case *tc, int idx, int ncases, const char *status, const char *error, const char *system_prompt, const char *question_prompt, const char *model_output, ds4_think_mode effective_think_mode, int prompt_tokens, int generated_tokens, double elapsed_sec, const char *picked, const eval_think_close_info *think_close);
const char *trace_find_next_case(const char *start, const char *end);
char *trace_copy_model_output(const char *case_start, const char *case_end);
int regrade_trace_file(const char *path);

/* eval_extract.c */
void find_case_answer(const eval_case *tc, const char *generated, char *dst, size_t dstlen);
bool answer_matches(const eval_case *tc, const char *got);
int run_extractor_self_tests(void);

#endif /* DS4_EVAL_INTERNAL_H */
