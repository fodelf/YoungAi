/* ds4-eval case execution: prompt construction, context auto-sizing, and
 * the per-case prefill/generate/grade loop (moved verbatim from ds4_eval.c). */

#include "eval_internal.h"

static const char *eval_system_prompt(void) {
    return "You are solving a hard benchmark question. Reason carefully. "
           "The final answer must follow the requested format exactly.";
}

static char *build_question_prompt(const eval_case *tc) {
    byte_buf b = {0};
    int nchoices = eval_case_nchoices(tc);
    buf_appendf(&b, "%s\n", tc->question);
    if (nchoices > 0) {
        buf_append(&b, "\nChoices:\n", strlen("\nChoices:\n"));
        for (int i = 0; i < nchoices; i++) {
            buf_appendf(&b, "%c. %s\n", 'A' + i, tc->choice[i]);
        }
        buf_append(&b,
            "\nSolve the question. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <letter>",
            strlen("\nSolve the question. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <letter>"));
    } else if (eval_case_is_compsec(tc)) {
        buf_append(&b,
            "\nAt the end, write exactly one final line in this format and do not "
            "write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>",
            strlen("\nAt the end, write exactly one final line in this format and do not "
                   "write anything after it:\n"
                   "Answer: <line number or comma-separated line numbers>"));
    } else {
        buf_append(&b,
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <integer>",
            strlen("\nSolve the problem. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <integer>"));
    }
    if (b.v) return b.v;
    char *empty = malloc(1);
    if (empty) empty[0] = '\0';
    return empty;
}

int eval_max_prompt_tokens(ds4_engine *engine,
                           const eval_config *cfg,
                           int ncases,
                           int ctx_for_think_mode,
                           int *max_case_out)
{
    int max_prompt = 0;
    int max_case = -1;
    const ds4_think_mode think_mode =
        ds4_think_mode_for_context(cfg->think_mode, ctx_for_think_mode);

    for (int i = 0; i < ncases; i++) {
        char *question = build_question_prompt(eval_case_at(i));
        if (!question) {
            fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
            exit(1);
        }
        ds4_tokens prompt = {0};
        ds4_encode_chat_prompt(engine, eval_system_prompt(), question, think_mode, &prompt);
        if (prompt.len > max_prompt) {
            max_prompt = prompt.len;
            max_case = i;
        }
        ds4_tokens_free(&prompt);
        free(question);
    }
    if (max_case_out) *max_case_out = max_case;
    return max_prompt;
}

int eval_auto_context_size(ds4_engine *engine,
                           eval_config *cfg,
                           int ncases,
                           int *max_prompt_out,
                           int *max_case_out)
{
    int ctx = EVAL_MAX_CONTEXT;
    int max_prompt = 0;
    int max_case = -1;
    const int min_ctx = cfg->think_mode == DS4_THINK_MAX ?
                        (int)ds4_think_max_min_context() : 1;

    /* Think Max downgrades to normal thinking under its minimum context.  Size
     * the prompts iteratively so the prompt tokenizer sees the same effective
     * thinking mode that the actual run will use. */
    for (int iter = 0; iter < 3; iter++) {
        max_prompt = eval_max_prompt_tokens(engine, cfg, ncases, ctx, &max_case);
        long long required = (long long)max_prompt + (long long)cfg->max_tokens;
        if (required < min_ctx) required = min_ctx;
        if (required > EVAL_MAX_CONTEXT) {
            fprintf(stderr,
                    "ds4-eval: largest prompt (%d tokens, case %d) + --tokens (%d) exceeds the %d token context cap\n",
                    max_prompt, max_case + 1, cfg->max_tokens, EVAL_MAX_CONTEXT);
            exit(2);
        }
        if ((int)required == ctx) break;
        ctx = (int)required;
    }

    if (max_prompt_out) *max_prompt_out = max_prompt;
    if (max_case_out) *max_case_out = max_case;
    return ctx;
}

static void eval_prefill_progress(void *ud, const char *event, int current, int total) {
    eval_ui *ui = ud;
    if (!ui || !event) return;
    if (strcmp(event, "prefill_chunk") && strcmp(event, "prefill_display"))
        return;
    tui_consume_input(ui);
    tui_run_clock_tick(ui);
    ui->prefill_current = current;
    ui->prefill_total = total;
    double elapsed = now_sec() - ui->phase_start_sec;
    ui->speed_tps = elapsed > 0.001 ? (double)current / elapsed : 0.0;
    tui_refresh(ui, "prefill");
    double paused_sec = tui_wait_if_paused(ui, "prefill");
    if (paused_sec > 0.0) ui->phase_start_sec += paused_sec;
}

eval_run_result run_one_case(ds4_engine *engine, ds4_session *session,
                             const eval_config *cfg, eval_ui *ui,
                             FILE *trace, int idx, uint64_t *rng) {
    const eval_case *tc = eval_case_at(idx);
    const bool tty = ui->enabled;
    const bool use_plain_color = !tty && isatty(STDOUT_FILENO);
    const ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size);
    const char *system = eval_system_prompt();

    char *question = build_question_prompt(tc);
    if (!question) {
        fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
        return EVAL_RUN_ERROR;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, system, question, think_mode, &prompt);
    ui->prompt_tokens[idx] = prompt.len;
    ui->generated_tokens[idx] = 0;

    if (prompt.len >= cfg->ctx_size) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt does not fit context", system, question, "",
                         think_mode, prompt.len, 0, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    int generation_limit = cfg->max_tokens;
    int ctx_generation_limit = cfg->ctx_size - prompt.len;
    if (ctx_generation_limit < generation_limit) generation_limit = ctx_generation_limit;
    if (generation_limit < 1) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt leaves no generation room", system, question, "",
                         think_mode, prompt.len, 0, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    ui->active_case = idx;
    if (!ui->selection_active) ui->selected_case = idx;
    ui->requested_case = -1;
    ui->guess[idx][0] = '\0';
    ui->status[idx] = EVAL_PREFILL;
    ui->max_tokens = generation_limit;
    ui->think_max_tokens = generation_limit - cfg->hard_limit_reply_budget;
    if (ui->think_max_tokens < 0) ui->think_max_tokens = 0;
    tui_run_clock_start(ui);
    if (tty) {
        tui_reset_stream(ui, tc, ds4_think_mode_enabled(think_mode));
        tui_refresh(ui, "prefill");
    } else {
        printf("\n%s[%d/%d] %s%s%s/%s%s%s %s%s\n",
               use_plain_color ? ANSI_BOLD : "",
               idx + 1, ui->ncases,
               use_plain_color ? ANSI_CYAN : "", tc->source,
               use_plain_color ? ANSI_RESET : "",
               use_plain_color ? ANSI_CYAN : "", tc->domain,
               use_plain_color ? ANSI_RESET : "", tc->title,
               use_plain_color ? ANSI_RESET : "");
        fflush(stdout);
    }

    char err[256];
    ds4_session_set_progress(session, eval_prefill_progress, ui);
    ds4_session_set_display_progress(session, eval_prefill_progress, ui);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        tui_run_clock_stop(ui);
        fprintf(stderr, "ds4-eval: prefill failed for %s: %s\n", tc->id, err);
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                         system, question, "", think_mode, prompt.len, 0, 0.0, "?", NULL);
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_ERROR;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    int prompt_tokens = prompt.len;
    ui->prompt_tokens[idx] = prompt_tokens;
    ds4_tokens_free(&prompt);

    tui_consume_input(ui);
    tui_wait_if_paused(ui, "prefill");
    if (tui_has_quit_request(ui)) {
        ui->status[idx] = EVAL_STOPPED;
        ui->generated_tokens[idx] = 0;
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_QUIT;
    }
    if (tui_has_switch_request(ui, idx)) {
        mark_case_pending(ui, idx);
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_SWITCH;
    }

    ui->status[idx] = EVAL_THINKING;
    ui->generated = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;
    byte_buf raw = {0};
    bool plain_in_think = ds4_think_mode_enabled(think_mode);
    bool generation_in_think = ds4_think_mode_enabled(think_mode);
    eval_think_close_info think_close = {0};
    ds4_tokens think_close_tokens = {0};
    if (generation_in_think) ds4_tokenize_text(engine, "</think>", &think_close_tokens);
    if (!tty && plain_in_think) plain_set_thinking_color(use_plain_color);
    tui_refresh(ui, "thinking");

    const int eos = ds4_token_eos(engine);
    double t0 = ui->phase_start_sec;
    int forced_close_pos = -1;
    for (int i = 0; i < generation_limit; i++) {
        if (tty) {
            tui_consume_input(ui);
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
            double paused_sec = tui_wait_if_paused(ui, ui->in_think ? "thinking" : "answer");
            if (paused_sec > 0.0) {
                ui->phase_start_sec += paused_sec;
                t0 += paused_sec;
            }
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
        }

        int remaining_budget = generation_limit - ui->generated;
        int close_rank = 0;
        int token = -1;
        eval_think_close_kind close_kind = EVAL_THINK_CLOSE_NONE;

        /* Benchmarks usually cap generation length, but DeepSeek can spend the
         * entire budget in <think>.  This controller only acts while the model is
         * still in thinking mode.  The soft limit is conservative: it accepts the
         * model's own desire to end thinking when </think> is already near the
         * top of the distribution.  The hard limit is a uniform benchmark rule:
         * leave a fixed answer reserve instead of failing a case because all
         * tokens were spent before the visible answer could start. */
        if (generation_in_think && think_close_tokens.len > 0) {
            if (forced_close_pos >= 0) {
                token = think_close_tokens.v[forced_close_pos++];
                if (forced_close_pos >= think_close_tokens.len) forced_close_pos = -1;
            } else if (remaining_budget <= cfg->hard_limit_reply_budget) {
                close_kind = EVAL_THINK_CLOSE_HARD;
                token = think_close_tokens.v[0];
                forced_close_pos = think_close_tokens.len > 1 ? 1 : -1;
            } else if (remaining_budget <= cfg->soft_limit_reply_budget &&
                       think_close_tokens.len == 1) {
                close_rank = token_rank_in_top(session, think_close_tokens.v[0],
                                               cfg->soft_limit_think_close_rank);
                if (close_rank > 0) {
                    close_kind = EVAL_THINK_CLOSE_SOFT;
                    token = think_close_tokens.v[0];
                }
            }
        }
        if (token < 0)
            token = ds4_session_sample(session, cfg->temperature, 0,
                                       cfg->top_p, cfg->min_p, rng);
        if (token == eos) break;
        if (close_kind != EVAL_THINK_CLOSE_NONE &&
            think_close.kind == EVAL_THINK_CLOSE_NONE) {
            think_close.kind = close_kind;
            think_close.token_index = ui->generated + 1;
            think_close.remaining_budget = remaining_budget;
            think_close.rank = close_rank;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            plain_reset_color(use_plain_color);
            ui->generated_tokens[idx] = ui->generated;
            tui_run_clock_stop(ui);
            fprintf(stderr, "ds4-eval: decode failed for %s: %s\n", tc->id, err);
            trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                             system, question, raw.v ? raw.v : "", think_mode,
                             prompt_tokens, ui->generated, now_sec() - t0, "?",
                             &think_close);
            free(question);
            ds4_tokens_free(&think_close_tokens);
            buf_free(&raw);
            return EVAL_RUN_ERROR;
        }

        size_t len = 0;
        char *text = ds4_token_text(engine, token, &len);
        buf_append(&raw, text, len);
        ui->generated++;
        ui->generated_tokens[idx] = ui->generated;
        tui_run_clock_tick(ui);
        if (generation_in_think && raw.v && strstr(raw.v, "</think>")) {
            generation_in_think = false;
            if (think_close.kind == EVAL_THINK_CLOSE_NONE) {
                think_close.kind = EVAL_THINK_CLOSE_NATURAL;
                think_close.token_index = ui->generated;
                think_close.remaining_budget = remaining_budget;
                think_close.rank = 0;
            }
        }
        double elapsed = now_sec() - ui->phase_start_sec;
        ui->speed_tps = elapsed > 0.001 ? (double)ui->generated / elapsed : 0.0;

        if (tty) {
            stream_append_token_text(ui, text, len, false);
            tui_refresh(ui, ui->in_think ? "thinking" : "answer");
        } else {
            if (plain_in_think && strstr(raw.v ? raw.v : "", "</think>")) {
                plain_in_think = false;
                plain_reset_color(use_plain_color);
            }
            fwrite(text, 1, len, stdout);
            fflush(stdout);
        }
        free(text);
    }
    if (tty) {
        stream_append_token_text(ui, NULL, 0, true);
        tui_draw_stream(ui);
    } else {
        plain_reset_color(use_plain_color);
        if (!raw.v || raw.len == 0 || raw.v[raw.len - 1] != '\n') fputc('\n', stdout);
    }

    char got[EVAL_ANSWER_MAX];
    find_case_answer(tc, raw.v ? raw.v : "", got, sizeof(got));
    snprintf(ui->guess[idx], EVAL_ANSWER_MAX, "%s", got);
    bool pass = answer_matches(tc, got);
    ui->status[idx] = pass ? EVAL_PASSED : EVAL_FAILED;
    ui->generated_tokens[idx] = ui->generated;
    tui_run_clock_stop(ui);
    double sec = now_sec() - t0;
    tui_refresh(ui, pass ? "passed" : "failed");
    trace_write_case(trace, cfg, tc, idx, ui->ncases, pass ? "PASSED" : "FAILED", NULL,
                     system, question, raw.v ? raw.v : "", think_mode, prompt_tokens,
                     ui->generated, sec, got, &think_close);

    if (!tty) {
        printf("%s%s%s got %s expected %s (%.1fs, %d tokens)\n",
               use_plain_color ? (pass ? ANSI_GREEN : ANSI_RED) : "",
               pass ? "PASSED" : "FAIL",
               use_plain_color ? ANSI_RESET : "",
               got, tc->answer, sec, ui->generated);
    }

    if (tty && cfg->pause_ms > 0) usleep((useconds_t)cfg->pause_ms * 1000);
    free(question);
    ds4_tokens_free(&think_close_tokens);
    buf_free(&raw);
    return EVAL_RUN_OK;
}
