/* ds4-eval entry point: case scheduling, engine/session setup, and the
 * final report (moved verbatim from ds4_eval.c; see eval_internal.h for the
 * module overview). */

#include "eval_internal.h"

static int next_pending_case(eval_ui *ui, int start) {
    if (ui->ncases <= 0) return -1;
    for (int off = 0; off < ui->ncases; off++) {
        int i = (start + off) % ui->ncases;
        if (ui->status[i] == EVAL_PENDING) return i;
    }
    return -1;
}

static int parse_case_sequence(const char *arg, int ncases, int **seq_out, int *len_out) {
    int cap = 8;
    int len = 0;
    int *seq = malloc((size_t)cap * sizeof(*seq));
    if (!seq) {
        fprintf(stderr, "ds4-eval: out of memory while parsing --case-sequence\n");
        return -1;
    }

    const char *p = arg;
    while (p && *p) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        errno = 0;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || errno != 0 || v < 1 || v > ncases) {
            fprintf(stderr,
                    "ds4-eval: invalid --case-sequence entry near '%s' "
                    "(valid range: 1..%d)\n",
                    p, ncases);
            free(seq);
            return -1;
        }
        if (len == cap) {
            cap *= 2;
            int *new_seq = realloc(seq, (size_t)cap * sizeof(*seq));
            if (!new_seq) {
                fprintf(stderr, "ds4-eval: out of memory while parsing --case-sequence\n");
                free(seq);
                return -1;
            }
            seq = new_seq;
        }
        seq[len++] = (int)v - 1;
        p = end;
        while (isspace((unsigned char)*p)) p++;
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p) {
            fprintf(stderr, "ds4-eval: expected comma in --case-sequence near '%s'\n", p);
            free(seq);
            return -1;
        }
    }
    if (len == 0) {
        fprintf(stderr, "ds4-eval: --case-sequence cannot be empty\n");
        free(seq);
        return -1;
    }
    *seq_out = seq;
    *len_out = len;
    return 0;
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-eval: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-eval: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-eval: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-eval: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static const char *report_status_name(eval_status st) {
    switch (st) {
    case EVAL_PASSED: return "PASSED";
    case EVAL_FAILED: return "FAILED";
    case EVAL_SKIPPED: return "SKIPPED";
    case EVAL_STOPPED: return "STOPPED";
    case EVAL_PREFILL: return "PREFILL";
    case EVAL_THINKING: return "RUNNING";
    case EVAL_PENDING:
    default: return "PENDING";
    }
}

static void print_eval_report(const eval_ui *ui, int ncases, int passed, int failed) {
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));

    printf("ds4-eval: %d/%d passed", passed, ncases);
    if (failed) printf(", %d failed", failed);
    printf(", runtime %s\n", elapsed);
    printf("%-3s %-8s %8s %8s %8s %-8s %-8s %s\n",
           "#", "state", "prompt", "gen", "total", "given", "correct", "test");
    for (int i = 0; i < ncases; i++) {
        int prompt_tokens = ui->prompt_tokens ? ui->prompt_tokens[i] : 0;
        int generated_tokens = ui->generated_tokens ? ui->generated_tokens[i] : 0;
        int total_tokens = prompt_tokens + generated_tokens;
        const char *given = ui->guess && ui->guess[i][0] ? ui->guess[i] : "-";
        printf("%3d %-8s %8d %8d %8d %-8s %-8s %s/%s\n",
               i + 1,
               report_status_name(ui->status[i]),
               prompt_tokens,
               generated_tokens,
               total_tokens,
               given,
               eval_case_at(i)->answer,
               eval_case_at(i)->source,
               eval_case_at(i)->id);
    }
}

int main(int argc, char **argv) {
    eval_config cfg = parse_options(argc, argv);
    if (cfg.self_test_extractors) return run_extractor_self_tests();
    if (cfg.regrade_trace_path) return regrade_trace_file(cfg.regrade_trace_path);

    int ncases = (int)eval_case_count();
    if (cfg.question_limit > 0 && cfg.question_limit < ncases) ncases = cfg.question_limit;
    if (cfg.question_limit > (int)eval_case_count()) {
        fprintf(stderr, "ds4-eval: only %zu questions are embedded\n",
                eval_case_count());
        return 2;
    }
    int *case_sequence = NULL;
    int case_sequence_len = 0;
    if (cfg.case_sequence &&
        parse_case_sequence(cfg.case_sequence, ncases, &case_sequence, &case_sequence_len) != 0) {
        return 2;
    }
    if (!cfg.seed) {
        cfg.seed = (uint64_t)time(NULL) ^
                   ((uint64_t)getpid() << 32) ^
                   (uint64_t)clock();
    }

    FILE *trace = NULL;
    if (cfg.trace_path) {
        trace = fopen(cfg.trace_path, "w");
        if (!trace) {
            fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                    cfg.trace_path, strerror(errno));
            free(case_sequence);
            return 2;
        }
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .distributed = cfg.dist,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-eval: %s\n", dist_err);
        if (trace) fclose(trace);
        free(case_sequence);
        return 2;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        if (trace) fclose(trace);
        free(case_sequence);
        return 1;
    }

    int max_prompt_tokens = 0;
    int max_prompt_case = -1;
    const bool auto_ctx = cfg.ctx_size <= 0;
    if (auto_ctx) {
        cfg.ctx_size = eval_auto_context_size(engine, &cfg, ncases,
                                              &max_prompt_tokens, &max_prompt_case);
        fprintf(stderr,
                "ds4-eval: context auto-sized to %d tokens "
                "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
                cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1, cfg.max_tokens);
    } else {
        max_prompt_tokens = eval_max_prompt_tokens(engine, &cfg, ncases,
                                                   cfg.ctx_size, &max_prompt_case);
        fprintf(stderr,
                "ds4-eval: context set to %d tokens "
                "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
                cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1, cfg.max_tokens);
        eval_warn_context_budget(&cfg, max_prompt_tokens, max_prompt_case);
    }
    fprintf(stderr, "ds4-eval: model shape %s\n", ds4_engine_model_name(engine));
    eval_warn_think_max_downgraded(&cfg);
    trace_write_header(trace, &cfg, ds4_engine_model_name(engine), ncases, max_prompt_tokens);
    log_context_memory(cfg.backend, cfg.ctx_size);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        fprintf(stderr, "ds4-eval: failed to create session\n");
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        free(case_sequence);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        free(case_sequence);
        return 1;
    }

    eval_ui ui;
    bool split_ui = !cfg.plain && isatty(STDOUT_FILENO);
    tui_start(&ui, ncases, cfg.max_tokens, split_ui);

    uint64_t rng = cfg.seed;
    int rc = 0;
    int next = 0;
    int sequence_pos = 0;
    while (next >= 0) {
        if (case_sequence_len > 0) {
            if (sequence_pos >= case_sequence_len) break;
            next = case_sequence[sequence_pos++];
            ui.selected_case = next;
            ui.selection_active = false;
        }
        tui_consume_input(&ui);
        tui_wait_if_paused(&ui, "idle");
        if (ui.quit_requested) break;
        if (case_sequence_len == 0 && ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
        }

        eval_run_result result = run_one_case(engine, session, &cfg, &ui, trace, next, &rng);
        if (result == EVAL_RUN_ERROR) {
            rc = 1;
            break;
        }
        if (result == EVAL_RUN_QUIT) break;
        /* A successful case should advance to the next pending benchmark.  If a
         * stale quit flag ever survives here, clear it; real q handling either
         * returns EVAL_RUN_QUIT from run_one_case() or is consumed at the top of
         * the next idle iteration. */
        ui.quit_requested = false;
        if (case_sequence_len > 0) continue;
        if (ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
            continue;
        }
        next = next_pending_case(&ui, next + 1);
        if (next >= 0) {
            ui.selected_case = next;
            ui.selection_active = false;
        }
    }

    int passed = 0;
    int failed = 0;
    for (int i = 0; i < ncases; i++) {
        if (ui.status[i] == EVAL_PASSED) passed++;
        else if (ui.status[i] == EVAL_FAILED) failed++;
    }

    if (ui.active) tui_restore();
    print_eval_report(&ui, ncases, passed, failed);
    if (trace) {
        fprintf(trace,
                "===== SUMMARY =====\n"
                "passed: %d\n"
                "failed: %d\n"
                "total: %d\n",
                passed, failed, ncases);
        fflush(trace);
    }

    tui_free(&ui);
    ds4_session_free(session);
    ds4_engine_close(engine);
    if (trace) fclose(trace);
    free(case_sequence);
    return rc || failed ? 1 : 0;
}
