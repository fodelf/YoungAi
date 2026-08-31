#include "ds4.h"
#include "ds4_distributed.h"
#include "linenoise.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single DeepSeek chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include "cli_internal.h"

static char *trim_inplace(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void print_repl_help(void) {
    puts("Commands:");
    puts("  /help          Show this help.");
    puts("  /think         Use normal thinking mode.");
    puts("  /think-max     Use Think Max only when context is at least 393216 tokens.");
    puts("  /nothink       Disable thinking mode.");
    puts("  /ctx N         Set context size for following prompts.");
    puts("  /power N       Set GPU duty cycle percentage, 1..100.");
    puts("  /read FILE     Read a prompt from FILE and run it.");
    puts("  /quit, /exit   Leave the prompt.");
    puts("  Ctrl+C         Stop generation and return to the prompt.");
}

static bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static void history_file_path(char *buf, size_t len) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, len, "%s/.ds4_history", home);
}

static void tokens_insert(ds4_tokens *dst, int pos, const ds4_tokens *src) {
    if (!src || src->len <= 0) return;
    if (pos < 0) pos = 0;
    if (pos > dst->len) pos = dst->len;
    while (dst->len + src->len > dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 64;
        int *next = realloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
        if (!next) {
            perror("ds4: realloc");
            exit(1);
        }
        dst->v = next;
    }
    memmove(dst->v + pos + src->len, dst->v + pos,
            (size_t)(dst->len - pos) * sizeof(dst->v[0]));
    memcpy(dst->v + pos, src->v, (size_t)src->len * sizeof(src->v[0]));
    dst->len += src->len;
}

static void tokens_remove(ds4_tokens *dst, int pos, int n) {
    if (n <= 0 || pos < 0 || pos >= dst->len) return;
    if (pos + n > dst->len) n = dst->len - pos;
    memmove(dst->v + pos, dst->v + pos + n,
            (size_t)(dst->len - pos - n) * sizeof(dst->v[0]));
    dst->len -= n;
}

/* Insert/remove the Think Max prefix inside the existing transcript.  The
 * prefix lives after BOS, before any system/developer text, which mirrors the
 * API rendering path.  Changing it invalidates the session because every later
 * token position would otherwise refer to the wrong prefix. */
static void repl_chat_apply_max_prefix(ds4_engine *engine, repl_chat *chat, bool enable) {
    if (enable && chat->max_prefix_tokens == 0) {
        ds4_tokens prefix = {0};
        ds4_chat_append_max_effort_prefix(engine, &prefix);
        tokens_insert(&chat->transcript, 1, &prefix);
        chat->max_prefix_tokens = prefix.len;
        ds4_tokens_free(&prefix);
        if (chat->session) ds4_session_invalidate(chat->session);
    } else if (!enable && chat->max_prefix_tokens > 0) {
        tokens_remove(&chat->transcript, 1, chat->max_prefix_tokens);
        chat->max_prefix_tokens = 0;
        if (chat->session) ds4_session_invalidate(chat->session);
    }
}

static int repl_chat_create_session(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4: interactive chat KV cache requires a session backend\n");
        return 1;
    }
    if (chat->session) ds4_session_free(chat->session);
    chat->session = session;
    chat->ctx_size = ctx_size;
    return 0;
}

static int repl_chat_init(ds4_engine *engine, repl_chat *chat, const cli_config *cfg) {
    memset(chat, 0, sizeof(*chat));
    ds4_chat_begin(engine, &chat->transcript);
    repl_chat_apply_max_prefix(engine, chat,
                               cli_effective_think_mode(&cfg->gen) == DS4_THINK_MAX);
    if (cfg->gen.system && cfg->gen.system[0]) {
        ds4_chat_append_message(engine, &chat->transcript, "system", cfg->gen.system);
    }
    return repl_chat_create_session(engine, chat, cfg->gen.ctx_size);
}

static void repl_chat_free(repl_chat *chat) {
    if (!chat) return;
    ds4_session_free(chat->session);
    ds4_tokens_free(&chat->transcript);
    memset(chat, 0, sizeof(*chat));
}

static int repl_chat_set_ctx(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session_free(chat->session);
    chat->session = NULL;
    chat->ctx_size = 0;
    return repl_chat_create_session(engine, chat, ctx_size);
}

/* Run one interactive turn.  The transcript is tentatively extended with user
 * and assistant markers, then ds4_session_sync() decides whether this is a KV
 * continuation.  If prompt processing fails, the transcript rolls back before
 * returning to the prompt. */
static int run_chat_turn(ds4_engine *engine, cli_config *cfg, repl_chat *chat, const char *user_text) {
    if (!chat->session) {
        fprintf(stderr, "ds4: no active interactive KV cache\n");
        return 1;
    }

    ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                           chat->ctx_size);
    repl_chat_apply_max_prefix(engine, chat, think_mode == DS4_THINK_MAX);
    const int rollback_len = chat->transcript.len;
    ds4_chat_append_message(engine, &chat->transcript, "user", user_text);
    ds4_chat_append_assistant_prefix(engine, &chat->transcript, think_mode);

    const int old_pos = ds4_session_pos(chat->session);
    const int common = ds4_session_common_prefix(chat->session, &chat->transcript);
    const int cached = common == old_pos && chat->transcript.len >= old_pos ? common : 0;
    const int suffix = chat->transcript.len - cached;

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = cached,
        .input_tokens = suffix,
        .use_color = ds4_log_is_tty(stderr),
    };
    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(chat->session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int sync_rc = ds4_session_sync(chat->session, &chat->transcript, err, sizeof(err));
    cli_dist_busy_set(cfg, false);
    if (sync_rc != 0) {
        ds4_session_set_progress(chat->session, NULL, NULL);
        ds4_session_set_display_progress(chat->session, NULL, NULL);
        chat->transcript.len = rollback_len;
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        return 1;
    }
    ds4_session_set_progress(chat->session, NULL, NULL);
    ds4_session_set_display_progress(chat->session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    /* 每轮重钉生成区边界(本轮 prompt=至此的全部 transcript, 含此前生成的轮次);
     * 投机贪心门按本轮实际温度。 */
    ds4_session_mark_generation_start(chat->session);
    ds4_session_set_spec_greedy(chat->session, cfg->gen.temperature <= 0.0f);

    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(chat->session) - ds4_session_pos(chat->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(chat->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       cfg->gen.min_p,
                                       &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[513];  /* wave 69: chained spec-pipe, up to 1 + DEPTH(<=7) verify cycles
                         * (8*64 rows + first_token). Was [65] (1 cycle) which blocked the
                         * spec-pipe 2nd cycle on full accepts (n_acc+spec_next_kb>65 -> the
                         * ~1/9 fire rate); 129 enabled 1 lookahead; 513 enables the chain. */
        int ntok = 0;
        /* Greedy decode always takes the speculative entry: it self-gates to
         * plain flat decode unless a drafter is armed (--spec). */
        if (cfg->gen.temperature <= 0.0f) {
            cli_dist_busy_set(cfg, true);
            ntok = ds4_session_eval_speculative_argmax(chat->session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
        } else {
            cli_dist_busy_set(cfg, true);
            int eval_rc = ds4_session_eval(chat->session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (eval_rc != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            ds4_tokens_push(&chat->transcript, toks[j]);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);

    const bool interrupted = cli_interrupt_requested();
    if (interrupted && generated == 0) {
        chat->transcript.len = rollback_len;
        ds4_session_invalidate(chat->session);
    } else {
        ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
    }

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    if (interrupted) cli_interrupt_clear();
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)suffix / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    /* PC.4 per-turn TTFT metric: in a multi-turn session the prefix KV is
     * reused (cached) and only the appended suffix is prefilled, so TTFT
     * (wall-clock prefill of the suffix) is the real coding-UX denominator,
     * not the cold full-context prefill.  Parsed by tools/mtp_replay_q2.sh. */
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: per-turn: cached=%d suffix=%d TTFT=%.0f ms decode=%d tok %.2f t/s\n",
            cached,
            suffix,
            prefill_s * 1000.0,
            generated,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    return 0;
}

int run_repl(ds4_engine *engine, cli_config *cfg) {
    repl_chat chat;
    if (repl_chat_init(engine, &chat, cfg) != 0) return 1;

    /* The one-shot (-p) paths wait for the worker layer route before
     * generating; the interactive/REPL path skipped it, so the first turn's
     * prefill hit "distributed route incomplete: missing layer N" (PC.4
     * multi-turn replay needs the route up across turns).  No-op for non-dist. */
    if (cli_wait_distributed_route(cfg, chat.session) != 0) {
        repl_chat_free(&chat);
        return 1;
    }

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = cli_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    cli_interrupt_clear();

    char hist[PATH_MAX];
    history_file_path(hist, sizeof(hist));
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    print_repl_help();

    int rc = 0;
    for (;;) {
        errno = 0;
        char *line = linenoise("ds4> ");
        if (!line) {
            if (errno == EAGAIN || cli_interrupt_requested()) {
                cli_interrupt_clear();
                continue;
            }
            break;
        }
        char *cmd = trim_inplace(line);
        if (!cmd[0]) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(cmd);
        linenoiseHistorySave(hist);

        if (!strcmp(cmd, "/help")) {
            print_repl_help();
        } else if (!strcmp(cmd, "/think")) {
            cfg->gen.think_mode = DS4_THINK_HIGH;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: high.");
        } else if (!strcmp(cmd, "/think-max")) {
            cfg->gen.think_mode = DS4_THINK_MAX;
            bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                     chat.ctx_size) == DS4_THINK_MAX;
            repl_chat_apply_max_prefix(engine, &chat, active);
            cli_warn_think_max_downgraded(&cfg->gen, "/think-max");
            printf("Thinking mode: %s.\n", active ? "max" : "high (ctx below 393216)");
        } else if (!strcmp(cmd, "/nothink")) {
            cfg->gen.think_mode = DS4_THINK_NONE;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: none.");
        } else if (!strncmp(cmd, "/power", 6) && (cmd[6] == '\0' || isspace((unsigned char)cmd[6]))) {
            char *arg = trim_inplace(cmd + 6);
            if (!arg[0]) {
                printf("Power: %d%%.\n", ds4_session_power(chat.session));
            } else {
                int power = 0;
                if (!parse_power_percent(arg, &power)) {
                    fprintf(stderr, "ds4: /power must be between 1 and 100\n");
                } else if (ds4_session_set_power(chat.session, power) != 0) {
                    fprintf(stderr, "ds4: failed to set /power\n");
                } else {
                    cfg->engine.power_percent = power;
                    printf("Power: %d%%.\n", power);
                }
            }
        } else if (!strncmp(cmd, "/ctx", 4) && (cmd[4] == '\0' || isspace((unsigned char)cmd[4]))) {
            char *arg = trim_inplace(cmd + 4);
            if (!arg[0]) {
                fprintf(stderr, "ds4: /ctx needs a positive integer\n");
            } else {
                cfg->gen.ctx_size = parse_int(arg, "/ctx");
                log_context_memory(cfg->engine.backend, cfg->gen.ctx_size);
                rc = repl_chat_set_ctx(engine, &chat, cfg->gen.ctx_size);
                if (rc != 0) {
                    linenoiseFree(line);
                    break;
                }
                bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                         chat.ctx_size) == DS4_THINK_MAX;
                repl_chat_apply_max_prefix(engine, &chat, active);
                cli_warn_think_max_downgraded(&cfg->gen, "/ctx");
            }
        } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            linenoiseFree(line);
            break;
        } else if (!strncmp(cmd, "/read", 5) && (cmd[5] == '\0' || isspace((unsigned char)cmd[5]))) {
            char *path = trim_inplace(cmd + 5);
            if (!path[0]) {
                fprintf(stderr, "ds4: /read needs a file path\n");
            } else {
                char *prompt = read_prompt_file(path, false);
                if (prompt) {
                    rc = run_chat_turn(engine, cfg, &chat, prompt);
                    free(prompt);
                }
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "ds4: unknown command: %s\n", cmd);
            fprintf(stderr, "ds4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, cfg, &chat, cmd);
        }
        linenoiseFree(line);
    }
    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    repl_chat_free(&chat);
    return rc;
}
