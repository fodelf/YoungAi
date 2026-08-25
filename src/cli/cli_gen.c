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

bool is_rendered_chat_prompt(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out) {
    if (is_rendered_chat_prompt(gen->prompt)) {
        ds4_tokenize_rendered_chat(engine, gen->prompt, out);
    } else {
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt,
                               cli_effective_think_mode(gen), out);
    }
}

int run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: sampled CLI generation requires a session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    ds4_think_mode think_mode = cli_effective_think_mode(&cfg->gen);
    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };

    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int sync_rc = ds4_session_sync(session, prompt, err, sizeof(err));
    cli_dist_busy_set(cfg, false);
    if (sync_rc != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    /* 生成区边界 + 投机贪心门: anticycle 只扫 [prompt 末, end); argmax 接受
     * (copy-spec/MTP)只在 temp==0 保分布, 带温采样时关断。 */
    ds4_session_mark_generation_start(session);
    ds4_session_set_spec_greedy(session, cfg->gen.temperature <= 0.0f);

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(session, cfg->gen.temperature, 0,
                                       cfg->gen.top_p, cfg->gen.min_p, &rng);
        if (getenv("DS4_DECODE_DIAG")) {
            fprintf(stderr,
                    "ds4: [decode-diag] sample#%d token=%d eos=%d max_tokens=%d\n",
                    generated, token, ds4_token_eos(engine), max_tokens);
        }
        if (token == ds4_token_eos(engine)) break;

        int toks[513];  /* wave 69: chained spec-pipe, up to 1 + DEPTH(<=7) verify cycles
                         * (8*64 rows + first_token). Was [65] (1 cycle) which blocked the
                         * spec-pipe 2nd cycle on full accepts (n_acc+spec_next_kb>65 -> the
                         * ~1/9 fire rate); 129 enabled 1 lookahead; 513 enables the chain. */
        int ntok = 0;
        /* Greedy decode always takes the speculative path: the n-gram copy
         * drafter is the natural default (self-gating, lossless), with MTP
         * drafting when a draft model is loaded. */
        if (cfg->gen.temperature <= 0.0f &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            cli_dist_busy_set(cfg, true);
            ntok = ds4_session_eval_speculative_argmax(session,
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
                ds4_session_free(session);
                return 1;
            }
        } else {
            cli_dist_busy_set(cfg, true);
            int eval_rc = ds4_session_eval(session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (eval_rc != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
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
    if (cli_interrupt_requested()) cli_interrupt_clear();

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);

    ds4_session_free(session);
    return 0;
}

int run_generation(ds4_engine *engine, const cli_config *cfg) {
    ds4_tokens prompt = {0};
    build_prompt(engine, &cfg->gen, &prompt);

    int rc = 0;
    if (cfg->gen.metal_graph_test) {
        rc = ds4_engine_metal_graph_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_full_test) {
        rc = ds4_engine_metal_graph_full_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_prompt_test) {
        rc = ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg->gen.ctx_size);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.score_ids_path) {
        rc = run_score_ids(engine, cfg);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logits_path) {
        rc = run_logits_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logprobs_path) {
        rc = run_logprob_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }

    const bool diagnostic = cfg->gen.dump_tokens ||
                            cfg->gen.head_test ||
                            cfg->gen.first_token_test;
    if (cfg->gen.head_test) {
        rc = ds4_engine_head_test(engine, &prompt);
    }
    if (rc == 0 && cfg->gen.first_token_test) {
        rc = ds4_engine_first_token_test(engine, &prompt);
    }
    if (cfg->gen.dump_tokens) {
        ds4_engine_dump_tokens(engine, &prompt);
    }

    if (diagnostic) {
        if (rc == 0) {
            fprintf(stderr, "ds4: diagnostic run completed on the native %s path.\n",
                    ds4_backend_name(cfg->engine.backend));
        }
    } else {
        /* Generation always runs the sampled-generation loop: it hosts the
         * speculative verifier for the always-armed copy-spec drafter (and MTP
         * when loaded); greedy temp=0 is just a special case of it. */
        rc = run_sampled_generation(engine, cfg, &prompt);
    }

    ds4_tokens_free(&prompt);
    return rc;
}
