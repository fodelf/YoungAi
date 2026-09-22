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

static volatile sig_atomic_t cli_interrupted;
static volatile sig_atomic_t cli_dist_busy;
static volatile sig_atomic_t cli_dist_notice_printed;

static const char cli_dist_drain_msg[] =
    "\nds4: stopping after the distributed cluster finishes the current token/chunk...\n";

void cli_sigint_handler(int sig) {
    (void)sig;
    cli_interrupted = 1;
    if (cli_dist_busy && !cli_dist_notice_printed) {
        cli_dist_notice_printed = 1;
        ssize_t ignored = write(STDERR_FILENO,
                                cli_dist_drain_msg,
                                sizeof(cli_dist_drain_msg) - 1u);
        (void)ignored;
    }
}

bool cli_interrupt_requested(void) {
    return cli_interrupted != 0;
}

void cli_interrupt_clear(void) {
    cli_interrupted = 0;
    cli_dist_notice_printed = 0;
}

bool cli_distributed_coordinator(const cli_config *cfg) {
    return cfg && cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR;
}

void cli_dist_busy_set(const cli_config *cfg, bool busy) {
    if (!cli_distributed_coordinator(cfg)) return;
    cli_dist_busy = busy ? 1 : 0;
    if (!busy) cli_dist_notice_printed = 0;
}

int cli_wait_distributed_route(const cli_config *cfg, ds4_session *session) {
    if (!cli_distributed_coordinator(cfg)) return 0;

    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }

        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

int main(int argc, char **argv) {
    cli_config cfg = parse_options(argc, argv);
    if (cfg.gen.classify_only) {
        /* Mode P/G router decision, no model loaded: programming -> resident
         * programming model; everyday -> full cached model. */
        if (cfg.gen.prompt == NULL) {
            fprintf(stderr, "ds4: --classify requires -p or --prompt-file\n");
            free(cfg.prompt_owned);
            return 2;
        }
        bool prog = ds4_prompt_is_programming(cfg.gen.prompt);
        printf("%s\n", prog ? "programming" : "daily");
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return 0;
    }
    if (cfg.gen.route && cfg.gen.prompt) {
        /* Mode P/G dynamic routing: classify the prompt, then point the engine at
         * the resident programming model (Mode P) or the full cached model (Mode
         * G) BEFORE it is opened. The rest of the run is unchanged. */
        bool prog = ds4_prompt_is_programming(cfg.gen.prompt);
        const char *picked = prog
            ? (cfg.gen.route_prog  ? cfg.gen.route_prog  : "gguf/reactgo-prog.gguf")
            : (cfg.gen.route_daily ? cfg.gen.route_daily : cfg.engine.model_path);
        cfg.engine.model_path = picked;
        fprintf(stderr, "ds4: route -> Mode %s [%s] -> %s\n",
                prog ? "P (编程/常驻快)" : "G (日常/全模型缓存)",
                prog ? "programming" : "daily", picked);
    }
    if (cfg.gen.dump_tokens) {
        if (cfg.gen.prompt == NULL) {
            fprintf(stderr, "ds4: --dump-tokens requires -p or --prompt-file\n");
            free(cfg.prompt_owned);
            return 2;
        }
        int rc = ds4_dump_text_tokenization(cfg.engine.model_path,
                                            cfg.gen.prompt,
                                            stdout);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return rc;
    }
    cfg.engine.inspect_only = cfg.inspect;
    /* ★投机开关必须在**开模型之前**就告诉引擎★(2026-09-16, mtp-1.md M3′): 它决定 DSpark 三塔
     * 那 7.3 GiB 在装载时排第几优先级 —— 投机不开时三塔一次都不读, 排最后; 投机开着时它每轮都读,
     * 就不能再排在主干专家后面。run_v41_generation 里那句 set_dspark 是开完模型才执行的, 太晚了。
     * (这里设一次、那里再设一次不冲突: 同一个值。) */
    ds4_engine_v41_set_dspark(cfg.gen.dspark && !cfg.gen.no_dspark);
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return 1;
    }
    if (cfg.dist && (cfg.dist->role == DS4_DISTRIBUTED_WORKER || cfg.dist->tp_enabled)) {
        ds4_dist_generation_options dist_gen = {
            .prompt = cfg.gen.prompt,
            .system = cfg.gen.system,
            .dump_logits_path = cfg.gen.dump_logits_path,
            .dump_logprobs_path = cfg.gen.dump_logprobs_path,
            .dump_logprobs_top_k = cfg.gen.dump_logprobs_top_k,
            .n_predict = cfg.gen.n_predict,
            .ctx_size = cfg.gen.ctx_size,
            .temperature = cfg.gen.temperature,
            .top_p = cfg.gen.top_p,
            .min_p = cfg.gen.min_p,
            .seed = cfg.gen.seed,
            .think_mode = cfg.gen.think_mode,
        };
        int rc = ds4_dist_run(engine, cfg.dist, &dist_gen);
        ds4_engine_close(engine);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return rc;
    }
    /* V4.1 的上下文来自模型元数据(用户 2026-09-22: 引擎里没有写死的上下文): 写进 cfg, 后面按 ctx 判 Think Max / 打 JSON 头
     * 的那些 V4 时代代码看到的就是同一个数, 不会拿 V4 会话默认的 32768 去降级 Think Max。 */
    if (ds4_engine_is_v41(engine)) cfg.gen.ctx_size = ds4_engine_v41_ctx();
    if (!cfg.inspect) {
        /* V4.1 不打 V4 那行"context buffer N MiB"估算: 它按 V4 会话算, 与 V4.1 无关, 09-20 就被当成真分配写进过脚本注释 */
        if (ds4_engine_is_v41(engine)) fprintf(stderr, "ds4: V4.1 上下文 %d(模型元数据 deepseek4.context_length; 状态按本趟位置分配)\n", cfg.gen.ctx_size);
        else log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);
        cli_warn_think_max_downgraded(&cfg.gen, "--think-max");
    }
    int rc = 0;
    if (cfg.inspect) {
        ds4_engine_summary(engine);
    } else if (cfg.gen.imatrix_output_path) {
        rc = ds4_engine_collect_imatrix(engine,
                                        cfg.gen.imatrix_dataset_path,
                                        cfg.gen.imatrix_output_path,
                                        cfg.gen.ctx_size,
                                        cfg.gen.imatrix_max_prompts,
                                        cfg.gen.imatrix_max_tokens);
    } else if (cfg.gen.perplexity_file_path) {
        rc = run_perplexity_file(engine, &cfg);
    } else if (cfg.gen.gen_ids_path) {
        rc = run_gen_ids(engine, &cfg);
    } else if (cfg.gen.score_ids_path) {
        /* score-ids 不需要 prompt; 放 REPL 判断之前, 否则无 -p 时被吞进交互模式 */
        rc = run_score_ids(engine, &cfg);
    } else if (cfg.gen.prompt == NULL) {
        rc = run_repl(engine, &cfg);
    } else {
        rc = run_generation(engine, &cfg);
    }
    ds4_engine_close(engine);
    ds4_dist_options_free(cfg.dist);
    free(cfg.prompt_owned);
    return rc;
}
