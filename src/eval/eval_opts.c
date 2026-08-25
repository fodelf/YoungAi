/* ds4-eval command-line parsing, usage text, and configuration warnings
 * (moved verbatim from ds4_eval.c). */

#include "eval_internal.h"

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64_arg(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_arg(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-eval: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "ds4-eval: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-eval [options]\n"
        "\n"
        "Runs a small built-in GPQA Diamond/audited SuperGPQA/AIME2025/COMPSEC integration test.\n"
        "The TTY UI keeps the question list on the left and streams sampled\n"
        "tokens live on the right; thinking text is dim grey until </think>.\n"
        "In the TTY UI, Up/Down selects a question, Enter runs it next,\n"
        "p pauses or resumes evaluation, and q exits with a report.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  -c, --ctx N            Allocated session context. Default: auto-sized.\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --warm-weights         Touch mapped tensor pages before evaluation.\n"
        "  --power N              Target GPU duty cycle percentage, 1..100. Default: 100\n"
        "\n"
        "Distributed:\n");
    ds4_dist_usage(fp);
    fprintf(fp,
        "\n"
        "\n"
        "Evaluation:\n"
        "  -n, --tokens N         Max generated tokens per question. Default: 16000\n"
        "  --questions N          Run only the first N embedded questions.\n"
        "  --case-sequence LIST   Run 1-based case numbers in this comma-separated order.\n"
        "  --temp F               Sampling temperature. Default: 0\n"
        "  --top-p F              Nucleus sampling probability. Default: 1\n"
        "  --min-p F              Keep tokens scoring at least F times the top token. Default: 0.05\n"
        "  --seed N               Sampling seed. Default: time-based\n"
        "  --trace FILE           Write questions, outputs, and grading decisions.\n"
        "  --regrade-trace FILE   Regrade a prior --trace file without loading the model.\n"
        "  --think                Enable thinking mode. Default\n"
        "  --think-max            Use Think Max. Auto context allocates at least 393216 tokens.\n"
        "  --nothink              Disable thinking mode.\n"
        "  --soft-limit-reply-budget N\n"
        "                         Inside the last N tokens, close thinking if\n"
        "                         </think> is already among the top close ranks.\n"
        "                         Default: 1024\n"
        "  --hard-limit-reply-budget N\n"
        "                         Force </think> with N tokens left for the answer.\n"
        "                         Default: 512\n"
        "  --soft-limit-think-close-rank N\n"
        "                         Soft-close when </think> is in the top N tokens.\n"
        "                         Default: 3\n"
        "  --pause-ms N           Pause after each result in the TTY UI. Default: 350\n"
        "  --plain                Disable split-screen ANSI UI.\n"
        "  --self-test-extractors Run answer-extractor self-tests and exit.\n"
        "  -h, --help             Show this help.\n");
}

eval_config parse_options(int argc, char **argv) {
    eval_config c = {
        .model_path = "ds4flash.gguf",
        .backend = default_backend(),
        .max_tokens = 16000,
        .top_p = DS4_DEFAULT_TOP_P,
        .min_p = DS4_DEFAULT_MIN_P,
        .pause_ms = 350,
        .soft_limit_reply_budget = 1024,
        .hard_limit_reply_budget = 512,
        .soft_limit_think_close_rank = 3,
        .think_mode = DS4_THINK_HIGH,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-eval: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.max_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--questions")) {
            c.question_limit = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--case-sequence")) {
            c.case_sequence = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--regrade-trace")) {
            c.regrade_trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--soft-limit-reply-budget")) {
            c.soft_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--hard-limit-reply-budget")) {
            c.hard_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--soft-limit-think-close-rank")) {
            c.soft_limit_think_close_rank = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--pause-ms")) {
            c.pause_ms = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-eval: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--think")) {
            c.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--plain")) {
            c.plain = true;
        } else if (!strcmp(arg, "--self-test-extractors")) {
            c.self_test_extractors = true;
        } else {
            fprintf(stderr, "ds4-eval: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (c.self_test_extractors || c.regrade_trace_path) return c;

    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-eval: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-eval: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }

    if (c.max_tokens > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --tokens (%d) exceeds the %d token context cap\n",
                c.max_tokens, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (c.ctx_size > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --ctx (%d) exceeds the %d token context cap\n",
                c.ctx_size, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (ds4_think_mode_enabled(c.think_mode) &&
        c.hard_limit_reply_budget >= c.max_tokens) {
        fprintf(stderr,
                "ds4-eval: --hard-limit-reply-budget (%d) must be smaller than --tokens (%d)\n",
                c.hard_limit_reply_budget, c.max_tokens);
        exit(2);
    }
    if (ds4_think_mode_enabled(c.think_mode) &&
        c.soft_limit_reply_budget < c.hard_limit_reply_budget) {
        fprintf(stderr,
                "ds4-eval: --soft-limit-reply-budget (%d) must be >= --hard-limit-reply-budget (%d)\n",
                c.soft_limit_reply_budget, c.hard_limit_reply_budget);
        exit(2);
    }
    return c;
}

void eval_warn_think_max_downgraded(const eval_config *cfg) {
    if (cfg->think_mode != DS4_THINK_MAX ||
        ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size) == DS4_THINK_MAX) {
        return;
    }
    fprintf(stderr,
            "ds4-eval: warning: --think-max needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
            ds4_think_max_min_context(),
            cfg->ctx_size);
}

void eval_warn_context_budget(const eval_config *cfg, int max_prompt_tokens, int max_prompt_case) {
    if (max_prompt_tokens >= cfg->ctx_size) {
        fprintf(stderr,
                "ds4-eval: warning: largest prompt (%d tokens, case=%d) does not fit ctx=%d\n",
                max_prompt_tokens,
                max_prompt_case + 1,
                cfg->ctx_size);
        return;
    }

    const int room = cfg->ctx_size - max_prompt_tokens;
    if (room < cfg->max_tokens) {
        fprintf(stderr,
                "ds4-eval: warning: largest prompt (%d tokens, case=%d) leaves %d generation tokens in ctx=%d; requested %d\n",
                max_prompt_tokens,
                max_prompt_case + 1,
                room,
                cfg->ctx_size,
                cfg->max_tokens);
    }
}
