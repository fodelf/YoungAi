/* agent_util.c — 机械拆分自 ds4_agent.c: Small Utilities And Command-Line Parsing。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

/* ============================================================================
 * Small Utilities And Command-Line Parsing
 * ============================================================================
 */

void agent_sigint_handler(int sig) {
    (void)sig;
    agent_sigint = 1;
}


char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

void write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t wr = write(fd, p, n);
        if (wr < 0) {
            if (errno == EINTR) continue;
            return;
        }
        p += wr;
        n -= (size_t)wr;
    }
}


void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static bool agent_slash_command_with_args(const char *cmd, const char *name) {
    size_t len = strlen(name);
    return !strncmp(cmd, name, len) &&
           (cmd[len] == '\0' || isspace((unsigned char)cmd[len]));
}

bool agent_slash_command_known(const char *cmd) {
    return !strcmp(cmd, "/help") ||
           !strcmp(cmd, "/save") ||
           !strcmp(cmd, "/compact") ||
           !strcmp(cmd, "/list") ||
           !strcmp(cmd, "/quit") ||
           !strcmp(cmd, "/exit") ||
           !strcmp(cmd, "/new") ||
           agent_slash_command_with_args(cmd, "/power") ||
           agent_slash_command_with_args(cmd, "/corr") ||
           agent_slash_command_with_args(cmd, "/switch") ||
           agent_slash_command_with_args(cmd, "/del") ||
           agent_slash_command_with_args(cmd, "/strip") ||
           agent_slash_command_with_args(cmd, "/history");
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-agent: invalid backend: %s\n", s);
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

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-agent [options]\n"
        "\n"
        "This is an experimental native DS4 agent MVP. It keeps the terminal\n"
        "responsive with linenoise's multiplexed API while a model worker owns\n"
        "the live KV session.\n"
        "\n"
        "Options:\n"
        "  -m, --model FILE        GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE             Optional MTP support GGUF.\n"
        "  --mtp-draft N          Maximum MTP draft tokens. Default: 1\n"
        "  --mtp-margin F         MTP verifier margin. Default: 3\n"
        "  -c, --ctx N            Context size. Default: 100000\n"
        "  -n, --tokens N         Max generated tokens per turn. Default: 50000\n"
        "  -p, --prompt TEXT      Submit an initial prompt after startup.\n"
        "  --non-interactive      Run without the TUI. With -p: one turn and exit;\n"
        "                         without -p: read repeated prompts from stdin.\n"
        "  -sys, --system TEXT    Extra system prompt. Empty disables extra text.\n"
        "  --trace FILE           Write prompt, token, and DSML debug trace.\n"
        "  --temp F               Sampling temperature. Default: 1\n"
        "  --top-p F              Nucleus sampling probability. Default: 1\n"
        "  --min-p F              Min-p sampling threshold. Default: 0.05\n"
        "  --seed N               Sampling seed.\n"
        "  --think                Use normal thinking mode.\n"
        "  --think-max            Use Think Max when context is large enough.\n"
        "  --nothink              Disable thinking. Default.\n"
        "  --backend NAME         metal, cuda, or cpu.\n"
        "  --metal, --cuda, --cpu Select backend explicitly.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --chdir DIR            Change working directory before loading runtime assets.\n"
        "  --quality              Prefer exact kernels where available.\n"
        "  --warm-weights         Touch mapped tensor pages before generation.\n"
        "  --power N              Target GPU duty cycle percentage, 1..100. Default: 100\n"
        "  --dir-steering-file FILE\n"
        "  --dir-steering-ffn F\n"
        "  --dir-steering-attn F\n"
        "\n"
        "Distributed:\n");
    ds4_dist_usage(fp);
    fprintf(fp,
        "\n"
        "  -h, --help             Show this help.\n"
        "\n"
        "Commands:\n"
        "  /help                  Show runtime help.\n"
        "  /save                  Save the current agent session.\n"
        "  /compact               Compact the current session context now.\n"
        "  /list                  List saved sessions in ~/.ds4/kvcache.\n"
        "  /switch SHA            Load a saved session and show recent history.\n"
        "  /del SHA               Delete a saved session.\n"
        "  /strip SHA             Remove KV payload from a saved session.\n"
        "  /history [N]           Show N recent user turns from the current session.\n"
        "  /power N               Set GPU duty cycle percentage, 1..100.\n"
        "  /corr [name|path|off]  List gguf/sidecars/ plugins or hot-swap the z domain.\n"
        "  /new                   Start a fresh session from the system prompt.\n"
        "  /quit, /exit           Exit.\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-agent: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

agent_config parse_options(int argc, char **argv) {
    agent_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
        },
        .gen = {
            .system = "You are a helpful coding assistant running inside ds4-agent.",
            .n_predict = 50000,
            .ctx_size = 100000,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .think_mode = DS4_THINK_NONE,   /* 默认不思考(2026-09-30 用户定, 与服务端/CLI 同; --think 才开) */
        },
    };

    bool steering_scale_set = false;
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
                                   &c.engine.distributed,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-agent: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "--mem-budget-mb")) {
            ds4_set_mem_budget_mb(atoi(need_arg(&i, argc, argv, arg)));
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--trace")) {
            c.gen.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--corr")) {
            c.engine.corr_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4-agent: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else {
            fprintf(stderr, "ds4-agent: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.engine.distributed,
                                        &c.engine,
                                        dist_err,
                                        sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-agent: %s\n", dist_err);
        exit(2);
    }
    if (c.engine.distributed.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-agent: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-agent: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

ds4_think_mode effective_think_mode(const agent_config *cfg) {
    return ds4_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}
