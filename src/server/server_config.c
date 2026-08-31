/* server_config.c — 机械拆分自 ds4_server.c (12787-13152 行): 命令行解析与 usage。 */

#include "server_internal.h"

/* 服务端默认输出 token 上限(384K)。与 DS4_THINK_MAX_MIN_CONTEXT 同数是巧合
 * (一个是输出上限, 一个是 think-max 的最小 ctx), 语义独立, 别合并。 */
#define SERVER_DEFAULT_MAX_TOKENS 393216
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonneg_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static float parse_float_arg(const char *s, const char *opt, float minv, float maxv) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || v < minv || v > maxv) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: missing value for %s", opt);
        exit(2);
    }
    return argv[++(*i)];
}

void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    server_log(DS4_LOG_DEFAULT,
               "ds4-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)",
               (double)m.total_bytes / (1024.0 * 1024.0),
               ctx_size,
               ds4_backend_name(backend),
               m.prefill_cap,
               m.raw_cap,
               m.comp_cap);
}

void server_close_resources(server *s) {
    if (s->trace) {
        fclose(s->trace);
        s->trace = NULL;
    }
    kv_cache_close(&s->kv);
    tool_memory_free(&s->tool_mem);
    live_tool_state_free(&s->responses_live);
    live_tool_state_free(&s->anthropic_live);
    visible_live_free(&s->thinking_live);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    ds4_session_free(s->session);
    ds4_engine_close(s->engine);
    memset(s, 0, sizeof(*s));
}

/* 跨文件全局 (声明在 server_types2.h): 渲染/引导层没有 cfg 可传, 走全局。 */
int g_base_native = 0;
bool g_primer_compact = false;

void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-server [options]\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: ds4flash.gguf\n"
        "  -c, --ctx N\n"
        "      Context size allocated at startup. Default: " DS4_STRINGIFY(DS4_DEFAULT_CTX_SIZE) "\n"
        "  -n, --tokens N\n"
        "      Default max output tokens when the client omits a limit. Default: "
        DS4_STRINGIFY(SERVER_DEFAULT_MAX_TOKENS) " (384K)\n"
        "  --max-output-tokens N\n"
        "      Hard server-side cap on output tokens per request, overriding larger client limits.\n"
        "      0 disables; protects a single-worker local server from runaway generations. Default: 0\n"
        "  --nothink\n"
        "      Force non-thinking mode for every request, ignoring client thinking configs.\n"
        "      For served base models without think training.\n"
        "  --tool-primer\n"
        "      Seed tool-enabled turns with the DSML tool-call opener (base/continuation\n"
        "      models act by continuing a prefix, not by following instructions).\n"
        "  --residual FILE\n"
        "      1-bit residual expert sidecar GGUF layered over the base quant.\n"
        "  --vq-dir DIR\n"
        "      VQ codebook sidecar directory (takes precedence over --residual).\n"
        "  --spec\n"
        "      DSpark speculative decoding + online scheduler (greedy-lossless).\n"
        "  --draft-gguf FILE | --draft-zchain FILE\n"
        "      Standalone DSpark drafter GGUF and its amplifier sidecar.\n"
        "  --mm-image-cmd CMD\n"
        "      External multimodal image encoder command (default: probe ./mm-ui).\n"
        "  --mem-budget-mb N\n"
        "      Arm the memory guardrails (watchdog 90%% abort, L1 85%% load gate,\n"
        "      expert resident/stream AUTO verdict). Unset = disarmed.\n"
        "  --prefill-chunk N\n"
        "      Prefill batch chunk cap in tokens (0 = whole prompt as one batch).\n"
        "  --batch N\n"
        "      Merge up to N concurrent non-streaming tool-free chat requests into one\n"
        "      batched decode (0 disables; max 8). Default: 0\n"
        "  --base-native\n"
        "      Render chat as base-model native scaffolding (# User:/# Assistant:)\n"
        "      instead of DSML role frames; for served base models.\n"
        "  --primer-compact\n"
        "      Tool-primer injects only semantic anchors into the KV (client-visible\n"
        "      text remains full DSML).\n"
        "  -t, --threads N\n"
        "      CPU helper threads for lightweight host-side work.\n"
        "  --chdir DIR\n"
        "      Change working directory before loading the model or runtime assets.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; MTP uses strict verification.\n"
        "  --dir-steering-file FILE\n"
        "      Load one f32 direction vector per layer for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before serving. Slower startup, fewer first-use stalls.\n"
        "  --power N\n"
        "      Target GPU duty cycle percentage, 1..100. Default: 100\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS and CUDA on CUDA builds.\n"
        "  --strict-fp\n"
        "      Strict IEEE-754 shader math (safe math + f32 raw KV + exp2/log2 RoPE)\n"
        "      for cross-GPU parity lanes. Metal only.\n"
        "  --expert-pool-mb N | --expert-pool-pinned SPEC | --expert-pool-auto-pin-top N | --expert-pool-prefetch-top N\n"
        "      Resident routed-expert LRU pool: MiB budget (0 = off), pin whitelist\n"
        "      (\"L20:1,2;L21:7\"), auto-pin top-N, prefetch margin. Metal only.\n"
        "  --expert-pin-file FILE | --expert-pin-mlock-mb N | --resid-pin-mlock-mb N\n"
        "      Frequency hot-expert mlock pins: pin list file, wired budget, and the\n"
        "      residual sidecar's wired budget (0 = off). Metal only.\n"
        "\n"
        "HTTP API:\n"
        "  --host HOST\n"
        "      Bind address. Default: 127.0.0.1\n"
        "  --port N\n"
        "      Bind port. Default: 8000\n"
        "  --cors\n"
        "      Add Access-Control-Allow-* headers for browser JS clients. Does not change --host.\n"
        "  --trace FILE\n"
        "      Write a human-readable session trace: prompts, cache decisions, output, tool calls.\n"
        "\n"
        "Thinking and sampling:\n"
        "  DeepSeek-compatible chat requests default to thinking mode with high effort.\n"
        "  Only reasoning_effort=max or output_config.effort=max requests Think Max.\n"
        "  Think Max is applied only when --ctx is at least " DS4_STRINGIFY(DS4_THINK_MAX_MIN_CONTEXT)
        " tokens; smaller contexts use high.\n"
        "  thinking={type:disabled}, think=false, or model=deepseek-chat selects non-thinking mode.\n"
        "  API defaults are temperature=1, top_p=1, min_p=0.05, and no top-k cap.\n"
        "  In thinking mode, client sampling knobs are ignored like the official API.\n"
        "\n"
        "Disk KV cache:\n"
        "  --kv-disk-dir DIR\n"
        "      Enable disk KV checkpoints in DIR. The directory is created if needed.\n"
        "  --kv-disk-space-mb N\n"
        "      Disk budget for checkpoint files. Default when enabled: 4096\n"
        "  --kv-cache-min-tokens N\n"
        "      Do not save or load checkpoints shorter than N tokens. Default: 512\n"
        "  --kv-cache-cold-max-tokens N\n"
        "      Cold first prompts in [min,N] are saved automatically. 0 disables cold saves. Default: 30000\n"
        "  --kv-cache-continued-interval-tokens N\n"
        "      Save at absolute aligned frontiers spaced about N tokens apart. 0 disables. Default: 10000\n"
        "  --kv-cache-boundary-trim-tokens N\n"
        "      Trim this many tail tokens before cold boundary saves to avoid tokenizer boundary merges. Default: 32\n"
        "  --kv-cache-boundary-align-tokens N\n"
        "      Align cold boundary saves down to this token multiple. 0 disables alignment. Default: 2048\n"
        "  --kv-cache-reject-different-quant\n"
        "      Refuse checkpoints written by the same model with a different routed-expert quantization.\n"
        "  --disable-exact-dsml-tool-replay\n"
        "      Disable the tool-id -> exact sampled DSML map. Tool history falls back to canonical JSON rendering.\n"
        "  --tool-memory-max-ids N\n"
        "      Maximum exact tool-call IDs kept in RAM for replay. Default: 100000\n"
        "\n"
        "  Cache triggers:\n"
        "      cold       save a stable prefix of a long first prompt before generation starts\n"
        "      continued  save absolute aligned restart frontiers during long prefill or generation\n"
        "      evict      save the live conversation before another request replaces it\n"
        "      shutdown   save the live conversation when the server exits cleanly\n"
        "\n"
        "Normal server command:\n"
        "  ./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192\n"
        "\n"
        "Notes:\n"
        "  Use /v1/chat/completions, /v1/responses, /v1/completions, or /v1/messages.\n"
        "  GET / serves a browser chat page from web/chat.html (same origin, no --cors needed).\n"
        "  Larger --ctx values allocate more KV memory at startup; the startup log prints the estimate.\n"
        "  Disk KV caching is best for agents that resend long prompts with stable prefixes.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
    fprintf(fp, "\nDistributed inference:\n");
    ds4_dist_usage(fp);
}

static ds4_backend parse_backend_arg(const char *s, const char *arg) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    server_log(DS4_LOG_DEFAULT, "ds4-server: invalid %s value: %s", arg, s);
    server_log(DS4_LOG_DEFAULT, "ds4-server: valid server backends are: metal, cuda, cpu");
    exit(2);
}

static ds4_backend default_server_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

server_config parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_server_backend(),
        },
        .host = "127.0.0.1",
        .port = 8000,
        .ctx_size = DS4_DEFAULT_CTX_SIZE,
        .default_tokens = SERVER_DEFAULT_MAX_TOKENS,
        .max_output_tokens = 0,
        .force_nothink = false,
        .tool_primer = false,
        .tool_memory_max_ids = DS4_TOOL_MEMORY_DEFAULT_MAX_IDS,
    };
    c.kv_cache = kv_cache_default_options();

#ifndef DS4_NO_GPU
    /* GPU 侧成组 setter 的累积量: 解析完一次性下发(池 setter 一次收全四项)。 */
    uint64_t expert_pool_mb = 0;
    const char *expert_pool_pinned = NULL;
    uint32_t expert_pool_auto_pin_top = 0;
    uint32_t expert_pool_prefetch_top = 0;
    const char *expert_pin_file = NULL;
    uint64_t expert_pin_mlock_mb = 0;
    uint64_t resid_pin_mlock_mb = 0;
#endif
    bool directional_steering_scale_set = false;
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
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s",
                       dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--corr")) {
            c.engine.corr_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--zchain")) {
            c.engine.zchain_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--residual")) {
            c.engine.residual_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vq-dir")) {
            c.engine.vq_dir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--spec")) {
            c.engine.spec = true;
        } else if (!strcmp(arg, "--draft-gguf")) {
            c.engine.draft_gguf_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--draft-zchain")) {
            c.engine.draft_zchain_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mm-image-cmd")) {
            c.engine.mm_image_cmd = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mem-budget-mb")) {
            ds4_set_mem_budget_mb(parse_int_arg(need_arg(&i, argc, argv, arg), arg));
        } else if (!strcmp(arg, "--prefill-chunk")) {
            ds4_tool_set_prefill_chunk(parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg));
        } else if (!strcmp(arg, "--batch")) {
            c.batch_max = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.batch_max > DS4_SERVER_BATCH_LANES) {
                fprintf(stderr, "ds4-server: --batch %d exceeds lane cap, clamped to %d\n",
                        c.batch_max, DS4_SERVER_BATCH_LANES);   /* 静默钳=用户以为开了更多路 */
                c.batch_max = DS4_SERVER_BATCH_LANES;
            }
        } else if (!strcmp(arg, "--base-native")) {
            g_base_native = 1;
        } else if (!strcmp(arg, "--primer-compact")) {
            g_primer_compact = true;
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.default_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-output-tokens")) {
            c.max_output_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--nothink")) {
            c.force_nothink = true;
        } else if (!strcmp(arg, "--tool-primer")) {
            c.tool_primer = true;
        } else if (!strcmp(arg, "--soul")) {
            const char *soul_path = need_arg(&i, argc, argv, arg);
            FILE *sf = fopen(soul_path, "rb");
            if (!sf) { fprintf(stderr, "ds4-server: cannot open --soul %s\n", soul_path); exit(1); }
            fseek(sf, 0, SEEK_END);
            long sn = ftell(sf);
            fseek(sf, 0, SEEK_SET);
            g_soul_text = xmalloc((size_t)sn + 1u);
            if (fread(g_soul_text, 1, (size_t)sn, sf) != (size_t)sn) {
                fprintf(stderr, "ds4-server: short read on --soul %s\n", soul_path); exit(1);
            }
            g_soul_text[sn] = '\0';
            fclose(sf);
        } else if (!strcmp(arg, "--knowledge")) {
            knowledge_load(need_arg(&i, argc, argv, arg));   /* 知识环检索库 (--- 分块) */
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cors")) {
            c.enable_cors = true;
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = (uint64_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-min-tokens")) {
            c.kv_cache.min_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-cold-max-tokens")) {
            c.kv_cache.cold_max_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-continued-interval-tokens")) {
            c.kv_cache.continued_interval_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-trim-tokens")) {
            c.kv_cache.boundary_trim_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-align-tokens")) {
            c.kv_cache.boundary_align_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-reject-different-quant")) {
            c.kv_cache_reject_different_quant = true;
        } else if (!strcmp(arg, "--disable-exact-dsml-tool-replay")) {
            c.disable_exact_dsml_tool_replay = true;
        } else if (!strcmp(arg, "--tool-memory-max-ids")) {
            c.tool_memory_max_ids = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.tool_memory_max_ids <= 0) {   /* 旧行为: 0 被 getter 静默兜回默认 100000 */
                fprintf(stderr, "ds4-server: --tool-memory-max-ids must be > 0 (got %d); "
                                "0 does not mean unlimited\n", c.tool_memory_max_ids);
                exit(1);
            }
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                server_log(DS4_LOG_DEFAULT, "ds4-server: --power must be between 1 and 100");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
#ifndef DS4_NO_GPU
        } else if (!strcmp(arg, "--strict-fp")) {
            ds4_gpu_set_strict_fp(1);
        } else if (!strcmp(arg, "--expert-pool-mb")) {
            expert_pool_mb = (uint64_t)parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expert-pool-pinned")) {
            expert_pool_pinned = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-pool-auto-pin-top")) {
            expert_pool_auto_pin_top = (uint32_t)parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expert-pool-prefetch-top")) {
            expert_pool_prefetch_top = (uint32_t)parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expert-pin-file")) {
            expert_pin_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-pin-mlock-mb")) {
            expert_pin_mlock_mb = (uint64_t)parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--resid-pin-mlock-mb")) {
            resid_pin_mlock_mb = (uint64_t)parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
#endif
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else {
            server_log(DS4_LOG_DEFAULT, "ds4-server: unknown option: %s", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (c.kv_cache.cold_max_tokens > 0 &&
        c.kv_cache.cold_max_tokens < c.kv_cache.min_tokens)
    {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens");
        exit(2);
    }
#ifndef DS4_NO_GPU
    ds4_gpu_set_expert_pool(expert_pool_mb, expert_pool_pinned,
                            expert_pool_auto_pin_top, expert_pool_prefetch_top);
    ds4_gpu_set_expert_pin(expert_pin_file, expert_pin_mlock_mb, resid_pin_mlock_mb);
#endif
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.engine.distributed,
                                        &c.engine,
                                        dist_err,
                                        sizeof(dist_err)) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: %s", dist_err);
        exit(2);
    }
    return c;
}
