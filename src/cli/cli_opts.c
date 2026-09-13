#include "ds4.h"
#include "ds4_distributed.h"
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif
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

int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4: invalid backend: %s\n", s);
    fprintf(stderr, "ds4: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

char *read_prompt_file(const char *path, bool fatal) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open prompt file: %s\n", path);
        if (fatal) exit(2);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4: failed to seek prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fprintf(stderr, "ds4: failed to size prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    rewind(fp);

    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "ds4: out of memory reading prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    size_t nread = fread(buf, 1, (size_t)len, fp);
    if (nread != (size_t)len) {
        fprintf(stderr, "ds4: failed to read prompt file: %s\n", path);
        free(buf);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close prompt file: %s\n", path);
        free(buf);
        if (fatal) exit(2);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
        },
        .gen = {
            .prompt = NULL,
            .system = "",   /* default system prompt OFF: assistant-persona system text derails
                             * base code continuation (model answers the persona instead of
                             * continuing the code). Pass -sys "..." to set one explicitly. */
            .n_predict = 50000,
            .ctx_size = DS4_DEFAULT_CTX_SIZE,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .dump_logprobs_top_k = 20,
            .think_mode = DS4_THINK_HIGH,
        },
    };

    c.dist = ds4_dist_options_create();
    if (!c.dist) {
        fprintf(stderr, "ds4: out of memory creating distributed options\n");
        exit(1);
    }

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
        ds4_dist_cli_parse_result dist_parse = ds4_dist_parse_cli_arg(arg,
                                                                      &i,
                                                                      argc,
                                                                      argv,
                                                                      c.dist,
                                                                      dist_parse_err,
                                                                      sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr, "ds4: %s\n", dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_prompt_file(need_arg(&i, argc, argv, arg), true);
            c.gen.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--corr")) {
            c.engine.corr_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--residual")) {
            c.engine.residual_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cap-dir")) {
            /* 取料入口(2026-08-22 由 DS4_CAP_DIR 迁来): 逐层捕获 x̂/路由/routed 输出 */
            ds4_tool_set_cap_dir(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--eval-ids")) {
            ds4_tool_set_eval_ids(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--eval-hdump")) {
            ds4_tool_set_eval_hdump(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--eval-logits")) {
            ds4_tool_set_eval_logits(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--eval-nll")) {
            ds4_tool_set_eval_nll(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--eval-topk")) {
            const int k = atoi(need_arg(&i, argc, argv, arg));
            ds4_tool_set_eval_topk(k, need_arg(&i, argc, argv, "--eval-topk <K> <out>"));
        } else if (!strcmp(arg, "--eval-no-bos")) {
            ds4_tool_set_eval_no_bos(1);
        } else if (!strcmp(arg, "--cap-layers")) {
            ds4_tool_set_cap_layers(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--amp-anchor")) {
            ds4_tool_set_amp_anchor(need_arg(&i, argc, argv, arg),
                                    ds4_tool_amp_anchor_route());
        } else if (!strcmp(arg, "--amp-anchor-route")) {
            ds4_tool_set_amp_anchor(ds4_tool_amp_anchor(), 1);
        } else if (!strcmp(arg, "--multi-bench")) {
            ds4_tool_set_multi_bench(parse_int(need_arg(&i, argc, argv, arg), arg));
        } else if (!strcmp(arg, "--prefill-chunk")) {
            ds4_tool_set_prefill_chunk(atoi(need_arg(&i, argc, argv, arg)));
        } else if (!strcmp(arg, "--mem-budget-mb")) {
            ds4_set_mem_budget_mb(parse_int(need_arg(&i, argc, argv, arg), arg));
        } else if (!strcmp(arg, "--spec")) {
            c.engine.spec = true;
        } else if (!strcmp(arg, "--draft-gguf")) {
            c.engine.draft_gguf_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--draft-zchain")) {
            c.engine.draft_zchain_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vq-dir")) {
            c.engine.vq_dir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--zchain")) {
            c.engine.zchain_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--zchain-scale")) {
            /* 全局 setter(同 --mem-budget-mb): 只有 V4.1 放大器目录形态消费它, 不进 ds4_engine_opts */
            ds4_engine_v41_set_amp_scale((float)atof(need_arg(&i, argc, argv, arg)));
        } else if (!strcmp(arg, "--finetune")) {
            c.engine.finetune_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.gen.dump_tokens = true;
        } else if (!strcmp(arg, "--classify")) {
            c.gen.classify_only = true;
        } else if (!strcmp(arg, "--route")) {
            c.gen.route = true;
        } else if (!strcmp(arg, "--route-prog")) {
            c.gen.route_prog = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--route-daily")) {
            c.gen.route_daily = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-logits")) {
            c.gen.dump_logits_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--score-ids")) {
            c.gen.score_ids_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--score-out")) {
            c.gen.score_out_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--v41-no-engram")) {
            c.gen.v41_no_engram = 1;
        } else if (!strcmp(arg, "--v41-chunk")) {
            c.gen.v41_chunk = atoi(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--v41-prof")) {
            c.gen.v41_prof = 1;
        } else if (!strcmp(arg, "--dump-logprobs")) {
            c.gen.dump_logprobs_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.gen.dump_logprobs_top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--perplexity-file")) {
            c.gen.perplexity_file_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-dataset")) {
            c.gen.imatrix_dataset_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-out")) {
            c.gen.imatrix_output_path = need_arg(&i, argc, argv, arg);
            /* 后端不在这里强改: default_backend() 已经是 Mac=Metal / Linux=CUDA,
             * 旧代码硬写 METAL 是 Mac 独占时代的遗留, 在 CUDA 构建上会把用户显式
             * 传的 --cuda 覆盖掉直接启动失败(2026-08-21 实锤)。 */
        } else if (!strcmp(arg, "--imatrix-max-prompts")) {
            c.gen.imatrix_max_prompts = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-max-tokens")) {
            c.gen.imatrix_max_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--head-test")) {
            c.gen.head_test = true;
        } else if (!strcmp(arg, "--first-token-test")) {
            c.gen.first_token_test = true;
        } else if (!strcmp(arg, "--metal-graph-test")) {
            c.gen.metal_graph_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-full-test")) {
            c.gen.metal_graph_full_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-prompt-test")) {
            c.gen.metal_graph_prompt_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-generate")) {
            fprintf(stderr, "ds4: --metal-graph-generate was removed; --metal is the graph path\n");
            exit(2);
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
#ifndef DS4_NO_GPU
        } else if (!strcmp(arg, "--no-residency")) {
            ds4_gpu_set_no_residency(1);
        } else if (!strcmp(arg, "--strict-fp")) {
            ds4_gpu_set_strict_fp(1);
        } else if (!strcmp(arg, "--expert-pool-mb")) {
            expert_pool_mb = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expert-pool-pinned")) {
            expert_pool_pinned = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-pool-auto-pin-top")) {
            expert_pool_auto_pin_top = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expert-pool-prefetch-top")) {
            expert_pool_prefetch_top = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--expert-pin-file")) {
            expert_pin_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-pin-mlock-mb")) {
            expert_pin_mlock_mb = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--resid-pin-mlock-mb")) {
            resid_pin_mlock_mb = parse_u64(need_arg(&i, argc, argv, arg), arg);
#endif
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--server")) {
            fprintf(stderr, "ds4: use ds4-server for the HTTP server\n");
            exit(2);
        } else {
            fprintf(stderr, "ds4: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

#ifndef DS4_NO_GPU
    ds4_gpu_set_expert_pool(expert_pool_mb, expert_pool_pinned,
                            expert_pool_auto_pin_top, expert_pool_prefetch_top);
    ds4_gpu_set_expert_pin(expert_pin_file, expert_pin_mlock_mb, resid_pin_mlock_mb);
#endif
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if (c.gen.imatrix_output_path && !c.gen.imatrix_dataset_path) {
        fprintf(stderr, "ds4: --imatrix-out requires --imatrix-dataset\n");
        exit(2);
    }
    if (c.gen.imatrix_dataset_path && !c.gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --imatrix-dataset requires --imatrix-out\n");
        exit(2);
    }
    if (c.gen.perplexity_file_path && c.gen.prompt) {
        fprintf(stderr, "ds4: --perplexity-file does not use -p/--prompt-file\n");
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(c.dist, &c.engine, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4: %s\n", dist_err);
        exit(2);
    }

    return c;
}
