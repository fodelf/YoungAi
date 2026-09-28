#include <limits.h>

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
            /* 不设上限: 生成到 EOS 或 ctx 边界(两条生成路各自钳)。以前写 50000 —— 与服务端那个 393216
             * 同一类"谁也说不出依据"的数, 2026-09-22 一起删(用户: "这些东西都不对")。要短输出就显式 -n。 */
            .n_predict = INT_MAX,
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
        } else if (!strcmp(arg, "--weight-cache-mb")) {
            /* 设备权重缓存封顶(反修拟合/判决用, 见 ds4_gpu_core.h): 全局 setter(同 --mem-budget-mb), 不进 ds4_engine_opts */
            ds4_gpu_set_model_cache_limit_mb((uint64_t)parse_int(need_arg(&i, argc, argv, arg), arg));
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
        } else if (!strcmp(arg, "--posttrain")) {
            /* 三文件部署的第三件(V4.1 形态): 与 --zchain 同构的增益目录, 表逐元素相乘。
             * 走全局 setter(同 --zchain-scale): ds4.h 已 500 行顶格, 不进 ds4_engine_opts。 */
            ds4_engine_v41_set_posttrain_dir(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--engram-dir")) {
            ds4_engine_v41_set_engram_dir(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--score-nll")) {
            c.gen.score_nll_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--score-topk")) {
            c.gen.score_topk = atoi(need_arg(&i, argc, argv, arg));
            c.gen.score_topk_path = need_arg(&i, argc, argv, "--score-topk <K> <out>");
        } else if (!strcmp(arg, "--score-rms")) {
            /* 出口 RMSNorm 的 inv(见 core_score_aux.h)。后训练第二版靠它把"增益改动"换算成
             * "logit 差改动"; 不传就只有形状没有单位。 */
            c.gen.score_rms_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--score-no-logits")) {
            c.gen.score_no_logits = 1;
        } else if (!strcmp(arg, "--score-split")) {
            c.gen.score_split = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            /* ★上下文没有参数★(用户 2026-09-22 "不要任何写死的上下文, 上下文大小只有 1M 这一个选择"): V4.1 的边界
             * 是模型元数据 deepseek4.context_length(装载时读进 g_ds4_v41.ctx), 状态按本趟位置分配, 没有别的档。
             * 拒而不是忽略 —— 忽略 = 用户以为设了其实没设, 本仓"不报错只出错"的坑踩够了。
             * V4 会话路的 ctx_size 留默认值, 只是没有入口再改它。 */
            fprintf(stderr, "ds4: 上下文由模型元数据(deepseek4.context_length)决定, 没有 %s 这个参数\n", arg);
            exit(2);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--dry-multiplier")) {
            c.gen.dry_multiplier = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--dry-base")) {
            c.gen.dry_base = parse_float_range(need_arg(&i, argc, argv, arg), arg, 1.0f, 100.0f);
        } else if (!strcmp(arg, "--dry-allowed-length")) {
            c.gen.dry_allowed_length = parse_int(need_arg(&i, argc, argv, arg), arg);
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
        } else if (!strcmp(arg, "--gen-ids")) {
            /* 按 token id 续写: 真实请求的序列只有 id 是准的(文本重新分词拼不回去, 09-20 实撞 79 vs 75) */
            c.gen.gen_ids_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--score-ids")) {
            c.gen.score_ids_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--score-out")) {
            c.gen.score_out_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--draft-amp")) {
            c.gen.draft_amp = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--draft-amp-scale")) {
            c.gen.draft_amp_scale = (float)atof(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--dspark-capture")) {
            c.gen.dcap_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--v41-no-engram")) {
            c.gen.v41_no_engram = 1;
        } else if (!strcmp(arg, "--v41-chunk")) {
            c.gen.v41_chunk = atoi(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--decoder-full")) {
            c.gen.decoder_full = 1;
        } else if (!strcmp(arg, "--no-dspark")) {
            c.gen.no_dspark = 1;
        } else if (!strcmp(arg, "--no-graph")) {
            c.gen.no_graph = 1;
        } else if (!strcmp(arg, "--no-vq-group")) {
            c.gen.no_vq_group = 1;
        } else if (!strcmp(arg, "--dspark")) {
            c.gen.dspark = 1;
        } else if (!strcmp(arg, "--emit-trace")) {
            c.gen.emit_trace = 1;
        } else if (!strcmp(arg, "--dspark-block")) {
            c.gen.dspark_block = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--dspark-verify")) {
            c.gen.verify_k = parse_int(need_arg(&i, argc, argv, arg), arg);
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
