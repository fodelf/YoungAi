/* cli_internal.h — ds4 CLI 模块内部头(重构阶段3, 自 ds4_cli.c 机械拆分)。
 * 只声明跨文件符号; 命名与拆分前完全一致。文件分工:
 *   cli_main.c  信号/分布式等待 + main
 *   cli_opts.c  参数解析原语 + read_prompt_file + parse_options
 *   cli_print.c usage/内存日志/think 提示/时钟/预填进度/token printer/JSON 输出
 *   cli_gen.c   构建 prompt + 采样生成主循环 + run_generation 分派
 *   cli_diag.c  诊断模式族(score-ids/logits/logprobs/perplexity)
 *   cli_repl.c  交互 REPL 与多轮会话 */
#ifndef DS4_CLI_INTERNAL_H
#define DS4_CLI_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4.h"
#include "ds4_distributed.h"

typedef struct {
    const char *prompt;
    const char *system;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool temp_given;      /* --temp 显式给过。V4.1 生成路只在显式给了 --temp 时采样: V4 的默认温度是 1.0,
                           * 但 V4.1 生成路一直是裸 argmax、所有尺脚本都靠它 ⇒ 不给 --temp 仍是 argmax(零改动) */
    float dry_multiplier; /* --dry-multiplier F: DRY 序列复读惩罚强度(0 = 关, llama.cpp 推荐 0.8); 温 0 也生效 */
    float dry_base;       /* --dry-base F: 惩罚随匹配长度指数增长的底(默认 1.75) */
    int dry_allowed_length;   /* --dry-allowed-length N: 匹配到这么长才开始罚(默认 2) */
    bool dump_tokens;
    bool classify_only;   /* --classify: print Mode P/G route for -p prompt, no model load */
    bool route;           /* --route: classify prompt, then load prog/daily model */
    const char *route_prog;   /* --route-prog: resident programming model (Mode P) */
    const char *route_daily;  /* --route-daily: full cached model (Mode G) */
    const char *dump_logits_path;
    const char *score_ids_path;   /* --score-ids: teacher-forced 逐位打分(公开对拍) */
    const char *gen_ids_path;     /* --gen-ids: 把文件里的 token id 当提示续写(真实请求复现: 文本重新分词拼不回原序列) */
    const char *score_out_path;
    float draft_amp_scale;        /* --draft-amp-scale β(默认 1.0) */
    const char *draft_amp;        /* --draft-amp FILE: 草稿器对齐边车(mtp.md M6) */
    const char *dcap_path;        /* --dspark-capture FILE: 草稿器对齐取料(mtp.md M6), 与 --score-ids 同用 */
    int v41_no_engram;           /* --v41-no-engram: V4.1 前向跳过 engram 层(与 Python --no-engram 同口径的对拍夹具) */
    int v41_chunk;               /* --v41-chunk N: V4.1 --score-ids 的分块大小(0=默认 512; 对拍夹具, 看分块与整批自洽) */
    int v41_prof;                /* --v41-prof: V4.1 每次前向打逐层毫秒(查速度用) */
    int decoder_full;            /* --decoder-full: 关 CED, 提示每块跑满 40 层(精确路, 跟 CED 对质量用) */
    int no_dspark;               /* --no-dspark: 关投机解码(09-24 起默认开, 量纯解码/跑判决尺的脚本都靠它显式关) */
    int no_graph;                /* --no-graph: 关解码整步 CUDA graph(默认开; 只作 A/B 与同轨定位, 两条路输出逐字节同) */
    int no_vq_group;             /* --no-vq-group: 验证批/草稿塔的 VQ 专家核回逐对形态(默认走分组核; 只作 A/B, 输出逐字节同) */
    int emit_trace;              /* --emit-trace: 逐 token 打 [emit] 行(同轨定位用, 不改执行路径) */
    int dspark_block;            /* --dspark-block N: 钉死草稿块长(诊断用) */
    int verify_k;                /* --dspark-verify N: 投机每轮验证几位(0=引擎默认) */
    int dspark;                  /* --dspark: 显式要投机(默认本来就开); 与采样同开时硬拒而不是悄悄走纯解码, 见 core_v41_api.c */
    const char *score_nll_path;  /* --score-nll FILE: 逐位 NLL f32[S](后训练判决尺, 4 B/位置) */
    const char *score_topk_path; /* --score-topk K FILE: 逐位 top-K (id,p)+目标 p+覆盖质量(后训练靶) */
    int score_topk;
    const char *score_rms_path;  /* --score-rms FILE: 逐位 inv=rsqrt(mean(x²)+eps) f32[S](后训练靶的单位) */
    int score_no_logits;         /* --score-no-logits: 不写全词表 logits(统一内存机器上那是 517 KB/位置) */
    int score_split;             /* --score-split P: [0,P) 照生成路 CED 预填, [P,n) 跑满解码器(部署同路; 0 = 老口径) */
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    const char *perplexity_file_path;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    ds4_engine_options engine;
    ds4_dist_options *dist;
    cli_generation_options gen;
    char *prompt_owned;
    bool inspect;
} cli_config;

typedef struct {
    int base_tokens;
    int input_tokens;
    bool use_color;
    bool finished;
} cli_prefill_progress;

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    char pending[16];
    size_t pending_len;
} token_printer;

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    int ctx_size;
    /* 对话头 = BOS 之后、第一条 user 之前的一段([<｜System｜>] [effort 前缀] [system 正文]), 整段重建, 见 cli_repl.c */
    int head_tokens;      /* 头里 BOS 之后有几个 token */
    bool head_built;      /* 建过一次(之后再改头 = 会话作废) */
    bool head_max;        /* 当前头里带的是 max 档前缀 */
    const char *system;   /* --system 文本(cfg 持有), 重建头时要重新拼 */
} repl_chat;

/* cli_main.c */
void cli_sigint_handler(int sig);
bool cli_interrupt_requested(void);
void cli_interrupt_clear(void);
bool cli_distributed_coordinator(const cli_config *cfg);
void cli_dist_busy_set(const cli_config *cfg, bool busy);
int  cli_wait_distributed_route(const cli_config *cfg, ds4_session *session);

/* cli_opts.c */
int parse_int(const char *s, const char *opt);
uint64_t parse_u64(const char *s, const char *opt);
float parse_float_range(const char *s, const char *opt, float min, float max);
ds4_backend parse_backend(const char *s);
ds4_backend default_backend(void);
cli_config parse_options(int argc, char **argv);
char *read_prompt_file(const char *path, bool fatal);

/* cli_print.c */
void usage(FILE *fp);
void log_context_memory(ds4_backend backend, int ctx_size);
ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen);
bool cli_think_max_downgraded(const cli_generation_options *gen);
void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name);
double cli_now_sec(void);
void cli_prefill_progress_cb(void *ud, const char *event, int current, int total);
void token_printer_process(token_printer *p, const char *text, size_t len, bool finish);
void token_printer_finish(token_printer *p);
void token_printer_write_text(token_printer *p, const char *text, size_t len);
void generation_done(void *ud);
void json_write_string(FILE *fp, const char *s, size_t n);
void json_write_token(FILE *fp, ds4_engine *engine, int token);

/* cli_gen.c */
bool is_rendered_chat_prompt(const char *prompt);
void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out);
int  run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt);
int  run_generation(ds4_engine *engine, const cli_config *cfg);

/* cli_diag.c */
int run_score_ids(ds4_engine *engine, const cli_config *cfg);
int run_gen_ids(ds4_engine *engine, const cli_config *cfg);
int run_v41_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt);   /* V4.1 贪心生成(cli_diag.c) */
int run_logits_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt);
int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt);
int run_perplexity_file(ds4_engine *engine, const cli_config *cfg);

/* cli_repl.c */
int run_repl(ds4_engine *engine, cli_config *cfg);

#endif /* DS4_CLI_INTERNAL_H */
