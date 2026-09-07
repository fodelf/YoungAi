/* bench_internal.h — ds4-bench 模块内部头(重构阶段3, 自 ds4_bench.c 机械拆分)。
 * bench_opts.c=配置解析, bench_main.c=前沿测量与入口。 */
#ifndef DS4_BENCH_INTERNAL_H
#define DS4_BENCH_INTERNAL_H

#include <stdbool.h>
#include <stdio.h>
#include "ds4.h"

typedef struct {
    const char *model_path;
    /* 与 ds4 CLI 同义的两个侧车入口: 速度尺必须能量"骨架 + VQ 层目录 + 放大器链"这一
     * 部署形态, 否则冠军态(从未合并成单文件 GGUF)在 ds4-bench 上根本挂不起来。 */
    const char *zchain_path;
    const char *vq_dir_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int gen_final_only;   /* 只在最后前沿解码且不快照(1M 尺: 快照=整段 KV 主机拷贝, 放不下) */
    int fill_ctx;         /* 合成上下文: 先把前 N 个 token 摆成"已处理"(内容零), 前沿从 N 之后起量 */
    int spec;             /* --spec: 生成段走 DSpark 投机(与 ds4 CLI 同一条 ds4_session_eval_speculative_argmax 路), 量投机在长上下文的真速度 */
    const char *draft_gguf_path;   /* --draft-gguf: 独立 drafter GGUF, 同 ds4 */
    int power_percent;
    double step_mul;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    bool warm_weights;
    bool quality;
} bench_config;

/* bench_opts.c */
double bench_now_sec(void);
bench_config parse_options(int argc, char **argv);
char *read_file(const char *path);

#endif /* DS4_BENCH_INTERNAL_H */
