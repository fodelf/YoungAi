/* bench_internal.h — ds4-bench 模块内部头(重构阶段3, 自 ds4_bench.c 机械拆分)。
 * bench_opts.c=配置解析, bench_main.c=前沿测量与入口。 */
#ifndef DS4_BENCH_INTERNAL_H
#define DS4_BENCH_INTERNAL_H

#include <stdbool.h>
#include <stdio.h>
#include "ds4.h"

typedef struct {
    const char *model_path;
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
