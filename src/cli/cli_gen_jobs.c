/* cli_gen_jobs.c — --gen-jobs 清单: 合批出题(2026-10-09, kdgen 用)。
 *
 * 为什么要它: kdgen 以前起服务、curl 一份一份串行要(服务端并发调度器在"采样 + 长提示"下首 token 就 decode failed, 10-01 起只敢 1 路),
 * 0009 那趟 161 份 5784 s(单流 ~40 t/s)。这里走训练器同一条合批出口(ds4_v41_gen_run, 随空随补): 8 路一起解码, 哪路写完立刻补下一份。
 * 渲染 = 部署同一个函数 ds4_encode_chat_prompt(无 system、不思考), 与服务端 model=deepseek-chat 同一串 token(门: 提示 token 数逐份对服务端 usage)。
 * 采样面 = 命令行(不给 --temp 就是模型卡默认 温 1 / top_p 1 / min_p 0, 与服务端缺省同); 种子按份给 ⇒ 同清单同二进制可复现(服务端是按时钟)。
 *
 * 清单每行: <提示文件>\t<输出文件>\t<种子>\t<上限 token(含 EOS 那一位)>; 提示文件内容 = user 消息原文。
 * 输出文件已存在且非空就跳过(断点续跑)。以 EOS 收口才落盘(先写 .tmp 再改名), 到上限没收口 = 失败、不落盘(与服务端 finish_reason != stop 同口径)。
 * 用法: ./ds4 --cuda -m <gguf> --zchain <dir> --no-dspark --gen-jobs <清单>   (投机在合批纯解码路上不用, 关掉省草稿塔内存) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ds4.h"
#include "cli_internal.h"

typedef struct { char *out; } gj_meta;
typedef struct { gj_meta *m; uint32_t n, done, fail; uint64_t tok; double t0; } gj_ctx;

static double gj_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + ts.tv_nsec * 1e-9; }

static void gj_done(void *ud, uint32_t job, const char *text, size_t len, int eos, uint32_t ntok) {
    gj_ctx *c = (gj_ctx *)ud;
    const char *out = c->m[job].out;
    c->done++; c->tok += ntok;
    if (!eos) { c->fail++; fprintf(stderr, "★生成失败 %s: %u token 到上限没收口★\n", out, ntok); }
    else {
        char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp", out);
        FILE *f = fopen(tmp, "wb");
        if (!f || fwrite(text, 1, len, f) != len || fclose(f) != 0 || rename(tmp, out) != 0) {
            c->fail++; fprintf(stderr, "★写不了 %s★\n", out); if (f) remove(tmp);
        }
    }
    const double dt = gj_now() - c->t0;
    fprintf(stderr, "ds4: [gen-jobs] %u/%u 份 %s %u token, 累计 %.0f s %.1f t/s, 失败 %u\n", c->done, c->n, eos ? "收口" : "截断", ntok, dt, dt > 0 ? (double)c->tok / dt : 0.0, c->fail);
}

int run_gen_jobs(ds4_engine *engine, const cli_config *cfg) {
    if (!ds4_engine_is_v41(engine)) { fprintf(stderr, "ds4: --gen-jobs 只接 V4.1 路\n"); return 1; }
    FILE *fl = fopen(cfg->gen.gen_jobs_path, "r");
    if (!fl) { fprintf(stderr, "ds4: --gen-jobs 打不开 %s\n", cfg->gen.gen_jobs_path); return 1; }
    ds4_engine_v41_set_prof(cfg->gen.v41_prof);
    ds4_engine_v41_set_decoder_full(cfg->gen.decoder_full);
    ds4_engine_v41_set_dspark(cfg->gen.no_dspark ? 0 : (cfg->gen.dspark ? 2 : 1));
    ds4_engine_v41_set_graph(!cfg->gen.no_graph);
    uint32_t cap = 64, n = 0, skip = 0;
    ds4_v41_gen_job *jobs = calloc(cap, sizeof *jobs);
    ds4_tokens *tk = calloc(cap, sizeof *tk);
    gj_meta *meta = calloc(cap, sizeof *meta);
    char line[8192];
    int rc = 0;
    while (fgets(line, sizeof line, fl)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0] || line[0] == '#') continue;
        char *pf = strtok(line, "\t"), *of = strtok(NULL, "\t"), *sd = strtok(NULL, "\t"), *mx = strtok(NULL, "\t");
        if (!pf || !of || !sd || !mx) { fprintf(stderr, "ds4: --gen-jobs 行要 4 列(提示\\t输出\\t种子\\t上限): %s\n", line); rc = 1; break; }
        FILE *ex = fopen(of, "rb");
        if (ex) { fseek(ex, 0, SEEK_END); const long sz = ftell(ex); fclose(ex); if (sz > 0) { skip++; continue; } }
        char *text = read_prompt_file(pf, false);
        if (!text) { fprintf(stderr, "ds4: --gen-jobs 读不了提示 %s\n", pf); rc = 1; break; }
        if (n == cap) {
            cap *= 2;
            jobs = realloc(jobs, cap * sizeof *jobs); tk = realloc(tk, cap * sizeof *tk); meta = realloc(meta, cap * sizeof *meta);
            memset(tk + n, 0, (cap - n) * sizeof *tk);
        }
        ds4_encode_chat_prompt(engine, NULL, text, DS4_THINK_NONE, &tk[n]);
        free(text);
        meta[n].out = strdup(of);
        jobs[n] = (ds4_v41_gen_job){ .ids = tk[n].v, .len = (uint32_t)tk[n].len, .max_tok = (uint32_t)strtoul(mx, NULL, 10),
            .sp = { .temperature = cfg->gen.temperature, .top_p = cfg->gen.top_p, .min_p = cfg->gen.min_p, .top_k = 0,
                    .seed = strtoull(sd, NULL, 10) } };
        fprintf(stderr, "ds4: [gen-jobs] 作业 %u 提示 %u token → %s\n", n + 1u, jobs[n].len, of);
        n++;
    }
    fclose(fl);
    if (rc == 0) {
        fprintf(stderr, "ds4: [gen-jobs] %u 份要生成, %u 份已有产物跳过\n", n, skip);
        gj_ctx ctx = { meta, n, 0, 0, 0, gj_now() };
        /* bcap 给到 UINT32_MAX = 用满合批上限(驱动里钳到 DS4_V41_MULTI_MAX) */
        if (!ds4_v41_gen_run(engine, jobs, n, UINT32_MAX, 1, gj_done, &ctx)) rc = 1;
        fprintf(stderr, "ds4: [gen-jobs] 完 %u/%u 份, 失败 %u, %.0f s, 生成 %llu token\n", ctx.done, n, ctx.fail, gj_now() - ctx.t0, (unsigned long long)ctx.tok);
    }
    for (uint32_t i = 0; i < n; i++) { ds4_tokens_free(&tk[i]); free(meta[i].out); }
    free(jobs); free(tk); free(meta);
    return rc;
}
