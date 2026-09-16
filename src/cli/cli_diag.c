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


/* teacher-forced 逐位打分(2026-08-14 公开对拍): 读 ids 文件(空白分隔 token id),
 * 首 token prefill 后逐位: 导出全词表 raw logits → 强制喂真值下一 token。
 * 输出=量化器 DS4_DUMP_LOGITS 同构二进制(int32 S,V + fp32[S*V]) → anchor_metrics 直接对表。 */
/* V4.1 贪心生成(P2c 冒烟出口): 逐 token 打印文本, 采样/会话/服务接入是 P5 */
static int v41_emit_print(int token, void *ud) {
    ds4_engine *engine = (ds4_engine *)ud;
    if (token == ds4_token_eos(engine)) return 1;
    size_t len = 0; char *txt = ds4_token_text(engine, token, &len);
    if (txt) { fwrite(txt, 1, len, stdout); fflush(stdout); }
    return 0;
}
int run_v41_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    if (cfg->gen.temperature > 0.0f) fprintf(stderr, "ds4: V4.1 当前只有贪心解码(temp 0), 忽略 --temp %.2f\n", (double)cfg->gen.temperature);
    ds4_engine_v41_set_prof(cfg->gen.v41_prof);
    ds4_engine_v41_set_decoder_full(cfg->gen.decoder_full);
    ds4_engine_v41_set_dspark(!cfg->gen.no_dspark);
    ds4_engine_v41_set_draft_amp(cfg->gen.draft_amp);
    ds4_engine_v41_set_chunk(cfg->gen.v41_chunk);
    int rc = ds4_engine_v41_generate_argmax(engine, prompt->v, (int)prompt->len, cfg->gen.n_predict, cfg->gen.ctx_size, v41_emit_print, engine);
    fputc('\n', stdout);
    return rc;
}

int run_score_ids(ds4_engine *engine, const cli_config *cfg) {
    FILE *fi = fopen(cfg->gen.score_ids_path, "r");
    if (!fi) { fprintf(stderr, "ds4: --score-ids 打不开 %s\n", cfg->gen.score_ids_path); return 1; }
    int cap = 8192, n = 0, t;
    int *ids = malloc((size_t)cap * sizeof(int));
    while (fscanf(fi, "%d", &t) == 1) {
        if (n >= cap) { cap *= 2; ids = realloc(ids, (size_t)cap * sizeof(int)); }
        ids[n++] = t;
    }
    fclose(fi);
    if (n < 2) { fprintf(stderr, "ds4: score-ids 少于 2 token\n"); free(ids); return 1; }
    /* ★取料模式(mtp.md M6)★: 同一份 ids, 改成"一位一块 + 每位跑一轮草稿器", 出 (草稿隐态, 主模型隐态) 对
     * 与首位一致率。必须配 --decoder-full —— 块 1 时 CED 会让非末块不出 logits, 靶直接是错的(不报错)。 */
    if (cfg->gen.dcap_path) {
        /* ★这条路不经过 run_v41_generation, 所以那边设的几个开关这里要自己设一遍★
         * (2026-09-16 实撞: --draft-amp 在取料路上静默没生效, 判决数一模一样, 看着像"方法没用"。) */
        ds4_engine_v41_set_decoder_full(cfg->gen.decoder_full);
        ds4_engine_v41_set_draft_amp(cfg->gen.draft_amp);
        ds4_engine_v41_set_draft_amp_scale(cfg->gen.draft_amp_scale > 0.f ? cfg->gen.draft_amp_scale : 1.0f);
        ds4_engine_v41_set_prof(cfg->gen.v41_prof);
        if (!cfg->gen.decoder_full)
            fprintf(stderr, "ds4: ★--dspark-capture 必须配 --decoder-full★(否则 CED 让每块不出 logits, 靶是错的)\n");
        const int rc = ds4_engine_v41_dspark_capture(engine, ids, n, cfg->gen.dcap_path);
        free(ids);
        return rc;
    }
    /* V4.1(2026-09-12): 分块增量前向出全位置 logits(同构输出, anchor_metrics 直接对表) */
    if (ds4_engine_is_v41(engine)) {
        ds4_engine_v41_set_prof(cfg->gen.v41_prof);
        ds4_engine_v41_set_score_aux(cfg->gen.score_nll_path, cfg->gen.score_topk_path,
                                     cfg->gen.score_topk, cfg->gen.score_rms_path, cfg->gen.score_no_logits);
        int rc = ds4_engine_v41_score_ids(engine, ids, n, cfg->gen.score_out_path ? cfg->gen.score_out_path : "/tmp/ds4_score.bin",
                                          cfg->gen.v41_no_engram, cfg->gen.v41_chunk);
        free(ids); return rc;
    }
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --score-ids 需要图会话后端\n"); free(ids); return 1;
    }
    char err[160];
    /* 首 token 走正常 S=1 sync 预填(与 --dump-logits 同路)。旧"CUDA 挂死绕道"(空会话
     * eval 单步)已撤: ①当年的"sync 挂死"实为 timeout 进程组 SIGTTIN 停机误诊;
     * ②空会话 pos=0 的 decode 反而是双后端都没验证过的边角, CUDA 上实测
     * "cuda decode failed"(与模型无关, v2/allq2 同挂)。 */
    ds4_tokens first = { .v = ids, .len = 1, .cap = 1 };
    if (ds4_session_sync(session, &first, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: 首 token 预填失败: %s\n", err);
        ds4_session_free(session); free(ids); return 1;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(float));
    FILE *fo = fopen(cfg->gen.score_out_path ? cfg->gen.score_out_path : "/tmp/ds4_score.bin", "wb");
    if (!fo || !logits) { fprintf(stderr, "ds4: score 输出打不开\n"); ds4_session_free(session); free(ids); return 1; }
    int hd[2] = { n, vocab };
    fwrite(hd, 4, 2, fo);
    for (int i = 1; i <= n; i++) {
        if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
            fprintf(stderr, "ds4: 位置 %d 取 logits 失败\n", i - 1); break;
        }
        fwrite(logits, 4, (size_t)vocab, fo);
        if (i == n) break;
        if (ds4_session_eval(session, ids[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: 位置 %d 强制喂入失败: %s\n", i, err); break;
        }
        if (i % 128 == 0) fprintf(stderr, "[score] %d/%d\n", i, n);
    }
    fclose(fo); free(logits); free(ids);
    ds4_session_free(session);
    fprintf(stderr, "[score] 完成 S=%d V=%d → %s\n", n, vocab,
            cfg->gen.score_out_path ? cfg->gen.score_out_path : "/tmp/ds4_score.bin");
    return 0;
}

int run_logits_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logits requires a graph session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        ds4_session_free(session);
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4: failed to copy session logits\n");
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    FILE *fp = fopen(cfg->gen.dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"model\":");
    json_write_string(fp, cfg->engine.model_path, strlen(cfg->engine.model_path));
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"vocab\":%d,\n",
            ds4_backend_name(cfg->engine.backend),
            ds4_engine_routed_quant_bits(engine),
            prompt->len,
            cfg->gen.ctx_size,
            vocab);
    const int argmax = ds4_session_argmax(session);
    fputs("  \"argmax_token\":", fp);
    json_write_token(fp, engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) {
            fprintf(fp, "%.9g", logits[i]);
        } else {
            fputs("null", fp);
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    free(logits);
    ds4_session_free(session);
    return 0;
}

int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logprobs requires a graph session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    /* 贪心 argmax 续写: 钉生成区边界(anticycle 不再扫 prompt), 贪心路径投机门=1。 */
    ds4_session_mark_generation_start(session);
    ds4_session_set_spec_greedy(session, 1);

    FILE *fp = fopen(cfg->gen.dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        ds4_session_free(session);
        return 1;
    }

    int k = cfg->gen.dump_logprobs_top_k > 0 ? cfg->gen.dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_token_score *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n",
            prompt->len, cfg->gen.ctx_size, k);
    int generated = 0;
    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    for (; generated < max_tokens; generated++) {
        int n = ds4_session_top_logprobs(session, scores, k);
        int token = ds4_session_argmax(session);
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        json_write_token(fp, engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n && scores[i].id >= 0; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            json_write_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed while dumping logprobs: %s\n", err);
            free(scores);
            fclose(fp);
            ds4_session_free(session);
            return 1;
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        free(scores);
        ds4_session_free(session);
        return 1;
    }
    free(scores);
    ds4_session_free(session);
    return 0;
}

int run_perplexity_file(ds4_engine *engine, const cli_config *cfg) {
    char *text = read_prompt_file(cfg->gen.perplexity_file_path, true);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);

    /* Seed the graph with enough real context to stay on the normal Metal
     * prefill path; scoring starts immediately after this fixed prefix. */
    const int prefix_len = 32;
    if (tokens.len <= prefix_len) {
        fprintf(stderr, "ds4: --perplexity-file needs more than %d tokens\n", prefix_len);
        ds4_tokens_free(&tokens);
        return 1;
    }

    int scored = tokens.len - prefix_len;
    if (cfg->gen.n_predict > 0 && scored > cfg->gen.n_predict) scored = cfg->gen.n_predict;
    if (scored > cfg->gen.ctx_size - prefix_len) scored = cfg->gen.ctx_size - prefix_len;
    if (scored <= 0) {
        fprintf(stderr, "ds4: context too small for perplexity scoring\n");
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --perplexity-file requires a graph session backend\n");
        ds4_tokens_free(&tokens);
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_tokens prefix = {0};
    for (int i = 0; i < prefix_len; i++) ds4_tokens_push(&prefix, tokens.v[i]);
    char err[160];
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: perplexity initial token failed: %s\n", err);
        ds4_tokens_free(&prefix);
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }
    ds4_tokens_free(&prefix);

    double nll = 0.0;
    for (int j = 0; j < scored; j++) {
        const int i = prefix_len + j;
        ds4_token_score score;
        if (!ds4_session_token_logprob(session, tokens.v[i], &score)) {
            fprintf(stderr, "ds4: failed to score token %d\n", i);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
        nll -= (double)score.logprob;

        if (((j + 1) % 256) == 0 || j + 1 == scored) {
            fprintf(stderr, "ds4: perplexity scored %d/%d\r", j + 1, scored);
            fflush(stderr);
        }

        if (j + 1 < scored && ds4_session_eval(session, tokens.v[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "\nds4: perplexity decode failed at token %d: %s\n", i, err);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
    }
    fputc('\n', stderr);

    const double avg_nll = nll / (double)scored;
    printf("tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f\n",
           tokens.len, scored, nll, avg_nll, exp(avg_nll));

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return 0;
}
