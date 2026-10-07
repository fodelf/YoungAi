/* cli_multi.c — 并发探针 --multi-probe N(2026-09-30, batch.md; 用户"三路并发最起码也应该一百以上")。
 *
 * 为什么要它: 服务端量并发要起服 + 三路 curl + 看日志, 一趟十几分钟, 还被预填串行墙和时间片搅着 —— 判"合批一步到底几毫秒、钱在哪"
 * 要一把秒级出数的尺(铁律: 逐函数表 → 照最大项改)。这里: 同一提示开 N 个请求态(与服务端同一条 ds4_v41_req 出口), 各自预填完,
 * 合批解码 -n 步(默认投机: 每路各出草稿, 验证行拼进一次前向; --no-dspark = 纯解码一行), 报每轮毫秒(中位)、每路 t/s、总 t/s;
 * 配 --v41-prof 时引擎每步打逐段账([multi-prof], core_v41_multi.c)。
 * 门: N 路同提示温 0 ⇒ 每轮 N 路吐出的 token 串必须相同(不同 = 状态串台), 不同就当场喊; 投机路的输出与纯解码也必须同一串(温 0)。
 * 用法: ./ds4 -m <gguf> --zchain <dir> --temp 0 -n 64 --multi-probe 3 --prompt-file <文件> [--no-dspark] [--dspark-verify K] [--v41-prof] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ds4.h"
#include "cli_internal.h"

static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + ts.tv_nsec * 1e-9; }
static int cmp_dbl(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

int run_v41_multi_probe(ds4_engine *e, const cli_config *cfg, const ds4_tokens *prompt) {
    const int N = cfg->gen.multi_probe, steps = cfg->gen.n_predict > 0 ? cfg->gen.n_predict : 64;
    if (N < 1 || N > 8) { fprintf(stderr, "ds4: --multi-probe 只收 1..8 路(解码小批核路上限)\n"); return 1; }
    ds4_engine_v41_set_prof(cfg->gen.v41_prof);
    ds4_engine_v41_set_decoder_full(cfg->gen.decoder_full);
    ds4_engine_v41_set_chunk(cfg->gen.v41_chunk);
    ds4_engine_v41_set_lanes(!cfg->gen.no_lanes);
    ds4_engine_v41_set_dspark(cfg->gen.no_dspark ? 0 : (cfg->gen.dspark ? 2 : 1));
    ds4_engine_v41_set_verify_k((unsigned)cfg->gen.verify_k);
    ds4_engine_v41_set_graph(!cfg->gen.no_graph);
    const int spec = !cfg->gen.no_dspark;
    const ds4_decode_sampling sp = {
        .temperature = cfg->gen.temperature, .top_p = cfg->gen.top_p, .min_p = cfg->gen.min_p, .top_k = 0, .seed = cfg->gen.seed,
        .dry_base = 1.75f, .dry_allowed_length = 2,
    };
    /* 批态行数 = 解码小批核路上限 8, 不是路数: 投机的验证批每路要 1+k 行(N=1 就是 6 行; N=3 时 8 行只够 k=2,2,1)。
     * ★实撞(09-30 17:23)★: 按路数开批态, N=1 时 cap=1, 草稿每轮都出(走图 62 次)却被"装不下就削 k"削成 0 —— 投机 0 轮, 每轮还白付草稿 9 ms。 */
    struct ds4_v41_batch *b = ds4_v41_batch_open(e, 8);
    if (!b) { fprintf(stderr, "ds4: 批态开不出来\n"); return 1; }
    struct ds4_v41_req *r[8] = {0};
    int rc = 1;
    double t0 = now_sec();
    for (int i = 0; i < N; i++) {
        r[i] = ds4_v41_req_open(e, prompt->v, prompt->len, steps + 8, &sp);
        if (!r[i]) { fprintf(stderr, "ds4: 第 %d 路请求态开不出来\n", i); goto out; }
        int prc;
        while ((prc = ds4_v41_req_prefill_step(r[i])) == 0) {}
        if (prc < 0) { fprintf(stderr, "ds4: 第 %d 路预填失败\n", i); goto out; }
        fprintf(stderr, "[multi] 第 %d 路预填 %d token 完, 首 token %d, 累计 %.1fs\n", i, prompt->len, ds4_v41_req_next(r[i]), now_sec() - t0);
    }
    double *ms = malloc((size_t)steps * sizeof(double));
    /* 各路吐出的 token 串(投机下各路每轮 k 不同、每轮吐出的个数不同, 门只能比整串): [路][steps+16] */
    const int seqcap = steps + 16;
    int *seq = malloc((size_t)N * (size_t)seqcap * sizeof(int)), nseq[8] = {0};
    if (!ms || !seq) { free(ms); free(seq); goto out; }
    int done = 0, mismatch = 0, ntok = 0, stop = 0;
    const int eos = ds4_token_eos(e);
    t0 = now_sec();
    /* 轮数 ≤ 步数, 且吐够 steps 个 token 就停: 投机一轮出多个, 请求态的上下文只按 steps 个位置开(spec2 N=3 实撞: 第 40 轮撞"上下文满") */
    for (int s = 0; s < steps && !stop && ntok < steps; s++) {
        const double ts = now_sec();
        const int src = spec ? ds4_v41_multi_round(b, r, N) : ds4_v41_multi_step(b, r, N);
        if (src != 0) { fprintf(stderr, "\nds4: 合批第 %d 轮失败\n", s); free(ms); free(seq); goto out; }
        ms[done++] = (now_sec() - ts) * 1e3;
        int outi[16];
        for (int i = 0; i < N; i++) {
            const int ni = ds4_v41_req_take(r[i], outi, 16);
            for (int j = 0; j < ni && nseq[i] < seqcap; j++) seq[i * seqcap + nseq[i]++] = outi[j];
        }
        while (ntok < nseq[0]) {   /* 第 0 路新吐出的打出来 */
            const int t = seq[ntok];
            if (t == eos) { stop = 1; break; }
            ntok++;
            size_t len = 0; char *txt = ds4_token_text(e, t, &len);
            if (txt) { fwrite(txt, 1, len, stdout); fflush(stdout); free(txt); }
        }
    }
    for (int i = 1; i < N; i++) {   /* 门: 各路整串对第 0 路(只比两路都吐到的前缀; 投机下各路进度可差几位) */
        const int n = nseq[i] < nseq[0] ? nseq[i] : nseq[0];
        if (memcmp(seq + i * seqcap, seq, (size_t)n * sizeof(int)) != 0) mismatch++;
    }
    free(seq);
    const double wall = now_sec() - t0;
    fputc('\n', stdout);
    if (done) {
        double sum = 0; for (int i = 0; i < done; i++) sum += ms[i];
        qsort(ms, (size_t)done, sizeof(double), cmp_dbl);
        const double med = done & 1 ? ms[done / 2] : 0.5 * (ms[done / 2 - 1] + ms[done / 2]);
        const double tpr = (double)ntok / done;   /* 每轮每路吐出的 token(投机 > 1)*/
        fprintf(stderr, "[multi] %d 路 × %d 轮(%s): 每轮 中位 %.1f ms / 平均 %.1f / 最小 %.1f / 最大 %.1f, 每轮每路 %.2f token ⇒ 每路 %.1f t/s, 总 %.1f t/s(中位); 壁钟口径 每路 %.1f 总 %.1f\n",
                N, done, spec ? "投机" : "纯解码", med, sum / done, ms[0], ms[done - 1], tpr, tpr * 1000.0 / med, N * tpr * 1000.0 / med,
                ntok / wall, N * ntok / wall);
        if (spec) { int rd = 0, of = 0, ac = 0; ds4_v41_req_spec_stats(r[0], &rd, &of, &ac); fprintf(stderr, "[multi] 第 0 路投机 %d 轮, 出草稿 %d 位, 平均接受 %.2f 位\n", rd, of, rd ? (double)ac / rd : 0.0); }
    }
    fprintf(stderr, "[multi] 逐字节门: %d 路同提示温 0, 各路吐出的整串 %s(不同的路 %d), 第 0 路共 %d token\n", N, mismatch ? "★不同★" : "全同", mismatch, ntok);
    free(ms);
    rc = mismatch ? 2 : 0;
out:
    for (int i = 0; i < N; i++) ds4_v41_req_close(r[i]);
    ds4_v41_batch_close(b);
    return rc;
}
