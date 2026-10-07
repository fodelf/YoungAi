/* core_v41_score.c — V4.1 的 --score-ids(teacher-forced 逐位打分)与它的小出口配置(2026-10-07 从 core_v41_api.c 拆出守 500 行; 内容未改)。
 * 生成路在 core_v41_api.c; 两边共用 g_ds4_v41_decoder_full(--decoder-full: CED 块也跑满解码器)。 */
#include "core_internal.h"
#include <unistd.h>

extern int g_ds4_v41_decoder_full;
/* --score-nll / --score-topk / --score-no-logits(2026-09-13, 后训练取梯度用):
 * 与 V4 的 --eval-nll/--eval-topk 是同一份实现(core_score_aux.c), 同一种字节。
 * skip_logits: 后训练一趟 5.8 万行, 全词表 logits 就是 30 GB —— 统一内存上写它 = 掏 GPU 内存。 */
static const char *g_v41_score_nll = NULL, *g_v41_score_topk_path = NULL, *g_v41_score_rms = NULL;
static int g_v41_score_topk = 0, g_v41_score_skip_logits = 0;
void ds4_engine_v41_set_score_aux(const char *nll_path, const char *topk_path, int topk,
                                  const char *rms_path, int skip_logits) {
    g_v41_score_nll = (nll_path && nll_path[0]) ? nll_path : NULL;
    g_v41_score_topk_path = (topk_path && topk_path[0]) ? topk_path : NULL;
    g_v41_score_topk = topk;
    g_v41_score_rms = (rms_path && rms_path[0]) ? rms_path : NULL;
    g_v41_score_skip_logits = skip_logits;
}
/* --score-ids 的部署同路切分点(2026-09-23, 后训练 ③ 实撞): 0 = 老口径(整条按块跑满解码器)。P > 0 = [0,P) 照生成路预填
 * (同分块、末块至少留一个窗口、非末块 CED 只跑编码器段), [P,n) 按块跑满解码器 —— 与"提示预填 + 逐 token 解码"同一种状态。
 * 为什么要它: 生成时提示走 CED, 老口径把提示也跑满解码器, 两边在同一位置的状态不是一回事。09-23 实撞: ③ 按老口径的表
 * 解出来, 老口径下决策点翻了(−7.03 → +5.5), 服务端端到端一个字没翻。CED 块没有 logits/钩子行, 这些行在表里留空。 */
static uint32_t g_v41_score_split = 0;
void ds4_engine_v41_set_score_split(int p) { g_v41_score_split = p > 0 ? (uint32_t)p : 0u; }

#ifndef DS4_NO_GPU
int ds4_engine_v41_score_ids(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int no_engram, int chunk) {
    if (!e || !ids || n_ids < 1 || !ds4_engine_is_v41(e)) return 1;
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 前向需要 GPU 后端\n"); return 1; }
    const uint32_t n = (uint32_t)n_ids;
    const uint32_t ck = chunk > 0 ? (uint32_t)chunk : DS4_V41_CHUNK;
    const uint32_t cap = ck < n ? ck : n;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, n, 0)) return 1;   /* 打分路: 每个位置都要 logits, 按 cap 开 */
    st.dump_prefix = (n <= 64u && cap == n) ? out_path : NULL;   /* 单块小样本自动落逐层 x/y, 对拍定位用 */
    st.no_engram = no_engram;
    if (no_engram) fprintf(stderr, "[v41] ★no-engram★ 对拍口径, 跳过 engram 层\n");
    if (ck < n) fprintf(stderr, "[v41] 分块 %u(%u 块)\n", ck, (n + ck - 1) / ck);
    ds4_score_aux *aux = ds4_score_aux_open(g_v41_score_nll, g_v41_score_topk_path,
                                            g_v41_score_topk, g_v41_score_rms, n, DS4_N_VOCAB, "v41");
    const int aux_nll = aux && g_v41_score_nll;   /* 只开了 rms 时 aux 非空但没算 NLL, 冒烟 PPL 仍要自己算 */
    /* 小出口开着时默认仍写全词表 logits(老对拍口径不变); --score-no-logits 才关掉它。 */
    FILE *fo = NULL;
    if (!g_v41_score_skip_logits) {
        fo = fopen(out_path, "wb");
        if (!fo) { fprintf(stderr, "ds4: 写不了 %s\n", out_path); ds4_score_aux_close(aux); v41_state_free(&st); return 1; }
        int hd[2] = { (int)n, (int)DS4_N_VOCAB };
        fwrite(hd, 4, 2, fo);
    } else if (!aux) {
        fprintf(stderr, "ds4: --score-no-logits 却没给 --score-nll/--score-topk, 这趟什么都不会产 -- aborting\n");
        v41_state_free(&st); return 1;
    }
    float *lg = xmalloc((size_t)cap * DS4_N_VOCAB * 4);
    /* --score-rms: 出口 RMSNorm 前的隐状态整块读回来算 inv。一块 512×5120×4 = 10 MB,
     * 相对这一块本来就要读的 logits(512×129280×4 = 265 MB)是零头。 */
    float *hx = g_v41_score_rms ? xmalloc((size_t)cap * DS4_N_EMBD * 4) : NULL;
    double nll = 0.0; const double t0 = now_sec();
    bool ok = true; int stopped = 0;
    const uint32_t split = g_v41_score_split < n ? g_v41_score_split : 0u;
    if (split && fo) { fprintf(stderr, "ds4: 部署同路切分(P=%u)下 CED 块没有 logits, 全词表文件写不完整 —— 要配 --score-no-logits\n", split); ok = false; }
    if (split) fprintf(stderr, "[v41] ★部署同路★ [0,%u) 照生成路预填(CED), [%u,%u) 跑满解码器\n", split, split, n);
    for (uint32_t c0 = 0, nc = 0; ok && c0 < n; c0 += nc) {
        nc = n - c0 < cap ? n - c0 : cap;
        st.ced_skip = 0;
        if (c0 < split) {   /* 提示段: 与 ds4_engine_v41_generate_argmax 的预填循环同一套切法 */
            if (c0 + nc > split) nc = split - c0;
            const uint32_t rest = split - c0 - nc;
            if (rest > 0u && rest < DS4_N_SWA && nc > DS4_N_SWA) nc -= DS4_N_SWA - rest;
            st.ced_skip = (!g_ds4_v41_decoder_full && c0 + nc < split) ? 1 : 0;
        } else if (split) {
            /* 报告段的尾块不许 ≤ DS4_V41_GEMV_MAX_TOK: 那么小的块走解码 GEMV 路, 不物化逐专家输出, 后训练取料的钩子拿不到 ye
             * 就停车(09-24 实撞: 报告段 2562 = 5×512 + 2)。从这一块匀出几个位置给尾块, 结果只差累加序, 不改语义。 */
            const uint32_t rest = n - c0 - nc;
            if (rest > 0u && rest <= DS4_V41_GEMV_MAX_TOK && nc > 2u * (DS4_V41_GEMV_MAX_TOK + 1u)) nc -= DS4_V41_GEMV_MAX_TOK + 1u - rest;
        }
        ok = v41_forward(e, &st, ids + c0, nc);
        if (ok && st.ced_skip) { ds4_score_aux_skip_rows(aux, c0, nc); continue; }   /* CED 块: 没有 logits, 表里写占位行 */
        if (ok && st.stop_early) { stopped = 1; continue; }   /* 反修钩子提前结束: 本块没有 logits, 文件不完整, 不算 PPL */
        if (ok) ok = ds4_gpu_tensor_read(st.logits, 0, lg, (uint64_t)nc * DS4_N_VOCAB * 4) != 0;
        if (ok && hx) {
            ok = ds4_gpu_tensor_read(st.x, 0, hx, (uint64_t)nc * DS4_N_EMBD * 4) != 0;
            if (ok) ds4_score_aux_rms_rows(aux, c0, hx, nc, DS4_N_EMBD, DS4_RMS_EPS);
        }
        if (!ok) break;
        if (fo) fwrite(lg, 4, (size_t)nc * DS4_N_VOCAB, fo);
        for (uint32_t i = 0; i < nc; i++) {   /* teacher-forcing: 第 c0+i 位预测 ids[c0+i+1] */
            const uint32_t row_i = c0 + i;
            const float *row = lg + (size_t)i * DS4_N_VOCAB;
            const int tgt = row_i + 1u < n ? ids[row_i + 1u] : -1;
            ds4_score_aux_row(aux, row_i, row, tgt);
            if (aux_nll || tgt < 0) continue;   /* aux 已经算过这一行的 NLL, 不重复扫 12.9 万个数 */
            float mx = row[0]; for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > mx) mx = row[v];
            double se = 0.0; for (uint32_t v = 0; v < DS4_N_VOCAB; v++) se += exp((double)row[v] - mx);
            nll += -((double)row[tgt] - mx - log(se));
        }
    }
    if (fo) fclose(fo);
    free(lg); free(hx);
    ds4_score_aux_close(aux);   /* 平均 NLL/PPL 与 topK 覆盖率由它打印 */
    if (ok && stopped) {
        if (fo) unlink(out_path);
        fprintf(stderr, "[v41] 钩子取料提前结束, 不出 logits(已删 %s)  %.1fs\n", out_path, now_sec() - t0);
    } else if (ok && !aux_nll) fprintf(stderr, "[v41] 完成 S=%u V=%u → %s  ★PPL(本段 %u token) = %.4f★  %.1fs\n", n, DS4_N_VOCAB, out_path, n,
                    n > 1 ? exp(nll / (double)(n - 1)) : 0.0, now_sec() - t0);
    else if (ok) fprintf(stderr, "[v41] 完成 S=%u V=%u%s  %.1fs\n", n, DS4_N_VOCAB,
                         fo ? " (logits 已写)" : " (只出小文件)", now_sec() - t0);
    v41_state_free(&st);
    return ok ? 0 : 1;
}
#else
int ds4_engine_v41_score_ids(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int no_engram, int chunk) {
    (void)e; (void)ids; (void)n_ids; (void)out_path; (void)no_engram; (void)chunk;
    fprintf(stderr, "ds4: V4.1 只有 GPU 路\n"); return 1;
}
#endif
