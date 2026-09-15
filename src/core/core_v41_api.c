/* core_v41_api.c — DeepSeek V4.1 对外出口(2026-09-12 战役 P2c): --score-ids 分块全位置 logits + 贪心生成。
 * 两者都走 core_v41_forward.c 的同一条增量前向; 解码 = n=1 的前向。采样/会话/服务接入是 P5。 */
#include "core_internal.h"

int ds4_engine_is_v41(ds4_engine *e) { (void)e; return DS4_MODEL_VARIANT == DS4_VARIANT_V41; }
int g_ds4_v41_prof = 0;
void ds4_engine_v41_set_prof(int on) { g_ds4_v41_prof = on; }
const char *g_ds4_v41_amp_dir = NULL;
void ds4_engine_v41_set_amp_dir(const char *dir) { g_ds4_v41_amp_dir = (dir && dir[0]) ? dir : NULL; }
const char *g_ds4_v41_pt_dir = NULL;
/* 三文件部署第三件: 单独挂 ③ 也允许(用户要求"独立使用量化文件也可以"的对称面), 但 ③ 是解在
 * ①+② 那个态上的, 单挂 = 把修正打在另一个基线上, 所以打一行明白话再放行。 */
void ds4_engine_v41_set_posttrain_dir(const char *dir) {
    g_ds4_v41_pt_dir = (dir && dir[0]) ? dir : NULL;
    if (g_ds4_v41_pt_dir && !g_ds4_v41_amp_dir)
        fprintf(stderr, "ds4: ★只挂了后训练件、没挂反修件 —— 它是解在反修态上的, 这个组合不是判决态★\n");
}
float g_ds4_v41_amp_scale = 1.0f;
/* β ≤ 0 当"没传"处理(1.0 = 原样)。负 β 是把修正反向注入, 没有任何用途, 不留这条路。 */
void ds4_engine_v41_set_amp_scale(float s) { g_ds4_v41_amp_scale = s > 0.f ? s : 1.0f; }
ds4_v41_moe_hook_fn g_ds4_v41_hook = NULL;
void *g_ds4_v41_hook_ud = NULL;
void ds4_engine_v41_set_moe_hook(ds4_v41_moe_hook_fn fn, void *ud) { g_ds4_v41_hook = fn; g_ds4_v41_hook_ud = ud; }
/* 只在第 il 层回调(-1 = 每层都回调)。★不设这个就是每层白拷一遍★: 取料的 D2H(x/y/sel/rw/hc,
 * 外加 prefill 路的逐专家输出 63 MB/块)发生在回调【之前】, 而解算方通常只要一层 —— 40 层里
 * 39 层的拷贝全是白做的。实测 512 token 一块从 33 秒降到个位数。 */
int g_ds4_v41_hook_layer = -1;
void ds4_engine_v41_set_moe_hook_layer(int il) { g_ds4_v41_hook_layer = il; }

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

#ifndef DS4_NO_GPU
int ds4_engine_v41_score_ids(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int no_engram, int chunk) {
    if (!e || !ids || n_ids < 1 || !ds4_engine_is_v41(e)) return 1;
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 前向需要 GPU 后端\n"); return 1; }
    const uint32_t n = (uint32_t)n_ids;
    const uint32_t ck = chunk > 0 ? (uint32_t)chunk : DS4_V41_CHUNK;
    const uint32_t cap = ck < n ? ck : n;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, n)) return 1;
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
    for (uint32_t c0 = 0; ok && c0 < n; c0 += cap) {
        const uint32_t nc = n - c0 < cap ? n - c0 : cap;
        ok = v41_forward(e, &st, ids + c0, nc);
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

int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, int ctx_size,
                                   ds4_v41_emit_fn emit, void *ud) {
    if (!e || !prompt || n_prompt < 1 || !ds4_engine_is_v41(e)) return 1;
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 前向需要 GPU 后端\n"); return 1; }
    const uint32_t np = (uint32_t)n_prompt;
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : np + (uint32_t)(n_predict > 0 ? n_predict : 0) + 1;
    if (ctx < np + 1) ctx = np + 1;
    if (ctx > DS4_V41_MAX_CTX_P2C) ctx = DS4_V41_MAX_CTX_P2C;
    const uint32_t cap = DS4_V41_CHUNK < np ? DS4_V41_CHUNK : np;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, ctx)) return 1;
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc(16);   /* argmax 在设备上做, 只读回 4 B */
    const int eos = ds4_token_eos(e);
    int rc = 1;
    do {
        if (!am) break;
        const double t0 = now_sec();
        bool ok = true;
        for (uint32_t c0 = 0; ok && c0 < np; c0 += cap) {
            const uint32_t nc = np - c0 < cap ? np - c0 : cap;
            ok = v41_forward(e, &st, prompt + c0, nc);
        }
        if (!ok) break;
        const double t1 = now_sec();
        fprintf(stderr, "[v41] prefill %u token %.1fs (%.1f t/s)\n", np, t1 - t0, (double)np / (t1 - t0 + 1e-9));
        /* 末位 logits → argmax → 逐 token 解码(n=1 前向) */
        int32_t tok32 = 0;
        if (!ds4_gpu_v41_argmax_tensor(am, st.logits, st.n - 1, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(am, 0, &tok32, 4)) break;
        {   /* 设备 argmax 自检(一次): 与主机顺序扫对一下, 不同就报 —— 核错会静默吐错 token */
            float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
            if (ds4_gpu_tensor_read(st.logits, (uint64_t)(st.n - 1) * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
                uint32_t best = 0; for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > row[best]) best = v;
                if ((int32_t)best != tok32) fprintf(stderr, "ds4: ★V4.1 argmax 核 %d ≠ 主机 %u (logit %.4f vs %.4f)★\n", tok32, best, (double)row[tok32], (double)row[best]);
            }
            free(row);
        }
        int tok = (int)tok32;
        int produced = 0;
        while (produced < n_predict) {
            produced++;
            if (emit && emit(tok, ud) != 0) break;
            if (tok == eos) break;
            if (st.n_past + 1 > st.ctx) { fprintf(stderr, "\n[v41] 上下文满 %u\n", st.ctx); break; }
            const int32_t t32 = (int32_t)tok;
            if (!v41_forward(e, &st, &t32, 1)) { ok = false; break; }
            if (!ds4_gpu_v41_argmax_tensor(am, st.logits, 0, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(am, 0, &tok32, 4)) { ok = false; break; }
            tok = (int)tok32;
        }
        const double t2 = now_sec();
        if (produced > 1) fprintf(stderr, "\n[v41] decode %d token %.1fs (%.2f t/s)\n", produced - 1, t2 - t1, (double)(produced - 1) / (t2 - t1 + 1e-9));
        rc = ok ? 0 : 1;
    } while (0);
    if (am) ds4_gpu_tensor_free(am);
    v41_state_free(&st);
    return rc;
}
#else
int ds4_engine_v41_score_ids(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int no_engram, int chunk) {
    (void)e; (void)ids; (void)n_ids; (void)out_path; (void)no_engram; (void)chunk;
    fprintf(stderr, "ds4: V4.1 只有 GPU 路\n"); return 1;
}
int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, int ctx_size, ds4_v41_emit_fn emit, void *ud) {
    (void)e; (void)prompt; (void)n_prompt; (void)n_predict; (void)ctx_size; (void)emit; (void)ud;
    fprintf(stderr, "ds4: V4.1 只有 GPU 路\n"); return 1;
}
#endif
