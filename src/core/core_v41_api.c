/* core_v41_api.c — DeepSeek V4.1 对外出口(2026-09-12 战役 P2c): --score-ids 分块全位置 logits + 贪心生成。
 * 两者都走 core_v41_forward.c 的同一条增量前向; 解码 = n=1 的前向。采样/会话/服务接入是 P5。 */
#include "core_internal.h"
#include <time.h>
#include <unistd.h>

int ds4_engine_is_v41(ds4_engine *e) { (void)e; return DS4_MODEL_VARIANT == DS4_VARIANT_V41; }
/* 解码采样设置面(2026-09-21, 113-1.md §4; 语义见 ds4_v41_api.h)。全零 = 裸 argmax = 今天的路。 */
static ds4_decode_sampling g_decode_sampling;
void ds4_engine_set_decode_sampling(const ds4_decode_sampling *sp) {
    if (sp) g_decode_sampling = *sp;
    else memset(&g_decode_sampling, 0, sizeof g_decode_sampling);
}
int g_ds4_v41_prof = 0;
/* ★--decoder-full★: 关掉 CED, 提示的每一块都跑满 40 层(精确路, 用来跟 CED 对质量)。
 * 默认是官方部署语义(CED 开), 见 core_v41_forward.c 的 ced_edge 注释。 */
int g_ds4_v41_decoder_full = 0;
void ds4_engine_v41_set_decoder_full(int on) { g_ds4_v41_decoder_full = on; }
void ds4_engine_v41_set_prof(int on) { g_ds4_v41_prof = on; }
/* ★--dspark★: 开投机解码(默认**关**)。温 0 下两条路必须**逐字节同** ——
 * 投机的接受条件就是"主模型自己也会选这个 token", 所以它只省时间不改输出; 不同就是回滚漏了东西。
 *
 * ★为什么默认是关的(2026-09-16, 用户令"速度大幅度提升才改默认开启")★: 两条理由, 缺一条都不该关。
 * ①**同轨门今天不绿**: 同一份 2K 提示、温 0, 投机与纯解码的输出第二句就分叉(mtp-1.md §4)。
 *   引擎默认路径必须是裸模型的真值 —— 默认开着一个会改输出的近似路, 等于每个跑分都掺了别的东西。
 * ②**它今天还是亏的**: 一轮 188 ms 只产 1.96 个 token(96 ms/token), 而纯解码一步 52 ms。
 * 等 mtp-1.md 的 M1′(同轨)与 M2′/M3′/M4′(核形态)过门、且真比纯解码快之后, 再把默认翻回来。
 * `--no-dspark` 保留: 老脚本一路在传它, 现在是"再确认一次关", 不是错。 */
int g_ds4_v41_dspark = 0;
void ds4_engine_v41_set_dspark(int on) { g_ds4_v41_dspark = on; }
/* --dspark-verify N: 每轮验证几位(0 = 用下面钉死的默认)。
 * 为什么要这个旋钮: ①同轨出问题时, k=1(验证批只有 2 行)是最小的多 token 批, 拿它跟 k=3 一比就知道
 * "病在批本身"还是"病在批大了以后"; ②字节账(mtp-1.md §3)要按 k 逐档量, 每档一个二进制是浪费。
 * 它不是兜底开关 —— 调度器(M5′)落地后由置信度定 k, 这个旋钮只作诊断与逐档量尺用。 */
/* --emit-trace: 逐 token 打 `[emit] <绝对位置> <token id>`(同轨定位, mtp-1.md M0′(d))。
 * ★为什么不搭 --v41-prof 的顺风车★: prof 会让前向**逐层 flush**(core_v41_forward.c), 时序一变,
 * 偶发的分叉可能就复现不出来了 —— 那样这把尺就在骗人。它只打这一行, 不改任何执行路径。 */
int g_ds4_v41_emit_trace = 0;
void ds4_engine_v41_set_emit_trace(int on) { g_ds4_v41_emit_trace = on; }
/* --dspark-block N: 把草稿块长钉成 N(0 = 用模型元数据里的 5)。**只作诊断**。
 * 为什么要它: 块长 5 + 4 个 noise 位 + 块内全可见, 是我们照官方 forward_embed/get_dspark_topk_idxs
 * 自己写的一段, 从来没有单独验过。拿 N=1 跑同一把取料尺(首位一致率)一比就知道那几位 noise 是在
 * 帮忙还是在捣乱 —— 训练时是带着它们训的, 所以 N=1 明显更准 = 我们这段写错了。
 * ★不是速度旋钮★: 块长是模型定的, 平时别动。 */
uint32_t g_ds4_v41_block = 0;
void ds4_engine_v41_set_block(uint32_t b) { g_ds4_v41_block = b; }
uint32_t g_ds4_v41_verify_k = 0;
void ds4_engine_v41_set_verify_k(uint32_t k) { g_ds4_v41_verify_k = k; }
const char *g_ds4_v41_draft_amp = NULL;
void ds4_engine_v41_set_draft_amp(const char *path) { g_ds4_v41_draft_amp = path; }
/* --draft-amp-scale β: 装载时 A *= β。用来分辨"方向错"还是"幅度过头" ——
 * β 小了就好转 = 幅度问题(过拟合/欠定); β 怎么调都不如不挂 = 目标函数选错了(L2 隐态 ≠ argmax)。 */
float g_ds4_v41_draft_amp_scale = 1.0f;
void ds4_engine_v41_set_draft_amp_scale(float s) { g_ds4_v41_draft_amp_scale = s; }
const char *g_ds4_v41_amp_dir = NULL;
#ifndef DS4_NO_GPU
const ds4_model *g_ds4_v41_model = NULL;
#endif
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
/* 服务接入的预填进度回调(2026-09-19): 服务端一条 15k token 的提示要预填两分钟, 流式客户端没有心跳就断线。
 * 只在生成路的预填块间回调, --score-ids 那条不回(它没有客户端在等)。 */
static ds4_v41_progress_fn g_v41_progress = NULL;
static void *g_v41_progress_ud = NULL;
void ds4_engine_v41_set_progress(ds4_v41_progress_fn fn, void *ud) { g_v41_progress = fn; g_v41_progress_ud = ud; }
int ds4_engine_v41_max_ctx(void) { return (int)DS4_V41_MAX_CTX_P2C; }

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

/* --v41-chunk 也管生成路的预填分块(以前只管 --score-ids, 生成路写死 DS4_V41_CHUNK=512)。
 * 为什么要管: 段 5 的专家路每层要把 384 个专家全解一遍成 NVFP4, 这个代价**与块里有几个 token 无关** ——
 * 块 512 时它摊不开(实测 TTFT 193 t/s, 还不如融合路的 206), 块开大才反超。 */
int g_ds4_v41_chunk = 0;
void ds4_engine_v41_set_chunk(int n) { g_ds4_v41_chunk = n > 0 ? n : 0; }

/* 取下一个 token。rowbuf == NULL(采样关): 设备 argmax, 只读回 4 B —— 老路一字不动。
 * rowbuf 非 NULL(采样开): 把第 row 行 logits 读回主机(129280 × 4 B; 统一内存上一次 cudaMemcpy), 交给 V4 路同一份采样器。
 * 为什么不把采样做进图: 图的输入是主机给的 token id, 采样夹在 wait 与下一次 launch 之间, 图一个节点不动;
 * 每步多付一次 517 KB 读回 + 一遍主机采样, 代价按 2K 尺量(113-1.md §4.3 门 4: ≤ +0.3 ms/步), 超了再做设备预筛核。 */
/* 生成段 token 史(只在读回 logits 的路上记): 复读惩罚只看它, 不看提示。brk = 断点表(DRY 开时才建)。 */
typedef struct { int32_t *tok; uint32_t n, cap; const uint8_t *brk; } v41_hist;
static bool v41_next_token(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row, float *rowbuf, uint64_t *rng, v41_hist *h, int32_t *out) {
    if (!rowbuf)
        return ds4_gpu_v41_argmax_tensor(am, st->logits, row, DS4_N_VOCAB) && ds4_gpu_synchronize() &&
               ds4_gpu_tensor_read(am, 0, out, 4) != 0;
    const ds4_decode_sampling *sp = &g_decode_sampling;
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(st->logits, (uint64_t)row * DS4_N_VOCAB * 4u, rowbuf, (uint64_t)DS4_N_VOCAB * 4u)) return false;
    if (h && h->n && (sp->dry_multiplier > 0.f || sp->freq_penalty != 0.f || sp->presence_penalty != 0.f))
        ds4_decode_penalize(rowbuf, DS4_N_VOCAB, h->tok, h->n, h->brk, sp->freq_penalty, sp->presence_penalty,
                            sp->dry_multiplier, sp->dry_base, sp->dry_allowed_length);
    *out = (int32_t)ds4_sample_logits(rowbuf, (int)DS4_N_VOCAB, sp->temperature, sp->top_k, sp->top_p, sp->min_p, rng);
    if (h && h->n < h->cap) h->tok[h->n++] = *out;
    return true;
}

int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, int ctx_size,
                                   ds4_v41_emit_fn emit, void *ud) {
    if (!e || !prompt || n_prompt < 1 || !ds4_engine_is_v41(e)) return 1;
    if (!e->metal_ready) { fprintf(stderr, "ds4: V4.1 前向需要 GPU 后端\n"); return 1; }
    const uint32_t np = (uint32_t)n_prompt;
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : np + (uint32_t)(n_predict > 0 ? n_predict : 0) + 1;
    if (ctx < np + 1) ctx = np + 1;
    if (ctx > DS4_V41_MAX_CTX_P2C) ctx = DS4_V41_MAX_CTX_P2C;
    /* ★只按这一趟真正用得到的位置分配★(2026-09-22): 状态里几个大块是 cap_tok × ctx 的
     * (iscore f32 + cand u8 = ctx × 2.5 KB, 再加 kv 源层的 ctx/ratio 格), 传进来的 ctx 是服务端
     * 配的**上限**, 不是这条请求能用到的长度。以前照上限分: --ctx 1M 时连一条 22 token 的请求
     * 都要吃 2.5 GiB 打分矩阵, 于是"把上下文配大"被误当成"每条请求都贵"。
     * 这一趟最多走到 np + n_predict 个位置(投机一轮会临时多推 ≤ 块长, 回滚前也要有地方放), 就分这么多。
     * ctx 仍然是硬边界: 生成到 st->ctx 就停, 与配的上限语义一致(配得再大也不会多花一个字节)。 */
    const uint64_t need = (uint64_t)np + (uint64_t)(n_predict > 0 ? n_predict : 0) + DS4_MTP_MAX_BLOCK + 2u;
    if ((uint64_t)ctx > need) ctx = (uint32_t)need;
    const uint32_t ck = g_ds4_v41_chunk > 0 ? (uint32_t)g_ds4_v41_chunk : DS4_V41_CHUNK;
    const uint32_t cap = ck < np ? ck : np;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, ctx)) return 1;
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc(16);   /* argmax 在设备上做, 只读回 4 B */
    const int eos = ds4_token_eos(e);
    int rc = 1;
    /* 采样(温度 > 0)与投机不能同开: 投机的接受条件是"主模型的 argmax 也是这个 token", 采样下要换成拒绝采样式验证, 还没接。
     * 硬拒而不是静默改走纯解码 —— 用户以为开着投机, 其实没开, 这种"不报错只出错"的坑本仓踩够了。 */
    const bool penal = g_decode_sampling.dry_multiplier > 0.f || g_decode_sampling.freq_penalty != 0.f || g_decode_sampling.presence_penalty != 0.f;
    const bool sampling = g_decode_sampling.temperature > 0.f || penal;   /* 惩罚开着时温 0 也要读回 logits 行(罚完再 argmax) */
    if (sampling && g_ds4_v41_dspark) {
        fprintf(stderr, "ds4: ★解码采样/惩罚(temp %.2f dry %.2f)与 --dspark 投机不能同开★(采样下的投机验证还没接; 去掉 --dspark 或关采样)\n",
                (double)g_decode_sampling.temperature, (double)g_decode_sampling.dry_multiplier);
        if (am) ds4_gpu_tensor_free(am);
        v41_state_free(&st);
        return 1;
    }
    float *rowbuf = sampling ? xmalloc((size_t)DS4_N_VOCAB * 4u) : NULL;
    v41_hist hist = {NULL, 0, 0, NULL};
    if (penal) {
        hist.cap = (uint32_t)(n_predict > 0 ? n_predict : 0) + 2u;
        hist.tok = xmalloc((size_t)hist.cap * sizeof(int32_t));
        if (g_decode_sampling.dry_multiplier > 0.f) hist.brk = ds4_decode_breakers(e, DS4_N_VOCAB);
    }
    uint64_t rng = g_decode_sampling.seed ? g_decode_sampling.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());   /* 与 V4 CLI(cli_gen.c)同一条规则 */
    do {
        if (!am) break;
        /* --emit-trace 连提示也打(`[ptok] 位置 id`): 提示是 build_prompt 套过聊天模板的(比 --dump-tokens 的原文分词多
         * BOS/角色/think 几个 token, 09-20 实撞 79 vs 75), 事后重新分词拼不回引擎真吃的序列; 教师锚/取料要的是这一份。 */
        if (g_ds4_v41_emit_trace) for (uint32_t i = 0; i < np; i++) fprintf(stderr, "[ptok] %u %d\n", i, prompt[i]);
        const double t0 = now_sec();
        bool ok = true, aborted = false;
        for (uint32_t c0 = 0; ok && c0 < np; ) {
            uint32_t nc = np - c0 < cap ? np - c0 : cap;
            /* ★最后一块至少留 window 个位置★(2026-09-21, bug.md §6.1): 官方 Decoder SWA Bounded Replay = 最后 n_win 个 token
             * 跑满解码器; 这里最后一块就是那段回放, 以前它 = 提示长 mod 512(1~512), 不足 128 时首批生成 token 的窗口直接
             * 缺位。余下不足 window 就从这一块匀过去(这一块缩短, 最后一块正好 window 个)。 */
            const uint32_t rest = np - c0 - nc;
            if (rest > 0u && rest < DS4_N_SWA && nc > DS4_N_SWA) nc -= DS4_N_SWA - rest;
            /* CED: 除最后一块外, 只跑编码器段 + 分界层 KV(最后一块同时充当官方说的"解码器有界回放") */
            st.ced_skip = (!g_ds4_v41_decoder_full && c0 + nc < np) ? 1 : 0;
            ok = v41_forward(e, &st, prompt + c0, nc);
            c0 += nc;
            /* 回调非 0 = 调用方不要这趟了(服务端: 客户端已挂断)。立刻停, 别把剩下的块算完 —— 13 万 token
             * 的提示预填 400 秒, 算给一个走掉的连接就是让后面排队的请求跟着超时(2026-09-22 早盘实撞)。 */
            if (ok && g_v41_progress &&
                g_v41_progress(g_v41_progress_ud, "prefill_chunk", (int)c0, (int)np) != 0) {
                fprintf(stderr, "[v41] 预填在 %u/%u 处按调用方要求中止\n", c0, np);
                aborted = true;
                break;
            }
        }
        if (!ok || aborted) break;
        const double t1 = now_sec();
        fprintf(stderr, "[v41] prefill %u token %.1fs (%.1f t/s)\n", np, t1 - t0, (double)np / (t1 - t0 + 1e-9));
        if (sampling) fprintf(stderr, "[v41] 解码采样 temp %.2f top_p %.2f min_p %.2f top_k %d seed %llu%s dry %.2f/%.2f/%d freq %.2f presence %.2f\n",
                              (double)g_decode_sampling.temperature, (double)g_decode_sampling.top_p, (double)g_decode_sampling.min_p,
                              g_decode_sampling.top_k, (unsigned long long)rng, penal ? " 惩罚" : "", (double)g_decode_sampling.dry_multiplier,
                              (double)g_decode_sampling.dry_base, g_decode_sampling.dry_allowed_length,
                              (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        /* 末位 logits → argmax(或采样) → 逐 token 解码(n=1 前向) */
        int32_t tok32 = 0;
        if (!v41_next_token(&st, am, st.n - 1, rowbuf, &rng, &hist, &tok32)) break;
        if (!rowbuf) {   /* 设备 argmax 自检(一次): 与主机顺序扫对一下, 不同就报 —— 核错会静默吐错 token */
            float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
            if (ds4_gpu_tensor_read(st.logits, (uint64_t)(st.n - 1) * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
                uint32_t best = 0; for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > row[best]) best = v;
                if ((int32_t)best != tok32) fprintf(stderr, "ds4: ★V4.1 argmax 核 %d ≠ 主机 %u (logit %.4f vs %.4f)★\n", tok32, best, (double)row[tok32], (double)row[best]);
            }
            free(row);
        }
        int tok = (int)tok32;
        int produced = 0;
        /* ---- DSpark 投机解码(speed.md 段 6 D1) ----
         * 没带三塔/没带运行参数的 GGUF: dr.ready=0, 下面整段跳过, 走原来的单 token 环。
         * 一轮 = 草稿器出 block 位 → 主模型一次验证 1+k 位 → 逐位比贪心结果, 接受最长前缀。
         * ★温 0 下这与纯解码逐 token 是同一串输出★: 接受的条件就是"主模型自己也会选这个 token",
         * 不接受的位置全部回滚。所以它是纯粹的省时间, 不是近似 —— 门也就是逐字节同。 */
        ds4_v41_draft dr;
        const bool spec = g_ds4_v41_dspark && v41_draft_alloc(e, &dr);
        uint32_t spec_rounds = 0, spec_acc = 0, spec_hist[DS4_MTP_MAX_BLOCK + 1];
        /* 一轮的壁钟分账(2026-09-16): 投机赢不赢是个除法 —— 一轮的耗时要压到 E[接受+1] × 纯解码一步
         * 以下。实测一轮 118 ms 对预算 72 ms, 超 64%, 而这 118 从来没拆过。四项分开计, 就能分清
         * "草稿器自己太贵"(那接受率再高也救不回来)还是"验证/回滚的边角料吃掉了"。 */
        double ms_draft = 0, ms_verify = 0, ms_argmax = 0, ms_snap = 0;
        /* 调度器判"这一轮不值得投机"之后歇几轮(见下面 v41_draft_pick_k 那一段)。
         * 09-16 定 16(太短就一直在亏本的文本上反复试, 太长就错过文本变好猜的那一段)。
         * ★09-19 改 4★: 陪审团 gguf-tools/bench/dspark_sim 在金融提示的取料上重放整段: 冷却 16 = 27.86 t/s, 8 = 28.12, 4 = 28.29,
         * 0 = 28.03 —— 文本好不好猜是逐 token 变的, 歇 16 步错过的好轮比省下的草稿钱多; 验证批再便宜一档(投机轮进图/专家按序)后 0 最优。 */
        #define V41_SPEC_COOLDOWN 4u
        uint32_t spec_skip = 0, spec_skipped = 0;
        for (uint32_t i = 0; i <= DS4_MTP_MAX_BLOCK; i++) spec_hist[i] = 0;
        while (produced < n_predict) {
            produced++;
            /* ★同轨定位用(mtp-1.md M0′(d))★: 逐 token 打"绝对位置 + token id"。
             * 为什么不看生成的文字: 文字把 token 边界抹掉了, 两条路差一个 token 可能只差半个词,
             * 而且分叉处往往两句都通顺(近平局翻面), 看不出来。把两条路的这几行 diff 一下,
             * 第一个不同的位置就是要查的那一步 —— 位置对得上、id 不同 = 那一步的 logits 不同(核);
             * 位置本身对不上 = 回滚把状态推歪了(快照漏项)。 */
            if (g_ds4_v41_emit_trace) fprintf(stderr, "[emit] %u %d\n", st.n_past, tok);
            if (spec_skip) spec_skip--;
            const bool draft_now = spec && spec_skip == 0;   /* 这一步出不出草稿; 不出的步就是一个普通的单 token 步 */
            /* ★解码整步 CUDA graph(core_decode_graph.c)★: 单 token 步(没开投机, 或投机这一步歇着)且暖过一步直发之后,
             * 一步 = 写槽 + 一发图 + 读设备 argmax。第一步仍走下面的直发(把平面副本/暂存那些懒分配全建好, 捕获态下不许分配)。
             * ★投机歇轮的步也走图★(2026-09-18): 图里的 hc_mean 核按设备位置把 main_hidden 落进环, 下次出草稿时按位置差补窗口,
             * 所以歇着的步不必再走直发付 5 ms 发射间隙 —— 以前 --dspark 一开, 连歇着的步都比纯解码慢(23.65 vs 25.9 t/s)。
             * ★先发图再 emit★: emit 是 fwrite+fflush 到文件, 实测 300 µs/步 —— 放在两步之间 GPU 就干等 300 µs,
             * 放在图跑着的时候做就是白赚。eos/上下文满的判断在发图之前(它们决定还发不发), 输出顺序与原来逐字相同。 */
            /* ★投机歇轮的步走图(2026-09-19 恢复)★: 09-18 第三版 2K 投机 + 图混跑 14 步 "illegal memory access" 已定罪 ——
             * 验证批(直发, n≤6)把后端暂存扩容换了指针, 图里烤死的旧指针失效; 现在 core_decode_graph.c 发图前对暂存代号,
             * 变了就重捕获(见 ds4_gpu_v41_scratch_generation)。fix4 那趟 58% 的步在歇, 每步直发比走图多付 3 ms。 */
            if (!draft_now && v41_graph_ready(&st) && tok != eos && st.n_past + 1 <= st.ctx) {
                int32_t nt = 0;
                if (!v41_graph_launch(e, &st, (int32_t)tok)) { ok = false; break; }
                if (emit && emit(tok, ud) != 0) { (void)v41_graph_wait(e, &st, &nt); break; }   /* 图已发, 等完再走 */
                if (!v41_graph_wait(e, &st, &nt)) { ok = false; break; }
                /* 采样开: 图末尾算的 argmax 不用, 从这一步的 logits 行(row 0, n=1)采; 关: nt 就是设备 argmax */
                if (rowbuf && !v41_next_token(&st, am, 0, rowbuf, &rng, &hist, &nt)) { ok = false; break; }
                tok = (int)nt;
                continue;
            }
            if (emit && emit(tok, ud) != 0) break;
            if (tok == eos) break;
            if (st.n_past + 1 > st.ctx) { fprintf(stderr, "\n[v41] 上下文满 %u\n", st.ctx); break; }
            uint32_t k = 0;
            int32_t batch[DS4_MTP_MAX_BLOCK + 1];
            batch[0] = (int32_t)tok;
            const double tr0 = now_sec();
            float sched_val = 0.f;                 /* 调度器预测的"产出/成本"(k=0 的轮也有, 诊断行要打) */
            bool drafted = false;
            const uint32_t pos_round = st.n_past;  /* 本轮块首位(tok)的绝对位置, 诊断行用 */
            if (draft_now && v41_draft_step(e, &st, &dr, (int32_t)tok, st.n_past - 1u)) {
                /* ★验证几位由置信度定(mtp-1.md M5′, core_draft_sched.c)★
                 * 以前这里钉死 3。钉死的毛病在两头: 文本好猜时少赚(README 英文满打满算能接受 1.39/5),
                 * 难猜时白读专家(金融文本 0.78/3, 每多验一位就多读一份 1.61 GB 的专家权重)。
                 * 调度器拿草稿器自己报的 conf 算"再验一位期望多拿多少 token / 多付多少成本", 取最优。
                 * --dspark-verify N 仍可钉死一个 k(诊断与逐档量字节账用)。 */
                drafted = true;
                const uint32_t use = g_ds4_v41_verify_k ? g_ds4_v41_verify_k
                                                       : v41_draft_pick_k(dr.host_conf, dr.block, &sched_val);
                /* ★预测连草稿钱都赚不回来 ⇒ 接下来几轮不出草稿★: 难文本上投机是净亏的, 而"亏不亏"
                 * 只有出过一次草稿才知道。歇 V41_SPEC_COOLDOWN 轮再试一次 —— 既不会一直亏,
                 * 也不会错过文本变好猜的那一段。只由 token 序列决定 ⇒ 温 0 下可复现。 */
                if (!g_ds4_v41_verify_k && sched_val < 1.0f) { spec_skip = V41_SPEC_COOLDOWN; spec_skipped++; }
                k = dr.block < use ? dr.block : use;
                for (uint32_t i = 0; i < k; i++) batch[i + 1u] = dr.host_ids[i + 1u];
            }
            const uint32_t nb = 1u + k;
            const double tr1 = now_sec();
            if (k && !v41_spec_snapshot(&st, nb)) { ok = false; break; }
            const double tr2 = now_sec();
            if (!v41_forward(e, &st, batch, nb)) { ok = false; break; }
            if (k && !ds4_gpu_synchronize()) { ok = false; break; }   /* 分账要真壁钟, 不同步量到的是发射时间 */
            const double tr3 = now_sec();
            /* 逐位取主模型的贪心结果, 与草稿比: 第 i 位的 logits 预测的是 batch[i] 之后那一位 */
            int32_t want[DS4_MTP_MAX_BLOCK + 1];
            for (uint32_t i = 0; i < nb; i++)
                if (!ds4_gpu_v41_argmax_tensor(am, st.logits, i, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(am, 0, &want[i], 4)) { ok = false; break; }
            if (!ok) break;
            /* 采样开时投机已拒 ⇒ nb 恒 1, 只有 want[0]; 直发这一步(暖身步/捕获失败的重来路)也按采样取 */
            if (rowbuf && !v41_next_token(&st, am, 0, rowbuf, &rng, &hist, &want[0])) { ok = false; break; }
            if (k) { ms_draft += (tr1 - tr0) * 1e3; ms_snap += (tr2 - tr1) * 1e3;
                     ms_verify += (tr3 - tr2) * 1e3; ms_argmax += (now_sec() - tr3) * 1e3; }
            uint32_t a = 0;
            while (a < k && want[a] == batch[a + 1u]) a++;   /* 接受最长前缀 */
            /* ★每出一次草稿打一行账(2026-09-19, --emit-trace 或 --v41-prof)★: 位置 / 调度器选的 k 与预测比值 / 实际接受 /
             * 五位的 sigmoid(conf)。k=0 的轮(草稿白跑、要歇 16 步)也打 —— 调度器"该不该歇"的判决只能拿这张表复核:
             * 每格 conf 对上实际接受率才算校准, 对不上就是调度器在按错的概率算账。 */
            if (drafted && (g_ds4_v41_prof || g_ds4_v41_emit_trace)) {
                fprintf(stderr, "[dspark] pos %u k %u val %.3f acc %u conf", pos_round, k, (double)sched_val, a);
                for (uint32_t i = 0; i < dr.block; i++) {
                    const float c = dr.host_conf[i];
                    fprintf(stderr, " %.3f", isfinite(c) ? 1.0 / (1.0 + exp(-(double)c)) : 0.0);
                }
                if (g_ds4_v41_prof) {   /* 草稿这几位 vs 主模型自己的几位 —— 看是"接近但不同"还是"完全不搭" */
                    fprintf(stderr, " | 草稿");
                    for (uint32_t i = 0; i < k; i++) fprintf(stderr, " %d", batch[i + 1u]);
                    fprintf(stderr, " | 主模型");
                    for (uint32_t i = 0; i < nb; i++) fprintf(stderr, " %d", want[i]);
                }
                fprintf(stderr, "\n");
            }
            if (k) {
                spec_rounds++; spec_acc += a; spec_hist[a]++;
                const double tb0 = now_sec();
                if (!v41_spec_rollback(&st, 1u + a)) { ok = false; break; }
                ms_snap += (now_sec() - tb0) * 1e3;
            }
            for (uint32_t i = 0; i < a; i++) {   /* 白赚的那几位: 草稿与主模型一致, 直接吐 */
                produced++;
                const int t = (int)batch[i + 1u];
                /* 位置 = 回滚后的 n_past 减去还没吐的那几位(与纯解码那条打印的是同一个绝对位置口径) */
                if (g_ds4_v41_emit_trace) fprintf(stderr, "[emit] %u %d\n", st.n_past - a + i, t);
                if ((emit && emit(t, ud) != 0) || t == eos) { produced = n_predict; break; }
            }
            tok = (int)want[a];
        }
        if (spec_rounds) {
            fprintf(stderr, "\n[v41] DSpark: %u 轮(调度器判亏本歇了 %u 次), 平均接受 %.2f/%u 位; 直方图",
                    spec_rounds, spec_skipped, (double)spec_acc / spec_rounds, dr.block);
            for (uint32_t i = 0; i <= dr.block; i++) fprintf(stderr, " %u:%u", i, spec_hist[i]);
            const double R = (double)spec_rounds, ea = 1.0 + (double)spec_acc / R;
            const double per = (ms_draft + ms_snap + ms_verify + ms_argmax) / R;
            fprintf(stderr, "\n[v41] 一轮 %.1f ms = 草稿 %.1f + 快照回滚 %.1f + 验证 %.1f + argmax %.1f"
                            "; 产出 %.2f token ⇒ %.1f ms/token\n",
                    per, ms_draft / R, ms_snap / R, ms_verify / R, ms_argmax / R, ea, per / ea);
        }
        if (spec) v41_draft_free(&dr);
        const double t2 = now_sec();
        if (produced > 1) fprintf(stderr, "\n[v41] decode %d token %.1fs (%.2f t/s)\n", produced - 1, t2 - t1, (double)(produced - 1) / (t2 - t1 + 1e-9));
        rc = ok ? 0 : 1;
    } while (0);
    free(rowbuf); free(hist.tok); free((void *)hist.brk);
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
