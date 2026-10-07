/* core_ptrain_layer.c — 后训练 ③ 第八版第二阶段: 解码器段逐层反传(2026-10-01)。总述见 core_ptrain.h。
 *
 * 为什么要逐层: 末层放大器只改读出, 模型被追问时会把旧信念说回来(1+1 实验); 文献里文档写进权重的方案都把低秩件铺满各层的 MLP。
 * 做法 = 梯度检查点: 训练前向时只存每个训练层入口的 hc(bf16)、pre_mix 与注意力用的压缩组号; 反传时从第 hi 层往下,
 * 每层先用引擎自己的前向函数把本层重算一遍(中间量抄进 pt_layer_buf), 再按 ffn 半层 → 注意力半层的倒序推梯度:
 *   hc_post(ffn) → 放大器 / shared 专家 / routed 专家 / 路由权重 → ffn_norm → hc_pre → mHC 混合系数(ffn)
 *   → hc_post(attn) → wo_b/wo_a → RoPE → 稀疏注意力 → q 路(q_b/q_norm/q_a) 与 kv 路(kv_norm/kv) → attn_norm → hc_pre → mHC 混合系数(attn)
 * 不求导的东西(离散或常量): 路由选哪几个专家、indexer 选哪些压缩组(含索引键那一支)、源层在训练段之下时它的压缩 KV、
 * engram 查表得来的 key/value、各处 bf16/fp8/fp4 舍入(直通)。
 * 分界层以下(10-01 夜补全, 训练段可到 L0):
 *   - kv 源层把 xn 压成压缩 KV, 之后每一层的注意力都读它 ⇒ 读取层倒着算时把"对压缩行的梯度"累加进 lb.gcomp[源层];
 *     轮到源层时(它自己也读自己的压缩行, 先加进去)经 RoPE 反向 → 压缩器 RMSNorm 反向 → 池化反向(ratio>1) → 两块投影的转置乘, 加进本层 g_xn。
 *   - engram 层: 存档是门之前的 hc, 重算时用存档的 key|value 先过门; 本层倒推完再过门的反向, 得到对门之前 hc 的梯度往下传。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define PT_LB_LIST(b) &b->mixa, &b->posta, &b->comba, &b->prea, &b->xa, &b->xna, &b->qr, &b->kv, &b->o, &b->hcin, \
    &b->mixf, &b->postf, &b->combf, &b->pref, &b->xf, &b->xnf, &b->g_hcm, &b->g_hcin, &b->g_y, &b->g_xnf, &b->g_xf, &b->g_pre_a, \
    &b->g_postf, &b->g_combf, &b->g_attn, &b->g_posta, &b->g_comba, &b->g_low, &b->g_o, &b->g_q, &b->g_kvn, &b->g_qrn, &b->g_qr, \
    &b->g_kv, &b->g_xna, &b->g_xa, &b->g_prein, &b->g_sh, &b->g_sg, &b->g_su, &b->g_rw, &b->g_glog, \
    &b->cckv, &b->ccsc, &b->cpool, &b->cgpool, &b->cgkv, &b->cgsc, &b->ekvf, &b->hpre

bool pt_layer_alloc(const pt_cfg *c, pt_run *r) {
    pt_layer_buf *b = &r->lb;
    const uint64_t n = c->maxlen, E = DS4_N_EMBD, HC = DS4_N_HC, MIX = 2u * HC + HC * HC, HD = DS4_N_HEAD_DIM, QH = (uint64_t)DS4_N_HEAD * HD;
    const uint64_t Q = DS4_N_LORA_Q, LOW = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT, KU = DS4_N_EXPERT_USED;
    bool ok = true;
    const struct { ds4_gpu_tensor **t; uint64_t bytes; } plan[] = {
        { &b->mixa, n * MIX * 4 }, { &b->posta, n * HC * 4 }, { &b->comba, n * HC * HC * 4 }, { &b->prea, n * HC * 4 },
        { &b->xa, n * E * 4 }, { &b->xna, n * E * 4 }, { &b->qr, n * Q * 4 }, { &b->kv, n * HD * 4 }, { &b->o, n * QH * 4 },
        { &b->hcin, n * HC * E * 4 }, { &b->mixf, n * MIX * 4 }, { &b->postf, n * HC * 4 }, { &b->combf, n * HC * HC * 4 },
        { &b->pref, n * HC * 4 }, { &b->xf, n * E * 4 }, { &b->xnf, n * E * 4 },
        { &b->g_hcm, n * HC * E * 4 }, { &b->g_hcin, n * HC * E * 4 }, { &b->g_y, n * E * 4 }, { &b->g_xnf, n * E * 4 }, { &b->g_xf, n * E * 4 },
        { &b->g_pre_a, n * HC * 4 }, { &b->g_postf, n * HC * 4 }, { &b->g_combf, n * HC * HC * 4 }, { &b->g_attn, n * E * 4 },
        { &b->g_posta, n * HC * 4 }, { &b->g_comba, n * HC * HC * 4 }, { &b->g_low, n * LOW * 4 }, { &b->g_o, n * QH * 4 }, { &b->g_q, n * QH * 4 },
        { &b->g_kvn, n * HD * 4 }, { &b->g_qrn, n * Q * 4 }, { &b->g_qr, n * Q * 4 }, { &b->g_kv, n * HD * 4 }, { &b->g_xna, n * E * 4 },
        { &b->g_xa, n * E * 4 }, { &b->g_prein, n * HC * 4 }, { &b->g_sh, n * FF * 4 }, { &b->g_sg, n * FF * 4 }, { &b->g_su, n * FF * 4 },
        { &b->g_rw, n * KU * 4 }, { &b->g_glog, n * NE * 4 },
    };
    for (size_t i = 0; i < sizeof plan / sizeof plan[0] && ok; i++) *plan[i].t = v41_alloc(plan[i].bytes, &ok);
    bool any_src = false, any_eng = false;
    int32_t *pg = xmalloc((size_t)n * 4);
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        r->save.hc_in[il] = v41_alloc(n * HC * E * 2, &ok);
        r->save.pm_in[il] = v41_alloc(n * HC * 4, &ok);
        r->save.idx[il] = v41_alloc(n * DS4_N_INDEXER_TOP_K * 4, &ok);
        r->save.sel[il] = v41_alloc(n * KU * 4, &ok);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ok && g_ds4_v41.is_kv_source[il] && ratio) {   /* 源层在训练段里: 压缩行梯度 + 组位置表(从位置 0 起, 第 g 组在 g·ratio) */
            any_src = true;
            b->gcomp[il] = v41_alloc(n * HD * 4, &ok); b->posg[il] = v41_alloc(n * 4, &ok);
            for (uint64_t g = 0; g < n; g++) pg[g] = (int32_t)(g * ratio);
            if (ok) ok = ds4_gpu_tensor_write(b->posg[il], 0, pg, n * 4) != 0;
        }
        if (ok && g_ds4_v41.engram_index_of[il] >= 0) { any_eng = true; r->save.ekv[il] = v41_alloc(n * (HC + 1u) * E * 2, &ok); }
    }
    free(pg);
    if (ok && any_src) { b->cckv = v41_alloc(n * HD * 4, &ok); b->ccsc = v41_alloc(n * HD * 4, &ok); b->cpool = v41_alloc(n * HD * 4, &ok);
                         b->cgpool = v41_alloc(n * HD * 4, &ok); b->cgkv = v41_alloc(n * HD * 4, &ok); b->cgsc = v41_alloc(n * HD * 4, &ok); }
    if (ok && any_eng) { b->ekvf = v41_alloc(n * (HC + 1u) * E * 4, &ok); b->hpre = v41_alloc(n * HC * E * 4, &ok); }
    b->ready = ok;
    if (!ok) fprintf(stderr, "ds4: [ptrain] 逐层反传缓冲分配失败(maxlen %u)\n", c->maxlen);
    return ok;
}

void pt_layer_free(pt_run *r) {
    pt_layer_buf *b = &r->lb;
    ds4_gpu_tensor **all[] = { PT_LB_LIST(b) };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) { ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor **pl[] = { &r->save.idx[il], &r->save.ekv[il], &r->save.sel[il], &b->gcomp[il], &b->posg[il] };
        for (size_t i = 0; i < sizeof pl / sizeof pl[0]; i++) if (*pl[i]) { ds4_gpu_tensor_free(*pl[i]); *pl[i] = NULL; }
    }
    b->ready = 0;
}

#define PT_CP(dst, src, bytes) ds4_gpu_tensor_copy((dst), 0, (src), 0, (bytes))
#define PT_MT(gx, w, in, out, gy, acc) ds4_gpu_bwd_matmul_t_tensor((gx), m->map, m->size, (w)->type, (w)->abs_offset, (in), (out), (gy), n, (acc))

static bool pt_has_engram(const ds4_v41_state *st, uint32_t il) { return !st->no_engram && g_ds4_v41.engram_index_of[il] >= 0; }

/* 注意力缓存段的重算/反传按题分段(合批见 core_ptrain_pack.c): 一段 = 一道题在批态里的行区间 + 它自己的组数/topk/窗口/压缩行(g0 = 它在
 * 源层压缩行里的起始组号, 压缩行梯度也从这一组起累加)。单题路就是一段: 行 0 起整段, 组数等存档在 r->save。
 * 各题的 idx 存档都在 save.idx 第 r0·TOPK 格起(行距是本题的 topk), 单题路 r0 = 0, 与 v41_layer 存的位置一致。 */
typedef struct { uint32_t r0, n, ng, topk, iratio, g0; ds4_gpu_tensor *win, *comp; } pt_seg;
static uint32_t pt_segs(const pt_run *r, uint32_t il, uint32_t n, pt_seg *sg) {
    const pt_pack *pk = &r->pk; const ds4_v41_state *st = &r->st; const v41_tsave *sv = &r->save;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const int16_t src = ratio ? g_ds4_v41.kv_source_of[il] : -1;
    if (!pk->nb) { sg[0] = (pt_seg){ 0, n, sv->ng[il], sv->topk[il], sv->iratio[il], 0, st->win[il], src >= 0 ? st->comp_kv[src] : NULL }; return 1; }
    for (uint32_t i = 0; i < pk->nb; i++)
        sg[i] = (pt_seg){ pk->r0[i], pk->n[i], pk->ng[i][il], pk->topk[i][il], pk->iratio[i][il], src >= 0 ? pk->g0[i][src] : 0u,
                          pk->rq[i].win[il], src >= 0 ? pk->rq[i].comp_kv[src] : NULL };
    return pk->nb;
}
static ds4_gpu_tensor *pt_idx_view(const pt_run *r, uint32_t il, const pt_seg *g) {
    return ds4_gpu_tensor_view(r->save.idx[il], (uint64_t)g->r0 * DS4_N_INDEXER_TOP_K * 4, (uint64_t)g->n * g->topk * 4);
}
static void pt_free_views(ds4_gpu_tensor **v, size_t n) { for (size_t i = 0; i < n; i++) if (v[i]) ds4_gpu_tensor_free(v[i]); }

/* 重算第 il 层(从存档的入口起, 与训练前向同一串引擎函数), 中间量抄进 lb; 结束时 st 里留着本层 MoE 的 y/sel/rw/glog/sg/su 与 attn_out */
static bool pt_layer_recompute(ds4_engine *e, pt_run *r, uint32_t il, uint32_t n) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    pt_layer_buf *b = &r->lb; const v41_tsave *sv = &r->save;
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, MIX = 2u * HC + HC * HC, HD = DS4_N_HEAD_DIM, QH = (uint64_t)DS4_N_HEAD * HD, SWA = DS4_N_SWA;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    st->n = n; st->pos0 = 0; st->graph = 0;
    /* L0 的入口是词嵌入(q4_K 解码值, 不在 bf16 格点上): 按 bf16 存档是有损的(10-01 重算对拍: 只有 L0 出口差 2.9e-3, 其余各层恰好 0),
     * 所以 L0 从 token 直接重算(st->tok 还是这一题的), 别的层入口都是 hc_post 舍过 bf16 的, 存档无损 */
    bool ok = il == 0u ? v41_embed(m, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, n, E) && ds4_gpu_v41_expand_hc_tensor(b->hcin, st->x, (uint32_t)E, (uint32_t)HC, n)
                       : ds4_gpu_bwd_unpack_bf16_tensor(b->hcin, sv->hc_in[il], n * HC * E);
    if (ok && pt_has_engram(st, il))   /* 存档是门之前的 hc: 先用存档的 key|value 过门(与前向同一个核), b->hcin 变成门之后、hpre 留门之前 */
        ok = sv->ekv[il] && ds4_gpu_bwd_unpack_bf16_tensor(b->ekvf, sv->ekv[il], n * (HC + 1u) * E) && PT_CP(b->hpre, b->hcin, n * HC * E * 4) &&
             ds4_gpu_v41_engram_gate_tensor(b->hcin, b->ekvf, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, (uint32_t)E, (uint32_t)HC, n, DS4_RMS_EPS);
    ok = ok && PT_CP(st->hc, b->hcin, n * HC * E * 4) &&
              PT_CP(st->pre_mix, sv->pm_in[il], n * HC * 4) && v41_hc_half(e, st, l, true) &&
              PT_CP(b->mixa, st->mix, n * MIX * 4) && PT_CP(b->posta, st->post, n * HC * 4) && PT_CP(b->comba, st->comb, n * HC * HC * 4) &&
              PT_CP(b->prea, st->pre_mix, n * HC * 4) && PT_CP(b->xa, st->x, n * E * 4) && PT_CP(b->xna, st->xn, n * E * 4) &&
              v41_attn_in(e, st, il) && PT_CP(b->qr, st->qr, n * DS4_N_LORA_Q * 4) && PT_CP(b->kv, st->kv, n * HD * 4) &&
              ds4_gpu_tensor_copy(st->win[il], SWA * HD * 4, st->kvn, 0, n * HD * 4);   /* 合批时各题的块区首尾相接, 合起来也正是这一段 */
    pt_seg sg[PT_PACK_MAX];
    const uint32_t nsg = pt_segs(r, il, n, sg);
    for (uint32_t i = 0; ok && i < nsg; i++) {   /* 稀疏注意力按题: 各题自己的窗口 / 压缩行 / 冻结的选组 */
        const pt_seg *g = &sg[i];
        const bool hasc = ratio && g->ng && g->topk;
        ds4_gpu_tensor *v[3] = { pt_rows(st->o, g->r0, g->n, QH), pt_rows(st->q, g->r0, g->n, QH), hasc ? pt_idx_view(r, il, g) : NULL };
        ok = v[0] && v[1] && (!hasc || v[2]) &&
             ds4_gpu_v41_sparse_attn_tensor(v[0], v[1], g->win, hasc ? g->comp : NULL, v[2], m->map, m->size, l->attn_sinks->abs_offset, g->n, 0, (uint32_t)SWA,
                                            g->ng, hasc ? g->topk : 0u, hasc ? g->iratio : 0u, DS4_N_HEAD, (uint32_t)HD,
                                            (float)(1.0 / sqrt((double)HD)), 0, 1, 0u, NULL, 0u) != 0;
        pt_free_views(v, 3);
    }
    if (ok) ok = PT_CP(b->o, st->o, n * QH * 4) && v41_attn_out(e, st, il) &&
                 ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, (uint32_t)E, (uint32_t)HC, n);
    if (ok) { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }   /* st->hc = hc_mid, st->hc2 = 本层入口 */
    /* st->hc(注意力半层出口 = ffn 半层入口)与 st->q(RoPE 之后的 q)在重算剩下的部分与整个反传里都不再被写, 反传直接读, 不另抄
     * (全层训练只剩 ~3 GB 余量, 这两份拷贝 218 MB)。改动重算/反传次序时要重核这一条。 */
    /* 专家前向开着截留(ds4_gpu_bwd_moe_capture): 本层逐对 H_g/A/H_u/O 留给紧接着的 routed 专家反向直接用, 不再在反传里把它算第二遍
     * (10-02 逐核表: 前向/重算/反传三遍专家前向合计 37%)。只开这一发, 前向与评估路都关着。 */
    if (ok) ok = v41_hc_half(e, st, l, false) &&
                 PT_CP(b->mixf, st->mix, n * MIX * 4) && PT_CP(b->postf, st->post, n * HC * 4) && PT_CP(b->combf, st->comb, n * HC * HC * 4) &&
                 PT_CP(b->pref, st->pre_mix, n * HC * 4) && PT_CP(b->xf, st->x, n * E * 4) && PT_CP(b->xnf, st->xn, n * E * 4) &&
                 ds4_gpu_bwd_moe_capture(1) && v41_moe(m, l, st, il);
    (void)ds4_gpu_bwd_moe_capture(0);
    return ok;
}

/* 源层的压缩器反向: lb.gcomp[il](读取层与本层注意力累加好的、对 RoPE 之后压缩行的梯度)→ 逆 RoPE → RMSNorm 反向 → 池化反向 → 投影转置乘,
 * 加进 g_xna。先按 v41_compress_source 同一串核重算归一化之前的 pooled(从位置 0 起一块, 尾巴不满一组的行不进池化)。
 * 按题调: 第 r0 行起 n 行是一道题, 它的组在 gcomp 第 g0 组起(组位置表 posg 从 0 起, 每题的位置都从 0 起); 暂存 cckv..cgsc 各题依次复用。 */
static bool pt_compress_bwd(ds4_engine *e, pt_run *r, uint32_t il, uint32_t r0, uint32_t n, uint32_t g0) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    pt_layer_buf *b = &r->lb;
    const uint32_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, ratio = ds4_layer_compress_ratio(il), ng = n / ratio;
    if (!ng) return true;   /* 一组都没凑满: 前向没产压缩行, 也就没人读 */
    ds4_gpu_tensor *v[3] = { pt_rows(b->xna, r0, n, E), pt_rows(b->g_xna, r0, n, E), pt_rows(b->gcomp[il], g0, ng, HD) };
    ds4_gpu_tensor *xna = v[0], *gxna = v[1], *gc = v[2];
    bool ok = xna && gxna && gc;
    if (ok && ratio > 1u) ok = ds4_gpu_v41_matmul_bf16_tensor(b->cckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, xna, n) &&
                               ds4_gpu_v41_matmul_bf16_tensor(b->ccsc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, xna, n) &&
                               ds4_gpu_v41_compress_pool_tensor(b->cpool, b->cckv, b->ccsc, n, ratio, HD);
    else if (ok) ok = ds4_gpu_v41_matmul_bf16_tensor(b->cpool, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, xna, n) &&
                      ds4_gpu_v41_round_bf16_tensor(b->cpool, (uint64_t)n * HD);
    ok = ok && ds4_gpu_bwd_rope_tensor(gc, b->posg[il], ng, 1, HD, DS4_N_ROT, DS4_COMPRESS_ROPE_FREQ_BASE, (uint32_t)DS4_ROPE_ORIG_CTX,
                                       DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
         ds4_gpu_bwd_rms_norm_tensor(b->cgpool, gc, b->cpool, m->map, m->size, l->attn_compressor_norm->abs_offset, HD, ng, DS4_RMS_EPS, 0);
    if (ratio > 1u) ok = ok && ds4_gpu_tensor_fill_f32(b->cgkv, 0.f, (uint64_t)n * HD) && ds4_gpu_tensor_fill_f32(b->cgsc, 0.f, (uint64_t)n * HD) &&
                         ds4_gpu_bwd_compress_pool_tensor(b->cgkv, b->cgsc, b->cgpool, b->cckv, b->ccsc, ng, ratio, HD) &&
                         PT_MT(gxna, l->attn_compressor_kv, E, HD, b->cgkv, 1) && PT_MT(gxna, l->attn_compressor_gate, E, HD, b->cgsc, 1);
    else ok = ok && PT_MT(gxna, l->attn_compressor_kv, E, HD, b->cgpool, 1);
    pt_free_views(v, 3);
    if (!ok) fprintf(stderr, "ds4: [ptrain] L%u 压缩器反向失败(ratio %u, 行 %u+%u, %u 组)\n", il, ratio, r0, n, ng);
    return ok;
}

/* prof: 段边界同步后取墙钟(只作诊断; 不开时零开销)。整步各段(前向/损失/出口/Adam)也用它, 见 core_ptrain.h 的 tm 下标 */
double pt_tick(const pt_cfg *c) { if (c->prof) (void)ds4_gpu_synchronize(); return c->prof ? now_sec() : 0.0; }

static double pt_row_absmax(const float *v, uint64_t n) { double m = 0; for (uint64_t i = 0; i < n; i++) if (fabs(v[i]) > m) m = fabs(v[i]); return m; }
/* dbglayer 探针: 第一题那几行(合批 = 第 0 段, 单题 = 整段)的梯度范数 + 最大行(尖峰压在哪个 token 上), 同一层单题路 vs 合批并排看哪一步先分叉。
 * 10-02 用它追到 STE 尖峰: token 10 的入口梯度从 L20 起逐层放大, 到 L0 占整题 80%(fable5 同日条) */
static void pt_dbg(const pt_cfg *c, const pt_run *r, uint32_t il, const char *tag, const ds4_gpu_tensor *t, uint64_t rowf) {
    if (c->dbg_layer != (int32_t)il && !(c->dbg_layer == 99 && !strcmp(tag, "层入口 g_hc"))) return;
    const uint32_t n0 = r->pk.nb ? r->pk.n[0] : r->st.n;
    float *h = xmalloc((size_t)n0 * rowf * 4);   /* 读回逐行范数: 尖峰压在哪一行(token)上, 是哪一步先冒出来的 */
    (void)ds4_gpu_synchronize();
    if (ds4_gpu_tensor_read(t, 0, h, (uint64_t)n0 * rowf * 4)) {
        double s = 0, best = -1; uint32_t bi = 0;
        for (uint32_t q = 0; q < n0; q++) {
            double rs = 0;
            for (uint64_t k = 0; k < rowf; k++) rs += (double)h[q * rowf + k] * h[q * rowf + k];
            s += rs;
            if (rs > best) { best = rs; bi = q; }
        }
        fprintf(stderr, "ds4: [dbg L%02u %s] %-22s |·| = %.6e(%u 行), 最大行 %u: %.4e(占 %.0f%%), 该行最大绝对值 %.3g\n", il, r->pk.nb ? "合批" : "单题", tag,
                sqrt(s), n0, bi, sqrt(best), s > 0 ? 100.0 * best / s : 0.0, pt_row_absmax(h + (uint64_t)bi * rowf, rowf));
    }
    free(h);
}

static bool pt_layer_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t il, uint32_t n) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    pt_layer_buf *b = &r->lb;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, FF = DS4_N_FF_EXP;
    const uint32_t G = DS4_N_OUT_GROUP, LOW = DS4_N_OUT_GROUP * DS4_N_LORA_O, ratio = ds4_layer_compress_ratio(il);
    const float theta = ratio ? DS4_COMPRESS_ROPE_FREQ_BASE : DS4_ROPE_FREQ_BASE;
    const uint32_t osl = ratio ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u;
    v41_tsave *keep = st->tsave;
    st->tsave = NULL;   /* 重算不再存档 */
    double t0 = pt_tick(c);
    bool ok = pt_layer_recompute(e, r, il, n);
    double t1 = pt_tick(c);
    if (ok && c->rccheck && il < c->layer_hi && r->save.hc_in[il + 1u]) {
        /* 重算对拍: 补算本层出口 hc, 与前向存档的下一层入口比。两者都在 bf16 格点上(hc_post 出口舍过 bf16), 重算忠实就该恰好 0;
         * 不是 0 = 重算没复现前向(组号/窗口/压缩行/路由哪一样对不上), 反传就是对着另一个函数求的导。g_hcm/g_hcin 此刻还没用, 借作暂存。 */
        double ref = 0, d = 0;
        ok = ds4_gpu_v41_hc_post_tensor(b->g_hcm, st->y, st->hc, b->postf, b->combf, E, HC, n) &&
             ds4_gpu_bwd_unpack_bf16_tensor(b->g_hcin, r->save.hc_in[il + 1u], (uint64_t)n * HC * E) &&
             ds4_gpu_bwd_sumsq_tensor(b->g_hcin, (uint64_t)n * HC * E, &ref) &&
             ds4_gpu_bwd_axpy_tensor(b->g_hcm, b->g_hcin, -1.f, (uint64_t)n * HC * E) &&
             ds4_gpu_bwd_sumsq_tensor(b->g_hcm, (uint64_t)n * HC * E, &d);
        fprintf(stderr, "ds4: [ptrain 重算对拍] L%02u 重算出口 vs 前向存档的 L%02u 入口: 相对差 %.3e(|ref| %.3e)\n", il, il + 1u, sqrt(d / (ref > 0 ? ref : 1)), sqrt(ref));
    }
    /* ---- ffn 半层 ---- */
    pt_dbg(c, r, il, "进来 g_hc(层出口)", r->ghc, (uint64_t)HC * E);
    if (ok) ok = ds4_gpu_tensor_fill_f32(b->g_hcm, 0.f, (uint64_t)n * HC * E) &&
                 ds4_gpu_bwd_hc_post_tensor(b->g_y, b->g_hcm, b->g_postf, b->g_combf, r->ghc, st->y, st->hc, b->postf, b->combf, E, HC, n) &&
                 ds4_gpu_tensor_fill_f32(b->g_xnf, 0.f, (uint64_t)n * E);
    pt_dbg(c, r, il, "g_y(MoE 出口)", b->g_y, E); pt_dbg(c, r, il, "g_hcm(hc_post 残差)", b->g_hcm, (uint64_t)HC * E);
    if (ok && r->gA[il]) ok = ds4_gpu_bwd_amp_tensor(r->gA[il], r->gB[il], b->g_xnf, b->g_y, b->xnf, st->ampA[il], st->ampB[il], r->T, r->gT, n, E, r->K);
    pt_dbg(c, r, il, "g_xnf 放大器后", b->g_xnf, E);
    if (ok) ok = PT_MT(b->g_sh, l->ffn_down_shexp, FF, E, b->g_y, 0) &&
                 ds4_gpu_bwd_swiglu_tensor(b->g_sg, b->g_su, b->g_sh, st->sg, st->su, (uint64_t)n * FF, DS4_SWIGLU_CLAMP_EXP) &&
                 PT_MT(b->g_xnf, l->ffn_gate_shexp, E, FF, b->g_sg, 1) && PT_MT(b->g_xnf, l->ffn_up_shexp, E, FF, b->g_su, 1);
    pt_dbg(c, r, il, "g_xnf 共享专家后", b->g_xnf, E);
    pt_dbg(c, r, il, "共享: sg(前向)", st->sg, FF); pt_dbg(c, r, il, "共享: su(前向)", st->su, FF); pt_dbg(c, r, il, "共享: g_sh", b->g_sh, FF);
    pt_dbg(c, r, il, "共享: g_sg", b->g_sg, FF); pt_dbg(c, r, il, "共享: g_su", b->g_su, FF); pt_dbg(c, r, il, "xnf(前向)", b->xnf, E);
    double t2 = pt_tick(c);
    if (ok) {
        char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il);
        const ds4_tensor *blob = model_find_tensor(m, nm);
        ok = blob && ds4_gpu_bwd_routed_moe_tensor(b->g_xnf, b->g_rw, b->g_y, b->xnf, st->sel, st->rw, m->map, m->size, blob->abs_offset, blob->bytes,
                                                   E, FF, E, DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, il, n) &&
             ds4_gpu_bwd_router_tensor(b->g_glog, b->g_rw, st->sel, st->glog, n, DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE) &&
             PT_MT(b->g_xnf, l->ffn_gate_inp, E, DS4_N_EXPERT, b->g_glog, 1);
    }
    pt_dbg(c, r, il, "g_xnf routed+路由后", b->g_xnf, E);
    double t3 = pt_tick(c);
    if (ok) ok = ds4_gpu_bwd_rms_norm_tensor(b->g_xf, b->g_xnf, b->xf, m->map, m->size, l->ffn_norm->abs_offset, E, n, DS4_RMS_EPS, 0);
    pt_dbg(c, r, il, "g_xf(ffn_norm 前)", b->g_xf, E);
    if (ok) ok = ds4_gpu_bwd_hc_pre_tensor(b->g_hcm, b->g_pre_a, b->g_xf, st->hc, b->prea, E, HC, n);
    pt_dbg(c, r, il, "g_hcm hc_pre 后", b->g_hcm, (uint64_t)HC * E); pt_dbg(c, r, il, "g_pre_a", b->g_pre_a, HC);
    pt_dbg(c, r, il, "g_postf", b->g_postf, HC); pt_dbg(c, r, il, "g_combf", b->g_combf, (uint64_t)HC * HC); pt_dbg(c, r, il, "进来 g_pre(层出口)", r->gpre, HC);
    if (ok) ok = ds4_gpu_bwd_hc_mix_tensor(b->g_hcm, r->gpre, b->g_postf, b->g_combf, st->hc, b->mixf, m->map, m->size, l->hc_ffn_fn->abs_offset,
                                           l->hc_ffn_scale->abs_offset, l->hc_ffn_base->abs_offset, E, HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, n);
    pt_dbg(c, r, il, "g_hcm hc_mix 后(中段)", b->g_hcm, (uint64_t)HC * E);
    double t4 = pt_tick(c);
    /* hc 边界检查(中段): 此刻 g_hcm = 对注意力半层出口 hc 的梯度 */
    if (ok && r->hcap && r->hcap_mid && r->hcap_layer == (int32_t)il) ok = PT_CP(r->hcap, b->g_hcm, (uint64_t)n * HC * E * 4);
    /* ---- 注意力半层 ---- */
    if (ok) ok = ds4_gpu_tensor_fill_f32(b->g_hcin, 0.f, (uint64_t)n * HC * E) &&
                 ds4_gpu_bwd_hc_post_tensor(b->g_attn, b->g_hcin, b->g_posta, b->g_comba, b->g_hcm, st->attn_out, b->hcin, b->posta, b->comba, E, HC, n) &&
                 PT_MT(b->g_low, l->attn_output_b, LOW, E, b->g_attn, 0) &&
                 ds4_gpu_bwd_grouped_matmul_t_tensor(b->g_o, m->map, m->size, l->attn_output_a->type, l->attn_output_a->abs_offset, G,
                                                     (uint64_t)(NH / G) * HD, DS4_N_LORA_O, b->g_low, n, 0) &&
                 /* 前向在 o 上做的是逆旋转 ⇒ 反向转正向 */
                 ds4_gpu_bwd_rope_tensor(b->g_o, st->pos, n, NH, HD, DS4_N_ROT, theta, osl, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, false);
    pt_dbg(c, r, il, "注意力: 进来 g_hcm", b->g_hcm, (uint64_t)HC * E); pt_dbg(c, r, il, "注意力: g_attn", b->g_attn, E);
    pt_dbg(c, r, il, "注意力: g_o", b->g_o, (uint64_t)NH * HD); pt_dbg(c, r, il, "注意力: g_hcin hc_post 后", b->g_hcin, (uint64_t)HC * E);
    if (ok) {   /* 稀疏注意力反向按题(各题的窗口/压缩行/选组各自的; 压缩行梯度累加进 gcomp 里本题那几组) */
        const int16_t src = ratio ? g_ds4_v41.kv_source_of[il] : -1;
        const bool wantc = src >= 0 && (uint32_t)src >= c->layer_lo && b->gcomp[src];   /* 源层在训练段里才要对压缩行求导 */
        pt_seg sg[PT_PACK_MAX];
        const uint32_t nsg = pt_segs(r, il, n, sg);
        for (uint32_t i = 0; ok && i < nsg; i++) {
            const pt_seg *g = &sg[i];
            const bool hasc = ratio && g->ng && g->topk, gcw = hasc && wantc;
            ds4_gpu_tensor *v[7] = { pt_rows(b->g_q, g->r0, g->n, (uint64_t)NH * HD), pt_rows(b->g_kvn, g->r0, g->n, HD), pt_rows(b->g_o, g->r0, g->n, (uint64_t)NH * HD),
                                     pt_rows(b->o, g->r0, g->n, (uint64_t)NH * HD), pt_rows(st->q, g->r0, g->n, (uint64_t)NH * HD),
                                     gcw ? pt_rows(b->gcomp[src], g->g0, g->ng, HD) : NULL, hasc ? pt_idx_view(r, il, g) : NULL };
            ok = v[0] && v[1] && v[2] && v[3] && v[4] && (!gcw || v[5]) && (!hasc || v[6]) &&
                 ds4_gpu_bwd_sparse_attn_tensor(v[0], v[1], v[5], v[2], v[3], v[4], g->win, hasc ? g->comp : NULL, v[6], m->map, m->size,
                                                l->attn_sinks->abs_offset, g->n, DS4_N_SWA, g->ng, hasc ? g->topk : 0u, NH, HD, (float)(1.0 / sqrt((double)HD)));
            pt_free_views(v, 7);
        }
    }
    pt_dbg(c, r, il, "注意力: g_q 反传后", b->g_q, (uint64_t)NH * HD); pt_dbg(c, r, il, "注意力: g_kvn 反传后", b->g_kvn, HD);
    if (ok) ok = ds4_gpu_bwd_rope_tensor(b->g_q, st->pos, n, NH, HD, DS4_N_ROT, theta, osl, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
                 PT_MT(b->g_qrn, l->attn_q_b, Q, (uint64_t)NH * HD, b->g_q, 0) &&
                 ds4_gpu_bwd_rms_norm_tensor(b->g_qr, b->g_qrn, b->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS, 0) &&
                 PT_MT(b->g_xna, l->attn_q_a, E, Q, b->g_qr, 0);
    pt_dbg(c, r, il, "注意力: g_qrn", b->g_qrn, Q); pt_dbg(c, r, il, "注意力: g_qr", b->g_qr, Q); pt_dbg(c, r, il, "注意力: g_xna q 支后", b->g_xna, E);
    if (ok) ok = ds4_gpu_bwd_rope_tensor(b->g_kvn, st->pos, n, 1, HD, DS4_N_ROT, theta, osl, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
                 ds4_gpu_bwd_rms_norm_tensor(b->g_kv, b->g_kvn, b->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS, 0) &&
                 PT_MT(b->g_xna, l->attn_kv, E, HD, b->g_kv, 1);
    pt_dbg(c, r, il, "注意力: g_kv", b->g_kv, HD); pt_dbg(c, r, il, "注意力: g_xna kv 支后", b->g_xna, E);
    if (ok && b->gcomp[il]) {   /* 本层是训练段里的 kv 源层: 压缩器那一支也读 xn(按题, 各题的组从自己的位置 0 起) */
        pt_dbg(c, r, il, "源层: gcomp(压缩前)", b->gcomp[il], HD); pt_dbg(c, r, il, "源层: g_xna 压缩器前", b->g_xna, E);
        const uint32_t nb = r->pk.nb ? r->pk.nb : 1u;
        for (uint32_t i = 0; ok && i < nb; i++)
            ok = r->pk.nb ? pt_compress_bwd(e, r, il, r->pk.r0[i], r->pk.n[i], r->pk.g0[i][il]) : pt_compress_bwd(e, r, il, 0, n, 0);
        pt_dbg(c, r, il, "源层: g_xna 压缩器后", b->g_xna, E); pt_dbg(c, r, il, "源层: cpool(重算)", b->cpool, HD);
        pt_dbg(c, r, il, "源层: cgpool", b->cgpool, HD);
    }
    if (ok) ok = ds4_gpu_bwd_rms_norm_tensor(b->g_xa, b->g_xna, b->xa, m->map, m->size, l->attn_norm->abs_offset, E, n, DS4_RMS_EPS, 0);
    pt_dbg(c, r, il, "注意力: g_xa", b->g_xa, E); pt_dbg(c, r, il, "注意力: xa(前向)", b->xa, E); pt_dbg(c, r, il, "注意力: hcin(前向)", b->hcin, (uint64_t)HC * E);
    pt_dbg(c, r, il, "注意力: pm_in(前向)", r->save.pm_in[il], HC);
    if (ok) ok = ds4_gpu_bwd_hc_pre_tensor(b->g_hcin, b->g_prein, b->g_xa, b->hcin, r->save.pm_in[il], E, HC, n);
    pt_dbg(c, r, il, "注意力: g_hcin hc_pre 后", b->g_hcin, (uint64_t)HC * E); pt_dbg(c, r, il, "注意力: g_prein", b->g_prein, HC);
    if (ok) ok = ds4_gpu_bwd_hc_mix_tensor(b->g_hcin, b->g_pre_a, b->g_posta, b->g_comba, b->hcin, b->mixa, m->map, m->size, l->hc_attn_fn->abs_offset,
                                           l->hc_attn_scale->abs_offset, l->hc_attn_base->abs_offset, E, HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, n);
    pt_dbg(c, r, il, "注意力: g_hcin hc_mix 后", b->g_hcin, (uint64_t)HC * E);
    /* engram 层: g_hcin 此刻是对门之后 hc 的梯度, 过门的反向变成对门之前(= 层入口)hc 的梯度 */
    if (ok && pt_has_engram(st, il))
        ok = ds4_gpu_bwd_engram_gate_tensor(b->g_hcin, b->hpre, b->ekvf, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, E, HC, n, DS4_RMS_EPS);
    if (ok) ok = PT_CP(r->ghc, b->g_hcin, (uint64_t)n * HC * E * 4) && PT_CP(r->gpre, b->g_prein, (uint64_t)n * HC * 4);
    double t5 = pt_tick(c);
    if (c->prof) { r->tm[PT_TM_RECOMP] += t1 - t0; r->tm[PT_TM_ROUTED] += t3 - t2; r->tm[PT_TM_ATTN] += t5 - t4; r->tm[PT_TM_LREST] += (t2 - t1) + (t4 - t3); }
    st->tsave = keep;
    if (!ok) fprintf(stderr, "ds4: [ptrain] L%u 逐层反传失败(n %u)\n", il, n);
    return ok;
}

/* ---- poison=1 毒值测试(只作诊断) ---- 运行期造 NaN(-ffast-math 下常量 NaN 会被折掉), 判 NaN 也按位判 */
static float pt_nanf(void) { volatile uint32_t u = 0x7fc00000u; const uint32_t v = u; float f; memcpy(&f, &v, 4); return f; }
static bool pt_is_nan(double x) { uint64_t b; memcpy(&b, &x, 8); return ((b >> 52) & 0x7ffu) == 0x7ffu && (b & 0xfffffffffffffull); }
static void pt_poison_rows(ds4_gpu_tensor *t, uint32_t n, uint64_t rowf) {   /* 第 n 行以后填 NaN */
    if (!t) return;
    const uint64_t have = ds4_gpu_tensor_bytes(t) / 4u, from = (uint64_t)n * rowf;
    if (from >= have) return;
    ds4_gpu_tensor *v = ds4_gpu_tensor_view(t, from * 4u, (have - from) * 4u);
    if (v) { (void)ds4_gpu_tensor_fill_f32(v, pt_nanf(), have - from); ds4_gpu_tensor_free(v); }
}
static void pt_poison_layer(pt_run *r, uint32_t il, uint32_t n) {
    pt_layer_buf *b = &r->lb; ds4_v41_state *st = &r->st;
    ds4_gpu_tensor **all[] = { PT_LB_LIST(b) };   /* 梯度暂存 / 重算暂存整块(每层都先写后读; gcomp/posg 跨层, 不在表里) */
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) pt_poison_rows(*all[i], 0, 1);
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, QH = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM, HD = DS4_N_HEAD_DIM;
    const struct { ds4_gpu_tensor *t; uint64_t f; } rows[] = {
        { st->hc, HC * E }, { st->hc2, HC * E }, { st->mix, 2u * HC + HC * HC }, { st->pre, HC }, { st->post, HC }, { st->comb, HC * HC },
        { st->pre_mix, HC }, { st->x, E }, { st->xn, E }, { st->attn_out, E }, { st->glog, DS4_N_EXPERT }, { st->rw, DS4_N_EXPERT_USED },
        { st->routed, E }, { st->sg, DS4_N_FF_EXP }, { st->su, DS4_N_FF_EXP }, { st->sh, DS4_N_FF_EXP }, { st->so, E }, { st->y, E },
        { st->qr, DS4_N_LORA_Q }, { st->qrn, DS4_N_LORA_Q }, { st->q, QH }, { st->kv, HD }, { st->kvn, HD }, { st->o, QH },
        { st->low, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O }, { r->ghc, HC * E }, { r->gpre, HC } };
    for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++) pt_poison_rows(rows[i].t, n, rows[i].f);
    pt_poison_rows(st->win[il], 0, 1);   /* 本层窗口整块: 重算只写块区 [SWA, SWA+n), 别处位置从 0 起的注意力不该读 */
}

bool pt_layers_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t n) {
    for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++)   /* 压缩行梯度按题累加: 每题从 0 起 */
        if (r->lb.gcomp[il] && !ds4_gpu_tensor_fill_f32(r->lb.gcomp[il], 0.f, (uint64_t)n * DS4_N_HEAD_DIM)) return false;
    int said = 0;
    for (uint32_t il = c->layer_hi + 1u; il-- > c->layer_lo; ) {
        if (c->poison) pt_poison_layer(r, il, n);
        /* 层内 bf16 权重缓存: 本层重算解出的 q4_K 稠密矩阵留给本层反传的转置乘(省掉第二遍解码); 只在这一层的"重算 + 反传"里开 */
        (void)ds4_gpu_bwd_wcache(1);
        const bool lok = pt_layer_bwd(e, c, r, il, n);
        (void)ds4_gpu_bwd_wcache(0);
        if (!lok) return false;
        if (c->dbg_layer == 99) pt_dbg(c, r, il, "层入口 g_hc", r->ghc, (uint64_t)DS4_N_HC * DS4_N_EMBD);   /* 逐层追尖峰 token(dbglayer=99) */
        if (c->poison && !said) {
            double s = 0, sp = 0;
            (void)ds4_gpu_bwd_sumsq_tensor(r->ghc, (uint64_t)n * DS4_N_HC * DS4_N_EMBD, &s);
            (void)ds4_gpu_bwd_sumsq_tensor(r->gpre, (uint64_t)n * DS4_N_HC, &sp);
            if (pt_is_nan(s) || pt_is_nan(sp)) { fprintf(stderr, "ds4: ★[poison] L%02u 反传之后入口梯度出 NaN(hc %s / pre %s): 本层读了没写过的内存★\n", il,
                                                         pt_is_nan(s) ? "NaN" : "好", pt_is_nan(sp) ? "NaN" : "好"); said = 1; }
            else if (il == c->layer_lo) fprintf(stderr, "ds4: [poison] %u 行一路到 L%02u 都没读到毒值\n", n, il);
        }
        /* hc 边界检查(pt_hccheck)要的: 第 il 层入口处的梯度(r->ghc 此刻就是它) */
        if (r->hcap && !r->hcap_mid && r->hcap_layer == (int32_t)il && !ds4_gpu_tensor_copy(r->hcap, 0, r->ghc, 0, (uint64_t)n * DS4_N_HC * DS4_N_EMBD * 4)) return false;
    }
    return true;   /* prof 的累计与打印在 core_ptrain.c(pt_prof_print): 整步各段一张表, 不再只打逐层反传那四段 */
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_layer_nonempty_tu;
