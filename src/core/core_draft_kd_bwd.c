/* core_draft_kd_bwd.c — 草稿器蒸馏: 出口反传 + 逐塔重算反传 + 件落盘(2026-10-07)。总述见 core_draft_kd.h。
 *
 * 出口(闭式一段): g_xn(头的转置乘, core_draft_kd_fwd.c 已填) → 出口件(xn_post = bf16(xn_pre + xn_pre·BᵀA): g_xn_pre = g_xn_post + 件的反向)
 *   → out_norm 反向 → hc_pre 反向 ⇒ 末塔出口处的 g_hc / g_pre。
 * 逐塔(梯度检查点, 与 core_ptrain_layer.c 同一套路): 从末塔到塔 0, 先用 dk_tower_fwd 同一串函数把本塔重算一遍(中间量抄进 lb), 再按
 *   ffn 半层 → 注意力半层倒推: hc_post → 塔件 / shared / routed / 路由 → ffn_norm → hc_pre → mHC(ffn)
 *   → hc_post → wo_b/wo_a → RoPE → 块注意力(只出 q 与块内 kv 的梯度) → q 路 / kv 路 → attn_norm → hc_pre → mHC(attn)。
 * 不求导的: 路由选哪几个专家(离散)、窗口里的历史行(main_x 冻结)、各处 bf16/fp8 舍入(直通)、markov 偏置(不依赖隐态)。
 * 塔件梯度攒进 r->gA/gB[T], 出口件进 r->gxA/gxB; Adam 在 core_draft_kd.c。 */
#include "core_draft_kd.h"
#include "../common/ds4_gr_fnv.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

#define DK_LB_LIST(b) &b->mixa, &b->posta, &b->comba, &b->prea, &b->xa, &b->xna, &b->qr, &b->kv, &b->o, &b->hcin, \
    &b->mixf, &b->postf, &b->combf, &b->pref, &b->xf, &b->xnf, &b->g_hcm, &b->g_hcin, &b->g_y, &b->g_xnf, &b->g_xf, &b->g_pre_a, \
    &b->g_postf, &b->g_combf, &b->g_attn, &b->g_posta, &b->g_comba, &b->g_low, &b->g_o, &b->g_q, &b->g_kvn, &b->g_qrn, &b->g_qr, \
    &b->g_kv, &b->g_xna, &b->g_xa, &b->g_prein, &b->g_sh, &b->g_sg, &b->g_su, &b->g_rw, &b->g_glog, &b->g_blk

bool dk_layer_alloc(dk_run *r) {
    dk_layer_buf *b = &r->lb;
    const uint64_t n = r->R, E = DS4_N_EMBD, HC = DS4_N_HC, MIX = 2u * HC + HC * HC, HD = DS4_N_HEAD_DIM, QH = (uint64_t)DS4_N_HEAD * HD;
    const uint64_t Q = DS4_N_LORA_Q, LOW = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, FF = DS4_N_FF_EXP, NE = g_ds4_v41.mtp_experts, KU = g_ds4_v41.mtp_used;
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
        { &b->g_rw, n * KU * 4 }, { &b->g_glog, n * NE * 4 }, { &b->g_blk, n * HD * 4 },
    };
    for (size_t i = 0; i < sizeof plan / sizeof plan[0] && ok; i++) *plan[i].t = v41_alloc(plan[i].bytes, &ok);
    if (!ok) fprintf(stderr, "ds4: [dk] 逐塔反传缓冲分配失败(R %u)\n", r->R);
    return ok;
}
void dk_layer_free(dk_run *r) {
    dk_layer_buf *b = &r->lb;
    ds4_gpu_tensor **all[] = { DK_LB_LIST(b) };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) { ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
}

#define DK_CP(dst, src, bytes) ds4_gpu_tensor_copy((dst), 0, (src), 0, (bytes))
#define DK_MT(gx, w, in, out, gy, acc) ds4_gpu_bwd_matmul_t_tensor((gx), m->map, m->size, (w)->type, (w)->abs_offset, (in), (out), (gy), n, (acc))

/* 出口反传: r->g_xn(对出口件之后、舍 bf16 之后的 xn 的梯度)→ r->ghc / r->gpre(末塔出口处) */
bool dk_exit_bwd(ds4_engine *e, dk_run *r) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model;
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC;
    bool ok = DK_CP(r->g_xn_pre, r->g_xn, (uint64_t)n * E * 4);   /* 恒等支 */
    if (ok && r->Kx) ok = ds4_gpu_bwd_amp_tensor(r->gxA, r->gxB, r->g_xn_pre, r->g_xn, r->xn_pre, r->xA, r->xB, r->xT, r->gxT, n, E, r->Kx) != 0;
    ok = ok && ds4_gpu_bwd_rms_norm_tensor(r->g_h, r->g_xn_pre, r->h, m->map, m->size, e->weights.mtp.out_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
         ds4_gpu_tensor_fill_f32(r->ghc, 0.f, (uint64_t)n * HC * E) &&
         ds4_gpu_bwd_hc_pre_tensor(r->ghc, r->gpre, r->g_h, st->hc, st->pre_mix, E, HC, n);
    if (!ok) fprintf(stderr, "ds4: [dk] 出口反传失败\n");
    return ok;
}

/* 重算第 T 塔(从存档的入口起, 与训练前向同一串函数), 中间量抄进 lb; 结束时 st 里留着本塔 MoE 的 y/sel/rw/glog/sg/su 与 attn_out, st->hc = hc_mid */
static bool dk_tower_recompute(ds4_engine *e, dk_run *r, uint32_t T) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.mtp.tower[T];
    dk_layer_buf *b = &r->lb;
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint64_t MIX = 2u * HC + HC * HC, QH = (uint64_t)NH * HD;
    bool ok = DK_CP(b->hcin, r->hc_in[T], (uint64_t)n * HC * E * 4) && DK_CP(st->hc, b->hcin, (uint64_t)n * HC * E * 4) &&
              DK_CP(st->pre_mix, r->pm_in[T], (uint64_t)n * HC * 4) && v41_hc_half(e, st, l, true) &&
              DK_CP(b->mixa, st->mix, n * MIX * 4) && DK_CP(b->posta, st->post, (uint64_t)n * HC * 4) && DK_CP(b->comba, st->comb, (uint64_t)n * HC * HC * 4) &&
              DK_CP(b->prea, st->pre_mix, (uint64_t)n * HC * 4) && DK_CP(b->xa, st->x, (uint64_t)n * E * 4) && DK_CP(b->xna, st->xn, (uint64_t)n * E * 4);
    /* 注意力(与 dk_tower_fwd 同一串核): q 路 / kv 路 / 块注意力 / 逆 RoPE / 输出投影 */
    ok = ok && v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1) && DK_CP(b->qr, st->qr, (uint64_t)n * Q * 4) &&
         ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS) &&
         v41_tproj(m, st->q, l->attn_q_b, Q, QH, st->qrn, n, 1) &&
         ds4_gpu_v41_rope_tensor(st->q, st->pos, n, NH, HD, DS4_N_ROT, DS4_ROPE_FREQ_BASE, 0u, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, false) &&
         v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1) && DK_CP(b->kv, st->kv, (uint64_t)n * HD * 4) &&
         ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS) &&
         ds4_gpu_v41_rope_tensor(st->kvn, st->pos, n, 1, HD, DS4_N_ROT, DS4_ROPE_FREQ_BASE, 0u, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, false) &&
         ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32) &&
         ds4_gpu_draft_attn_fwd_tensor(st->o, r->lse[T], st->q, r->hist[T], r->hbase, st->kvn, r->bpos, r->nb, r->B, SWA,
                                       m->map, m->size, l->attn_sinks->abs_offset, NH, HD, (float)(1.0 / sqrt((double)HD))) &&
         DK_CP(b->o, st->o, n * QH * 4) &&
         ds4_gpu_v41_rope_tensor(st->o, st->pos, n, NH, HD, DS4_N_ROT, DS4_ROPE_FREQ_BASE, 0u, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
         v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP, (uint64_t)(NH / DS4_N_OUT_GROUP) * HD, DS4_N_LORA_O, st->o, n, 1) &&
         v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1) &&
         ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, E, HC, n);
    if (ok) { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }   /* st->hc = hc_mid(ffn 半层入口) */
    /* ffn 半层: 专家前向开截留(逐对中间量留给紧接着的 routed 反向), 只这一发 */
    if (ok) ok = v41_hc_half(e, st, l, false) &&
                 DK_CP(b->mixf, st->mix, n * MIX * 4) && DK_CP(b->postf, st->post, (uint64_t)n * HC * 4) && DK_CP(b->combf, st->comb, (uint64_t)n * HC * HC * 4) &&
                 DK_CP(b->pref, st->pre_mix, (uint64_t)n * HC * 4) && DK_CP(b->xf, st->x, (uint64_t)n * E * 4) && DK_CP(b->xnf, st->xn, (uint64_t)n * E * 4) &&
                 ds4_gpu_bwd_moe_capture(1) && v41_moe(m, l, st, T);
    (void)ds4_gpu_bwd_moe_capture(0);
    return ok;
}

static bool dk_tower_bwd(ds4_engine *e, dk_run *r, uint32_t T) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.mtp.tower[T];
    dk_layer_buf *b = &r->lb;
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, FF = DS4_N_FF_EXP, SWA = DS4_N_SWA;
    const uint32_t G = DS4_N_OUT_GROUP, LOW = DS4_N_OUT_GROUP * DS4_N_LORA_O, NE = g_ds4_v41.mtp_experts, KU = g_ds4_v41.mtp_used;
    const float theta = DS4_ROPE_FREQ_BASE, scale = (float)(1.0 / sqrt((double)HD));
    bool ok = dk_tower_recompute(e, r, T);
    /* ---- ffn 半层 ---- */
    ok = ok && ds4_gpu_tensor_fill_f32(b->g_hcm, 0.f, (uint64_t)n * HC * E) &&
         ds4_gpu_bwd_hc_post_tensor(b->g_y, b->g_hcm, b->g_postf, b->g_combf, r->ghc, st->y, st->hc, b->postf, b->combf, E, HC, n) &&
         ds4_gpu_tensor_fill_f32(b->g_xnf, 0.f, (uint64_t)n * E);
    if (ok && r->gA[T]) ok = ds4_gpu_bwd_amp_tensor(r->gA[T], r->gB[T], b->g_xnf, b->g_y, b->xnf, st->ampA[T], st->ampB[T], st->ampT, r->gT, n, E, r->K) != 0;
    ok = ok && DK_MT(b->g_sh, l->ffn_down_shexp, FF, E, b->g_y, 0) &&
         ds4_gpu_bwd_swiglu_tensor(b->g_sg, b->g_su, b->g_sh, st->sg, st->su, (uint64_t)n * FF, DS4_SWIGLU_CLAMP_EXP) &&
         DK_MT(b->g_xnf, l->ffn_gate_shexp, E, FF, b->g_sg, 1) && DK_MT(b->g_xnf, l->ffn_up_shexp, E, FF, b->g_su, 1);
    if (ok) {
        const ds4_tensor *blob = st->tower_exps_vq[T];
        ok = blob && ds4_gpu_bwd_routed_moe_tensor(b->g_xnf, b->g_rw, b->g_y, b->xnf, st->sel, st->rw, m->map, m->size, blob->abs_offset, blob->bytes,
                                                   E, FF, E, NE, KU, DS4_SWIGLU_CLAMP_EXP, DS4_N_LAYER + T, n) &&
             ds4_gpu_bwd_router_tensor(b->g_glog, b->g_rw, st->sel, st->glog, n, NE, KU, DS4_EXPERT_WEIGHT_SCALE) &&
             DK_MT(b->g_xnf, l->ffn_gate_inp, E, NE, b->g_glog, 1);
    }
    ok = ok && ds4_gpu_bwd_rms_norm_tensor(b->g_xf, b->g_xnf, b->xf, m->map, m->size, l->ffn_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
         ds4_gpu_bwd_hc_pre_tensor(b->g_hcm, b->g_pre_a, b->g_xf, st->hc, b->prea, E, HC, n) &&
         ds4_gpu_bwd_hc_mix_tensor(b->g_hcm, r->gpre, b->g_postf, b->g_combf, st->hc, b->mixf, m->map, m->size, l->hc_ffn_fn->abs_offset,
                                   l->hc_ffn_scale->abs_offset, l->hc_ffn_base->abs_offset, E, HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, n);
    /* ---- 注意力半层 ---- */
    ok = ok && ds4_gpu_tensor_fill_f32(b->g_hcin, 0.f, (uint64_t)n * HC * E) &&
         ds4_gpu_bwd_hc_post_tensor(b->g_attn, b->g_hcin, b->g_posta, b->g_comba, b->g_hcm, st->attn_out, b->hcin, b->posta, b->comba, E, HC, n) &&
         DK_MT(b->g_low, l->attn_output_b, LOW, E, b->g_attn, 0) &&
         ds4_gpu_bwd_grouped_matmul_t_tensor(b->g_o, m->map, m->size, l->attn_output_a->type, l->attn_output_a->abs_offset, G, (uint64_t)(NH / G) * HD, DS4_N_LORA_O, b->g_low, n, 0) &&
         ds4_gpu_bwd_rope_tensor(b->g_o, st->pos, n, NH, HD, DS4_N_ROT, theta, 0u, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, false) &&
         ds4_gpu_draft_attn_bwd_tensor(b->g_q, b->g_blk, b->g_o, b->o, st->q, r->lse[T], r->hist[T], r->hbase, st->kvn, r->bpos, r->nb, r->B, SWA, NH, HD, scale) &&
         ds4_gpu_bwd_rope_tensor(b->g_q, st->pos, n, NH, HD, DS4_N_ROT, theta, 0u, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
         DK_MT(b->g_qrn, l->attn_q_b, Q, (uint64_t)NH * HD, b->g_q, 0) &&
         ds4_gpu_bwd_rms_norm_tensor(b->g_qr, b->g_qrn, b->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS, 0) &&
         DK_MT(b->g_xna, l->attn_q_a, E, Q, b->g_qr, 0) &&
         DK_CP(b->g_kvn, b->g_blk, (uint64_t)n * HD * 4) &&
         ds4_gpu_bwd_rope_tensor(b->g_kvn, st->pos, n, 1, HD, DS4_N_ROT, theta, 0u, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
         ds4_gpu_bwd_rms_norm_tensor(b->g_kv, b->g_kvn, b->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS, 0) &&
         DK_MT(b->g_xna, l->attn_kv, E, HD, b->g_kv, 1) &&
         ds4_gpu_bwd_rms_norm_tensor(b->g_xa, b->g_xna, b->xa, m->map, m->size, l->attn_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
         ds4_gpu_bwd_hc_pre_tensor(b->g_hcin, b->g_prein, b->g_xa, b->hcin, r->pm_in[T], E, HC, n) &&
         ds4_gpu_bwd_hc_mix_tensor(b->g_hcin, b->g_pre_a, b->g_posta, b->g_comba, b->hcin, b->mixa, m->map, m->size, l->hc_attn_fn->abs_offset,
                                   l->hc_attn_scale->abs_offset, l->hc_attn_base->abs_offset, E, HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, n) &&
         DK_CP(r->ghc, b->g_hcin, (uint64_t)n * HC * E * 4) && DK_CP(r->gpre, b->g_prein, (uint64_t)n * HC * 4);
    if (!ok) fprintf(stderr, "ds4: [dk] 塔 %u 反传失败(n %u)\n", T, n);
    return ok;
}

bool dk_towers_bwd(ds4_engine *e, dk_run *r, uint32_t nb) {
    (void)nb;
    for (uint32_t T = r->NT; T-- > 0; ) {
        (void)ds4_gpu_bwd_wcache(1);   /* 本塔重算解出的稠密矩阵留给本塔反传的转置乘 */
        const bool ok = dk_tower_bwd(e, r, T);
        (void)ds4_gpu_bwd_wcache(0);
        if (!ok) return false;
    }
    return true;
}

/* 件落盘: tower_Tn.bin = {E,K,1} + A + B(amp_Lnn.bin 同格式); exit.dspa = {"DSPA",D,K,0} + A + B(现成 --draft-amp 的格式, 引擎直接挂);
 * base.fnv = ② 目录指纹(件是对着 ①+② 这个底座训的, 挂到别的底座上接受率只会掉, 不报错) */
bool dk_save(const dk_cfg *c, dk_run *r, const char *dir) {
    mkdir(dir, 0755);
    const uint32_t E = DS4_N_EMBD;
    const uint32_t Kmax = r->K > r->Kx ? r->K : r->Kx;
    float *buf = xmalloc((size_t)Kmax * E * 4 + 16);
    bool ok = true;
    for (uint32_t T = 0; ok && r->K && T < r->NT; T++) {
        char p[1200]; snprintf(p, sizeof p, "%s/tower_T%u.bin", dir, T);
        FILE *f = fopen(p, "wb");
        const uint64_t nel = (uint64_t)r->K * E; const int32_t hd[3] = { (int32_t)E, (int32_t)r->K, 1 };
        ok = f && fwrite(hd, 4, 3, f) == 3 && ds4_gpu_tensor_read(r->st.ampA[T], 0, buf, nel * 4) && fwrite(buf, 4, nel, f) == nel &&
             ds4_gpu_tensor_read(r->st.ampB[T], 0, buf, nel * 4) && fwrite(buf, 4, nel, f) == nel;
        if (f) fclose(f);
    }
    if (ok && r->Kx) {
        char p[1200]; snprintf(p, sizeof p, "%s/exit.dspa", dir);
        FILE *f = fopen(p, "wb");
        const uint64_t nel = (uint64_t)r->Kx * E; const struct { char magic[4]; uint32_t d, k, rsv; } hd = { { 'D', 'S', 'P', 'A' }, E, r->Kx, 0u };
        ok = f && fwrite(&hd, sizeof hd, 1, f) == 1 && ds4_gpu_tensor_read(r->xA, 0, buf, nel * 4) && fwrite(buf, 4, nel, f) == nel &&
             ds4_gpu_tensor_read(r->xB, 0, buf, nel * 4) && fwrite(buf, 4, nel, f) == nel;
        if (f) fclose(f);
    }
    free(buf);
    if (ok && r->mkE) {   /* 偏置表: markov_embd.bin {Vm,R,1} + f32 / markov_head.bin {V,R,1} + f32(与原件同形, 引擎挂件时顶替 GGUF 里那两张) */
        const struct { const char *nm; ds4_gpu_tensor *t; uint32_t rows; } tb[2] = { { "markov_embd.bin", r->mkE, r->Vm }, { "markov_head.bin", r->mkH, DS4_N_VOCAB } };
        for (int i = 0; ok && i < 2; i++) {
            char p[1200]; snprintf(p, sizeof p, "%s/%s", dir, tb[i].nm);
            const uint64_t nel = (uint64_t)tb[i].rows * r->Rk; float *tbuf = xmalloc((size_t)nel * 4);
            FILE *f = fopen(p, "wb"); const int32_t hd[3] = { (int32_t)tb[i].rows, (int32_t)r->Rk, 1 };
            ok = f && fwrite(hd, 4, 3, f) == 3 && ds4_gpu_tensor_read(tb[i].t, 0, tbuf, nel * 4) && fwrite(tbuf, 4, nel, f) == nel;
            if (f) fclose(f);
            free(tbuf);
        }
    }
    if (ok) {
        unsigned n = 0;
        const uint64_t h = (g_ds4_v41_amp_dir && g_ds4_v41_amp_dir[0]) ? ds4_gr_dir_fnv(g_ds4_v41_amp_dir, DS4_N_LAYER, &n) : DS4_GR_FNV_SEED;
        char p[1200]; snprintf(p, sizeof p, "%s/base.fnv", dir);
        FILE *f = fopen(p, "w");
        ok = f && fprintf(f, "%016llx %u\n", (unsigned long long)h, n) > 0;
        if (f) fclose(f);
        snprintf(p, sizeof p, "%s/drafter.cfg", dir);
        f = fopen(p, "w");
        if (f) { fprintf(f, "rank=%u\nrank_exit=%u\nmarkov=%u\ntopk=%u\nlr=%g\nlr_markov=%g\nseed=%u\n", r->K, r->Kx, r->mkE ? 1u : 0u, c->topk, (double)c->lr, (double)c->lr_markov, c->seed); fclose(f); }
    }
    if (!ok) fprintf(stderr, "ds4: [dk] 件落盘失败 %s\n", dir);
    return ok;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_draft_kd_bwd_nonempty_tu;
