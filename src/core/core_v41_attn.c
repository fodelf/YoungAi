/* core_v41_attn.c — DeepSeek V4.1 注意力块, 增量形态(2026-09-12 战役 P2a 批前向 → P2c 带缓存), 逐式对照官方
 * Attention.forward / Compressor / Indexer / select_candidate_blocks / sparse_attn。
 *
 * 接线(g_ds4_v41): 只有 kv 源层压缩并持有 comp_kv/index_k; 消费层读最近源层; 只有 indexer 源层产 topk,
 * 消费层复用最近源层的 topk; candidate 源层筛候选块, 之后的 indexer 源层在块内 topk。
 * RoPE 常量按层: 压缩层(ratio>0)用 compress θ=160000 + YaRN(orig 65536), 窗口层 θ=10000 无 YaRN。
 * 增量语义(官方 start_pos>0 的等价写法): 可见性全按绝对位置; 压缩组只在凑满 ratio 个 token 时产出(尾巴留到下块)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

static float v41_theta(uint32_t ratio) { return ratio ? DS4_COMPRESS_ROPE_FREQ_BASE : DS4_ROPE_FREQ_BASE; }
static uint32_t v41_osl(uint32_t ratio) { return ratio ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u; }

/* rope(x[rows][n_head][hd] 末 n_rot 维) → 位置表 pos, 层常量 */
static bool v41_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t rows, uint32_t n_head, uint32_t hd, uint32_t ratio, bool inverse) {
    return ds4_gpu_v41_rope_tensor(x, pos, rows, n_head, hd, DS4_N_ROT, v41_theta(ratio), v41_osl(ratio), DS4_ROPE_SCALE_FACTOR,
                                   DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, inverse) != 0;
}

/* 压缩源层: 新 token 的压缩器输入接到余行后 → 池化出新完成的组 → latent(norm, 未 rope)
 * → index_k(wk+k_norm+rope+fp4) 与 comp_kv(rope+fp4 e4m3) 追加进源层缓存。 */
static bool v41_compress_source(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, IK = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t rowb = (uint64_t)HD * 4;
    const uint32_t g0 = st->pos0 / ratio;     /* 缓存里已有的组数 */
    uint32_t ng_new;
    if (ratio > 1) {   /* Compressor ratio>1: f32 权重, x 进, 组内逐维 softmax 池化 → bf16 */
        if (!ds4_gpu_v41_matmul_f32_tensor(st->ckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_matmul_f32_tensor(st->csc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, st->xn, n)) return false;
        const uint32_t pend = st->cpend[il];
        if (!ds4_gpu_tensor_copy(st->cpre_kv[il], (uint64_t)pend * rowb, st->ckv, 0, (uint64_t)n * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->cpre_sc[il], (uint64_t)pend * rowb, st->csc, 0, (uint64_t)n * rowb)) return false;
        const uint32_t tot = pend + n, rem = tot % ratio;
        ng_new = tot / ratio;
        if (ng_new) {
            if (!ds4_gpu_v41_compress_pool_tensor(st->pooled, st->cpre_kv[il], st->cpre_sc[il], tot, ratio, HD)) return false;
            if (rem) {   /* 余行挪到头(源行号 ≥ ratio > rem, 不重叠) */
                if (!ds4_gpu_tensor_copy(st->cpre_kv[il], 0, st->cpre_kv[il], (uint64_t)ng_new * ratio * rowb, (uint64_t)rem * rowb)) return false;
                if (!ds4_gpu_tensor_copy(st->cpre_sc[il], 0, st->cpre_sc[il], (uint64_t)ng_new * ratio * rowb, (uint64_t)rem * rowb)) return false;
            }
        }
        st->cpend[il] = rem;
    } else {           /* ratio 1: 纯投影(bf16 权重值) → bf16 */
        if (!ds4_gpu_v41_matmul_f32_tensor(st->pooled, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_round_bf16_tensor(st->pooled, (uint64_t)n * HD)) return false;
        ng_new = n;
    }
    st->ng_src[il] = g0 + ng_new;
    if (!ng_new) return true;
    {   /* 新组位置 g·ratio */
        int32_t *pg = xmalloc((size_t)ng_new * 4);
        for (uint32_t g = 0; g < ng_new; g++) pg[g] = (int32_t)((g0 + g) * ratio);
        const bool okp = ds4_gpu_tensor_write(st->posg, 0, pg, (uint64_t)ng_new * 4) != 0;
        free(pg);
        if (!okp) return false;
    }
    if (!ds4_gpu_v41_rms_norm_tensor(st->latent, st->pooled, m->map, m->size, l->attn_compressor_norm->abs_offset, HD, ng_new, DS4_RMS_EPS)) return false;
    /* indexer 键: k = k_norm(wk(latent)) → rope(组位置) → fp4(ue8m0/32) —— 用 latent 的未 rope 形; ckv 借作 [ng_new][IK] 出口 */
    if (!ds4_gpu_v41_matmul_f32_tensor(st->ktmp, m->map, m->size, l->indexer_wk->abs_offset, HD, IK, st->latent, ng_new)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ktmp, (uint64_t)ng_new * IK)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->ckv, st->ktmp, m->map, m->size, l->indexer_k_norm->abs_offset, IK, ng_new, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->ckv, st->posg, ng_new, 1, IK, ratio, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp4_tensor(st->ckv, ng_new, IK, 32, false)) return false;
    if (!ds4_gpu_tensor_copy(st->index_k[il], (uint64_t)g0 * IK * 4, st->ckv, 0, (uint64_t)ng_new * IK * 4)) return false;
    /* 压缩 KV: latent → rope(组位置) → fp4(e4m3 scale/16) → 源层缓存 */
    if (!ds4_gpu_tensor_copy(st->pooled, 0, st->latent, 0, (uint64_t)ng_new * rowb)) return false;
    if (!v41_rope(st->pooled, st->posg, ng_new, 1, HD, ratio, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp4_tensor(st->pooled, ng_new, HD, 16, true)) return false;
    return ds4_gpu_tensor_copy(st->comp_kv[il], (uint64_t)g0 * rowb, st->pooled, 0, (uint64_t)ng_new * rowb) != 0;
}

/* indexer 源层: q = wq_b(qr_norm) → rope → fp4; weights = proj(x)·scale; 对源层整段键打分 → [候选块] → topk */
static bool v41_index_source(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t n = st->n, E = DS4_N_EMBD, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    const int16_t src = v->kv_source_of[il];
    if (src < 0 || !st->index_k[src]) return false;
    const uint32_t ng = st->ng_src[src];
    st->idx_owner = (int16_t)il;
    if (!ng) { st->idx_topk = 0; return true; }   /* 还没有任何完成的组: 本层只看窗口 */
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->iq, m->map, m->size, l->indexer_attn_q_b->abs_offset, DS4_N_LORA_Q, (uint64_t)IH * IK, st->qrn, n, 1)) return false;
    if (!v41_rope(st->iq, st->pos, n, IH, IK, ratio, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp4_tensor(st->iq, (uint64_t)n * IH, IK, 32, false)) return false;
    if (!ds4_gpu_v41_matmul_f32_tensor(st->iw, m->map, m->size, l->indexer_proj->abs_offset, E, IH, st->xn, n)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->iw, (uint64_t)n * IH)) return false;
    /* weights = proj(x) * (softmax_scale · n_heads^-0.5), 官方在 bf16 上乘 → 再舍 bf16 */
    if (!ds4_gpu_v41_scale_round_tensor(st->iw, (uint64_t)n * IH, (float)(1.0 / sqrt((double)IK) / sqrt((double)IH)))) return false;
    const bool uses_cand = v->candidate_source_layer >= 0 && (int32_t)il > v->candidate_source_layer && st->cand_owner >= 0;
    if (!ds4_gpu_v41_indexer_score_tensor(st->iscore, st->iq, st->index_k[src], st->iw, uses_cand ? st->cand : NULL, n, st->pos0, ng, IH, IK, ratio)) return false;
    if ((int32_t)il == v->candidate_source_layer) {
        if (!ds4_gpu_v41_candidate_blocks_tensor(st->cand, st->iscore, n, st->pos0, ng, ratio, (uint32_t)v->candidate_topk_blocks, (uint32_t)v->candidate_block_size)) return false;
        st->cand_owner = (int16_t)il;
    }
    const uint32_t topk = DS4_N_INDEXER_TOP_K < ng ? DS4_N_INDEXER_TOP_K : ng;   /* min(index_topk, end_pos // ratio) */
    if (!ds4_gpu_v41_indexer_topk_tensor(st->idx, st->iscore, n, ng, topk)) return false;
    st->idx_topk = topk;
    return true;
}

bool v41_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint64_t rowb = (uint64_t)HD * 4;
    /* q 路: q_a → bf16 → q_norm → q_b → bf16 → rope(绝对位置) */
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->qr, m->map, m->size, l->attn_q_a->abs_offset, E, Q, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS)) return false;
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->q, m->map, m->size, l->attn_q_b->abs_offset, Q, (uint64_t)NH * HD, st->qrn, n, 1)) return false;
    if (!v41_rope(st->q, st->pos, n, NH, HD, ratio, false)) return false;
    /* 窗口 kv: wkv → bf16 → kv_norm → rope(末 64 维) → fp8 act_quant(按 32 块) → 进窗口缓冲的后 n 行 */
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->kv, m->map, m->size, l->attn_kv->abs_offset, E, HD, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->kvn, st->pos, n, 1, HD, ratio, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) return false;
    if (!ds4_gpu_tensor_copy(st->win[il], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)n * rowb)) return false;
    /* 压缩侧: 源层先产出, 消费层读最近源层的缓存 + 本 chunk 最近 indexer 源层的 topk */
    uint32_t ng = 0, topk = 0; const ds4_gpu_tensor *comp = NULL;
    if (ratio) {
        if (v->is_kv_source[il] && !v41_compress_source(e, st, il, ratio)) return false;
        if (v->is_index_source[il] && !v41_index_source(e, st, il, ratio)) return false;
        const int16_t src = v->kv_source_of[il];
        if (src < 0 || !st->comp_kv[src]) { fprintf(stderr, "ds4: V4.1 L%u 压缩层无源\n", il); return false; }
        ng = st->ng_src[src];
        if (ng) {
            if (st->idx_owner < 0) { fprintf(stderr, "ds4: V4.1 L%u 压缩层无 topk\n", il); return false; }
            comp = st->comp_kv[src]; topk = st->idx_topk;
        }
    }
    /* 稀疏注意力(窗口 128 + topk 压缩行, sink 进分母) → bf16 → 逆 rope */
    if (!ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, st->win[il], (ng && topk) ? comp : NULL, (ng && topk) ? st->idx : NULL, m->map, m->size,
                                        l->attn_sinks->abs_offset, n, st->pos0, SWA, ng, topk, NH, HD, (float)(1.0 / sqrt((double)HD)))) return false;
    /* 窗口缓冲平移: 行 [n, n+SWA) → [0, SWA)(经暂存, 免重叠) */
    if (!ds4_gpu_tensor_copy(st->wintmp, 0, st->win[il], (uint64_t)n * rowb, (uint64_t)SWA * rowb)) return false;
    if (!ds4_gpu_tensor_copy(st->win[il], 0, st->wintmp, 0, (uint64_t)SWA * rowb)) return false;
    if (!v41_rope(st->o, st->pos, n, NH, HD, ratio, true)) return false;
    /* 输出投影: 分组 wo_a(块对角) → bf16 → wo_b → bf16 */
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!ds4_gpu_v41_grouped_matmul_fp4x32_tensor(st->low, m->map, m->size, l->attn_output_a->abs_offset, DS4_N_OUT_GROUP,
                                                  (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1)) return false;
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->attn_out, m->map, m->size, l->attn_output_b->abs_offset,
                                          (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1)) return false;
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_attn_nonempty_tu;
