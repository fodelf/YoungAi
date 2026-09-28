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
/* ★解码整步 graph 的压缩源层(2026-09-18; 口径见 ds4_gpu_v41.h "设备位置")★ n=1, 位置只在设备槽 st->pos 里:
 * 追加/池化一发(核里按 pos%ratio 定格、凑满才池化), 之后的 norm→wk→rope→打包链**每步无条件发**, 组号由打包核按
 * 位置算 —— 没凑满时算的是陈值、写进垃圾槽(缓存末尾那一格, 组号上限之外, 永远没人读)。
 * 为什么不在主机上判"凑满没有": 图捕一次要重放几百步, 主机的每一个 if 都得变成"每步都发、设备上判"。
 * 主机计数(cpend/ng_src)这里不动 —— graph 路由 core_decode_graph.c 每步按闭式推进。 */
/* ★n 行(投机验证批进图, 2026-09-22)★: 追加核逐行按批内顺序追加、每凑满一组池化一次, 池化行最多 np = (ratio−1+n)/ratio 个;
 * 之后的 norm→wk→rope→打包链按 np 行**无条件发**, 打包核按位置算真组号, 没凑满的行写垃圾槽。n=1 就是原来的整步图。
 * 快照(投机回滚要)也在追加核里: [旧余行 | 本批 n 行] 线性存进 snap_cpre —— 与直发路的布局同, v41_spec_rollback 一个字不改。 */
static bool v41_compress_source_graph(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, IK = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t rowb = (uint64_t)HD * 4;
    const uint32_t g_trash = st->ctx / ratio;   /* 与 v41_state_alloc 的 ngcap = ctx/ratio + 2 同源: 这一格在组号上限之外 */
    const ds4_gpu_tensor *pg = st->posg;        /* 新组位置表: ratio>1 由追加核写; ratio 1 每行一组, 位置就是 pos 自己 */
    uint32_t np;
    if (ratio > 1) {
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->ckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->csc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, st->xn, n)) return false;
        /* 快照只有验证批(n>1)要; 缓冲由直发那一轮 v41_compress_source 建(捕获态不许分配), 没建就是调用方没先直发暖过一轮 */
        ds4_gpu_tensor *sk = n > 1u ? st->snap_cpre_kv[il] : NULL, *ss = n > 1u ? st->snap_cpre_sc[il] : NULL;
        if (n > 1u && (!sk || !ss)) { fprintf(stderr, "ds4: [graph] L%u 压缩器余行的快照缓冲还没建(验证批要先直发跑一轮)\n", il); return false; }
        if (!ds4_gpu_v41_compress_step_n_tensor(st->pooled, st->posg, st->cpre_kv[il], st->cpre_sc[il], sk, ss, st->ckv, st->csc, st->pos, ratio, HD, n)) return false;
        np = (ratio - 1u + n) / ratio;
    } else {
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->pooled, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_round_bf16_tensor(st->pooled, (uint64_t)n * HD)) return false;
        pg = st->pos; np = n;
    }
    if (!ds4_gpu_v41_rms_norm_tensor(st->latent, st->pooled, m->map, m->size, l->attn_compressor_norm->abs_offset, HD, np, DS4_RMS_EPS)) return false;
    if (!ds4_gpu_v41_matmul_bf16_tensor(st->ktmp, m->map, m->size, l->indexer_wk->abs_offset, HD, IK, st->latent, np)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ktmp, (uint64_t)np * IK)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->ckv, st->ktmp, m->map, m->size, l->indexer_k_norm->abs_offset, IK, np, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->ckv, pg, np, 1, IK, ratio, false)) return false;
    if (!ds4_gpu_v41_idxk_pack_tensor(st->index_k[il], 0, st->ckv, np, st->pos, ratio, g_trash, n)) return false;
    if (!ds4_gpu_tensor_copy(st->pooled, 0, st->latent, 0, (uint64_t)np * rowb)) return false;
    if (!v41_rope(st->pooled, pg, np, 1, HD, ratio, false)) return false;
    return ds4_gpu_v41_ckv_pack_tensor(st->comp_kv[il], 0, st->pooled, np, st->pos, ratio, g_trash, n) != 0;
}

static bool v41_compress_source(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    if (st->graph) return v41_compress_source_graph(e, st, il, ratio);
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, IK = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t rowb = (uint64_t)HD * 4;
    const uint32_t g0 = st->pos0 / ratio;     /* 缓存里已有的组数 */
    uint32_t ng_new;
    if (ratio > 1) {   /* Compressor ratio>1: f32 权重, x 进, 组内逐维 softmax 池化 → bf16 */
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->ckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->csc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, st->xn, n)) return false;
        const uint32_t pend = st->cpend[il];
        if (!ds4_gpu_tensor_copy(st->cpre_kv[il], (uint64_t)pend * rowb, st->ckv, 0, (uint64_t)n * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->cpre_sc[il], (uint64_t)pend * rowb, st->csc, 0, (uint64_t)n * rowb)) return false;
        const uint32_t tot = pend + n, rem = tot % ratio;
        ng_new = tot / ratio;
        /* ★投机验证的回滚点★(speed.md 段 6 D1): 下面的"余行挪到头"是破坏性的, 挪完就找不回
         * 本批那几个 token 的压缩器输入了。所以在挪之前把整块存一份, 回滚时按接受数从里面取。 */
        if (st->snap_on) {
            bool sok = true;
            const uint64_t nb = ((uint64_t)ratio + st->cap_tok) * rowb;   /* 与 v41_state_alloc 里 cpre_* 的分配式同源 */
            if (!st->snap_cpre_kv[il]) { st->snap_cpre_kv[il] = v41_alloc(nb, &sok); st->snap_cpre_sc[il] = v41_alloc(nb, &sok); }
            if (!sok) return false;
            if (!ds4_gpu_tensor_copy(st->snap_cpre_kv[il], 0, st->cpre_kv[il], 0, (uint64_t)tot * rowb)) return false;
            if (!ds4_gpu_tensor_copy(st->snap_cpre_sc[il], 0, st->cpre_sc[il], 0, (uint64_t)tot * rowb)) return false;
            st->snap_cpend[il] = pend;
        }
        if (ng_new) {
            if (!ds4_gpu_v41_compress_pool_tensor(st->pooled, st->cpre_kv[il], st->cpre_sc[il], tot, ratio, HD)) return false;
            if (rem) {   /* 余行挪到头(源行号 ≥ ratio > rem, 不重叠) */
                if (!ds4_gpu_tensor_copy(st->cpre_kv[il], 0, st->cpre_kv[il], (uint64_t)ng_new * ratio * rowb, (uint64_t)rem * rowb)) return false;
                if (!ds4_gpu_tensor_copy(st->cpre_sc[il], 0, st->cpre_sc[il], (uint64_t)ng_new * ratio * rowb, (uint64_t)rem * rowb)) return false;
            }
        }
        st->cpend[il] = rem;
    } else {           /* ratio 1: 纯投影(bf16 权重值) → bf16 */
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->pooled, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
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
    if (!ds4_gpu_v41_matmul_bf16_tensor(st->ktmp, m->map, m->size, l->indexer_wk->abs_offset, HD, IK, st->latent, ng_new)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ktmp, (uint64_t)ng_new * IK)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->ckv, st->ktmp, m->map, m->size, l->indexer_k_norm->abs_offset, IK, ng_new, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->ckv, st->posg, ng_new, 1, IK, ratio, false)) return false;
    /* ★量化 + 打包一发进缓存★(decode.md D1): 原来是"act_quant 就地改 f32" + "整行 f32 拷进缓存"两发,
     * 缓存按 512 个 f32 存一个只有 16 种取值的量 —— 现在按官方的 288 B/组存(见 cuda_kv_pack.inc.cu)。
     * 值逐位不变: 打包的 (nibble, scale) 读回来算 bf16(nibble×scale), 正是 act_quant 写回去的那个数。 */
    if (!ds4_gpu_v41_idxk_pack_tensor(st->index_k[il], g0, st->ckv, ng_new, NULL, 0, 0, 0)) return false;
    /* 压缩 KV: latent → rope(组位置) → fp4(e4m3 scale/16) → 源层缓存 */
    if (!ds4_gpu_tensor_copy(st->pooled, 0, st->latent, 0, (uint64_t)ng_new * rowb)) return false;
    if (!v41_rope(st->pooled, st->posg, ng_new, 1, HD, ratio, false)) return false;
    return ds4_gpu_v41_ckv_pack_tensor(st->comp_kv[il], g0, st->pooled, ng_new, NULL, 0, 0, 0) != 0;
}

/* indexer 源层: q = wq_b(qr_norm) → rope → fp4; weights = proj(x)·scale; 对源层整段键打分 → [候选块] → topk */
static bool v41_index_source(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t n = st->n, E = DS4_N_EMBD, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    const int16_t src = v->kv_source_of[il];
    if (src < 0 || !st->index_k[src]) return false;
    /* graph 路: ng/topk 传桶上限, 位置走设备槽(st->pos); 直发路: 主机真值。"还没有完成的组"那条早退在 graph 路
     * 由核自己处理(ng 自算为 0 ⇒ 打分/topk 都空跑, 注意力只看窗口), 主机不分支。 */
    const ds4_gpu_tensor *posd = st->graph ? st->pos : NULL;
    const uint32_t ng = st->graph ? (st->graph_pos_cap + n) / ratio : st->ng_src[src];   /* 桶上限: 末行位置 cap+n−1 之后的组数 */
    st->idx_owner = (int16_t)il;
    if (!ng) { st->idx_topk = 0; st->idx_ratio = 0; return true; }   /* 还没有任何完成的组: 本层只看窗口 */
    /* 打分草稿按本层要用的组数长(走图那条已在 capture 前按桶上限长够, 这里恒真; 见 core_v41_forward.c) */
    if (!v41_index_scratch_prepare(st, ng)) return false;
    if (!v41_tproj(m, st->iq, l->indexer_attn_q_b, DS4_N_LORA_Q, (uint64_t)IH * IK, st->qrn, n, 1)) return false;
    if (!v41_rope(st->iq, st->pos, n, IH, IK, ratio, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp4_tensor(st->iq, (uint64_t)n * IH, IK, 32, false)) return false;
    if (!ds4_gpu_v41_matmul_bf16_tensor(st->iw, m->map, m->size, l->indexer_proj->abs_offset, E, IH, st->xn, n)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->iw, (uint64_t)n * IH)) return false;
    /* weights = proj(x) * (softmax_scale · n_heads^-0.5), 官方在 bf16 上乘 → 再舍 bf16 */
    if (!ds4_gpu_v41_scale_round_tensor(st->iw, (uint64_t)n * IH, (float)(1.0 / sqrt((double)IK) / sqrt((double)IH)))) return false;
    const bool uses_cand = v->candidate_source_layer >= 0 && (int32_t)il > v->candidate_source_layer && st->cand_owner >= 0;
    if (!ds4_gpu_v41_indexer_score_tensor(st->iscore, st->iq, st->index_k[src], st->iw, uses_cand ? st->cand : NULL, n, st->pos0, ng, IH, IK, ratio, posd)) return false;
    if ((int32_t)il == v->candidate_source_layer) {
        if (!ds4_gpu_v41_candidate_blocks_tensor(st->cand, st->iscore, n, st->pos0, ng, ratio, (uint32_t)v->candidate_topk_blocks, (uint32_t)v->candidate_block_size, posd)) return false;
        st->cand_owner = (int16_t)il;
    }
    const uint32_t topk = DS4_N_INDEXER_TOP_K < ng ? DS4_N_INDEXER_TOP_K : ng;   /* min(index_topk, end_pos // ratio); graph 路 = 上限 */
    if (!ds4_gpu_v41_indexer_topk_tensor(st->idx, st->iscore, n, ng, topk, ratio, posd)) return false;
    st->idx_topk = topk;
    st->idx_ratio = ratio;   /* 注意力按每个 query 自己的位置算段长要用它(见 ds4_gpu_v41.h 的 ratio 注释) */
    return true;
}

/* ★CED 的分界层专用★: 只把本层的压缩 KV + 索引键写进源层缓存, 其余(q 路/窗口 kv/注意力/输出投影)全不做。
 * 为什么够: 解码器段所有层的全局 KV 都读这一层的缓存(kv_source_of), 而中间块的解码器段不跑, 也就没人要
 * 这一层的注意力输出。出错会怎样: 漏了这一步, 解码器段在最后一块里读到的全局 KV 只有最后一块那几百个位置,
 * 前面的上下文整段丢失 —— 表现是长提示答非所问, 不报错。 */
bool v41_attention_kv_only(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (!ratio || !g_ds4_v41.is_kv_source[il]) return true;
    return v41_compress_source(e, st, il, ratio);
}

/* ★三塔的投影按盘上类型分发(2026-09-17, mtp-1.md M6 真因)★
 * 三塔在原件里是 FP8(E4M3 + 32×32), 2026-09-17 之前转换器把它们跟主干骨架一起压成了 FP4 ——
 * 草稿器对 FP 原模型的首位一致率因此只剩 0.50(最容易的一档也只有 0.81, 而底座 0.977)。
 * 现在盘上存回 FP8, 这里按登记类型走两条路。★老 GGUF 仍然能跑★: 它们登记的是 fp4x32, 走下面那条。 */
bool v41_tproj(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                      uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n, int round_out) {
    if (w->type == DS4_TENSOR_FP8_32X32)
        return ds4_gpu_v41_matmul_fp8blk_round_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n, round_out) != 0;
    if (w->type == DS4_TENSOR_Q4_K)   /* 100 GB 配方的骨架(2026-09-19); 老 GGUF 的 fp4x32 走下面 */
        return ds4_gpu_v41_matmul_q4k_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n, round_out) != 0;
    return ds4_gpu_v41_matmul_fp4x32_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n, round_out) != 0;
}
/* 嵌入取行也要按类型分发: token_embd 与 output 跟骨架同一档(fp4x32 或 q4_K)。
 * 少了这一支的后果不是报错 —— 是拿 fp4x32 的解码器去读 q4_K 的字节, 每个 token 的词向量全错。 */
bool v41_embed(const ds4_model *m, ds4_gpu_tensor *out, const ds4_gpu_tensor *tok,
               const ds4_tensor *w, uint64_t n_vocab, uint32_t n, uint64_t dim) {
    if (w->type == DS4_TENSOR_Q4_K)
        return ds4_gpu_v41_embed_q4k_tensor(out, tok, m->map, m->size, w->abs_offset, n_vocab, n, dim) != 0;
    return ds4_gpu_v41_embed_fp4x32_tensor(out, tok, m->map, m->size, w->abs_offset, (uint32_t)n_vocab, n, (uint32_t)dim) != 0;
}
bool v41_tproj_grouped(const ds4_model *m, ds4_gpu_tensor *low, const ds4_tensor *w, uint32_t n_groups,
                              uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n, int round_out) {
    if (w->type == DS4_TENSOR_FP8_32X32)
        return ds4_gpu_v41_grouped_matmul_fp8blk_tensor(low, m->map, m->size, w->abs_offset, n_groups, group_dim, rank, heads, n, round_out) != 0;
    if (w->type == DS4_TENSOR_Q4_K)
        return ds4_gpu_v41_grouped_matmul_q4k_tensor(low, m->map, m->size, w->abs_offset, n_groups, group_dim, rank, heads, n, round_out) != 0;
    return ds4_gpu_v41_grouped_matmul_fp4x32_tensor(low, m->map, m->size, w->abs_offset, n_groups, group_dim, rank, heads, n, round_out) != 0;
}

/* ★DSpark 草稿塔的注意力(官方 DSparkAttention)★
 * 与主路的三处不同: ①没有压缩 KV/indexer(compress_ratio==0), 只有 SWA 窗口;
 * ②窗口里装的不是塔自己的 kv, 而是**主模型那一位的 main_x 投影**(由 v41_draft_push_main 推进);
 * ③块内 n 位互相全可见(官方 get_dspark_topk_idxs), 所以 sparse_attn 传 full_block。
 * ★块的 kv 不进历史窗口★: 草稿是试算, 只有主模型确认过的位置才配进窗口 —— 写进去了下一轮就在
 * 一份含"没被接受的草稿"的历史上出草稿, 不报错, 接受率慢慢烂掉。 */
static bool v41_draft_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.mtp.tower[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS)) return false;
    if (!v41_tproj(m, st->q, l->attn_q_b, Q, (uint64_t)NH * HD, st->qrn, n, 1)) return false;
    if (!v41_rope(st->q, st->pos, n, NH, HD, 0, false)) return false;
    if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->kvn, st->pos, n, 1, HD, 0, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) return false;
    if (!ds4_gpu_tensor_copy(st->win[il], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)n * rowb)) return false;
    /* ring=0: 草稿塔的窗口是 v41_draft_push_main 用整体左移维护的线性段, 不是主路那个环 */
    if (!ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, st->win[il], NULL, NULL, m->map, m->size,
                                        l->attn_sinks->abs_offset, n, st->pos0, SWA, 0, 0, 0, NH, HD,
                                        (float)(1.0 / sqrt((double)HD)), 1, 0, 0u, NULL, 0u)) return false;   /* 草稿塔窗口由 push_main 整段维护, 无脏槽 */
    if (!v41_rope(st->o, st->pos, n, NH, HD, 0, true)) return false;
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP, (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1)) return false;
    return v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1);
}

/* 把 rows 个**已确认位置**的 main_x 投影进每个塔的窗口(官方 self.window_kv_cache[start_pos % win] = main_kv)。
 * 我们的窗口是线性缓冲不是环形: 新行先写块区, 再整体左移 rows 行 —— 效果一样, 省一个取模索引。
 * pos 张量由调用方填成这 rows 个位置的绝对位置(rope 要它)。 */
bool v41_draft_push_main(ds4_engine *e, ds4_v41_state *st, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint32_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA;
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!rows) return true;
    for (uint32_t T = 0; T < g_ds4_v41.mtp_towers; T++) {
        const ds4_layer_weights *l = &e->weights.mtp.tower[T];
        if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->main_x, rows, 1)) return false;
        if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, rows, DS4_RMS_EPS)) return false;
        if (!v41_rope(st->kvn, st->pos, rows, 1, HD, 0, false)) return false;
        if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, rows, HD, 32)) return false;
        if (!ds4_gpu_tensor_copy(st->win[T], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)rows * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->wintmp, 0, st->win[T], (uint64_t)rows * rowb, (uint64_t)SWA * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->win[T], 0, st->wintmp, 0, (uint64_t)SWA * rowb)) return false;
    }
    return true;
}

bool v41_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    if (st->draft) return v41_draft_attention(e, st, il);
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint64_t rowb = (uint64_t)HD * 4;
    /* ★kv 支挂侧流, 与 q 支并行(2026-09-23, 纯解码 n=1)★: 两支只共读 xn/pos, 各写各的(kv/kvn 对 qr/qrn/q), 原来串行 ——
     * kv 支 ~18 µs(kv GEMV + rms + rope + act_quant, 后三个是单/小 block 核)藏到 q_b(~107 µs)后面。算式与次序不变 ⇒ 逐字节同。
     * 进窗口环的那发拷贝没有流参数(落主流), 放到汇合之后。
     * ★放宽到 n ≤ DS4_V41_GEMV_MAX_TOK(2026-09-24, 投机验证批)★: 原来只开 n=1, 理由是"预填/验证批的 GEMM 路要用共享暂存"。
     * 可 n ≤ 8 两支走的全是 GEMV(q4k/fp4/fp8 即乘)+ rms/rope/act_quant, 没有一个碰 g_v41_xbf 这类共享暂存 —— 只有 n > 8 的预填
     * GEMM 路才转 bf16 进共享缓冲。验证批(1+k ≤ 6 行)因此一直串行, 比纯解码多付 ~3 ms 截距。n > 8 仍不分叉。 */
    const int fork = n <= DS4_V41_GEMV_MAX_TOK && ds4_gpu_side_mark() && ds4_gpu_side_begin();
    /* 窗口 kv: wkv → bf16 → kv_norm → rope(末 64 维) → fp8 act_quant(按 32 块) → 进窗口缓冲的后 n 行 */
    if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1) ||
        !ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS) ||
        !v41_rope(st->kvn, st->pos, n, 1, HD, ratio, false) ||
        !ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) { if (fork) (void)ds4_gpu_side_join(); return false; }
    if (fork) (void)ds4_gpu_side_main();
    /* q 路: q_a → bf16 → q_norm → q_b → bf16 → rope(绝对位置) */
    if (!v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1) ||
        !ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS) ||
        !v41_tproj(m, st->q, l->attn_q_b, Q, (uint64_t)NH * HD, st->qrn, n, 1) ||
        !v41_rope(st->q, st->pos, n, NH, HD, ratio, false)) { if (fork) (void)ds4_gpu_side_join(); return false; }
    if (fork && !ds4_gpu_side_join()) return false;
    if (!ds4_gpu_tensor_copy(st->win[il], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)n * rowb)) return false;
    /* 压缩侧: 源层先产出, 消费层读最近源层的缓存 + 本 chunk 最近 indexer 源层的 topk */
    uint32_t ng = 0, topk = 0, iratio = 0; const ds4_gpu_tensor *comp = NULL;
    const ds4_gpu_tensor *posd = st->graph ? st->pos : NULL;   /* graph 路: 位置走设备槽, ng/topk 是桶上限 */
    if (ratio) {
        if (v->is_kv_source[il] && !v41_compress_source(e, st, il, ratio)) return false;
        if (v->is_index_source[il] && !v41_index_source(e, st, il, ratio)) return false;
        const int16_t src = v->kv_source_of[il];
        if (src < 0 || !st->comp_kv[src]) { fprintf(stderr, "ds4: V4.1 L%u 压缩层无源\n", il); return false; }
        ng = st->graph ? (st->graph_pos_cap + n) / ratio : st->ng_src[src];
        if (ng) {
            if (st->idx_owner < 0) { fprintf(stderr, "ds4: V4.1 L%u 压缩层无 topk\n", il); return false; }
            comp = st->comp_kv[src]; topk = st->idx_topk; iratio = st->idx_ratio;
            /* graph 路的核按一个 ratio 同时推 ng 与段长, 所以 topk 来源层的压缩比必须就是本层的(接线表保证; 不成立就停车) */
            if (st->graph && iratio != ratio) { fprintf(stderr, "ds4: V4.1 L%u graph 路: topk 源层压缩比 %u ≠ 本层 %u\n", il, iratio, ratio); return false; }
        }
    }
    /* 稀疏注意力(窗口 128 + topk 压缩行, sink 进分母) → bf16 → 逆 rope */
    if (!ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, st->win[il], (ng && topk) ? comp : NULL, (ng && topk) ? st->idx : NULL, m->map, m->size,
                                        l->attn_sinks->abs_offset, n, st->pos0, SWA, ng, topk, iratio, NH, HD,
                                        (float)(1.0 / sqrt((double)HD)), 0, 1, st->win_from[il], posd, st->graph_pos_cap)) return false;   /* 窗口范围也随位置走, 纯窗口层同样要 posd; win_from 钳掉 CED 没写过的环槽 */
    /* ★本批的 n 行进环★(decode.md D1; 原来是"整段左移两发 256 KB 拷贝", 每层每步白搬 512 KB):
     * 注意力已经算完 —— 这一步之前环里装的还是历史 [pos0-SWA, pos0-1], 正是上面要读的;
     * 算完才提交, 顺序不能倒。投机的部分接受由 v41_spec_rollback 把没接受的格子还原回去。 */
    /* ★验证批进图(2026-09-22)★: "存 commit 将盖掉的那 n 格"这一发进图(位置从设备槽), 与直发路 v41_spec_snapshot 存的是同几格;
     * 快照缓冲由直发那一轮建(捕获态不许分配), 没建 = 调用方没先直发暖过这个 n。 */
    if (st->graph && n > 1u) {
        if (!st->snap_win[il]) { fprintf(stderr, "ds4: [graph] L%u 窗口环的快照缓冲还没建(验证批要先直发跑一轮)\n", il); return false; }
        if (!ds4_gpu_v41_win_ring_snap_tensor(st->win[il], st->snap_win[il], 0u, 0u, n, SWA, HD, 0, posd)) return false;
    }
    if (!ds4_gpu_v41_win_commit_tensor(st->win[il], st->pos0, n, SWA, HD, posd)) return false;
    if (!v41_rope(st->o, st->pos, n, NH, HD, ratio, true)) return false;
    /* 输出投影: 分组 wo_a(块对角) → bf16 → wo_b → bf16 */
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP,
                                                  (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1)) return false;
    /* ★wo_b 也必须走分发器★(2026-09-20 定罪): 这一行原来直接调 fp4x32 核, 是骨架矩阵里唯一漏改的一处。
     * q4_K 骨架下它拿 fp4x32 解码器读 q4_K 的字节 —— fp4x32 每 32 元素末尾那个字节当 ue8m0 指数(2^(b−127)),
     * q4_K 的 f16/6-bit 字节随机落到那个位置就是 2^100 量级 ⇒ attn_out 溢出成 inf/NaN, L00 之后整份 hc
     * 非有限、路由塌到一个专家、生成全是 BOS(fable5 09-19 夜的现象逐条对上)。不报错, 只出假数。 */
    if (!v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1)) return false;
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_attn_nonempty_tu;
