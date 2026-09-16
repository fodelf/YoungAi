/* core_v41_forward.c — DeepSeek V4.1 增量前向驱动(2026-09-12 战役 P2a 批前向 → P2c 持久状态): 一次喂 n 个 token
 * (prefill 块或解码的 1 个), 过 40 层, 出这 n 个位置的 logits, 各层缓存追加。
 *
 * 逐式对照官方 inference/model.py 的 Transformer.forward / Block.forward / MoE.forward(那是 V4.1 唯一
 * ground truth): 嵌入 → 展成 hc 份 → 每层 [engram] → hc_mixes(attn) → hc_pre(pre_mix) → attn_norm →
 * attention → hc_post → hc_mixes(ffn) → hc_pre(attn_pre) → ffn_norm → MoE → hc_post, pre_mix ← ffn_pre →
 * 末层后 hc_pre(pre_mix) → norm → head。bf16 舍入点跟官方 dtype 流走(见 ds4_gpu_v41.h 头注释)。
 * 出口(--score-ids/生成)在 core_v41_api.c。 */
#include "core_internal.h"
#include "src/common/ds4_quantfmt.h"   /* DS4_GGT_*: 路由 gate 在盘上是 f32 还是 bf16, 按登记类型认 */
#ifndef DS4_NO_GPU

ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes ? bytes : 16);
    if (!t) *ok = false;
    return t;
}

bool v41_state_alloc(ds4_v41_state *st, uint32_t cap, uint32_t ctx) {
    memset(st, 0, sizeof *st);
    st->cap_tok = cap; st->ctx = ctx; st->idx_owner = -1; st->cand_owner = -1;
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) st->eshard[i].fd = -1;
    if (cap == 0 || ctx < cap) { fprintf(stderr, "ds4: V4.1 状态参数错(cap %u ctx %u)\n", cap, ctx); return false; }
    if (ctx > DS4_V41_MAX_CTX_P2C) { fprintf(stderr, "ds4: V4.1 当前 ctx 上限 %u(topk 核 shared 容量), 给了 %u\n", DS4_V41_MAX_CTX_P2C, ctx); return false; }
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint64_t IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED;
    const uint64_t mix = 2 * HC + HC * HC, low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    bool ok = true;
    st->hist = xmalloc((size_t)ctx * 4);
    st->tok = v41_alloc((uint64_t)cap * 4, &ok);          st->pos = v41_alloc((uint64_t)cap * 4, &ok);
    st->posg = v41_alloc((uint64_t)cap * 4, &ok);
    st->hc = v41_alloc((uint64_t)cap * HC * E * 4, &ok);  st->hc2 = v41_alloc((uint64_t)cap * HC * E * 4, &ok);
    st->mix = v41_alloc((uint64_t)cap * mix * 4, &ok);    st->pre = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->post = v41_alloc((uint64_t)cap * HC * 4, &ok);    st->comb = v41_alloc((uint64_t)cap * HC * HC * 4, &ok);
    st->pre_mix = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->x = v41_alloc((uint64_t)cap * E * 4, &ok);        st->xn = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->qr = v41_alloc((uint64_t)cap * Q * 4, &ok);       st->qrn = v41_alloc((uint64_t)cap * Q * 4, &ok);
    st->q = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);
    st->kv = v41_alloc((uint64_t)cap * HD * 4, &ok);      st->kvn = v41_alloc((uint64_t)cap * HD * 4, &ok);
    st->ckv = v41_alloc((uint64_t)cap * HD * 4, &ok);     st->csc = v41_alloc((uint64_t)cap * HD * 4, &ok);
    st->pooled = v41_alloc((uint64_t)cap * HD * 4, &ok);  st->latent = v41_alloc((uint64_t)cap * HD * 4, &ok);
    st->ktmp = v41_alloc((uint64_t)cap * IK * 4, &ok);
    st->wintmp = v41_alloc(SWA * HD * 4, &ok);
    st->iq = v41_alloc((uint64_t)cap * IH * IK * 4, &ok); st->iw = v41_alloc((uint64_t)cap * IH * 4, &ok);
    st->iscore = v41_alloc((uint64_t)cap * ctx * 4, &ok); st->cand = v41_alloc((uint64_t)cap * ctx, &ok);
    st->idx = v41_alloc((uint64_t)cap * DS4_N_INDEXER_TOP_K * 4, &ok);
    st->o = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);  st->low = v41_alloc((uint64_t)cap * low * 4, &ok);
    st->attn_out = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->glog = v41_alloc((uint64_t)cap * NE * 4, &ok);    st->sel = v41_alloc((uint64_t)cap * K * 4, &ok);
    st->rw = v41_alloc((uint64_t)cap * K * 4, &ok);       st->routed = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->sg = v41_alloc((uint64_t)cap * FF * 4, &ok);      st->su = v41_alloc((uint64_t)cap * FF * 4, &ok);
    st->sh = v41_alloc((uint64_t)cap * FF * 4, &ok);      st->so = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->y = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->logits = v41_alloc((uint64_t)cap * DS4_N_VOCAB * 4, &ok);
    /* DSpark 的 main_hidden: 留最后 SWA 行 —— 预填完要用它把三塔的 128 行窗口一次填满
     * (官方 DSparkBlock.forward(start_pos=0) 就是拿整段 main_kv 灌窗口), 之后每轮只进几行。 */
    if (g_ds4_v41.mtp_block && g_ds4_v41.n_mtp_target) {
        st->mainh_cap = DS4_N_SWA;
        st->mainh = v41_alloc((uint64_t)st->mainh_cap * g_ds4_v41.n_mtp_target * E * 4, &ok);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        st->win[il] = v41_alloc((SWA + cap) * HD * 4, &ok);
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) { fprintf(stderr, "ds4: V4.1 kv 源层 L%u 压缩比为 0\n", il); ok = false; break; }
        const uint64_t ngcap = (uint64_t)ctx / ratio + 1;
        /* ★按官方打包尺寸分配★(decode.md D1): 主 KV 288 B/组、索引 K 72 B/组, 不是 512/128 个 f32。
         * 常量在 ds4_gpu_v41.h 只有一份 —— 这里与核里的解包各写各的就会静默越界。 */
        st->comp_kv[il] = v41_alloc(ngcap * DS4_V41_CKV_BYTES, &ok);
        st->index_k[il] = v41_alloc(ngcap * DS4_V41_IDXK_BYTES, &ok);
        if (ratio > 1) {
            st->cpre_kv[il] = v41_alloc((ratio + (uint64_t)cap) * HD * 4, &ok);
            st->cpre_sc[il] = v41_alloc((ratio + (uint64_t)cap) * HD * 4, &ok);
        }
    }
    /* 三文件部署: ②反修目录 与 ③后训练目录 各自可缺席; 显式要了的挂不上就停车, 不许静默裸跑。 */
    if (ok && (g_ds4_v41_amp_dir || g_ds4_v41_pt_dir) &&
        !v41_amp_load(st, g_ds4_v41_amp_dir, g_ds4_v41_pt_dir)) {
        fprintf(stderr, "ds4: 插件挂不上, 停车(不做静默裸模型对照): 反修 %s / 后训练 %s\n",
                g_ds4_v41_amp_dir ? g_ds4_v41_amp_dir : "(无)", g_ds4_v41_pt_dir ? g_ds4_v41_pt_dir : "(无)");
        ok = false;
    }
    if (!ok) { fprintf(stderr, "ds4: V4.1 状态缓冲分配失败(cap=%u ctx=%u)\n", cap, ctx); v41_state_free(st); }
    return ok;
}

void v41_state_free(ds4_v41_state *st) {
    v41_amp_free(st);
    v41_engram_close(st);
    ds4_gpu_tensor **all[] = { &st->tok, &st->pos, &st->posg, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix,
        &st->x, &st->xn, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->ckv, &st->csc, &st->pooled, &st->latent, &st->ktmp, &st->wintmp,
        &st->iq, &st->iw, &st->iscore, &st->cand, &st->idx, &st->o, &st->low, &st->attn_out, &st->glog, &st->sel, &st->rw,
        &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->mainh };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { if (*all[i]) ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor **per[] = { &st->win[il], &st->comp_kv[il], &st->index_k[il], &st->cpre_kv[il], &st->cpre_sc[il],
                                   &st->snap_win[il], &st->snap_cpre_kv[il], &st->snap_cpre_sc[il] };
        for (size_t i = 0; i < sizeof(per) / sizeof(per[0]); i++) { if (*per[i]) ds4_gpu_tensor_free(*per[i]); *per[i] = NULL; }
    }
    free(st->hist); st->hist = NULL;
}

/* hc 三件: mix → pre/post/comb(官方 hc_mixes + hc_split_sinkhorn) */
static bool v41_hc_mixes(const ds4_model *m, ds4_v41_state *st, const ds4_tensor *fn, const ds4_tensor *scale, const ds4_tensor *base) {
    if (!ds4_gpu_v41_hc_mix_tensor(st->mix, st->hc, m->map, m->size, fn->abs_offset, DS4_N_EMBD, DS4_N_HC, st->n, DS4_RMS_EPS)) return false;
    return ds4_gpu_v41_hc_split_tensor(st->pre, st->post, st->comb, st->mix, m->map, m->size, scale->abs_offset, base->abs_offset,
                                       DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, st->n) != 0;
}

/* 半层的入口三件(single.md S3): mix(要多 block 的 GEMV, 单独一发) → 合一核(split + hc_pre + norm)。
 * 出来后 st->pre_mix = **本层**新产的 pre(给下半层/下一层), st->xn = 归一化后的子层输入。
 * ★顺序要紧★: 合一核里 hc_pre 用的是**进来时**的 pre_mix(上一层的), 交换必须在核之后做。 */
static bool v41_hc_half(ds4_engine *e, ds4_v41_state *st, const ds4_layer_weights *l, bool attn_half) {
    const ds4_model *m = &e->model;
    const ds4_tensor *fn = attn_half ? l->hc_attn_fn : l->hc_ffn_fn;
    const ds4_tensor *sc = attn_half ? l->hc_attn_scale : l->hc_ffn_scale;
    const ds4_tensor *bs = attn_half ? l->hc_attn_base : l->hc_ffn_base;
    const ds4_tensor *nm = attn_half ? l->attn_norm : l->ffn_norm;
    if (!ds4_gpu_v41_hc_mix_tensor(st->mix, st->hc, m->map, m->size, fn->abs_offset, DS4_N_EMBD, DS4_N_HC, st->n, DS4_RMS_EPS)) return false;
    if (!ds4_gpu_v41_hc_fused_tensor(st->pre, st->post, st->comb, st->x, st->xn, st->mix, st->hc, st->pre_mix,
                                     m->map, m->size, sc->abs_offset, bs->abs_offset, nm->abs_offset,
                                     DS4_N_EMBD, DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, st->n)) return false;
    { ds4_gpu_tensor *t = st->pre; st->pre = st->pre_mix; st->pre_mix = t; }
    return true;
}

/* MoE(官方 MoE.forward): 路由 f32 → routed(VQ) + shared(fp4) → bf16 */
static bool v41_moe(const ds4_model *m, const ds4_layer_weights *l, ds4_v41_state *st, uint32_t il) {
    const uint32_t n = st->n, E = DS4_N_EMBD, FF = DS4_N_FF_EXP;
    /* 草稿塔: 128 个专家 top-3, 每个专家是一个独立 fp4x32 张量(不是 VQ blob) —— 路由的形状也是塔自己的 */
    const uint32_t NE = st->draft ? g_ds4_v41.mtp_experts : DS4_N_EXPERT;
    const uint32_t KU = st->draft ? g_ds4_v41.mtp_used : DS4_N_EXPERT_USED;
    /* 路由 gate: 主干层盘上是 bf16, 草稿塔是 f32(转换器两条 plan 不同) —— 按登记类型认 */
    if (l->ffn_gate_inp->type == DS4_GGT_F32) {
        if (!ds4_gpu_v41_matmul_f32_tensor(st->glog, m->map, m->size, l->ffn_gate_inp->abs_offset, E, NE, st->xn, n)) return false;
    } else if (!ds4_gpu_v41_matmul_bf16_tensor(st->glog, m->map, m->size, l->ffn_gate_inp->abs_offset, E, NE, st->xn, n)) return false;
    if (!ds4_gpu_v41_router_tensor(st->sel, st->rw, st->glog, m->map, m->size, l->ffn_exp_probs_b->abs_offset, n, NE,
                                   KU, DS4_EXPERT_WEIGHT_SCALE)) return false;
    if (st->draft) {
        if (!ds4_gpu_v41_mtp_moe_tensor(st->routed, m->map, il, st->tower_exp_off[il], E, FF, E, st->sel, st->rw,
                                        NE, KU, DS4_SWIGLU_CLAMP_EXP, st->xn, n)) return false;
    } else {
    char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il);
    const ds4_tensor *blob = model_find_tensor(m, nm);
    if (!blob) return false;
    if (!ds4_gpu_v41_routed_moe_tensor(st->routed, m->map, m->size, blob->abs_offset, blob->bytes, E, FF, E, st->sel, st->rw,
                                       DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, st->xn, il, n)) return false;
    }
    /* shared expert: w1/w3 → bf16 → swiglu(截断) → bf16 → w2 → bf16 */
    /* ★判负存档(single.md S5①, 09-16)★: gate 与 up 同形状同输入, 合成一发试过 —— 55.2 vs 55.2, 持平。
     * 小矩阵的固定开销不在"发数"上(每发才 41 µs, 启动只占几微秒), 合发省不出东西。pair 那条核路已删。 */
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->sg, m->map, m->size, l->ffn_gate_shexp->abs_offset, E, FF, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->su, m->map, m->size, l->ffn_up_shexp->abs_offset, E, FF, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_swiglu_tensor(st->sh, st->sg, st->su, n, FF, DS4_SWIGLU_CLAMP_EXP)) return false;
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->so, m->map, m->size, l->ffn_down_shexp->abs_offset, FF, E, st->sh, n, 1)) return false;
    /* y = routed(f32 累加的 bf16 专家输出) + shared(bf16) → .type_as(x) bf16 */
    if (!ds4_gpu_tensor_copy(st->y, 0, st->routed, 0, (uint64_t)n * E * 4)) return false;
    if (!ds4_gpu_v41_add_tensor(st->y, st->so, (uint64_t)n * E)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->y, (uint64_t)n * E)) return false;
    if (st->draft) return true;   /* 反修/取料钩子只挂主干; 草稿塔是另一个模型, 不挂 */
    const int hk = v41_amp_hook(st, il);   /* 反修取料钩子(解算时才挂): 0 继续 / 1 取完了·本次前向到此为止 / <0 失败 */
    if (hk < 0) return false;
    if (hk > 0) return true;
    return v41_amp_apply(st, il);   /* 反修放大器(挂了才动): y += x·(B·A) → bf16, 与 Python 判决态 hook 同位 */
}

/* --v41-prof 下每层后扫一遍 hc 找非有限值(偶发 NaN/垃圾定位用): 报首个坏层/坏 token */
static void v41_nan_scan(const ds4_v41_state *st, uint32_t il) {
    const uint64_t cnt = (uint64_t)st->n * DS4_N_HC * DS4_N_EMBD;
    float *buf = xmalloc((size_t)cnt * 4);
    if (ds4_gpu_tensor_read(st->hc, 0, buf, cnt * 4)) {
        uint64_t bad = 0, first = 0;
        for (uint64_t i = 0; i < cnt; i++) if (!isfinite(buf[i])) { if (!bad) first = i; bad++; }
        if (bad) fprintf(stderr, "\n[v41-prof] ★L%02u 后 hc 有 %llu 个非有限值, 首个在 token %llu 路 %llu★\n", il, (unsigned long long)bad,
                         (unsigned long long)(first / ((uint64_t)DS4_N_HC * DS4_N_EMBD)), (unsigned long long)((first / DS4_N_EMBD) % DS4_N_HC));
    }
    free(buf);
}

static void v41_dump_rows(const ds4_v41_state *st, const ds4_gpu_tensor *t, const char *kind, uint32_t il, uint64_t rowf) {
    char p[4400]; float *buf = xmalloc((size_t)st->n * rowf * 4);
    ds4_gpu_synchronize();
    if (ds4_gpu_tensor_read(t, 0, buf, (uint64_t)st->n * rowf * 4)) {
        snprintf(p, sizeof p, "%s.%s_L%02u.bin", st->dump_prefix, kind, il);
        FILE *f = fopen(p, "wb"); if (f) { fwrite(buf, 4, (size_t)st->n * rowf, f); fclose(f); }
    }
    free(buf);
}

/* 一层(官方 Block.forward): 返回后 st->hc 是层输出, st->pre_mix 已换成本层 ffn_pre */
bool v41_layer(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model;
    const ds4_layer_weights *l = st->draft ? &e->weights.mtp.tower[il] : &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD;
    /* engram 层(官方 Transformer.forward: layer(h) 之前先 h = engram(h)); 草稿塔没有 engram */
    if (!st->draft && !st->no_engram && g_ds4_v41.engram_index_of[il] >= 0 && !v41_engram(e, st, il)) return false;
    /* DSpark: 目标层的注意力输入(= engram 之后、本层之前的 hc 四路均值)拼成 main_hidden。
     * 只搬本 chunk 末尾 mainh_cap 行 —— 草稿器只从"最后那几个已定 token"出发。 */
    if (!st->draft && st->mainh && g_ds4_v41.mtp_target_slot[il] >= 0) {
        const uint32_t rows = n < st->mainh_cap ? n : st->mainh_cap;
        if (!ds4_gpu_v41_hc_mean_tensor(st->mainh, st->hc, E, DS4_N_HC, rows, n - rows,
                                        (uint32_t)g_ds4_v41.mtp_target_slot[il], g_ds4_v41.n_mtp_target)) return false;
        st->mainh_rows = rows;
    }
    /* attn 半层: 本层 attn 的 mixes 产 attn_pre(给下面 ffn 用)/attn_post/attn_comb; 入口用上一层传来的 pre_mix */
    if (!v41_hc_half(e, st, l, true)) return false;
    if (!v41_attention(e, st, il)) return false;
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, E, DS4_N_HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    /* ffn 半层: mixes 产 ffn_pre(传给下一层)/ffn_post/ffn_comb; 入口用本层 attn_pre(在 pre_mix 槽) */
    if (!v41_hc_half(e, st, l, false)) return false;
    if (!v41_moe(m, l, st, il)) return false;
    if (st->stop_early) return true;   /* 钩子取完料: 余下半层不算(状态随即作废, 只要位置推进) */
    if (st->dump_prefix) { v41_dump_rows(st, st->xn, "x", il, E); v41_dump_rows(st, st->y, "y", il, E); }   /* 对拍夹具: MoE 入/出 */
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->y, st->hc, st->post, st->comb, E, DS4_N_HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }   /* pre_mix ← ffn_pre 的交换已在 v41_hc_half 里做过 */
    return true;
}

/* ---- 投机验证的快照与回滚(speed.md 段 6 D1; 窗口部分 2026-09-16 decode.md D1 改环) ----
 * 为什么要备份: 验证批把 1+k 个位置全算进缓存, 其中没被接受的那几位必须撤销。
 *   窗口(win): 本批的行先待在缓冲的块区, 层算完由 win_commit 写进环 —— 写进去就盖掉了
 *     128 步之前那一格(它还在下一步的可见窗口里)。所以**只备份将被盖掉的那 n 格**:
 *     40 层 × n(≤6) 行 × 512 × 4 B ≈ 0.5 MB。改环之前是整份 128 行 = 10.5 MB, 22 倍。
 *   压缩器余行(cpre_*): 那是真的"整体左移", 备份点在 core_v41_attn.c 的平移之前。
 * comp_kv/index_k 按绝对组号写、ng_src/cpend 只是计数 ⇒ 回退计数下一轮直接覆盖, 不用备份。 */
bool v41_spec_snapshot(ds4_v41_state *st, uint32_t n) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    const uint32_t nmax = DS4_MTP_MAX_BLOCK + 1u;   /* 验证批 = 1 个已确认位 + 最多 block 个草稿位 */
    bool ok = true;
    if (!n || n > nmax) { fprintf(stderr, "ds4: V4.1 投机快照批 %u 超上限 %u\n", n, nmax); return false; }
    /* 存的是"环里即将被本批第 i 行盖掉的那一格", 按批内行号 i 排 ⇒ 还原区间正好是 [keep, n)。 */
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        if (!st->snap_win[il]) st->snap_win[il] = v41_alloc((uint64_t)nmax * rowb, &ok);
        if (ok && !ds4_gpu_v41_win_ring_snap_tensor(st->win[il], st->snap_win[il], st->n_past,
                                                    0u, n, DS4_N_SWA, DS4_N_HEAD_DIM, 0)) ok = false;
    }
    st->snap_n = n;
    /* 压缩器余行的快照点不在这里, 而在 core_v41_attn.c 的 v41_compress_source 里"平移之前" ——
     * 那一刻缓冲里才是"旧余行 + 本批全部新行", 回滚要的正是它(平移一走就盖掉了)。 */
    st->snap_past = st->n_past;
    st->snap_on = ok ? 1u : 0u;
    return ok;
}

/* keep = 这一批里被接受的位置数(含那个已确认的首位)。三件事:
 *   ①窗口(环): 本批第 keep..n-1 行是没被接受的, 它们在层里已经被 commit 进环、盖掉了 128 步前的真行 ——
 *      把快照里那几格写回去。被接受的前 keep 行本来就该留在环里, 一个字节不动。
 *   ②压缩 KV/索引键: 只把组数回退到 (pos0+keep)/ratio, 内容不用管(按绝对组号写, 下一轮覆盖);
 *   ③压缩器余行: 从"平移前"的快照里取最后 pend' 行接回头部 —— 那几行是被接受 token 的压缩器输入。
 * 出错会怎样: 漏了 ③, 下一组池化会把一个没被接受的草稿 token 混进去, 不报错, 只是那一组的
 * 全局 KV 与纯解码路对不上(温 0 逐字节门会抓); 漏了 ①, 下一步把"未来的草稿"当历史读, 症状相同。 */
bool v41_spec_rollback(ds4_v41_state *st, uint32_t keep) {
    if (!st->snap_on) return false;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    const uint32_t pos0 = st->snap_past, np = pos0 + keep;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (keep < st->snap_n &&
            !ds4_gpu_v41_win_ring_snap_tensor(st->win[il], st->snap_win[il], pos0,
                                              keep, st->snap_n, DS4_N_SWA, DS4_N_HEAD_DIM, 1)) return false;
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        st->ng_src[il] = np / ratio;
        if (ratio == 1u || !st->cpre_kv[il]) { st->cpend[il] = 0; continue; }
        const uint32_t pend2 = np % ratio;
        if (pend2) {   /* 快照里第 (snap_cpend + keep - pend2) 行起的 pend2 行 = 最后几个被接受 token 的输入 */
            const uint64_t src = (uint64_t)(st->snap_cpend[il] + keep - pend2) * rowb;
            if (!ds4_gpu_tensor_copy(st->cpre_kv[il], 0, st->snap_cpre_kv[il], src, (uint64_t)pend2 * rowb)) return false;
            if (!ds4_gpu_tensor_copy(st->cpre_sc[il], 0, st->snap_cpre_sc[il], src, (uint64_t)pend2 * rowb)) return false;
        }
        st->cpend[il] = pend2;
    }
    st->n_past = np;
    st->snap_on = 0;
    return true;
}

bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    const uint32_t E = DS4_N_EMBD;
    if (n == 0 || n > st->cap_tok) { fprintf(stderr, "ds4: V4.1 前向块 %u 超 cap %u\n", n, st->cap_tok); return false; }
    if (st->n_past + n > st->ctx) { fprintf(stderr, "ds4: V4.1 上下文满(%u+%u > %u)\n", st->n_past, n, st->ctx); return false; }
    st->n = n; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->stop_early = 0;
    memcpy(st->hist + st->pos0, ids, (size_t)n * 4);
    {
        int32_t *pos = xmalloc((size_t)n * 4);
        for (uint32_t i = 0; i < n; i++) pos[i] = (int32_t)(st->pos0 + i);
        const bool ok = ds4_gpu_tensor_write(st->tok, 0, ids, (uint64_t)n * 4) && ds4_gpu_tensor_write(st->pos, 0, pos, (uint64_t)n * 4);
        free(pos);
        if (!ok) return false;
    }
    float *pm = xmalloc((size_t)n * DS4_N_HC * 4);   /* pre_mix = one-hot(第 0 路) */
    for (uint32_t i = 0; i < n * DS4_N_HC; i++) pm[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    const bool okpm = ds4_gpu_tensor_write(st->pre_mix, 0, pm, (uint64_t)n * DS4_N_HC * 4) != 0;
    free(pm);
    if (!okpm) return false;
    if (!st->no_engram && !v41_engram_prefetch(e, st)) return false;   /* engram 行读盘与前面几层的 GPU 算重叠 */
    if (ds4_gpu_begin_commands() == 0) return false;
    bool ok = ds4_gpu_v41_embed_fp4x32_tensor(st->x, st->tok, e->model.map, e->model.size, e->weights.token_embd->abs_offset, DS4_N_VOCAB, n, E) != 0;
    if (ok) ok = ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, DS4_N_HC, n) != 0;
    const double t0 = now_sec();
    double lt[DS4_N_LAYER + 1], tprev = t0;
    /* ★CED 分界层 = 最后一个 kv 源层★(元数据驱动, 不写死层号: 见铁律"引擎禁版本名硬编码")。
     * 官方 V4.1 把 40 层切成编码器/解码器两段, 解码器段所有层的全局 KV 都读分界层的那一槽,
     * 所以提示的中间块跑到分界层、把 KV 写进缓存就可以收工, 后半段的层没有任何下游消费者。 */
    uint32_t ced_edge = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) if (g_ds4_v41.is_kv_source[il]) ced_edge = il;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (st->ced_skip && il == ced_edge) {   /* 只补全局 KV, 本块到此为止 */
            ok = v41_hc_half(e, st, &e->weights.layer[il], true) && v41_attention_kv_only(e, st, il);
            if (ok && ds4_gpu_flush_commands() == 0) ok = false;
            if (!ok) break;
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_synchronize() == 0) return false;
            st->n_past += n;
            if (n >= 64u) fputc('\n', stderr);
            return true;   /* 中间块不出 logits(没人读: 生成只要最后一块的末位) */
        }
        ok = v41_layer(e, st, il);
        if (ok && (g_ds4_v41_prof || n >= 64u) && ds4_gpu_flush_commands() == 0) ok = false;   /* 解码不逐层同步(省 40 次停等), 查速度/大块才同步 */
        if (g_ds4_v41_prof) { const double tn = now_sec(); lt[il] = tn - tprev; if (ok) v41_nan_scan(st, il); tprev = now_sec(); }
        if (n >= 64u) fprintf(stderr, "[v41] pos %u+%u L%02u %s %.1fs\r", st->pos0, n, il, ok ? "ok" : "★失败★", now_sec() - t0);
        if (ok && st->stop_early) break;   /* 反修钩子取完料: 余下层与出口不算 */
    }
    if (n >= 64u) fputc('\n', stderr);
    if (!ok) { fprintf(stderr, "ds4: V4.1 前向失败(pos0 %u n %u)\n", st->pos0, n); return false; }
    if (st->stop_early) {   /* 位置照常推进(下一块的缓存/位置接得上), 不出 logits */
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_synchronize() == 0) return false;
        st->n_past += n;
        return true;
    }
    /* 出口: h = hc_pre(hc, pre_mix) → norm → head(fp4x32, f32 logits), 全 n 行 */
    if (!ds4_gpu_v41_hc_pre_tensor(st->x, st->hc, st->pre_mix, E, DS4_N_HC, n)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, st->x, e->model.map, e->model.size, e->weights.output_norm->abs_offset, E, n, DS4_RMS_EPS)) return false;
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->logits, e->model.map, e->model.size, e->weights.output->abs_offset, E, DS4_N_VOCAB, st->xn, n, 0)) return false;
    if (ds4_gpu_end_commands() == 0) return false;
    if (ds4_gpu_synchronize() == 0) return false;
    if (g_ds4_v41_prof) {   /* 逐层毫秒: 一眼看出哪层在吃时间(engram 层 L1/L14, 源层 L2/8/14/20, 候选层 L20) */
        lt[DS4_N_LAYER] = now_sec() - tprev;
        fprintf(stderr, "[v41-prof] pos %u+%u 总 %.1f ms | 层(ms):", st->pos0, n, (now_sec() - t0) * 1e3);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) fprintf(stderr, " %.1f", lt[il] * 1e3);
        fprintf(stderr, " | 出口 %.1f\n", lt[DS4_N_LAYER] * 1e3);
    }
    st->n_past += n;
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_forward_nonempty_tu;
