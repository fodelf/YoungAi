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

/* hc 三件: mix → pre/post/comb(官方 hc_mixes + hc_split_sinkhorn) */
static bool v41_hc_mixes(const ds4_model *m, ds4_v41_state *st, const ds4_tensor *fn, const ds4_tensor *scale, const ds4_tensor *base) {
    if (!ds4_gpu_v41_hc_mix_tensor(st->mix, st->hc, m->map, m->size, fn->abs_offset, DS4_N_EMBD, DS4_N_HC, st->n, DS4_RMS_EPS)) return false;
    return ds4_gpu_v41_hc_split_tensor(st->pre, st->post, st->comb, st->mix, m->map, m->size, scale->abs_offset, base->abs_offset,
                                       DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, st->n) != 0;
}

/* 半层的入口三件(single.md S3): mix(要多 block 的 GEMV, 单独一发) → 合一核(split + hc_pre + norm)。
 * 出来后 st->pre_mix = **本层**新产的 pre(给下半层/下一层), st->xn = 归一化后的子层输入。
 * ★顺序要紧★: 合一核里 hc_pre 用的是**进来时**的 pre_mix(上一层的), 交换必须在核之后做。 */
bool v41_hc_half(ds4_engine *e, ds4_v41_state *st, const ds4_layer_weights *l, bool attn_half) {
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

/* ★小批的专家重合度探针(--v41-prof, n>1 才有意义)★
 * 现役 VQ 路是 n_tok × top-k 个 (token, 槽) 对, 每对独立读一份专家权重 —— 同一个专家被两个 token 选中
 * 就把它读两遍。投机验证批一次 1+k 行, 所以"验证多一位要多付多少专家字节"就等于这里的唯一专家数增量。
 * 09-17 实测(12k): 唯一专家 6 / 9.96 / 13.26 / 17.38 / 21.71(n=1..5) ⇒ **每多一个 token 多 3.9 个新专家**,
 * 这是单流投机的上限来源(mtp-2.md §2.2), 也是张量核并集核该省的那一笔。
 * ★2026-09-17 扩到草稿塔★(mtp-2.md §6.1): 塔的 MoE 是 3 塔 × 5 位 × top-3 = 45 对, 而块里第 1..4 位吃的是
 * 同一个 noise 嵌入(只差位置) ⇒ 路由重合应当远高于主干。它定 §6.1 那一刀的字节账, 量出来才知道值不值。
 * 代价: 要把 sel 读回主机 = 每层一次同步, 所以只在 prof 下开。 */
static void v41_moe_uniq_probe(ds4_v41_state *st, uint32_t il, uint32_t topk, const char *tag) {
    int32_t s[8u * 8u];
    const uint32_t ns = st->n * topk;
    uint32_t uniq = 0;
    if (!g_ds4_v41_prof || st->n < 2u) return;
    if (ns > sizeof(s) / sizeof(s[0])) {
        /* ★预填块的路由分布(2026-09-29)★: 预填专家核按专家开工作项(≤ BN 个 token 一项, 超了再开一项 = 位流再解一遍), 所以每层"有 token 的专家数 /
         * 最大专家的 token 数 / 项数"决定它的成本; 微基准按均匀哈希路由(384 个专家各 n/64 个)算, 真路由偏斜多少只有量了才知道。 */
        int32_t *sb = xmalloc((size_t)ns * 4); uint32_t cnt[1024]; memset(cnt, 0, sizeof cnt);
        if (ds4_gpu_synchronize() && ds4_gpu_tensor_read(st->sel, 0, sb, (uint64_t)ns * 4)) {
            uint32_t ne = 0, mx = 0, items32 = 0, items64 = 0, items128 = 0;
            for (uint32_t i = 0; i < ns; i++) if (sb[i] >= 0 && sb[i] < 1024) cnt[sb[i]]++;
            for (uint32_t e = 0; e < 1024u; e++) if (cnt[e]) { ne++; if (cnt[e] > mx) mx = cnt[e]; items32 += (cnt[e] + 31u) / 32u; items64 += (cnt[e] + 63u) / 64u; items128 += (cnt[e] + 127u) / 128u; }
            fprintf(stderr, "[%s] L%02u n=%u: 有 token 的专家 %u, 最大 %u 个 token, 工作项 BN32 %u / BN64 %u / BN128 %u\n", tag, il, st->n, ne, mx, items32, items64, items128);
            /* 首块的逐专家计数落盘(/tmp/v41_route_Lnn_nNNNN.txt, 一行一个专家): 给 gguf-tools/bench/v41_vq_prefill_mma_bench.cu 当真路由用 */
            if (st->pos0 == 0u) {
                char p[96]; snprintf(p, sizeof p, "/tmp/v41_route_L%02u_n%u.txt", il, st->n);
                FILE *f = fopen(p, "w");
                if (f) { for (uint32_t e = 0; e < DS4_N_EXPERT; e++) fprintf(f, "%u\n", cnt[e]); fclose(f); }
            }
        }
        free(sb);
        return;
    }
    if (!ds4_gpu_synchronize() || !ds4_gpu_tensor_read(st->sel, 0, s, (uint64_t)ns * 4)) return;
    for (uint32_t i = 0; i < ns; i++) { uint32_t j = 0; while (j < i && s[j] != s[i]) j++; if (j == i) uniq++; }
    fprintf(stderr, "[%s] L%02u n=%u: 唯一专家 %u / %u\n", tag, il, st->n, uniq, ns);
}

/* MoE(官方 MoE.forward): 路由 f32 → routed(VQ) + shared(fp4) → bf16 */
bool v41_moe(const ds4_model *m, const ds4_layer_weights *l, ds4_v41_state *st, uint32_t il) {
    const uint32_t n = st->n, E = DS4_N_EMBD, FF = DS4_N_FF_EXP;
    const bool tail = !st->draft && n <= DS4_V41_GEMV_MAX_TOK;   /* 解码小批走 VQ 即乘核 ⇒ MoE 尾巴可以一发收 */
    /* 草稿塔: 128 个专家 top-3, 盘上是逐专家 fp4x32 张量或一个 VQ blob(按绑定时看到的形态) —— 路由的形状也是塔自己的 */
    const uint32_t NE = st->draft ? g_ds4_v41.mtp_experts : DS4_N_EXPERT;
    const uint32_t KU = st->draft ? g_ds4_v41.mtp_used : DS4_N_EXPERT_USED;
    /* ★shared 专家挂侧流, 与路由 + routed 专家并行(2026-09-23, 纯解码 n=1)★: 两支只共读 xn, 写 sg/su/sh/so 对 glog/sel/rw/专家暂存,
     * 原来串行。路由那一段(router GEMV + 单 block 的 router 核 + xpack)的延迟藏到 shared 三发 GEMV 后面。算式不变 ⇒ 逐字节同。
     * 只开主干(草稿塔另有暂存共用, 不动)。★n ≤ DS4_V41_GEMV_MAX_TOK 都开(2026-09-24)★: 验证批两支同样只走 GEMV/逐元素核,
     * routed 的 VQ 小批核用自己的暂存, 与 shared 三发不相交; n > 8 的预填 GEMM 路共用 bf16 暂存, 不分叉。 */
    const int fork = !st->draft && n <= DS4_V41_GEMV_MAX_TOK && ds4_gpu_side_mark() && ds4_gpu_side_begin();
    if (fork) {
        if (!v41_tproj(m, st->sg, l->ffn_gate_shexp, E, FF, st->xn, n, 1) ||
            !v41_tproj(m, st->su, l->ffn_up_shexp, E, FF, st->xn, n, 1) ||
            !ds4_gpu_v41_swiglu_tensor(st->sh, st->sg, st->su, n, FF, DS4_SWIGLU_CLAMP_EXP) ||
            !v41_tproj(m, st->so, l->ffn_down_shexp, FF, E, st->sh, n, 1)) { (void)ds4_gpu_side_join(); return false; }
        (void)ds4_gpu_side_main();
    }
    /* 路由 gate: 主干层盘上是 bf16, 草稿塔是 f32(转换器两条 plan 不同) —— 按登记类型认 */
    if (l->ffn_gate_inp->type == DS4_GGT_F32) {
        if (!ds4_gpu_v41_matmul_f32_tensor(st->glog, m->map, m->size, l->ffn_gate_inp->abs_offset, E, NE, st->xn, n)) return false;
    } else if (!ds4_gpu_v41_matmul_bf16_tensor(st->glog, m->map, m->size, l->ffn_gate_inp->abs_offset, E, NE, st->xn, n)) return false;
    if (!ds4_gpu_v41_router_tensor(st->sel, st->rw, st->glog, m->map, m->size, l->ffn_exp_probs_b->abs_offset, n, NE,
                                   KU, DS4_EXPERT_WEIGHT_SCALE)) return false;
    /* 后训练: 存路由选择 / 梯度检查的冻结选择前向照存的选(见 v41_tsave.replay)。草稿态只有草稿器蒸馏的训练态带 tsave(塔号当层号),
     * 推理的草稿态 tsave 恒 NULL —— 同一个判断, 不另开分支 */
    if (st->tsave && st->tsave->sel[il]) {
        v41_tsave *sv = st->tsave;
        const uint64_t sb = (uint64_t)n * KU * 4;
        if (sv->replay ? !ds4_gpu_tensor_copy(st->sel, 0, sv->sel[il], 0, sb) ||
                         !ds4_gpu_bwd_router_fixed_tensor(st->rw, st->sel, st->glog, n, NE, KU, DS4_EXPERT_WEIGHT_SCALE)
                       : !ds4_gpu_tensor_copy(sv->sel[il], 0, st->sel, 0, sb)) return false;
    }
    if (st->draft) {
        v41_moe_uniq_probe(st, il, KU, "mtp-uniq");
        const ds4_tensor *tb = st->tower_exps_vq[il];
        if (tb) {
            /* ★塔专家是 VQ blob(100 GB 配方, 2026-09-19 起)★: 直接用主干的解码即乘核 —— blob 槽表按专家号寻址, 128 个专家
             * 与 384 个只差表长, 塔的 sel/rw 本来就是 [n][KU]。层号传 DS4_N_LAYER+塔号: 它只用来挑 gr 侧车与头缓存的槽位
             * (塔没有 gr, 那几格恒空)。块 n ≤ DS4_MTP_MAX_BLOCK ≤ DS4_V41_GEMV_MAX_TOK ⇒ 永远走解码即乘核, 不进预填 GEMM 路。
             * 09-20 之前对 blob 直接不武装(拿逐专家核读 blob 会出一整套假草稿, 只表现为接受率低, 不报错)。 */
            if (!ds4_gpu_v41_routed_moe_tensor(st->routed, m->map, m->size, tb->abs_offset, tb->bytes, E, FF, E, st->sel, st->rw,
                                               NE, KU, DS4_SWIGLU_CLAMP_EXP, st->xn, (uint32_t)DS4_N_LAYER + il, n)) return false;
        } else if (!ds4_gpu_v41_mtp_moe_tensor(st->routed, m->map, il, st->tower_exp_off[il], E, FF, E, st->sel, st->rw,
                                               NE, KU, DS4_SWIGLU_CLAMP_EXP, st->xn, n)) return false;
    } else {
    v41_moe_uniq_probe(st, il, DS4_N_EXPERT_USED, "moe-uniq");
    char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il);
    const ds4_tensor *blob = model_find_tensor(m, nm);
    if (!blob) return false;
    /* 解码小批: 不做归约那一发, 部分和留在核侧, 下面 shared 专家算完后一发收尾(ds4_gpu_v41.h 的 out==NULL 口径) */
    if (!ds4_gpu_v41_routed_moe_tensor(tail ? NULL : st->routed, m->map, m->size, blob->abs_offset, blob->bytes, E, FF, E, st->sel, st->rw,
                                       DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, st->xn, il, n)) return false;
    }
    /* shared expert: w1/w3 → bf16 → swiglu(截断) → bf16 → w2 → bf16 */
    /* ★判负存档(single.md S5①, 09-16)★: gate 与 up 同形状同输入, 合成一发试过 —— 55.2 vs 55.2, 持平。
     * 小矩阵的固定开销不在"发数"上(每发才 41 µs, 启动只占几微秒), 合发省不出东西。pair 那条核路已删。 */
    /* ★按盘上类型分发★: 主干的 shared 专家是 fp4x32; 三塔的在原件里是 FP8, 2026-09-17 起原样存
     * (v41_tproj 两条路都走, 老 GGUF 仍然是 fp4x32) */
    if (fork) { if (!ds4_gpu_side_join()) return false; }
    else if (!v41_tproj(m, st->sg, l->ffn_gate_shexp, E, FF, st->xn, n, 1) ||
             !v41_tproj(m, st->su, l->ffn_up_shexp, E, FF, st->xn, n, 1) ||
             !ds4_gpu_v41_swiglu_tensor(st->sh, st->sg, st->su, n, FF, DS4_SWIGLU_CLAMP_EXP) ||
             !v41_tproj(m, st->so, l->ffn_down_shexp, FF, E, st->sh, n, 1)) return false;
    /* y = routed(f32 累加的 bf16 专家输出) + shared(bf16) → .type_as(x) bf16 */
    if (tail) {   /* 解码: 归约 + 相加 + 舍 bf16 一发(2026-09-18 小核合并; 算式同序, 逐位同) */
        if (!ds4_gpu_v41_moe_tail_tensor(st->y, st->so, st->rw, n, DS4_N_EXPERT_USED, E)) return false;
    } else {
        if (!ds4_gpu_tensor_copy(st->y, 0, st->routed, 0, (uint64_t)n * E * 4)) return false;
        if (!ds4_gpu_v41_add_tensor(st->y, st->so, (uint64_t)n * E)) return false;
        if (!ds4_gpu_v41_round_bf16_tensor(st->y, (uint64_t)n * E)) return false;
    }
    /* 草稿塔: 反修/取料钩子不挂(塔是另一个模型); 只有草稿器蒸馏的塔件(--draft-amp <目录>/tower_Tn.bin, core_v41_draft.c 装进 st->ampA[塔号])
     * 在这里应用 y += xn·(B·A) → bf16 —— 没挂时 ampA 为 NULL, v41_amp_apply 一个核都不发, 与 2026-10-07 之前逐位同 */
    if (st->draft) return v41_amp_apply(st, il);
    /* 后训练前向存档: 放大器的梯度要本层 MoE 输入(此刻 xn 还是 ffn_norm 出口, 出口的 rms_norm 之后就被盖掉) */
    if (st->tsave && st->tsave->moe_in[il] && !ds4_gpu_tensor_copy(st->tsave->moe_in[il], 0, st->xn, 0, (uint64_t)n * E * 4)) return false;
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
    /* 后训练的 hc 边界检查: 在这一层入口给 hc 加一个方向扰动(推理时 tsave 恒 NULL, 不走这里) */
    if (!st->draft && st->tsave && st->tsave->hpert && st->tsave->hpert_layer == (int32_t)il && !st->tsave->hpert_mid &&
        !ds4_gpu_bwd_axpy_tensor(st->hc, st->tsave->hpert, st->tsave->hpert_eps, (uint64_t)n * DS4_N_HC * E)) return false;
    /* 后训练前向存档: 层入口的 hc 与 pre_mix(反传时从这里逐层重算本层, 不必存几十份中间量) */
    if (!st->draft && st->tsave && st->tsave->hc_in[il] &&
        (!ds4_gpu_bwd_pack_bf16_tensor(st->tsave->hc_in[il], st->hc, (uint64_t)n * DS4_N_HC * E) ||
         !ds4_gpu_tensor_copy(st->tsave->pm_in[il], 0, st->pre_mix, 0, (uint64_t)n * DS4_N_HC * 4))) return false;
    /* engram 层(官方 Transformer.forward: layer(h) 之前先 h = engram(h)); 草稿塔没有 engram */
    if (!st->draft && !st->no_engram && g_ds4_v41.engram_index_of[il] >= 0) {
        if (!v41_engram(e, st, il)) return false;
        if (st->tsave && st->tsave->ekv[il] && !ds4_gpu_bwd_pack_bf16_tensor(st->tsave->ekv[il], st->ekv, (uint64_t)n * (DS4_N_HC + 1u) * E)) return false;
    }
    /* DSpark: 目标层的注意力输入(= engram 之后、本层之前的 hc 四路均值)拼成 main_hidden。
     * 只搬本 chunk 末尾 mainh_cap 行 —— 草稿器只从"最后那几个已定 token"出发。 */
    if (!st->draft && st->mainh && g_ds4_v41.mtp_target_slot[il] >= 0) {
        const uint32_t rows = n < st->mainh_cap ? n : st->mainh_cap, src0 = n - rows;
        /* 落环: 第 t 行 → 位置 pos0+src0+t 的格; graph 路位置在设备槽(st->pos), 核里自算 */
        if (!ds4_gpu_v41_hc_mean_tensor(st->mainh, st->hc, E, DS4_N_HC, rows, src0,
                                        (uint32_t)g_ds4_v41.mtp_target_slot[il], g_ds4_v41.n_mtp_target,
                                        st->pos0 + src0, st->mainh_cap, st->graph ? st->pos : NULL)) return false;
        st->mainh_wrote = rows;
    }
    /* attn 半层: 本层 attn 的 mixes 产 attn_pre(给下面 ffn 用)/attn_post/attn_comb; 入口用上一层传来的 pre_mix */
    if (!v41_hc_half(e, st, l, true)) return false;
    if (!v41_attention(e, st, il)) return false;
    if (!st->draft && st->tsave && st->tsave->hc_in[il]) {   /* 后训练存档: 本层注意力用的压缩组号(倒着重算时它的来源层已经不在手边) */
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const int16_t src = ratio ? g_ds4_v41.kv_source_of[il] : -1;
        v41_tsave *sv = st->tsave;
        sv->ng[il] = src >= 0 ? st->ng_src[src] : 0u;
        sv->topk[il] = (src >= 0 && sv->ng[il]) ? st->idx_topk : 0u;
        sv->iratio[il] = sv->topk[il] ? st->idx_ratio : 0u;
        if (sv->topk[il] && (!sv->idx[il] || !ds4_gpu_tensor_copy(sv->idx[il], 0, st->idx, 0, (uint64_t)n * sv->topk[il] * 4))) return false;
    }
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, E, DS4_N_HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    if (!st->draft && st->tsave && st->tsave->hpert && st->tsave->hpert_layer == (int32_t)il && st->tsave->hpert_mid &&   /* hc 边界检查的中段扰动点 */
        !ds4_gpu_bwd_axpy_tensor(st->hc, st->tsave->hpert, st->tsave->hpert_eps, (uint64_t)n * DS4_N_HC * E)) return false;
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
                                                    0u, n, DS4_N_SWA, DS4_N_HEAD_DIM, 0, NULL)) ok = false;
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
                                              keep, st->snap_n, DS4_N_SWA, DS4_N_HEAD_DIM, 1, NULL)) return false;
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
    if (st->mainh) {   /* mainh 环: 被拒的 snap_n-keep 行作废(它们的格下一步会被真 token 盖掉, 现在只是不许被取) */
        const uint32_t dropped = st->snap_n - keep;
        st->mainh_n = st->mainh_n > dropped ? st->mainh_n - dropped : 0;
        st->mainh_end = (int64_t)np - 1;
    }
    return true;
}

/* 前向主体: embed → 各层 → 出口 head。直发(v41_forward)与解码整步 graph 的捕获(core_decode_graph.c)共用 ——
 * 里面没有任何主机等待/同步拷贝/位置推进, 那些各归调用方。CED 收工与钩子早停走 st->ced_done / st->stop_early 出去。 */
bool v41_forward_body(ds4_engine *e, ds4_v41_state *st) {
    const uint32_t n = st->n, E = DS4_N_EMBD;
    st->ced_done = 0;
    bool ok = v41_embed(&e->model, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, n, E) != 0;
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
            if (n >= 64u) fputc('\n', stderr);
            st->ced_done = 1;
            /* 分界层及其后的层这一块没写窗口环 ⇒ 它们的环里最早有效位置推进到下一块开头(注意力核按它钳窗口下界) */
            for (uint32_t l = ced_edge; l < DS4_N_LAYER; l++) st->win_from[l] = st->pos0 + n;
            return ok;   /* 中间块不出 logits(没人读: 生成只要最后一块的末位) */
        }
        ok = v41_layer(e, st, il);
        if (ok && (g_ds4_v41_prof || n >= 64u) && ds4_gpu_flush_commands() == 0) ok = false;   /* 解码不逐层同步(省 40 次停等), 查速度/大块才同步 */
        if (g_ds4_v41_prof) { const double tn = now_sec(); lt[il] = tn - tprev; if (ok) v41_nan_scan(st, il); tprev = now_sec(); }
        if (n >= 64u) fprintf(stderr, "[v41] pos %u+%u L%02u %s %.1fs\r", st->pos0, n, il, ok ? "ok" : "★失败★", now_sec() - t0);
        if (ok && st->stop_early) break;   /* 反修钩子取完料: 余下层与出口不算 */
    }
    if (n >= 64u) fputc('\n', stderr);
    if (!ok) { fprintf(stderr, "ds4: V4.1 前向失败(pos0 %u n %u)\n", st->pos0, n); return false; }
    if (st->stop_early) return true;   /* 位置照常推进(调用方做), 不出 logits */
    /* 出口: h = hc_pre(hc, pre_mix) → norm → head(f32 logits) */
    if (!ds4_gpu_v41_hc_pre_tensor(st->x, st->hc, st->pre_mix, E, DS4_N_HC, n)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, st->x, e->model.map, e->model.size, e->weights.output_norm->abs_offset, E, n, DS4_RMS_EPS)) return false;
    if (st->head_last_only && n > DS4_V41_GEMV_MAX_TOK) {
        /* ★生成路的预填块只算末位, 走解码同款 GEMV★(2026-09-29, 见 core_v41.h head_last_only): 整块走 GEMM 路 = 出口头 1.3 GB bf16 暂存 +
         * 2048 行 × 129280 列的白算, 而生成只读末位。★与整块 GEMM 路不逐字节同★: GEMM 路先把 q4_K 头权重舍成 bf16 再乘, GEMV 用 q4_K 原值
         * (与之后每一步解码同一条核路) —— 首个生成 token 的 logits 从"预填精度"变成"解码精度", 判据是文本/五指标, 不是 cmp。 */
        if (!ds4_gpu_tensor_copy(st->xlast, 0, st->xn, (uint64_t)(n - 1u) * E * 4, (uint64_t)E * 4)) return false;
        if (!v41_tproj(&e->model, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xlast, 1, 0)) return false;
        st->last_logit_row = 0;
    } else {
        if (n > st->logits_rows) { fprintf(stderr, "ds4: V4.1 出口要 %u 行 logits, 缓冲只开了 %u 行\n", n, st->logits_rows); return false; }
        if (!v41_tproj(&e->model, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xn, n, 0)) return false;
        st->last_logit_row = n - 1u;
    }
    if (g_ds4_v41_prof) {   /* 逐层毫秒: 一眼看出哪层在吃时间(engram 层 L1/L14, 源层 L2/8/14/20, 候选层 L20) */
        if (ds4_gpu_flush_commands() == 0) return false;
        lt[DS4_N_LAYER] = now_sec() - tprev;
        fprintf(stderr, "[v41-prof] pos %u+%u 总 %.1f ms | 层(ms):", st->pos0, n, (now_sec() - t0) * 1e3);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) fprintf(stderr, " %.1f", lt[il] * 1e3);
        fprintf(stderr, " | 出口 %.1f\n", lt[DS4_N_LAYER] * 1e3);
    }
    return true;
}

/* 一块预填: 单请求路(generate_argmax)与并发路(ds4_v41_req_prefill_step)共用同一份 —— 分块 / CED / 窗口尾的规则只写这一处。
 * ★最后一块至少留 window 个位置★(2026-09-21, bug.md §6.1): 官方 Decoder SWA Bounded Replay = 最后 n_win 个 token 跑满解码器;
 * 这里最后一块就是那段回放, 以前它 = 提示长 mod 512(1~512), 不足 128 时首批生成 token 的窗口直接缺位。余下不足 window 就从这一块匀过去。
 * ★判负存档(2026-09-29)★: 试过"最后一块只留正好 window 个"(CFO 14103 token 预填 15.1 → 13.8 s) —— 最后 window 位的解码器窗口被截到块头,
 * 隐态变差: 同一 CFO 请求投机接受 1.87 → 1.41/5 位、解码 43.5 → 38.3 t/s, 文本第 1 字就变。质量不为速度让步, 回退。
 * CED: 除最后一块外, 只跑编码器段 + 分界层 KV(最后一块同时充当官方说的"解码器有界回放")。 */
bool v41_prefill_chunk(ds4_engine *e, ds4_v41_state *st, const int32_t *prompt, uint32_t np, uint32_t *c0, uint32_t cap) {
    uint32_t nc = np - *c0 < cap ? np - *c0 : cap;
    const uint32_t rest = np - *c0 - nc;
    if (rest > 0u && rest < DS4_N_SWA && nc > DS4_N_SWA) nc -= DS4_N_SWA - rest;
    st->ced_skip = (!g_ds4_v41_decoder_full && *c0 + nc < np) ? 1 : 0;
    if (!v41_forward(e, st, prompt + *c0, nc)) return false;
    *c0 += nc;
    return true;
}

bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    if (n == 0 || n > st->cap_tok) { fprintf(stderr, "ds4: V4.1 前向块 %u 超 cap %u\n", n, st->cap_tok); return false; }
    if (st->n_past + n > st->ctx) { fprintf(stderr, "ds4: V4.1 上下文满(%u+%u > %u)\n", st->n_past, n, st->ctx); return false; }
    st->n = n; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->idx_ratio = 0; st->stop_early = 0;
    st->graph = 0;
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
    st->mainh_wrote = 0;
    if (ds4_gpu_begin_commands() == 0) return false;
    if (!v41_forward_body(e, st)) return false;
    if (ds4_gpu_end_commands() == 0 || ds4_gpu_synchronize() == 0) return false;
    if (st->mainh && st->mainh_wrote) {   /* mainh 环的账: 与上一段连续就接着数, 断了(CED 跳过的块)就从头数 */
        const int64_t first = (int64_t)st->pos0 + (int64_t)(n - st->mainh_wrote);
        const bool contig = st->mainh_n > 0 && st->mainh_end == first - 1;
        const uint32_t nn = contig ? st->mainh_n + st->mainh_wrote : st->mainh_wrote;
        st->mainh_n = nn > st->mainh_cap ? st->mainh_cap : nn;
        st->mainh_end = (int64_t)st->pos0 + (int64_t)n - 1;
    }
    st->n_past += n;
    if (n == 1u && !st->draft && !st->ced_done && !st->stop_early) st->n_direct1++;   /* 解码整步 graph 的"暖过了"计数 */
    /* 投机验证批(n=2..)同样要各暖一次直发(每个 n 的暂存/核属性各自懒建), 之后那个 n 才许捕获(core_decode_graph.c 批图) */
    if (n < DS4_MTP_MAX_BLOCK + 2u && !st->draft && !st->ced_done && !st->stop_early) st->n_direct_n[n]++;
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_forward_nonempty_tu;
