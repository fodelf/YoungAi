/* core_v41_draft.c — DSpark 草稿器(speed.md 段 6 D1, 2026-09-15): 一次前向出 block 个草稿位 + 每位的接受概率。
 *
 * 说人话: 解码慢是因为每出一个 token 就要把 6 GB 权重重读一遍。DSpark 是官方自带的解法 ——
 * 三个小塔(每塔 128 专家 top-3, 只有 SWA 128 的窗口注意力)一次猜出 5 个 token, 主模型一次验证 1+k 个,
 * 猜对几个就白赚几个 token 的时间。猜得对不对由**接受率**决定, 所以草稿的质量就是速度。
 *
 * 逐式对照官方 model.py: Transformer.forward_spec / DSparkBlock.forward_embed / forward_head。
 *   main_x       = main_norm(main_proj(主模型 L37/38/39 的注意力输入拼起来))   ← 块注意力的 KV 来源
 *   草稿块输入   = [上一个真 token, noise, noise, noise, noise] 的嵌入(noise id 由元数据给)
 *   三塔逐层     = 与普通层同构(core_v41_forward.c 的 v41_layer 带 draft 标志复用)
 *   逐位贪心     = head(out_norm(h)) 的第 i 行 + markov_head(第 i 位 token) 的偏置 → argmax → 第 i+1 位
 *   confidence   = proj([h_i ; markov_embed_i]) → 第 i 位的条件接受概率
 *
 * 出错会怎样: ①main_hidden 取成层输出而不是注意力输入 ⇒ 不报错, 接受率掉到 1 附近;
 * ②块的 kv 混进历史窗口 ⇒ 不报错, 接受率随会话慢慢烂掉(见 core_v41_attn.c 的注释);
 * ③这份 GGUF 没带 DSpark 运行参数 ⇒ ready=0, 调用方照常单 token 解码, 不停车。 */
#include "core_internal.h"
#include "src/common/ds4_quantfmt.h"   /* DS4_GGT_*: 张量在盘上的类型(f32 / bf16 / fp4x32) */
#ifndef DS4_NO_GPU

/* ★盘上是 f32 还是 bf16, 按 GGUF 登记的类型认, 不假设★
 * 实撞(2026-09-15): mtp 的 gate/markov/confidence 走转换器的 plan_small, 存的是 f32, 而主路同名矩阵
 * 走 plan_bf16 存 bf16。按 bf16 去读 f32 的字节不报错, 读出来是垃圾 —— 症状就是接受率 0.14/5。 */
static bool v41_small_matmul(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                             uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n) {
    if (w->type == DS4_GGT_BF16)
        return ds4_gpu_v41_matmul_bf16_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    if (w->type == DS4_GGT_F32)
        return ds4_gpu_v41_matmul_f32_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    fprintf(stderr, "ds4: [v41] 草稿器: %.*s 的类型 %u 不是 f32/bf16\n", (int)w->name.len, w->name.ptr, w->type);
    return false;
}

/* 草稿器的专家偏移表: [塔][3 个矩阵 × n_expert]。绑定时算一次, 之后每层前向直接递给核。
 * 为什么不在核里按名字找: 那是每层一次的字符串查找 + 128 次 range 解析, 纯白花。 */
static bool v41_draft_exp_off(ds4_engine *e, ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    for (uint32_t T = 0; T < v->mtp_towers; T++) {
        uint64_t *off = xmalloc((size_t)3u * v->mtp_experts * 8u);
        for (uint32_t i = 0; i < v->mtp_experts; i++) {
            const ds4_tensor *g = e->weights.mtp.exp_gate[T][i], *u = e->weights.mtp.exp_up[T][i], *d = e->weights.mtp.exp_down[T][i];
            if (!g || !u || !d) { free(off); fprintf(stderr, "ds4: [v41] 草稿塔 %u 专家 %u 张量缺失\n", T, i); return false; }
            off[i] = g->abs_offset; off[v->mtp_experts + i] = u->abs_offset; off[2u * v->mtp_experts + i] = d->abs_offset;
        }
        dr->exp_off[T] = off;
        dr->st.tower_exp_off[T] = off;
    }
    return true;
}

/* 挂草稿器对齐边车(mtp.md M6; gguf-tools/amp/dspark_align 的产物)。
 * 盘上: 16 B 头 {"DSPA", D, K} + A[K][D] f32 + B[K][D] f32。
 * ★只改草稿器★: 主模型一个字节不碰, 所以它**不可能**动五指标, 只动接受率 —— 这是它能独立发车的前提。
 * 出错会怎样: 文件在但 D 对不上 = 不是这个模型解的, 直接停车(挂上去只会让草稿变垃圾, 而且不报错)。 */
static bool v41_draft_amp_load(ds4_v41_draft *dr, const char *path) {
    struct { char magic[4]; uint32_t d, k, rsv; } h;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ds4: [v41] --draft-amp 打不开 %s\n", path); return false; }
    bool ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "DSPA", 4);
    if (ok && h.d != DS4_N_EMBD) {
        fprintf(stderr, "ds4: [v41] --draft-amp 维度 %u ≠ %u, 不是这个模型解的\n", h.d, (unsigned)DS4_N_EMBD);
        ok = false;
    }
    if (!ok) { fclose(f); fprintf(stderr, "ds4: [v41] --draft-amp %s 不是对齐边车\n", path); return false; }
    const uint64_t nb = (uint64_t)h.k * h.d * 4;
    float *buf = xmalloc((size_t)nb);
    bool alloc_ok = true;
    dr->ampA = v41_alloc(nb, &alloc_ok);
    dr->ampB = v41_alloc(nb, &alloc_ok);
    dr->ampT = v41_alloc((uint64_t)(dr->st.cap_tok ? dr->st.cap_tok : 8u) * h.k * 4, &alloc_ok);
    ok = alloc_ok && fread(buf, 1, (size_t)nb, f) == nb;
    if (ok && g_ds4_v41_draft_amp_scale != 1.0f)
        for (uint64_t i = 0; i < nb / 4; i++) buf[i] *= g_ds4_v41_draft_amp_scale;
    if (ok) ok = ds4_gpu_tensor_write(dr->ampA, 0, buf, nb) != 0;
    if (ok) ok = fread(buf, 1, (size_t)nb, f) == nb && ds4_gpu_tensor_write(dr->ampB, 0, buf, nb);
    free(buf); fclose(f);
    if (!ok) { fprintf(stderr, "ds4: [v41] --draft-amp 读取失败\n"); return false; }
    dr->ampK = h.k;
    fprintf(stderr, "ds4: [v41] 草稿器对齐边车已挂: K=%u D=%u β=%.3f (%s)\n", h.k, h.d, (double)g_ds4_v41_draft_amp_scale, path);
    return true;
}

/* 草稿器一次前向要读多少字节(2026-09-16)。
 *
 * 为什么要这张表: 投机赢不赢是纯字节账 —— 一轮读的总字节 ÷ 一轮产出的 token, 要小于纯解码的
 * 6.0 GB/token。官方那边主模型 FP8 一个 token 几十 GB, 草稿器几 GB 可以忽略不计; 我们把主模型
 * 压到 1.5 bit 专家 + 4.25 bit 骨架 = 6.0 GB, **草稿器的相对分量就翻上来了**。实测一轮草稿 39.7 ms,
 * 按解码路的有效带宽折算是 4.5 GB —— 跟主模型一个 token 一样贵。这张表就是查这 4.5 GB 落在哪。
 *
 * 怎么读: 密集部分(注意力/共享专家/main_proj)是每步必读的固定成本, 激活专家按 top-k 折算,
 * 出口头是主模型的那一份(草稿器借用, 所以严格说不算草稿器额外付的钱, 单列)。
 * 出错会怎样: 表里"密集"一项如果和主模型一层的量级相当, 说明三塔根本不小, 投机从根上不成立。 */
static void v41_draft_byte_report(ds4_engine *e, const ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_weights *w = &e->weights;
    uint64_t dense = 0, exp_all = 0;
    for (uint32_t T = 0; T < v->mtp_towers; T++) {
        const ds4_layer_weights *t = &w->mtp.tower[T];
        const ds4_tensor *d[] = { t->hc_attn_fn, t->hc_attn_scale, t->hc_attn_base, t->attn_norm, t->attn_q_a,
            t->attn_q_a_norm, t->attn_q_b, t->attn_kv, t->attn_kv_a_norm, t->attn_sinks, t->attn_output_a,
            t->attn_output_b, t->hc_ffn_fn, t->hc_ffn_scale, t->hc_ffn_base, t->ffn_norm, t->ffn_gate_inp,
            t->ffn_exp_probs_b, t->ffn_gate_shexp, t->ffn_up_shexp, t->ffn_down_shexp };
        for (size_t i = 0; i < sizeof(d) / sizeof(d[0]); i++) if (d[i]) dense += d[i]->bytes;
        for (uint32_t x = 0; x < v->mtp_experts; x++) {
            const ds4_tensor *g = w->mtp.exp_gate[T][x], *u = w->mtp.exp_up[T][x], *dn = w->mtp.exp_down[T][x];
            if (g) exp_all += g->bytes; if (u) exp_all += u->bytes; if (dn) exp_all += dn->bytes;
        }
    }
    /* 激活专家: 每塔 top-k, 但一块 B 位各自选各自的 —— 最坏情况 B×k 份互不相同 */
    const double per_exp = v->mtp_experts ? (double)exp_all / (double)(v->mtp_towers * v->mtp_experts) : 0.0;
    const double act = per_exp * v->mtp_used * v->mtp_towers * dr->block;
    const uint64_t head = (w->output ? w->output->bytes : 0) + (w->mtp.main_proj ? w->mtp.main_proj->bytes : 0);
    const double G = 1024.0 * 1024.0 * 1024.0;
    fprintf(stderr, "ds4: [v41] 草稿器字节账: 密集(三塔) %.2f GB + 激活专家(最坏 %u×%u×%u 份) %.2f GB"
                    " + 出口头/main_proj %.2f GB = **%.2f GB/块**  [专家全量 %.2f GB]\n",
            (double)dense / G, v->mtp_towers, v->mtp_used, dr->block, act / G, (double)head / G,
            ((double)dense + act + (double)head) / G, (double)exp_all / G);
}

bool v41_draft_alloc(ds4_engine *e, ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    memset(dr, 0, sizeof *dr);
    if (!v->mtp_towers || !v->mtp_block || !v->n_mtp_target) return false;   /* 元数据不全 = 投机路不武装 */
    if (!e->weights.mtp.main_proj || !e->weights.mtp.markov_embd || !e->weights.mtp.confidence) return false;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint32_t FF = DS4_N_FF_EXP, R = v->mtp_markov_rank;
    const uint32_t B = v->mtp_block, cap = B + 1u;   /* +1: 验证批最多 1+block 行, main_x 也按它分配 */
    const uint64_t low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, mix = 2u * HC + HC * HC;
    ds4_v41_state *st = &dr->st;
    st->draft = 1; st->cap_tok = cap; st->ctx = cap; st->idx_owner = -1; st->cand_owner = -1;
    bool ok = true;
    st->tok = v41_alloc(cap * 4, &ok);          st->pos = v41_alloc((uint64_t)(cap > SWA ? cap : SWA) * 4, &ok);
    st->hc = v41_alloc((uint64_t)cap * HC * E * 4, &ok);   st->hc2 = v41_alloc((uint64_t)cap * HC * E * 4, &ok);
    st->mix = v41_alloc((uint64_t)cap * mix * 4, &ok);      st->pre = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->post = v41_alloc((uint64_t)cap * HC * 4, &ok);      st->comb = v41_alloc((uint64_t)cap * HC * HC * 4, &ok);
    st->pre_mix = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->x = v41_alloc((uint64_t)cap * E * 4, &ok);          st->xn = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->qr = v41_alloc((uint64_t)cap * Q * 4, &ok);         st->qrn = v41_alloc((uint64_t)cap * Q * 4, &ok);
    st->q = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);
    /* kv/kvn/main_x 按 SWA 行分配: 预填后要一次把 128 个已确认位置的 main_kv 推进窗口 */
    const uint32_t pcap = cap > SWA ? cap : SWA;
    st->kv = v41_alloc((uint64_t)pcap * HD * 4, &ok);       st->kvn = v41_alloc((uint64_t)pcap * HD * 4, &ok);
    st->wintmp = v41_alloc((uint64_t)SWA * HD * 4, &ok);
    st->o = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);    st->low = v41_alloc((uint64_t)cap * low * 4, &ok);
    st->attn_out = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->glog = v41_alloc((uint64_t)cap * v->mtp_experts * 4, &ok);
    st->sel = v41_alloc((uint64_t)cap * v->mtp_used * 4, &ok);
    st->rw = v41_alloc((uint64_t)cap * v->mtp_used * 4, &ok);
    st->routed = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->sg = v41_alloc((uint64_t)cap * FF * 4, &ok);        st->su = v41_alloc((uint64_t)cap * FF * 4, &ok);
    st->sh = v41_alloc((uint64_t)cap * FF * 4, &ok);        st->so = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->y = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->logits = v41_alloc((uint64_t)cap * DS4_N_VOCAB * 4, &ok);
    st->main_x = v41_alloc((uint64_t)pcap * E * 4, &ok);
    for (uint32_t T = 0; T < v->mtp_towers; T++) st->win[T] = v41_alloc((uint64_t)(SWA + cap) * HD * 4, &ok);
    dr->mainx_raw = v41_alloc((uint64_t)pcap * E * 4, &ok);
    dr->mk_embed = v41_alloc((uint64_t)cap * R * 4, &ok);
    dr->mk_cur = v41_alloc((uint64_t)R * 4, &ok);
    dr->mk_bias = v41_alloc((uint64_t)DS4_N_VOCAB * 4, &ok);
    dr->conf_in = v41_alloc((uint64_t)cap * (E + R) * 4, &ok);
    dr->conf = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids_next = v41_alloc(16, &ok);
    dr->h = v41_alloc((uint64_t)cap * E * 4, &ok);
    if (!ok || !v41_draft_exp_off(e, dr)) { v41_draft_free(dr); return false; }
    dr->block = B;
    if (g_ds4_v41_draft_amp && !v41_draft_amp_load(dr, g_ds4_v41_draft_amp)) { v41_draft_free(dr); return false; }
    dr->ready = 1;
    fprintf(stderr, "ds4: [v41] DSpark 草稿器已武装: %u 塔 × %u 专家 top-%u, 一块 %u 位, 目标层",
            v->mtp_towers, v->mtp_experts, v->mtp_used, B);
    for (uint32_t i = 0; i < v->n_mtp_target; i++) fprintf(stderr, " L%02d", (int)v->mtp_target[i]);
    fprintf(stderr, "\n");
    v41_draft_byte_report(e, dr);
    return true;
}

void v41_draft_free(ds4_v41_draft *dr) {
    ds4_v41_state *st = &dr->st;
    ds4_gpu_tensor **all[] = { &st->tok, &st->pos, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix,
        &st->x, &st->xn, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->wintmp, &st->o, &st->low, &st->attn_out,
        &st->glog, &st->sel, &st->rw, &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->main_x,
        &dr->mainx_raw, &dr->mk_embed, &dr->mk_cur, &dr->mk_bias, &dr->conf_in, &dr->conf, &dr->ids, &dr->ids_next, &dr->h,
        &dr->ampA, &dr->ampB, &dr->ampT };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { if (*all[i]) ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t T = 0; T < DS4_MTP_MAX_TOWERS; T++) {
        if (st->win[T]) { ds4_gpu_tensor_free(st->win[T]); st->win[T] = NULL; }
        free(dr->exp_off[T]); dr->exp_off[T] = NULL; st->tower_exp_off[T] = NULL;
    }
    dr->ready = 0;
}

/* main_x = main_norm(main_proj(main_hidden 的 rows 行)) —— 官方 DSparkBlock.forward_embed 的前两步。 */
static bool v41_draft_main_x(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint64_t in = (uint64_t)DS4_N_EMBD * g_ds4_v41.n_mtp_target;
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(dr->mainx_raw, m->map, m->size, e->weights.mtp.main_proj->abs_offset,
                                          in, DS4_N_EMBD, main_st->mainh, rows, 1)) return false;
    return ds4_gpu_v41_rms_norm_tensor(dr->st.main_x, dr->mainx_raw, m->map, m->size,
                                       e->weights.mtp.main_norm->abs_offset, DS4_N_EMBD, rows, DS4_RMS_EPS) != 0;
}

/* 草稿块的一次前向: 三塔 → 出口 → 逐位 markov 贪心 → confidence。ids[0] 由调用方写好(= 上一个真 token)。 */
static bool v41_draft_block(ds4_engine *e, ds4_v41_draft *dr, uint32_t pos0) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_model *m = &e->model;
    ds4_v41_state *st = &dr->st;
    const uint32_t B = dr->block, E = DS4_N_EMBD, R = v->mtp_markov_rank;
    st->n = B; st->pos0 = pos0;
    {   /* 块输入 = [真 token, noise × (B-1)](官方 draft_input_ids), 位置 pos0..pos0+B-1 */
        int32_t tk[DS4_MTP_MAX_BLOCK], ps[DS4_MTP_MAX_BLOCK];
        for (uint32_t i = 0; i < B; i++) { tk[i] = i ? (int32_t)v->mtp_noise_id : dr->host_ids[0]; ps[i] = (int32_t)(pos0 + i); }
        if (!ds4_gpu_tensor_write(st->tok, 0, tk, (uint64_t)B * 4) || !ds4_gpu_tensor_write(st->pos, 0, ps, (uint64_t)B * 4)) return false;
    }
    {
        float *pm = xmalloc((size_t)B * DS4_N_HC * 4);
        for (uint32_t i = 0; i < B * DS4_N_HC; i++) pm[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
        const bool okpm = ds4_gpu_tensor_write(st->pre_mix, 0, pm, (uint64_t)B * DS4_N_HC * 4) != 0;
        free(pm);
        if (!okpm) return false;
    }
    if (!ds4_gpu_v41_embed_fp4x32_tensor(st->x, st->tok, m->map, m->size, e->weights.token_embd->abs_offset, DS4_N_VOCAB, B, E)) return false;
    if (!ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, DS4_N_HC, B)) return false;
    for (uint32_t T = 0; T < v->mtp_towers; T++) if (!v41_layer(e, st, T)) return false;
    /* 出口: 官方 mtp[-1] 借主模型的 head, norm 用 mtp.2.norm */
    if (!ds4_gpu_v41_hc_pre_tensor(dr->h, st->hc, st->pre_mix, E, DS4_N_HC, B)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, dr->h, m->map, m->size, e->weights.mtp.out_norm->abs_offset, E, B, DS4_RMS_EPS)) return false;
    /* ★对齐修正(mtp.md M6)★: xn += xn·(Bᵀ·A) —— 把草稿器喂给出口头的隐态掰到主模型喂给**同一个头**的那个。
     * 位置就在这里: norm 之后、head 之前, 与取料时 X 的取点逐字对应(取错点解出来的映射就是错的, 且不报错)。
     * 没挂边车时 ampK=0, 这一行整条跳过, 输出与挂之前逐位相同。 */
    if (dr->ampK) {
        if (!ds4_gpu_v41_amp_apply_tensor(st->xn, st->xn, dr->ampA, dr->ampB, dr->ampT, B, E, dr->ampK)) return false;
        /* ★补回 bf16 格点★: 出口头那个 GEMV 核**假定进来的激活已经在 bf16 格点上**(它为此省掉了
         * 内层的舍入, 见 cuda_v41_4.inc.cu 的注释)。修正是 f32 加出来的, 不补这一下就破了那个前提 ——
         * 不报错, 只差一个舍入位, 但那条不变量一旦破了后面没人再守。 */
        if (!ds4_gpu_v41_round_bf16_tensor(st->xn, (uint64_t)B * E)) return false;
    }
    if (!ds4_gpu_v41_matmul_fp4x32_tensor(st->logits, m->map, m->size, e->weights.output->abs_offset, E, DS4_N_VOCAB, st->xn, B, 0)) return false;
    /* 逐位: logits[i] += markov_head(第 i 位 token) → argmax → 第 i+1 位。全程在设备上,
     * 每位一次 D2H 就是每轮 5 次停等 —— 投机省下来的时间还不够付。 */
    const uint64_t nrow = e->weights.mtp.markov_embd->ndim > 1 ? e->weights.mtp.markov_embd->dim[1] : DS4_N_VOCAB;
    const uint32_t eb = e->weights.mtp.markov_embd->type == DS4_GGT_BF16 ? 2u : 4u;
    for (uint32_t i = 0; i < B; i++) {
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_embed, m->map, m->size, e->weights.mtp.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, i)) return false;
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_cur, m->map, m->size, e->weights.mtp.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, 0)) return false;
        if (!v41_small_matmul(m, dr->mk_bias, e->weights.mtp.markov_head, R, DS4_N_VOCAB, dr->mk_cur, 1)) return false;
        if (!ds4_gpu_v41_row_add_tensor(st->logits, i, dr->mk_bias, DS4_N_VOCAB)) return false;
        /* argmax 只会写自己那块的第 0 个 int, 所以先落 ids_next 再拷到 ids[i+1](官方 output_ids[:, i+1]) */
        if (!ds4_gpu_v41_argmax_tensor(dr->ids_next, st->logits, i, DS4_N_VOCAB)) return false;
        if (!ds4_gpu_tensor_copy(dr->ids, (uint64_t)(i + 1u) * 4, dr->ids_next, 0, 4)) return false;
    }
    /* confidence = proj([h_i ; markov_embed_i]) */
    for (uint32_t i = 0; i < B; i++) {
        if (!ds4_gpu_tensor_copy(dr->conf_in, (uint64_t)i * (E + R) * 4, dr->h, (uint64_t)i * E * 4, (uint64_t)E * 4)) return false;
        if (!ds4_gpu_tensor_copy(dr->conf_in, ((uint64_t)i * (E + R) + E) * 4, dr->mk_embed, (uint64_t)i * R * 4, (uint64_t)R * 4)) return false;
    }
    return v41_small_matmul(m, dr->conf, e->weights.mtp.confidence, (uint64_t)E + R, 1, dr->conf_in, B);
}

bool v41_draft_step(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main, uint32_t rows) {
    if (!dr->ready || !main_st->mainh || !main_st->mainh_rows) return false;
    if (rows > main_st->mainh_rows) rows = main_st->mainh_rows;
    if (!rows) return false;
    {   /* 这 rows 个已确认位置的绝对位置(rope 要) */
        int32_t *ps = xmalloc((size_t)rows * 4);
        for (uint32_t i = 0; i < rows; i++) ps[i] = (int32_t)(pos_main + 1u - rows + i);
        const bool okp = ds4_gpu_tensor_write(dr->st.pos, 0, ps, (uint64_t)rows * 4) != 0;
        free(ps);
        if (!okp) return false;
    }
    if (!v41_draft_main_x(e, main_st, dr, rows)) return false;
    if (!v41_draft_push_main(e, &dr->st, rows)) return false;
    /* 块注意力的 main_x 只用最后一行(官方 forward_spec 的 main_hidden 是当前这一位) */
    if (rows > 1 && !ds4_gpu_tensor_copy(dr->st.main_x, 0, dr->st.main_x, (uint64_t)(rows - 1u) * DS4_N_EMBD * 4,
                                         (uint64_t)DS4_N_EMBD * 4)) return false;
    dr->host_ids[0] = tok;
    if (!ds4_gpu_tensor_write(dr->ids, 0, &tok, 4)) return false;
    if (!v41_draft_block(e, dr, pos_main + 1u)) return false;
    if (!ds4_gpu_synchronize()) return false;
    if (!ds4_gpu_tensor_read(dr->ids, 0, dr->host_ids, (uint64_t)(dr->block + 1u) * 4)) return false;
    if (!ds4_gpu_tensor_read(dr->conf, 0, dr->host_conf, (uint64_t)dr->block * 4)) return false;
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_draft_nonempty_tu;
