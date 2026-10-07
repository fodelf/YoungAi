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
        /* 盘上是 VQ blob 的塔(100 GB 配方): 不建偏移表, 把 blob 递给状态, MoE 走主干的融合核(core_v41_forward.c v41_moe) */
        if (e->weights.mtp.exps_vq[T]) {
            dr->st.tower_exps_vq[T] = e->weights.mtp.exps_vq[T]; dr->exp_off[T] = NULL; dr->st.tower_exp_off[T] = NULL;
            continue;
        }
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
        if (w->mtp.exps_vq[T]) { exp_all += w->mtp.exps_vq[T]->bytes; continue; }   /* VQ blob 形态: 一塔一张 */
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
    /* 三塔专家两种盘上形态都能武装(2026-09-20): 逐专家 fp4x32 走 cuda_v41_draft.inc.cu 的 dense MoE 核, VQ blob 走主干的
     * 解码即乘核(v41_draft_exp_off 按形态分流)。09-19~20 之间 blob 形态曾被这里拦成"不武装"(当时草稿器只有逐专家那条核)。 */
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
    /* ★窗口后面的暂存区要装得下一次补进来的最多 SWA 行★(2026-09-18 实撞): 以前只留 block+1 行, 而预填后第一轮草稿要把
     * 提示末尾最多 128 个位置的 main_kv 一次推进窗口 —— 越界的拷贝静默失败, 第一轮草稿从来没出过, 窗口里也从来没有提示的上下文;
     * 老口径下后面每轮只推 1 行所以还能跑, 新口径(按位置差补窗口)则每轮都要补整段, 于是整条投机路静默失效。 */
    for (uint32_t T = 0; T < v->mtp_towers; T++) st->win[T] = v41_alloc((uint64_t)(SWA + pcap) * HD * 4, &ok);
    dr->mainx_raw = v41_alloc((uint64_t)pcap * E * 4, &ok);
    dr->mk_embed = v41_alloc((uint64_t)cap * R * 4, &ok);
    dr->mk_cur = v41_alloc((uint64_t)R * 4, &ok);
    dr->mk_bias = v41_alloc((uint64_t)DS4_N_VOCAB * 4, &ok);
    /* markov 偏置缓存: 槽 id 全 −1(空)、轮换计数 0; 只在原件 bf16 markov_head 那条路用(蒸馏的偏置表 mkH_dev 不缓存) */
    dr->mk_cache = v41_alloc((uint64_t)DS4_V41_MKCACHE_SLOTS * DS4_N_VOCAB * 4, &ok);
    dr->mk_cache_ids = v41_alloc((uint64_t)DS4_V41_MKCACHE_SLOTS * 4, &ok);
    dr->mk_cache_next = v41_alloc(16, &ok);
    dr->mk_hit = v41_alloc(16, &ok);
    if (ok) {
        int32_t neg[DS4_V41_MKCACHE_SLOTS]; for (uint32_t i = 0; i < DS4_V41_MKCACHE_SLOTS; i++) neg[i] = -1;
        const uint32_t zero4[4] = { 0u, 0u, 0u, 0u };
        if (!ds4_gpu_tensor_write(dr->mk_cache_ids, 0, neg, sizeof neg) || !ds4_gpu_tensor_write(dr->mk_cache_next, 0, zero4, 16)) ok = false;
    }
    dr->conf_in = v41_alloc((uint64_t)cap * (E + R) * 4, &ok);
    dr->conf = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids_next = v41_alloc(16, &ok);
    dr->h = v41_alloc((uint64_t)cap * E * 4, &ok);
    dr->mainh_lin = v41_alloc((uint64_t)SWA * v->n_mtp_target * E * 4, &ok);
    dr->win_end = -1;
    /* ★草稿图的槽★(2026-09-22): pinned(零拷贝小核直接读/写, 图每次重放读那一刻槽里的值) + 一个设备 int(ring_rows 的起始行) */
    dr->p_tok = ds4_gpu_host_alloc((uint64_t)cap * 4);  dr->p_bpos = ds4_gpu_host_alloc((uint64_t)cap * 4);
    /* p_wpos 按窗宽分: 预填后第一轮补窗口一次最多 SWA 行(v41_draft_step 钳的), 不是图分档上限 DS4_V41_DRAFT_GROWS ——
     * 09-22 按 8 分配, 第一轮就越界写主机槽, 随后 512 B 的 cudaMemcpy 读 32 B 的锁页区报 invalid argument(09-24 实撞)。 */
    dr->p_wpos = ds4_gpu_host_alloc((uint64_t)SWA * 4);  dr->p_first = ds4_gpu_host_alloc(4);
    dr->p_ids = ds4_gpu_host_alloc((uint64_t)cap * 4);  dr->p_conf = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_onehot = ds4_gpu_host_alloc((uint64_t)cap * DS4_N_HC * 4);
    dr->firstd = v41_alloc(16, &ok);
    if (!dr->p_tok || !dr->p_bpos || !dr->p_wpos || !dr->p_first || !dr->p_ids || !dr->p_conf || !dr->p_onehot) ok = false;
    if (ok) for (uint32_t i = 0; i < cap * DS4_N_HC; i++) dr->p_onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    if (!ok || !v41_draft_exp_off(e, dr)) { v41_draft_free(dr); return false; }
    /* --dspark-block: 诊断时钉死块长(不超过元数据给的 B, 缓冲是按 B 分配的) */
    dr->block = (g_ds4_v41_block && g_ds4_v41_block <= B) ? g_ds4_v41_block : B;
    if (g_ds4_v41_draft_amp && !v41_draft_amp_mount(dr, g_ds4_v41_draft_amp)) { v41_draft_free(dr); return false; }   /* 文件 = 出口对齐边车; 目录 = 蒸馏的件(core_v41_draft_amp.c) */
    dr->ready = 1;
    fprintf(stderr, "ds4: [v41] DSpark 草稿器已武装: %u 塔 × %u 专家 top-%u(%s), 一块 %u 位, 目标层",
            v->mtp_towers, v->mtp_experts, v->mtp_used, e->weights.mtp.exps_vq[0] ? "VQ blob" : "逐专家 fp4x32", B);
    for (uint32_t i = 0; i < v->n_mtp_target; i++) fprintf(stderr, " L%02d", (int)v->mtp_target[i]);
    fprintf(stderr, "\n");
    v41_draft_byte_report(e, dr);
    return true;
}

void v41_draft_free(ds4_v41_draft *dr) {
    ds4_v41_state *st = &dr->st;
    if (dr->gsteps || dr->gcaps) fprintf(stderr, "ds4: [graph] 草稿一轮走图 %u 次, 捕获 %u 次\n", dr->gsteps, dr->gcaps);
    for (uint32_t r = 0; r <= DS4_V41_DRAFT_GROWS; r++) { if (dr->gexec[r]) ds4_gpu_decode_graph_free(dr->gexec[r]); dr->gexec[r] = NULL; }
    ds4_gpu_host_free(dr->p_tok); ds4_gpu_host_free(dr->p_bpos); ds4_gpu_host_free(dr->p_wpos); ds4_gpu_host_free(dr->p_first);
    ds4_gpu_host_free(dr->p_ids); ds4_gpu_host_free(dr->p_conf); ds4_gpu_host_free(dr->p_onehot);
    dr->p_tok = dr->p_bpos = dr->p_wpos = dr->p_first = dr->p_ids = NULL; dr->p_conf = dr->p_onehot = NULL;
    ds4_gpu_tensor **all[] = { &st->tok, &st->pos, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix,
        &st->x, &st->xn, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->wintmp, &st->o, &st->low, &st->attn_out,
        &st->glog, &st->sel, &st->rw, &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->main_x,
        &dr->mainx_raw, &dr->mk_embed, &dr->mk_cur, &dr->mk_bias, &dr->conf_in, &dr->conf, &dr->ids, &dr->ids_next, &dr->h,
        &dr->ampA, &dr->ampB, &dr->ampT, &dr->mainh_lin, &dr->firstd, &dr->mkE_dev, &dr->mkH_dev,
        &dr->mk_cache, &dr->mk_cache_ids, &dr->mk_cache_next, &dr->mk_hit };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { if (*all[i]) ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t T = 0; T < DS4_MTP_MAX_TOWERS; T++) {
        if (st->win[T]) { ds4_gpu_tensor_free(st->win[T]); st->win[T] = NULL; }
        free(dr->exp_off[T]); dr->exp_off[T] = NULL; st->tower_exp_off[T] = NULL; st->tower_exps_vq[T] = NULL;
    }
    v41_amp_free(st);   /* 塔件(ampA/ampB/ampT)住草稿态, 这里一起放; 没挂时全 NULL */
    dr->ready = 0;
}

/* main_x = main_norm(main_proj(dr->mainh_lin 的 rows 行)) —— 官方 DSparkBlock.forward_embed 的前两步。 */
static bool v41_draft_main_x(ds4_engine *e, ds4_v41_draft *dr, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint64_t in = (uint64_t)DS4_N_EMBD * g_ds4_v41.n_mtp_target;
    /* main_proj 盘上可能是 FP8(原件精度)/ fp4x32(2026-09-17 之前的老 GGUF)/ q4_K(骨架配方跟着换) —— 一律交给
     * 按登记类型分发的 v41_tproj, 这里不再自己二选一(09-20 wo_b 那处漏改的教训: 少一支不报错, 只出假数)。 */
    if (!v41_tproj(m, dr->mainx_raw, e->weights.mtp.main_proj, in, DS4_N_EMBD, dr->mainh_lin, rows, 1)) return false;
    return ds4_gpu_v41_rms_norm_tensor(dr->st.main_x, dr->mainx_raw, m->map, m->size,
                                       e->weights.mtp.main_norm->abs_offset, DS4_N_EMBD, rows, DS4_RMS_EPS) != 0;
}

/* 主机 → 设备张量: 直发时同步拷; 捕获时改零拷贝小核(源必须是 pinned 槽 —— 图每次重放读那一刻槽里的值, 所以主机每轮发图前先把槽填好) */
static bool v41_draft_put(const ds4_v41_draft *dr, ds4_gpu_tensor *t, uint64_t off, const void *pinned, uint64_t bytes) {
    return dr->cap_mode ? ds4_gpu_tensor_write_zerocopy(t, off, pinned, bytes) != 0 : ds4_gpu_tensor_write(t, off, pinned, bytes) != 0;
}

/* 本轮的主机槽: 补窗口的位置/环起始行 + 块输入 [真 token, noise × (B-1)](官方 draft_input_ids) 与块位置 pos0..pos0+B-1。
 * ★直发、捕获、重放三条路都必须先调它★: 草稿图只烤了槽的地址, 值是重放那一刻从槽里读的。09-22 版只在直发/捕获那趟
 * 顺手填(写在 fill/block 里), 重放时位置和环起始行全是捕获那一刻的旧值 ⇒ rope 用旧位置、从 main_hidden 环取旧行,
 * 窗口一路写坏, 在线接受率 40 轮后从 0.47 掉到 0.03, 不报错(09-24 实撞)。 */
static void v41_draft_slots(const ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    for (uint32_t i = 0; i < rows; i++) dr->p_wpos[i] = (int32_t)(first + (int64_t)i);   /* 这 rows 个位置的绝对位置(rope 要) */
    if (rows) dr->p_first[0] = (int32_t)(first % (int64_t)main_st->mainh_cap);
    for (uint32_t i = 0; i < dr->block; i++) {
        dr->p_tok[i] = i ? (int32_t)g_ds4_v41.mtp_noise_id : dr->host_ids[0];
        dr->p_bpos[i] = (int32_t)(pos0 + i);
    }
    __sync_synchronize();
}

/* 草稿块的一次前向: 三塔 → 出口 → 逐位 markov 贪心 → confidence。槽由 v41_draft_slots 填好(ids[0] = 上一个真 token)。 */
static bool v41_draft_block(ds4_engine *e, ds4_v41_draft *dr, uint32_t pos0) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_model *m = &e->model;
    ds4_v41_state *st = &dr->st;
    const uint32_t B = dr->block, E = DS4_N_EMBD, R = v->mtp_markov_rank;
    st->n = B; st->pos0 = pos0;
    if (!v41_draft_put(dr, st->tok, 0, dr->p_tok, (uint64_t)B * 4) || !v41_draft_put(dr, st->pos, 0, dr->p_bpos, (uint64_t)B * 4)) return false;
    if (!v41_draft_put(dr, st->pre_mix, 0, dr->p_onehot, (uint64_t)B * DS4_N_HC * 4)) return false;
    if (!v41_embed(m, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, B, E)) return false;
    if (!ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, DS4_N_HC, B)) return false;
    for (uint32_t T = 0; T < v->mtp_towers; T++) if (!v41_layer(e, st, T)) return false;
    /* 出口: 官方 mtp[-1] 借主模型的 head, norm 用 mtp.2.norm */
    if (!ds4_gpu_v41_hc_pre_tensor(dr->h, st->hc, st->pre_mix, E, DS4_N_HC, B)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, dr->h, m->map, m->size, e->weights.mtp.out_norm->abs_offset, E, B, DS4_RMS_EPS)) return false;
    /* ★对齐修正(mtp.md M6)★: xn += xn·(Bᵀ·A) —— 把草稿器喂给出口头的隐态掰到主模型喂给**同一个头**的那个。
     * 位置就在这里: norm 之后、head 之前, 与取料时 X 的取点逐字对应(取错点解出来的映射就是错的, 且不报错)。
     * 没挂边车时 ampK=0, 这一行整条跳过, 输出与挂之前逐位相同。 */
    if (dr->ampK) {
        if (!ds4_gpu_draft_amp_apply_tensor(st->xn, st->xn, dr->ampA, dr->ampB, dr->ampT, B, E, dr->ampK)) return false;   /* 小批两发小核, 见 cuda_draft_attn */
        /* ★补回 bf16 格点★: 出口头那个 GEMV 核**假定进来的激活已经在 bf16 格点上**(它为此省掉了
         * 内层的舍入, 见 cuda_v41_4.inc.cu 的注释)。修正是 f32 加出来的, 不补这一下就破了那个前提 ——
         * 不报错, 只差一个舍入位, 但那条不变量一旦破了后面没人再守。 */
        if (!ds4_gpu_v41_round_bf16_tensor(st->xn, (uint64_t)B * E)) return false;
    }
    if (!v41_tproj(m, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xn, B, 0)) return false;
    /* 逐位: logits[i] += markov_head(第 i 位 token) → argmax → 第 i+1 位。全程在设备上,
     * 每位一次 D2H 就是每轮 5 次停等 —— 投机省下来的时间还不够付。 */
    const uint64_t nrow = e->weights.mtp.markov_embd->ndim > 1 ? e->weights.mtp.markov_embd->dim[1] : DS4_N_VOCAB;
    const uint32_t eb = e->weights.mtp.markov_embd->type == DS4_GGT_BF16 ? 2u : 4u;
    /* 偏置缓存(ds4_gpu_v41.h): 原件 bf16 markov_head 那条路才缓存; 命中 ⇒ GEMV 整网格直接退, 偏置从缓存槽加 */
    const bool mkc = dr->mk_cache && !dr->mkH_dev && e->weights.mtp.markov_head->type == DS4_GGT_BF16;
    for (uint32_t i = 0; i < B; i++) {
        if (mkc && !ds4_gpu_v41_mkcache_lookup_tensor(dr->mk_hit, dr->mk_cache_ids, dr->mk_cache_next, dr->ids, i, DS4_V41_MKCACHE_SLOTS)) return false;
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_embed, m->map, m->size, e->weights.mtp.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, i)) return false;   /* confidence 头吃的仍是原件的行 */
        /* 偏置用的行与头: 挂了蒸馏的偏置表(core_v41_draft_amp.c)就走设备表, 否则原件 */
        if (dr->mkE_dev) {
            if (!ds4_gpu_draft_rows_dev_tensor(dr->mk_cur, dr->mkE_dev, dr->ids, i, 1u, R, dr->mk_rows)) return false;
        } else if (!ds4_gpu_v41_row_gather_tensor(dr->mk_cur, m->map, m->size, e->weights.mtp.markov_embd->abs_offset,
                                                  nrow, R, eb, dr->ids, i, 0)) return false;
        /* ★判负存档(2026-09-17): 这里试过"有界剪枝的精确 argmax"★ —— 用 |bias_v| ≤ ‖W_v‖·‖e‖ 把不可能
         * 夺冠的词剪掉, 只对候选读那 512 B。实测**候选 129280/129280, 一个都没剪掉**, 草稿 13.2 → 15.9 ms。
         * 真因: markov 表的行与 embed 近正交, C-S 的界比 logits 的整个动态范围还大(详见 ds4_gpu_v41.h 的存档)。
         * 所以这三发保持原样: 读整张表算全词表偏置 → 加进 logits → argmax。它是带宽受限的(248 GB/s, 贴墙),
         * 占草稿一轮的 10%、整轮的 1.5% —— 要省只剩"换更低精度存表"或"只算 logits 的 top-K(近似)"。 */
        if (dr->mkH_dev) { if (!ds4_gpu_draft_bias_tensor(dr->mk_bias, dr->mk_cur, dr->mkH_dev, 1u, R, DS4_N_VOCAB)) return false; }
        else if (mkc) { if (!ds4_gpu_v41_matmul_bf16_skip_tensor(dr->mk_bias, m->map, m->size, e->weights.mtp.markov_head->abs_offset,
                                                                 R, DS4_N_VOCAB, dr->mk_cur, 1, dr->mk_hit)) return false; }
        else if (!v41_small_matmul(m, dr->mk_bias, e->weights.mtp.markov_head, R, DS4_N_VOCAB, dr->mk_cur, 1)) return false;
        if (mkc ? !ds4_gpu_v41_mkcache_add_tensor(st->logits, i, dr->mk_bias, dr->mk_cache, dr->mk_hit, DS4_N_VOCAB)
                : !ds4_gpu_v41_row_add_tensor(st->logits, i, dr->mk_bias, DS4_N_VOCAB)) return false;
        /* argmax 只会写自己那块的第 0 个 int, 所以先落 ids_next 再拷到 ids[i+1](官方 output_ids[:, i+1])。
         * 主路在采样时草稿也按塔的分布抽(2026-09-29): 接受率上限从 p(argmax q) 变成 Σmin(p,q); 样本落同一个槽的第 0 个 int。 */
        if (dr->dev_sample) {
            if (!ds4_gpu_v41_sample_tensor(dr->ids_next, st->logits, i, 1u, DS4_N_VOCAB, st->pos, st->tok, &dr->samp, NULL)) return false;
        } else if (!ds4_gpu_v41_argmax_tensor(dr->ids_next, st->logits, i, DS4_N_VOCAB)) return false;
        if (!ds4_gpu_tensor_copy(dr->ids, (uint64_t)(i + 1u) * 4, dr->ids_next, 0, 4)) return false;
    }
    /* confidence = proj([h_i ; markov_embed_i]) */
    for (uint32_t i = 0; i < B; i++) {
        if (!ds4_gpu_tensor_copy(dr->conf_in, (uint64_t)i * (E + R) * 4, dr->h, (uint64_t)i * E * 4, (uint64_t)E * 4)) return false;
        if (!ds4_gpu_tensor_copy(dr->conf_in, ((uint64_t)i * (E + R) + E) * 4, dr->mk_embed, (uint64_t)i * R * 4, (uint64_t)R * 4)) return false;
    }
    return v41_small_matmul(m, dr->conf, e->weights.mtp.confidence, (uint64_t)E + R, 1, dr->conf_in, B);
}

/* 补窗口: 从主态 mainh 环取 [first, first+rows) 行 → main_proj/main_norm → 推进三塔窗口。位置与起始行走 pinned 槽(进图时零拷贝读)。
 * ★这三步失败必须出声★(2026-09-18 实撞): 以前静默 return false, 调用方只当"这轮不出草稿", 于是一个缓冲区太小(窗口的块区只留了
 * block+1 行, 补 79 行直接越界)让投机整段静默失效 —— 门上"投机 == 纯解码"还是绿的(一轮都没投机当然逐字节同), 只有 t/s 露馅。 */
static bool v41_draft_fill(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, uint32_t rows) {
    if (!v41_draft_put(dr, dr->st.pos, 0, dr->p_wpos, (uint64_t)rows * 4)) { fprintf(stderr, "ds4: [v41] 草稿器: 写 %u 行窗口位置失败\n", rows); return false; }
    if (dr->cap_mode && !ds4_gpu_tensor_write_zerocopy(dr->firstd, 0, dr->p_first, 4)) { fprintf(stderr, "ds4: [v41] 草稿器: 写起始行失败\n"); return false; }
    if (!ds4_gpu_v41_ring_rows_tensor(dr->mainh_lin, main_st->mainh, g_ds4_v41.n_mtp_target * DS4_N_EMBD, main_st->mainh_cap,
                                      (uint32_t)dr->p_first[0], rows, dr->cap_mode ? dr->firstd : NULL)) {
        fprintf(stderr, "ds4: [v41] 草稿器: 从 main_hidden 环取 %u 行失败\n", rows); return false;
    }
    if (!v41_draft_main_x(e, dr, rows)) { fprintf(stderr, "ds4: [v41] 草稿器: main_proj/main_norm %u 行失败\n", rows); return false; }
    if (!v41_draft_push_main(e, &dr->st, rows)) { fprintf(stderr, "ds4: [v41] 草稿器: 推 %u 行进三塔窗口失败\n", rows); return false; }
    /* 块注意力的 main_x 只用最后一行(官方 forward_spec 的 main_hidden 是当前这一位) */
    if (rows > 1 && !ds4_gpu_tensor_copy(dr->st.main_x, 0, dr->st.main_x, (uint64_t)(rows - 1u) * DS4_N_EMBD * 4,
                                         (uint64_t)DS4_N_EMBD * 4)) return false;
    return true;
}

/* 一轮的 GPU 部分(补窗口 + 写 ids[0] + 块前向): 直发与捕获共用同一串调用, 差的只是主机写张量走同步拷还是零拷贝小核 */
static bool v41_draft_gpu_round(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    v41_draft_slots(main_st, dr, first, rows, pos0);
    if (rows && !v41_draft_fill(e, main_st, dr, rows)) return false;
    if (!v41_draft_put(dr, dr->ids, 0, dr->p_tok, 4)) return false;   /* p_tok[0] = 真 token(调用方已填) */
    return v41_draft_block(e, dr, pos0);
}

/* ★草稿一轮走图★(2026-09-22): 按 rows 各一张。捕获失败(捕获态下有分配/同步 ⇒ 作废)就永久回直发并出声; 捕获不执行任何核,
 * 所以失败后调用方按直发重来是干净的。返回 false = 没走图(调用方走直发), 状态一个没动。 */
static bool v41_draft_graph_round(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    const uint64_t gen = ds4_gpu_v41_scratch_generation();
    if (dr->gexec[rows] && dr->ggen[rows] != gen) {   /* 暂存换过指针 ⇒ 所有草稿图作废(它们烤死的都是捕获那一刻的指针) */
        for (uint32_t r = 0; r <= DS4_V41_DRAFT_GROWS; r++) { if (dr->gexec[r]) ds4_gpu_decode_graph_free(dr->gexec[r]); dr->gexec[r] = NULL; }
        fprintf(stderr, "ds4: [graph] 草稿图: 后端暂存换过指针, 全部重捕获\n");
    }
    if (!dr->gexec[rows]) {
        const double tc0 = now_sec();
        dr->cap_mode = 1;
        if (!ds4_gpu_decode_graph_capture_begin()) { dr->cap_mode = 0; dr->graph_off = 1; return false; }
        bool ok = v41_draft_gpu_round(e, main_st, dr, first, rows, pos0);
        if (ok) ok = ds4_gpu_tensor_read_zerocopy(dr->p_ids, dr->ids, 0, (uint64_t)(dr->block + 1u) * 4) != 0 &&
                     ds4_gpu_tensor_read_zerocopy(dr->p_conf, dr->conf, 0, (uint64_t)dr->block * 4) != 0;
        dr->cap_mode = 0;
        void *exec = ds4_gpu_decode_graph_capture_end();   /* 不管 ok 与否都要收捕获, 否则流一直停在捕获态 */
        if (!ok || !exec) {
            if (exec) ds4_gpu_decode_graph_free(exec);
            fprintf(stderr, "ds4: ★[graph] 草稿图(补 %u 行)捕获失败, 草稿一律直发★\n", rows);
            dr->graph_off = 1;
            return false;
        }
        dr->gexec[rows] = exec; dr->ggen[rows] = gen; dr->gcaps++; dr->h_capture += now_sec() - tc0;
        fprintf(stderr, "ds4: [graph] 草稿图(补 %u 行)已捕获\n", rows);
    }
    /* 重放: 图读的是重放那一刻槽里的值 ⇒ 先按本轮填(捕获那趟 gpu_round 已填过, 重填同值无害)。
     * ★只发不等(2026-10-07)★: 等与读回在 v41_draft_wait —— 调用方把上一轮的接受 token emit(fwrite+fflush)压在草稿跑着的时候做。 */
    const double tl0 = now_sec();
    v41_draft_slots(main_st, dr, first, rows, pos0);
    const double tl1 = now_sec();
    if (!ds4_gpu_decode_graph_launch(dr->gexec[rows])) return false;
    dr->t_launched = now_sec(); dr->h_slots += tl1 - tl0; dr->h_launch += dr->t_launched - tl1;
    dr->pending = 1;
    return true;
}
/* 草稿图发出之后的另一半: 等完、读回 host_ids/host_conf。没有在飞的图(直发路已同步跑完)就是空操作。 */
bool v41_draft_wait(ds4_v41_draft *dr) {
    if (!dr->pending) return true;
    dr->pending = 0;
    if (!ds4_gpu_synchronize()) return false;
    __sync_synchronize();
    memcpy(dr->host_ids, dr->p_ids, (size_t)(dr->block + 1u) * 4);
    memcpy(dr->host_conf, dr->p_conf, (size_t)dr->block * 4);
    dr->gsteps++;
    return true;
}

/* 返回 0 = 出不了草稿(料不齐); 1 = 草稿图已发出(host_ids/conf 要等 v41_draft_wait); 2 = 直发路已同步跑完(结果就绪, wait 是空操作)。 */
int v41_draft_launch(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main) {
    if (!dr->ready || !main_st->mainh) return 0;
    if (main_st->mainh_end != (int64_t)pos_main) {   /* 主前向没把这一位写进环(CED 跳过/早停)= 料不齐, 不出草稿 */
        fprintf(stderr, "ds4: [v41] 草稿器: main_hidden 环末位 %lld ≠ pos_main %u, 本轮不出草稿\n", (long long)main_st->mainh_end, pos_main);
        return 0;
    }
    /* ★补窗口: (win_end, pos_main] 缺多少补多少★ —— 歇过的轮、走过 graph 的步都在这里补齐; 缺口超过窗宽就整窗重建。
     * 以前按调用方递的"上一批确认了几行"推进, 歇轮那些步的 main_x 永远进不了窗口(实撞: 在线 p1 0.42 对教师强制 0.75)。 */
    const uint32_t SWA = DS4_N_SWA;
    int64_t first = dr->win_end + 1;
    if ((int64_t)pos_main - first + 1 > (int64_t)SWA) first = (int64_t)pos_main - (int64_t)SWA + 1;
    if (first < 0) first = 0;
    const uint32_t rows = (uint32_t)((int64_t)pos_main - first + 1);
    if (rows) {
        const int64_t avail_first = main_st->mainh_end - (int64_t)main_st->mainh_n + 1;
        if (first < avail_first) {
            fprintf(stderr, "ds4: [v41] 草稿器: 窗口要补 %lld..%u, 环里连续段只从 %lld 起(%u 行), 本轮不出草稿\n",
                    (long long)first, pos_main, (long long)avail_first, main_st->mainh_n);
            return 0;
        }
    }
    dr->host_ids[0] = tok; dr->p_tok[0] = tok;
    const uint32_t pos0 = pos_main + 1u;
    /* 走图的条件: 主路允许走图、这个 rows 直发暖过(暂存/核属性全建好)、位置 ≥ 窗宽(块注意力的 pos0 烤进图, 只有 lo = pos0−window
     * 那一支与 pos0 无关 —— 提示短于窗宽时窗口从 0 起、键数随位置变, 进不了一张图)、rows 在档里、没捕获失败过 */
    if (v41_graph_allowed(main_st) && !dr->graph_off && rows >= 1u && rows <= DS4_V41_DRAFT_GROWS && pos0 >= SWA && dr->gwarm[rows] > 0 &&
        v41_draft_graph_round(e, main_st, dr, first, rows, pos0)) {
        dr->win_end = pos_main;
        dr->rounds++; dr->last_warm = 0;
        return 1;
    }
    /* 暖身轮 = 本请求第一轮(草稿态的懒分配/首次触碰全在这一轮付), 或这个 rows 档第一次直发且下次能走图(这一轮在暖它的图档);
     * 位置 < 窗宽 / rows 超档 / 草稿图已关 的直发不是暖身, 那就是它的真实成本(调度器按 last_warm 决定记不记账) */
    const bool in_grows = rows >= 1u && rows <= DS4_V41_DRAFT_GROWS;
    dr->last_warm = dr->rounds == 0u || (in_grows && dr->gwarm[rows] == 0u && pos0 >= SWA && !dr->graph_off);
    if (!v41_draft_gpu_round(e, main_st, dr, first, rows, pos0)) return 0;
    if (rows) dr->win_end = pos_main;
    if (!ds4_gpu_synchronize()) return 0;
    if (!ds4_gpu_tensor_read(dr->ids, 0, dr->host_ids, (uint64_t)(dr->block + 1u) * 4)) return 0;
    if (!ds4_gpu_tensor_read(dr->conf, 0, dr->host_conf, (uint64_t)dr->block * 4)) return 0;
    if (in_grows) dr->gwarm[rows]++;
    dr->rounds++;
    dr->pending = 0;
    return 2;
}
bool v41_draft_step(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main) {
    const int r = v41_draft_launch(e, main_st, dr, tok, pos_main);
    return r == 2 || (r == 1 && v41_draft_wait(dr));
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_draft_nonempty_tu;
