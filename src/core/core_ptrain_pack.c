/* core_ptrain_pack.c — 后训练 ③ 第八版: 合批(几道题拼进一个批态, 2026-10-02)。总述见 core_ptrain.h 的 pt_pack。
 *
 * 一包 = nb 道题首尾相接进批态 r->st(第 i 题占 [r0_i, r0_i + n_i), 总行数 R ≤ maxlen), 每题的位置各自从 0 起。
 * 逐 token 的段(embed / engram 门 / hc 三件 / 注意力投影进出 / MoE / 出口头)在批态上按 R 行一次发 —— 每遍专家位流只读一次, 摊给 nb 道题;
 * 注意力缓存段(kvn 进窗口 / 压缩源 / indexer / 稀疏注意力)按题在各自的请求槽上发(核与单请求路同一批, 一个没改), 与服务端多路合批
 * (core_v41_multi.c)同一个拆法。反传: 逐 token 的段整批走, 注意力反传与压缩器反传按题分段(core_ptrain_layer.c 的 pt_segs)。
 *
 * 请求槽的缓存字段全是批态自己缓冲的视图, 不另开池:
 *   - 窗口: 第 i 题切批态窗口的 [r0_i, r0_i + SWA + n_i) 行。块区 [SWA + r0_i, +n_i) 与别题首尾相接(合起来正好是单题路的 [SWA, SWA+R));
 *     环头 [r0_i, r0_i + SWA) 压在前几题的块区上 —— 位置从 0 起的注意力不读环头(v41_win_row: a ≥ pos0 = 0 恒走块区), 前向末尾的环提交
 *     写进去时前几题本层的注意力已经算完; 反传重算时整批的 kvn 一发拷回块区, 不再发环提交。
 *   - 压缩器余行(cpre)与索引中间量从第 r0_i 行切, 压缩行 / 索引键从第 g0_i 组切(前面各题 n_j/ratio 之和) —— 都只写自己那段。
 * 出错会怎样: 视图一重叠, 后算的题盖掉先算的题的 KV, 不报错只出错的注意力 —— 门 = packcheck(同一批题单题路 vs 合批逐题对账)。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

static void pt_free_t(ds4_gpu_tensor **t) { if (*t) { ds4_gpu_tensor_free(*t); *t = NULL; } }

/* 槽里每包重挂的缓存视图(行视图由 v41_detach 摘) */
static void pt_slot_views_free(ds4_v41_state *m) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { pt_free_t(&m->win[il]); pt_free_t(&m->comp_kv[il]); pt_free_t(&m->index_k[il]); pt_free_t(&m->cpre_kv[il]); pt_free_t(&m->cpre_sc[il]); }
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) pt_free_t(&m->eraw[k]);
    ds4_gpu_tensor **v[] = { &m->posg, &m->ckv, &m->csc, &m->pooled, &m->latent, &m->ktmp, &m->iq, &m->iw, &m->idx };
    for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) pt_free_t(v[i]);
}

/* engram 原始行每行字节(24 列 × (256 + 8)): 与 v41_engram_rows 的 eraw 分配式同源 */
static uint64_t pt_eraw_rowb(void) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    return (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads * (v->engram_head_dim + v->engram_head_dim / 32u);
}

bool pt_pack_alloc(const pt_cfg *c, pt_run *r) {
    (void)c;
    pt_pack *pk = &r->pk; ds4_v41_state *B = &r->st;
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint64_t cap = B->cap_tok;
    memset(pk, 0, sizeof *pk);
    bool ok = true;
    if (v->n_engram && !B->no_engram && !B->erows) {   /* 槽的 erows/eraw 是批态这几块的视图: 先按单题路懒建的同一份分配式建好(v41_engram_rows) */
        const uint64_t cols = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads;
        for (uint32_t k = 0; k < v->n_engram; k++) B->eraw[k] = v41_alloc(cap * pt_eraw_rowb(), &ok);
        B->erows = v41_alloc(cap * cols * v->engram_head_dim * 4, &ok);
        B->ekv = v41_alloc(cap * (DS4_N_HC + 1u) * DS4_N_EMBD * 4, &ok);
    }
    pk->rq = xmalloc_zeroed(PT_PACK_MAX, sizeof *pk->rq);
    for (uint32_t i = 0; i < PT_PACK_MAX; i++)   /* 先全标"没开": 中途分配失败时 pt_pack_free 会对每个槽 close 分片 fd, 清零的 0 是 stdin */
        for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) pk->rq[i].eshard[k].fd = -1;
    for (uint32_t i = 0; i < PT_PACK_MAX && ok; i++) {   /* 槽自己的只有: 位置历史 / 组位置 pinned 槽 / engram 取行任务(第一次预取时按 cap 建) */
        ds4_v41_state *m = &pk->rq[i];
        m->cap_tok = B->cap_tok; m->ctx = B->ctx; m->idx_owner = -1; m->cand_owner = -1; m->no_engram = B->no_engram;
        m->hist = xmalloc((size_t)m->ctx * 4);
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++)
            if (v->is_kv_source[il] && !(m->posg_pin[il] = ds4_gpu_host_alloc(cap * 4u))) ok = false;
    }
    pk->ready = ok;
    if (!ok) fprintf(stderr, "ds4: [ptrain] 合批请求槽分配失败(%u 槽, cap %llu)\n", PT_PACK_MAX, (unsigned long long)cap);
    return ok;
}

void pt_pack_free(pt_run *r) {
    pt_pack *pk = &r->pk;
    for (uint32_t i = 0; pk->rq && i < PT_PACK_MAX; i++) {
        ds4_v41_state *m = &pk->rq[i];
        v41_detach(m, &pk->rv[i]);
        pt_slot_views_free(m);
        v41_engram_close(m);   /* 取行任务 + 分片 fd; eraw/erows/ekv 此刻都是 NULL(视图已摘), 不会去放批态的 */
        pt_free_t(&m->iscore); pt_free_t(&m->cand);
        for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) if (m->posg_pin[il]) { ds4_gpu_host_free(m->posg_pin[il]); m->posg_pin[il] = NULL; }
        free(m->hist); m->hist = NULL;
    }
    free(pk->rq); pk->rq = NULL;
    pk->ready = 0; pk->nb = 0;
}

/* 第 i 题挂到批态第 r0 行起的 n 行: 缓存视图从批态缓冲切, 计数清零, 行视图挂进批态 */
static bool pt_slot_bind(pt_run *r, uint32_t i, const pt_sample *s) {
    pt_pack *pk = &r->pk; ds4_v41_state *m = &pk->rq[i]; const ds4_v41_state *B = &r->st;
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t r0 = pk->r0[i], n = pk->n[i];
    const uint64_t HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA, rowb = HD * 4u, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    v41_detach(m, &pk->rv[i]);
    pt_slot_views_free(m);
    m->n_past = 0; m->graph = 0; m->snap_on = 0; m->draft = 0;
    m->tsave = NULL;   /* 槽不存档: 训练存档整批记在批态(各题的 idx 由 pt_pack_forward 按行搬进去) */
    memset(m->cpend, 0, sizeof m->cpend); memset(m->ng_src, 0, sizeof m->ng_src); memset(m->win_from, 0, sizeof m->win_from);
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ok = (m->win[il] = ds4_gpu_tensor_view(B->win[il], (uint64_t)r0 * rowb, (SWA + n) * rowb)) != NULL;
        if (!ok || !v->is_kv_source[il]) continue;
        /* +2 与 v41_state_alloc 的 ngcap 同式; 直发路只写前 n/ratio 组, 后两格压在下一题头上也没人写 */
        const uint64_t ratio = ds4_layer_compress_ratio(il), g0 = pk->g0[i][il], ng = n / ratio + 2u;
        m->comp_kv[il] = ds4_gpu_tensor_view(B->comp_kv[il], g0 * DS4_V41_CKV_BYTES, ng * DS4_V41_CKV_BYTES);
        m->index_k[il] = ds4_gpu_tensor_view(B->index_k[il], g0 * DS4_V41_IDXK_BYTES, ng * DS4_V41_IDXK_BYTES);
        ok = m->comp_kv[il] && m->index_k[il];
        if (ok && ratio > 1u) {
            m->cpre_kv[il] = ds4_gpu_tensor_view(B->cpre_kv[il], (uint64_t)r0 * rowb, (ratio + n) * rowb);
            m->cpre_sc[il] = ds4_gpu_tensor_view(B->cpre_sc[il], (uint64_t)r0 * rowb, (ratio + n) * rowb);
            ok = m->cpre_kv[il] && m->cpre_sc[il];
        }
    }
    if (ok) {
        m->posg = pt_rows(B->posg, r0, n, 1); m->ckv = pt_rows(B->ckv, r0, n, HD); m->csc = pt_rows(B->csc, r0, n, HD);
        m->pooled = pt_rows(B->pooled, r0, n, HD); m->latent = pt_rows(B->latent, r0, n, HD); m->ktmp = pt_rows(B->ktmp, r0, n, IK);
        m->iq = pt_rows(B->iq, r0, n, IH * IK); m->iw = pt_rows(B->iw, r0, n, IH); m->idx = pt_rows(B->idx, r0, n, DS4_N_INDEXER_TOP_K);
        ok = m->posg && m->ckv && m->csc && m->pooled && m->latent && m->ktmp && m->iq && m->iw && m->idx;
    }
    for (uint32_t k = 0; ok && !m->no_engram && k < v->n_engram; k++)
        ok = (m->eraw[k] = ds4_gpu_tensor_view(B->eraw[k], (uint64_t)r0 * pt_eraw_rowb(), (uint64_t)n * pt_eraw_rowb())) != NULL;
    memcpy(m->hist, s->sids, (size_t)n * 4);   /* engram 按本题自己的 token 历史拼 n-gram(题与题之间不许串) */
    if (ok) ok = v41_attach(m, B, r0, n, &pk->rv[i]);
    if (!ok) fprintf(stderr, "ds4: [ptrain] 合批第 %u 题(行 %u+%u)挂槽失败\n", i, r0, n);
    return ok;
}

/* 合批前向: 与 v41_layer 逐层同一串函数, 逐 token 的段按 R 行在批态上发、缓存段按题在槽上发; 训练存档(hc_in/pm_in/ekv/sel)按 R 行,
 * 注意力用的组号/组数/topk 按题存(各题的压缩行是各自的)。 */
static bool pt_pack_forward(ds4_engine *e, pt_run *r) {
    pt_pack *pk = &r->pk; ds4_v41_state *B = &r->st; v41_tsave *sv = &r->save;
    const ds4_model *md = &e->model;
    const uint32_t R = pk->R, E = DS4_N_EMBD, HC = DS4_N_HC, TK = DS4_N_INDEXER_TOP_K;
    pt_state_reset(B);
    B->n = R; B->pos0 = 0; B->graph = 0; B->ced_skip = 0; B->head_last_only = 0; B->stop_early = 0; B->idx_owner = -1; B->cand_owner = -1;
    bool ok = true;
    {   /* 行: 各题 token 首尾相接, 位置各自从 0 起; pre_mix = one-hot(第 0 路), 与 v41_forward 同 */
        int32_t *tok = xmalloc((size_t)R * 4), *pos = xmalloc((size_t)R * 4);
        float *pm = xmalloc((size_t)R * HC * 4);
        for (uint32_t i = 0; i < pk->nb; i++) for (uint32_t j = 0; j < pk->n[i]; j++) { tok[pk->r0[i] + j] = pk->rq[i].hist[j]; pos[pk->r0[i] + j] = (int32_t)j; }
        for (uint32_t q = 0; q < R * HC; q++) pm[q] = (q % HC) == 0 ? 1.0f : 0.0f;
        ok = ds4_gpu_tensor_write(B->tok, 0, tok, (uint64_t)R * 4) && ds4_gpu_tensor_write(B->pos, 0, pos, (uint64_t)R * 4) &&
             ds4_gpu_tensor_write(B->pre_mix, 0, pm, (uint64_t)R * HC * 4);
        free(tok); free(pos); free(pm);
    }
    for (uint32_t i = 0; ok && i < pk->nb; i++) if (!pk->rq[i].no_engram && !v41_engram_prefetch(e, &pk->rq[i])) ok = false;   /* 盘读与前几层重叠 */
    if (ok && ds4_gpu_begin_commands() == 0) ok = false;
    if (ok) ok = v41_embed(md, B->x, B->tok, e->weights.token_embd, DS4_N_VOCAB, R, E) && ds4_gpu_v41_expand_hc_tensor(B->hc, B->x, E, HC, R);
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (sv->hc_in[il] && (!ds4_gpu_bwd_pack_bf16_tensor(sv->hc_in[il], B->hc, (uint64_t)R * HC * E) ||
                              !ds4_gpu_tensor_copy(sv->pm_in[il], 0, B->pre_mix, 0, (uint64_t)R * HC * 4))) ok = false;
        if (ok && !B->no_engram && g_ds4_v41.engram_index_of[il] >= 0) {   /* 取行按题, wkv/门按批(同 v41_multi_body) */
            for (uint32_t i = 0; ok && i < pk->nb; i++) if (!v41_engram_rows(e, &pk->rq[i], il)) ok = false;
            if (ok && !v41_engram_apply(e, B, il)) ok = false;
            if (ok && sv->ekv[il] && !ds4_gpu_bwd_pack_bf16_tensor(sv->ekv[il], B->ekv, (uint64_t)R * (HC + 1u) * E)) ok = false;
        }
        if (ok) ok = v41_hc_half(e, B, l, true) && v41_attn_in(e, B, il);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const int16_t src = ratio ? g_ds4_v41.kv_source_of[il] : -1;
        for (uint32_t i = 0; ok && i < pk->nb; i++) {
            ds4_v41_state *m = &pk->rq[i];
            if (!v41_attn_cache(e, m, il)) { ok = false; break; }
            pk->ng[i][il] = src >= 0 ? m->ng_src[src] : 0u;   /* 与 v41_layer 的存档同式, 按题 */
            pk->topk[i][il] = (src >= 0 && pk->ng[i][il]) ? m->idx_topk : 0u;
            pk->iratio[i][il] = pk->topk[i][il] ? m->idx_ratio : 0u;
            if (pk->topk[i][il] && sv->idx[il] &&   /* 第 i 题的 idx 落在存档第 r0·TK 格起(行距 topk_i, 与单题路同一个紧排) */
                !ds4_gpu_tensor_copy(sv->idx[il], (uint64_t)pk->r0[i] * TK * 4u, m->idx, 0, (uint64_t)pk->n[i] * pk->topk[i][il] * 4u)) ok = false;
        }
        if (ok) ok = v41_attn_out(e, B, il) && ds4_gpu_v41_hc_post_tensor(B->hc2, B->attn_out, B->hc, B->post, B->comb, E, HC, R);
        if (ok) { ds4_gpu_tensor *t = B->hc; B->hc = B->hc2; B->hc2 = t; }
        if (ok) ok = v41_hc_half(e, B, l, false) && v41_moe(md, l, B, il) && ds4_gpu_v41_hc_post_tensor(B->hc2, B->y, B->hc, B->post, B->comb, E, HC, R);
        if (ok) { ds4_gpu_tensor *t = B->hc; B->hc = B->hc2; B->hc2 = t; }
        if (ok && ds4_gpu_flush_commands() == 0) ok = false;
    }
    if (ok) ok = ds4_gpu_v41_hc_pre_tensor(B->x, B->hc, B->pre_mix, E, HC, R) &&
                 ds4_gpu_v41_rms_norm_tensor(B->xn, B->x, md->map, md->size, e->weights.output_norm->abs_offset, E, R, DS4_RMS_EPS) &&
                 v41_tproj(md, B->logits, e->weights.output, E, DS4_N_VOCAB, B->xn, R, 0);
    if (ok && (ds4_gpu_end_commands() == 0 || ds4_gpu_synchronize() == 0)) ok = false;
    if (!ok) fprintf(stderr, "ds4: [ptrain] 合批前向失败(%u 题 %u 行)\n", pk->nb, R);
    return ok;
}

bool pt_pack_step(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss, uint32_t nb, int grad, double *loss_each, uint32_t *ntok_each) {
    pt_pack *pk = &r->pk; ds4_v41_state *B = &r->st;
    const ds4_model *mdl = &e->model;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, V = DS4_N_VOCAB;
    if (!pk->ready || !nb || nb > PT_PACK_MAX) return false;
    uint32_t R = 0, gacc[DS4_MAX_LAYER];
    memset(gacc, 0, sizeof gacc);
    for (uint32_t i = 0; i < nb; i++) {
        pk->r0[i] = R; pk->n[i] = ss[i]->sn; R += ss[i]->sn; loss_each[i] = 0.0; ntok_each[i] = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++)
            if (g_ds4_v41.is_kv_source[il]) { pk->g0[i][il] = gacc[il]; gacc[il] += ss[i]->sn / ds4_layer_compress_ratio(il); }
    }
    if (R > B->cap_tok) { fprintf(stderr, "ds4: [ptrain] 合批 %u 行超批态 %u\n", R, B->cap_tok); return false; }
    pk->nb = nb; pk->R = R;
    const double q0 = pt_tick(c);
    bool ok = true;
    for (uint32_t i = 0; ok && i < nb; i++) ok = pt_slot_bind(r, i, ss[i]);
    if (ok) ok = pt_pack_forward(e, r);
    const double q1 = pt_tick(c);
    /* 各题答案位的 KL: 教师表逐题上传, 核读第 row 行起 m 行 logits、梯度就地写回那几行(与 pt_step_sample 同一个核同一个缩放) */
    ds4_gpu_tensor *glog[PT_PACK_MAX];
    memset(glog, 0, sizeof glog);
    for (uint32_t i = 0; ok && i < nb; i++)
        ok = pt_loss_rows(c, r, B->logits, pk->r0[i] + ss[i]->sa0 - 1u, ss[i], &loss_each[i], &ntok_each[i], &glog[i]);
    const double q2 = pt_tick(c);
    double q3 = q2, q4 = q2;
    if (ok && grad) {   /* 出口: 各题答案行的 g_xn = glog·W_head(其余行 0) → RMSNorm → hc_pre, 整批 R 行; 再逐层往下(按题分段的在层里) */
        ok = ds4_gpu_tensor_fill_f32(r->gxn, 0.f, (uint64_t)R * E);
        for (uint32_t i = 0; ok && i < nb; i++) {
            const uint32_t m = ss[i]->m, row = pk->r0[i] + ss[i]->sa0 - 1u;
            ds4_gpu_tensor *gv = pt_rows(r->gxn, row, m, E);
            ok = gv && ds4_gpu_bwd_matmul_t_tensor(gv, mdl->map, mdl->size, e->weights.output->type, e->weights.output->abs_offset, E, V, glog[i], m, 0);
            if (gv) ds4_gpu_tensor_free(gv);
        }
        if (ok) ok = ds4_gpu_bwd_rms_norm_tensor(r->gx, r->gxn, B->x, mdl->map, mdl->size, e->weights.output_norm->abs_offset, E, R, DS4_RMS_EPS, 0) &&
                     ds4_gpu_tensor_fill_f32(r->ghc, 0.f, (uint64_t)R * HC * E) &&
                     ds4_gpu_bwd_hc_pre_tensor(r->ghc, r->gpre, r->gx, B->hc, B->pre_mix, E, HC, R);
        q3 = pt_tick(c);
        if (ok) ok = pt_layers_bwd(e, c, r, R);
        q4 = pt_tick(c);
    }
    for (uint32_t i = 0; i < nb; i++) if (glog[i]) ds4_gpu_tensor_free(glog[i]);
    if (c->prof && grad) {   /* 段时间是整包的, 题数 += nb: 打出来的"秒/题"与单题路同口径可比 */
        r->tm[PT_TM_FWD] += q1 - q0; r->tm[PT_TM_LOSS] += q2 - q1; r->tm[PT_TM_EXIT] += q3 - q2; r->tm[PT_TM_LAYERS] += q4 - q3;
        r->tm[PT_TM_Q] += pt_tick(c) - q0; r->tm_n += nb;
    }
    pk->nb = 0;   /* 包用完: 之后的单题路(检查/探针)按单题口径走 */
    if (!ok) fprintf(stderr, "ds4: [ptrain] 合批一步失败(%u 题 %u 行)\n", nb, R);
    return ok;
}

bool pt_run_list(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss, uint32_t n, int grad,
                 double *loss_sum, uint32_t *ntok, double *loss_each, uint32_t *ntok_each) {
    if (loss_each) for (uint32_t k = 0; k < n; k++) { loss_each[k] = 0.0; ntok_each[k] = 0; }
    if (!r->pk.ready || c->nopack) {   /* 不合批: 一题一题过(10-02 之前的路, 只作对照) */
        for (uint32_t k = 0; k < n; k++) {
            double l = 0; uint32_t t = 0;
            if (!pt_step_sample(e, c, r, ss[k], grad, &l, &t)) return false;
            *loss_sum += l; *ntok += t;
            if (loss_each) { loss_each[k] = l; ntok_each[k] = t; }
        }
        return true;
    }
    /* 装包: 从长到短, 每题进第一个装得下的包(行数 ≤ cap、题数 ≤ PT_PACK_MAX)。梯度是各题之和, 怎么分包不改数学(只差浮点加法次序) */
    uint32_t *ix = xmalloc((size_t)(n + 1) * 4), *bin = xmalloc((size_t)(n + 1) * 4), *brow = xmalloc((size_t)(n + 1) * 4), *bcnt = xmalloc((size_t)(n + 1) * 4), nbin = 0, nx = 0;
    for (uint32_t k = 0; k < n; k++) if (pt_runnable(r, ss[k])) ix[nx++] = k;   /* 教师没给表 / 超长的题: 与 pt_step_sample 一样跳过 */
    for (uint32_t a = 1; a < nx; a++) for (uint32_t b = a; b > 0 && ss[ix[b]]->sn > ss[ix[b - 1]]->sn; b--) { const uint32_t t = ix[b]; ix[b] = ix[b - 1]; ix[b - 1] = t; }
    for (uint32_t a = 0; a < nx; a++) {
        const uint32_t len = ss[ix[a]]->sn;
        uint32_t b = 0;
        while (b < nbin && (brow[b] + len > r->st.cap_tok || bcnt[b] >= PT_PACK_MAX)) b++;
        if (b == nbin) { brow[nbin] = 0; bcnt[nbin] = 0; nbin++; }
        bin[a] = b; brow[b] += len; bcnt[b]++;
    }
    bool ok = true;
    for (uint32_t b = 0; ok && b < nbin; b++) {
        const pt_sample *ps[PT_PACK_MAX]; uint32_t who[PT_PACK_MAX], np = 0, nt[PT_PACK_MAX];
        double le[PT_PACK_MAX];
        for (uint32_t a = 0; a < nx; a++) if (bin[a] == b) { who[np] = ix[a]; ps[np++] = ss[ix[a]]; }
        if (np == 1) { le[0] = 0; nt[0] = 0; ok = pt_step_sample(e, c, r, ps[0], grad, &le[0], &nt[0]); }   /* 一题一包: 单题路就是它 */
        else ok = pt_pack_step(e, c, r, ps, np, grad, le, nt);
        for (uint32_t q = 0; ok && q < np; q++) {
            *loss_sum += le[q]; *ntok += nt[q];
            if (loss_each) { loss_each[who[q]] = le[q]; ntok_each[who[q]] = nt[q]; }
        }
    }
    free(ix); free(bin); free(brow); free(bcnt);
    return ok;
}

/* 预热探底: 训练题里从长到短装一个最满的包(行数峰值)、再从短到长装一个题数最多的包(槽数峰值, 每个槽第一次用时建取行任务), 各走一趟
 * 前向 + 反传 —— 按需长的暂存与槽一次长到顶, 内存峰值在开训前量出来。梯度由调用方清掉。不合批时 = 最长那题单走一趟(10-02 之前的预热)。 */
bool pt_pack_warm(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d, uint32_t *rows) {
    const pt_sample **ss = xmalloc((size_t)(d->ns + 1) * sizeof *ss);
    uint32_t n = 0;
    for (uint32_t i = 0; i < d->ns; i++) if (!d->s[i].eval && pt_runnable(r, &d->s[i])) ss[n++] = &d->s[i];
    for (uint32_t a = 1; a < n; a++) for (uint32_t b = a; b > 0 && ss[b]->sn > ss[b - 1]->sn; b--) { const pt_sample *t = ss[b]; ss[b] = ss[b - 1]; ss[b - 1] = t; }
    const bool pack = r->pk.ready && !c->nopack;
    bool ok = true;
    *rows = 0;
    for (int pass = 0; ok && n && pass < (pack ? 2 : 1); pass++) {
        const pt_sample *ps[PT_PACK_MAX]; uint32_t np = 0, R = 0, nt[PT_PACK_MAX];
        double le[PT_PACK_MAX];
        for (uint32_t a = 0; a < n && np < (pack ? PT_PACK_MAX : 1u); a++) {
            const pt_sample *s = ss[pass ? n - 1u - a : a];
            if (R + s->sn > r->st.cap_tok) continue;
            ps[np++] = s; R += s->sn;
        }
        if (R > *rows) *rows = R;
        le[0] = 0; nt[0] = 0;
        ok = np == 1 ? pt_step_sample(e, c, r, ps[0], 1, &le[0], &nt[0]) : pt_pack_step(e, c, r, ps, np, 1, le, nt);
        fprintf(stderr, "ds4: [ptrain] 预热第 %d 包: %u 题 %u 行\n", pass + 1, np, R);
    }
    free(ss);
    return ok;
}

#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_pack_nonempty_tu;
