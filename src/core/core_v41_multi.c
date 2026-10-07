/* core_v41_multi.c — 多状态一步(2026-09-30, batch.md §3.1; 用户令"支持 CUDA 架构的并发, 量三并发的输出速度")。
 *
 * 说人话: N 条请求各有自己的 KV(请求态), 每步各喂 1 行(纯解码)或 1+k 行(投机验证批); 稠密段(embed / hc 三件 / norm / MoE / 出口头)把
 * 全部行拼成一次发, 权重每步只读一遍 —— GB10 上解码是字节墙(6.3 GB/步), 这是并发唯一的红利; 注意力的投影段(q_a/q_b/kv/wo_a/wo_b, 一层 71 MB、
 * 40 层 2.85 GB = 骨架的 70%)也按批发, 只有缓存段(压缩源 / indexer / 稀疏注意力 / 环提交)逐请求发(各路一条流):
 * 各请求的 tok/pos/xn/erows/qrn/q/kvn/o 是批态行缓冲里自己那几行的**视图**(ds4_gpu_tensor_view, 不拷贝), 其余中间量在请求态里 ——
 * core_v41_attn.c 的核一个不改(它们只认 tensor + n + pos)。engram 同理: 取行按请求(各自的 hist / 取行任务), wkv 那一发按批。
 * ★第一版把整段注意力逐请求发, 3 路一步 85 ms(1 路 37): 那 2.85 GB 按路数重复读, +24 ms/两路 —— 09-30 实撞, 拆三段后修。★
 * 投机(batch.md 第二期): 每路各出草稿, 验证行(1+k)拼进同一次前向, 各路按自己的行取 token、接受最长前缀、各自回滚(core_v41_req.c)。
 * ★直发 vs 捕图★(09-30 nsys 定罪): 缓存段的核 11~13 µs 一发、主机发射 ~5 µs 一发, 三路分流也喂不饱 GPU(道间几乎不重叠), 主流自己 835 发/步 ≈ 4 ms 间隙
 * ⇒ 整步捕成 CUDA graph(core_v41_mgraph.c): 前向主体 v41_multi_body 直发与捕获共用, 捕获时各路按"设备位置"口径(st->graph=1), 各道的分支由图执行器并行跑。
 * 门 = 温 0 下每路输出与单请求路逐字节同(fable5 09-30)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

int g_ds4_v41_lanes = 1;   /* 合批缓存段按路分流(默认开); --no-lanes 关掉只作 A/B(输出逐字节同, 只是发法) */
void ds4_engine_v41_set_lanes(int on) { g_ds4_v41_lanes = on ? 1 : 0; }

bool v41_batch_alloc(ds4_v41_batch *b, uint32_t cap) {
    memset(b, 0, sizeof *b);
    if (!cap || cap > DS4_V41_GEMV_MAX_TOK) { fprintf(stderr, "ds4: V4.1 批态行数 %u 超解码小批核路上限 %u\n", cap, DS4_V41_GEMV_MAX_TOK); return false; }
    if (!v41_batch_rows_alloc(&b->rows, cap)) return false;
    b->cap = cap;
    b->am = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    b->onehot = xmalloc((size_t)cap * DS4_N_HC * 4);
    for (uint32_t i = 0; i < cap * DS4_N_HC; i++) b->onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    if (g_ds4_v41.n_engram) {   /* engram 解码行 / wkv 出口按批开(取行按请求, 各自的 erows 是这里的行视图) */
        const ds4_v41_cfg *v = &g_ds4_v41;
        const uint64_t in_dim = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads * v->engram_head_dim;
        b->rows.erows = ds4_gpu_tensor_alloc((uint64_t)cap * in_dim * 4);
        b->rows.ekv = ds4_gpu_tensor_alloc((uint64_t)cap * (DS4_N_HC + 1) * DS4_N_EMBD * 4);
        if (!b->rows.erows || !b->rows.ekv) { v41_batch_free(b); return false; }
    }
    if (!b->am) { v41_batch_free(b); return false; }
    return true;
}

void v41_batch_free(ds4_v41_batch *b) {
    v41_mgraph_free(b);
    if (b->am) ds4_gpu_tensor_free(b->am);
    free(b->onehot);
    v41_state_free(&b->rows);
    memset(b, 0, sizeof *b);
}

/* 请求态挂到批态第 r 行起的 rows 行: 八个视图(tok/pos/xn/erows + 注意力投影段的 qrn/q/kvn/o) + 本步的行数 / 位置 n_past。
 * 缓存段(v41_attn_cache)从视图读本路的 q/qrn/kvn、写本路的 o; 投影段在批态上按 R 行发。detach 归还视图, 字段回 NULL。 */
bool v41_attach(ds4_v41_state *m, const ds4_v41_state *B, uint32_t r, uint32_t rows, v41_rowview *v) {
    const uint64_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q;
    const ds4_v41_cfg *cfg = &g_ds4_v41;
    const uint64_t erow = cfg->n_engram ? (uint64_t)(cfg->engram_max_ngram - 1) * cfg->engram_heads * cfg->engram_head_dim * 4 : 0;
    v->tok = ds4_gpu_tensor_view(B->tok, (uint64_t)r * 4, (uint64_t)rows * 4);
    v->pos = ds4_gpu_tensor_view(B->pos, (uint64_t)r * 4, (uint64_t)rows * 4);
    v->xn = ds4_gpu_tensor_view(B->xn, (uint64_t)r * E * 4, (uint64_t)rows * E * 4);
    v->erows = erow ? ds4_gpu_tensor_view(B->erows, (uint64_t)r * erow, (uint64_t)rows * erow) : NULL;
    v->qrn = ds4_gpu_tensor_view(B->qrn, (uint64_t)r * Q * 4, (uint64_t)rows * Q * 4);
    v->q = ds4_gpu_tensor_view(B->q, (uint64_t)r * NH * HD * 4, (uint64_t)rows * NH * HD * 4);
    v->kvn = ds4_gpu_tensor_view(B->kvn, (uint64_t)r * HD * 4, (uint64_t)rows * HD * 4);
    v->o = ds4_gpu_tensor_view(B->o, (uint64_t)r * NH * HD * 4, (uint64_t)rows * NH * HD * 4);
    if (!v->tok || !v->pos || !v->xn || (erow && !v->erows) || !v->qrn || !v->q || !v->kvn || !v->o) return false;
    m->tok = v->tok; m->pos = v->pos; m->xn = v->xn; m->erows = v->erows; m->qrn = v->qrn; m->q = v->q; m->kvn = v->kvn; m->o = v->o;
    m->n = rows; m->pos0 = m->n_past; m->idx_owner = -1; m->cand_owner = -1; m->idx_topk = 0; m->idx_ratio = 0; m->stop_early = 0;
    m->ced_skip = 0; m->mainh_wrote = 0;
    return true;
}

void v41_detach(ds4_v41_state *m, v41_rowview *v) {
    ds4_gpu_tensor **t[] = { &v->tok, &v->pos, &v->xn, &v->erows, &v->qrn, &v->q, &v->kvn, &v->o };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) { if (*t[i]) ds4_gpu_tensor_free(*t[i]); *t[i] = NULL; }
    m->tok = NULL; m->pos = NULL; m->xn = NULL; m->erows = NULL; m->qrn = NULL; m->q = NULL; m->kvn = NULL; m->o = NULL;
}

/* --v41-prof 下的逐段账: 每段后 flush + synchronize 记墙钟(会把总时长撑大, 但各段的比例是对的 —— d0a 那套的口径), 每步打一行。
 * 段: embed / engram / hc 三件 / 注意力投影进 / 缓存段(逐路) / 注意力投影出 / MoE / 出口。 */
typedef struct { double embed, engram, hc, attn_in, cache[DS4_V41_GEMV_MAX_TOK], attn_out, moe, exit; } v41_mprof;
static double mp_tick(double *acc, double t) { if (ds4_gpu_flush_commands()) (void)ds4_gpu_synchronize(); const double n = now_sec(); *acc += n - t; return n; }

/* 前向主体: embed → 40 层 → 出口 logits(R 行)。直发(v41_multi_step)与捕获(core_v41_mgraph.c)共用: 里面没有主机等待/同步拷贝/位置推进。
 * 各路已由调用方挂好视图(n / pos0 / 视图); 捕获时各路 st->graph=1, 缓存段按设备位置口径发。 */
bool v41_multi_body(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const uint32_t *r0, const uint32_t *nr, uint32_t nm, uint32_t R) {
    ds4_v41_state *B = &b->rows;
    const ds4_model *md = &e->model;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC;
    const int prof = g_ds4_v41_prof;
    v41_mprof P; memset(&P, 0, sizeof P);
    double tp = prof ? now_sec() : 0;
    B->n = R; B->pos0 = 0; B->graph = 0; B->stop_early = 0; B->ced_skip = 0; B->idx_owner = -1; B->cand_owner = -1;
    bool ok = v41_embed(md, B->x, B->tok, e->weights.token_embd, DS4_N_VOCAB, R, E) != 0 &&
              ds4_gpu_v41_expand_hc_tensor(B->hc, B->x, E, HC, R) != 0;
    if (prof) tp = mp_tick(&P.embed, tp);
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (g_ds4_v41.engram_index_of[il] >= 0) {   /* 官方 layer(h) 之前 h = engram(h): 取行按请求, wkv/门按批 */
            for (uint32_t i = 0; ok && i < nm; i++) if (!m[i]->no_engram && !v41_engram_rows(e, m[i], il)) ok = false;
            if (ok && !v41_engram_apply(e, B, il)) ok = false;
            if (prof) tp = mp_tick(&P.engram, tp);
        }
        /* DSpark 目标层: 各路本步的注意力输入(engram 之后、层之前的 hc 四路均值)按路落进各自的 main_hidden 环(与 v41_layer 同一处、同一核) */
        if (ok && g_ds4_v41.mtp_target_slot[il] >= 0)
            for (uint32_t i = 0; ok && i < nm; i++) {
                if (!m[i]->mainh) continue;
                const uint32_t rws = nr[i] < m[i]->mainh_cap ? nr[i] : m[i]->mainh_cap, src0 = r0[i] + (nr[i] - rws);
                if (!ds4_gpu_v41_hc_mean_tensor(m[i]->mainh, B->hc, E, HC, rws, src0, (uint32_t)g_ds4_v41.mtp_target_slot[il],
                                                g_ds4_v41.n_mtp_target, m[i]->pos0 + (nr[i] - rws), m[i]->mainh_cap,
                                                m[i]->graph ? m[i]->pos : NULL)) ok = false;
                m[i]->mainh_wrote = rws;
            }
        if (ok && !v41_hc_half(e, B, l, true)) ok = false;
        if (prof) tp = mp_tick(&P.hc, tp);
        /* 注意力三段(core_v41_attn.c): 投影进按批(q_a/q_b/kv 权重读一遍) → 缓存段逐请求(各自的 KV) → 投影出按批(wo_a/wo_b 读一遍) */
        if (ok && !v41_attn_in(e, B, il)) ok = false;
        if (prof) tp = mp_tick(&P.attn_in, tp);
        /* 缓存段各路互不相干 ⇒ 各路一条流(ds4_gpu_lanes_fork; 不可用/关掉/逐段计时时串行), 汇合后才投影出; 捕获态里 fork/join 事件成图的分支 */
        const int lanes = (ok && g_ds4_v41_lanes && !prof && nm > 1u) ? ds4_gpu_lanes_fork((int)nm) : 0;
        for (uint32_t i = 0; ok && i < nm; i++) {
            if (lanes && !ds4_gpu_lane_begin((int)i)) ok = false;
            if (ok && !v41_attn_cache(e, m[i], il)) ok = false;
            if (lanes) ds4_gpu_lane_end();
            if (prof) tp = mp_tick(&P.cache[i], tp);
        }
        if (lanes && !ds4_gpu_lanes_join()) ok = false;
        if (ok && !v41_attn_out(e, B, il)) ok = false;
        if (prof) tp = mp_tick(&P.attn_out, tp);
        if (ok && !ds4_gpu_v41_hc_post_tensor(B->hc2, B->attn_out, B->hc, B->post, B->comb, E, HC, R)) ok = false;
        { ds4_gpu_tensor *t = B->hc; B->hc = B->hc2; B->hc2 = t; }
        if (ok && !v41_hc_half(e, B, l, false)) ok = false;
        if (prof) tp = mp_tick(&P.hc, tp);
        if (ok && !v41_moe(md, l, B, il)) ok = false;
        if (prof) tp = mp_tick(&P.moe, tp);
        if (ok && !ds4_gpu_v41_hc_post_tensor(B->hc2, B->y, B->hc, B->post, B->comb, E, HC, R)) ok = false;
        { ds4_gpu_tensor *t = B->hc; B->hc = B->hc2; B->hc2 = t; }
        if (prof) tp = mp_tick(&P.hc, tp);
    }
    /* 出口: h = hc_pre(hc, pre_mix) → norm → head(R 行, 解码同款 q4_K GEMV) */
    if (ok && !ds4_gpu_v41_hc_pre_tensor(B->x, B->hc, B->pre_mix, E, HC, R)) ok = false;
    if (ok && !ds4_gpu_v41_rms_norm_tensor(B->xn, B->x, md->map, md->size, e->weights.output_norm->abs_offset, E, R, DS4_RMS_EPS)) ok = false;
    if (ok && !v41_tproj(md, B->logits, e->weights.output, E, DS4_N_VOCAB, B->xn, R, 0)) ok = false;
    B->last_logit_row = R - 1u;
    if (prof) {
        tp = mp_tick(&P.exit, tp);
        double cache = 0; for (uint32_t i = 0; i < nm; i++) cache += P.cache[i];
        fprintf(stderr, "[multi-prof] R=%u ms: embed %.1f engram %.1f hc %.1f 注意力投影进 %.1f 缓存段 %.1f(每路", R,
                P.embed * 1e3, P.engram * 1e3, P.hc * 1e3, P.attn_in * 1e3, cache * 1e3);
        for (uint32_t i = 0; i < nm; i++) fprintf(stderr, " %.1f", P.cache[i] * 1e3);
        fprintf(stderr, ") 投影出 %.1f MoE %.1f 出口 %.1f | 合计 %.1f\n", P.attn_out * 1e3, P.moe * 1e3, P.exit * 1e3,
                (P.embed + P.engram + P.hc + P.attn_in + cache + P.attn_out + P.moe + P.exit) * 1e3);
    }
    return ok;
}

/* 一步结束后各路的账(直发与走图同一份): mainh 环连续段、位置推进、暖身计数 */
void v41_multi_advance(ds4_v41_state *st, uint32_t n) {
    if (st->mainh && st->mainh_wrote) {   /* 与 v41_forward 收尾同式: 与上一段连续就接着数, 断了就从头数 */
        const int64_t first = (int64_t)st->pos0 + (int64_t)(n - st->mainh_wrote);
        const bool contig = st->mainh_n > 0 && st->mainh_end == first - 1;
        const uint32_t nn = contig ? st->mainh_n + st->mainh_wrote : st->mainh_wrote;
        st->mainh_n = nn > st->mainh_cap ? st->mainh_cap : nn;
        st->mainh_end = (int64_t)st->pos0 + (int64_t)n - 1;
    }
    st->n_past += n;
    if (n == 1u) st->n_direct1++;
    if (n < DS4_MTP_MAX_BLOCK + 2u) st->n_direct_n[n]++;
}

bool v41_multi_step(ds4_engine *e, ds4_v41_batch *b, ds4_v41_state **m, const int32_t *tok, const uint32_t *rows, uint32_t nm) {
    ds4_v41_state *B = &b->rows;
    const uint32_t HC = DS4_N_HC;
    uint32_t r0[DS4_V41_GEMV_MAX_TOK], nr[DS4_V41_GEMV_MAX_TOK], R = 0;
    if (!nm || nm > b->cap) { fprintf(stderr, "ds4: V4.1 合批 %u 路超批态 %u\n", nm, b->cap); return false; }
    for (uint32_t i = 0; i < nm; i++) { nr[i] = rows ? rows[i] : 1u; r0[i] = R; R += nr[i]; }
    if (!R || R > b->cap) { fprintf(stderr, "ds4: V4.1 合批 %u 行超批态 %u\n", R, b->cap); return false; }
    int32_t posv[DS4_V41_GEMV_MAX_TOK];
    v41_rowview rv[DS4_V41_GEMV_MAX_TOK];
    memset(rv, 0, sizeof rv);
    for (uint32_t i = 0; i < nm; i++) {
        if (m[i]->n_past + nr[i] > m[i]->ctx) { fprintf(stderr, "ds4: V4.1 合批第 %u 路上下文满(%u)\n", i, m[i]->ctx); return false; }
        if (m[i]->hc || m[i]->cap_tok < nr[i]) { fprintf(stderr, "ds4: V4.1 合批第 %u 路不是收缩后的请求态或行数 %u 超 cap %u\n", i, nr[i], m[i]->cap_tok); return false; }
        for (uint32_t j = 0; j < nr[i]; j++) { posv[r0[i] + j] = (int32_t)(m[i]->n_past + j); m[i]->hist[m[i]->n_past + j] = tok[r0[i] + j]; }
    }
    bool ok = ds4_gpu_tensor_write(B->tok, 0, tok, (uint64_t)R * 4) && ds4_gpu_tensor_write(B->pos, 0, posv, (uint64_t)R * 4) &&
              ds4_gpu_tensor_write(B->pre_mix, 0, b->onehot, (uint64_t)R * HC * 4);
    for (uint32_t i = 0; ok && i < nm; i++) {
        if (!v41_attach(m[i], B, r0[i], nr[i], &rv[i])) ok = false;
        else { m[i]->graph = 0; if (!m[i]->no_engram && !v41_engram_prefetch(e, m[i])) ok = false; }   /* 各路自己的行: 盘读与前几层的 GPU 算重叠 */
    }
    if (ok && ds4_gpu_begin_commands() == 0) ok = false;
    if (ok) ok = v41_multi_body(e, b, m, r0, nr, nm, R);
    if (ok && (ds4_gpu_end_commands() == 0 || ds4_gpu_synchronize() == 0)) ok = false;
    for (uint32_t i = 0; i < nm; i++) { v41_detach(m[i], &rv[i]); if (ok) v41_multi_advance(m[i], nr[i]); }
    if (!ok) fprintf(stderr, "ds4: V4.1 合批前向失败(%u 路 %u 行)\n", nm, R);
    return ok;
}

/* 第 row 行按请求 m 的采样面取 token: rowbuf 非 NULL = 主机惩罚路(读回整行), 否则设备采样核 / argmax(温 0) */
bool v41_multi_pick(ds4_v41_batch *b, ds4_v41_state *m, uint32_t row, float *rowbuf, uint64_t *rng, v41_hist *h, int32_t *out) {
    ds4_v41_state *B = &b->rows;
    B->dev_sample = m->dev_sample; B->samp = m->samp; B->psamp = m->psamp;
    return v41_next_token(B, b->am, row, rowbuf, rng, h, out);
}

/* 请求 m 的 rows 行(批态第 row0 行起)一次出 want[rows](设备采样核: 接受 ⇒ 草稿 / 拒绝 ⇒ 残差; 否则逐行 argmax) —— 与 core_v41_api.c 验证批同一口径 */
bool v41_multi_pick_rows(ds4_v41_batch *b, ds4_v41_state *m, uint32_t row0, uint32_t rows, const int32_t *batch, int32_t *want) {
    ds4_v41_state *B = &b->rows;
    B->dev_sample = m->dev_sample; B->samp = m->samp; B->psamp = m->psamp; B->spec_q = m->spec_q;
    return v41_device_next(B, b->am, row0, rows, batch, want);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_multi_nonempty_tu;
