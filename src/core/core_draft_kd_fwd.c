/* core_draft_kd_fwd.c — 草稿器蒸馏: 训练缓冲、批量教师强制前向、损失与陪审团(2026-10-07)。总述见 core_draft_kd.h。
 *
 * 一批 = 文本里连续 nb 个块(块 i 吃 ids[i]), 展成 R = nb·B 行一次过三塔: 第 r 行 = 第 r/B 块的第 r%B 位, token = [ids[i], noise×(B−1)],
 * 位置 i..i+B−1, 与 v41_draft_block 一字不差(官方 draft_input_ids); 窗口 = 块前面 ≤ window 个位置的 main_x 经各塔 kv 投影(v41_draft_push_main
 * 那串核, 批量算一次全放 hist[T], 块注意力按块首位切), 块内全可见。出口 = hc_pre → out_norm → (出口件) → 头 → 第 r 行 += markov_head(ids[i+j])
 * (教师强制: 第 j 位之前的草稿假定全被接受 = 真 token), 教师 = 底座第 i+j 行的分布。
 * ★与部署同一串函数★: 塔内核全是 v41_draft_attention / v41_moe / v41_hc_half 那几发(只有注意力换成按块切窗口的批量版, 值逐位同);
 * 所以件为零时, 任何位置的首位 logits 与 dcap 取料路给出的 q 只差专家核的累加序(n>8 走预填 GEMM 路), 陪审团首位应复现 accjury。 */
#include "core_draft_kd.h"
#include "../common/ds4_quantfmt.h"
#include "../common/ds4_float.h"
#ifndef DS4_NO_GPU

static float v41_theta0(void) { return DS4_ROPE_FREQ_BASE; }
static bool dk_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t rows, uint32_t n_head, uint32_t hd, bool inverse) {
    return ds4_gpu_v41_rope_tensor(x, pos, rows, n_head, hd, DS4_N_ROT, v41_theta0(), 0u, DS4_ROPE_SCALE_FACTOR,
                                   DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, inverse) != 0;
}
/* markov 头/embed 盘上是 f32 还是 bf16, 按 GGUF 登记类型认(core_v41_draft.c 2026-09-15 实撞: 按 bf16 读 f32 不报错, 接受率 0.14/5) */
static bool dk_small_matmul(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n) {
    if (w->type == DS4_GGT_BF16) return ds4_gpu_v41_matmul_bf16_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    if (w->type == DS4_GGT_F32) return ds4_gpu_v41_matmul_f32_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    fprintf(stderr, "ds4: [dk] %.*s 的类型 %u 不是 f32/bf16\n", (int)w->name.len, w->name.ptr, w->type);
    return false;
}
static double dk_tick(const dk_cfg *c) { if (c->prof) (void)ds4_gpu_synchronize(); return c->prof ? now_sec() : 0.0; }

static uint64_t dk_rng;
static double dk_randn(void) {   /* Box-Muller, xorshift64*: 只给件的 B 初值, 种子固定可复现(与 core_ptrain_bwd.c 同式) */
    double u[2];
    for (int i = 0; i < 2; i++) {
        dk_rng ^= dk_rng >> 12; dk_rng ^= dk_rng << 25; dk_rng ^= dk_rng >> 27;
        u[i] = ((double)((dk_rng * 2685821657736338717ull) >> 11) + 0.5) / 9007199254740992.0;
    }
    return sqrt(-2.0 * log(u[0])) * cos(6.283185307179586 * u[1]);
}
/* markov 表上设备(f32): 缺省从 GGUF 原件复制(f32 原样 / bf16 展开), init 非空时从 <init>/<name> 读({rows, R, 1} + f32) */
static bool dk_table_load(const ds4_model *m, const ds4_tensor *w, uint64_t rows, uint32_t R, const char *init, const char *name, ds4_gpu_tensor **out) {
    const uint64_t nel = rows * R;
    float *buf = xmalloc((size_t)nel * 4);
    bool ok = true;
    if (init && init[0]) {
        char p[1200]; snprintf(p, sizeof p, "%s/%s", init, name);
        FILE *f = fopen(p, "rb"); int32_t hd[3] = { 0, 0, 0 };
        ok = f && fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)rows && hd[1] == (int32_t)R && hd[2] == 1 && fread(buf, 4, nel, f) == nel;
        if (f) fclose(f);
        if (!ok) fprintf(stderr, "ds4: [dk] init: %s 读不了或形状不对(要 %llu×%u f32)\n", p, (unsigned long long)rows, R);
    } else if (w->type == DS4_GGT_F32) memcpy(buf, (const uint8_t *)m->map + w->abs_offset, (size_t)nel * 4);
    else if (w->type == DS4_GGT_BF16) { const uint16_t *src = (const uint16_t *)((const uint8_t *)m->map + w->abs_offset); for (uint64_t i = 0; i < nel; i++) buf[i] = ds4_bf16_to_f32(src[i]); }
    else { fprintf(stderr, "ds4: [dk] %.*s 的类型 %u 不是 f32/bf16\n", (int)w->name.len, w->name.ptr, w->type); ok = false; }
    if (ok) { *out = v41_alloc(nel * 4, &ok); if (ok) ok = ds4_gpu_tensor_write(*out, 0, buf, nel * 4) != 0; }
    free(buf);
    return ok;
}
/* 一件: A = 0(初态恒等, 起点就是部署态), B ~ N(0, std²); init 非空时从 <init>/<name> 读({E,K,1} + A + B) */
static bool dk_amp_init(const dk_cfg *c, ds4_gpu_tensor **A, ds4_gpu_tensor **B, uint32_t K, const char *name, float *buf) {
    const uint32_t E = DS4_N_EMBD; const uint64_t nel = (uint64_t)K * E;
    bool ok = true;
    *A = v41_alloc(nel * 4, &ok); *B = v41_alloc(nel * 4, &ok);
    if (!ok) return false;
    const float std = c->init_std > 0.f ? c->init_std : (float)(1.0 / sqrt((double)E));
    for (uint64_t t = 0; t < nel; t++) buf[t] = (float)(dk_randn() * std);
    ok = ds4_gpu_tensor_fill_f32(*A, 0.f, nel) && ds4_gpu_tensor_write(*B, 0, buf, nel * 4);
    if (ok && c->init[0]) {
        char p[1200]; snprintf(p, sizeof p, "%s/%s", c->init, name);
        FILE *f = fopen(p, "rb"); int32_t hd[4] = { 0, 0, 0, 0 };
        bool got = f && fread(hd, 4, 3, f) == 3;
        if (got && !memcmp(hd, "DSPA", 4)) got = fread(hd + 3, 4, 1, f) == 1 && hd[1] == (int32_t)E && hd[2] == (int32_t)K;   /* 出口件的 DSPA 头 {magic, D, K, rsv} */
        else got = got && hd[0] == (int32_t)E && hd[1] == (int32_t)K && hd[2] == 1;
        if (got) got = fread(buf, 4, nel, f) == nel && ds4_gpu_tensor_write(*A, 0, buf, nel * 4) && fread(buf, 4, nel, f) == nel && ds4_gpu_tensor_write(*B, 0, buf, nel * 4);
        if (f) fclose(f);
        if (!got) { fprintf(stderr, "ds4: [dk] init: %s 读不了或形状不对(要 E %u K %u)\n", p, E, K); return false; }
    }
    return ok;
}

bool dk_run_alloc(ds4_engine *e, const dk_cfg *c, dk_run *r) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    memset(r, 0, sizeof *r);
    if (!v->mtp_towers || !v->mtp_block || !v->n_mtp_target || !e->weights.mtp.main_proj || !e->weights.mtp.markov_embd || !e->weights.mtp.out_norm) {
        fprintf(stderr, "ds4: [dk] 这份 GGUF 没带 DSpark 三塔, 训不了\n"); return false;
    }
    for (uint32_t T = 0; T < v->mtp_towers; T++) if (!e->weights.mtp.exps_vq[T]) { fprintf(stderr, "ds4: [dk] 塔 %u 的专家不是 VQ blob, 反传只接 blob 形态\n", T); return false; }
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA, FF = DS4_N_FF_EXP;
    const uint32_t B = v->mtp_block, NT = v->mtp_towers, R = c->batch_blocks * B, Rk = v->mtp_markov_rank, V = DS4_N_VOCAB, K = c->topk;
    const uint64_t low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, mix = 2u * HC + HC * HC;
    ds4_v41_state *st = &r->st;
    r->R = R; r->B = B; r->NT = NT; r->K = c->rank; r->Kx = c->rank_exit;
    r->Lr = c->logits_rows && c->logits_rows < R ? c->logits_rows : R;
    r->hist_rows = SWA + c->batch_blocks;
    st->draft = 1; st->cap_tok = R; st->ctx = R; st->idx_owner = -1; st->cand_owner = -1;
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) st->eshard[i].fd = -1;
    bool ok = true;
    st->tok = v41_alloc((uint64_t)R * 4, &ok);          st->pos = v41_alloc((uint64_t)R * 4, &ok);
    st->hc = v41_alloc((uint64_t)R * HC * E * 4, &ok);  st->hc2 = v41_alloc((uint64_t)R * HC * E * 4, &ok);
    st->mix = v41_alloc((uint64_t)R * mix * 4, &ok);    st->pre = v41_alloc((uint64_t)R * HC * 4, &ok);
    st->post = v41_alloc((uint64_t)R * HC * 4, &ok);    st->comb = v41_alloc((uint64_t)R * HC * HC * 4, &ok);
    st->pre_mix = v41_alloc((uint64_t)R * HC * 4, &ok);
    st->x = v41_alloc((uint64_t)R * E * 4, &ok);        st->xn = v41_alloc((uint64_t)R * E * 4, &ok);
    st->qr = v41_alloc((uint64_t)R * Q * 4, &ok);       st->qrn = v41_alloc((uint64_t)R * Q * 4, &ok);
    st->q = v41_alloc((uint64_t)R * NH * HD * 4, &ok);  st->kv = v41_alloc((uint64_t)R * HD * 4, &ok);
    st->kvn = v41_alloc((uint64_t)R * HD * 4, &ok);     st->o = v41_alloc((uint64_t)R * NH * HD * 4, &ok);
    st->low = v41_alloc((uint64_t)R * low * 4, &ok);    st->attn_out = v41_alloc((uint64_t)R * E * 4, &ok);
    st->glog = v41_alloc((uint64_t)R * v->mtp_experts * 4, &ok);
    st->sel = v41_alloc((uint64_t)R * v->mtp_used * 4, &ok);  st->rw = v41_alloc((uint64_t)R * v->mtp_used * 4, &ok);
    st->routed = v41_alloc((uint64_t)R * E * 4, &ok);
    st->sg = v41_alloc((uint64_t)R * FF * 4, &ok);      st->su = v41_alloc((uint64_t)R * FF * 4, &ok);
    st->sh = v41_alloc((uint64_t)R * FF * 4, &ok);      st->so = v41_alloc((uint64_t)R * E * 4, &ok);
    st->y = v41_alloc((uint64_t)R * E * 4, &ok);
    st->logits_rows = r->Lr; st->logits = v41_alloc((uint64_t)r->Lr * V * 4, &ok);
    st->tsave = &r->save;
    for (uint32_t T = 0; T < NT; T++) {
        st->tower_exps_vq[T] = e->weights.mtp.exps_vq[T];
        r->save.sel[T] = v41_alloc((uint64_t)R * v->mtp_used * 4, &ok);   /* 各塔路由选择的存档(梯度检查冻结用; 每次前向都存, 一次 1.9 KB 拷贝) */
        r->hist[T] = v41_alloc((uint64_t)r->hist_rows * HD * 4, &ok);
        r->lse[T] = v41_alloc((uint64_t)R * NH * 4, &ok);
        r->hc_in[T] = v41_alloc((uint64_t)R * HC * E * 4, &ok); r->pm_in[T] = v41_alloc((uint64_t)R * HC * 4, &ok);
    }
    r->mx_dev = v41_alloc((uint64_t)r->hist_rows * E * 4, &ok); r->posh = v41_alloc((uint64_t)r->hist_rows * 4, &ok);
    r->bpos = v41_alloc((uint64_t)c->batch_blocks * 4, &ok);
    r->h = v41_alloc((uint64_t)R * E * 4, &ok); r->xn_pre = v41_alloc((uint64_t)R * E * 4, &ok);
    r->g_xn = v41_alloc((uint64_t)R * E * 4, &ok); r->g_xn_pre = v41_alloc((uint64_t)R * E * 4, &ok); r->g_h = v41_alloc((uint64_t)R * E * 4, &ok);
    r->ghc = v41_alloc((uint64_t)R * HC * E * 4, &ok); r->gpre = v41_alloc((uint64_t)R * HC * 4, &ok);
    r->mk_ids = v41_alloc((uint64_t)R * 4, &ok); r->mk_emb = v41_alloc((uint64_t)R * Rk * 4, &ok); r->mk_bias = v41_alloc((uint64_t)r->Lr * V * 4, &ok);
    r->tid = v41_alloc((uint64_t)R * K * 4, &ok); r->tp = v41_alloc((uint64_t)R * K * 4, &ok); r->trest = v41_alloc((uint64_t)R * 4, &ok);
    r->loss = v41_alloc((uint64_t)R * 4, &ok); r->tlog_dev = v41_alloc((uint64_t)r->Lr * V * 4, &ok); r->jury = v41_alloc((uint64_t)r->Lr * 4 * 4, &ok);
    r->loss_h = xmalloc((size_t)R * 4); r->jury_h = xmalloc((size_t)r->Lr * 16); r->bpos_h = xmalloc((size_t)c->batch_blocks * 4); r->mk_h = xmalloc((size_t)R * 4);
    r->mx_h = xmalloc((size_t)r->hist_rows * E * 4);
    r->tt_id = xmalloc((size_t)(c->batch_blocks + B) * K * 4); r->tt_p = xmalloc((size_t)(c->batch_blocks + B) * K * 4); r->tt_rest = xmalloc((size_t)(c->batch_blocks + B) * 4);
    r->tl_rows = r->Lr / B + B + 2u; r->tl_h = xmalloc((size_t)r->tl_rows * V * 4);   /* 一片 Lr 行的教师行号只跨 Lr/B + B − 1 个 */
    /* 件: 塔件挂草稿态 st.ampA/B[T](v41_moe 的草稿分支应用), 出口件单放; Adam 动量从零起 */
    const uint32_t Kmax = r->K > r->Kx ? r->K : r->Kx;
    float *buf = xmalloc((size_t)Kmax * E * 4 + 16);
    dk_rng = 0x9E3779B97F4A7C15ull ^ ((uint64_t)c->seed * 0xBF58476D1CE4E5B9ull);
    if (ok && r->K) for (uint32_t T = 0; T < NT && ok; T++) {
        char nm[64]; snprintf(nm, sizeof nm, "tower_T%u.bin", T);
        ok = dk_amp_init(c, &st->ampA[T], &st->ampB[T], r->K, nm, buf);
        st->ampK[T] = r->K;
        const uint64_t nel = (uint64_t)r->K * E;
        r->gA[T] = v41_alloc(nel * 4, &ok); r->gB[T] = v41_alloc(nel * 4, &ok); r->mA[T] = v41_alloc(nel * 4, &ok); r->vA[T] = v41_alloc(nel * 4, &ok);
        r->mB[T] = v41_alloc(nel * 4, &ok); r->vB[T] = v41_alloc(nel * 4, &ok);
        ds4_gpu_tensor *z[] = { r->gA[T], r->gB[T], r->mA[T], r->vA[T], r->mB[T], r->vB[T] };
        for (size_t i = 0; ok && i < 6; i++) ok = ds4_gpu_tensor_fill_f32(z[i], 0.f, nel) != 0;
    }
    if (ok && r->Kx) {
        ok = dk_amp_init(c, &r->xA, &r->xB, r->Kx, "exit.dspa", buf);
        const uint64_t nel = (uint64_t)r->Kx * E;
        r->gxA = v41_alloc(nel * 4, &ok); r->gxB = v41_alloc(nel * 4, &ok); r->mxA = v41_alloc(nel * 4, &ok); r->vxA = v41_alloc(nel * 4, &ok);
        r->mxB = v41_alloc(nel * 4, &ok); r->vxB = v41_alloc(nel * 4, &ok);
        r->xT = v41_alloc((uint64_t)R * r->Kx * 4, &ok); r->gxT = v41_alloc((uint64_t)R * r->Kx * 4, &ok);
        ds4_gpu_tensor *z[] = { r->gxA, r->gxB, r->mxA, r->vxA, r->mxB, r->vxB };
        for (size_t i = 0; ok && i < 6; i++) ok = ds4_gpu_tensor_fill_f32(z[i], 0.f, nel) != 0;
    }
    free(buf);
    if (ok && Kmax) st->ampT = v41_alloc((uint64_t)R * Kmax * 4, &ok);
    if (ok && r->K) r->gT = v41_alloc((uint64_t)R * r->K * 4, &ok);
    if (ok && c->markov) {   /* 偏置表当训练变量: 起点 = 原件(或 init 的表), 梯度/Adam 各一份同形 */
        const ds4_tensor *we = e->weights.mtp.markov_embd, *wh = e->weights.mtp.markov_head;
        r->Rk = Rk; r->Vm = (uint32_t)(we->ndim > 1 ? we->dim[1] : V);
        ok = wh && dk_table_load(&e->model, we, r->Vm, Rk, c->init, "markov_embd.bin", &r->mkE) && dk_table_load(&e->model, wh, V, Rk, c->init, "markov_head.bin", &r->mkH);
        const uint64_t ne = (uint64_t)r->Vm * Rk, nh = (uint64_t)V * Rk;
        r->gmkE = v41_alloc(ne * 4, &ok); r->mE = v41_alloc(ne * 4, &ok); r->vE = v41_alloc(ne * 4, &ok);
        r->gmkH = v41_alloc(nh * 4, &ok); r->mH = v41_alloc(nh * 4, &ok); r->vH = v41_alloc(nh * 4, &ok);
        r->ge = v41_alloc((uint64_t)r->Lr * Rk * 4, &ok);
        ds4_gpu_tensor *z[] = { r->gmkE, r->mE, r->vE }; ds4_gpu_tensor *zh[] = { r->gmkH, r->mH, r->vH };
        for (size_t i = 0; ok && i < 3; i++) ok = ds4_gpu_tensor_fill_f32(z[i], 0.f, ne) && ds4_gpu_tensor_fill_f32(zh[i], 0.f, nh);
    }
    if (ok) ok = dk_layer_alloc(r);
    if (!ok) { fprintf(stderr, "ds4: [dk] 训练缓冲分配失败(R %u)\n", R); dk_run_free(r); }
    return ok;
}

void dk_run_free(dk_run *r) {
    dk_layer_free(r);
    ds4_gpu_tensor **all[] = { &r->xA, &r->xB, &r->gxA, &r->gxB, &r->mxA, &r->vxA, &r->mxB, &r->vxB, &r->xT, &r->gxT, &r->gT, &r->mx_dev, &r->posh, &r->bpos,
                               &r->mkE, &r->mkH, &r->gmkE, &r->gmkH, &r->mE, &r->vE, &r->mH, &r->vH, &r->ge,
                               &r->h, &r->xn_pre, &r->g_xn, &r->g_xn_pre, &r->g_h, &r->ghc, &r->gpre, &r->mk_ids, &r->mk_emb, &r->mk_bias,
                               &r->tid, &r->tp, &r->trest, &r->loss, &r->tlog_dev, &r->jury };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) { ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t T = 0; T < DS4_MTP_MAX_TOWERS; T++) {
        ds4_gpu_tensor **pt[] = { &r->gA[T], &r->gB[T], &r->mA[T], &r->vA[T], &r->mB[T], &r->vB[T], &r->hist[T], &r->lse[T], &r->hc_in[T], &r->pm_in[T], &r->save.sel[T] };
        for (size_t i = 0; i < sizeof pt / sizeof pt[0]; i++) if (*pt[i]) { ds4_gpu_tensor_free(*pt[i]); *pt[i] = NULL; }
    }
    r->st.tsave = NULL;
    v41_state_free(&r->st);   /* 塔件 A/B 与 ampT 在状态里, v41_amp_free 一起放 */
    free(r->loss_h); free(r->jury_h); free(r->bpos_h); free(r->mk_h); free(r->mx_h); free(r->tt_id); free(r->tt_p); free(r->tt_rest); free(r->tl_h);
    r->loss_h = r->jury_h = r->mx_h = r->tt_p = r->tt_rest = r->tl_h = NULL; r->bpos_h = r->mk_h = r->tt_id = NULL;
}

/* 一塔前向 = v41_layer 的草稿分支, 注意力换批量块核(其余核与 v41_draft_attention 一字不差) */
bool dk_tower_fwd(ds4_engine *e, dk_run *r, uint32_t T) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.mtp.tower[T];
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    if (!v41_hc_half(e, st, l, true)) return false;
    if (!v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1) ||
        !ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS) ||
        !v41_tproj(m, st->q, l->attn_q_b, Q, (uint64_t)NH * HD, st->qrn, n, 1) || !dk_rope(st->q, st->pos, n, NH, HD, false)) return false;
    if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1) ||
        !ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS) ||
        !dk_rope(st->kvn, st->pos, n, 1, HD, false) || !ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) return false;
    if (!ds4_gpu_draft_attn_fwd_tensor(st->o, r->lse[T], st->q, r->hist[T], r->hbase, st->kvn, r->bpos, r->nb, r->B, SWA,
                                       m->map, m->size, l->attn_sinks->abs_offset, NH, HD, (float)(1.0 / sqrt((double)HD)))) return false;
    if (!dk_rope(st->o, st->pos, n, NH, HD, true)) return false;
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP, (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1) ||
        !v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1)) return false;
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, E, HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    if (!v41_hc_half(e, st, l, false) || !v41_moe(m, l, st, T)) return false;
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->y, st->hc, st->post, st->comb, E, HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    return true;
}

/* 本批的窗口: 历史位置 [hbase, i0+nb−1) 的 main_x → 每塔 kv 投影/归一/rope/fp8 格点(v41_draft_push_main 同一串) → hist[T] */
static bool dk_window(ds4_engine *e, dk_run *r, const dk_text *t, uint32_t i0, uint32_t nb) {
    const ds4_model *m = &e->model;
    const uint32_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA;
    const uint32_t hlo = i0 > SWA ? i0 - SWA : 0u, nh = i0 + nb - 1u - hlo;
    r->hbase = hlo; r->nb = nb;
    int32_t *ph = xmalloc((size_t)nh * 4);
    for (uint32_t k = 0; k < nh; k++) ph[k] = (int32_t)(hlo + k);
    bool ok = dk_text_pread(t, t->o_mx + (uint64_t)hlo * E * 4, r->mx_h, (uint64_t)nh * E * 4) &&
              ds4_gpu_tensor_write(r->mx_dev, 0, r->mx_h, (uint64_t)nh * E * 4) && ds4_gpu_tensor_write(r->posh, 0, ph, (uint64_t)nh * 4);
    free(ph);
    for (uint32_t T = 0; ok && T < r->NT; T++) {
        const ds4_layer_weights *l = &e->weights.mtp.tower[T];
        ok = v41_tproj(m, r->st.kv, l->attn_kv, E, HD, r->mx_dev, nh, 1) &&
             ds4_gpu_v41_rms_norm_tensor(r->hist[T], r->st.kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, nh, DS4_RMS_EPS) &&
             dk_rope(r->hist[T], r->posh, nh, 1, HD, false) && ds4_gpu_v41_act_quant_fp8_tensor(r->hist[T], nh, HD, 32);
    }
    return ok;
}

bool dk_batch(ds4_engine *e, const dk_cfg *c, dk_run *r, const dk_text *t, uint32_t i0, uint32_t nb, int grad,
              double *loss_sum, uint32_t *nrows, double *jury, uint32_t *jrows) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t B = r->B, R = nb * B, E = DS4_N_EMBD, HC = DS4_N_HC, V = DS4_N_VOCAB, K = c->topk, Rk = v->mtp_markov_rank;
    if (nb == 0 || nb * B > r->R) return false;
    const double q0 = dk_tick(c);
    st->n = R; st->pos0 = 0;
    if (!dk_window(e, r, t, i0, nb)) return false;
    const double q1 = dk_tick(c);
    /* 块行: token / 位置 / 块首位 / 各行"前一个真 token"(markov) / 教师表行(教师第 i0..i0+nb+B−2 行从缓存连续 pread 一段, 按 b+j 取) */
    const uint32_t ntr = nb + B - 1u;
    if (!dk_text_pread(t, t->o_tid + (uint64_t)i0 * K * 4, r->tt_id, (uint64_t)ntr * K * 4) || !dk_text_pread(t, t->o_tp + (uint64_t)i0 * K * 4, r->tt_p, (uint64_t)ntr * K * 4) ||
        !dk_text_pread(t, t->o_trest + (uint64_t)i0 * 4, r->tt_rest, (uint64_t)ntr * 4)) return false;
    int32_t *tok = xmalloc((size_t)R * 4), *pos = xmalloc((size_t)R * 4);
    int32_t *tid = xmalloc((size_t)R * K * 4); float *tp = xmalloc((size_t)R * K * 4), *tr = xmalloc((size_t)R * 4), *pm = xmalloc((size_t)R * HC * 4);
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t i = i0 + b;
        r->bpos_h[b] = (int32_t)i;
        for (uint32_t j = 0; j < B; j++) {
            const uint32_t rr = b * B + j;
            tok[rr] = j ? (int32_t)v->mtp_noise_id : t->ids[i]; pos[rr] = (int32_t)(i + j); r->mk_h[rr] = t->ids[i + j];
            memcpy(tid + (size_t)rr * K, r->tt_id + (size_t)(b + j) * K, (size_t)K * 4); memcpy(tp + (size_t)rr * K, r->tt_p + (size_t)(b + j) * K, (size_t)K * 4);
            tr[rr] = r->tt_rest[b + j];
        }
    }
    for (uint32_t i = 0; i < R * HC; i++) pm[i] = (i % HC) == 0 ? 1.0f : 0.0f;
    bool ok = ds4_gpu_tensor_write(st->tok, 0, tok, (uint64_t)R * 4) && ds4_gpu_tensor_write(st->pos, 0, pos, (uint64_t)R * 4) &&
              ds4_gpu_tensor_write(r->bpos, 0, r->bpos_h, (uint64_t)nb * 4) && ds4_gpu_tensor_write(r->mk_ids, 0, r->mk_h, (uint64_t)R * 4) &&
              ds4_gpu_tensor_write(r->tid, 0, tid, (uint64_t)R * K * 4) && ds4_gpu_tensor_write(r->tp, 0, tp, (uint64_t)R * K * 4) &&
              ds4_gpu_tensor_write(r->trest, 0, tr, (uint64_t)R * 4) && ds4_gpu_tensor_write(st->pre_mix, 0, pm, (uint64_t)R * HC * 4);
    free(tok); free(pos); free(tid); free(tp); free(tr); free(pm);
    if (!ok) return false;
    if (!v41_embed(m, st->x, st->tok, e->weights.token_embd, V, R, E) || !ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, HC, R)) return false;
    for (uint32_t T = 0; T < r->NT; T++) {   /* 入口存档(反传重算起点)后过塔 */
        if (!ds4_gpu_tensor_copy(r->hc_in[T], 0, st->hc, 0, (uint64_t)R * HC * E * 4) || !ds4_gpu_tensor_copy(r->pm_in[T], 0, st->pre_mix, 0, (uint64_t)R * HC * 4)) return false;
        if (!dk_tower_fwd(e, r, T)) return false;
    }
    const double q2 = dk_tick(c);
    /* 出口: hc_pre → out_norm → (出口件 → 补回 bf16 格点) → 头, 按 Lr 行一片(logits 一行 517 KB) */
    if (!ds4_gpu_v41_hc_pre_tensor(r->h, st->hc, st->pre_mix, E, HC, R) ||
        !ds4_gpu_v41_rms_norm_tensor(r->xn_pre, r->h, m->map, m->size, e->weights.mtp.out_norm->abs_offset, E, R, DS4_RMS_EPS) ||
        !ds4_gpu_tensor_copy(st->xn, 0, r->xn_pre, 0, (uint64_t)R * E * 4)) return false;
    if (r->Kx && (!ds4_gpu_v41_amp_apply_tensor(st->xn, r->xn_pre, r->xA, r->xB, r->xT, R, E, r->Kx) || !ds4_gpu_v41_round_bf16_tensor(st->xn, (uint64_t)R * E))) return false;
    const uint64_t nrow_mk = e->weights.mtp.markov_embd->ndim > 1 ? e->weights.mtp.markov_embd->dim[1] : V;
    const uint32_t eb = e->weights.mtp.markov_embd->type == DS4_GGT_BF16 ? 2u : 4u;
    if (r->mkE ? !ds4_gpu_draft_rows_dev_tensor(r->mk_emb, r->mkE, r->mk_ids, 0u, R, Rk, r->Vm)
               : !ds4_gpu_draft_rows_gather_tensor(r->mk_emb, m->map, m->size, e->weights.mtp.markov_embd->abs_offset, nrow_mk, Rk, eb, r->mk_ids, R)) return false;
    const float scale = 1.0f / (float)R;
    for (uint32_t c0 = 0; c0 < R; c0 += r->Lr) {
        const uint32_t mm = R - c0 < r->Lr ? R - c0 : r->Lr;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(st->xn, (uint64_t)c0 * E * 4, (uint64_t)mm * E * 4), *ev = ds4_gpu_tensor_view(r->mk_emb, (uint64_t)c0 * Rk * 4, (uint64_t)mm * Rk * 4);
        ds4_gpu_tensor *tiv = ds4_gpu_tensor_view(r->tid, (uint64_t)c0 * K * 4, (uint64_t)mm * K * 4), *tpv = ds4_gpu_tensor_view(r->tp, (uint64_t)c0 * K * 4, (uint64_t)mm * K * 4);
        ds4_gpu_tensor *trv = ds4_gpu_tensor_view(r->trest, (uint64_t)c0 * 4, (uint64_t)mm * 4), *lv = ds4_gpu_tensor_view(r->loss, (uint64_t)c0 * 4, (uint64_t)mm * 4);
        ds4_gpu_tensor *gv = grad ? ds4_gpu_tensor_view(r->g_xn, (uint64_t)c0 * E * 4, (uint64_t)mm * E * 4) : NULL;
        ok = xv && ev && tiv && tpv && trv && lv && (!grad || gv) &&
             v41_tproj(m, st->logits, e->weights.output, E, V, xv, mm, 0) &&
             (r->mkH ? ds4_gpu_draft_bias_tensor(r->mk_bias, ev, r->mkH, mm, Rk, V) : dk_small_matmul(m, r->mk_bias, e->weights.mtp.markov_head, Rk, V, ev, mm)) &&
             ds4_gpu_v41_add_tensor(st->logits, r->mk_bias, (uint64_t)mm * V);
        if (ok && jury) {   /* 留出: 全词表教师 logits(这一片涉及的教师行号连续, 从缓存 pread 一段)上传, 逐行陪审团(在损失核改写 logits 之前) */
            const uint32_t tlo = i0 + c0 / B, thi = i0 + (c0 + mm - 1u) / B + B - 1u, nrow = thi - tlo + 1u;
            ok = nrow <= r->tl_rows && tlo >= t->n_prompt &&
                 dk_text_pread(t, t->o_tlog + (uint64_t)(tlo - t->n_prompt) * V * 4, r->tl_h, (uint64_t)nrow * V * 4);
            for (uint32_t k = 0; ok && k < mm; k++) {
                const uint32_t rr = c0 + k, trow = i0 + rr / B + rr % B;
                ok = ds4_gpu_tensor_write(r->tlog_dev, (uint64_t)k * V * 4, r->tl_h + (size_t)(trow - tlo) * V, (uint64_t)V * 4) != 0;
            }
            ok = ok && ds4_gpu_draft_jury_tensor(r->jury, st->logits, r->tlog_dev, mm, V, c->temp) && ds4_gpu_synchronize() &&
                 ds4_gpu_tensor_read(r->jury, 0, r->jury_h, (uint64_t)mm * 16);
            for (uint32_t k = 0; ok && k < mm; k++) { const uint32_t j = (c0 + k) % B; for (int q = 0; q < 3; q++) jury[j * 3 + q] += r->jury_h[k * 4 + q]; jrows[j]++; }
        }
        /* 损失与对 logits 的梯度(同址改写): 总变差(目标 = 1 − 期望接受率)或前向 KL(对照) */
        ok = ok && (c->loss_kl ? ds4_gpu_bwd_kl_topk_tensor(st->logits, lv, st->logits, 0, mm, V, tiv, tpv, trv, NULL, K, scale)
                               : ds4_gpu_draft_tv_tensor(st->logits, lv, st->logits, mm, V, tiv, tpv, trv, K, c->temp, scale)) &&
             ds4_gpu_synchronize() && ds4_gpu_tensor_read(lv, 0, r->loss_h, (uint64_t)mm * 4);
        if (ok) { for (uint32_t k = 0; k < mm; k++) *loss_sum += r->loss_h[k]; *nrows += mm; }
        if (ok && grad) ok = ds4_gpu_bwd_matmul_t_tensor(gv, m->map, m->size, e->weights.output->type, e->weights.output->abs_offset, E, V, st->logits, mm, 0) != 0;
        if (ok && grad && r->mkE) {   /* 偏置表的梯度: 头 += g_logitsᵀ·e; 行 = g_logits·head 按 token 散加进 embd 的梯度 */
            ds4_gpu_tensor *idv = ds4_gpu_tensor_view(r->mk_ids, (uint64_t)c0 * 4, (uint64_t)mm * 4);
            ok = idv && ds4_gpu_draft_bias_bwd_tensor(r->gmkH, r->ge, st->logits, ev, r->mkH, mm, Rk, V) &&
                 ds4_gpu_draft_rows_scatter_tensor(r->gmkE, r->ge, idv, mm, Rk, r->Vm);
            if (idv) ds4_gpu_tensor_free(idv);
        }
        ds4_gpu_tensor *vs[] = { xv, ev, tiv, tpv, trv, lv, gv };
        for (size_t i = 0; i < sizeof vs / sizeof vs[0]; i++) if (vs[i]) ds4_gpu_tensor_free(vs[i]);
        if (!ok) return false;
    }
    const double q3 = dk_tick(c);
    double q4 = q3, q5 = q3;
    if (grad) {
        if (!dk_exit_bwd(e, r)) return false;
        q4 = dk_tick(c);
        if (!dk_towers_bwd(e, r, nb)) return false;
        q5 = dk_tick(c);
    }
    if (c->prof) { r->tm[DK_TM_WIN] += q1 - q0; r->tm[DK_TM_TOWERS] += q2 - q1; r->tm[DK_TM_EXIT] += q3 - q2; r->tm[DK_TM_EXITB] += q4 - q3; r->tm[DK_TM_TOWERB] += q5 - q4; r->tm[DK_TM_BATCH] += q5 - q0; r->tm_n++; }
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_draft_kd_fwd_nonempty_tu;
