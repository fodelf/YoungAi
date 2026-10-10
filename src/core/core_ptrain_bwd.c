/* core_ptrain_bwd.c — 后训练 ③ 第八版: 一个样本的前向 + 余量桶 KL + 反传(2026-10-01)。总述见 core_ptrain.h。
 *
 * 第一阶段只训末层(L39)的放大器, 因为末层到出口这一段的反传是闭式的(不穿注意力):
 *   logits = W_head · rms(x_exit),  x_exit = Σ_c pre_f[c]·hc_out[c],  hc_out[c] = post_f[c]·y + Σ_j comb_f[j][c]·hc_mid[j],
 *   y = MoE(xn) + xn·B·A
 * ⇒ g_xn_exit = g_logits·W_head → RMSNorm 反向 → hc_pre 反向 → hc_post 反向得 g_y → 放大器两件的梯度。
 * 只有答案位(教师给了分布的那 m 行)有梯度: 末层之后没有注意力, 别的行对答案位没有影响。
 * 多层模式逐层重算并穿过注意力/MoE, 在 core_ptrain_layer.c, 本文件的出口段不变。 */
#include "core_ptrain.h"
#include "../common/ds4_gr_fnv.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

static uint64_t pt_rng_state;
static double pt_randn(void) {   /* Box-Muller, xorshift64*: 只用于放大器 B 的初值, 种子固定可复现 */
    double u[2];
    for (int i = 0; i < 2; i++) {
        pt_rng_state ^= pt_rng_state >> 12; pt_rng_state ^= pt_rng_state << 25; pt_rng_state ^= pt_rng_state >> 27;
        u[i] = ((double)((pt_rng_state * 2685821657736338717ull) >> 11) + 0.5) / 9007199254740992.0;
    }
    return sqrt(-2.0 * log(u[0])) * cos(6.283185307179586 * u[1]);
}

bool pt_run_alloc(ds4_engine *e, const pt_cfg *c, pt_run *r) {
    (void)e;
    memset(r, 0, sizeof *r);
    const uint32_t cap = c->maxlen, K = c->rank, E = DS4_N_EMBD, HC = DS4_N_HC;
    const bool multi = c->layer_lo < DS4_N_LAYER - 1u;   /* 末层以外也训 ⇒ 逐层反传(core_ptrain_layer.c) */
    if (!v41_state_alloc(&r->st, cap, cap + 8u, 0)) return false;   /* logits 每个位置都要(答案位那几行算 KL) */
    r->K = K;
    bool ok = true;
    const uint64_t nel = (uint64_t)K * E;
    float *buf = xmalloc((size_t)nel * 4);
    pt_rng_state = 0x9E3779B97F4A7C15ull ^ ((uint64_t)c->seed * 0xBF58476D1CE4E5B9ull);
    const float std = c->init_std > 0.f ? c->init_std : (float)(1.0 / sqrt((double)E));
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        if (r->st.ampA[il]) { fprintf(stderr, "ds4: [ptrain] ② 在 L%u 已有放大器, 本版不接按秩拼接训练\n", il); ok = false; break; }
        /* A = 0(初态 ③ 恒等, 起点就是部署态), B ~ N(0, std²): 与 LoRA 一样一边置零一边随机, 否则两边梯度都是 0 */
        r->st.ampA[il] = v41_alloc(nel * 4, &ok); r->st.ampB[il] = v41_alloc(nel * 4, &ok);
        r->gA[il] = v41_alloc(nel * 4, &ok); r->gB[il] = v41_alloc(nel * 4, &ok);
        r->mA[il] = v41_alloc(nel * 4, &ok); r->vA[il] = v41_alloc(nel * 4, &ok);
        r->mB[il] = v41_alloc(nel * 4, &ok); r->vB[il] = v41_alloc(nel * 4, &ok);
        if (!ok) break;
        for (uint64_t t = 0; t < nel; t++) buf[t] = (float)(pt_randn() * std);
        ok = ds4_gpu_tensor_fill_f32(r->st.ampA[il], 0.f, nel) && ds4_gpu_tensor_write(r->st.ampB[il], 0, buf, nel * 4) &&
             ds4_gpu_tensor_fill_f32(r->gA[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(r->gB[il], 0.f, nel) &&
             ds4_gpu_tensor_fill_f32(r->mA[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(r->vA[il], 0.f, nel) &&
             ds4_gpu_tensor_fill_f32(r->mB[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(r->vB[il], 0.f, nel);
        r->st.ampK[il] = K;
        if (!multi) r->save.moe_in[il] = v41_alloc((uint64_t)cap * E * 4, &ok);
        if (ok && c->init[0]) {   /* 从已有 ③ 读起: 头 {E, K, 1=f32} + A[K][E] + B[K][E](pt_save_amp 的格式), 形状不对就停车, 不静默用随机初值 */
            char p[1200]; snprintf(p, sizeof p, "%s/amp_L%02u.bin", c->init, il);
            FILE *f = fopen(p, "rb");
            int32_t hd[3] = { 0, 0, 0 };
            const bool got = f && fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)E && hd[1] == (int32_t)K && hd[2] == 1 &&
                             fread(buf, 4, nel, f) == nel && ds4_gpu_tensor_write(r->st.ampA[il], 0, buf, nel * 4) &&
                             fread(buf, 4, nel, f) == nel && ds4_gpu_tensor_write(r->st.ampB[il], 0, buf, nel * 4);
            if (f) fclose(f);
            if (!got) { fprintf(stderr, "ds4: [ptrain] init: %s 读不了或形状不对(要 E %u K %u f32)\n", p, E, K); ok = false; break; }
        }
    }
    if (ok && c->init[0]) {   /* 叠加训练只许叠在同一份 ② 上: init 的 ③ 带 base.fnv(pt_save_amp 写的 ② 指纹), 与现挂的 ② 对不上就停车 */
        char p[1200]; snprintf(p, sizeof p, "%s/base.fnv", c->init);
        FILE *f = fopen(p, "r");
        unsigned long long want = 0; unsigned wn = 0, hn = 0;
        if (f && fscanf(f, "%llx %u", &want, &wn) == 2) {
            const uint64_t have = (g_ds4_v41_amp_dir && g_ds4_v41_amp_dir[0]) ? ds4_gr_dir_fnv(g_ds4_v41_amp_dir, DS4_N_LAYER, &hn) : DS4_GR_FNV_SEED;
            if (have != (uint64_t)want || hn != wn) {
                fprintf(stderr, "ds4: ★[ptrain] init 的 ③ %s 解在另一份 ② 上(指纹 %016llx/%u, 现挂 %016llx/%u): 换了底座不许叠加, 要从零重蒸馏★\n",
                        c->init, want, wn, (unsigned long long)have, hn);
                ok = false;
            }
        } else fprintf(stderr, "ds4: [ptrain] init 的 ③ %s 没有 base.fnv, 核对不了底座(放行)\n", c->init);
        if (f) fclose(f);
        if (ok) fprintf(stderr, "ds4: [ptrain] 放大器 A/B 从 %s 读起(L%u..L%u), Adam 动量从零起\n", c->init, c->layer_lo, c->layer_hi);
    }
    if (ok && multi) ok = pt_layer_alloc(c, r);
    if (ok && multi && !c->nopack) ok = pt_pack_alloc(c, r);   /* 合批只接多层模式(只训末层的那条出口反传是按单题写的) */
    free(buf);
    if (ok && !r->st.ampT) r->st.ampT = v41_alloc((uint64_t)cap * K * 4, &ok);
    r->st.tsave = &r->save;
    const uint32_t TK = c->topk;
    r->tid = v41_alloc((uint64_t)cap * TK * 4, &ok); r->tp = v41_alloc((uint64_t)cap * TK * 4, &ok);
    r->trest = v41_alloc((uint64_t)cap * 4, &ok); r->loss = v41_alloc((uint64_t)cap * 4, &ok);
    r->gxn = v41_alloc((uint64_t)cap * E * 4, &ok); r->gx = v41_alloc((uint64_t)cap * E * 4, &ok);
    r->ghc = v41_alloc((uint64_t)cap * HC * E * 4, &ok); r->gpre = v41_alloc((uint64_t)cap * HC * 4, &ok);
    r->gy = v41_alloc((uint64_t)cap * E * 4, &ok);
    r->T = v41_alloc((uint64_t)cap * K * 4, &ok); r->gT = v41_alloc((uint64_t)cap * K * 4, &ok);
    r->loss_h = xmalloc((size_t)cap * 4);
    if (!ok) { fprintf(stderr, "ds4: [ptrain] 训练缓冲分配失败(maxlen %u)\n", cap); pt_run_free(r); }
    return ok;
}

void pt_run_free(pt_run *r) {
    pt_pack_free(r);   /* 槽的视图指着批态的缓冲: 先摘槽再放批态 */
    pt_layer_free(r);
    ds4_gpu_tensor **all[] = { &r->tid, &r->tp, &r->trest, &r->loss, &r->gxn, &r->gx, &r->ghc, &r->gpre, &r->gy, &r->T, &r->gT };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) { ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor **pl[] = { &r->gA[il], &r->gB[il], &r->mA[il], &r->vA[il], &r->mB[il], &r->vB[il], &r->save.moe_in[il], &r->save.hc_in[il], &r->save.pm_in[il] };
        for (size_t i = 0; i < sizeof pl / sizeof pl[0]; i++) if (*pl[i]) { ds4_gpu_tensor_free(*pl[i]); *pl[i] = NULL; }
    }
    r->st.tsave = NULL;
    v41_state_free(&r->st);   /* 放大器 A/B 与 ampT 在状态里, 由 v41_amp_free 一起放 */
    free(r->loss_h); r->loss_h = NULL;
}

/* 损失 = 本题答案位的平均 KL(不带材料的题: 表是 one-hot ⇒ 交叉熵); 梯度再除以批大小 ⇒ 一批 = 各题平均的平均(长答案不压过短答案)。
 * KL 核就地写梯度(读完 logits 就盖掉), 读回损失 = 同步, 之后才能覆盖教师表。 */
bool pt_loss_rows(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *logits, uint32_t row, const pt_sample *s, double *loss, uint32_t *ntok, ds4_gpu_tensor **glog) {
    const uint32_t m = s->m, V = DS4_N_VOCAB, TK = c->topk;
    const float scale = 1.0f / ((float)m * (float)c->batch);
    *glog = pt_rows(logits, row, m, V);
    bool ok = *glog && ds4_gpu_tensor_write(r->tid, 0, s->top_id, (uint64_t)m * TK * 4) && ds4_gpu_tensor_write(r->tp, 0, s->top_p, (uint64_t)m * TK * 4) &&
              ds4_gpu_tensor_write(r->trest, 0, s->top_rest, (uint64_t)m * 4) &&
              ds4_gpu_bwd_kl_topk_tensor(*glog, r->loss, logits, row, m, V, r->tid, r->tp, r->trest, NULL, TK, scale) &&
              ds4_gpu_tensor_read(r->loss, 0, r->loss_h, (uint64_t)m * 4);
    if (ok) { double l = 0; for (uint32_t k = 0; k < m; k++) l += r->loss_h[k]; *loss += l; *ntok += m; }
    return ok;
}

bool pt_step_sample(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s, int grad, double *loss_sum, uint32_t *ntok) {
    ds4_v41_state *st = &r->st;
    const ds4_model *mdl = &e->model;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, V = DS4_N_VOCAB, K = r->K, m = s->m, r0 = s->sa0 - 1u;
    if (!s->top_id || s->sn > st->cap_tok) return true;   /* 教师没给表(续算块超长)的题不进 */
    const double q0 = pt_tick(c);
    pt_state_reset(st);
    st->ced_skip = 0; st->head_last_only = 0;
    if (!v41_forward(e, st, s->sids, s->sn)) return false;
    const double q1 = pt_tick(c);
    ds4_gpu_tensor *glog = NULL;
    bool ok = pt_loss_rows(c, r, st->logits, r0, s, loss_sum, ntok, &glog);
    const double q2 = pt_tick(c);
    double q3 = q2, q4 = q2;
    if (ok && grad && c->layer_lo < DS4_N_LAYER - 1u) {
        /* 多层: 出口梯度铺回整段(答案位以外的行为 0) → RMSNorm → hc_pre 得末层输出处的 g_hc / g_pre, 再逐层往下 */
        const uint32_t n = s->sn;
        ds4_gpu_tensor *gv = pt_rows(r->gxn, r0, m, E);
        ok = gv && ds4_gpu_tensor_fill_f32(r->gxn, 0.f, (uint64_t)n * E) &&
             ds4_gpu_bwd_matmul_t_tensor(gv, mdl->map, mdl->size, e->weights.output->type, e->weights.output->abs_offset, E, V, glog, m, 0) &&
             ds4_gpu_bwd_rms_norm_tensor(r->gx, r->gxn, st->x, mdl->map, mdl->size, e->weights.output_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
             ds4_gpu_tensor_fill_f32(r->ghc, 0.f, (uint64_t)n * HC * E) &&
             ds4_gpu_bwd_hc_pre_tensor(r->ghc, r->gpre, r->gx, st->hc, st->pre_mix, E, HC, n);
        q3 = pt_tick(c);
        if (ok) ok = pt_layers_bwd(e, c, r, n);
        q4 = pt_tick(c);
        if (gv) ds4_gpu_tensor_free(gv);
    } else if (ok && grad) {
        const uint32_t il = DS4_N_LAYER - 1u;
        ds4_gpu_tensor *x = pt_rows(st->x, r0, m, E), *hc = pt_rows(st->hc, r0, m, (uint64_t)HC * E), *hcm = pt_rows(st->hc2, r0, m, (uint64_t)HC * E);
        ds4_gpu_tensor *pre = pt_rows(st->pre_mix, r0, m, HC), *post = pt_rows(st->post, r0, m, HC), *comb = pt_rows(st->comb, r0, m, (uint64_t)HC * HC);
        ds4_gpu_tensor *y = pt_rows(st->y, r0, m, E), *xin = pt_rows(r->save.moe_in[il], r0, m, E);
        ok = x && hc && hcm && pre && post && comb && y && xin &&
             ds4_gpu_bwd_matmul_t_tensor(r->gxn, mdl->map, mdl->size, e->weights.output->type, e->weights.output->abs_offset, E, V, glog, m, 0) &&
             ds4_gpu_bwd_rms_norm_tensor(r->gx, r->gxn, x, mdl->map, mdl->size, e->weights.output_norm->abs_offset, E, m, DS4_RMS_EPS, 0) &&
             ds4_gpu_tensor_fill_f32(r->ghc, 0.f, (uint64_t)m * HC * E) &&
             ds4_gpu_bwd_hc_pre_tensor(r->ghc, r->gpre, r->gx, hc, pre, E, HC, m) &&
             ds4_gpu_bwd_hc_post_tensor(r->gy, NULL, NULL, NULL, r->ghc, y, hcm, post, comb, E, HC, m) &&
             ds4_gpu_bwd_amp_tensor(r->gA[il], r->gB[il], NULL, r->gy, xin, st->ampA[il], st->ampB[il], r->T, r->gT, m, E, K);
        ds4_gpu_tensor *vs[] = { x, hc, hcm, pre, post, comb, y, xin };
        for (size_t i = 0; i < sizeof vs / sizeof vs[0]; i++) if (vs[i]) ds4_gpu_tensor_free(vs[i]);
        q3 = q4 = pt_tick(c);   /* 只训末层: 出口到放大器是闭式的一段, 整段记在出口反传里 */
    }
    if (glog) ds4_gpu_tensor_free(glog);
    if (c->prof && grad) {   /* 只记训练题(评估/探针不算), 各段与整题墙钟对账: 段和 ≈ 墙钟, 差的就是段外的开销 */
        r->tm[PT_TM_FWD] += q1 - q0; r->tm[PT_TM_LOSS] += q2 - q1; r->tm[PT_TM_EXIT] += q3 - q2; r->tm[PT_TM_LAYERS] += q4 - q3;
        r->tm[PT_TM_Q] += pt_tick(c) - q0; r->tm_n++;
    }
    if (!ok) fprintf(stderr, "ds4: [ptrain] 样本前向/反传失败(长 %u, 答案 %u 行)\n", s->sn, m);
    return ok;
}

bool pt_save_amp(const pt_cfg *c, pt_run *r, const char *dir) {
    mkdir(dir, 0755);
    const uint32_t E = DS4_N_EMBD, K = r->K;
    const uint64_t nel = (uint64_t)K * E;
    float *buf = xmalloc((size_t)nel * 4);
    bool ok = true;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        char p[1200]; snprintf(p, sizeof p, "%s/amp_L%02u.bin", dir, il);
        FILE *f = fopen(p, "wb");
        if (!f) { ok = false; break; }
        const int32_t hd[3] = { (int32_t)E, (int32_t)K, 1 };   /* 1 = f32(训练产物原值落盘; 要省盘再走 fp4x32) */
        ok = fwrite(hd, 4, 3, f) == 3;
        if (ok) ok = ds4_gpu_tensor_read(r->st.ampA[il], 0, buf, nel * 4) && fwrite(buf, 4, nel, f) == nel;
        if (ok) ok = ds4_gpu_tensor_read(r->st.ampB[il], 0, buf, nel * 4) && fwrite(buf, 4, nel, f) == nel;
        fclose(f);
    }
    free(buf);
    if (ok) {   /* 底座指纹: ③ 是解在这一份 ② 上的, 引擎挂 ③ 时重算比对(core_v41_amp.c v41_pt_base_ok) */
        unsigned n = 0;
        const uint64_t h = (g_ds4_v41_amp_dir && g_ds4_v41_amp_dir[0]) ? ds4_gr_dir_fnv(g_ds4_v41_amp_dir, DS4_N_LAYER, &n) : DS4_GR_FNV_SEED;
        char p[1200]; snprintf(p, sizeof p, "%s/base.fnv", dir);
        FILE *f = fopen(p, "w");
        ok = f && fprintf(f, "%016llx %u\n", (unsigned long long)h, n) > 0;
        if (f) fclose(f);
    }
    if (!ok) fprintf(stderr, "ds4: [ptrain] ③ 落盘失败 %s\n", dir);
    return ok;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_bwd_nonempty_tu;
