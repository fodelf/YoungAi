/* diag=2：重复短训固定 TF32；只用 FP32 量因子贡献，避免低精度分解误差掩盖读数。
 * diag=3 只做一步数值核验：固定默认 TF32 前向/反传，同一更新额外用 FP32 分解，不跑评分集。
 * diag=4 重建一步后的状态，在训练doc6的L15/序列行471检查激活方向；不改冻结权重。
 * diag=5 核对实际 Adam 更新前后的同批目标，确认激活局部失配是否真的导致跨步过大。
 * diag=6：原写法重复两次，再测 Hadamard/倒数缩放；前向等价过闸才反传，单步且不存盘。
 * diag=7：原 Adam 两臂，再做两臂 LoRA-Pro X=0 梯度单步，训练输入匹配有效更新幅度。
 * 每臂恢复同一份 GPU 权重快照并清空 Adam；只选每篇开头一个完整短段，
 * 因而不能把这里的结果说成长上下文成绩。独立留出只评分，不反传。
 * 所有矩阵计算复用生产 GPU 原语；退出时恢复快照，不保存 checkpoint。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define OC_DOCS 8u
typedef struct {
    const pt_sample *s;
    uint32_t index;
    int32_t *base, *first;
} oc_sample;

static void oc_free(ds4_gpu_tensor **t) { if (*t) ds4_gpu_tensor_free(*t); *t = NULL; }

static bool oc_restore(const pt_cfg *c, pt_run *r, ds4_gpu_tensor **a, ds4_gpu_tensor **b) {
    const uint64_t n = (uint64_t)r->K * DS4_N_EMBD;
    for (uint32_t l = c->layer_lo; l <= c->layer_hi; l++) {
        if (!ds4_gpu_tensor_copy(r->st.ampA[l], 0, a[l], 0, n * 4) ||
            !ds4_gpu_tensor_copy(r->st.ampB[l], 0, b[l], 0, n * 4)) return false;
        ds4_gpu_tensor *z[] = {r->gA[l], r->gB[l], r->mA[l], r->mB[l], r->vA[l], r->vB[l]};
        for (unsigned k = 0; k < sizeof z / sizeof z[0]; k++)
            if (!ds4_gpu_tensor_fill_f32(z[k], 0.f, n)) return false;
    }
    r->cont_s = NULL; r->grad_tokens = 0;
    pt_state_reset(&r->st);
    return true;
}

/* top-K 和 KL 两个核都会覆盖 logits；榜首必须从副本取，原件留给 KL。 */
static bool oc_score(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s,
                     int32_t *ids, double *loss, uint32_t *nt) {
    const uint32_t off = s->l0, m = s->m - off, row = s->sa0 - 1u + off;
    r->cont_s = NULL; r->grad_tokens = 0;
    pt_state_reset(&r->st); r->st.ced_skip = 0; r->st.head_last_only = 0;
    if (!v41_forward(e, &r->st, s->sids, s->sn)) return false;
    bool copied = true; const uint64_t rowb = (uint64_t)DS4_N_VOCAB * 4;
    ds4_gpu_tensor *scores = v41_alloc((uint64_t)m * rowb, &copied);
    if (copied) copied = ds4_gpu_tensor_copy(scores, 0, r->st.logits, (uint64_t)row * rowb, (uint64_t)m * rowb) &&
                         ds4_gpu_bwd_topk_tensor(r->tid, r->tp, r->trest, scores, 0, m, DS4_N_VOCAB, 1) &&
                         ds4_gpu_tensor_read(r->tid, 0, ids, (uint64_t)m * 4);
    oc_free(&scores);
    if (!copied) return false;
    ds4_gpu_tensor *g = NULL;
    *loss = 0; *nt = 0;
    const bool ok = pt_loss_rows(c, r, r->st.logits, row, s, off, loss, nt, &g);
    oc_free(&g);
    return ok && *nt == m;
}

static bool oc_eval(ds4_engine *e, const pt_cfg *c, pt_run *r, oc_sample *q, uint32_t nq,
                    const char *stage, const char *split, int mode, FILE *f, FILE *rows) {
    uint64_t total = 0, hit = 0, repaired = 0, broken = 0, peer = 0;
    double sum = 0;
    if (!pt_loss_mode(c, false)) return false;
    for (uint32_t k = 0; k < nq; k++) {
        const pt_sample *s = q[k].s;
        int32_t *id = xmalloc((size_t)s->m * 4);
        double loss = 0; uint32_t n = 0, h = 0, fix = 0, bad = 0, diff = 0;
        const bool ok = oc_score(e, c, r, s, id, &loss, &n);
        if (!ok) { free(id); return false; }
        if (mode == 0) memcpy(q[k].base, id, (size_t)n * 4);
        for (uint32_t j = 0; j < n; j++) {
            const size_t o = (size_t)(j + s->l0) * c->topk;
            uint32_t best = 0;
            for (uint32_t z = 1; z < c->topk; z++) if (s->top_p[o + z] > s->top_p[o + best]) best = z;
            const int32_t t = s->top_id[o + best];
            const int now = id[j] == t, before = q[k].base[j] == t;
            h += now; fix += now && !before; bad += !now && before;
            if (mode == 2) diff += id[j] != q[k].first[j];
            fprintf(rows, "%s\t%s\t%u\t%u\t%d\t%d\t%d\n", stage, split, s->doc, j, t, id[j], q[k].base[j]);
        }
        if (mode == 1) memcpy(q[k].first, id, (size_t)n * 4);
        fprintf(f, "%s\t%s\t%u\t%u\t%u\t%u\t%u\t%u\t%.12g\t%u\n",
                stage, split, s->doc, q[k].index, n, h, fix, bad, loss, diff);
        total += n; hit += h; repaired += fix; broken += bad; peer += diff; sum += loss;
        free(id);
    }
    fflush(f); fflush(rows);
    fprintf(stderr, "[optcheck] %s %s n=%llu same=%llu (%.6f%%) fixed=%llu broken=%llu KL=%.9f peer_flips=%llu\n",
            stage, split, (unsigned long long)total, (unsigned long long)hit, total ? 100.0 * hit / total : 0,
            (unsigned long long)repaired, (unsigned long long)broken, total ? sum / total : 0, (unsigned long long)peer);
    return pt_loss_mode(c, true);
}

static bool oc_score_check(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s) {
    int32_t *id = xmalloc((size_t)s->m * 4); double a = 0, b = 0; uint32_t na = 0, nb = 0;
    bool ok = pt_loss_mode(c, false) && oc_score(e, c, r, s, id, &a, &na);
    r->cont_s = NULL;
    if (ok) ok = pt_step_sample(e, c, r, s, 0, &b, &nb);
    if (ok) ok = na == nb && na && fabs(a / na - b / nb) <= 1e-6;
    fprintf(stderr, "[optcheck] score_check diag_KL=%.12g native_KL=%.12g %s\n", na ? a / na : 0, nb ? b / nb : 0, ok ? "PASS" : "FAIL");
    free(id);
    return pt_loss_mode(c, true) && ok;
}

/* 保持训练的排序项和装包方式，只关闭反传；不选学习率，也不按评分集调参数。 */
static bool oc_objective(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *const *ss,
                         uint32_t n, uint32_t expected, double *mean) {
    double loss = 0; uint32_t nt = 0; r->cont_s = NULL;
    if (!pt_run_list(e, c, r, ss, n, 0, &loss, &nt, NULL, NULL, NULL) || !nt || nt != expected) return false;
    *mean = loss / nt;
    return isfinite(*mean);
}

/* 先排除有限精度让“等价参数”变成不同模型；失败的分支跳过，不能靠放宽阈值放行。 */
static bool oc_basis_check(ds4_engine *e, const pt_cfg *c, pt_run *r, oc_sample *tr, uint32_t nt,
                           unsigned arm, double original, bool *pass) {
    uint64_t flips = 0, total = 0;
    for (uint32_t k = 0; k < nt; k++) {
        int32_t *id = xmalloc((size_t)tr[k].s->m * 4); double loss = 0; uint32_t n = 0;
        bool ok = oc_score(e, c, r, tr[k].s, id, &loss, &n);
        if (ok) for (uint32_t j = 0; j < n; j++) flips += id[j] != tr[k].base[j];
        free(id); if (!ok) return false; total += n;
    }
    const pt_sample *ss[OC_DOCS];
    for (uint32_t j = 0; j < c->batch; j++) ss[j] = tr[j].s;
    const uint32_t expected = (uint32_t)pt_batch_loss_tokens(r, ss, c->batch);
    double pre = 0, repeat = 0;
    if (!oc_objective(e, c, r, ss, c->batch, expected, &pre) ||
        !oc_objective(e, c, r, ss, c->batch, expected, &repeat)) return false;
    *pass = !flips && fabs(pre - original) <= 1e-7 && fabs(pre - repeat) <= 1e-7;
    fprintf(stderr, "[geometry] arm=%u pre_gate flips=%llu/%llu objective=%.12g original=%.12g difference=%.12g repeat=%.12g %s\n",
            arm, (unsigned long long)flips, (unsigned long long)total, pre, original, fabs(pre - original), fabs(pre - repeat), *pass ? "PASS" : "SKIP");
    return true;
}

static bool oc_frozen(const pt_cfg *c, pt_run *r, ds4_gpu_tensor **a, ds4_gpu_tensor **b) {
    const size_t n = (size_t)r->k2 * DS4_N_EMBD * 4;
    if (!n) return true;
    unsigned char *x = xmalloc(n), *y = xmalloc(n); bool ok = true;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *w[] = {r->st.ampA[l], r->st.ampB[l]}, *z[] = {a[l], b[l]};
        for (unsigned k = 0; ok && k < 2; k++) ok = ds4_gpu_tensor_read(w[k], 0, x, n) &&
            ds4_gpu_tensor_read(z[k], 0, y, n) && !memcmp(x, y, n);
    }
    free(x); free(y); fprintf(stderr, "[geometry] frozen_prefix %s\n", ok ? "PASS" : "FAIL"); return ok;
}

/* 固定一步更新前的真实层输入，拆解 Δy = xBΔA + xΔBA + xΔBΔA。
 * 这是局部输出贡献，不是端到端准确率的因果证明。冻结的前 k2 行差为零。 */
static int oc_contrib(const pt_cfg *c, pt_run *r, ds4_gpu_tensor **a, ds4_gpu_tensor **b,
                       uint32_t nrow, unsigned arm, FILE *f) {
    bool ok = true, identity_ok = true;
    const uint64_t n = (uint64_t)r->K * DS4_N_EMBD, ny = (uint64_t)nrow * DS4_N_EMBD;
    ds4_gpu_tensor *da = v41_alloc(n * 4, &ok), *db = v41_alloc(n * 4, &ok);
    ds4_gpu_tensor *ya = v41_alloc(ny * 4, &ok), *yb = v41_alloc(ny * 4, &ok), *yc = v41_alloc(ny * 4, &ok);
    ds4_gpu_tensor *tmp = v41_alloc((uint64_t)nrow * r->K * 4, &ok), *yf = v41_alloc(ny * 4, &ok);
    double qa = 0, qb = 0, qs = 0, qc = 0;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        if (!r->save.moe_in[l]) { fprintf(stderr, "[optcheck] L%u 没有保存层输入，不能量输出贡献\n", l); ok = false; break; }
        double va = 0, vb = 0, vs = 0, vc = 0;
        ok = ds4_gpu_tensor_copy(da, 0, r->st.ampA[l], 0, n * 4) && ds4_gpu_bwd_axpy_tensor(da, a[l], -1.f, n) &&
             ds4_gpu_tensor_copy(db, 0, r->st.ampB[l], 0, n * 4) && ds4_gpu_bwd_axpy_tensor(db, b[l], -1.f, n) &&
             ds4_gpu_tensor_fill_f32(ya, 0.f, ny) && ds4_gpu_tensor_fill_f32(yb, 0.f, ny) && ds4_gpu_tensor_fill_f32(yc, 0.f, ny) &&
             ds4_gpu_v41_amp_apply_tensor(ya, r->save.moe_in[l], da, b[l], tmp, nrow, DS4_N_EMBD, r->K) &&
             ds4_gpu_v41_amp_apply_tensor(yb, r->save.moe_in[l], a[l], db, tmp, nrow, DS4_N_EMBD, r->K) &&
             ds4_gpu_v41_amp_apply_tensor(yc, r->save.moe_in[l], da, db, tmp, nrow, DS4_N_EMBD, r->K) &&
             ds4_gpu_bwd_sumsq_tensor(ya, ny, &va) && ds4_gpu_bwd_sumsq_tensor(yb, ny, &vb) &&
             ds4_gpu_bwd_sumsq_tensor(yc, ny, &vc) && ds4_gpu_bwd_axpy_tensor(ya, yb, 1.f, ny) &&
             ds4_gpu_bwd_sumsq_tensor(ya, ny, &vs);
        double whole = 0, err = 0, frozen_a = 0, frozen_b = 0;
        if (ok && r->k2) ok = ds4_gpu_bwd_sumsq_tensor(da, (uint64_t)r->k2 * DS4_N_EMBD, &frozen_a) &&
                              ds4_gpu_bwd_sumsq_tensor(db, (uint64_t)r->k2 * DS4_N_EMBD, &frozen_b) && frozen_a == 0 && frozen_b == 0;
        /* 冻结大背景数学上抵消，浮点相减却会损失有效位；参照只乘可训练后缀。 */
        ds4_gpu_tensor *an = pt_rows(r->st.ampA[l], r->k2, c->rank, DS4_N_EMBD), *bn = pt_rows(r->st.ampB[l], r->k2, c->rank, DS4_N_EMBD);
        ds4_gpu_tensor *ao = pt_rows(a[l], r->k2, c->rank, DS4_N_EMBD), *bo = pt_rows(b[l], r->k2, c->rank, DS4_N_EMBD);
        if (ok) ok = an && bn && ao && bo && ds4_gpu_tensor_fill_f32(yf, 0.f, ny) && ds4_gpu_tensor_fill_f32(yb, 0.f, ny) &&
                     ds4_gpu_v41_amp_apply_tensor(yf, r->save.moe_in[l], an, bn, tmp, nrow, DS4_N_EMBD, c->rank) &&
                     ds4_gpu_v41_amp_apply_tensor(yb, r->save.moe_in[l], ao, bo, tmp, nrow, DS4_N_EMBD, c->rank) &&
                     ds4_gpu_bwd_axpy_tensor(yf, yb, -1.f, ny) && ds4_gpu_bwd_sumsq_tensor(yf, ny, &whole) &&
                     ds4_gpu_bwd_axpy_tensor(yf, ya, -1.f, ny) && ds4_gpu_bwd_axpy_tensor(yf, yc, -1.f, ny) &&
                     ds4_gpu_bwd_sumsq_tensor(yf, ny, &err);
        const double relative = whole > 0 ? sqrt(err / whole) : err > 0 ? -1.0 : 0;
        if (ok) { fprintf(f, "%u\t%u\t%.12g\t%.12g\t%.12g\t%.12g\t%.9g\t%.12g\t%.12g\n", arm, l, va, vb, vs, vc, relative, whole, err); qa += va; qb += vb; qs += vs; qc += vc; }
        oc_free(&an); oc_free(&bn); oc_free(&ao); oc_free(&bo);
        if (ok && (relative < 0 || relative > 0.01)) { fprintf(stderr, "[optcheck] L%u 输出分解误差 %.6g 不在 [0,1%%]，负值代表零参照却有残差\n", l, relative); identity_ok = false; }
        if (!ok) fprintf(stderr, "[optcheck] L%u 贡献核验失败 frozen_A=%.9g frozen_B=%.9g\n", l, frozen_a, frozen_b);
    }
    if (ok && identity_ok && qa > 0 && qb > 0) {
        const double ratio = sqrt(qa / qb), aa = 1.0 / sqrt(ratio), bb = sqrt(ratio);
        const double dot2 = qs - qa - qb, predicted = aa * aa * qa + bb * bb * qb + aa * bb * dot2;
        const double match = predicted > 0 ? sqrt(qs / predicted) : 0;
        fprintf(stderr, "[optcheck] contribution arm=%u A/B=%.9g cross/sum=%.9g match_lrA=%.9g match_lrB=%.9g\n",
                arm, ratio, qs > 0 ? sqrt(qc / qs) : 0, c->lr * aa * match, (c->lr_b > 0 ? c->lr_b : c->lr) * bb * match);
    }
    fflush(f);
    oc_free(&da); oc_free(&db); oc_free(&ya); oc_free(&yb); oc_free(&yc); oc_free(&tmp); oc_free(&yf);
    return !ok ? -1 : identity_ok ? 1 : 0;
}

bool pt_optcheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    if (!c->init[0] || !c->evalf[0] || c->evalkl[0] || c->data_eval || !c->max_steps || c->max_steps > 32 || c->batch > OC_DOCS ||
        !(c->lr > 0) || !isfinite(c->lr) || c->lr_b < 0 || !isfinite(c->lr_b)) {
        fprintf(stderr, "[optcheck] 要 init、独立 eval、max_steps=1..32、batch<=8；不混用 eval_kl/data_eval\n"); return false;
    }
    ds4_gpu_set_quality(false);
    oc_sample tr[OC_DOCS] = {0}, ev[OC_DOCS] = {0}; uint32_t nt = 0, ne = 0;
    for (uint32_t i = 0; i < d->ns; i++) {
        const pt_sample *s = &d->s[i];
        if (!s->top_id || !s->raw || s->sn > r->st.cap_tok || s->m <= s->l0 || d->ch[s->chunk].hold || (s->eval != 0 && s->eval != 2)) continue;
        oc_sample *q = s->eval ? ev : tr; uint32_t *nq = s->eval ? &ne : &nt;
        bool seen = false;
        for (uint32_t j = 0; j < *nq; j++) seen |= q[j].s->doc == s->doc;
        if (seen || *nq == OC_DOCS) continue;
        q[*nq].s = s; q[*nq].index = i; (*nq)++;
    }
    if (nt < c->batch || ne < 4) { fprintf(stderr, "[optcheck] 独立短篇不足 train=%u eval=%u\n", nt, ne); return false; }
    if (c->diag == 4 && (nt < 7 || tr[6].s->doc != 6 || c->max_steps != 1 || c->batch != 4)) {
        fprintf(stderr, "[optcheck] 激活诊断固定原料doc6、batch4、max_steps=1；拒绝错位样本\n"); return false;
    }
    for (uint32_t i = 0; i < nt; i++) for (uint32_t j = 0; j < ne; j++)
        if ((tr[i].s->fpkey && tr[i].s->fpkey == ev[j].s->fpkey) ||
            (tr[i].s->sn == ev[j].s->sn && !memcmp(tr[i].s->sids, ev[j].s->sids, (size_t)tr[i].s->sn * 4))) {
            fprintf(stderr, "[optcheck] 训练/留出重合，拒绝验证\n"); return false;
        }
    char p[1200]; snprintf(p, sizeof p, "%s/optcheck.tsv", c->out); FILE *f = fopen(p, "w");
    snprintf(p, sizeof p, "%s/optcheck_rows.tsv", c->out); FILE *rows = fopen(p, "w");
    snprintf(p, sizeof p, "%s/optcheck_components.tsv", c->out); FILE *comp = fopen(p, "w");
    snprintf(p, sizeof p, "%s/optcheck_components_fp32.tsv", c->out); FILE *precise = c->diag == 3 ? fopen(p, "w") : NULL;
    if (!f || !rows || !comp || (c->diag == 3 && !precise)) { if (f) fclose(f); if (rows) fclose(rows); if (comp) fclose(comp); if (precise) fclose(precise); return false; }
    fprintf(f, "stage\tsplit\tdoc\tsample\tn\tcorrect\tfixed\tbroken\tkl_sum\tpeer_flips\n");
    fprintf(rows, "stage\tsplit\tdoc\tpos\tteacher\tstudent\tbase\n");
    fprintf(comp, "arm\tlayer\tA_sumsq\tB_sumsq\tsum_sumsq\tcross_sumsq\tidentity_rel_error\treference_sumsq\terror_sumsq\n");
    if (precise) fprintf(precise, "arm\tlayer\tA_sumsq\tB_sumsq\tsum_sumsq\tcross_sumsq\tidentity_rel_error\treference_sumsq\terror_sumsq\n");
    ds4_gpu_tensor *a[DS4_MAX_LAYER] = {0}, *b[DS4_MAX_LAYER] = {0};
    bool ok = true, contribution_ok = true; const uint64_t bytes = (uint64_t)r->K * DS4_N_EMBD * 4;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        a[l] = v41_alloc(bytes, &ok); b[l] = v41_alloc(bytes, &ok);
        if (ok) ok = ds4_gpu_tensor_copy(a[l], 0, r->st.ampA[l], 0, bytes) && ds4_gpu_tensor_copy(b[l], 0, r->st.ampB[l], 0, bytes);
    }
    for (unsigned group = 0; group < 2; group++) {
        oc_sample *q = group ? ev : tr; uint32_t nq = group ? ne : nt;
        for (uint32_t j = 0; j < nq; j++) { q[j].base = xmalloc((size_t)q[j].s->m * 4); q[j].first = xmalloc((size_t)q[j].s->m * 4); }
    }
    fprintf(stderr, "[optcheck] train_docs=%u eval_docs=%u steps=%u lrA=%.9g lrB=%.9g；每篇只取开头一段\n", nt, ne, c->max_steps, c->lr, c->lr_b > 0 ? c->lr_b : c->lr);
    fprintf(stderr, "[optcheck] 训练/评分固定 TF32；贡献测量 %s；恒等式阈值仍为 1%%\n", c->diag == 3 ? "TF32 与 FP32" : "FP32");
    pt_geometry *geo = c->diag >= 6 ? pt_geometry_new(c, r) : NULL;
    double original = 0, reference_norm = 0;
    if (c->diag >= 6 && !geo) ok = false;
    if (ok) ok = oc_score_check(e, c, r, tr[0].s);
    if (ok && c->diag != 3) ok = oc_eval(e, c, r, tr, nt, "base", "train", 0, f, rows) && oc_eval(e, c, r, ev, ne, "base", "eval", 0, f, rows);
    if (ok && geo) {
        const pt_sample *ss[OC_DOCS]; for (uint32_t j = 0; j < c->batch; j++) ss[j] = tr[j].s;
        ok = oc_objective(e, c, r, ss, c->batch, (uint32_t)pt_batch_loss_tokens(r, ss, c->batch), &original);
    }
    for (unsigned arm = 1; ok && arm <= (geo ? 4u : 2u); arm++) {
        ok = oc_restore(c, r, a, b);
        if (ok && geo) {
            bool pass = false;
            if (c->diag == 6) ok = pt_geometry_basis(c, r, geo, arm) && pt_geometry_equivalence(c, r, geo, arm);
            if (ok) ok = oc_frozen(c, r, a, b) && oc_basis_check(e, c, r, tr, nt, arm, original, &pass);
            if (ok && !pass) { if (arm <= 2) ok = false; else continue; }
        }
        char tag[24]; snprintf(tag, sizeof tag, "run%u_pre", arm);
        if (ok && c->diag != 3) ok = oc_eval(e, c, r, ev, ne, tag, "eval", 3, f, rows);
        for (uint32_t step = 1; ok && step <= (c->diag == 3 ? 1u : c->max_steps); step++) {
            const pt_sample *ss[OC_DOCS];
            for (uint32_t j = 0; j < c->batch; j++) ss[j] = tr[((step - 1u) * c->batch + j) % nt].s;
            double loss = 0, gn = 0; uint32_t n = 0; int clipped = 0;
            r->cont_s = NULL;
            ok = pt_run_list(e, c, r, ss, c->batch, 1, &loss, &n, NULL, NULL, NULL);
            if (ok && (c->diag == 5 || geo)) {
                double zero = 0;
                ok = n && oc_objective(e, c, r, ss, c->batch, n, &zero) && fabs(zero - loss / n) <= 1e-7;
                fprintf(stderr, "[stepcheck] arm=%u step=%u pre_repeat_error=%.12g %s\n", arm, step, n ? fabs(zero - loss / n) : -1., ok ? "PASS" : "FAIL");
            }
            /* 用同一训练篇的更新前输入量贡献；这趟只前向，保留刚累计的 gA/gB。 */
            if (ok && step == 1) {
                /* 多层反传平时只存 hc 并重算 xn；诊断这趟临时留 xn，否则 moe_in 是 NULL。 */
                for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++)
                    if (!r->save.moe_in[l]) r->save.moe_in[l] = v41_alloc((uint64_t)r->st.cap_tok * DS4_N_EMBD * 4, &ok);
                int32_t *id = xmalloc((size_t)tr[0].s->m * 4); double v = 0; uint32_t m = 0;
                if (ok) ok = oc_score(e, c, r, tr[0].s, id, &v, &m);
                free(id);
            }
            if (ok && geo) ok = pt_geometry_before(c, r, geo, tr[0].s->sn, arm);
            if (ok) ok = c->diag == 7 && arm > 2 ? pt_geometry_project(c, r, geo, &gn) : pt_opt_step(c, r, step, &gn, &clipped);
            fprintf(stderr, "[optcheck] arm=%u step=%u loss=%.9f gnorm=%.9g clip=%d mem=%ld\n", arm, step, n ? loss / n : 0, gn, clipped, pt_mem_mb());
            if (ok && geo) {
                double norm = 0, matched = 0;
                ok = pt_geometry_measure(c, r, geo, arm, false, &norm) && oc_frozen(c, r, a, b);
                if (arm == 1) reference_norm = norm;
                if (ok && arm > 2) {
                    double raw_loss = 0;
                    ok = oc_objective(e, c, r, ss, c->batch, n, &raw_loss) && pt_geometry_rescale(c, r, geo, norm) &&
                         pt_geometry_measure(c, r, geo, arm, true, &matched) && oc_frozen(c, r, a, b);
                    const bool match_ok = reference_norm > 0 && fabs(matched / reference_norm - 1) <= .01;
                    if (!match_ok) contribution_ok = false;
                    fprintf(stderr, "[geometry] arm=%u raw_post_loss=%.12g matched_norm_ratio=%.12g %s\n", arm, raw_loss,
                            reference_norm > 0 ? matched / reference_norm : 0, match_ok ? "PASS" : "FAIL");
                }
            }
            if (ok && step == 1 && !geo) {
                if (c->diag != 3) ds4_gpu_set_quality(true);
                const int result = oc_contrib(c, r, a, b, tr[0].s->sn, arm, comp);
                ds4_gpu_set_quality(false); /* 仅测量用 FP32，不能改变下一步训练或评分。 */
                if (result < 0) ok = false;
                if (result <= 0) contribution_ok = false; /* 恒等式未过仍完成独立的 A/A 指标，不通过双学习率准入。 */
                if (ok && precise) {
                    ds4_gpu_set_quality(true);
                    const int fp32 = oc_contrib(c, r, a, b, tr[0].s->sn, arm, precise);
                    ds4_gpu_set_quality(false); /* 必须恢复，后面的训练/评估仍走原 TF32 路。 */
                    fprintf(stderr, "[optcheck] 同一输入/同一更新: TF32=%d FP32=%d；FP32诊断不改变准入状态\n", result, fp32);
                    if (fp32 < 0) ok = false;
                }
            }
            if (step == 1 && c->layer_lo < c->layer_hi)
                for (uint32_t l = c->layer_lo; l <= c->layer_hi; l++) oc_free(&r->save.moe_in[l]);
            if (ok && step == 1 && c->diag == 4) ok = pt_actcheck(e, c, r, tr[6].s, 15, 471, arm);
            if (ok && (c->diag == 5 || geo)) {
                double post = 0, repeat = 0;
                ok = oc_objective(e, c, r, ss, c->batch, n, &post) &&
                     oc_objective(e, c, r, ss, c->batch, n, &repeat) && fabs(post - repeat) <= 1e-7;
                fprintf(stderr, "[stepcheck] arm=%u step=%u before=%.12g after=%.12g gain=%.12g post_repeat_error=%.12g %s\n",
                        arm, step, loss / n, post, loss / n - post, fabs(post - repeat), ok ? "PASS" : "FAIL");
            }
            if (pt_mem_mb() >= 0 && pt_mem_mb() < 3000) { fprintf(stderr, "[optcheck] 内存低于 3000 MB，停止\n"); ok = false; }
        }
        snprintf(tag, sizeof tag, "run%u_post", arm);
        if (ok && c->diag != 3) ok = oc_eval(e, c, r, tr, nt, tag, "train", arm == 1 ? 1 : 2, f, rows) && oc_eval(e, c, r, ev, ne, tag, "eval", arm == 1 ? 1 : 2, f, rows);
    }
    if (ok) ok = oc_restore(c, r, a, b);
    if (ok && c->diag != 3) ok = oc_eval(e, c, r, ev, ne, "restored", "eval", 3, f, rows);
    for (uint32_t l = c->layer_lo; l <= c->layer_hi; l++) { oc_free(&a[l]); oc_free(&b[l]); }
    for (unsigned group = 0; group < 2; group++) {
        oc_sample *q = group ? ev : tr; uint32_t nq = group ? ne : nt;
        for (uint32_t j = 0; j < nq; j++) { free(q[j].base); free(q[j].first); }
    }
    fclose(f); fclose(rows); fclose(comp);
    if (precise) fclose(precise);
    pt_geometry_free(geo);
    fprintf(stderr, "[optcheck] %s；输出分解%s；没有保存 checkpoint\n", ok ? "重复对照完成" : "失败", ok && contribution_ok ? "通过" : "未过门");
    return ok && contribution_ok;
}

#endif
typedef int ds4_core_ptrain_optcheck_nonempty_tu;
