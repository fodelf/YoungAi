/* diag=12：固定首篇原文的1022个教师目标，只改变前向行数；diag=13导出1022行梯度作驻留/流式对拍。
 * out/ref_short.topk、out/ref_aligned.topk须来自同权重、同原文的冻结评分程序。
 * 概率复用评分出口：训练GPU top-K与评分CPU归一化的舍入不同，不能混着判前向。
 * 本入口没有优化器或存盘调用；对齐失败仍落盘诊断，但不做反传准入检查。 */
#include "core_ptrain.h"
#include "core_ptrain_split.h"
#include "core_score_aux.h"
#include "../common/ds4_etgd.h"
#ifndef DS4_NO_GPU

enum { SHAPE_TARGETS = 1022, SHAPE_ROWS = 2048 };

/* 构建启用fast-math，不能用可能被折掉的isfinite守诊断门。 */
static bool shape_finite(double value) {
    uint64_t bits; memcpy(&bits, &value, sizeof bits);
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

bool pt_diag_weights(const pt_cfg *c, const pt_run *r) {
    const size_t bytes = (size_t)c->rank * DS4_N_EMBD * 4;
    void *disk = xmalloc(bytes), *gpu = xmalloc(bytes);
    bool ok = true;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        char path[1200]; snprintf(path, sizeof path, "%s/amp_L%02u.bin", c->init, l);
        FILE *f = fopen(path, "rb"); int32_t hd[3];
        ok = f && fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)DS4_N_EMBD && hd[1] == (int32_t)c->rank && hd[2] == 1;
        ds4_gpu_tensor *ab[2] = { r->st.ampA[l], r->st.ampB[l] };
        for (unsigned k = 0; ok && k < 2; k++)
            ok = fread(disk, 1, bytes, f) == bytes &&
                 ds4_gpu_tensor_read(ab[k], (uint64_t)r->k2 * DS4_N_EMBD * 4, gpu, bytes) && !memcmp(disk, gpu, bytes);
        if (f) fclose(f);
    }
    free(disk); free(gpu); return ok;
}

static bool shape_emit(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s,
                       const char *name, FILE *report) {
    pt_state_reset(&r->st); r->cont_s = NULL; r->grad_tokens = 0;
    r->st.ced_skip = 0; r->st.head_last_only = 0;
    if (!pt_loss_mode(c, false) || !v41_forward(e, &r->st, s->sids, s->sn)) return false;
    char path[1200]; snprintf(path, sizeof path, "%s/%s.topk", c->out, name);
    ds4_score_aux *a = ds4_score_aux_open(NULL, path, 2, NULL, SHAPE_TARGETS, DS4_N_VOCAB, name);
    float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
    bool ok = a != NULL;
    for (uint32_t j = 0; ok && j < SHAPE_TARGETS; j++) {
        ok = ds4_gpu_tensor_read(r->st.logits, (uint64_t)j * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4);
        if (ok) ds4_score_aux_row(a, j, row, s->sids[j + 1u]);
    }
    free(row); ds4_score_aux_close(a);
    double loss = 0; uint32_t nt = 0; ds4_gpu_tensor *g = NULL;
    if (ok) ok = pt_loss_rows(c, r, r->st.logits, 0, s, 0, &loss, &nt, &g);
    if (g) ds4_gpu_tensor_free(g);
    fprintf(report, "forward\t%s\t%u\t%u\t%.12g\n", name, s->sn, nt, nt ? loss / nt : -1.0);
    fflush(report);
    return ok && nt == SHAPE_TARGETS && shape_finite(loss);
}

/* ETGD的头可能含整篇8192行；只比较预登记的1022目标，不读取其他位置作选择。 */
static bool shape_compare(const pt_cfg *c, const pt_sample *s, const char *a_name,
                          const char *b_name, FILE *report, bool *equal) {
    char path[1200]; ds4_etgd a = {0}, b = {0};
    snprintf(path, sizeof path, "%s/%s.topk", c->out, a_name);
    bool ok = ds4_etgd_read(path, &a) == 0;
    snprintf(path, sizeof path, "%s/%s.topk", c->out, b_name);
    if (ok) ok = ds4_etgd_read(path, &b) == 0;
    ok = ok && a.K == 2 && b.K == 2 && a.n >= SHAPE_TARGETS && b.n >= SHAPE_TARGETS &&
         a.vocab == (int)DS4_N_VOCAB && b.vocab == (int)DS4_N_VOCAB;
    uint32_t arg = 0, ids = 0, prob = 0, target = 0; double maxdp = 0;
    for (uint32_t j = 0; ok && j < SHAPE_TARGETS; j++) {
        target += a.tgt[j] != s->sids[j + 1u] || b.tgt[j] != s->sids[j + 1u];
        arg += ds4_etgd_top1(&a, j, NULL) != ds4_etgd_top1(&b, j, NULL);
        ids += memcmp(a.ids + j * 2u, b.ids + j * 2u, 8) != 0;
        prob += memcmp(a.ps + j * 2u, b.ps + j * 2u, 8) != 0;
        for (unsigned k = 0; k < 2; k++) {
            if (!shape_finite(a.ps[j * 2u + k]) || !shape_finite(b.ps[j * 2u + k])) ok = false;
            if (a.ids[j * 2u + k] == b.ids[j * 2u + k]) {
                const double dp = fabs((double)a.ps[j * 2u + k] - b.ps[j * 2u + k]);
                if (dp > maxdp) maxdp = dp;
            }
        }
    }
    *equal = ok && !target && !arg && !ids && !prob;
    fprintf(report, "compare\t%s\t%s\ttop1_diff=%u\ttop2_ids_diff=%u\tprob_bits_diff=%u\ttarget_diff=%u\tmax_common_slot_dp=%.12g\tequal=%u\n",
            a_name, b_name, arg, ids, prob, target, maxdp, *equal);
    fflush(report); ds4_etgd_free(&a); ds4_etgd_free(&b);
    return ok && !target;
}

bool pt_diag_gradient(const pt_cfg *c, const pt_run *r, FILE *report) {
    char path[1200]; snprintf(path, sizeof path, "%s/gradient.bin", c->out);
    FILE *f = fopen(path, "wb"); if (!f) return false;
    const uint32_t hd[5] = {0x31445247u, c->layer_lo, c->layer_hi, c->rank, DS4_N_EMBD};
    const uint64_t n = (uint64_t)c->rank * DS4_N_EMBD;
    void *buf = xmalloc(n * 4); double total = 0;
    bool ok = fwrite(hd, sizeof hd, 1, f) == 1;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *ab[2] = {r->gA[l], r->gB[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *v = pt_rows(ab[k], r->k2, c->rank, DS4_N_EMBD); double ss = 0;
            ok = v && ds4_gpu_bwd_sumsq_tensor(v, n, &ss) && shape_finite(ss) &&
                 ds4_gpu_tensor_read(v, 0, buf, n * 4) && fwrite(buf, 4, n, f) == n;
            if (v) ds4_gpu_tensor_free(v);
            total += ss;
        }
    }
    free(buf); if (fclose(f)) ok = false;
    fprintf(report, "gradient\tsumsq=%.12g\tfinite_nonzero=%u\n", total, ok && total > 0);
    return ok && total > 0;
}

bool pt_shapecheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    const uint32_t needed = c->diag == 13 ? SHAPE_TARGETS : SHAPE_ROWS;
    if (c->epochs || c->diag_save || c->max_steps || c->sample_n || !c->init[0] ||
        r->st.cap_tok < needed || c->layer_lo != 0 || c->layer_hi != DS4_N_LAYER - 1u) {
        fprintf(stderr, "[shape] 要epochs=0、禁止保存/采样、maxlen>=%u、全层固定init；配置不符停车\n", needed); return false;
    }
    const pt_sample *source = NULL;
    for (uint32_t i = 0; i < d->ns; i++) {
        const pt_sample *s = &d->s[i];
        if (s->doc == 0 && !s->eval && s->raw && s->top_id && !s->l0 && s->sa0 == 1 && s->m >= SHAPE_TARGETS &&
            d->docids[0] && d->docn[0] > SHAPE_ROWS) { source = s; break; }
    }
    if (!source) { fprintf(stderr, "[shape] 第0篇要保留至少2049个原始ID与开头1022位教师表\n"); return false; }
    pt_sample s = *source;
    s.sids = d->docids[0]; s.m = SHAPE_TARGETS; s.sn = SHAPE_TARGETS;
    char path[1200]; snprintf(path, sizeof path, "%s/shape.tsv", c->out);
    FILE *f = fopen(path, "w"); if (!f) return false;
    bool ok = pt_diag_weights(c, r), control = false, aligned = false, repeat = false, post = false;
    if (ok) ok = shape_emit(e, c, r, &s, "short", f) && shape_compare(c, &s, "ref_short", "short", f, &control);
    if (c->diag == 13) {
        double loss = 0; uint32_t nt = 0;
        r->grad_tokens = SHAPE_TARGETS;
        ok = ok && control && pt_mem_mb() >= 6000 && pt_loss_mode(c, true) && pt_step_sample(e, c, r, &s, 1, &loss, &nt);
        if (ok) ok = nt == SHAPE_TARGETS && shape_finite(loss) && pt_diag_gradient(c, r, f);
        if (ok) ok = shape_emit(e, c, r, &s, "post_bwd", f) && shape_compare(c, &s, "short", "post_bwd", f, &post);
        const bool weights = pt_diag_weights(c, r), pass = ok && post && weights;
        fprintf(f, "verdict\t%s\tcontrol=%u\ttargets=%u\tobjective=%.12g\tpost=%u\tweights_equal=%u\tavailable_MiB=%ld\n",
                pass ? "PASS" : "FAIL", control, nt, nt ? loss / nt : -1.0, post, weights, pt_mem_mb());
        fclose(f); return pass;
    }
    s.sn = SHAPE_ROWS;
    if (ok) ok = shape_emit(e, c, r, &s, "aligned", f) && shape_compare(c, &s, "ref_aligned", "aligned", f, &aligned);
    if (ok) ok = shape_emit(e, c, r, &s, "repeat", f) && shape_compare(c, &s, "aligned", "repeat", f, &repeat);
    if (ok) { bool unused = false; ok = shape_compare(c, &s, "short", "aligned", f, &unused); }
    if (ok && repeat && !aligned) {
        /* 前向仍分叉时只隔离训练存档标志，不把这条无反传路径当作通过。 */
        v41_tsave *saved = r->st.tsave; bool without_save = false;
        r->st.tsave = NULL;
        ok = shape_emit(e, c, r, &s, "without_save", f) && shape_compare(c, &s, "ref_aligned", "without_save", f, &without_save);
        r->st.tsave = saved;
    }
    double tail_ss = -1, loss = 0; uint32_t nt = 0;
    bool backward = false;
    const long before = pt_mem_mb();
    if (ok && control && aligned && repeat && before >= 6000) {
        r->grad_tokens = SHAPE_TARGETS;
        ok = pt_loss_mode(c, true) && pt_step_sample(e, c, r, &s, 1, &loss, &nt);
        /* r->gxn是出口矩阵反传结果，逐层反传不复用它；新增行必须没有出口梯度。 */
        ds4_gpu_tensor *tail = ok ? pt_rows(r->gxn, SHAPE_TARGETS, SHAPE_ROWS - SHAPE_TARGETS, DS4_N_EMBD) : NULL;
        ok = ok && tail && ds4_gpu_bwd_sumsq_tensor(tail, (uint64_t)(SHAPE_ROWS - SHAPE_TARGETS) * DS4_N_EMBD, &tail_ss);
        if (tail) ds4_gpu_tensor_free(tail);
        backward = ok && nt == SHAPE_TARGETS && tail_ss == 0.0 && shape_finite(loss);
        fprintf(f, "backward\ttargets=%u\ttail_rows=%u\ttail_gradient_sumsq=%.12g\tobjective=%.12g\tpass=%u\n",
                nt, SHAPE_ROWS - SHAPE_TARGETS, tail_ss, nt ? loss / nt : -1.0, backward);
        /* 2048准入也保存完整参数梯度，不能只凭范数相同判断重复性。 */
        if (ok && backward) ok = pt_diag_gradient(c, r, f);
        if (ok) ok = shape_emit(e, c, r, &s, "post_bwd", f) && shape_compare(c, &s, "aligned", "post_bwd", f, &post);
    } else fprintf(f, "backward\tSKIP\tforward_gate_or_memory\tavailable_MiB=%ld\n", before);
    const bool weights = pt_diag_weights(c, r);
    const bool pass = ok && control && aligned && repeat && backward && post && weights;
    fprintf(f, "verdict\t%s\tcontrol=%u\taligned=%u\trepeat=%u\tbackward=%u\tpost=%u\tweights_equal=%u\tavailable_MiB=%ld\n",
            pass ? "PASS" : "FAIL", control, aligned, repeat, backward, post, weights, pt_mem_mb());
    fclose(f);
    fprintf(stderr, "[shape] %s：固定%u目标、%u前向行；无优化器步进、无权重存盘\n", pass ? "PASS" : "FAIL", SHAPE_TARGETS, SHAPE_ROWS);
    return pass;
}
#endif
typedef int ds4_core_ptrain_shape_nonempty_tu;
