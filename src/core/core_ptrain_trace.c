/* core_ptrain_trace.c — 末层只读张量对照诊断(100轮): 定位单步缩步非单调跳变的精确阶段。
 * 只读合法训练位(切片B doc3, pos575, 教师769 vs 竞争6946), 不保存候选, 结束后完整恢复父本。 */
#include "core_ptrain_split.h"
#include "core_v41_trace.h"
#ifndef DS4_NO_GPU

v41_trace_tap *g_v41_trace_tap = NULL;

void v41_trace_capture_row(const ds4_gpu_tensor *t, uint32_t row, uint32_t cols, float *dst, bool *ok) {
    if (!*ok || !t || !dst) { *ok = false; return; }
    if (!ds4_gpu_flush_commands() || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(t, (uint64_t)row * cols * 4u, dst, (uint64_t)cols * 4u)) *ok = false;
}

#ifdef __APPLE__
bool pt_trace_run(ds4_engine *e, const pt_cfg *c, pt_run *r, pt_sample *s,
                  ds4_gpu_tensor **m, ds4_gpu_tensor *snap[DS4_MAX_LAYER][2], double radius) {
    (void)e; (void)c; (void)r; (void)s; (void)m; (void)snap; (void)radius; return false;
}
#else
enum { TR_DOC = 3, TR_ROW = 575, TR_TEACHER = 769, TR_COMP = 6946, TR_STAGES = 5 };

typedef struct {
    float *buf[6]; /* 0:y_moe, 1:y_pre, 2:y_post, 3:hc_post, 4:hc_pre, 5:xn_out */
    double l_t, l_c, margin;
    int32_t argmax;
} tr_snap;

typedef struct {
    uint32_t nnz, top_idx;
    double l2, max_abs, cos1, dmargin, top_dv, top_dm;
} tr_stat;

static void tr_free_gpu(ds4_gpu_tensor **p) { if (*p) ds4_gpu_tensor_free(*p); *p = NULL; }
static uint32_t tr_stage_len(unsigned s) { return s == 3 ? DS4_N_HC * DS4_N_EMBD : DS4_N_EMBD; }
static const char *tr_stage_name(unsigned s) {
    static const char *const nm[6] = { "y_moe", "y_pre", "y_post", "hc_post", "hc_pre", "xn_out" };
    return nm[s];
}

static bool tr_snap_alloc(tr_snap *x) {
    memset(x, 0, sizeof *x);
    for (unsigned s = 0; s < 6; s++) x->buf[s] = xmalloc((size_t)tr_stage_len(s) * 4u);
    return true;
}
static void tr_snap_free(tr_snap *x) {
    for (unsigned s = 0; s < 6; s++) { free(x->buf[s]); x->buf[s] = NULL; }
}
static bool tr_snap_equal(const tr_snap *a, const tr_snap *b) {
    if (a->l_t != b->l_t || a->l_c != b->l_c || a->argmax != b->argmax) return false;
    for (unsigned s = 0; s < 6; s++)
        if (memcmp(a->buf[s], b->buf[s], (size_t)tr_stage_len(s) * 4u) != 0) return false;
    return true;
}

static void tr_diff_stats(const float *cur, const float *base, const float *ref1_diff,
                          const float *jac, uint32_t n, tr_stat *out) {
    memset(out, 0, sizeof *out);
    double ss = 0, rr = 0, dr = 0, dm = 0, worst_dm = -1.0;
    for (uint32_t i = 0; i < n; i++) {
        const double dv = (double)cur[i] - (double)base[i];
        if (dv != 0.0) out->nnz++;
        ss += dv * dv;
        const double ad = fabs(dv);
        if (ad > out->max_abs) out->max_abs = ad;
        if (ref1_diff) {
            const double rv = (double)ref1_diff[i];
            rr += rv * rv; dr += dv * rv;
        }
        if (jac) {
            const double c = (double)jac[i] * dv;
            dm += c;
            if (fabs(c) > worst_dm) { worst_dm = fabs(c); out->top_idx = i; out->top_dv = dv; out->top_dm = c; }
        }
    }
    out->l2 = sqrt(ss);
    out->dmargin = dm;
    out->cos1 = (ref1_diff && ss > 0.0 && rr > 0.0) ? (dr / sqrt(ss * rr)) : 0.0;
}

static bool tr_gpu_pair_norm(ds4_gpu_tensor *a[2], uint32_t n, double *norm) {
    double s0 = 0, s1 = 0;
    const bool ok = ds4_gpu_bwd_sumsq_tensor(a[0], n, &s0) && ds4_gpu_bwd_sumsq_tensor(a[1], n, &s1);
    *norm = sqrt(s0 + s1);
    return ok && isfinite(*norm);
}

/* 极化恒等式 (||u_hat+v_hat||^2 - ||u_hat-v_hat||^2)/4: 避免范数舍入偏置。 */
static bool tr_gpu_pair_cos(ds4_gpu_tensor *u[2], ds4_gpu_tensor *v[2], double nu, double nv, uint32_t n, double *cos_out) {
    bool ok = nu > 0.0 && nv > 0.0;
    ds4_gpu_tensor *tmp = v41_alloc((uint64_t)n * 4u, &ok);
    double sp = 0, sm = 0;
    for (unsigned k = 0; ok && k < 2; k++) {
        double qp = 0, qm = 0;
        ok = ds4_gpu_tensor_fill_f32(tmp, 0.f, n) &&
             ds4_gpu_bwd_axpy_tensor(tmp, u[k], (float)(1.0 / nu), n) &&
             ds4_gpu_bwd_axpy_tensor(tmp, v[k], (float)(1.0 / nv), n) &&
             ds4_gpu_bwd_sumsq_tensor(tmp, n, &qp) &&
             ds4_gpu_tensor_fill_f32(tmp, 0.f, n) &&
             ds4_gpu_bwd_axpy_tensor(tmp, u[k], (float)(1.0 / nu), n) &&
             ds4_gpu_bwd_axpy_tensor(tmp, v[k], (float)(-1.0 / nv), n) &&
             ds4_gpu_bwd_sumsq_tensor(tmp, n, &qm);
        sp += qp; sm += qm;
    }
    tr_free_gpu(&tmp);
    *cos_out = (sp - sm) * 0.25;
    return ok && isfinite(*cos_out);
}

static bool tr_fixture(void) {
    const float base[4] = { 1, 2, -3, 4 }, cur[4] = { 2, 2, -1, 1 }, ref1[4] = { 1, 0, 2, -3 }, jac[4] = { .5f, -1, 2, 3 };
    tr_stat st; tr_diff_stats(cur, base, ref1, jac, 4, &st);
    bool ok = st.nnz == 3 && fabs(st.l2 - sqrt(14.0)) < 1e-12 && fabs(st.max_abs - 3.0) < 1e-12 &&
              fabs(st.cos1 - 1.0) < 1e-12 && fabs(st.dmargin - (-4.5)) < 1e-12 &&
              st.top_idx == 3 && fabs(st.top_dv - (-3.0)) < 1e-12 && fabs(st.top_dm - (-9.0)) < 1e-12;
    const float u0[4] = { 3, 4, 0, 0 }, v0[4] = { 4, -3, 0, 0 }, w0[4] = { 6, 8, 0, 0 }, z0[4] = { 0 };
    ds4_gpu_tensor *u[2] = { 0 }, *v[2] = { 0 }, *w[2] = { 0 };
    for (unsigned k = 0; ok && k < 2; k++) {
        u[k] = v41_alloc(16, &ok); v[k] = v41_alloc(16, &ok); w[k] = v41_alloc(16, &ok);
        if (ok) ok = ds4_gpu_tensor_write(u[k], 0, k ? z0 : u0, 16) &&
                     ds4_gpu_tensor_write(v[k], 0, k ? z0 : v0, 16) &&
                     ds4_gpu_tensor_write(w[k], 0, k ? z0 : w0, 16);
    }
    double nu = 0, nv = 0, nw = 0, c_orth = 9, c_par = 9;
    if (ok) ok = tr_gpu_pair_norm(u, 4, &nu) && tr_gpu_pair_norm(v, 4, &nv) && tr_gpu_pair_norm(w, 4, &nw) &&
                 tr_gpu_pair_cos(u, v, nu, nv, 4, &c_orth) && tr_gpu_pair_cos(u, w, nu, nw, 4, &c_par) &&
                 fabs(nu - 5.0) < 1e-6 && fabs(nv - 5.0) < 1e-6 && fabs(nw - 10.0) < 1e-6 &&
                 fabs(c_orth) < 1e-6 && fabs(c_par - 1.0) < 1e-6;
    for (unsigned k = 0; k < 2; k++) { tr_free_gpu(&u[k]); tr_free_gpu(&v[k]); tr_free_gpu(&w[k]); }
    fprintf(stderr, "[trace] known_answer c_orth=%.12g c_par=%.12g dmargin=%.12g %s\n", c_orth, c_par, st.dmargin, ok ? "PASS" : "FAIL");
    return ok;
}

static bool tr_capture(ds4_engine *e, pt_run *r, const pt_sample *s, int skip_amp_bf16, tr_snap *out) {
    pt_state_reset(&r->st); r->cont_s = NULL; r->st.ced_skip = 0; r->st.head_last_only = 0;
    v41_trace_tap tap = {
        .layer = DS4_N_LAYER - 1u, .row = TR_ROW, .skip_amp_bf16 = skip_amp_bf16,
        .y_moe = out->buf[0], .y_pre = out->buf[1], .y_post = out->buf[2],
        .hc_post = out->buf[3], .hc_pre = out->buf[4], .xn_out = out->buf[5], .ok = true
    };
    g_v41_trace_tap = &tap;
    bool ok = v41_forward(e, &r->st, s->sids, s->sn) && tap.ok;
    g_v41_trace_tap = NULL;
    float a = 0, b = 0; int32_t am = -1;
    ds4_gpu_tensor *idx = v41_alloc(4, &ok);
    const uint64_t row_off = (uint64_t)TR_ROW * DS4_N_VOCAB;
    if (ok) ok = ds4_gpu_tensor_read(r->st.logits, (row_off + TR_TEACHER) * 4u, &a, 4) &&
                 ds4_gpu_tensor_read(r->st.logits, (row_off + TR_COMP) * 4u, &b, 4) &&
                 ds4_gpu_v41_argmax_tensor(idx, r->st.logits, TR_ROW, DS4_N_VOCAB) &&
                 ds4_gpu_tensor_read(idx, 0, &am, 4);
    tr_free_gpu(&idx);
    out->l_t = (double)a; out->l_c = (double)b; out->margin = out->l_t - out->l_c; out->argmax = am;
    return ok && isfinite(out->margin);
}

/* 对当前 st 中刚跑完的样本计算单位置 margin = logit[769]-logit[6946] 对出口链各阶段与 L39 参数的精确连续(STE)雅可比。 */
static bool tr_margin_jacobian(ds4_engine *e, const pt_cfg *c, pt_run *r,
                               float *jac[6], ds4_gpu_tensor *gm[2]) {
    const uint32_t il = c->layer_hi, E = DS4_N_EMBD, HC = DS4_N_HC, V = DS4_N_VOCAB, K = r->K;
    const ds4_model *mdl = &e->model;
    ds4_v41_state *st = &r->st;
    bool ok = true;
    ds4_gpu_tensor *glog = v41_alloc((uint64_t)V * 4u, &ok);
    const float pos1 = 1.f, neg1 = -1.f;
    if (ok) ok = ds4_gpu_tensor_fill_f32(glog, 0.f, V) &&
                 ds4_gpu_tensor_write(glog, (uint64_t)TR_TEACHER * 4u, &pos1, 4) &&
                 ds4_gpu_tensor_write(glog, (uint64_t)TR_COMP * 4u, &neg1, 4);
    for (unsigned k = 0; ok && k < 2; k++) ok = ds4_gpu_tensor_fill_f32(gm[k], 0.f, (uint64_t)K * E);
    ds4_gpu_tensor *x = pt_rows(st->x, TR_ROW, 1, E), *hc = pt_rows(st->hc, TR_ROW, 1, (uint64_t)HC * E);
    ds4_gpu_tensor *hcm = pt_rows(st->hc2, TR_ROW, 1, (uint64_t)HC * E);
    ds4_gpu_tensor *pre = pt_rows(st->pre_mix, TR_ROW, 1, HC), *post = pt_rows(st->post, TR_ROW, 1, HC);
    ds4_gpu_tensor *comb = pt_rows(st->comb, TR_ROW, 1, (uint64_t)HC * HC);
    ds4_gpu_tensor *y = pt_rows(st->y, TR_ROW, 1, E), *xin = pt_rows(r->save.moe_in[il], TR_ROW, 1, E);
    if (ok) ok = x && hc && hcm && pre && post && comb && y && xin &&
                 ds4_gpu_bwd_matmul_t_tensor(r->gxn, mdl->map, mdl->size, e->weights.output->type, e->weights.output->abs_offset, E, V, glog, 1, 0) &&
                 ds4_gpu_bwd_rms_norm_tensor(r->gx, r->gxn, x, mdl->map, mdl->size, e->weights.output_norm->abs_offset, E, 1, DS4_RMS_EPS, 0) &&
                 ds4_gpu_tensor_fill_f32(r->ghc, 0.f, (uint64_t)HC * E) &&
                 ds4_gpu_bwd_hc_pre_tensor(r->ghc, r->gpre, r->gx, hc, pre, E, HC, 1) &&
                 ds4_gpu_bwd_hc_post_tensor(r->gy, NULL, NULL, NULL, r->ghc, y, hcm, post, comb, E, HC, 1) &&
                 ds4_gpu_bwd_amp_tensor(gm[0], gm[1], NULL, r->gy, xin, st->ampA[il], st->ampB[il], r->T, r->gT, 1, E, K);
    if (ok && r->k2) {
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *pfx = pt_rows(gm[k], 0, r->k2, E);
            ok = pfx && ds4_gpu_tensor_fill_f32(pfx, 0.f, (uint64_t)r->k2 * E);
            tr_free_gpu(&pfx);
        }
    }
    if (ok) ok = ds4_gpu_tensor_read(r->gy, 0, jac[1], (uint64_t)E * 4u) &&
                 ds4_gpu_tensor_read(r->gy, 0, jac[2], (uint64_t)E * 4u) &&
                 ds4_gpu_tensor_read(r->ghc, 0, jac[3], (uint64_t)HC * E * 4u) &&
                 ds4_gpu_tensor_read(r->gx, 0, jac[4], (uint64_t)E * 4u) &&
                 ds4_gpu_tensor_read(r->gxn, 0, jac[5], (uint64_t)E * 4u);
    ds4_gpu_tensor *vs[] = { glog, x, hc, hcm, pre, post, comb, y, xin };
    for (size_t i = 0; i < sizeof vs / sizeof vs[0]; i++) tr_free_gpu(&vs[i]);
    return ok;
}

typedef struct {
    const char *suite;
    int skip_amp_bf16;
    ds4_gpu_tensor *dir[2];
    double dir_coeff; /* 乘以 scale 即 axpy 系数 */
    const tr_snap *ref_base;
    float *ref1_diff[6];
    float *jac[6];
    double dm_1st_unit; /* scale=1 时的一阶参数线性预测 Δmargin */
    FILE *fsum, *fstg;
} tr_ctx;

static bool tr_eval_scale(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s,
                          ds4_gpu_tensor *snap[DS4_MAX_LAYER][2], const tr_snap *b0,
                          tr_ctx *ctx, double scale, bool save_ref1, tr_snap *cur_out) {
    const uint32_t il = c->layer_hi, n = r->K * DS4_N_EMBD;
    const uint64_t bytes = (uint64_t)n * 4u;
    ds4_gpu_tensor *p[2] = { r->st.ampA[il], r->st.ampB[il] };
    bool ok = true;
    for (unsigned k = 0; ok && k < 2; k++) {
        ok = ds4_gpu_tensor_copy(p[k], 0, snap[il][k], 0, bytes);
        if (ok && scale != 0.0) ok = ds4_gpu_bwd_axpy_tensor(p[k], ctx->dir[k], (float)(scale * ctx->dir_coeff), n);
    }
    if (ok) ok = pt_kp_frozen(c, r, snap) && tr_capture(e, r, s, ctx->skip_amp_bf16, cur_out);
    const bool ymoe_eq = ok && memcmp(cur_out->buf[0], b0->buf[0], (size_t)DS4_N_EMBD * 4u) == 0;
    ok = ok && ymoe_eq;
    if (ok && save_ref1) {
        for (unsigned st = 1; st <= TR_STAGES; st++) {
            const uint32_t len = tr_stage_len(st);
            for (uint32_t i = 0; i < len; i++) ctx->ref1_diff[st][i] = cur_out->buf[st][i] - ctx->ref_base->buf[st][i];
        }
    }
    tr_stat ss[6];
    memset(ss, 0, sizeof ss);
    for (unsigned st = 1; ok && st <= TR_STAGES; st++) {
        const uint32_t len = tr_stage_len(st);
        tr_diff_stats(cur_out->buf[st], ctx->ref_base->buf[st], ctx->ref1_diff[st], ctx->jac[st], len, &ss[st]);
        const double inv_s = scale != 0.0 ? fabs(scale) : 1.0;
        fprintf(ctx->fstg, "%s\t%.9g\t%s\t%u\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%u\t%.12g\t%.12g\n",
                ctx->suite, scale, tr_stage_name(st), ss[st].nnz, ss[st].l2, ss[st].l2 / inv_s,
                ss[st].max_abs, ss[st].cos1, ss[st].dmargin, ss[st].top_idx, ss[st].top_dv, ss[st].top_dm);
    }
    const double dm_act = cur_out->margin - ctx->ref_base->margin;
    const double dm_abs = cur_out->margin - b0->margin;
    if (ok) {
        fprintf(ctx->fsum, "%s\t%.9g\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%u\n",
                ctx->suite, scale, cur_out->argmax, cur_out->margin, dm_act, dm_abs,
                scale * ctx->dm_1st_unit, ss[1].dmargin, ss[2].dmargin, ss[3].dmargin,
                ss[4].dmargin, ss[5].dmargin, dm_act - ss[5].dmargin, (unsigned)ymoe_eq);
        fprintf(stderr, "[trace] %s s=%.6g argmax=%d margin=%.12g dm=%.12g lin=%.12g ypre=%.12g ypost=%.12g hc=%.12g hcp=%.12g xn=%.12g cons_err=%.3e\n",
                ctx->suite, scale, cur_out->argmax, cur_out->margin, dm_act, scale * ctx->dm_1st_unit,
                ss[1].dmargin, ss[2].dmargin, ss[3].dmargin, ss[4].dmargin, ss[5].dmargin, dm_act - ss[5].dmargin);
    }
    return ok;
}

bool pt_trace_run(ds4_engine *e, const pt_cfg *c, pt_run *r, pt_sample *samples,
                  ds4_gpu_tensor **mask, ds4_gpu_tensor *snap[DS4_MAX_LAYER][2], double radius) {
    const uint32_t il = c->layer_hi, E = DS4_N_EMBD, n = r->K * E;
    const uint64_t bytes = (uint64_t)n * 4u;
    bool ok = radius > 0.0 && pt_kp_direction_fixture() && tr_fixture() && pt_kp_restore(c, r, snap);
    double pre = 0;
    if (ok) ok = pt_kp_objective(e, c, r, samples, mask, true, 1, &pre);
    ds4_gpu_tensor *gB[2] = { 0 }, *gm[2] = { 0 }, *d_orth[2] = { 0 };
    ds4_gpu_tensor *live[2] = { r->gA[il], r->gB[il] };
    for (unsigned k = 0; ok && k < 2; k++) {
        if (r->k2) {
            ds4_gpu_tensor *pfx = pt_rows(live[k], 0, r->k2, E);
            ok = pfx && ds4_gpu_tensor_fill_f32(pfx, 0.f, (uint64_t)r->k2 * E);
            tr_free_gpu(&pfx);
        }
        gB[k] = v41_alloc(bytes, &ok); gm[k] = v41_alloc(bytes, &ok); d_orth[k] = v41_alloc(bytes, &ok);
        if (ok) ok = ds4_gpu_tensor_copy(gB[k], 0, live[k], 0, bytes) && ds4_gpu_tensor_fill_f32(live[k], 0.f, n);
    }
    double gn_B = 0, gn_m = 0, cos_m_gB = 0, orth_n = 0, cos_orth_m = 0;
    if (ok) ok = tr_gpu_pair_norm(gB, n, &gn_B) && fabs(pre - .130914387941) < 1e-10 &&
                 fabs(gn_B - .0236464977571) < 1e-10 && gn_B < c->clip;
    fprintf(stderr, "[trace] sliceB pre=%.12g gnorm=%.12g baseline_grad_match=%u\n", pre, gn_B, ok);

    tr_snap b0, rep, b_noamp, cur;
    tr_snap_alloc(&b0); tr_snap_alloc(&rep); tr_snap_alloc(&b_noamp); tr_snap_alloc(&cur);
    float *jac[6] = { 0 }, *ref1[6] = { 0 };
    for (unsigned st = 1; st <= TR_STAGES; st++) {
        jac[st] = xmalloc((size_t)tr_stage_len(st) * 4u);
        ref1[st] = xmalloc((size_t)tr_stage_len(st) * 4u);
    }
    if (ok) ok = tr_capture(e, r, &samples[TR_DOC], 0, &b0) &&
                 tr_capture(e, r, &samples[TR_DOC], 0, &rep) &&
                 tr_snap_equal(&b0, &rep) && b0.argmax == TR_TEACHER &&
                 fabs(b0.margin - .000299453735352) < 1e-9;
    fprintf(stderr, "[trace] parent_doc3_pos575 argmax=%d l_t=%.12g l_c=%.12g margin=%.12g repeat_eq=%u\n",
            b0.argmax, b0.l_t, b0.l_c, b0.margin, ok);
    if (ok) ok = tr_margin_jacobian(e, c, r, jac, gm) &&
                 tr_gpu_pair_norm(gm, n, &gn_m) && tr_gpu_pair_cos(gm, gB, gn_m, gn_B, n, &cos_m_gB);
    const double cos_m_negB = -cos_m_gB, dm_1st_negB = radius * cos_m_negB * gn_m;
    for (unsigned k = 0; ok && k < 2; k++) {
        ok = ds4_gpu_tensor_fill_f32(d_orth[k], 0.f, n) &&
             ds4_gpu_bwd_axpy_tensor(d_orth[k], gB[k], (float)(-1.0 / gn_B), n) &&
             ds4_gpu_bwd_axpy_tensor(d_orth[k], gm[k], (float)(-cos_m_negB / gn_m), n);
    }
    if (ok) ok = tr_gpu_pair_norm(d_orth, n, &orth_n) && orth_n > 1e-6 &&
                 tr_gpu_pair_cos(d_orth, gm, orth_n, gn_m, n, &cos_orth_m) && fabs(cos_orth_m) < 1e-5;
    fprintf(stderr, "[trace] jacobian gn_m=%.12g gn_B=%.12g cos(gm,-gB)=%.12g dm_1st_R=%.12g orth_n=%.12g cos(orth,gm)=%.3e ok=%u\n",
            gn_m, gn_B, cos_m_negB, dm_1st_negB, orth_n, cos_orth_m, ok);

    char psum[1200], pstg[1200];
    snprintf(psum, sizeof psum, "%s/trace_summary.tsv", c->out);
    snprintf(pstg, sizeof pstg, "%s/trace_stages.tsv", c->out);
    FILE *fsum = fopen(psum, "w"), *fstg = fopen(pstg, "w");
    ok = ok && fsum && fstg;
    if (ok) {
        fprintf(fsum, "suite\tscale\targmax\tmargin\tdm_rel\tdm_abs\tdm_lin\tdm_ypre\tdm_ypost\tdm_hcpost\tdm_hcpre\tdm_xn\tcons_err\tymoe_eq\n");
        fprintf(fstg, "suite\tscale\tstage\tnnz\tl2\tl2_over_s\tmax_abs\tcos1\tdmargin\ttop_idx\ttop_dv\ttop_dm\n");
    }
    tr_ctx ctx = { .fsum = fsum, .fstg = fstg };
    for (unsigned st = 1; st <= TR_STAGES; st++) { ctx.jac[st] = jac[st]; ctx.ref1_diff[st] = ref1[st]; }

    /* Suite A: 生产前向沿 -gB (含99轮四档严格重放 + scale=1 双遍复现 + 更小/更大/反向档位) */
    const double s_prod[] = { 1.0, .5, .25, .125, .0625, .03125, .015625, 2.0, 4.0, -.125, -.25, -.5, -1.0 };
    const double exp99[4] = { -.00400543212891, -.00420379638672, -.000970840454102, -.00102615356445 };
    ctx.suite = "prod"; ctx.skip_amp_bf16 = 0; ctx.dir[0] = gB[0]; ctx.dir[1] = gB[1];
    ctx.dir_coeff = -radius / gn_B; ctx.ref_base = &b0; ctx.dm_1st_unit = dm_1st_negB;
    for (size_t i = 0; ok && i < sizeof s_prod / sizeof s_prod[0]; i++) {
        ok = tr_eval_scale(e, c, r, &samples[TR_DOC], snap, &b0, &ctx, s_prod[i], i == 0, &cur);
        if (ok && i < 4) {
            const bool m_ok = fabs(cur.margin - exp99[i]) < 1e-9 && cur.argmax == TR_COMP;
            fprintf(stderr, "[trace] replay99 idx=%zu scale=%.6g got=%.12g expect=%.12g match=%u\n", i, s_prod[i], cur.margin, exp99[i], m_ok);
            ok = m_ok;
        }
        if (ok && i == 0) {
            ok = tr_capture(e, r, &samples[TR_DOC], 0, &rep) && tr_snap_equal(&cur, &rep);
            fprintf(stderr, "[trace] prod_scale1_repeat_eq=%u\n", ok);
        }
    }

    /* Suite B: 因果对照 C1 — 仅跳过 L39 amp_apply 处 BF16 舍入 (core_v41_amp.c:451) */
    if (ok) {
        ok = pt_kp_restore(c, r, snap) && tr_capture(e, r, &samples[TR_DOC], 1, &b_noamp);
        fprintf(stderr, "[trace] no_amp_bf16_s0 argmax=%d margin=%.12g diff_from_prod0=%.12g ok=%u\n",
                b_noamp.argmax, b_noamp.margin, b_noamp.margin - b0.margin, ok);
    }
    const double s_noamp[] = { 1.0, .5, .25, .125, .0625, .03125, .015625 };
    ctx.suite = "no_amp_bf16"; ctx.skip_amp_bf16 = 1; ctx.ref_base = &b_noamp;
    for (size_t i = 0; ok && i < sizeof s_noamp / sizeof s_noamp[0]; i++)
        ok = tr_eval_scale(e, c, r, &samples[TR_DOC], snap, &b0, &ctx, s_noamp[i], i == 0, &cur);

    /* Suite C: 生产前向沿 +g_margin (直接检验该位上升方向在真实BF16前向中是否平滑有效) */
    const double s_dir[] = { 1.0, .5, .25, .125 };
    ctx.suite = "grad_m"; ctx.skip_amp_bf16 = 0; ctx.dir[0] = gm[0]; ctx.dir[1] = gm[1];
    ctx.dir_coeff = radius / gn_m; ctx.ref_base = &b0; ctx.dm_1st_unit = radius * gn_m;
    for (size_t i = 0; ok && i < sizeof s_dir / sizeof s_dir[0]; i++)
        ok = tr_eval_scale(e, c, r, &samples[TR_DOC], snap, &b0, &ctx, s_dir[i], i == 0, &cur);

    /* Suite D: 生产前向沿正交化后的 -gB (去掉与 g_margin 的线性分量, 检验投影方向能否抗住BF16舍入) */
    ctx.suite = "proj_orth"; ctx.skip_amp_bf16 = 0; ctx.dir[0] = d_orth[0]; ctx.dir[1] = d_orth[1];
    ctx.dir_coeff = radius / orth_n; ctx.ref_base = &b0; ctx.dm_1st_unit = 0.0;
    for (size_t i = 0; ok && i < sizeof s_dir / sizeof s_dir[0]; i++)
        ok = tr_eval_scale(e, c, r, &samples[TR_DOC], snap, &b0, &ctx, s_dir[i], i == 0, &cur);

    /* 最终恢复父本并复验逐位一致 */
    if (ok) ok = pt_kp_restore(c, r, snap) && pt_kp_frozen(c, r, snap) &&
                 tr_capture(e, r, &samples[TR_DOC], 0, &cur) && tr_snap_equal(&b0, &cur);
    fprintf(stderr, "[trace] final_restore_verify=%u\n", ok);
    if (fsum && fclose(fsum) != 0) ok = false;
    if (fstg && fclose(fstg) != 0) ok = false;
    for (unsigned st = 1; st <= TR_STAGES; st++) { free(jac[st]); free(ref1[st]); }
    tr_snap_free(&b0); tr_snap_free(&rep); tr_snap_free(&b_noamp); tr_snap_free(&cur);
    for (unsigned k = 0; k < 2; k++) { tr_free_gpu(&gB[k]); tr_free_gpu(&gm[k]); tr_free_gpu(&d_orth[k]); }
    return ok;
}
#endif
#endif
typedef int ds4_core_ptrain_trace_nonempty_tu;
