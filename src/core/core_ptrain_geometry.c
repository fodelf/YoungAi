/* diag=6：量低秩因子的方向集中程度和 Adam 对等价因子写法的依赖。
 * A/B 都是 [rank][E]，实际残差为 x Bᵀ A。只改可训练后缀：
 * Hadamard 的 QᵀQ=I；交替 2/0.5 缩放分别乘 A 和 B 的倒数。
 * 这些恒等式在有限精度下未必成立，端到端前向闸由 optcheck 先核对。
 * 矩阵运算复用 GPU 原语；主机只汇总 64×64 Gram 的标量统计。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define GEO_ROWS 64u
struct pt_geometry {
    ds4_gpu_tensor *q, *scratch, *discard, *zero, *gram, *tmp, *y, *diff;
    ds4_gpu_tensor *x[DS4_MAX_LAYER], *pre[DS4_MAX_LAYER], *ref[DS4_MAX_LAYER], *original[DS4_MAX_LAYER];
    ds4_gpu_tensor *a[DS4_MAX_LAYER], *b[DS4_MAX_LAYER];
    uint32_t rows;
    double ref2;
    FILE *f;
};

static void geo_drop(ds4_gpu_tensor **p) { if (*p) ds4_gpu_tensor_free(*p); *p = NULL; }

void pt_geometry_free(pt_geometry *g) {
    if (!g) return;
    geo_drop(&g->q); geo_drop(&g->scratch); geo_drop(&g->discard); geo_drop(&g->zero);
    geo_drop(&g->gram); geo_drop(&g->tmp); geo_drop(&g->y); geo_drop(&g->diff);
    for (uint32_t l = 0; l < DS4_MAX_LAYER; l++) {
        geo_drop(&g->x[l]); geo_drop(&g->pre[l]); geo_drop(&g->ref[l]); geo_drop(&g->original[l]);
        geo_drop(&g->a[l]); geo_drop(&g->b[l]);
    }
    if (g->f) fclose(g->f);
    free(g);
}

pt_geometry *pt_geometry_new(const pt_cfg *c, pt_run *r) {
    if (c->rank != 64 || r->K != r->k2 + c->rank || c->max_steps != 1) return NULL;
    pt_geometry *g = xmalloc_zeroed(1, sizeof *g); bool ok = true;
    const uint64_t nf = (uint64_t)c->rank * DS4_N_EMBD, ny = (uint64_t)GEO_ROWS * DS4_N_EMBD;
    g->q = v41_alloc(64 * 64 * 4, &ok); g->gram = v41_alloc(64 * 64 * 4, &ok);
    g->scratch = v41_alloc(nf * 4, &ok); g->discard = v41_alloc(nf * 4, &ok);
    g->zero = v41_alloc(nf * 4, &ok); g->tmp = v41_alloc(GEO_ROWS * 64 * 4, &ok);
    g->y = v41_alloc(ny * 4, &ok); g->diff = v41_alloc(ny * 4, &ok);
    float q[64 * 64];
    for (unsigned i = 0; i < 64; i++) for (unsigned j = 0; j < 64; j++) {
        unsigned z = i & j, parity = 0;
        while (z) { parity ^= z & 1u; z >>= 1; }
        q[i * 64 + j] = parity ? -.125f : .125f;
    }
    if (ok) ok = ds4_gpu_tensor_write(g->q, 0, q, sizeof q) && ds4_gpu_tensor_fill_f32(g->zero, 0.f, nf);
    char path[1200]; snprintf(path, sizeof path, "%s/geometry.tsv", c->out);
    g->f = fopen(path, "w");
    if (!ok || !g->f) { pt_geometry_free(g); return NULL; }
    fprintf(g->f, "kind\tarm\tlayer\tfactor\tvalue1\tvalue2\tvalue3\tvalue4\n");
    ds4_gpu_set_quality(true);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *all[] = {r->st.ampA[l], r->st.ampB[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *w = pt_rows(all[k], r->k2, 64, DS4_N_EMBD);
            float h[64 * 64]; double frob = 0, trace = 0, diag2 = 0, maxdiag = 0;
            ok = w && ds4_gpu_draft_bias_tensor(g->gram, w, w, 64, DS4_N_EMBD, 64) &&
                 ds4_gpu_bwd_sumsq_tensor(g->gram, 64 * 64, &frob) && ds4_gpu_tensor_read(g->gram, 0, h, sizeof h);
            if (ok) {
                for (unsigned j = 0; j < 64; j++) { double d = h[j * 64 + j]; trace += d; diag2 += d * d; if (d > maxdiag) maxdiag = d; }
                ok = trace > 0 && frob > 0;
                fprintf(g->f, "gram\t0\t%u\t%c\t%.12g\t%.12g\t%.12g\t%.12g\n", l, k ? 'B' : 'A',
                        trace, frob > 0 ? trace * trace / frob : 0, frob > 0 ? (frob - diag2) / frob : 0, trace > 0 ? maxdiag / trace : 0);
            }
            geo_drop(&w);
        }
    }
    /* Q²=I 用同一 GPU 矩阵乘核核验，不依赖手写矩阵乘。 */
    if (ok) {
        float h[64 * 64]; double err = 0;
        ok = ds4_gpu_draft_bias_tensor(g->gram, g->q, g->q, 64, 64, 64) && ds4_gpu_tensor_read(g->gram, 0, h, sizeof h);
        if (ok) for (unsigned i = 0; i < 64; i++) for (unsigned j = 0; j < 64; j++) err = fmax(err, fabs(h[i * 64 + j] - (i == j ? 1.f : 0.f)));
        fprintf(stderr, "[geometry] Hadamard orthogonality max_error=%.12g %s\n", err, ok && err <= 1e-7 ? "PASS" : "FAIL");
        ok = ok && err <= 1e-7;
    }
    ds4_gpu_set_quality(false); fflush(g->f);
    if (!ok) { pt_geometry_free(g); return NULL; }
    return g;
}

bool pt_geometry_basis(const pt_cfg *c, pt_run *r, pt_geometry *g, unsigned arm) {
    if (arm <= 2) return true;
    bool ok = arm <= 4; const uint64_t nf = (uint64_t)c->rank * DS4_N_EMBD;
    ds4_gpu_set_quality(true);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *all[] = {r->st.ampA[l], r->st.ampB[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *w = pt_rows(all[k], r->k2, c->rank, DS4_N_EMBD);
            ok = w != NULL;
            if (ok && arm == 3) {
                /* 复用 ge=gW；多算的 gW 用全零输入并丢弃，不污染真实梯度。 */
                ok = ds4_gpu_tensor_fill_f32(g->discard, 0.f, nf) &&
                     ds4_gpu_draft_bias_bwd_tensor(g->discard, g->scratch, g->q, g->zero, w, 64, DS4_N_EMBD, 64) &&
                     ds4_gpu_tensor_copy(w, 0, g->scratch, 0, nf * 4);
            } else if (ok) {
                ok = ds4_gpu_tensor_copy(g->scratch, 0, w, 0, nf * 4) && ds4_gpu_tensor_fill_f32(w, 0.f, nf);
                for (uint32_t j = 0; ok && j < c->rank; j++) {
                    const float scale = ((j + k) & 1u) ? .5f : 2.f;
                    ds4_gpu_tensor *to = pt_rows(w, j, 1, DS4_N_EMBD), *from = pt_rows(g->scratch, j, 1, DS4_N_EMBD);
                    ok = to && from && ds4_gpu_bwd_axpy_tensor(to, from, scale, DS4_N_EMBD);
                    geo_drop(&to); geo_drop(&from);
                }
            }
            geo_drop(&w);
        }
    }
    ds4_gpu_set_quality(false);
    return ok;
}

bool pt_geometry_before(const pt_cfg *c, pt_run *r, pt_geometry *g, uint32_t nrow, unsigned arm) {
    bool ok = true; g->rows = nrow < GEO_ROWS ? nrow : GEO_ROWS;
    const uint64_t ny = (uint64_t)g->rows * DS4_N_EMBD, nf = (uint64_t)c->rank * DS4_N_EMBD;
    ds4_gpu_set_quality(true);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        if (arm == 1) {
            g->x[l] = v41_alloc(ny * 4, &ok); g->pre[l] = v41_alloc(ny * 4, &ok); g->ref[l] = v41_alloc(ny * 4, &ok);
            g->original[l] = v41_alloc(ny * 4, &ok);
            g->a[l] = v41_alloc(nf * 4, &ok); g->b[l] = v41_alloc(nf * 4, &ok);
            if (ok) ok = r->save.moe_in[l] && ds4_gpu_tensor_copy(g->x[l], 0, r->save.moe_in[l], 0, ny * 4);
        }
        ds4_gpu_tensor *a = pt_rows(r->st.ampA[l], r->k2, c->rank, DS4_N_EMBD), *b = pt_rows(r->st.ampB[l], r->k2, c->rank, DS4_N_EMBD);
        if (ok) ok = a && b && ds4_gpu_tensor_copy(g->a[l], 0, a, 0, nf * 4) && ds4_gpu_tensor_copy(g->b[l], 0, b, 0, nf * 4) &&
                     ds4_gpu_tensor_fill_f32(g->pre[l], 0.f, ny) &&
                     ds4_gpu_v41_amp_apply_tensor(g->pre[l], g->x[l], a, b, g->tmp, g->rows, DS4_N_EMBD, c->rank);
        if (ok && arm == 1) ok = ds4_gpu_tensor_copy(g->original[l], 0, g->pre[l], 0, ny * 4);
        geo_drop(&a); geo_drop(&b);
    }
    ds4_gpu_set_quality(false); return ok;
}

/* 固定第一臂的原始输出，量纯乘积误差，区分换基舍入和端到端放大。 */
bool pt_geometry_equivalence(const pt_cfg *c, pt_run *r, pt_geometry *g, unsigned arm) {
    if (arm <= 2) return true;
    bool ok = true; double whole = 0, error = 0;
    const uint64_t ny = (uint64_t)g->rows * DS4_N_EMBD;
    ds4_gpu_set_quality(true);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *a = pt_rows(r->st.ampA[l], r->k2, c->rank, DS4_N_EMBD), *b = pt_rows(r->st.ampB[l], r->k2, c->rank, DS4_N_EMBD);
        double ss = 0, dd = 0;
        ok = a && b && g->original[l] && ds4_gpu_tensor_fill_f32(g->y, 0.f, ny) &&
             ds4_gpu_v41_amp_apply_tensor(g->y, g->x[l], a, b, g->tmp, g->rows, DS4_N_EMBD, c->rank) &&
             ds4_gpu_bwd_sumsq_tensor(g->original[l], ny, &ss) && ds4_gpu_bwd_axpy_tensor(g->y, g->original[l], -1.f, ny) &&
             ds4_gpu_bwd_sumsq_tensor(g->y, ny, &dd);
        whole += ss; error += dd;
        geo_drop(&a); geo_drop(&b);
    }
    fprintf(g->f, "pre_equivalence\t%u\t-1\tAB\t%.12g\t%.12g\t%.12g\t0\n", arm, whole, error, whole > 0 ? sqrt(error / whole) : 0);
    fprintf(stderr, "[geometry] arm=%u local_FP32_relative_error=%.12g\n", arm, whole > 0 ? sqrt(error / whole) : 0);
    ds4_gpu_set_quality(false); fflush(g->f); return ok && whole > 0;
}

/* 报实际乘积的变化，包含 ΔBᵀΔA；以 arm1 为参照，先量原幅度，再只用训练输入定标。 */
bool pt_geometry_measure(const pt_cfg *c, pt_run *r, pt_geometry *g, unsigned arm, bool matched, double *norm) {
    bool ok = true; double total = 0, diff = 0;
    const uint64_t ny = (uint64_t)g->rows * DS4_N_EMBD;
    ds4_gpu_set_quality(true);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *a = pt_rows(r->st.ampA[l], r->k2, c->rank, DS4_N_EMBD), *b = pt_rows(r->st.ampB[l], r->k2, c->rank, DS4_N_EMBD);
        double ss = 0, dd = 0;
        ok = a && b && ds4_gpu_tensor_fill_f32(g->y, 0.f, ny) &&
             ds4_gpu_v41_amp_apply_tensor(g->y, g->x[l], a, b, g->tmp, g->rows, DS4_N_EMBD, c->rank) &&
             ds4_gpu_bwd_axpy_tensor(g->y, g->pre[l], -1.f, ny) && ds4_gpu_bwd_sumsq_tensor(g->y, ny, &ss);
        if (ok && arm == 1) ok = ds4_gpu_tensor_copy(g->ref[l], 0, g->y, 0, ny * 4);
        if (ok) ok = ds4_gpu_tensor_copy(g->diff, 0, g->y, 0, ny * 4) &&
                     ds4_gpu_bwd_axpy_tensor(g->diff, g->ref[l], -1.f, ny) && ds4_gpu_bwd_sumsq_tensor(g->diff, ny, &dd);
        total += ss; diff += dd;
        fprintf(g->f, "%s\t%u\t%u\tAB\t%.12g\t%.12g\t0\t0\n", matched ? "matched_layer" : "raw_layer", arm, l, ss, dd);
        geo_drop(&a); geo_drop(&b);
    }
    if (arm == 1) g->ref2 = total;
    const double cosine = total > 0 && g->ref2 > 0 ? (total + g->ref2 - diff) / (2 * sqrt(total * g->ref2)) : 0;
    *norm = sqrt(total);
    fprintf(g->f, "%s\t%u\t-1\tAB\t%.12g\t%.12g\t%.12g\t%.12g\n", matched ? "matched_total" : "raw_total", arm, total, diff, cosine, g->ref2);
    fprintf(stderr, "[geometry] arm=%u %s norm=%.12g ref_norm=%.12g cosine=%.12g relative_diff=%.12g rows=%u\n",
            arm, matched ? "matched" : "raw", *norm, sqrt(g->ref2), cosine, g->ref2 > 0 ? sqrt(diff / g->ref2) : 0, g->rows);
    ds4_gpu_set_quality(false); fflush(g->f);
    return ok && total > 0 && isfinite(total) && isfinite(diff);
}

bool pt_geometry_rescale(const pt_cfg *c, pt_run *r, pt_geometry *g, double norm) {
    if (!(norm > 0) || !(g->ref2 > 0)) return false;
    const float scale = (float)(sqrt(g->ref2) / norm);
    if (!(scale >= .1f && scale <= 10.f)) return false;
    bool ok = true; const uint64_t nf = (uint64_t)c->rank * DS4_N_EMBD;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *all[] = {r->st.ampA[l], r->st.ampB[l]}, *start[] = {g->a[l], g->b[l]};
        for (unsigned k = 0; ok && k < 2; k++) {
            ds4_gpu_tensor *w = pt_rows(all[k], r->k2, c->rank, DS4_N_EMBD);
            ok = w && ds4_gpu_tensor_copy(g->scratch, 0, w, 0, nf * 4) && ds4_gpu_bwd_axpy_tensor(g->scratch, start[k], -1.f, nf) &&
                 ds4_gpu_tensor_copy(w, 0, start[k], 0, nf * 4) && ds4_gpu_bwd_axpy_tensor(w, g->scratch, scale, nf);
            geo_drop(&w);
        }
    }
    fprintf(stderr, "[geometry] training_input_scale=%.12g\n", scale);
    return ok;
}

bool pt_geometry_project(const pt_cfg *c, pt_run *r, pt_geometry *g, double *gnorm) {
    if (!pt_projected_gradient(c, r, gnorm)) return false;
    bool ok = true; double linear = 0; const uint64_t ny = (uint64_t)g->rows * DS4_N_EMBD;
    ds4_gpu_set_quality(true);
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *full[] = {r->st.ampA[l],r->st.ampB[l],r->mA[l],r->mB[l]}, *p[4] = {0};
        for (unsigned k = 0; k < 4; k++) { p[k] = pt_rows(full[k], r->k2, c->rank, DS4_N_EMBD); if (!p[k]) ok = false; }
        double ss = 0;
        if (ok) ok = ds4_gpu_tensor_fill_f32(g->y, 0.f, ny) &&
                     ds4_gpu_v41_amp_apply_tensor(g->y, g->x[l], p[2], p[1], g->tmp, g->rows, DS4_N_EMBD, c->rank) &&
                     ds4_gpu_v41_amp_apply_tensor(g->y, g->x[l], p[0], p[3], g->tmp, g->rows, DS4_N_EMBD, c->rank) &&
                     ds4_gpu_bwd_sumsq_tensor(g->y, ny, &ss);
        linear += ss;
        for (unsigned k = 0; k < 4; k++) geo_drop(&p[k]);
    }
    const float eta = linear > 0 ? (float)sqrt(g->ref2 / linear) : 0;
    if (!(eta > 0 && eta <= 1)) ok = false;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *full[] = {r->st.ampA[l],r->st.ampB[l],r->mA[l],r->mB[l]}, *p[4] = {0};
        for (unsigned k = 0; k < 4; k++) { p[k] = pt_rows(full[k], r->k2, c->rank, DS4_N_EMBD); if (!p[k]) ok = false; }
        if (ok) ok = ds4_gpu_bwd_axpy_tensor(p[0], p[2], -eta, (uint64_t)c->rank * DS4_N_EMBD) &&
                     ds4_gpu_bwd_axpy_tensor(p[1], p[3], -eta, (uint64_t)c->rank * DS4_N_EMBD);
        for (unsigned k = 0; k < 4; k++) geo_drop(&p[k]);
    }
    fprintf(stderr, "[projected] linear_norm=%.12g target_norm=%.12g eta=%.12g %s\n", sqrt(linear), sqrt(g->ref2), eta, ok ? "PASS" : "FAIL");
    ds4_gpu_set_quality(false); return ok;
}

#endif
typedef int ds4_core_ptrain_geometry_nonempty_tu;
