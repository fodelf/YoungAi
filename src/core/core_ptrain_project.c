/* LoRA-Pro 定理 2.1 的 X=0 单步对照；不是论文的完整 AdamW，也不是 LoRA-RITE。
 * 论文 B 对应这里 Aᵀ，论文 A 对应这里 B，缩放 s=1：
 * dB=(AAᵀ)^-1 gB；dA=(BBᵀ)^-1[gA-(gA Aᵀ)(AAᵀ)^-1 A]。
 * 不构造 E×E 矩阵；逆矩阵用 GPU Newton-Schulz，残差不过门直接失败，不能临时加 ridge。
 * 方向写入原 Adam 的 mA/mB 暂存；只有 diag=7 调用，正式训练仍走原 Adam。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

typedef struct {
    ds4_gpu_tensor *ha, *hb, *ia, *ib, *u, *v, *z, *id, *cross;
    ds4_gpu_tensor *scratch, *mid, *zero, *discard;
} proj_work;
static void pj_drop(ds4_gpu_tensor **p) { if (*p) ds4_gpu_tensor_free(*p); *p = NULL; }

/* ge=gW 给任意行主序矩阵乘；另一输出只乘零，放独立暂存。 */
static bool pj_mm(proj_work *w, ds4_gpu_tensor *out, const ds4_gpu_tensor *left, const ds4_gpu_tensor *right, uint32_t r, uint32_t e) {
    return ds4_gpu_tensor_fill_f32(w->discard, 0.f, (uint64_t)r * e) &&
           ds4_gpu_draft_bias_bwd_tensor(w->discard, out, left, w->zero, right, r, e, r);
}

static bool pj_inverse(proj_work *w, ds4_gpu_tensor *out, ds4_gpu_tensor *h, uint32_t r, double *residual) {
    float host[64 * 64], identity[64 * 64] = {0}; double trace = 0;
    if (!ds4_gpu_tensor_read(h, 0, host, (uint64_t)r * r * 4)) return false;
    for (uint32_t j = 0; j < r; j++) { trace += host[j * r + j]; identity[j * r + j] = 1; }
    if (!(trace > 0) || !isfinite(trace)) return false;
    const uint64_t nn = (uint64_t)r * r;
    bool ok = ds4_gpu_tensor_write(w->id, 0, identity, nn * 4) && ds4_gpu_tensor_fill_f32(out, 0.f, nn) &&
              ds4_gpu_bwd_axpy_tensor(out, w->id, (float)(1 / trace), nn);
    *residual = 1;
    for (unsigned it = 0; ok && it <= 32; it++) {
        double ss = 0;
        ok = pj_mm(w, w->u, h, out, r, r) && ds4_gpu_tensor_copy(w->v, 0, w->u, 0, nn * 4) &&
             ds4_gpu_bwd_axpy_tensor(w->v, w->id, -1.f, nn) && ds4_gpu_bwd_sumsq_tensor(w->v, nn, &ss);
        *residual = sqrt(ss / r);
        if (!ok || !isfinite(*residual)) return false;
        if (*residual <= .001) return true;
        if (it == 32) break;
        /* X <- X(2I-HX)，从 I/trace(H) 开始；H=WWᵀ 的谱范数不大于迹。 */
        ok = ds4_gpu_tensor_copy(w->v, 0, w->id, 0, nn * 4) && ds4_gpu_bwd_axpy_tensor(w->v, w->id, 1.f, nn) &&
             ds4_gpu_bwd_axpy_tensor(w->v, w->u, -1.f, nn) && pj_mm(w, w->z, out, w->v, r, r) &&
             ds4_gpu_tensor_copy(out, 0, w->z, 0, nn * 4);
    }
    return false;
}

static bool pj_direction(proj_work *w, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                          const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb, ds4_gpu_tensor *da, ds4_gpu_tensor *db,
                          uint32_t r, uint32_t e, double *ra, double *rb) {
    return ds4_gpu_draft_bias_tensor(w->ha, a, a, r, e, r) && ds4_gpu_draft_bias_tensor(w->hb, b, b, r, e, r) &&
           pj_inverse(w, w->ia, w->ha, r, ra) && pj_inverse(w, w->ib, w->hb, r, rb) &&
           pj_mm(w, db, w->ia, gb, r, e) &&
           ds4_gpu_draft_bias_tensor(w->cross, ga, a, r, e, r) && pj_mm(w, w->u, w->cross, w->ia, r, r) &&
           pj_mm(w, w->mid, w->u, a, r, e) && ds4_gpu_tensor_copy(w->scratch, 0, ga, 0, (uint64_t)r * e * 4) &&
           ds4_gpu_bwd_axpy_tensor(w->scratch, w->mid, -1.f, (uint64_t)r * e) && pj_mm(w, da, w->ib, w->scratch, r, e);
}

static bool pj_dot(proj_work *w, const ds4_gpu_tensor *g, const ds4_gpu_tensor *d, uint64_t n, double *dot, double *gn2) {
    double gg = 0, dd = 0, ss = 0;
    if (!ds4_gpu_bwd_sumsq_tensor(g, n, &gg) || !ds4_gpu_bwd_sumsq_tensor(d, n, &dd)) return false;
    *gn2 += gg;
    if (gg == 0 || dd == 0) return true;
    if (!(gg > 0 && dd > 0) || !ds4_gpu_tensor_fill_f32(w->scratch, 0.f, n) ||
        !ds4_gpu_bwd_axpy_tensor(w->scratch, g, (float)(1 / sqrt(gg)), n) ||
        !ds4_gpu_bwd_axpy_tensor(w->scratch, d, (float)(1 / sqrt(dd)), n) ||
        !ds4_gpu_bwd_sumsq_tensor(w->scratch, n, &ss)) return false;
    *dot += .5 * (ss - 2) * sqrt(gg * dd);
    return true;
}

/* 已知全梯度 1..16 的 4×4 小例子：投影只删除左上两输入行中后两个输出方向。
 * 检查实际等效梯度，能抓到 A/B 转置映射错和投影漏减一次，而非只检查公式两边同错。 */
static bool pj_selftest(proj_work *w) {
    const float h[6][8] = {{1,0,0,0,0,2,0,0}, {0,0,3,0,0,0,0,4},
        {27,30,33,36,52,56,60,64}, {1,5,9,13,4,12,20,28}, {0}, {0}};
    ds4_gpu_tensor *p[6] = {0}; bool ok = true; double ra = 0, rb = 0, err = 0;
    for (unsigned k = 0; k < 6; k++) { p[k] = v41_alloc(8 * 4, &ok); if (ok) ok = ds4_gpu_tensor_write(p[k], 0, h[k], 8 * 4); }
    if (ok) ok = pj_direction(w, p[0], p[1], p[2], p[3], p[4], p[5], 2, 4, &ra, &rb);
    float eye[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}, actual[16];
    const float expected[16] = {1,2,0,0,5,6,0,0,9,10,11,12,13,14,15,16};
    ds4_gpu_tensor *x = v41_alloc(16 * 4, &ok), *y = v41_alloc(16 * 4, &ok), *t = v41_alloc(8 * 4, &ok);
    if (ok) ok = ds4_gpu_tensor_write(x, 0, eye, sizeof eye) && ds4_gpu_tensor_fill_f32(y, 0.f, 16) &&
                 ds4_gpu_v41_amp_apply_tensor(y, x, p[4], p[1], t, 4, 4, 2) &&
                 ds4_gpu_v41_amp_apply_tensor(y, x, p[0], p[5], t, 4, 4, 2) && ds4_gpu_tensor_read(y, 0, actual, sizeof actual);
    if (ok) for (unsigned j = 0; j < 16; j++) err = fmax(err, fabs(actual[j] - expected[j]));
    ok = ok && err <= .02;
    fprintf(stderr, "[projected] analytic_full_gradient_test max_error=%.12g inverse=%.12g/%.12g %s\n", err, ra, rb, ok ? "PASS" : "FAIL");
    for (unsigned k = 0; k < 6; k++) pj_drop(&p[k]);
    pj_drop(&x); pj_drop(&y); pj_drop(&t); return ok;
}

bool pt_projected_gradient(const pt_cfg *c, pt_run *r, double *gnorm) {
    if (c->rank != 64 || c->diag != 7) return false;
    bool ok = true; proj_work w = {0}; const uint64_t nf = (uint64_t)c->rank * DS4_N_EMBD;
    ds4_gpu_tensor **small[] = {&w.ha,&w.hb,&w.ia,&w.ib,&w.u,&w.v,&w.z,&w.id,&w.cross};
    ds4_gpu_tensor **large[] = {&w.scratch,&w.mid,&w.zero,&w.discard};
    for (unsigned k = 0; k < sizeof small / sizeof small[0]; k++) *small[k] = v41_alloc(64 * 64 * 4, &ok);
    for (unsigned k = 0; k < sizeof large / sizeof large[0]; k++) *large[k] = v41_alloc(nf * 4, &ok);
    if (ok) ok = ds4_gpu_tensor_fill_f32(w.zero, 0.f, nf);
    ds4_gpu_set_quality(true);
    if (ok) ok = pj_selftest(&w);
    double dot = 0, gn2 = 0, worst = 0;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *full[] = {r->st.ampA[l],r->st.ampB[l],r->gA[l],r->gB[l],r->mA[l],r->mB[l]}, *p[6] = {0};
        for (unsigned k = 0; k < 6; k++) { p[k] = pt_rows(full[k], r->k2, c->rank, DS4_N_EMBD); if (!p[k]) ok = false; }
        double ra = 0, rb = 0;
        if (ok) ok = pj_direction(&w, p[0], p[1], p[2], p[3], p[4], p[5], c->rank, DS4_N_EMBD, &ra, &rb) &&
                     pj_dot(&w, p[2], p[4], nf, &dot, &gn2) && pj_dot(&w, p[3], p[5], nf, &dot, &gn2);
        worst = fmax(worst, fmax(ra, rb));
        fprintf(stderr, "[projected] L%u inverse_residual_A=%.12g B=%.12g %s\n", l, ra, rb, ok ? "PASS" : "FAIL");
        for (unsigned k = 0; k < 6; k++) pj_drop(&p[k]);
    }
    *gnorm = sqrt(gn2);
    ok = ok && dot > 0 && isfinite(dot);
    fprintf(stderr, "[projected] gradient_dot_direction=%.12g gnorm=%.12g worst_inverse=%.12g %s\n", dot, *gnorm, worst, ok ? "PASS" : "FAIL");
    ds4_gpu_set_quality(false);
    for (unsigned k = 0; k < sizeof small / sizeof small[0]; k++) pj_drop(small[k]);
    for (unsigned k = 0; k < sizeof large / sizeof large[0]; k++) pj_drop(large[k]);
    return ok;
}
#endif
typedef int ds4_core_ptrain_project_nonempty_tu;
