/* diag=4 的激活方向检查：在一步更新后的固定参数点测量真实损失变化。
 * 只扰动已有 hpert 接口的中间激活，不对任何权重做有限差分。
 * 固定路由与正常路由分开记录；梯度集中不等于错误，读数不自动放行训练。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

static void ac_drop(ds4_gpu_tensor **t) { if (*t) ds4_gpu_tensor_free(*t); *t = NULL; }

static void ac_reset(pt_run *r) {
    r->save.hpert = NULL; r->save.hpert_eps = 0; r->save.hpert_layer = -1;
    r->save.hpert_mid = 0; r->save.replay = 0; r->hcap_layer = -1; r->hcap_mid = 0;
    r->cont_s = NULL; r->grad_tokens = 0;
}

static bool ac_clear(const pt_cfg *c, pt_run *r) {
    for (uint32_t l = c->layer_lo; l <= c->layer_hi; l++)
        if (!ds4_gpu_tensor_fill_f32(r->gA[l], 0.f, (uint64_t)r->K * DS4_N_EMBD) ||
            !ds4_gpu_tensor_fill_f32(r->gB[l], 0.f, (uint64_t)r->K * DS4_N_EMBD)) return false;
    return true;
}

/* 字节核对包含冻结前缀和可训后缀；读回只做 memcmp，不在主机计算训练矩阵。 */
static bool ac_weights(const pt_cfg *c, pt_run *r, unsigned char *saved, bool check) {
    const size_t bytes = (size_t)r->K * DS4_N_EMBD * 4;
    unsigned char *tmp = check ? xmalloc(bytes) : NULL; size_t off = 0; bool ok = true;
    for (uint32_t l = c->layer_lo; ok && l <= c->layer_hi; l++) {
        ds4_gpu_tensor *w[] = {r->st.ampA[l], r->st.ampB[l]};
        for (unsigned k = 0; ok && k < 2; k++, off += bytes) {
            ok = ds4_gpu_tensor_read(w[k], 0, check ? tmp : saved + off, bytes);
            if (ok && check && memcmp(tmp, saved + off, bytes)) {
                fprintf(stderr, "[actcheck] L%u %c 权重发生变化，拒绝读数\n", l, k ? 'B' : 'A'); ok = false;
            }
        }
    }
    free(tmp); return ok;
}

static bool ac_loss(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s,
                    float eps, double *mean) {
    r->save.hpert_eps = eps; r->cont_s = NULL; r->grad_tokens = 0;
    double sum = 0; uint32_t n = 0;
    if (!pt_step_sample(e, c, r, s, 0, &sum, &n) || n != s->m - s->l0 || !n) return false;
    *mean = sum / n;
    return isfinite(*mean);
}

static bool ac_case(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s,
                    uint32_t layer, int mid, int row, unsigned arm, FILE *f, ds4_gpu_tensor *D) {
    const uint64_t rowf = (uint64_t)DS4_N_HC * DS4_N_EMBD, nel = (uint64_t)s->sn * rowf;
    const char *scope = row < 0 ? "full" : "row";
    ac_reset(r);
    if (!ac_clear(c, r)) return false;
    r->hcap_layer = (int32_t)layer; r->hcap_mid = mid;
    double sum = 0, ss = 0; uint32_t n = 0;
    bool ok = pt_step_sample(e, c, r, s, 1, &sum, &n);
    r->hcap_layer = -1;
    ds4_gpu_tensor *g = row < 0 ? pt_rows(r->hcap, 0, s->sn, rowf) : pt_rows(r->hcap, (uint32_t)row, 1, rowf);
    ds4_gpu_tensor *v = row < 0 ? pt_rows(D, 0, s->sn, rowf) : pt_rows(D, (uint32_t)row, 1, rowf);
    const uint64_t used = row < 0 ? nel : rowf;
    if (ok) ok = n == s->m - s->l0 && n && g && v && ds4_gpu_bwd_sumsq_tensor(g, used, &ss);
    const double gn = sqrt(ss), base = n ? sum / n : 0;
    if (ok) ok = isfinite(gn) && gn > 0 && isfinite(base) && ds4_gpu_tensor_fill_f32(D, 0.f, nel) &&
                 ds4_gpu_bwd_axpy_tensor(v, g, (float)(1.0 / gn), used);
    ac_drop(&g); ac_drop(&v);
    if (!ok) { ac_reset(r); return false; }
    r->save.hpert = D; r->save.hpert_layer = (int32_t)layer; r->save.hpert_mid = mid;
    /* 0.01 探舍入地板；连续多档用于区分局部斜率、舍入噪声和大步非线性。 */
    const float eps[] = {0.01f, 0.1f, 0.3f, 1.f, 3.f, 10.f};
    for (int route = 0; ok && route < 2; route++) {
        r->save.replay = route == 0;
        const char *routing = route ? "live" : "frozen";
        double z0 = 0, z1 = 0;
        ok = ac_loss(e, c, r, s, 0.f, &z0) && ac_loss(e, c, r, s, 0.f, &z1);
        const double noise = fabs(z1 - z0), shift = fabs(z0 - base);
        if (ok && (noise > 1e-7 || shift > 1e-7)) {
            fprintf(stderr, "[actcheck] 零扰动不一致 noise=%.9g shift=%.9g，停止\n", noise, shift); ok = false;
        }
        for (unsigned i = 0; ok && i < sizeof eps / sizeof eps[0]; i++) {
            double plus = 0, minus = 0;
            ok = ac_loss(e, c, r, s, eps[i], &plus) && ac_loss(e, c, r, s, -eps[i], &minus);
            if (!ok) break;
            const double fd = (plus - minus) / (2.0 * eps[i]), ratio = fd / gn;
            fprintf(f, "%u\t%u\t%u\t%d\t%s\t%d\t%s\t%.9g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.9g\t%.12g\t%.12g\t%.12g\n",
                    arm, s->doc, layer, mid, scope, row, routing, (double)eps[i], base, plus, minus, gn, fd, ratio, base - minus, noise, shift);
            fflush(f);
            fprintf(stderr, "[actcheck] arm=%u L%u%s %s %s eps=%.3g fd/g=%.6g loss_minus_change=%+.9g\n",
                    arm, layer, mid ? "m" : "", scope, routing, (double)eps[i], ratio, minus - base);
        }
        double end = 0;
        if (ok) ok = ac_loss(e, c, r, s, 0.f, &end) && fabs(end - base) <= 1e-7;
        if (pt_mem_mb() >= 0 && pt_mem_mb() < 3000) ok = false;
    }
    ac_reset(r);
    return ok && ac_clear(c, r);
}

bool pt_actcheck(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s,
                 uint32_t layer, uint32_t row, unsigned arm) {
    if (!r->lb.ready || r->hcap || s->sn > r->st.cap_tok || row >= s->sn ||
        layer < c->layer_lo || layer + 1u > c->layer_hi || !c->token_mean) return false;
    const uint64_t nel = (uint64_t)s->sn * DS4_N_HC * DS4_N_EMBD;
    const size_t bytes = (size_t)(c->layer_hi - c->layer_lo + 1u) * 2u * r->K * DS4_N_EMBD * 4;
    unsigned char *weights = xmalloc(bytes); bool ok = ac_weights(c, r, weights, false);
    ds4_gpu_tensor *D = v41_alloc(nel * 4, &ok); r->hcap = v41_alloc(nel * 4, &ok);
    char path[1200]; snprintf(path, sizeof path, "%s/activation.tsv", c->out);
    FILE *f = fopen(path, arm == 1 ? "w" : "a");
    if (!f) ok = false;
    if (f && arm == 1) fprintf(f, "arm\tdoc\tlayer\tmid\tscope\trow\trouting\tepsilon\tbase_loss\tplus_loss\tminus_loss\tgradient_norm\tfinite_difference\tratio\tdescent_gain\tzero_noise\tzero_shift\n");
    /* 单篇目标按有效 token 求均值；不沿用先前 batch4 的分母。只测量，不更新 Adam。 */
    pt_cfg one = *c; one.batch = 1; one.token_mean = 1;
    const struct { uint32_t layer; int mid, row; } cases[] = {
        {layer + 1u, 0, -1}, {layer + 1u, 0, (int)row}, {layer, 1, -1}, {layer, 1, (int)row}
    };
    fprintf(stderr, "[actcheck] arm=%u doc=%u row=%u；固定一步后的权重，扰动激活；含原②的全部A/B做字节核对\n", arm, s->doc, row);
    for (unsigned i = 0; ok && i < sizeof cases / sizeof cases[0]; i++)
        ok = ac_case(e, &one, r, s, cases[i].layer, cases[i].mid, cases[i].row, arm, f, D);
    ac_reset(r);
    const bool same = ac_weights(c, r, weights, true);
    fprintf(stderr, "[actcheck] arm=%u weights_unchanged=%s measurement=%s；此状态只代表测量完成，不代表下降方向有效\n", arm, same ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
    ac_drop(&D); ac_drop(&r->hcap); free(weights); if (f) fclose(f);
    return ac_clear(c, r) && same && ok;
}

#endif
typedef int ds4_core_ptrain_actcheck_nonempty_tu;
