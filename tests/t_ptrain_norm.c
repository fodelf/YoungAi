/* 同一批目标换分段方式，生产 KL 核的 loss/梯度应保持一致；不加载模型。
 * 这能抓到“每个小段各除自己的长度”和末批仍除固定 batch 的回归。 */
#include "../src/core/core_ptrain.h"
#ifndef DS4_NO_GPU
/* 与 t_units.c 相同：server/core 的 now_sec 声明冲突，不能包含 test_internal.h。 */
void test_assert(bool cond, const char *file, int line, const char *expr);
#define TEST_ASSERT(expr) test_assert((expr), __FILE__, __LINE__, #expr)
ds4_gpu_tensor *test_v41_tensor(const void *src, uint64_t bytes);
int test_v41_read(const ds4_gpu_tensor *t, void *dst, uint64_t bytes);

enum { NT = 9, PREFIX = 2, TK = 2 };

static void count_cases(void) {
    pt_run r = {0}; r.st.cap_tok = 1024; r.st.ctx = 65544;
    int32_t dummy = 1;
    const uint32_t a = pt_pfx_align(), seg = (1024u - 2u) / a * a;
    pt_sample full = {.sn = seg, .sa0 = 1, .m = seg, .raw = 1, .top_id = &dummy};
    pt_sample tail = {.sn = 2u * seg + 7u, .sa0 = 2u * seg + 1u, .m = 7, .raw = 1, .top_id = &dummy};
    pt_sample prefix = {.sn = 20, .sa0 = 1, .m = 19, .l0 = 12, .raw = 1, .top_id = &dummy};
    pt_sample missing = full; missing.top_id = NULL;
    pt_sample over = full; over.sn = r.st.ctx;
    pt_sample clipped = {.sn = 2u * seg, .sa0 = 1, .m = 2u * seg - 1u, .raw = 1, .top_id = &dummy};
    const pt_sample *ss[] = {&full, &tail, &prefix, &missing, &over};
    TEST_ASSERT(pt_batch_loss_tokens(&r, ss, 5) == seg + 14u);
    uint32_t pfx = 0;
    TEST_ASSERT(pt_loss_offset(&r, &tail, &pfx) == 0 && pfx == 2u * seg);
    const uint32_t off = pt_loss_offset(&r, &clipped, &pfx);
    TEST_ASSERT(off == pfx && clipped.m - off <= 1023);
    const pt_sample *one[] = {&clipped};
    TEST_ASSERT(pt_batch_loss_tokens(&r, one, 1) == clipped.m - off);
    pt_cfg c = {.batch = 4, .token_mean = 1};
    r.grad_tokens = 0;
    TEST_ASSERT(pt_loss_scale(&c, &r, 7) == 1.f / 7.f);
    r.grad_tokens = seg + 7;
    TEST_ASSERT(pt_loss_scale(&c, &r, 7) == pt_loss_scale(&c, &r, seg));
    c.token_mean = 0;
    TEST_ASSERT(pt_loss_scale(&c, &r, 7) == 1.f / 28.f);
}

static bool partition(const float *z, const int32_t *ids, const float *p, const float *rest,
                      const uint32_t *parts, uint32_t n, uint32_t token_mean, float *g, double *loss) {
    const uint32_t v = DS4_N_VOCAB;
    pt_cfg c = {.batch = 4, .topk = TK, .token_mean = token_mean};
    pt_run r = {0}; r.st.cap_tok = 1024; r.st.ctx = 65544;
    r.tid = test_v41_tensor(NULL, (NT + PREFIX) * TK * 4);
    r.tp = test_v41_tensor(NULL, (NT + PREFIX) * TK * 4);
    r.trest = test_v41_tensor(NULL, (NT + PREFIX) * 4);
    r.loss = test_v41_tensor(NULL, NT * 4);
    float lh[NT]; r.loss_h = lh;
    ds4_gpu_tensor *tz = test_v41_tensor(z, (uint64_t)NT * v * 4);
    pt_sample samples[4] = {0}; const pt_sample *ss[4];
    for (uint32_t k = 0, start = 0; k < n; start += parts[k++]) {
        samples[k] = (pt_sample){.sn = parts[k] + PREFIX, .sa0 = 1, .m = parts[k] + PREFIX,
            .l0 = PREFIX, .raw = 1, .top_id = (int32_t *)ids + start * TK,
            .top_p = (float *)p + start * TK, .top_rest = (float *)rest + start};
        ss[k] = &samples[k];
    }
    r.grad_tokens = pt_batch_loss_tokens(&r, ss, n);
    bool ok = r.grad_tokens == NT && tz && r.tid && r.tp && r.trest && r.loss;
    uint32_t nt = 0; *loss = 0;
    for (uint32_t k = 0, start = 0; ok && k < n; start += parts[k++]) {
        ds4_gpu_tensor *view = NULL;
        ok = pt_loss_rows(&c, &r, tz, start, ss[k], PREFIX, loss, &nt, &view);
        if (view) ds4_gpu_tensor_free(view);
    }
    ok = ok && nt == NT && test_v41_read(tz, g, (uint64_t)NT * v * 4);
    ds4_gpu_tensor_free(tz); ds4_gpu_tensor_free(r.tid); ds4_gpu_tensor_free(r.tp);
    ds4_gpu_tensor_free(r.trest); ds4_gpu_tensor_free(r.loss);
    return ok;
}

void test_ptrain_norm(void) {
    TEST_ASSERT(ds4_gpu_init());
    count_cases();
    const uint32_t v = DS4_N_VOCAB;
    const uint64_t size = (uint64_t)NT * v;
    float *z = calloc(size, 4), *base = malloc(size * 4), *g = malloc(size * 4);
    int32_t ids[(NT + PREFIX) * TK]; float p[(NT + PREFIX) * TK], rest[NT + PREFIX];
    for (uint32_t i = 0; i < NT + PREFIX; i++) {
        ids[i * TK] = 3; ids[i * TK + 1] = 7;
        p[i * TK] = i % 2 ? .7f : .49f; p[i * TK + 1] = .2f;
        rest[i] = 1.f - p[i * TK] - p[i * TK + 1];
    }
    for (uint32_t i = 0; i < NT; i++) { z[(uint64_t)i * v + 17] = 2.f; z[(uint64_t)i * v + 3] = .1f * i; }
    const uint32_t whole[] = {NT}, split[] = {1, 7, 1}, two[] = {4, 5};
    for (int margin = 0; margin < 2; margin++) {
#ifdef __APPLE__
        if (margin) break;   /* Metal 没有边际项；CUDA 两种生产损失都检查 */
#endif
        TEST_ASSERT(ds4_gpu_bwd_kl_margin(margin ? .5f : 0.f, .5f, .3f));
        double lb = 0, ls = 0, lt = 0;
        TEST_ASSERT(partition(z, ids, p, rest, whole, 1, 1, base, &lb));
        TEST_ASSERT(partition(z, ids, p, rest, split, 3, 1, g, &ls));
        float err = 0, peak = 0;
        for (uint64_t i = 0; i < size; i++) { err = fmaxf(err, fabsf(g[i] - base[i])); peak = fmaxf(peak, fabsf(base[i])); }
        TEST_ASSERT(peak > .01f && err < 1e-7f && fabs(lb - ls) < 1e-6);
        TEST_ASSERT(partition(z, ids, p, rest, two, 2, 1, g, &lt));
        for (uint64_t i = 0; i < size; i++) err = fmaxf(err, fabsf(g[i] - base[i]));
        TEST_ASSERT(err < 1e-7f && fabs(lb - lt) < 1e-6);
        TEST_ASSERT(partition(z, ids, p, rest, whole, 1, 0, g, &lt));
        float legacy_err = 0;
        for (uint64_t i = 0; i < size; i++) legacy_err = fmaxf(legacy_err, fabsf(g[i] * 4.f - base[i]));
        TEST_ASSERT(legacy_err < 1e-7f);
        fprintf(stderr, "ptrain-norm: margin=%d 分段 9 / 1+7+1 / 4+5 梯度差 %.3g，旧分母差 %.3g\n", margin, (double)err, (double)legacy_err);
    }
    TEST_ASSERT(ds4_gpu_bwd_kl_margin(0.f, .5f, 1.f));
    free(z); free(base); free(g);
}
#else
typedef int ds4_t_ptrain_norm_nonempty_tu;
#endif
