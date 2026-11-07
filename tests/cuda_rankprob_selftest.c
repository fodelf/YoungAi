/* 不加载模型：直接用 GPU 损失作有限差分，验证复合梯度、默认关闭和原地覆盖。
 * 平滑点两档扰动误差 <=2e-4；并列是折点，单独验证半梯度约定。 */
#include "ds4_gpu.h"
#include "ds4_gpu_rankprob.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static unsigned checks;

static void evaluate(const float *z, uint32_t V, const int32_t *ids, const float *q, uint32_t K,
                     float rest, float weight, float scale, float lam, int native, int inplace, int offset,
                     float *g, float *loss) {
    const uint64_t zb = (uint64_t)(offset + 1) * V * 4;
    ds4_gpu_tensor *tz = ds4_gpu_tensor_alloc(zb), *ti = ds4_gpu_tensor_alloc((uint64_t)K * 4);
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc((uint64_t)K * 4), *tr = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(4), *tl = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *tg = inplace ? ds4_gpu_tensor_view(tz, (uint64_t)offset * V * 4, (uint64_t)V * 4) : ds4_gpu_tensor_alloc((uint64_t)V * 4);
    CHECK(tz && ti && tq && tr && tw && tl && tg);
    CHECK(ds4_gpu_tensor_fill_f32(tz, -77.f, (uint64_t)(offset + 1) * V));
    CHECK(ds4_gpu_tensor_write(tz, (uint64_t)offset * V * 4, z, (uint64_t)V * 4));
    CHECK(ds4_gpu_tensor_write(ti, 0, ids, (uint64_t)K * 4) && ds4_gpu_tensor_write(tq, 0, q, (uint64_t)K * 4));
    CHECK(ds4_gpu_tensor_write(tr, 0, &rest, 4) && ds4_gpu_tensor_write(tw, 0, &weight, 4));
    CHECK(native ? ds4_gpu_bwd_kl_topk_tensor(tg, tl, tz, offset, 1, V, ti, tq, tr, tw, K, scale) :
        ds4_gpu_bwd_kl_rankprob_tensor(tg, tl, tz, offset, 1, V, ti, tq, tr, tw, K, scale, lam));
    CHECK(ds4_gpu_tensor_read(tg, 0, g, (uint64_t)V * 4) && ds4_gpu_tensor_read(tl, 0, loss, 4));
    CHECK(isfinite(*loss));
    for (uint32_t k = 0; k < V; k++) CHECK(isfinite(g[k]));
    ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tz); ds4_gpu_tensor_free(ti); ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tr); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tl);
    checks++;
}

static void smooth(const char *name, const float *z0, const int32_t *ids, const float *q) {
    float z[8], g[8], tmp[8], loss, other, native[8];
    const float scale = 0.37f, weight = 0.7f, rest = 0.1f, eps[] = {0.01f, 0.003f};
    memcpy(z, z0, sizeof z);
    evaluate(z, 8, ids, q, 4, rest, weight, scale, 1.f, 0, 0, 0, g, &loss);
    double sum = 0; for (unsigned k = 0; k < 8; k++) sum += g[k];
    CHECK(fabs(sum) <= 2e-6);
    double worst = 0;
    for (unsigned e = 0; e < 2; e++) for (unsigned k = 0; k < 8; k++) {
        float lp, lm;
        z[k] = z0[k] + eps[e]; evaluate(z, 8, ids, q, 4, rest, weight, scale, 1.f, 0, 1, 1, tmp, &lp);
        z[k] = z0[k] - eps[e]; evaluate(z, 8, ids, q, 4, rest, weight, scale, 1.f, 0, 1, 1, tmp, &lm);
        z[k] = z0[k];
        const double fd = (lp - lm) * scale / (2.0 * eps[e]), err = fabs(fd - g[k]);
        if (err > worst) worst = err;
        CHECK(err <= 2e-4);
    }
    evaluate(z, 8, ids, q, 4, rest, weight, scale, 1.f, 0, 1, 1, tmp, &other);
    CHECK(!memcmp(g, tmp, sizeof g) && loss == other);
    evaluate(z, 8, ids, q, 4, rest, weight, scale, 0.f, 0, 1, 0, tmp, &other);
    evaluate(z, 8, ids, q, 4, rest, weight, scale, 0.f, 1, 0, 0, native, &loss);
    CHECK(!memcmp(native, tmp, sizeof tmp) && loss == other);
    printf("%s finite_difference_max=%.9g gradient_sum=%.9g inplace=PASS disabled_bitwise=PASS\n", name, worst, sum);
}

static void gate(const char *name, const float *z, const int32_t *ids, const float *q,
                  int j, int t, int half) {
    float a[8], b[8], la, lb;
    evaluate(z, 8, ids, q, 4, 0.1f, 1.f, 1.f, 1.f, 0, 0, 0, a, &la);
    evaluate(z, 8, ids, q, 4, 0.1f, 1.f, 1.f, 0.f, 1, 0, 0, b, &lb);
    CHECK(la == lb);
    double denom = 0; for (unsigned k = 0; k < 8; k++) denom += exp(z[k]);
    for (int k = 0; k < 8; k++) {
        const double want = half ? 0.5 * (k == j ? exp(z[j]) / denom : k == t ? -exp(z[t]) / denom : 0) : 0;
        CHECK(fabs((a[k] - b[k]) - want) <= 2e-6);
    }
    printf("%s branch_gradient=PASS\n", name);
}

int main(void) {
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_bwd_kl_margin(0.f, 0.5f, 0.3f) && ds4_gpu_bwd_kl_tail(0.f, 1.f));
    const int32_t ids[4] = {3, 1, 6, 2}, rev[4] = {2, 6, 1, 3};
    const float q[4] = {0.5f, 0.25f, 0.1f, 0.05f}, qr[4] = {0.05f, 0.1f, 0.25f, 0.5f};
    const float correct[8] = {-0.8f, 0.1f, -1.2f, 1.7f, -0.7f, -0.9f, 0.2f, -1.1f};
    const float wrong[8] = {-0.8f, 0.5f, -1.2f, 0.2f, -0.7f, -0.9f, 0.1f, -1.1f};
    const float outside[8] = {-0.8f, 0.1f, -1.2f, 0.2f, -0.7f, -0.9f, 0.0f, 1.5f};
    const float strong[8] = {-3.f, 9.f, -4.f, -1.f, -5.f, -2.f, -3.f, -4.f};
    smooth("correct", correct, ids, q); smooth("wrong_low_confidence", wrong, ids, q);
    smooth("winner_outside_teacher_topk", outside, ids, q); smooth("wrong_high_confidence", strong, ids, q);
    smooth("teacher_table_permuted", wrong, rev, qr);
    gate("correct_adds_zero", correct, ids, q, 3, 3, 0);
    const float teacher_tie[4] = {0.375f, 0.375f, 0.1f, 0.05f};
    gate("teacher_tie_adds_zero", wrong, ids, teacher_tie, 1, 3, 0);
    float tie[8]; memcpy(tie, wrong, sizeof tie); tie[3] = tie[1];
    gate("student_tie_half_gradient", tie, ids, q, 1, 3, 1);
    const float flat_q[4] = {0.225f, 0.225f, 0.225f, 0.225f};
    gate("outside_teacher_bound_ambiguous", outside, ids, flat_q, 7, 3, 0);
    float a[8], b[8], la, lb, shifted[8];
    for (unsigned k = 0; k < 8; k++) shifted[k] = wrong[k] + 7.f;
    evaluate(wrong, 8, ids, q, 4, 0.1f, 1.f, 1.f, 1.f, 0, 0, 0, a, &la);
    evaluate(shifted, 8, ids, q, 4, 0.1f, 1.f, 1.f, 1.f, 0, 0, 0, b, &lb);
    CHECK(fabs(la - lb) < 2e-6);
    for (unsigned k = 0; k < 8; k++) CHECK(fabs(a[k] - b[k]) < 2e-6);
    evaluate(wrong, 8, rev, qr, 4, 0.1f, 1.f, 1.f, 1.f, 0, 0, 0, b, &lb);
    CHECK(fabs(la - lb) < 2e-6);
    for (unsigned k = 0; k < 8; k++) CHECK(fabs(a[k] - b[k]) < 2e-6);
    const uint32_t V = 129280;
    float *z = calloc(V, 4), *g = calloc(V, 4), *ref = calloc(V, 4);
    CHECK(z && g && ref); z[129279] = 3.f; z[3] = 1.f;
    evaluate(z, V, ids, q, 4, 0.1f, 1.f, 1.f, 1.f, 0, 1, 1, g, &la);
    evaluate(z, V, ids, q, 4, 0.1f, 1.f, 1.f, 1.f, 0, 0, 0, ref, &lb);
    CHECK(la == lb && !memcmp(g, ref, (size_t)V * 4));
    double sum = 0; for (uint32_t k = 0; k < V; k++) sum += g[k];
    CHECK(fabs(sum) < 2e-6);
    free(z); free(g); free(ref);
    printf("full_vocab=%u gradient_sum=%.9g shift_and_permutation=PASS GPU_evaluations=%u ALL_PASS\n", V, sum, checks);
    return 0;
}
