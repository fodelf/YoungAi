/* t_zfinetune.c — 微调侧车按秩拼接的离线自测(不需要模型/GPU)。
 *
 * 判什么: 拼接后的一个 rank(k1+k2) 的 z, 逐位等于原来两段各自 apply 之后相加。
 * 这是三文件部署的全部数学依据 —— 它一旦不成立, "zchain 冻结 + 微调独立"就是假的。
 * 顺带判两条守规矩: 层数不一致必须拒绝; 乘性(zmul!=0)的底座必须拒绝合并。
 *
 * 构造两个内存里的 DQZ2 链(不落盘, 直接填 struct), 因为要测的是合并算术, 不是文件解析。*/
#include "ds4_zfinetune.h"
#include "src/common/ds4_float.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 8u          /* 小维: 数学与 4096 同构, 跑得快 */

static uint16_t *mk_payload(uint32_t k, const float *z, const float *U, const float *V) {
    const size_t n = (size_t)k * (1u + D + D);
    uint16_t *p = malloc(n * sizeof(uint16_t));
    size_t o = 0;
    for (uint32_t c = 0; c < k; c++) p[o++] = ds4_f64_to_f16((double)z[c]);
    for (size_t i = 0; i < (size_t)D * k; i++) p[o++] = ds4_f64_to_f16((double)U[i]);
    for (size_t i = 0; i < (size_t)D * k; i++) p[o++] = ds4_f64_to_f16((double)V[i]);
    return p;
}

/* 直接按契约算 routed += U diag(z) Vᵀ x(fp16 载荷读回 f32), 当参照。 */
static void ref_apply(const uint16_t *pay, uint32_t k, const float *x, float *y) {
    const uint16_t *hU = pay + k, *hV = hU + (size_t)D * k;
    for (uint32_t c = 0; c < k; c++) {
        double vx = 0;
        for (uint32_t j = 0; j < D; j++) vx += (double)ds4_f16_to_f32(hV[(size_t)j * k + c]) * x[j];
        const double s = (double)ds4_f16_to_f32(pay[c]) * vx;
        for (uint32_t i = 0; i < D; i++) y[i] += (float)(s * ds4_f16_to_f32(hU[(size_t)i * k + c]));
    }
}

static ds4_zchain *mk_chain(uint32_t nl, uint32_t k, uint16_t *pay, uint32_t zmul) {
    ds4_zchain *z = calloc(1, sizeof(*z));
    z->n_layer = nl; z->n_expert = 4; z->d_model = D;
    z->layer = calloc(nl, sizeof(*z->layer));
    if (k) {
        z->layer[0].zl.zlk = k; z->layer[0].zl.zdin = D; z->layer[0].zl.zltr = 1.0f;
        z->layer[0].zl.zmul = zmul; z->layer[0].zl.zlm = pay;
    }
    return z;
}

/* 挂进 ds4_unit(test_common.c 调): 返回失败条数, 0 = 全过。 */
int unit_zfinetune(void) {
    srand(7);
    const uint32_t k1 = 3, k2 = 2;
    float z1[3], U1[D * 3], V1[D * 3], z2[2], U2[D * 2], V2[D * 2], x[D];
    for (uint32_t i = 0; i < k1; i++) z1[i] = (float)((rand() % 200 - 100) / 64.0);
    for (uint32_t i = 0; i < D * k1; i++) { U1[i] = (float)((rand() % 200 - 100) / 128.0);
                                            V1[i] = (float)((rand() % 200 - 100) / 128.0); }
    for (uint32_t i = 0; i < k2; i++) z2[i] = (float)((rand() % 200 - 100) / 64.0);
    for (uint32_t i = 0; i < D * k2; i++) { U2[i] = (float)((rand() % 200 - 100) / 128.0);
                                            V2[i] = (float)((rand() % 200 - 100) / 128.0); }
    for (uint32_t i = 0; i < D; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);

    uint16_t *p1 = mk_payload(k1, z1, U1, V1), *p2 = mk_payload(k2, z2, U2, V2);
    float want[D] = {0}, got[D] = {0};
    ref_apply(p1, k1, x, want);
    ref_apply(p2, k2, x, want);          /* 参照 = 两段各自 apply 后相加 */

    ds4_zchain *base = mk_chain(2, k1, p1, 0u);
    ds4_zchain *add  = mk_chain(2, k2, p2, 0u);
    int n = ds4_zfinetune_merge(base, add);
    if (n != 1) { printf("FAIL: 合并层数 %d ≠ 1\n", n); return 1; }
    if (base->layer[0].zl.zlk != k1 + k2) {
        printf("FAIL: 合并后秩 %u ≠ %u\n", base->layer[0].zl.zlk, k1 + k2); return 1; }
    ref_apply(base->layer[0].zl.zlm, k1 + k2, x, got);

    double worst = 0;
    for (uint32_t i = 0; i < D; i++) {
        const double d = fabs((double)got[i] - want[i]);
        if (d > worst) worst = d;
    }
    /* fp16 载荷原值搬运 ⇒ 应当逐位相等; 只留浮点求和次序的余量。 */
    if (!(worst < 1e-6)) { printf("FAIL: 拼接后 apply 与两段相加差 %.3g\n", worst); return 1; }
    printf("ok  拼接=两段相加(最大差 %.3g, 秩 %u+%u=%u)\n", worst, k1, k2, base->layer[0].zl.zlk);

    /* 守规矩: 层数不一致拒绝 */
    ds4_zchain *bad_nl = mk_chain(3, k2, p2, 0u);
    if (ds4_zfinetune_merge(base, bad_nl) != -1) { printf("FAIL: 层数不一致没被拒绝\n"); return 1; }
    printf("ok  层数不一致被拒绝\n");

    /* 守规矩: 乘性微调拒绝(加性与乘性不能按秩合并) */
    ds4_zchain *bad_mul = mk_chain(2, k2, p2, 1u);
    if (ds4_zfinetune_merge(base, bad_mul) != -1) { printf("FAIL: 乘性微调没被拒绝\n"); return 1; }
    printf("ok  乘性微调被拒绝\n");
    printf("t_zfinetune: PASS\n");
    return 0;
}
