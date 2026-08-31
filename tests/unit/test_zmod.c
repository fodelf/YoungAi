/* test_zmod.c — ut_zmod_parity: 引擎 zl_apply 双路数值对拍(离线, 无模型)。
 *
 * 2026-08-26 引擎 z 应用切换 ds4_z 模块(反修与引擎复用同一份实现)后, 同一份
 * type6 载荷走 [模块路(zmod, f32)] 与 [fp16 旧路(zmod=NULL 回落)] 必须同数:
 * 差异只许是 f32 vs f64 累加的尾位(闸 1e-4 相对)。两个用例:
 *   ① 常规修正(信任域不触发)  ② 大修正(‖Δ‖ > tr·‖routed‖, 夹持路径一致性)
 */
#include "../../ds4_zchain.h"
#include "../../ds4_z.h"
#include "ds4_float.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t zm_s = 0x2026A26ULL;
static float zm_rnd(void) {   /* 决定论: 固定种子 LCG, 与运行环境无关 */
    zm_s = zm_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((float)(uint32_t)(zm_s >> 32) / 4294967296.0f) * 2.0f - 1.0f;
}

static int zm_case(uint32_t d, uint32_t k, float zscale, const char *tag) {
    size_t nh = (size_t)k + (size_t)d * k + (size_t)d * k;
    uint16_t *pay = malloc(nh * 2);
    for (uint32_t c = 0; c < k; c++) pay[c] = ds4_f64_to_f16(zm_rnd() * zscale);
    for (size_t i = k; i < nh; i++) pay[i] = ds4_f64_to_f16(zm_rnd() * 0.05);

    ds4_zchain_zl a; memset(&a, 0, sizeof a);
    a.zlk = k; a.zdin = d; a.zltr = 0.5f; a.zlm = pay;
    /* 模块路: 镜像 loader 的 fp16→f32 转换 */
    ds4_z zm; memset(&zm, 0, sizeof zm);
    zm.d_in = d; zm.d_out = d; zm.rank = k; zm.k = k;
    zm.z = malloc(k * 4); zm.U = malloc((size_t)d * k * 4); zm.V = malloc((size_t)d * k * 4);
    const uint16_t *hz = pay, *hU = hz + k, *hV = hU + (size_t)d * k;
    for (uint32_t c = 0; c < k; c++) zm.z[c] = ds4_f16_to_f32(hz[c]);
    for (size_t i = 0; i < (size_t)d * k; i++) zm.U[i] = ds4_f16_to_f32(hU[i]);
    for (size_t i = 0; i < (size_t)d * k; i++) zm.V[i] = ds4_f16_to_f32(hV[i]);
    a.zmod = &zm;
    ds4_zchain_zl b = a; b.zmod = NULL;      /* fp16 旧路回落 */

    float *x = malloc(d * 4), *ra = malloc(d * 4), *rb = malloc(d * 4);
    for (uint32_t j = 0; j < d; j++) { x[j] = zm_rnd(); ra[j] = rb[j] = zm_rnd(); }
    ds4_zchain_zl_apply(&a, d, x, ra);
    ds4_zchain_zl_apply(&b, d, x, rb);
    double e2 = 0, n2 = 0;
    for (uint32_t j = 0; j < d; j++) {
        double dd = (double)ra[j] - rb[j];
        e2 += dd * dd; n2 += (double)rb[j] * rb[j];
    }
    double rel = sqrt(e2 / (n2 + 1e-30));
    int bad = !(rel < 1e-4);
    printf("  ut_zmod_parity %-8s d=%u k=%u rel=%.3e %s\n", tag, d, k, rel, bad ? "FAIL" : "PASS");
    free(pay); free(zm.z); free(zm.U); free(zm.V); free(x); free(ra); free(rb);
    return bad;
}

/* ---- ut_4l: 四损失参数在引擎侧真生效(2026-08-26 用户架构落地闸) ----
 * 引擎内调 z: ds4_zchain_posttrain_z 用四损失权闭式重解 z, 对种进去的真 z
 * 必须显著收回(align 口径的相对误差降一个量级以上)。
 * (旧①"前向调制夹持分叉"用例已删 2026-08-31: w4 加权夹持只有 CPU host 实现过,
 * CUDA/Metal/判决尺全是无权 —— 特例整族删除, 夹持契约全线无权。) */
static int fourl_case(void) {
    const uint32_t d = 256, k = 6;
    ds4_z zm; memset(&zm, 0, sizeof zm);
    zm.d_in = d; zm.d_out = d; zm.rank = k; zm.k = k;
    zm.z = malloc(k * 4); zm.U = malloc((size_t)d * k * 4); zm.V = malloc((size_t)d * k * 4);
    float *ztrue = malloc(k * 4);
    for (uint32_t c = 0; c < k; c++) { ztrue[c] = 0.5f + zm_rnd(); zm.z[c] = 0.0f; }
    for (size_t i = 0; i < (size_t)d * k; i++) { zm.U[i] = zm_rnd() * 0.1f; zm.V[i] = zm_rnd() * 0.1f; }

    ds4_zchain_zl a; memset(&a, 0, sizeof a);
    a.zlk = k; a.zdin = d; a.zltr = 0.5f; a.zmod = &zm; a.zlm = (const uint16_t *)zm.z;
    float *w4 = malloc(d * 4);
    for (uint32_t j = 0; j < d; j++) w4[j] = (j < d / 8) ? 8.0f : 0.5f;   /* 前 1/8 维=重要 */

    /* 引擎内调 z: 造 (X,R,Yt), R 由 ztrue 生成, 看闭式重解能否收回 */
    const uint32_t n = 400;
    float *X = malloc((size_t)n * d * 4), *R = malloc((size_t)n * d * 4), *Yt = malloc((size_t)n * d * 4);
    for (uint32_t c = 0; c < k; c++) zm.z[c] = ztrue[c];
    for (uint32_t t = 0; t < n; t++) {
        float *xr = X + (size_t)t * d, *rr = R + (size_t)t * d, *yr = Yt + (size_t)t * d;
        for (uint32_t j = 0; j < d; j++) xr[j] = zm_rnd();
        memset(rr, 0, d * 4);
        ds4_z_apply(&zm, xr, rr);
        for (uint32_t j = 0; j < d; j++) yr[j] = rr[j] + zm_rnd() * 0.01f;
    }
    ds4_zchain_4l l4; memset(&l4, 0, sizeof l4);
    l4.w[0] = 1.0f; l4.w[1] = 0.5f; l4.w[2] = 0.1f; l4.w[3] = 1e-3f;
    l4.seed = 1; l4.dscale = 0.04f; l4.d = d; l4.wnorm = w4;
    ds4_zchain zc; memset(&zc, 0, sizeof zc);
    ds4_zchain_layer lay; memset(&lay, 0, sizeof lay);
    lay.zl = a; lay.l4 = l4;
    zc.n_layer = 1; zc.d_model = d; zc.layer = &lay;
    double e0 = 0, tn = 0;
    for (uint32_t c = 0; c < k; c++) { zm.z[c] = 0.0f; e0 += (double)ztrue[c] * ztrue[c]; tn += (double)ztrue[c] * ztrue[c]; }
    int rc = ds4_zchain_posttrain_z(&zc, 0, X, R, Yt, n);
    double e1 = 0;
    for (uint32_t c = 0; c < k; c++) { double dz = (double)zm.z[c] - ztrue[c]; e1 += dz * dz; }
    double rel0 = sqrt(e0 / (tn + 1e-30)), rel1 = sqrt(e1 / (tn + 1e-30));
    int bad2 = (rc != 0) || !(rel1 < 0.1 * rel0);
    printf("  ut_4l 引擎内调z rc=%d 相对误差 %.3f→%.3f %s\n", rc, rel0, rel1, bad2 ? "FAIL" : "PASS");

    free(zm.z); free(zm.U); free(zm.V); free(ztrue); free(w4);
    free(X); free(R); free(Yt);
    return bad2;
}

int unit_zmod(void) {
    int fails = 0;
    fails += zm_case(512, 8, 1.0f, "常规");
    fails += zm_case(512, 8, 400.0f, "夹持");   /* 大 z ⇒ ‖Δ‖ 超 0.5·‖routed‖, 走信任域 */
    fails += fourl_case();                      /* 四损失参数引擎侧生效闸(2026-08-26) */
    return fails;
}

