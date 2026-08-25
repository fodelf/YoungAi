/* test_zmod.c — ut_zmod_parity: 引擎 zl_apply 双路数值对拍(离线, 无模型)。
 *
 * 2026-08-26 引擎 z 应用切换 ds4_z 模块(反修与引擎复用同一份实现)后, 同一份
 * type6 载荷走 [模块路(zmod, f32)] 与 [fp16 旧路(zmod=NULL 回落)] 必须同数:
 * 差异只许是 f32 vs f64 累加的尾位(闸 1e-4 相对)。两个用例:
 *   ① 常规修正(信任域不触发)  ② 大修正(‖Δ‖ > tr·‖routed‖, 夹持路径一致性)
 */
#include "../../ds4_zchain.h"
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

int unit_zmod(void) {
    int fails = 0;
    fails += zm_case(512, 8, 1.0f, "常规");
    fails += zm_case(512, 8, 400.0f, "夹持");   /* 大 z ⇒ ‖Δ‖ 超 0.5·‖routed‖, 走信任域 */
    return fails;
}
