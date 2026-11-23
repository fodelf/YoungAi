/* core_ptrain_cosine.c — 默认关闭的顶层局部冻结下界与余弦学习率退火(103轮)。
 * 默认 PT_TRAIN_LAYER_LO=0、PT_COSINE_MIN_RATIO=1.0f 与原训练器逐位等价。 */
#include "core_ptrain_split.h"
#include <math.h>
#ifndef DS4_NO_GPU

#ifndef PT_TRAIN_LAYER_LO
#define PT_TRAIN_LAYER_LO 0u
#endif
#ifndef PT_COSINE_MIN_RATIO
#define PT_COSINE_MIN_RATIO 1.0f
#endif

static uint32_t g_pt_cos_total;

uint32_t pt_train_lo(const pt_cfg *c) {
    const uint32_t floor_lo = (uint32_t)PT_TRAIN_LAYER_LO;
    return c->layer_lo > floor_lo ? c->layer_lo : floor_lo;
}

static double pt_cos_eval(uint32_t step, uint32_t total, double rmin) {
    if (rmin >= 1.0 || total <= 1u) return 1.0;
    const double t = step <= 1u ? 0.0 : (step >= total ? 1.0 : (double)(step - 1u) / (double)(total - 1u));
    return rmin + 0.5 * (1.0 - rmin) * (1.0 + cos(3.14159265358979323846 * t));
}

bool pt_cosine_init(uint32_t total) {
    g_pt_cos_total = total;
    const double rmin = (double)PT_COSINE_MIN_RATIO;
    if (rmin >= 1.0 && PT_TRAIN_LAYER_LO == 0u) return true;
    const double s0 = pt_cos_eval(1u, 11u, rmin), sm = pt_cos_eval(6u, 11u, rmin), s1 = pt_cos_eval(11u, 11u, rmin);
    const bool ok = rmin > 0.0 && rmin <= 1.0 &&
                    fabs(s0 - 1.0) < 1e-12 && fabs(s1 - rmin) < 1e-12 &&
                    fabs(sm - 0.5 * (1.0 + rmin)) < 1e-12;
    fprintf(stderr, "[cosine] known_answer total=%u min_ratio=%.6g layer_lo=%u s0=%.6g sm=%.6g s1=%.6g %s\n",
            total, rmin, (unsigned)PT_TRAIN_LAYER_LO, s0, sm, s1, ok ? "PASS" : "FAIL");
    return ok;
}

float pt_cosine_scale(uint32_t step) {
    const double rmin = (double)PT_COSINE_MIN_RATIO;
    const float sc = (float)pt_cos_eval(step, g_pt_cos_total, rmin);
    if (rmin < 1.0)
        fprintf(stderr, "[cosine] step=%u/%u scale=%.6f\n", step, g_pt_cos_total, (double)sc);
    return sc;
}
#endif
typedef int ds4_core_ptrain_cosine_nonempty_tu;
