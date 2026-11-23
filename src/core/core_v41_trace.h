#ifndef DS4_CORE_V41_TRACE_H
#define DS4_CORE_V41_TRACE_H
#include "core_internal.h"
#ifndef DS4_NO_GPU
/* 只读张量探针(默认 NULL, 零开销): 在指定层捕获补丁/BF16舍入前后及出口前张量。 */
typedef struct {
    uint32_t layer, row;
    int skip_amp_bf16;
    float *y_moe, *y_pre, *y_post, *hc_post, *hc_pre, *xn_out;
    bool ok;
} v41_trace_tap;
extern v41_trace_tap *g_v41_trace_tap;
void v41_trace_capture_row(const ds4_gpu_tensor *t, uint32_t row, uint32_t cols, float *dst, bool *ok);
#endif
#endif
