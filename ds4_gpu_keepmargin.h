#ifndef DS4_GPU_KEEPMARGIN_H
#define DS4_GPU_KEEPMARGIN_H
#include "ds4_gpu_bwd.h"
#ifdef __cplusplus
extern "C" {
#endif
/* 先保存原logits上的稀疏保护项，再算原KL；mask只来自冻结父本的训练预测。 */
int ds4_gpu_bwd_kl_keepmargin_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *,
    uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
    uint32_t, float, const ds4_gpu_tensor *, float, float, float);
#ifdef __cplusplus
}
#endif
#endif
