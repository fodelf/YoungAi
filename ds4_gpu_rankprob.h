/* 训练侧概率排序：原 KL 保留，额外项只处理学生榜首错排。 */
#ifndef DS4_GPU_RANKPROB_H
#define DS4_GPU_RANKPROB_H
#include "ds4_gpu_bwd.h"
#ifdef __cplusplus
extern "C" {
#endif
/* lambda=0 直接调用原 KL，不分配暂存。lambda>0 先读原 logits 计算排序梯度，
 * 再算原 KL 并相加，避免原地 KL 覆盖概率；loss 未归一，梯度乘 scale。 */
int ds4_gpu_bwd_kl_rankprob_tensor(ds4_gpu_tensor *g, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits,
    uint32_t row0, uint32_t m, uint32_t vocab, const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp,
    const ds4_gpu_tensor *rest, const ds4_gpu_tensor *w, uint32_t k, float scale, float lambda);
#ifdef __cplusplus
}
#endif
#endif
