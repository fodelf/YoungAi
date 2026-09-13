/* ds4_gpu.h — GPU 后端契约伞头(重构阶段3: 原 1242 行按 6 个既有章节拆成子头,
 * 本头保名零改动服务全部消费方)。Metal(ds4_metal.m 系)与 CUDA(ds4_cuda.cu 系)
 * 共同实现这份契约; 每个子头 ≤500 行:
 *   ds4_gpu_core.h   tensor/命令生命周期
 *   ds4_gpu_embed.h  embedding 与 indexer
 *   ds4_gpu_dense.h  稠密投影/norm/RoPE/KV rounding
 *   ds4_gpu_attn.h   KV 压缩与 attention
 *   ds4_gpu_moe.h    router/共享专家/routed MoE
 *   ds4_gpu_hc.h     Hyper-Connection */
#ifndef DS4_GPU_H
#define DS4_GPU_H

#include "ds4_gpu_core.h"
#include "ds4_gpu_embed.h"
#include "ds4_gpu_dense.h"
#include "ds4_gpu_attn.h"
#include "ds4_gpu_moe.h"
#include "ds4_gpu_hc.h"
#include "ds4_gpu_v41.h"   /* DeepSeek V4.1 批前向原语(2026-09-12) */

#endif
