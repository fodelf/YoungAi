/* cuda_internal.cuh — ds4_cuda.cu 机械拆分的共享前奏(聚合根第一个 include)。
 * 单 TU 共享前奏: system/CUDA include, GPU 契约头, 网格封顶, 量化块类型, iq2 表, q4_K kernel 前向声明。
 * 单 TU: 所有 .inc.cu 分片由 ds4_cuda.cu 按原文件顺序纹理包含, 行为零改动。 */
#include <cuda_runtime.h>

/* 网格封顶(2026-08-21 复查): 这些 384 是 08-20 在 decode 形状上标定的"48SM×4驻留块",
 * 属于写死的调优常量。做成可调以便复测(DS4_CUDA_GRID_CAP), 默认仍 384。 */
static unsigned ds4_grid_cap(void) {
    static unsigned v = 0u;
    if (v == 0u) { const char *e = ((const char *)0) /* DS4_CUDA_GRID_CAP: 路径开关已删(2026-08-22 隐形炸弹清理) */; v = e ? (unsigned)atoi(e) : 384u; }
    return v;
}

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <vector>

/* v2.2 VQ 码本解析(DQVL/DQVQ)。纯 C、零依赖、static inline —— 量化器/Metal/CUDA
 * 共用同一份解码, 保证三端逐位同义。 */
#include "vq_fmt.h"

/* 反修产物契约(λ clamp/z 秩上限): 与引擎 ds4_zchain.c、工具回放同一份定义 */
#include "src/common/ds4_amp_fmt.h"

/* GPU 契约头(子头带 extern "C" 守卫)。API 定义因此直接继承 C 链接与签名检查:
 * 实现与契约不一致会在编译期报 conflicting declaration, 而不是静默的 ABI 错位。
 * ds4_gpu_tensor 在契约里是 opaque typedef, 下面补上 CUDA 侧的具体定义;
 * ds4_gpu_residual_set 直接用契约里的那一份(历史上这里手抄过一份镜像)。 */
#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

/* 前向声明: q4_K attn_output kernel 组(定义在 q4_K dot 辅助之后, 调用点在前) */
__global__ static void grouped_q4_K_a_preq_warp8_kernel(
        float *low, const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t kblocks, int use_dp4a);
__global__ static void matmul_q4_K_hc_expand_preq_warp8_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc,
        uint64_t kblocks, int has_add, int use_dp4a);
__global__ static void grouped_q4_K_a_preq_warp8_dp4a_kernel(
        float *low, const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t kblocks);
__global__ static void matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc,
        uint64_t kblocks, int has_add);

