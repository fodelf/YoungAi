/* cuda_internal.cuh — ds4_cuda.cu 机械拆分的共享前奏(聚合根第一个 include)。
 * 单 TU 共享前奏: system/CUDA include, GPU 契约头, 网格封顶, 量化块类型, iq2 表, q4_K kernel 前向声明。
 * 单 TU: 所有 .inc.cu 分片由 ds4_cuda.cu 按原文件顺序纹理包含, 行为零改动。 */
#include <cuda_runtime.h>

/* 网格封顶(2026-08-21 复查): 这些 384 是 08-20 在 decode 形状上标定的"48SM×4驻留块",
 * 属于写死的调优常量。做成可调以便复测(DS4_CUDA_GRID_CAP), 默认仍 384。 */
/* decode 期只能在非 capture 时机分配的 scratch(split-K f16 partial 等)统一预建入口(定义在
 * cuda_api_matmul_1); token graph 开捕获前必须调一次 —— 否则第一次解码就在 capture 里,
 * 惰性分配被闸掉, kernel 走另一条归约序(09-05 定罪: 图/直发逐位分叉的根因)。 */
static void cuda_decode_scratch_prepare(void);
static unsigned ds4_grid_cap(void) {
    static unsigned v = 0u;
    if (v == 0u) { const char *e = ((const char *)0) /* DS4_CUDA_GRID_CAP: 路径开关已删(2026-08-22 隐形炸弹清理) */; v = e ? (unsigned)atoi(e) : 384u; }
    return v;
}

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cublasLt.h>   /* V4.1 预填 NVFP4 张量核路(cuda_v41_nvfp4.inc.cu)要块缩放 matmul */
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>

/* 单 token 解码注意力分行核(cuda_attn_kernels_6 / cuda_api_attention_4): 部分和暂存预建 +
 * 发射入口; 两个解码 attention 入口(cuda_api_attention_1)在 n_tokens==1 时先走它。 */
static void cuda_attn_split_scratch_prepare(void);
static int attention_decode_split_launch(float *heads, const float *sinks, const float *q,
                                         const float *raw_kv, const uint8_t *comp_kv, const int32_t *topk,
                                         uint32_t raw_cap, uint32_t raw_start, uint32_t raw_first_idx,
                                         uint32_t raw_count, uint32_t visible_comp, uint32_t comp_count,
                                         uint32_t n_head);
static int attention_decode_split_tokens(float *heads, const float *sinks, const float *q, const float *raw_kv,
                                         const uint8_t *comp_kv, const int32_t *topk, uint32_t top_k,
                                         uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                                         uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio,
                                         uint32_t n_head);   /* 小批逐 token 分行核(cuda_api_attention_4), 与解码同轨 */
/* f32→f16 逐元素转换核(定义在 cuda_embed_norm_kernels_1); lifecycle 里的 ds4_gpu_tensor_copy_f32_to_f16 先用 */
__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n);
/* 批路静态注意力 GEMM 版(cuda_api_attention_5): prefill 批对全部可见压缩行, n_comp 大时替代逐 token 在线核 */
static int attention_static_gemm_launch(float *heads, const float *sinks, const float *q, const float *raw_kv,
                                        const uint8_t *comp_kv, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
                                        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t window,
                                        uint32_t ratio, uint32_t n_head);
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

/* ===== 编程式依赖发射(PDL, 2026-09-07) =====
 * 解码一个 token 发 ~1900 个 kernel(nsys 逐核账 prof_decode_account.sh: 2048 ctx 空隙
 * 2.7 ms/token, 每个核边界 ~1.4 µs 的图节点派发延迟 + 前一核的尾波)。PDL 让下一个核在前一
 * 个核还没排空时就上 SM: 发射端带 programmaticStreamSerialization 属性, 核内第一句
 * DS4_PDL_WAIT() 等前序 grid 完成并刷内存后才碰任何全局内存 ⇒ 数值与串行发射逐位相同。
 * 规矩: 只有核体第一句是 DS4_PDL_WAIT() 的核才许用 ds4_launch_pdl 发射(没等就读=读旧值);
 * 普通 <<<>>> 发射的核里这句是空转(未编程式发射时 cudaGridDependencySynchronize 立即返回)。
 * 流捕获会把该属性变成图的编程式边(CUDA ≥12.3), 四相图 ExecUpdate 拓扑同构不受影响。
 * 前序若是 memset/memcpy 节点或跨流事件, 退化为普通全依赖, 同样正确只是没重叠。 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define DS4_PDL_WAIT() cudaGridDependencySynchronize()
/* 触发放在等待之后、核体最前: 本核所有 block 都已上 SM(多波核=最后一波已排定)时下一核才可发射,
 * 不会抢本核未排定 block 的槽位; 下一核的 block 只是提前占空槽在 WAIT 里等 ⇒ 尾波被填满。 */
#define DS4_PDL_TRIGGER() cudaTriggerProgrammaticLaunchCompletion()
#else
#define DS4_PDL_WAIT() ((void)0)
#define DS4_PDL_TRIGGER() ((void)0)
#endif
template <typename... KArgs, typename... Args>
static inline cudaError_t ds4_launch_pdl(void (*kern)(KArgs...), dim3 grid, dim3 block,
                                         size_t shm, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = shm; cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kern, args...);
}

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

/* 前向声明: 解码 Q4_K 稠密 gemv tile 核族的主机发射(定义在 cuda_q4k_tile.inc.cu, 在 dot 辅助之后;
 * 调用点 cuda_api_matmul_1 / cuda_api_attention_2,3 / cuda_qk_warp_2,4 在前) + q8_K 激活量化核
 * (定义在 cuda_qk_warp_1)。 */
__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows);
static int q4k_tile_supported(uint32_t blocks, uint32_t rows);
static int q4k_tile_launch(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t blocks,
                           uint32_t out_dim, uint32_t n_tok);
static int q4k_tile_pair_launch(float *out0, float *out1, const char *w0, const char *w1, const cuda_block_q8_K *xq,
                                uint32_t blocks, uint32_t out0_dim, uint32_t out1_dim);
static int q4k_tile_grouped_launch(float *low, const char *w, const int8_t *xq, const float *xs, uint32_t blocks,
                                   uint32_t rank, uint32_t n_groups, uint32_t n_tok);
static int q4k_rows_multi_launch(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t blocks,
                                 uint32_t out_dim, uint32_t n_tok);   /* 17..32 块行小批(verify), cuda_q4k_tile */
static int q4k_rows_q8_0_multi_launch(float *out, const char *w, const int8_t *xq, const float *xs, uint32_t kblocks,
                                      uint32_t out_dim, uint32_t n_tok);   /* out_b 小批, q8_0 激活 = 解码同轨 */
static int q4k_hc_expand_launch(float *out_hc, float *block_out, const float *residual_hc, const float *split,
                                const char *w, const int8_t *xq, const float *xs, uint32_t kblocks, uint32_t out_dim,
                                uint32_t n_embd, uint32_t n_hc);

