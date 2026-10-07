/* cuda_internal.cuh — ds4_cuda.cu 机械拆分的共享前奏(聚合根第一个 include)。
 * 单 TU 共享前奏: system/CUDA include, GPU 契约头, 网格封顶, 量化块类型, iq2 表, q4_K kernel 前向声明。
 * 单 TU: 所有 .inc.cu 分片由 ds4_cuda.cu 按原文件顺序纹理包含, 行为零改动。 */
#include <cuda_runtime.h>

/* decode 期只能在非 capture 时机分配的 scratch(split-K f16 partial 等)统一预建入口(定义在
 * cuda_api_matmul_1); token graph 开捕获前必须调一次 —— 否则第一次解码就在 capture 里,
 * 惰性分配被闸掉, kernel 走另一条归约序(09-05 定罪: 图/直发逐位分叉的根因)。 */
static void cuda_decode_scratch_prepare(void);
/* 本卡 SM 数(查一次缓存)。属性查不到就按 48 走(GB10 的数): 查不到只会发生在设备根本没起来的情况, 那时后面每一发核都会报错。 */
static unsigned ds4_sm_count(void) {
    static unsigned v = 0u;
    if (v == 0u) {
        int dev = 0, n = 0;
        (void)cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) { (void)cudaGetLastError(); n = 48; }
        v = (unsigned)n;
    }
    return v;
}
/* 网格封顶 = SM 数 × 8。08-20 阶梯审判在 decode 形状上定的 384(384 比 192 +4~8 GB/s)就是 48 SM × 8;
 * 2026-10-07 前写死 384(DS4_CUDA_GRID_CAP 旋钮删后的残值), 换一块卡就按 GB10 的 SM 数封顶 —— 改按本卡 SM 数自量, GB10 上仍 384。 */
static unsigned ds4_grid_cap(void) { return ds4_sm_count() * 8u; }
/* 统一内存判据(运行时; 2026-10-07 前是编译宏 DS4_CUDA_SPARK_HBM_CACHE, make cuda-spark 才开): GPU 经主机页表访问整机内存
 * (GB10/Grace 的 ATS)或 iGPU 共享内存(Jetson)时, "显存"就是整机内存 —— 启动缓存按整机内存减余量收编全部权重含专家;
 * 独显(含开了 HMM 的: pageable access 可为 1 但 ATS 为 0)显存是独立池, 只缓骨架、预算按显存算。 */
static int cuda_unified_memory_host(void) {
    static int v = -1;
    if (v < 0) {
        int dev = 0, ats = 0, integ = 0;
        (void)cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) != cudaSuccess) { (void)cudaGetLastError(); ats = 0; }
        if (cudaDeviceGetAttribute(&integ, cudaDevAttrIntegrated, dev) != cudaSuccess) { (void)cudaGetLastError(); integ = 0; }
        v = (ats || integ) ? 1 : 0;
    }
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


/* ==== PDL(programmatic dependent launch, 2026-09-23): 解码图里让下一个核在上一个核还没跑完时就发射 ====
 * 为什么: 走图之后核间空隙只剩 0.1 µs, 真正的浪费是①单 block 小核(hc_fused/rms/router 各 4~9 µs)跑的时候另外 47 个 SM 干等,
 * ②每个 GEMV 起步要先等第一批权重从 DRAM 回来。权重是常量, 与上一步无关 ⇒ 下一个核可以先把权重读进 shared, 再等上一步的激活。
 * 怎么接: 捕获收尾时(cuda_decode_graph.inc.cu)把"核 → 已登记核"的边改成程序化边(出口 = 上游所有 block 已开跑);
 * 已登记的核必须在读任何"前面的核产出的东西"之前调 v41_pdl_wait() —— 它等上游**完成且内存可见**。
 * 不经 PDL 发射(直发路/没被改边)时 griddepcontrol.wait 立即返回, 所以同一个核两条路都能用。
 * ★出错会怎样★: 登记了却漏了 wait(或 wait 之前就读了激活/写了全局)⇒ 读到上一步的半成品, 不报错, 输出悄悄变。
 * 门 = 走图 vs 直发逐字节同(d0a 的 prof.out 与 nsys.out)。 */
__device__ __forceinline__ static void v41_pdl_wait(void) { asm volatile("griddepcontrol.wait;" ::: "memory"); }
#define V41_PDL_MAX 64
static const void *g_v41_pdl_ready[V41_PDL_MAX];
static int g_v41_pdl_n = 0;
/* 发射端调: 声明"这个核函数已在读上游产出之前调了 v41_pdl_wait"。重复登记无害。 */
static void v41_pdl_register(const void *fn) {
    for (int i = 0; i < g_v41_pdl_n; i++) if (g_v41_pdl_ready[i] == fn) return;
    if (g_v41_pdl_n < V41_PDL_MAX) g_v41_pdl_ready[g_v41_pdl_n++] = fn;
}
static int v41_pdl_is_ready(const void *fn) {
    for (int i = 0; i < g_v41_pdl_n; i++) if (g_v41_pdl_ready[i] == fn) return 1;
    return 0;
}
