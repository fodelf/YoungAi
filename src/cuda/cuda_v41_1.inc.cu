/* cuda_v41_1.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 批前向原语 ①(2026-09-12 战役 P2)。
 * 稠密 fp4x32 GEMM / 分组投影 / 嵌入 / bf16 舍入 / hyper-connection 三件 / RMSNorm / 加 / 展开。
 * 契约见 ds4_gpu_v41.h。口径: f32 计算, 官方 bf16 模块边界处显式舍 bf16。
 * 暂存: 自管 grow-only 槽(cuda_tmp_alloc 全局只有一块, 同一发里要两块就撞)。 */
#include "src/common/ds4_fp8.h"

typedef struct { void *p; uint64_t cap; } v41_scratch;
/* 暂存槽: f16 那三个(g_v41_w16/x16/w16b)随 clear.md C0 的 f16 路一起删了, 只剩这一个杂项槽 */
static v41_scratch g_v41_misc;
/* ---- 块对角(wo_a)预填: FP4 权重 → bf16 张量核 ----
 * ★2026-09-15 clear.md C0: 这条路取代了原来的 f16 暂存 + cuBLAS hgemm, 引擎里再没有 __half★
 *
 * 【为什么是 bf16 而不是 NVFP4】同机器状态 A/B 实测(主尺 2048 token / 块 512):
 *   f16 老路  20.8~21.1 s  PPL 15.8403
 *   NVFP4     20.6~20.8 s  PPL 16.4556   ← 速度一模一样, 质量退 3.9%
 * wo_a 的输入是 64 个头拼出来的 32768 维注意力输出, 把它降到 FP4 激活(每 16 个一个 e4m3 缩放)
 * 扛不住; 而这个矩阵本来就不是预填的瓶颈, 降精度一毫秒都换不回来 ⇒ 判负, NVFP4 分组路已删。
 *
 * 【bf16 为什么是无损的, 不是"退一档"】FP4 的幅值只有 {0,.5,1,1.5,2,3,4,6}, 3 个尾数位装得下,
 * 缩放又是 2 的整数幂 ⇒ **fp4x32 权重转 bf16 逐位精确**; 激活本来就被各消费核舍在 bf16 格点上,
 * 转过去同样精确。所以这条路与 f16 老路数值完全一致(f16 对这些值也精确), 只是不再出现 f16 类型。
 * 盘上格式仍然是 FP4, bf16 只是喂张量核的那一瞬间的形状。 */
static v41_scratch g_v41_wbf, g_v41_xbf;
__global__ static void v41_fp4x32_to_bf16_kernel(__nv_bfloat16 *out, const uint8_t *w, uint64_t nblk) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const uint8_t *p = w + b * 17u;
    const float s = ds4_e8m0_to_f32(p[16]);
    __nv_bfloat16 *o = out + b * 32u;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        o[2 * j]     = __float2bfloat16(ds4_fp4_nibble_to_f32(p[j] & 0x0F) * s);
        o[2 * j + 1] = __float2bfloat16(ds4_fp4_nibble_to_f32(p[j] >> 4) * s);
    }
}
__global__ static void v41_x_to_bf16_kernel(__nv_bfloat16 *out, const float *x, uint64_t n);
/* 解码小批(n ≤ 8)走 cuda_v41_4.inc.cu 的融合核(不落 f16); 原型先声明, 定义在后面的分片(同一 TU) */
#define V41_GEMV_MAX_TOK 8u
static int v41_fp4x32_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                           const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                           uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what);
static int v41_vq_fused_moe(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                            const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                            const float *gr);
/* 预填稠密 GEMM 的 NVFP4 路(定义在 cuda_v41_nvfp4.inc.cu, 同一 TU) */
static int v41_matmul_nvfp4(const void *model_map, uint64_t model_size, uint64_t off,
                            uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                            uint32_t n_tok, const char *what);
/* cuBLAS 句柄的流: 我们的核都发在 PTDS(编译 -default-stream per-thread 下的"流 0"), 但 cuBLAS 库不是按这个
 * 开关编的, 递给它的 0 是 legacy 流 —— 两者的隐式同步靠文档一句话, 实测整批路偶发脏读(同一列全 token 偏/NaN), 与 cuBLAS
 * 参与的路一一对应。显式递 cudaStreamPerThread, 让 cuBLAS 与我们的核同一条流, 不赌隐式同步。 */
static inline cudaStream_t v41_cublas_stream(void) { return g_cur_stream ? g_cur_stream : cudaStreamPerThread; }
static void *v41_grow(v41_scratch *s, uint64_t bytes, const char *what) {
    if (bytes <= s->cap) return s->p;
    (void)cudaDeviceSynchronize();
    if (s->p) (void)cudaFree(s->p);
    s->p = NULL; s->cap = 0;
    if (cudaMalloc(&s->p, (size_t)bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] %s 暂存分配失败 (%.1f MB)\n", what, (double)bytes / 1048576.0);
        return NULL;
    }
    s->cap = bytes;
    return s->p;
}

__device__ __forceinline__ static float v41_bf16r(float x) {   /* RNE 舍到 bf16 再回 f32 */
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;            /* NaN/Inf 原样 */
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

/* 激活已经落在 bf16 格点上(各消费核出口都舍过), 这里只是换个存法, 不改值 */
__global__ static void v41_x_to_bf16_kernel(__nv_bfloat16 *out, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(x[i]);
}
__global__ static void v41_round_bf16_kernel(float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v41_bf16r(x[i]);
}
int ds4_gpu_v41_round_bf16_tensor(ds4_gpu_tensor *x, uint64_t n) {
    if (!x || x->bytes < n * 4) return 0;
    v41_round_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)x->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 round bf16");
}

/* ★2026-09-15 clear.md C0: 这里原来有四件 f16 的东西, 全删了★
 *   v41_fp4x32_to_f16_kernel / v41_x_to_f16_kernel / v41_gemm_f16(cuBLAS hgemm) / v41_dequant_fp4x32。
 * 它们是预填 wo_a 的老路: 把 fp4x32 权重解成 f16 落暂存, 激活也转 f16, 再逐组 cuBLAS。
 * 现在稠密预填走 v41_matmul_nvfp4(FP4 张量核), 块对角 wo_a 走本文件上面那条 bf16 张量核
 * (wo_a 降 FP4 激活退 3.9% PPL, 判负存档在 cuda_v41_nvfp4.inc.cu 末尾)。
 * ★引擎里再没有 __half★ —— bf16 对 FP4 权重逐位无损, 不是"退一档"。 */

int ds4_gpu_v41_matmul_fp4x32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * 4 || out->bytes < (uint64_t)n_tok * out_dim * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK)
        return v41_fp4x32_gemv(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr, (uint32_t)in_dim,
                               (float *)out->ptr, (uint32_t)out_dim, n_tok, 1u, 0u, 0u, round_out, "v41 fp4x32 gemv");
    /* 预填(n > 8): NVFP4 张量核(cuda_v41_nvfp4.inc.cu)。09-15 S0 实测比原 f16 路的 cuBLAS
     * 快 3.5×, 且暂存字节少 3.5×; 权重在 e4m3 可表示区内逐位恒等, 激活降到 FP4 是质量门的事。 */
    if (!v41_matmul_nvfp4(model_map, model_size, weight_offset, in_dim, out_dim,
                          (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 fp4x32 nvfp4")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;
}

int ds4_gpu_v41_grouped_matmul_fp4x32_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                             uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (heads->bytes < (uint64_t)n_tok * in_all * 4 || low->bytes < (uint64_t)n_tok * out_all * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK) {   /* 小批: 八组一发 GEMV(grid.y=组), 输入/输出按组段取(行步长 in_all/out_all) */
        if (group_dim % 32u) return 0;
        return v41_fp4x32_gemv(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr, (uint32_t)in_all,
                               (float *)low->ptr, (uint32_t)out_all, n_tok, n_groups, (uint32_t)group_dim, (uint32_t)rank, round_out, "v41 wo_a gemv");
    }
    /* 预填(n > 8): FP4 权重 → bf16 张量核, 每组一发(见上面那段"为什么是 bf16"的账) */
    const uint64_t nblk = out_all * (group_dim / 32u);
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * 17u, "v41 wo_a");
    __nv_bfloat16 *wb = (__nv_bfloat16 *)v41_grow(&g_v41_wbf, nblk * 32u * sizeof(__nv_bfloat16), "v41 wo_a bf16");
    if (!w || !wb) return 0;
    v41_fp4x32_to_bf16_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, g_cur_stream>>>(wb, w, nblk);
    if (!cuda_ok(cudaGetLastError(), "v41 wo_a fp4→bf16")) return 0;
    const uint64_t xn = (uint64_t)n_tok * in_all;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, xn * sizeof(__nv_bfloat16), "v41 heads bf16");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(xb, (const float *)heads->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 heads→bf16")) return 0;
    /* 每组一发 GEMM: 输入按行步长 in_all 取第 g 段, 输出按行步长 out_all 写第 g 段 */
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    for (uint32_t g = 0; g < n_groups; g++) {
        cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rank, (int)n_tok, (int)group_dim, &alpha,
                                         wb + (uint64_t)g * rank * group_dim, CUDA_R_16BF, (int)group_dim,
                                         xb + (uint64_t)g * group_dim, CUDA_R_16BF, (int)in_all, &beta,
                                         (float *)low->ptr + (uint64_t)g * rank, CUDA_R_32F, (int)out_all,
                                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "v41 wo_a bf16 gemm")) return 0;
    }
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;
}

/* 非 FP4 权重的小批 GEMV(cuda_v41_gemv_highprec.inc.cu, 同一 TU 后面定义) */
static int v41_f32_gemv(const float *w, uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                        uint32_t n_tok, const char *what);
static int v41_bf16_gemv(const __nv_bfloat16 *w, uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                         uint32_t n_tok, const char *what);
static int v41_fp8blk_gemv(const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t out_dim,
                           const float *x, float *out, uint32_t n_tok, const char *what);
static int v41_fp8blk_to_bf16(__nv_bfloat16 *o, const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t rows);

int ds4_gpu_v41_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t wbytes = in_dim * out_dim * 4;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const float *W = (const float *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 f32 w");
    if (!W) return 0;
    /* 解码/小批走自家 GEMV: cuBLAS 在 n=1 时把一发 Sgemm 拆成 200 多个小核(见 v41_f32_gemv 头注) */
    if (n_tok <= V41_GEMV_MAX_TOK && (in_dim % 128u) == 0u)
        return v41_f32_gemv(W, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 f32 gemv");
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                    W, (int)in_dim, (const float *)x->ptr, (int)in_dim, &beta, (float *)out->ptr, (int)out_dim);
    return cublas_ok(st, "v41 f32 gemm");
}

/* BF16 权重(官方原生精度) × f32 激活 → f32。clear.md C1 起路由 gate / compressor / indexer
 * 投影都存 BF16 —— 转换器不再把它们展开成 f32, 每 token 少读 0.20 GB, 而值是原件原样, 一位没动。
 * 预填(n > 8)直接把 mmap 上的 bf16 喂 cuBLAS, 连一次格式转换都不用。 */
int ds4_gpu_v41_matmul_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                   uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                   const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t wbytes = in_dim * out_dim * 2;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const __nv_bfloat16 *W = (const __nv_bfloat16 *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 bf16 w");
    if (!W) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK && (in_dim % 256u) == 0u)
        return v41_bf16_gemv(W, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 bf16 gemv");
    /* 预填: 激活转 bf16 一发 GEMM(激活本来就落在 bf16 格点上, 转过去不改值) */
    const uint64_t xn = (uint64_t)n_tok * in_dim;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, xn * sizeof(__nv_bfloat16), "v41 bf16 x");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(xb, (const float *)x->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 x→bf16")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                     W, CUDA_R_16BF, (int)in_dim, xb, CUDA_R_16BF, (int)in_dim, &beta,
                                     (float *)out->ptr, CUDA_R_32F, (int)out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "v41 bf16 gemm");
}

/* engram wkv: FP8(e4m3 + 32×32 块 ue8m0)权重 × f32 行 → f32。clear.md C1 起盘上就是官方格式,
 * 不再展开成 f16(字节减半, 值更准 —— 见 cuda_v41_gemv_highprec.inc.cu 的账)。
 * 这同时把 engram 从 V4 的 ds4_gpu_matmul_f16_tensor 上摘了下来, V4.1 前向不再借 V4 的核。 */
int ds4_gpu_v41_matmul_fp8blk_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || n_tok == 0) return 0;
    const uint64_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u;
    const uint64_t wbytes = in_dim * out_dim + sbr * sbc;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *W = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 fp8blk w");
    if (!W) return 0;
    const uint8_t *SC = W + in_dim * out_dim;
    if (n_tok <= V41_GEMV_MAX_TOK && (in_dim % 512u) == 0u)
        return v41_fp8blk_gemv(W, SC, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 fp8blk gemv");
    if (!g_cublas_ready) return 0;
    const uint64_t wn = in_dim * out_dim;
    __nv_bfloat16 *wb = (__nv_bfloat16 *)v41_grow(&g_v41_wbf, wn * sizeof(__nv_bfloat16), "v41 wkv bf16");
    if (!wb || !v41_fp8blk_to_bf16(wb, W, SC, in_dim, out_dim)) return 0;
    const uint64_t xn = (uint64_t)n_tok * in_dim;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, xn * sizeof(__nv_bfloat16), "v41 wkv x bf16");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(xb, (const float *)x->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 wkv x→bf16")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                     wb, CUDA_R_16BF, (int)in_dim, xb, CUDA_R_16BF, (int)in_dim, &beta,
                                     (float *)out->ptr, CUDA_R_32F, (int)out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "v41 wkv bf16 gemm");
}

/* ---- 嵌入: 行 = token, 每行 in_dim/32 块 ---- */
__global__ static void v41_embed_kernel(float *out, const int32_t *tok, const uint8_t *w, uint32_t n_vocab, uint32_t n_embd) {
    const uint32_t t = blockIdx.x, nb = n_embd / 32u;
    int32_t id = tok[t]; if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        const uint8_t *p = w + ((uint64_t)id * nb + b) * 17u;
        const float s = ds4_e8m0_to_f32(p[16]);
        float *o = out + (uint64_t)t * n_embd + b * 32u;
        for (int j = 0; j < 16; j++) {
            o[2 * j] = v41_bf16r(ds4_fp4_nibble_to_f32(p[j] & 0x0F) * s);
            o[2 * j + 1] = v41_bf16r(ds4_fp4_nibble_to_f32(p[j] >> 4) * s);
        }
    }
}
int ds4_gpu_v41_embed_fp4x32_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                    uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab,
                                    uint32_t n_tok, uint32_t n_embd) {
    if (!out || !tokens || n_tok == 0 || (n_embd % 32u)) return 0;
    const uint64_t wbytes = (uint64_t)n_vocab * (n_embd / 32u) * 17u;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 embed");
    if (!w || out->bytes < (uint64_t)n_tok * n_embd * 4) return 0;
    v41_embed_kernel<<<n_tok, 256, 0, g_cur_stream>>>((float *)out->ptr, (const int32_t *)tokens->ptr, w, n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "v41 embed");
}


/* RMSNorm 带权 → bf16(官方 RMSNorm: f32 算, weight 是 bf16 值, 结果 .to(bf16)) */
__global__ static void v41_rms_norm_kernel(float *out, const float *x, const float *w, uint32_t dim, float eps) {
    const uint32_t r = blockIdx.x; const float *xr = x + (uint64_t)r * dim; float *o = out + (uint64_t)r * dim;
    float s = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) o[i] = v41_bf16r(w[i] * (xr[i] * inv));
}
int ds4_gpu_v41_rms_norm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset, uint32_t dim, uint32_t n_tok, float eps) {
    if (!out || !x) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset, (uint64_t)dim * 4, "v41 norm w");
    if (!w) return 0;
    v41_rms_norm_kernel<<<n_tok, 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)x->ptr, w, dim, eps);
    return cuda_ok(cudaGetLastError(), "v41 rms norm");
}

__global__ static void v41_add_kernel(float *a, const float *b, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
int ds4_gpu_v41_add_tensor(ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n) {
    if (!a || !b) return 0;
    v41_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 add");
}
__global__ static void v41_expand_hc_kernel(float *hc, const float *x, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float v = x[(uint64_t)n * n_embd + d];
    for (uint32_t c = 0; c < n_hc; c++) hc[((uint64_t)n * n_hc + c) * n_embd + d] = v;
}
int ds4_gpu_v41_expand_hc_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!hc || !x) return 0;
    v41_expand_hc_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_cur_stream>>>((float *)hc->ptr, (const float *)x->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 expand hc");
}
