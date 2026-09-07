/* cuda_q4k_gemm.inc.cu — 稠密 q4_K 权重的批量(prefill) GEMM 路(2026-09-06)。
 *
 * 为什么: 稠密 q4_K 的 warp 核(cuda_qk_warp_2)是 decode 形态 —— grid.y 按 token 铺开, 每个
 * token 各自把整个矩阵从 L2 重读一遍并做 CUDA 核整数点积。8192 token 的 prefill 光
 * q_b/out_b 这类 18.9 MB 矩阵就要 76 s(15201 发 × 5 ms), 折合 9.3 ms/token, 比 decode 一整个
 * token(33 ms)的四分之一还多; 原版 IQ2 模型 5 月在 GB10 上 prefill 400 t/s 是 f16 骨架走
 * cuBLAS 拿到的。这里对 n_tok ≥ DS4_Q4K_GEMM_MIN_TOK 的调用: 先把 q4_K 按 llama.cpp
 * dequantize_row_q4_K 的公式解成 f16(与 src/common/ds4_deq_q4_K 逐字同式), 再
 * cublasGemmEx(f16×f16→f32 累加)。权重 f16 只在本次调用内存活(静态 scratch 按最大矩阵增长,
 * 不落缓存): 5.7 GB 骨架每 4096 token 块解一遍 = 20 GB 写 + 20 GB 读, 0.2 s/块。
 * 数值: 激活 f32→f16(与 f16 骨架 prefill 同口径), 权重解码精确; decode 路仍是 q8_K 激活 ×
 * q4_K 整数点积。两路不逐位同, 判决走批路五指标(kernel_parity_spark.sh … prefill)。
 * 改了会怎样: 阈值降到 ≤8 会把 verify 小批也拉进来, 那里要的是与单 token 有序路逐位对齐
 * (见 cuda_api_matmul_1 的 f16_exact_batch 注释); 阈值抬到几百则短 prompt 退回慢核。 */
#define DS4_Q4K_GEMM_MIN_TOK 16u

static __half *g_q4k_wf16 = NULL;        /* 解码后的 f16 权重 [out][in], 按需增长 */
static uint64_t g_q4k_wf16_elems = 0;

/* scale/min 解包同 ds4_quantfmt.c q4k_scale_min: 12 字节 scales, 8 个子块各 6 位 */
__device__ __forceinline__ static void dev_q4k_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
           *m = (uint8_t)((q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4)); }
}

/* 一个 block 一行, 256 线程各管 q4_K 块内一个位置 j, 沿行循环所有块。
 * 块布局(144 B): d f16 | dmin f16 | scales[12] | qs[128]; 64 值一对子块: 前 32 取 qs 低 4 位,
 * 后 32 取同一段 qs 的高 4 位。y = d·sc·q − dmin·m, 与 CPU 版同一运算序。 */
__global__ static void q4_K_to_f16_kernel(__half *out, const uint8_t *w, uint32_t blocks, uint32_t rows) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const uint32_t j = threadIdx.x;
    const uint32_t pair = j >> 6, l = j & 63u;
    const int is = (int)(pair * 2u + (l >> 5));
    const uint8_t *row = w + (uint64_t)r * blocks * 144u;
    __half *orow = out + (uint64_t)r * blocks * 256u;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = row + (uint64_t)b * 144u;
        uint16_t hd, hm; memcpy(&hd, blk, 2); memcpy(&hm, blk + 2, 2);
        const float d = __half2float(__ushort_as_half(hd));
        const float dmin = __half2float(__ushort_as_half(hm));
        uint8_t sc, mn; dev_q4k_scale_min(is, blk + 4, &sc, &mn);
        const uint8_t q = blk[16 + pair * 32u + (l & 31u)];
        const float v = (l < 32u) ? (float)(q & 0xF) : (float)(q >> 4);
        orow[b * 256u + j] = __float2half(d * (float)sc * v - dmin * (float)mn);
    }
}

static int cuda_q4k_wf16_ensure(uint64_t elems) {
    if (elems <= g_q4k_wf16_elems) return 1;
    (void)cudaStreamSynchronize(g_cur_stream);   /* 旧块可能仍被上一发 GEMM 读 */
    if (g_q4k_wf16) (void)cudaFree(g_q4k_wf16);
    g_q4k_wf16 = NULL; g_q4k_wf16_elems = 0;
    if (cudaMalloc(&g_q4k_wf16, elems * sizeof(__half)) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: q4_K gemm 权重 f16 scratch 分配失败 (%.1f MB)\n",
                (double)elems * sizeof(__half) / 1048576.0);
        return 0;
    }
    g_q4k_wf16_elems = elems;
    return 1;
}

/* 激活 f32 [n][dim] → f16 临时区(cuda_tmp_alloc 是单一共享临时区, 调用间在同一 stream 上串行) */
static __half *cuda_q4k_x_f16(const float *x, uint64_t n) {
    __half *xh = (__half *)cuda_tmp_alloc(n * sizeof(__half), "q4_K gemm activations");
    if (!xh) return NULL;
    ds4_launch_pdl(f32_to_f16_kernel, (unsigned)((n + 255u) / 256u), 256, 0, g_cur_stream, xh, x, n);
    return cuda_ok(cudaGetLastError(), "q4_K gemm activation convert launch") ? xh : NULL;
}

/* out[n_tok][out_dim] = x[n_tok][in_dim] · W[out_dim][in_dim]^T */
static int cuda_q4k_gemm_dense(float *out, const unsigned char *w, uint64_t in_dim, uint64_t out_dim,
                               const float *x, uint64_t n_tok) {
    if (!g_cublas_ready) return 0;
    if (!cuda_q4k_wf16_ensure(out_dim * in_dim)) return 0;
    q4_K_to_f16_kernel<<<(unsigned)out_dim, 256, 0, g_cur_stream>>>(
        g_q4k_wf16, w, (uint32_t)(in_dim / CUDA_QK_K), (uint32_t)out_dim);
    if (!cuda_ok(cudaGetLastError(), "q4_K to f16 launch")) return 0;
    __half *xh = cuda_q4k_x_f16(x, n_tok * in_dim);
    if (!xh) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, g_cur_stream);
    cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                     g_q4k_wf16, CUDA_R_16F, (int)in_dim,
                                     xh, CUDA_R_16F, (int)in_dim, &beta,
                                     out, CUDA_R_32F, (int)out_dim,
                                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "q4_K gemm");
}

/* 分组投影(attn_output a 段): 行 r 属组 r/rank, 只吃该组那一段激活 ——
 * low[t][g·rank + i] = Σ_k W[g·rank + i][k] · heads[t][g·group_dim + k], 即 n_groups 个独立
 * GEMM, 用 strided-batched 一发: A 组间跨 rank·group_dim, B 组间跨 group_dim(同一行内平移),
 * C 组间跨 rank。与 grouped_q4_K_a_preq_warp8 核同一公式。 */
static int cuda_q4k_gemm_grouped(float *low, const unsigned char *out_a, uint64_t group_dim, uint64_t rank,
                                 uint32_t n_groups, const float *heads, uint32_t n_tok) {
    if (!g_cublas_ready) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t x_dim = (uint64_t)n_groups * group_dim;
    if (!cuda_q4k_wf16_ensure(low_dim * group_dim)) return 0;
    q4_K_to_f16_kernel<<<(unsigned)low_dim, 256, 0, g_cur_stream>>>(
        g_q4k_wf16, out_a, (uint32_t)(group_dim / CUDA_QK_K), (uint32_t)low_dim);
    if (!cuda_ok(cudaGetLastError(), "q4_K grouped to f16 launch")) return 0;
    __half *xh = cuda_q4k_x_f16(heads, (uint64_t)n_tok * x_dim);
    if (!xh) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, g_cur_stream);
    cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                   (int)rank, (int)n_tok, (int)group_dim, &alpha,
                                                   g_q4k_wf16, CUDA_R_16F, (int)group_dim,
                                                   (long long)(rank * group_dim),
                                                   xh, CUDA_R_16F, (int)x_dim, (long long)group_dim,
                                                   &beta, low, CUDA_R_32F, (int)low_dim, (long long)rank,
                                                   (int)n_groups, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "q4_K grouped gemm");
}
