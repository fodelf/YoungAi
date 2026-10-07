/* cuda_qk_warp_3.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * q4_K/q2_K warp gemv kernel 族与 dense 入口。
 */
int ds4_gpu_attention_output_q2k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    /* q2_K 版批量 attn_output(2026-08-19 全q2): a=分组 q2_K matmul, b=dense q2_K。 */
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) return 0;
    if (group_dim % CUDA_QK_K != 0 || ((uint64_t)n_groups * rank) % CUDA_QK_K != 0) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint32_t blocks = (uint32_t)(group_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t a_bytes = row_bytes * low_dim;
    if (out_a_offset > model_size || a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;
    const char *wa = (const char *)cuda_model_range_ptr(model_map, out_a_offset, a_bytes, "attn_out_a_q2k");
    if (!wa) return 0;
    const uint64_t xrows = (uint64_t)n_tokens * n_groups;
    const uint64_t xq_need = xrows * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    ds4_launch_pdl(q8_K_quantize_kernel, dim3(blocks, (unsigned)xrows, 1), 256, 0, g_cur_stream, 
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)heads->ptr, (uint32_t)group_dim, (uint32_t)xrows);
    if (blocks <= 4u) {   /* 小块: 一 lane 一行(warp 模板空转病, 见 kernel 注释) */
        grouped_q2_K_rowlane_kernel<<<dim3((unsigned)((low_dim + 255u) / 256u), (unsigned)n_tokens, 1),
                                      256, 0, g_cur_stream>>>(
            (float *)low->ptr, wa, (const cuda_block_q8_K *)g_q4k_xq_sc,
            row_bytes, blocks, (uint32_t)low_dim, (uint32_t)rank, n_groups);
    } else {
        unsigned gx = (unsigned)((low_dim + 7u) / 8u);
        if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
        const size_t gsh = (blocks <= 32u && (blocks & 3u) == 0u)
                           ? (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4) : 0u;
        const uint32_t gtmin = 16u;   /* 分片核门限(与 dense 同 16 token) */
        if (n_tokens >= gtmin && blocks <= 32u && (blocks & 3u) == 0u && rank >= 8u) {
            const size_t wsh_bytes = (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4);
            const size_t budget = 96u * 1024u;
            uint32_t tile = (uint32_t)((budget - wsh_bytes) / ((size_t)blocks * sizeof(cuda_block_q8_K)));
            if (tile > n_tokens) tile = n_tokens;
            if (tile > 64u) tile = 64u;
            if (tile >= 2u) {
                const size_t sh = wsh_bytes + (size_t)tile * blocks * sizeof(cuda_block_q8_K);
                if (grouped_q2k_tiled_set_smem(blocks, (int)sh)) {
                    unsigned ggx = (unsigned)((low_dim + 7u) / 8u);
                    if (ggx > 512u) ggx = 512u;   /* 分片核网格封顶(与 dense 同 512) */
                    grouped_q2k_tiled_launch(ggx, sh, (float *)low->ptr, wa,
                                             (const cuda_block_q8_K *)g_q4k_xq_sc,
                                             row_bytes, blocks, (uint32_t)low_dim,
                                             (uint32_t)rank, n_groups, n_tokens, tile);
                    if (!cuda_ok(cudaGetLastError(), "attn_out_q2k grouped tiled launch")) return 0;
                    return ds4_gpu_matmul_q2_K_tensor(out, model_map, model_size, out_b_offset,
                                                      low_dim, out_dim, low, n_tokens);
                }
            }
        }
        const uint32_t gwstat = (n_tokens > 1u) ? (uint32_t)n_tokens : 1u;
        if (gwstat > 1u) gx = (unsigned)((low_dim + 7u) / 8u);
        grouped_q2_K_warp_kernel<<<dim3(gx, gwstat > 1u ? 1u : (unsigned)n_tokens, 1), 256, gsh, g_cur_stream>>>(
            (float *)low->ptr, wa, (const cuda_block_q8_K *)g_q4k_xq_sc,
            row_bytes, blocks, (uint32_t)low_dim, (uint32_t)rank, n_groups, gwstat);
    }
    if (!cuda_ok(cudaGetLastError(), "attn_out_q2k grouped launch")) return 0;
    return ds4_gpu_matmul_q2_K_tensor(out, model_map, model_size, out_b_offset,
                                      low_dim, out_dim, low, n_tokens);
}


/* pair 的分片版(2026-08-21): 两矩阵共用同一份激活, 正好一起分片 —— 激活按 tile 进
 * shared 块内共用, 权重每行 stage 一次。lane 映射/dot/求和序与 pair_batch 完全相同。 */
template <uint32_t NB>
__global__ static void __launch_bounds__(256, 1) matmul_q2_K_pair_tiled_kernel(
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq, uint64_t row_bytes, uint32_t blocks_dyn,
        uint32_t out0_dim, uint32_t out1_dim, uint32_t n_tok, uint32_t tile,
        uint32_t lv, uint32_t mb) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 ptshm[];
    cuda_block_q8_K *acts = (cuda_block_q8_K *)ptshm;
    uint4 *wsh = (uint4 *)(acts + (uint64_t)tile * blocks);
    const uint32_t total = out0_dim + out1_dim;
    const uint32_t base0 = blockIdx.x * 8u;
    const uint32_t stride = gridDim.x * 8u;
    const uint32_t n_iter = (base0 >= total) ? 0u : ((total - base0 + stride - 1u) / stride);

    for (uint32_t t0 = 0; t0 < n_tok; t0 += tile) {
        const uint32_t tn = (n_tok - t0 < tile) ? (n_tok - t0) : tile;
        __syncthreads();
        {
            const uint32_t words = (uint32_t)(((uint64_t)tn * blocks * sizeof(cuda_block_q8_K)) / sizeof(uint4));
            const uint4 *src = (const uint4 *)(xq + (uint64_t)t0 * blocks);
            uint4 *dst = (uint4 *)acts;
            for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = src[i];
        }
        __syncthreads();
        for (uint32_t it = 0; it < n_iter; it++) {
            const uint32_t r = base0 + warp + it * stride;
            const bool live = r < total;
            const bool second = live && (r >= out0_dim);
            const uint32_t row = second ? (r - out0_dim) : r;
            float *outp = second ? out1 : out0;
            const uint32_t od = second ? out1_dim : out0_dim;
            const cuda_block_q2_K *wr = NULL;
            if (live) {
                const char *wb = second ? w1 : w0;
                const uint4 *src16 = (const uint4 *)(wb + (uint64_t)row * row_bytes);
                uint4 *my = wsh + (uint64_t)warp * n16;
                for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
                __syncwarp();
                wr = (const cuda_block_q2_K *)my;
            }
            for (uint32_t t = 0; t < tn; t++) {
                float a2 = 0.0f;
                if (live) {
                    const cuda_block_q8_K *xt = acts + (uint64_t)t * blocks;
                    if (blocks == 4u && lv >= 2u && mb >= 4u) {
                        const uint32_t bi = lane >> 3u, qi = lane & 7u;
                        a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
                    } else if (blocks <= 16u && lv >= 2u && blocks <= mb) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (live && lane == 0) outp[(uint64_t)(t0 + t) * od + row] = a2;
            }
            __syncwarp();
        }
    }
}

static bool pair_q2k_tiled_set_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx]) return cap[idx] >= bytes;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

static inline void pair_q2k_tiled_launch(unsigned gx, size_t sh,
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq, uint64_t row_bytes, uint32_t blocks,
        uint32_t out0_dim, uint32_t out1_dim, uint32_t n_tok, uint32_t tile,
        uint32_t lv, uint32_t mb) {
    switch (blocks) {
    case 4u:  matmul_q2_K_pair_tiled_kernel<4u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    case 8u:  matmul_q2_K_pair_tiled_kernel<8u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    case 16u: matmul_q2_K_pair_tiled_kernel<16u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    case 32u: matmul_q2_K_pair_tiled_kernel<32u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    default:  matmul_q2_K_pair_tiled_kernel<0u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    }
}

/* 批版 pair 融合(2026-08-21): 同输入的两矩阵(shexp gate+up / q_a+kv)合成一发。
 * ①激活只量化一次 ②行数合并 ⇒ 小矩阵不再因 gx 小而并行度不足(35GB/s 病)
 * ③权重驻留(块内循环 token)。逐 (row,tok) 数学与单发路逐位相同。 */
template<uint32_t NB>
__global__ static void __launch_bounds__(256, DS4_Q2K_BLOCKS_PER_SM) matmul_q2_K_pair_batch_kernel(
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq, uint64_t row_bytes, uint32_t blocks_dyn,
        uint32_t out0_dim, uint32_t out1_dim, uint32_t ntb, uint32_t lv, uint32_t mb) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 pairshm[];
    const uint32_t total = out0_dim + out1_dim;
    for (uint32_t r = blockIdx.x * 8u + warp; r < total; r += gridDim.x * 8u) {
        const bool second = (r >= out0_dim);
        const uint32_t row = second ? (r - out0_dim) : r;
        const char *wb = second ? w1 : w0;
        float *outp = second ? out1 : out0;
        const uint32_t od = second ? out1_dim : out0_dim;
        const uint4 *src16 = (const uint4 *)(wb + (uint64_t)row * row_bytes);
        uint4 *my = pairshm + (uint64_t)warp * n16;
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)my;
        for (uint32_t t = 0; t < ntb; t++) {
            const cuda_block_q8_K *xt = xq + (uint64_t)t * blocks;
            float a2 = 0.0f;
            if (blocks == 4u && lv >= 2u && mb >= 4u) {
                const uint32_t bi = lane >> 3u, qi = lane & 7u;
                a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
            } else if (blocks <= 16u && lv >= 2u && blocks <= mb) {
                const uint32_t bi = lane >> 1u, h = lane & 1u;
                if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u)
                    a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
            }
            for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
            if (lane == 0) outp[(uint64_t)t * od + row] = a2;
        }
        __syncwarp();
    }
}

int ds4_gpu_matmul_q2_K_pair_batch_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || n_tok < 2u) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    if (blocks > 32u || (blocks & 3u) != 0u) return 0;
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t w0b = row_bytes * out0_dim, w1b = row_bytes * out1_dim;
    if (off0 > model_size || w0b > model_size - off0) return 0;
    if (off1 > model_size || w1b > model_size - off1) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float)) return 0;
    if (out0->bytes < n_tok * out0_dim * sizeof(float)) return 0;
    if (out1->bytes < n_tok * out1_dim * sizeof(float)) return 0;
    const char *w0 = cuda_model_range_ptr(model_map, off0, w0b, "dense_q2k_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, off1, w1b, "dense_q2k_pair1");
    if (!w0 || !w1) return 0;
    const uint64_t xq_need = n_tok * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    ds4_launch_pdl(q8_K_quantize_kernel, dim3(blocks, (unsigned)n_tok, 1), 256, 0, g_cur_stream, 
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    const uint32_t plv = 2u, pmb = 16u;   /* staging 档 / 半块整块分界, 与 dense 入口同 */
    const unsigned gx = (unsigned)((out0_dim + out1_dim + 7u) / 8u);
    const size_t sh = (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4);
    const cuda_block_q8_K *xq = (const cuda_block_q8_K *)g_q4k_xq_sc;
    {   /* 大批: 激活分片进 shared(与 dense/grouped 同一修法) */
        const uint32_t ptmin = 16u;
        if (n_tok >= ptmin) {
            const size_t budget = 96u * 1024u;
            uint32_t tile = (uint32_t)((budget - sh) / ((size_t)blocks * sizeof(cuda_block_q8_K)));
            if (tile > n_tok) tile = (uint32_t)n_tok;
            if (tile > 64u) tile = 64u;
            if (tile >= 2u) {
                const size_t psh = sh + (size_t)tile * blocks * sizeof(cuda_block_q8_K);
                if (pair_q2k_tiled_set_smem(blocks, (int)psh)) {
                    unsigned pgx = gx > 512u ? 512u : gx;   /* 分片核网格封顶 */
                    pair_q2k_tiled_launch(pgx, psh, (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                                          xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim,
                                          (uint32_t)n_tok, tile, plv, pmb);
                    return cuda_ok(cudaGetLastError(), "dense q2_K pair tiled launch");
                }
            }
        }
    }
    switch (blocks) {
    case 4u:  matmul_q2_K_pair_batch_kernel<4u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    case 8u:  matmul_q2_K_pair_batch_kernel<8u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    case 16u: matmul_q2_K_pair_batch_kernel<16u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    case 32u: matmul_q2_K_pair_batch_kernel<32u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    default:  matmul_q2_K_pair_batch_kernel<0u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    }
    return cuda_ok(cudaGetLastError(), "dense q2_K pair batch launch");
}

__global__ static void q2k_ksplit_reduce_kernel(float *out, const float *partial,
                                               uint32_t out_dim, uint32_t n_tok, uint32_t ksplit) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)out_dim * n_tok) return;
    const uint32_t t = (uint32_t)(idx / out_dim), row = (uint32_t)(idx - (uint64_t)t * out_dim);
    float acc = 0.0f;
    for (uint32_t z = 0; z < ksplit; z++)   /* 固定 z 序 = 确定性 */
        acc += partial[((uint64_t)z * n_tok + t) * out_dim + row];
    out[(uint64_t)t * out_dim + row] = acc;
}

/* 动态 shared 上限 opt-in(超 48KB 必须显式申请, 每形状只设一次)。 */
static bool q2k_set_dynamic_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx] && cap[idx] >= bytes) return true;
    if (done[idx] && cap[idx] < bytes) return false;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

/* 大批分片 GEMM(2026-08-21): 原 warp kernel 是"一行一 warp, 块内循环 token", 每个 warp
 * 为自己那一行把整批 token 的 q8 激活重读一遍, 一个 block 的 8 个 warp 又各读一遍 ——
 * 340 token 的块产生约 12.9GB 显存流量(权重才 11MB), prefill 因此只有 42 t/s。
 * 这里按 token 分片: 每个 tile 的激活块内协作搬进 shared 一次, 8 个 warp 共用; 权重每行
 * 仍只 stage 一次(行循环在 tile 内)。每 lane 的 block 映射与 dot 函数与原 kernel 完全相同,
 * 所以结果逐位一致, 只是少读几个数量级的字节。
 *   shared 布局: [acts: T × blocks × sizeof(q8_K)][weights: 8 warps × blocks*84B]
 *   行循环次数在块内统一(否则 __syncthreads 会挂), 越界行只跳过写回。 */
template <uint32_t NB>
__global__ static void __launch_bounds__(256, 1) matmul_q2_K_tiled_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks_dyn, uint32_t out_dim,
        uint32_t n_tok, uint32_t tile, uint32_t lv, uint32_t mb) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 tshm[];
    cuda_block_q8_K *acts = (cuda_block_q8_K *)tshm;
    uint4 *wsh = (uint4 *)(acts + (uint64_t)tile * blocks);

    /* 块内统一的行迭代次数: 本块最小起始行 = blockIdx.x*8, 步长 gridDim.x*8 */
    const uint32_t base0 = blockIdx.x * 8u;
    const uint32_t stride = gridDim.x * 8u;
    const uint32_t n_iter = (base0 >= out_dim) ? 0u : ((out_dim - base0 + stride - 1u) / stride);

    for (uint32_t t0 = 0; t0 < n_tok; t0 += tile) {
        const uint32_t tn = (n_tok - t0 < tile) ? (n_tok - t0) : tile;
        __syncthreads();
        {   /* 协作搬 tn 个 token 的激活(uint4 宽拷) */
            const uint32_t words = (uint32_t)(((uint64_t)tn * blocks * sizeof(cuda_block_q8_K)) / sizeof(uint4));
            const uint4 *src = (const uint4 *)(xq + (uint64_t)t0 * blocks);
            uint4 *dst = (uint4 *)acts;
            for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = src[i];
        }
        __syncthreads();
        for (uint32_t it = 0; it < n_iter; it++) {
            const uint32_t row = base0 + warp + it * stride;
            const bool live = row < out_dim;
            const cuda_block_q2_K *wr;
            if (live) {
                const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
                uint4 *my = wsh + (uint64_t)warp * n16;
                for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
                __syncwarp();
                wr = (const cuda_block_q2_K *)my;
            } else {
                wr = NULL;
            }
            for (uint32_t t = 0; t < tn; t++) {
                float a2 = 0.0f;
                if (live) {
                    const cuda_block_q8_K *xt = acts + (uint64_t)t * blocks;
                    if (blocks == 4u && lv >= 2u && mb >= 4u) {
                        const uint32_t bi = lane >> 3u, qi = lane & 7u;
                        a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
                    } else if (blocks <= 16u && lv >= 2u && blocks <= mb) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (live && lane == 0) out[(uint64_t)(t0 + t) * out_dim + row] = a2;
            }
            __syncwarp();
        }
    }
}

/* tiled kernel 的 shared opt-in(与 q2k_set_dynamic_smem 同款, 独立记账) */
static bool q2k_tiled_set_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx]) return cap[idx] >= bytes;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

static inline void q2k_tiled_launch(unsigned gx, size_t sh,
        float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t n_tok, uint32_t tile, uint32_t lv, uint32_t mb) {
    switch (blocks) {
    case 4u:  matmul_q2_K_tiled_kernel<4u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    case 8u:  matmul_q2_K_tiled_kernel<8u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    case 16u: matmul_q2_K_tiled_kernel<16u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    case 32u: matmul_q2_K_tiled_kernel<32u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    default:  matmul_q2_K_tiled_kernel<0u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    }
}

/* 特化分发: blocks∈{4,8,16,32} 走编译常量实例, 其他走动态 */
static inline void q2k_warp_launch(dim3 grid, size_t sh,
        float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t lv, uint32_t mb, const float *xraw, uint32_t ntb, uint32_t ntot,
        float *part, uint32_t ks, uint32_t xsm) {
    switch (blocks) {
    case 4u:  matmul_q2_K_warp_kernel<4u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    case 8u:  matmul_q2_K_warp_kernel<8u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    case 16u: matmul_q2_K_warp_kernel<16u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    case 32u: matmul_q2_K_warp_kernel<32u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    default:  matmul_q2_K_warp_kernel<0u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    }
}

