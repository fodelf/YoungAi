/* cuda_q4k_tile.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * 解码期 Q4_K 稠密 gemv 的单 token tile 核(2026-09-07), 取代 cuda_api_moe_corr_2 的 dp4a/半块家族; 块点积与 stage 助手在
 * cuda_q4k_dot, 小批多 token 核在 cuda_q4k_multi(本文件的发射器对 n_tok>1 委托过去)。 */
/* 单矩阵: out[row] = W[row] · xq(解码单 token; 小批走 q4k_tile_multi_kernel) */
template <uint32_t BLOCKS>
__global__ static void q4k_tile_kernel(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t out_dim) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    constexpr uint32_t R = 32u / BLOCKS;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 q4k_stage[];
    q4k_act_q8K x; x.x = q4k_stage_x1(q4k_stage + 8u * 32u * 9u, xq, BLOCKS);
    uint4 *my = q4k_stage + (uint64_t)warp * (R * BLOCKS * 9u);
    const uint32_t ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const float acc = q4k_tile_dot<BLOCKS>(my, w + (uint64_t)t * R * BLOCKS * sizeof(cuda_block_q4_K), x, lane);
        if (lane % BLOCKS == 0u) out[t * R + lane / BLOCKS] = acc;
    }
}

/* 同输入矩阵对(q_a+kv / shexp gate+up): tile 号先铺满矩阵 0 再矩阵 1, 一次发射 */
template <uint32_t BLOCKS>
__global__ static void q4k_tile_pair_kernel(float *out0, float *out1, const char *w0, const char *w1,
                                            const cuda_block_q8_K *xq, uint32_t out0_dim, uint32_t out1_dim) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    constexpr uint32_t R = 32u / BLOCKS;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 q4k_stage[];
    q4k_act_q8K x; x.x = q4k_stage_x1(q4k_stage + 8u * 32u * 9u, xq, BLOCKS);
    uint4 *my = q4k_stage + (uint64_t)warp * (R * BLOCKS * 9u);
    const uint32_t nt0 = out0_dim / R, ntiles = nt0 + out1_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const bool second = t >= nt0;
        const uint32_t tl = second ? t - nt0 : t;
        const char *w = second ? w1 : w0;
        const float acc = q4k_tile_dot<BLOCKS>(my, w + (uint64_t)tl * R * BLOCKS * sizeof(cuda_block_q4_K), x, lane);
        if (lane % BLOCKS == 0u) (second ? out1 : out0)[tl * R + lane / BLOCKS] = acc;
    }
}

/* attn_output_b + hc expand(解码单 token, q8_0 激活): 行长 32 块 ⇒ 一 warp 一行整行 stage(4608 B), lane 各一整块;
 * epilogue(块输出 → 4 路 hc 残差合成)与原 q8 labeled 版逐字同义。 */
__global__ static void q4k_hc_expand_kernel(float *out_hc, float *block_out, const float *residual_hc, const float *split,
                                            const char *w, const int8_t *xq, const float *xs,
                                            uint32_t kblocks, uint32_t out_dim, uint32_t n_embd, uint32_t n_hc) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 q4k_stage[];
    const uint32_t n16 = kblocks * 9u;
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    /* 激活(值+尺度)进 shared, 子块和 Σq8 只算一次(09-07 微基准 out_b: 169 → 225 GB/s, 逐位同): 原来每行每块都在
     * dot 里重算 64 个 dp4a 的 Σq8 —— 它只依赖激活, 与行无关。 */
    uint32_t *dst = (uint32_t *)(q4k_stage + 8u * n16);
    const uint32_t nq = kblocks * 64u, ns = kblocks * 8u;
    for (uint32_t i = threadIdx.x; i < nq; i += blockDim.x) dst[i] = ((const uint32_t *)xq)[i];
    for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) ((float *)dst)[nq + i] = xs[i];
    __syncthreads();
    const int8_t *xqa = (const int8_t *)dst; const float *xsa = (const float *)dst + nq;
    int32_t *s8 = (int32_t *)dst + nq + ns;
    for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) {
        const int8_t *q8 = xqa + (uint64_t)i * 32u; int32_t sm = 0;
        #pragma unroll
        for (uint32_t k = 0; k < 32u; k += 4u) sm = __dp4a(0x01010101, *(const int32_t *)(q8 + k), sm);
        s8[i] = sm;
    }
    __syncthreads();
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src16 = (const uint4 *)(w + (uint64_t)row * kblocks * sizeof(cuda_block_q4_K));
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        float acc = 0.0f;
        if (lane < kblocks) {   /* kblocks ≤ 32(发射方校验): lane 各一整块 */
            q4k_wblk wb; q4k_wblk_decode(wr + lane, &wb);
            acc = dev_dot_q4_K_q8_0x8_pre(wr + lane, wb, xqa + (uint64_t)lane * 256u, xsa + (uint64_t)lane * 8u, s8 + (uint64_t)lane * 8u);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            block_out[row] = acc;
            const float *post = split + n_hc;
            const float *comb = split + 2u * n_hc;
            for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
                float hc_acc = acc * post[dst_hc];
                for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++)
                    hc_acc += comb[dst_hc + (uint64_t)src_hc * n_hc] * residual_hc[(uint64_t)src_hc * n_embd + row];
                out_hc[(uint64_t)dst_hc * n_embd + row] = hc_acc;
            }
        }
        __syncwarp();
    }
}

static int q4k_tile_launch(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t blocks,
                           uint32_t out_dim, uint32_t n_tok) {
    const unsigned gx = q4k_tile_grid(out_dim / (32u / blocks));
    if (n_tok > 1u) return q4k_tile_multi_launch(out, w, xq, blocks, out_dim, n_tok);
    const size_t shm1 = Q4K_TILE_SHM + (size_t)blocks * sizeof(cuda_block_q8_K);   /* + 单 token 激活 */
    switch (blocks) {
        case 4u:  ds4_launch_pdl(q4k_tile_kernel<4u>, gx, 256, shm1, g_cur_stream, out, w, xq, out_dim); break;
        case 8u:  ds4_launch_pdl(q4k_tile_kernel<8u>, gx, 256, shm1, g_cur_stream, out, w, xq, out_dim); break;
        default:  ds4_launch_pdl(q4k_tile_kernel<16u>, gx, 256, shm1, g_cur_stream, out, w, xq, out_dim); break;
    }
    return cuda_ok(cudaGetLastError(), "q4k tile launch");
}
static int q4k_tile_pair_launch(float *out0, float *out1, const char *w0, const char *w1, const cuda_block_q8_K *xq,
                                uint32_t blocks, uint32_t out0_dim, uint32_t out1_dim) {
    const uint32_t R = 32u / blocks;
    const unsigned g = q4k_tile_grid(out0_dim / R + out1_dim / R);
    const size_t shm1 = Q4K_TILE_SHM + (size_t)blocks * sizeof(cuda_block_q8_K);
    switch (blocks) {
        case 4u:  ds4_launch_pdl(q4k_tile_pair_kernel<4u>, g, 256, shm1, g_cur_stream, out0, out1, w0, w1, xq, out0_dim, out1_dim); break;
        case 8u:  ds4_launch_pdl(q4k_tile_pair_kernel<8u>, g, 256, shm1, g_cur_stream, out0, out1, w0, w1, xq, out0_dim, out1_dim); break;
        default:  ds4_launch_pdl(q4k_tile_pair_kernel<16u>, g, 256, shm1, g_cur_stream, out0, out1, w0, w1, xq, out0_dim, out1_dim); break;
    }
    return cuda_ok(cudaGetLastError(), "q4k tile pair launch");
}
static int q4k_hc_expand_launch(float *out_hc, float *block_out, const float *residual_hc, const float *split,
                                const char *w, const int8_t *xq, const float *xs, uint32_t kblocks, uint32_t out_dim,
                                uint32_t n_embd, uint32_t n_hc) {
    if (kblocks > 32u) { fprintf(stderr, "ds4: q4k hc expand: 行 %u 块超过 32(整行 stage 上限)\n", kblocks); return 0; }
    const unsigned g = q4k_tile_grid(out_dim);
    const size_t shm = (size_t)8u * kblocks * 9u * sizeof(uint4) + (size_t)kblocks * (256u + 32u + 32u);   /* 权重 stage + 激活/尺度/Σq8 */
    ds4_launch_pdl(q4k_hc_expand_kernel, g, 256, shm, g_cur_stream,
                   out_hc, block_out, residual_hc, split, w, xq, xs, kblocks, out_dim, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "q4k hc expand launch");
}
