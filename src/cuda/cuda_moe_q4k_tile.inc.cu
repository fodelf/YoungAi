/* cuda_moe_q4k_tile.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * Q4_K 专家 MoE 的 tile 结构核(2026-09-07): drafter(DSpark mtp.0/1/2, 专家 Q4_K)每轮 5 token × 6 专家 = 30 对。
 *
 * 为什么: 现役 moe_gate_up_mid_decode_q4K_qwarp32 / moe_down_q4K_pairs_qwarp32 一对一 block, 8 lane 分一行、各 lane 直读
 * 自己的 144 B 块(warp 内不合并), 剖面 2.0 + 0.87 ms/段 = 141 GB/s; 稠密路同样形态 09-07 改 tile(整 tile 连续 stage 进
 * shared, lane 各一整块, 段内树归约, 激活进 shared)后 200~235 GB/s。这里把同一 tile 结构套到"块 = (行 tile, 对)":
 * gate/up 各 stage 一 tile(BLOCKS=16, 一 warp 2 行), 段首 lane 同时拿到本行的 g/u ⇒ silu·u·w 直接落 mid。
 * 数值: 归约序从"8 lane 各串 2 块 + 四分之一 warp 树"变"lane 各一块 + 段内树"= 容差级(ULP), 只影响草稿(verify 判官不受影响)。 */
template <uint32_t BLOCKS>
__global__ static void moe_gate_up_q4k_tile_kernel(
        float *gate_out, float *up_out, float *mid_out, const char *gate_base, const char *up_base,
        const cuda_block_q8_K *xq, const int32_t *selected, const float *weights,
        uint64_t expert_bytes, uint32_t expert_mid_dim, uint32_t n_expert, uint32_t write_aux, float clamp) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, pair = blockIdx.z;
    const uint32_t tok = pair / n_expert, slot = pair - tok * n_expert;
    int32_t e = selected[(uint64_t)tok * n_expert + slot];
    if (e < 0) e = 0;
    extern __shared__ uint4 q4k_stage[];
    const cuda_block_q8_K *xa = q4k_stage_x1(q4k_stage + 8u * n16, xq + (uint64_t)tok * BLOCKS, BLOCKS);
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    const char *gb = gate_base + (uint64_t)(uint32_t)e * expert_bytes;
    const char *ub = up_base + (uint64_t)(uint32_t)e * expert_bytes;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = expert_mid_dim / R;
    const float w = weights[(uint64_t)tok * n_expert + slot];
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint64_t toff = (uint64_t)t * R * BLOCKS * sizeof(cuda_block_q4_K);
        const uint4 *sg = (const uint4 *)(gb + toff);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(sg + i);
        __syncwarp();
        float g = dev_dot_q4_K_q8_K_block((const cuda_block_q4_K *)my + rin * BLOCKS + b, xa + b);
        g = q4k_seg_sum<BLOCKS>(g);
        __syncwarp();
        const uint4 *su = (const uint4 *)(ub + toff);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(su + i);
        __syncwarp();
        float u = dev_dot_q4_K_q8_K_block((const cuda_block_q4_K *)my + rin * BLOCKS + b, xa + b);
        u = q4k_seg_sum<BLOCKS>(u);
        __syncwarp();
        if (b == 0u) {
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + t * R + rin;
            if (write_aux) { gate_out[off] = g; up_out[off] = u; }
            mid_out[off] = (g / (1.0f + expf(-g))) * u * w;
        }
    }
}
template <uint32_t BLOCKS>
__global__ static void moe_down_q4k_tile_kernel(
        float *down_out, const char *down_base, const cuda_block_q8_K *midq, const int32_t *selected,
        uint64_t expert_bytes, uint32_t out_dim, uint32_t n_expert) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, pair = blockIdx.z;
    const uint32_t tok = pair / n_expert, slot = pair - tok * n_expert;
    int32_t e = selected[(uint64_t)tok * n_expert + slot];
    if (e < 0) e = 0;
    extern __shared__ uint4 q4k_stage[];
    const cuda_block_q8_K *xa = q4k_stage_x1(q4k_stage + 8u * n16, midq + (uint64_t)pair * BLOCKS, BLOCKS);
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    const char *db = down_base + (uint64_t)(uint32_t)e * expert_bytes;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint4 *sd = (const uint4 *)(db + (uint64_t)t * R * BLOCKS * sizeof(cuda_block_q4_K));
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(sd + i);
        __syncwarp();
        float acc = dev_dot_q4_K_q8_K_block((const cuda_block_q4_K *)my + rin * BLOCKS + b, xa + b);
        acc = q4k_seg_sum<BLOCKS>(acc);
        __syncwarp();
        if (b == 0u) down_out[(uint64_t)pair * out_dim + t * R + rin] = acc;
    }
}
/* 主机发射: 形状假设 IN=4096(16 块)/MID=2048(8 块) —— 其它形状回老核。grid = (tile 数/8 封顶, 1, 对数)。 */
static int moe_gate_up_q4k_tile_launch(float *gate_out, float *up_out, float *mid_out, const char *gate_w, const char *up_w,
                                       const cuda_block_q8_K *xq, const int32_t *selected, const float *weights,
                                       uint64_t expert_bytes, uint32_t xq_blocks, uint32_t expert_mid_dim,
                                       uint32_t n_expert, uint32_t n_tokens, uint32_t write_aux, float clamp) {
    if (xq_blocks != 16u || (expert_mid_dim & 1u) != 0u) return -1;
    unsigned gx = (expert_mid_dim / 2u + 7u) / 8u;
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();
    const size_t shm = (size_t)8u * 32u * 9u * sizeof(uint4) + 16u * sizeof(cuda_block_q8_K);
    ds4_launch_pdl(moe_gate_up_q4k_tile_kernel<16u>, dim3(gx, 1, n_tokens * n_expert), 256, shm, g_cur_stream,
                   gate_out, up_out, mid_out, gate_w, up_w, xq, selected, weights, expert_bytes, expert_mid_dim,
                   n_expert, write_aux, clamp);
    return cuda_ok(cudaGetLastError(), "moe gate_up q4k tile launch");
}
static int moe_down_q4k_tile_launch(float *down_out, const char *down_w, const cuda_block_q8_K *midq,
                                    const int32_t *selected, uint64_t expert_bytes, uint32_t midq_blocks,
                                    uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    if (midq_blocks != 8u || (out_dim & 3u) != 0u) return -1;
    unsigned gx = (out_dim / 4u + 7u) / 8u;
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();
    const size_t shm = (size_t)8u * 32u * 9u * sizeof(uint4) + 8u * sizeof(cuda_block_q8_K);
    ds4_launch_pdl(moe_down_q4k_tile_kernel<8u>, dim3(gx, 1, n_tokens * n_expert), 256, shm, g_cur_stream,
                   down_out, down_w, midq, selected, expert_bytes, out_dim, n_expert);
    return cuda_ok(cudaGetLastError(), "moe down q4k tile launch");
}
