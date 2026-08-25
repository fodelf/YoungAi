/* cuda_moe_kernels_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * routed MoE gate_up_mid/scatter_sorted/down kernel 族。
 */
/* 全 Q2_K 路由专家的 gate/up。引擎原有的 gate/up kernel 硬编码 cuda_block_iq2_xxs,
 * 只认 "gate=IQ2_XXS + down=Q2_K" 这一种组合; 而 Q2_K 每 16 元素一个 scale 直接解码,
 * 没有 IQ2_XXS 的 256 项 grid 查表(gather), 推理更快。down 路径本就支持 Q2_K, 解码函数
 * dev_dot_q2_K_q8_K_block 现成可用 —— 这里只是把同一套循环换到 Q2_K 的块布局上,
 * 行字节数由调用方的 gate_row_bytes 传入, 自适应两种块大小。 */
/* 向量化取块版 q2_K×q8_K dot: 块 84B 只保证 4B 对齐(84=21×4, 行距 672=42×16),
 * 用 uint32×21 整取进寄存器 —— 比逐字节 load 宽 4×。数学复用原 dot。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_vec(
        const cuda_block_q2_K *xg, const cuda_block_q8_K *y) {
    uint32_t v[21];
    #pragma unroll
    for (int i = 0; i < 21; i++) v[i] = ((const uint32_t *)xg)[i];
    return dev_dot_q2_K_q8_K_block((const cuda_block_q2_K *)v, y);
}

/* xq_blocks==16 专用拆分(2026-08-17 第五轮): 微基准判决 —— 同构+真dot 的隔离 kernel
 * REG:62 / 257 GB/s, 而合体 kernel 因泛化 else 分支被顶到 REG:128(2 块/SM) 只有 94GB/s。
 * 寄存器按最坏路径分配, 唯一解=物理拆 kernel。泛化配方仍走原 kernel。 */
__global__ static void moe_gate_up_mid_q2k_x16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t tok = blockIdx.x, pk = blockIdx.z;
    const uint32_t m = blockIdx.y * 8u + warp;
    if (m >= expert_mid_dim) return;
    const uint64_t pair = (uint64_t)tok * n_expert + pk;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * 16u;

    __shared__ uint4 stage[8][2 * 16 * 84 / 16];      /* gate 行 84 uint4 + up 行 84 uint4 */
    const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    for (uint32_t i = lane; i < 84u; i += 32u) {
        stage[warp][i] = __ldcs(gsrc + i);             /* 流式读: 权重只过一遍, 别挤 L2 */
        stage[warp][84u + i] = __ldcs(usrc + i);
    }
    __syncwarp();
    const uint32_t half = lane >> 4u;                  /* 0=gate 1=up */
    const uint32_t l16 = lane & 15u;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)((const uint8_t *)stage[warp] + (uint64_t)half * 1344u);
    float acc = dev_dot_q2_K_q8_K_block_smem(wr + l16, xqb + l16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    const float up_v = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0) {
        float gate = acc, up = up_v;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = pair * expert_mid_dim + m;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
    }
}

/* A/B(2026-08-17): ncu 实测本 kernel Block Limit Registers=2(理论 occupancy 33%)。
 * __launch_bounds__(256,4) 压到 REG:64 但 STACK:224 溢出, e2e 21.0 vs 21.09 中性偏负,
 * 已回退; 真凶=else 泛化分支的 _vec dot(寄存器按最坏路径配, b5 不走它却买单),
 * 已换普通 dot 消压, 不再需要 launch_bounds。 */
__global__ static void moe_gate_up_mid_q2k_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    /* 重构(2026-08-17 第三轮): 原 8-lane/行×84B 步距访存不合并(该 kernel 16% 时间)。
     * 现: 每 warp 一行, gate+up 两行权重协作合并搬 shared; 32 lane 拆两半 —
     * lanes 0-15 算 gate 的 16 块, lanes 16-31 算 up 的 16 块, 双矩阵并行。
     * grid = (ntok, (MID+7)/8, nexp)。 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t tok = blockIdx.x, pk = blockIdx.z;
    const uint32_t m = blockIdx.y * 8u + warp;
    if (m >= expert_mid_dim) return;
    const uint64_t pair = (uint64_t)tok * n_expert + pk;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    float acc = 0.0f;
    if (xq_blocks == 16u) {
        __shared__ uint4 stage[8][2 * 16 * 84 / 16];      /* gate 行 84 uint4 + up 行 84 uint4 */
        const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        for (uint32_t i = lane; i < 84u; i += 32u) {
            stage[warp][i] = gsrc[i];
            stage[warp][84u + i] = usrc[i];
        }
        __syncwarp();
        const uint32_t half = lane >> 4u;                  /* 0=gate 1=up */
        const uint32_t l16 = lane & 15u;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)((const uint8_t *)stage[warp] + (uint64_t)half * 1344u);
        acc = dev_dot_q2_K_q8_K_block_smem(wr + l16, xqb + l16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        const float up_v = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            float gate = acc, up = up_v;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = pair * expert_mid_dim + m;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
        }
    } else {
        /* 泛化回退: 逐块直读(xq_blocks != 16 的配方) */
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        float g = 0.0f, u = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            g += dev_dot_q2_K_q8_K_block(gr + b, xqb + b);
            u += dev_dot_q2_K_q8_K_block(ur + b, xqb + b);
        }
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, off);
            u += __shfl_down_sync(0xffffffffu, u, off);
        }
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const uint64_t off2 = pair * expert_mid_dim + m;
            gate_out[off2] = g;
            up_out[off2] = u;
            mid_out[off2] = (g / (1.0f + expf(-g))) * u * weights[pair];
        }
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* IQ2_XXS 版 x16 staging(2026-08-17 第七夜): lut 版 8-lane×66B 步距散读实测 111 GB/s
 * (final86 最大单洞 233μs/层)。q2k x16 同款药: gate+up 行(各 1056B=66 uint4, 16B 对齐)
 * warp 协作 __ldcs 合并搬 shared, 半 warp 各算一矩阵 16 块, LUT/xq 块级共享。 */
__global__ static void moe_gate_up_mid_iq2_x16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t tok = blockIdx.x, pk = blockIdx.z;
    const uint32_t m = blockIdx.y * 8u + warp;
    const uint64_t pair = (uint64_t)tok * n_expert + pk;
    /* A/B(2026-08-17 第九夜): 试过 LUT/xq 去 staging 全局直读 —— 31.4→22.1 大倒退回退。
     * LUT gather 是 warp 内随机索引, shared 是唯一不串行化的路; "28.7MB 搬运"是
     * 片上拷贝不占 DRAM, 账算错了。 */
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint4 stage[8][2 * 66];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_signs[i] = cuda_ksigns_iq2xs[i];
    {   /* xq 16 块 = 4672B, 按 uint32 粒度整块搬 */
        const uint32_t *src = (const uint32_t *)(xq + (uint64_t)tok * 16u);
        uint32_t *dst = (uint32_t *)sxq;
        const uint32_t n32 = (uint32_t)(16u * sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (m >= expert_mid_dim) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    for (uint32_t i = lane; i < 66u; i += 32u) {
        stage[warp][i] = __ldcs(gsrc + i);
        stage[warp][66u + i] = __ldcs(usrc + i);
    }
    __syncwarp();
    const uint32_t half = lane >> 4u;                  /* 0=gate 1=up */
    const uint32_t l16 = lane & 15u;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)((const uint8_t *)stage[warp] + (uint64_t)half * 1056u);
    float acc = dev_dot_iq2_xxs_q8_K_block_lut(wr + l16, sxq + l16, s_grid, s_signs);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    const float up_v = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0) {
        float gate = acc, up = up_v;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = pair * expert_mid_dim + m;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
    }
}

/* 批版 x16 tile(2026-08-21 verify kernel 质量战): decode 的 x16 kernel 实测 215GB/s
 * 近峰值, 而 verify 走的 tile8_row32 只有 74GB/s —— 差距全在访存形状。本 kernel =
 * x16 结构(warp 驻留一行 + 32 lane 合并 uint4 staging + 16 lane 覆盖 16 块×2 矩阵)
 * 叠加 expert-tile 权重驻留(同专家的 np 个 token 复用已 staged 的行, 权重只读一遍)。
 * 逐 (pair,row) 数学与 tile8 路一致(同 dot、同归约序)。 */
__device__ static uint32_t g_x16_ldg_dev = 0;   /* A/B: staging 用 __ldg(L1 缓存) vs __ldcs(流式) */
__global__ static void moe_gate_up_mid_iq2_x16_tile_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        uint32_t g_x16_ldg) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    /* tile 维 grid-stride(2026-08-21): 网格按上界开(37 列)但实际 tile 只有 ~15 ⇒ 59%
     * 空块白调度。改成按典型值开、块内循环补齐, 空块归零。 */
    const uint32_t n_tile = *tile_total;
    for (uint32_t tile = blockIdx.y; tile < n_tile; tile += gridDim.y) {
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    uint32_t pair_id[8], tokid[8], np = 0;
    for (; np < 8u; np++) {
        const uint32_t lp = local_start + np;
        if (lp >= counts[expert]) break;
        pair_id[np] = sorted_pairs[offsets[expert] + lp];
        tokid[np] = pair_id[np] / n_expert;
    }
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];
    __shared__ uint4 stage[8][2 * 66];
    /* 激活 staging(2026-08-21): 每行都从全局重读 np×16 块激活 = 每行 84B 权重配 ~5KB
     * 激活(21x 放大, 全靠 L1 兜)。tile 内 np 个 token 的激活对整块 8 warp 共用 ⇒ 一次
     * 搬进 shared(np≤6 时 28KB), decode x16 同款药。 */
    __shared__ cuda_block_q8_K sxq[6][16];
    const uint32_t np_sh = (np <= 6u) ? np : 0u;
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_signs[i] = cuda_ksigns_iq2xs[i];
    if (np_sh) {
        const uint32_t n32 = (uint32_t)(16u * sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t p = 0; p < np_sh; p++) {
            const uint32_t *src = (const uint32_t *)(xq + (uint64_t)tokid[p] * 16u);
            uint32_t *dst = (uint32_t *)sxq[p];
            for (uint32_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
        }
    }
    __syncthreads();
    const uint32_t m = blockIdx.x * 8u + warp;
    if (m >= expert_mid_dim || np == 0u) { __syncthreads(); continue; }   /* 下轮 tile 前对齐 */
    const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    if (g_x16_ldg) {
        for (uint32_t i = lane; i < 66u; i += 32u) {
            stage[warp][i] = __ldg(gsrc + i);
            stage[warp][66u + i] = __ldg(usrc + i);
        }
    } else {
        for (uint32_t i = lane; i < 66u; i += 32u) {
            stage[warp][i] = __ldcs(gsrc + i);
            stage[warp][66u + i] = __ldcs(usrc + i);
        }
    }
    __syncwarp();
    const uint32_t half = lane >> 4u, l16 = lane & 15u;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)((const uint8_t *)stage[warp] + (uint64_t)half * 1056u);
    /* LUT 解码一次 · 点乘 np 次(2026-08-21): 每 pair 单独调 _block_lut 会把 iq2 网格
     * 解码(shared 随机索引, bank 冲突重)重复 np 遍 —— 本 kernel 的真实瓶颈。 */
    const cuda_block_q8_K *yb[8];
    for (uint32_t p = 0; p < 8u; p++)
        yb[p] = (p < np) ? (np_sh ? (sxq[p] + l16) : (xq + (uint64_t)tokid[p] * 16u + l16)) : NULL;
    float acc8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    dev_dot_iq2_xxs_q8_K_block8_deq_lut(wr + l16, yb[0], yb[1], yb[2], yb[3], yb[4], yb[5], yb[6], yb[7],
                                        np, acc8, s_grid, s_signs);
    for (uint32_t p = 0; p < np; p++) {
        float acc = acc8[p];
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        const float up_v = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            float gate = acc, up = up_v;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair_id[p] * expert_mid_dim + m;
            if (write_aux) { gate_out[off] = gate; up_out[off] = up; }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair_id[p]];
        }
    }
    __syncthreads();   /* 下轮 tile 复写 stage/sxq 前全块对齐 */
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[256] = sum;
    }
}

