/* cuda_api_moe_corr_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * swiglu/add/corr/steering/router select GPU API。
 */
/* 半块 dot 族(2026-08-17 第九夜): blocks≤16 的 kernel 里整块 dot 让半个 warp 空转
 * (dot 循环 b<blocks 步长 32)。每 lane 算半块(j0=0 或 4 的 4 个子块), 32 lanes 全活跃。
 * 数值: 两半各自 float 化后相加, 与整块版有末位舍入差(同类重排已过断言纪律)。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_block_half(
        const cuda_block_q4_K *x, const cuda_block_q8_K *y, uint32_t j0) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = j0; j < j0 + 4u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        summs0 += (int)m0 * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)m1 * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        isum0 += (int)sc0 * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, 0);
        isum1 += (int)sc1 * dev_dot_q4_32(x->qs + byte_off, y->qs + (j + 1u) * 32u, 4);
    }
    return y->d * xd * (float)(isum0 + isum1) - y->d * xmin * (float)(summs0 + summs1);
}

__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem_half(
        const cuda_block_q4_K *x, const int8_t *xq, const float *xs, uint32_t j0) {
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = j0; j < j0 + 4u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* q4_K 权重块(256 值) × q8_0 预量化激活(8 个 32 值子块, int8 qs + 每块 f32 scale)。
 * q4_K 的 8 个子 scale/min 与 q8_0 的 32 值分块天然对齐: 每子块
 * acc += xs_j × (d·sc_j·Σ(q4·q8) − dmin·m_j·Σq8)。attn_output q8→q4 移植的核心。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8(
        const cuda_block_q4_K *xg, const int8_t *xq, const float *xs, int use_dp4a) {
    /* 块 144B=9×uint4 一次进寄存器(dense q4 kernel 上实测 +1.3% 的同款招),
     * 后续 scale 解包与 dp4a 全走寄存器。 */
    uint4 v[9];
    #pragma unroll
    for (int vi = 0; vi < 9; vi++) v[vi] = ((const uint4 *)xg)[vi];
    const cuda_block_q4_K *x = (const cuda_block_q4_K *)v;
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    /* 双 float 链(奇偶子块交错), 掩盖 dp4a 依赖延迟 */
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t dot0, dot1, s80 = 0, s81 = 0;
        if (use_dp4a) {
            dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
            dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
            #pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
                s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
            }
        } else {
            dot0 = dot1 = 0;
            #pragma unroll
            for (uint32_t i = 0; i < 32u; i++) {
                dot0 += ((x->qs[byte_off + i] >> 0) & 0xF) * q8a[i]; s80 += q8a[i];
                dot1 += ((x->qs[byte_off + i] >> 4) & 0xF) * q8b[i]; s81 += q8b[i];
            }
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* shared 源专用变体(2026-08-17): 数据已在片上, v[9] 寄存器整取无意义; 砍掉非 dp4a
 * 标量路径(寄存器按最坏路径分配, 把宿主 kernel 顶到 REG:121/127=2 块/SM 的元凶)。
 * 数学与 dev_dot_q4_K_q8_0x8 的 dp4a 路径逐位同义。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem(
        const cuda_block_q4_K *x, const int8_t *xq, const float *xs) {
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* dp4a+staged 特化版: 无 use_dp4a/直读分支 ⇒ 寄存器回落, SM 驻留翻倍 */
__global__ static void grouped_q4_K_a_preq_warp8_dp4a_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t kblocks) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t slot = threadIdx.x >> 4u;
    const uint64_t row = (uint64_t)blockIdx.x * 16u + slot;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * kblocks * 256u;
    const float *xsr = xscale + xrow * kblocks * 8u;
    extern __shared__ uint4 dstage_a[];
    const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K));
    const uint32_t n16 = (uint32_t)kblocks * 9u;
    uint4 *my = dstage_a + (uint64_t)slot * n16;
    for (uint32_t i = lane; i < n16; i += 16u) my[i] = __ldcs(src16 + i);
    __syncwarp();
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
    float acc = 0.0f;
    for (uint64_t b = lane; b < kblocks; b += 16u)
        acc += dev_dot_q4_K_q8_0x8_smem(wr + b, xqr + b * 256u, xsr + b * 8u);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t kblocks,
        int has_add) {
    /* epilogue 与原版逐字同义 */
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (row >= out_dim) return;
    float acc = 0.0f;
    extern __shared__ uint4 dstage_b[];
    const uint32_t seg = (kblocks > 16u) ? 16u : (uint32_t)kblocks;
    const uint32_t n16 = seg * 9u;
    uint4 *my = dstage_b + (uint64_t)warp * n16;
    for (uint32_t b0 = 0; b0 < (uint32_t)kblocks; b0 += seg) {
        const uint32_t nb = ((uint32_t)kblocks - b0 < seg) ? ((uint32_t)kblocks - b0) : seg;
        const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K) + (uint64_t)b0 * sizeof(cuda_block_q4_K));
        for (uint32_t i = lane; i < nb * 9u; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        /* 半块拆分: 16 块×2 半块=32 单元, 全 warp 活跃(原 b<nb 步长 32 半 warp 空转) */
        const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
        if (bi < nb)
            acc += dev_dot_q4_K_q8_0x8_smem_half(wr + bi, xq + (uint64_t)(b0 + bi) * 256u, xscale + (uint64_t)(b0 + bi) * 8u, hj);
        __syncwarp();
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void grouped_q4_K_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t kblocks,          /* group_dim/256 */
        int use_dp4a) {
    /* 16 lanes/行(2 行/warp): group_dim=4096 时 kblocks=16, 恰一 lane 一块 —
     * 32-lane 版半数 lane 空转(该 kernel 实测 139GB/s=63%)。 */
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t slot = threadIdx.x >> 4u;   /* 0..15 行槽 */
    const uint64_t row = (uint64_t)blockIdx.x * 16u + slot;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * kblocks * 256u;
    const float *xsr = xscale + xrow * kblocks * 8u;
    float acc = 0.0f;
    if (kblocks <= 16u) {   /* 半warp 协作 staging(动态 shared: 静态36KB钉死SM驻留2块) */
        extern __shared__ uint4 dstage_a[];
        const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K));
        const uint32_t n16 = (uint32_t)kblocks * 9u;
        uint4 *my = dstage_a + (uint64_t)slot * n16;
        for (uint32_t i = lane; i < n16; i += 16u) my[i] = src16[i];
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        for (uint64_t b = lane; b < kblocks; b += 16u)
            acc += dev_dot_q4_K_q8_0x8(wr + b, xqr + b * 256u, xsr + b * 8u, use_dp4a);
    } else {
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w + row * kblocks * sizeof(cuda_block_q4_K));
        for (uint64_t b = lane; b < kblocks; b += 16u)
            acc += dev_dot_q4_K_q8_0x8(wr + b, xqr + b * 256u, xsr + b * 8u, use_dp4a);
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void matmul_q4_K_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t kblocks,
        int has_add,
        int use_dp4a) {
    /* epilogue 与 matmul_q8_0_hc_expand_preq_warp8_kernel 逐字同义(post/comb 矩阵 +
     * has_add), 仅权重 dot 换 q4_K×q8_0x8。 */
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (row >= out_dim) return;
    float acc = 0.0f;
    if (kblocks <= 32u) {   /* warp 协作 staging(动态 shared, 同 dense q4 之药)。
         * 半行两段(2026-08-17): kblocks=32 时整行 staging 要 36KB shared → 2 块/SM
         * (160 GB/s 卡点); 按 16 块一段分两段搬+算, shared 减半 → 4 块/SM。 */
        extern __shared__ uint4 dstage_b[];
        const uint32_t seg = (kblocks > 16u) ? 16u : (uint32_t)kblocks;
        const uint32_t n16 = seg * 9u;
        uint4 *my = dstage_b + (uint64_t)warp * n16;
        for (uint32_t b0 = 0; b0 < (uint32_t)kblocks; b0 += seg) {
            const uint32_t nb = ((uint32_t)kblocks - b0 < seg) ? ((uint32_t)kblocks - b0) : seg;
            const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K) + (uint64_t)b0 * sizeof(cuda_block_q4_K));
            for (uint32_t i = lane; i < nb * 9u; i += 32u) my[i] = src16[i];
            __syncwarp();
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
            for (uint32_t b = lane; b < nb; b += 32u)
                acc += dev_dot_q4_K_q8_0x8(wr + b, xq + (uint64_t)(b0 + b) * 256u, xscale + (uint64_t)(b0 + b) * 8u, use_dp4a);
            __syncwarp();
        }
    } else {
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w + row * kblocks * sizeof(cuda_block_q4_K));
        for (uint64_t b = lane; b < kblocks; b += 32u)
            acc += dev_dot_q4_K_q8_0x8(wr + b, xq + b * 256u, xscale + b * 8u, use_dp4a);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

