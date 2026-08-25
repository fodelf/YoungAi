/* cuda_qk_warp_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * q4_K/q2_K warp gemv kernel 族与 dense 入口。
 */
/* block 内并行 q8_K 量化(2026-08-20 碎片税刀①): 8 warp × 各承包块, warp 内 lane 持
 * 连续 8 元素。amax 选择 = "高索引挑战低索引, 严格大于才胜" 与原 256 线程树同语义
 * (全局 lowest-index argmax) ⇒ qs/d/bsums 逐位一致。写进 shared xqsh。 */
__device__ static void q2k_fused_quantize(cuda_block_q8_K *xqsh, const float *x, uint32_t blocks) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (uint32_t b = warp; b < blocks; b += 8u) {
        const float *xr = x + (uint64_t)b * 256u;
        cuda_block_q8_K *yb = xqsh + b;
        /* lane 扫连续 8 元素: 保低 index 平手 */
        float amax = 0.0f, maxv = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const float v = xr[lane * 8u + i];
            const float a = fabsf(v);
            if (a > amax) { amax = a; maxv = v; }
        }
        /* shuffle 树: 高 lane 挑战低 lane, 严格大于才胜 */
        for (int o = 16; o; o >>= 1) {
            const float oa = __shfl_down_sync(0xffffffffu, amax, o);
            const float ov = __shfl_down_sync(0xffffffffu, maxv, o);
            if (oa > amax) { amax = oa; maxv = ov; }
        }
        amax = __shfl_sync(0xffffffffu, amax, 0);
        maxv = __shfl_sync(0xffffffffu, maxv, 0);
        if (amax == 0.0f) {
            if (lane == 0) yb->d = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; i++) yb->qs[lane * 8u + i] = 0;
            if (lane < 16u) yb->bsums[lane] = 0;
            continue;
        }
        const float iscale = -127.0f / maxv;
        int psum = 0;   /* lane 的 8 元素恰是半个 bsums 组(16 元素=2 lanes) */
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            int qv = (int)lrintf(iscale * xr[lane * 8u + i]);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            yb->qs[lane * 8u + i] = (int8_t)qv;
            psum += qv;
        }
        const int osum = __shfl_down_sync(0xffffffffu, psum, 1);
        if ((lane & 1u) == 0u && lane < 32u) yb->bsums[lane >> 1] = (int16_t)(psum + osum);
        if (lane == 0) yb->d = 1.0f / iscale;
    }
}

template <uint32_t NB>   /* 施工日3特化刀: blocks 编译常量(NB>0)→全展开+静态寻址
                            (阶梯常量核228 vs 真核动态173 的最后差异嫌疑); NB=0 动态。 */
__global__ static void __launch_bounds__(256, DS4_Q2K_BLOCKS_PER_SM) matmul_q2_K_warp_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks_dyn, uint32_t out_dim, uint32_t q2k_stage_lv,
        uint32_t q2k_split_maxblk, const float *xraw, uint32_t n_tok_batch, uint32_t n_tok_total,
        float *partial, uint32_t ksplit, uint32_t x_in_smem) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    /* dense Q2_K 占用率手术(2026-08-20)+量化融合(碎片税刀①): xraw!=NULL(decode n=1)
     * 时首段 block 内量化到 shared, 免独立 quantize 发射(414发/tok 的大头)。
     * staging: warp 协作 uint4(blocks%4==0)+半块/八分拆。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    /* 权重驻留批(2026-08-21 verify 8x 偏离根因): n_tok_batch>1 时 grid.y=1,
     * 块内循环 token 复用已 staged 的权重行 —— 此前 grid.y=n_tok 让同一份权重被
     * 每个 token 各读一遍(6 tok=6x 带宽)。逐 (row,tok) 数学与旧路逐位相同。 */
    /* token 分组(2026-08-21): grid.y = 组数, 每块吃 n_tok_batch 个 token。
     * 全驻留(组=1)省带宽但并行度只剩 gx 块 —— 小 out_dim 会饿死(o_b 实测 54GB/s vs
     * down tile 同形状 151GB/s, 差在 warp 数 3.75x)。分组让两者可调和。 */
    const uint32_t tpg = n_tok_batch ? n_tok_batch : 1u;
    const uint32_t ntot = n_tok_total ? n_tok_total : tpg;
    const uint32_t tok_base = blockIdx.y * tpg;
    if (tok_base >= ntot) return;
    const uint32_t ntb = (tpg < ntot - tok_base) ? tpg : (ntot - tok_base);
    const uint32_t stage_ok = (blocks <= 32u && (blocks & 3u) == 0u) && (q2k_stage_lv > 0u);
    const uint32_t n16 = blocks * 84u / 16u;   /* 行 uint4 数(blocks%4==0 ⇒ 整) */
    extern __shared__ uint4 q2shm[];
    const cuda_block_q8_K *xt;
    if (xraw) {   /* 融合量化(n_tok==1 门在入口) */
        cuda_block_q8_K *xqsh = (cuda_block_q8_K *)(q2shm + (stage_ok ? (uint64_t)8u * n16 : 0u));
        q2k_fused_quantize(xqsh, xraw, blocks);
        __syncthreads();
        xt = xqsh;
    } else if (stage_ok && x_in_smem) {
        /* 批激活进 shared(2026-08-21): 原本每个 warp 为自己的行各读一遍这 ntb 份 q8 激活,
         * 一个 block 8 个 warp ⇒ 同一份激活被读 8 遍, 整个矩阵每 token 约 38MB 冗余 L2 流量。
         * 块内协作搬一次, 之后全 warp 共用。数值不变(只换读取来源)。 */
        cuda_block_q8_K *xsh = (cuda_block_q8_K *)(q2shm + (uint64_t)8u * n16);
        const uint32_t words = (uint32_t)((ntb * blocks * sizeof(cuda_block_q8_K)) / sizeof(uint4));
        const uint4 *src4 = (const uint4 *)(xq + (uint64_t)tok_base * blocks);
        uint4 *dst4 = (uint4 *)xsh;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst4[i] = src4[i];
        __syncthreads();
        xt = xsh;
    } else {
        xt = xq + (uint64_t)tok_base * blocks;
    }
    /* K 维切分(2026-08-21): out_dim 小的矩阵(o_b 只有 512 块)并行度不足 — 同字节下
     * q_b(4096 块)跑 170GB/s 而 o_b 只有 88。按 K 分段各算部分和, reduce 固定 z 序合并。
     * ksplit==1 时与旧路逐位相同。 */
    const uint32_t kz = (ksplit > 1u) ? blockIdx.z : 0u;
    const uint32_t kchunk = (blocks + ksplit - 1u) / ksplit;
    const uint32_t kb0 = kz * kchunk;
    const uint32_t kb1 = (kb0 + kchunk < blocks) ? (kb0 + kchunk) : blocks;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        float acc = 0.0f;
        if (stage_ok) {
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = q2shm + (uint64_t)warp * n16;
            const uint32_t s_lo = (ksplit > 1u) ? (kb0 * 84u / 16u) : 0u;
            const uint32_t s_hi = (ksplit > 1u) ? (kb1 * 84u / 16u) : n16;
            for (uint32_t i = lane + s_lo; i < s_hi; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)my;
            if (q2k_stage_lv == 9u && blocks == 4u) {   /* 现场对质: 全行全tok计数 */
                float f = 0.0f, sp = 0.0f;
                if (lane == 0) {
                    for (uint32_t b = 0; b < blocks; b++) f += dev_dot_q2_K_q8_K_block(wr + b, xt + b);
                    for (uint32_t b = 0; b < blocks; b++)
                        for (uint32_t q = 0; q < 8u; q++) sp += dev_dot_q2_K_q8_K_block_eighth(wr + b, xt + b, q);
                    const float df = fabsf(f - sp);
                    if (df > 1e-4f && df > 1e-3f * fabsf(f))
                        printf("[q2bad] row=%u tok=%u full=%.6f split=%.6f\n", row, tok_base, f, sp);
                }
            }
            /* 走 smem 通路且是多 token 批: 用多 token 复用解量化版(4 个一组), 权重只解一次。
             * 这条只改 ALU 复用, 逐 (row,tok) 数值与单 token 路逐位相同。 */
            const bool multi_ok = (ntb > 1u) && !xraw &&
                                  !(blocks == 4u && q2k_stage_lv >= 2u && q2k_split_maxblk >= 4u) &&
                                  !(blocks <= 16u && q2k_stage_lv >= 2u && blocks <= q2k_split_maxblk);
            if (multi_ok) {
                for (uint32_t t0 = 0; t0 < ntb; t0 += 4u) {
                    const uint32_t nt = (ntb - t0 < 4u) ? (ntb - t0) : 4u;
                    float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    const cuda_block_q8_K *ybase = (x_in_smem ? xt : (xq + (uint64_t)tok_base * blocks))
                                                   + (uint64_t)t0 * blocks;
                    for (uint32_t b = lane + kb0; b < kb1; b += 32u) {
                        switch (nt) {
                        case 4: dev_dot_q2_K_q8_K_block_smem_multi<4>(wr + b, ybase + b, blocks, a4); break;
                        case 3: dev_dot_q2_K_q8_K_block_smem_multi<3>(wr + b, ybase + b, blocks, a4); break;
                        case 2: dev_dot_q2_K_q8_K_block_smem_multi<2>(wr + b, ybase + b, blocks, a4); break;
                        default: dev_dot_q2_K_q8_K_block_smem_multi<1>(wr + b, ybase + b, blocks, a4); break;
                        }
                    }
                    for (uint32_t i = 0; i < nt; i++) {
                        float v = a4[i];
                        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
                        if (lane == 0) {
                            if (ksplit > 1u) partial[((uint64_t)kz * n_tok_total + (tok_base + t0 + i)) * out_dim + row] = v;
                            else out[(uint64_t)(tok_base + t0 + i) * out_dim + row] = v;
                        }
                    }
                }
                __syncwarp();
                continue;
            }
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xtt = (xraw || x_in_smem) ? (xt + (uint64_t)(xraw ? 0u : t) * blocks)
                                                                 : (xq + (uint64_t)(tok_base + t) * blocks);
                float a2 = 0.0f;
                if (blocks == 4u && q2k_stage_lv >= 2u && q2k_split_maxblk >= 4u) {
                    const uint32_t bi = lane >> 3u, qi = lane & 7u;
                    a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xtt + bi, qi);
                } else if (blocks <= 16u && q2k_stage_lv >= 2u && blocks <= q2k_split_maxblk) {
                    const uint32_t bi = lane >> 1u, h = lane & 1u;
                    if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xtt + bi, h);
                } else {
                    for (uint32_t b = lane + kb0; b < kb1; b += 32u)
                        a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xtt + b);
                }
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (lane == 0) {
                    if (ksplit > 1u) partial[((uint64_t)kz * n_tok_total + (tok_base + t)) * out_dim + row] = a2;
                    else out[(uint64_t)(tok_base + t) * out_dim + row] = a2;
                }
            }
            __syncwarp();   /* 下轮复写 stage 前全 lane 必须算完 */
            continue;
        } else {
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(w_base + (uint64_t)row * row_bytes);
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xtt = xraw ? xt : (xq + (uint64_t)(tok_base + t) * blocks);
                float a2 = 0.0f;
                for (uint32_t b = lane; b < blocks; b += 32u)
                    a2 += dev_dot_q2_K_q8_K_block(wr + b, xtt + b);
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (lane == 0) out[(uint64_t)(tok_base + t) * out_dim + row] = a2;
            }
            continue;
        }
        (void)acc;
    }
}

__global__ static void matmul_q4_K_warp_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim) {
    /* 每 warp 一行(32 lanes 分块)。A/B 记录: 8-lane×32行/block 版实测更慢
     * (16.84 vs 17.30 t/s) —— dev_dot 每块开销大, 每 lane 串行 2 块比半数 lane
     * 空转更亏。保留本版。 */
    /* warp 协作 shared staging(2026-08-17 第三轮): 此前每 lane 直接读自己的块,
     * warp 内"第 i 次 load"跨 lane 相距 144B ⇒ 不合并。现在 warp 把整行权重
     * (≤32 块×144B)按 lane 连续 16B 粒度搬进 shared(完全合并), 再各 lane 从片上算。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t tok = blockIdx.y;
    const cuda_block_q8_K *xt = xq + (uint64_t)tok * blocks;
    /* grid-stride 行循环(2026-08-17): out=2048 的小矩阵原来发 256 块, 在飞上限 ~192
     * → 1.33 个 wave, 尾波白扔 ~25% 吞吐。现在固定小 grid 循环吃完所有行。 */
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        float acc = 0.0f;
        if (blocks <= 32u) {
            /* 动态 shared: 静态 [8][32*9] 常驻 36KB/块把 SM 驻留钉在 2 块(occupancy 墙);
             * 按实际行字节配(launch 第三参), blocks=8 时只要 9KB。 */
            extern __shared__ uint4 dstage[];
            const uint32_t n16 = blocks * 9u;
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = dstage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
            if (blocks <= 16u) {
                /* 半块拆分: blocks×2 单元≤32, 全 warp 活跃 */
                const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
                if (bi < blocks) acc += dev_dot_q4_K_q8_K_block_half(wr + bi, xt + bi, hj);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u)
                    acc += dev_dot_q4_K_q8_K_block(wr + b, xt + b);
            }
            __syncwarp();   /* 下轮复写 stage 前全 lane 必须算完 */
        } else {
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w_base + (uint64_t)row * row_bytes);
            for (uint32_t b = lane; b < blocks; b += 32u)
                acc += dev_dot_q4_K_q8_K_block_vec(wr + b, xt + b);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(uint64_t)tok * out_dim + row] = acc;
    }
}

int ds4_gpu_matmul_q4_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q4_K);
    const uint64_t w_bytes = row_bytes * out_dim;
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "dense_q4k");
    if (!w) return 0;

    const uint64_t xq_need = n_tok * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    /* 该 kernel 的 grid 语义: blockIdx.x 就是量化块号(一 CUDA block 一 q8_K 块) */
    q8_K_quantize_kernel<<<dim3(blocks, (unsigned)n_tok, 1), 256, 0, g_cur_stream>>>(
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    /* A/B 三连档案: 16-lane 变体在小矩阵上仍输(18.84 vs 19.13) — dense 场景 32-lane
     * 恒胜, dev_dot 每块开销决定一切, lane 空转无关紧要。变体保留但不启用。 */
    if (((const char *)0) /* DS4_Q4K_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
        static uint64_t seen[64][2]; static int nseen = 0;
        int hit = 0;
        for (int i = 0; i < nseen; i++) if (seen[i][0] == in_dim && seen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && nseen < 64) {
            seen[nseen][0] = in_dim; seen[nseen][1] = out_dim; nseen++;
            fprintf(stderr, "ds4: [q4k-dims] in=%llu out=%llu ntok=%llu\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, (unsigned long long)n_tok);
        }
    }
    /* 动态 shared: blocks≤32 时 kernel staging 需 8 warps×blocks×144B */
    const size_t q4k_shmem = (blocks <= 32u) ? (size_t)8u * blocks * 9u * sizeof(uint4) : 0;
    unsigned q4k_gx = (unsigned)((out_dim + 7u) / 8u);
    if (q4k_gx > ds4_grid_cap()) q4k_gx = ds4_grid_cap();   /* 48 SM × 4 驻留块: 单 wave 满载, 行循环吃尾 */
    matmul_q4_K_warp_kernel<<<dim3(q4k_gx, (unsigned)n_tok, 1), 256, q4k_shmem, g_cur_stream>>>(
        (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
        row_bytes, blocks, (uint32_t)out_dim);
    return cuda_ok(cudaGetLastError(), "dense q4_K matmul launch");
}

/* 分组投影的分片 GEMM(2026-08-21): 与 dense tiled 同一套修法。行 r 属组 r/rank, 只需要
 * 该组那一段激活; 一个 block 的 8 行同组(rank≫8), 于是 tile 内只搬这一组的 T 份激活。
 * 原 kernel 每 warp 每 token 各读一遍 ⇒ 8192 行 × 340 token × 2.3KB ≈ 6.4GB; 分片后
 * 只剩 blocks × token × 2.3KB。lane 映射与 dot 函数不变 ⇒ 逐位一致。 */
template <uint32_t NB>
__global__ static void __launch_bounds__(256, 1) grouped_q2_K_tiled_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks_dyn, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups, uint32_t n_tok, uint32_t tile) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 gtshm[];
    cuda_block_q8_K *acts = (cuda_block_q8_K *)gtshm;
    uint4 *wsh = (uint4 *)(acts + (uint64_t)tile * blocks);

    const uint32_t base0 = blockIdx.x * 8u;
    const uint32_t stride = gridDim.x * 8u;
    const uint32_t n_iter = (base0 >= out_dim) ? 0u : ((out_dim - base0 + stride - 1u) / stride);
    /* 本块第一行所在组: 块内 8 行同组(rank 是 1024 量级), 跨迭代可能换组 ⇒ 每迭代按行算组,
     * 但激活 tile 是按"块当前迭代的组"搬的 ⇒ 行循环放外层, tile 放内层。 */
    for (uint32_t it = 0; it < n_iter; it++) {
        const uint32_t row = base0 + warp + it * stride;
        const bool live = row < out_dim;
        const uint32_t grp = live ? (row / rank) : 0u;
        /* 块内 8 个 warp 的组必须一致才能共用 tile —— rank≥8 时天然成立 */
        const uint32_t grp_blk = (base0 + it * stride) / rank;
        const cuda_block_q2_K *wr = NULL;
        if (live) {
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = wsh + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            wr = (const cuda_block_q2_K *)my;
        }
        for (uint32_t t0 = 0; t0 < n_tok; t0 += tile) {
            const uint32_t tn = (n_tok - t0 < tile) ? (n_tok - t0) : tile;
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < tn * blocks; i += blockDim.x) {
                const uint32_t tt = i / blocks, bb = i - tt * blocks;
                acts[(uint64_t)tt * blocks + bb] =
                    xq[((uint64_t)(t0 + tt) * n_groups + grp_blk) * blocks + bb];
            }
            __syncthreads();
            for (uint32_t t = 0; t < tn; t++) {
                float acc = 0.0f;
                if (live && grp == grp_blk) {
                    const cuda_block_q8_K *xt = acts + (uint64_t)t * blocks;
                    if (blocks <= 16u) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                } else if (live) {
                    /* 组不一致(理论上不会发生, rank≥8): 退回全局读, 保正确 */
                    const cuda_block_q8_K *xt = xq + ((uint64_t)(t0 + t) * n_groups + grp) * blocks;
                    if (blocks <= 16u) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (live && lane == 0) out[(uint64_t)(t0 + t) * out_dim + row] = acc;
            }
        }
        __syncthreads();
    }
}

static bool grouped_q2k_tiled_set_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx]) return cap[idx] >= bytes;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

static inline void grouped_q2k_tiled_launch(unsigned gx, size_t sh,
        float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups, uint32_t n_tok, uint32_t tile) {
    switch (blocks) {
    case 4u:  grouped_q2_K_tiled_kernel<4u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    case 8u:  grouped_q2_K_tiled_kernel<8u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    case 16u: grouped_q2_K_tiled_kernel<16u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    case 32u: grouped_q2_K_tiled_kernel<32u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    default:  grouped_q2_K_tiled_kernel<0u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    }
}

__global__ static void __launch_bounds__(256, 6) grouped_q2_K_warp_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups, uint32_t n_tok_batch) {
    /* attn_output_a 分组结构(全q2, 2026-08-19): 行 r 属组 r/rank, 读 x 的第 g 段。
     * 2026-08-20 提速刀②改: 实测形状 blocks=16/低维8192 — 与 dense 同款"裸读克隆"病
     * (129GB/s), 移植同款 staging+半块手术。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t ntb = n_tok_batch ? n_tok_batch : 1u;   /* 权重驻留批(见 dense 同款) */
    const uint32_t tok_base = (ntb > 1u) ? 0u : blockIdx.y;
    const uint32_t stage_ok = (blocks <= 32u && (blocks & 3u) == 0u);
    const uint32_t n16 = blocks * 84u / 16u;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint32_t grp = row / rank;
        if (stage_ok) {
            extern __shared__ uint4 gq2stage[];
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = gq2stage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)my;
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xt = xq + ((uint64_t)(tok_base + t) * n_groups + grp) * blocks;
                float acc = 0.0f;
                if (blocks <= 16u) {
                    const uint32_t bi = lane >> 1u, h = lane & 1u;
                    if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                } else {
                    for (uint32_t b = lane; b < blocks; b += 32u)
                        acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                }
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)(tok_base + t) * out_dim + row] = acc;
            }
            __syncwarp();
        } else {
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(w_base + (uint64_t)row * row_bytes);
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xt = xq + ((uint64_t)(tok_base + t) * n_groups + grp) * blocks;
                float acc = 0.0f;
                for (uint32_t b = lane; b < blocks; b += 32u)
                    acc += dev_dot_q2_K_q8_K_block(wr + b, xt + b);
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)(tok_base + t) * out_dim + row] = acc;
            }
        }
    }
}

/* 小块专用变体(2026-08-20 提速刀②): blocks≤4 时 warp-per-row 模板 32 lane 里 28 个空转
 * (nsys 实测 3.68ms/tok ≈ 6GB/s 纯延迟病)。改一 lane 一行(256 行/块), 低寄存器 dot。 */
__global__ static void grouped_q2_K_rowlane_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups) {
    const uint32_t tok = blockIdx.y;
    const uint32_t row = blockIdx.x * 256u + threadIdx.x;
    if (row >= out_dim) return;
    const uint32_t grp = row / rank;
    const cuda_block_q8_K *xt = xq + ((uint64_t)tok * n_groups + grp) * blocks;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++)
        acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
    out[(uint64_t)tok * out_dim + row] = acc;
}

