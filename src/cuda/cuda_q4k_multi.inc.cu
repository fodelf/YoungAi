/* cuda_q4k_multi.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * Q4_K 稠密 gemv 的小批多 token 核(2026-09-07, 投机 verify 批 2..8 token; 09-07 晚从 cuda_q4k_tile 拆出并改预解 nibble)。
 * 结构: 一 tile/一行权重 stage 一次, lane 各一整块, 尺度 + nibble 各解一次, 逐 token 只剩 dp4a + 同一棵归约树
 * (每 token 与单 token 核逐位同; grid.y=token 的老法 08-21 定罪"verify 8× 偏离")。
 * ★地址空间(09-07 微基准定罪)★: "stage_x ? shared 指针 : 全局指针"这种运行期二选一让编译器只能发 generic load(ld 而非
 * ld.shared), 引擎里同核 384 µs vs 微基准 164。所以 STAGE 做成模板参数: 两条路分开编译, 各自指针的地址空间编译期可知。 */
/* tile 多 token(BLOCKS=4/8/16): 09-07 晚试过"三段式 stage + 16 B 向量读 + 预解 nibble"(链 30/32), 所有形状反慢 25~75%
 * (grid 32: 14.9 → 18.7 µs, shexp 128 块: 29 → 51), 只有超大网格受益于封顶; 故本核保留 p20 版核体(结构体 stage + 标量 dot),
 * 网格只对 >4 波的超大矩阵封顶(drafter 输出头 2.5 → 1.6 ms)。rows_q8_0/grouped 的向量读版是赚的, 各自保留。 */
template <uint32_t BLOCKS, bool STAGE>
__global__ static void q4k_tile_multi_kernel(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t out_dim,
                                             uint32_t n_tok) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 q4k_stage[];
    cuda_block_q8_K *xs_sh = (cuda_block_q8_K *)(q4k_stage + 8u * n16);
    if (STAGE) {
        uint32_t *dst = (uint32_t *)xs_sh;
        const uint32_t *src = (const uint32_t *)xq;
        const uint32_t n_words = n_tok * BLOCKS * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < n_words; i += blockDim.x) dst[i] = src[i];
        __syncthreads();
    }
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint4 *src16 = (const uint4 *)(w + (uint64_t)t * R * BLOCKS * sizeof(cuda_block_q4_K));
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my + rin * BLOCKS + b;
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = STAGE ? dev_dot_q4_K_q8_K_block(wr, xs_sh + (uint64_t)tk * BLOCKS + b)
                              : dev_dot_q4_K_q8_K_block(wr, xq + (uint64_t)tk * BLOCKS + b);
            acc = q4k_seg_sum<BLOCKS>(acc);
            if (b == 0u) out[(uint64_t)tk * out_dim + t * R + rin] = acc;
        }
        __syncwarp();
    }
}
/* attn_output_a(分组投影, q8_0 激活; 解码 n_tok=1 与 verify 小批共用): 行 row 属组 row/rank, 只乘该组的激活段
 * (第 (tk*n_groups+group)*BLOCKS 块起); 一 tile 的 R 行同组(rank 是 R 的倍数, 调用方校验)。
 * 权重块尺度/nibble 每 tile 解码一次供 n_tok 个 token 复用, 激活从全局读(q8_0 子块 32 B 对齐, 合并读;
 * 全表 Σq8 预算要每 block 读全部激活 128 KB, 实测 182 → 215 µs 反慢, 故 Σq8 lane 内现算)。 */
template <uint32_t BLOCKS>
__global__ static void q4k_tile_grouped_multi_kernel(float *low, const char *w, const int8_t *xq, const float *xs,
                                                     uint32_t rank, uint32_t n_groups, uint32_t n_tok) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 q4k_stage[];
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, low_dim = n_groups * rank, ntiles = low_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint32_t row0 = t * R, group = row0 / rank;
        const uint4 *src16 = (const uint4 *)(w + (uint64_t)row0 * BLOCKS * sizeof(cuda_block_q4_K));
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my + rin * BLOCKS + b;
        q4k_wblk wb; q4k_wblk_decode(wr, &wb);
        q4k_wnib nb; q4k_wnib_unpack(wr, &nb);
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            const uint64_t bi = ((uint64_t)tk * n_groups + group) * BLOCKS + b;
            float acc = dev_dot_q4_K_q8_0x8_wpre_nib_v(wb, nb, xq + bi * 256u, xs + bi * 8u);
            acc = q4k_seg_sum<BLOCKS>(acc);
            if (b == 0u) low[(uint64_t)tk * low_dim + row0 + rin] = acc;
        }
        __syncwarp();
    }
}
/* 17..32 块的行(q8_K 激活)多 token: 一 warp 一行整行 stage 一次, lane 各一整块, 逐 token 归约;
 * 与 cuda_qk_warp_2 matmul_q4_K_warp_kernel 的 blocks≤32 分支同 dot 同加序。 */
template <bool STAGE>
__global__ static void q4k_rows_multi_kernel(float *out, const char *w, const cuda_block_q8_K *xq,
                                             uint32_t blocks, uint32_t out_dim, uint32_t n_tok) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, n16 = blocks * 9u;
    extern __shared__ uint4 q4k_stage[];
    q4k_q8K_split xa; xa.qs = NULL; xa.d = NULL; xa.bs = NULL;
    if (STAGE) xa = q4k_stage_q8K_split(q4k_stage + 8u * n16, xq, n_tok * blocks);
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src16 = (const uint4 *)(w + (uint64_t)row * blocks * sizeof(cuda_block_q4_K));
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        q4k_wblk wb; q4k_wnib nb;   /* lane 各一整块(blocks ≤ 32, 发射方校验) */
        if (lane < blocks) { q4k_wblk_decode(wr + lane, &wb); q4k_wnib_unpack(wr + lane, &nb); }
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = 0.0f;
            if (lane < blocks) {
                const uint32_t bi = tk * blocks + lane;
                acc = STAGE ? dev_dot_q4_K_q8_K_split_nib(wb, nb, xa.qs + (uint64_t)bi * 256u, xa.d[bi], xa.bs + (uint64_t)bi * 16u)
                            : dev_dot_q4_K_q8_K_block_nib(wb, nb, xq + bi);
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
        }
        __syncwarp();
    }
}
/* 每 block 动态 shared 可选上限(GB10 101376 B); 权重 stage 区之外还装得下 n_tok 份激活才 stage_x。
 * 超过 48 KB 的核要 cudaFuncSetAttribute 放行一次(按模板实例各一次)。 */
static size_t q4k_shm_optin(void) {
    static size_t v = 0;
    if (!v) {
        int optin = 0, dev = 0;
        (void)cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) { (void)cudaGetLastError(); optin = 48 * 1024; }
        v = (size_t)optin;
    }
    return v;
}
template <typename K>
static int q4k_shm_plan(K kern, size_t base, size_t x_bytes, size_t *shm) {   /* 返回 stage_x */
    if (base + x_bytes <= 48u * 1024u) { *shm = base + x_bytes; return 1; }
    if (base + x_bytes <= q4k_shm_optin()) {
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(base + x_bytes));
        if (e == cudaSuccess) { *shm = base + x_bytes; return 1; }
        static int warned = 0;   /* 退回读全局的路慢 2~3×, 必须看得见 */
        if (!warned++) fprintf(stderr, "ds4: q4k 小批核 shared 放行失败(%zu B, 上限 %zu): %s; 激活改读全局\n",
                               base + x_bytes, q4k_shm_optin(), cudaGetErrorString(e));
    } else {
        static int warned2 = 0;
        if (!warned2++) fprintf(stderr, "ds4: q4k 小批核 shared 需 %zu B 超上限 %zu; 激活改读全局\n", base + x_bytes, q4k_shm_optin());
    }
    (void)cudaGetLastError();
    *shm = base; return 0;
}
/* attn_output_b 小批(verify), q8_0 激活口径: 与解码 q4k_hc_expand_kernel 同 dot(dev_dot_q4_K_q8_0x8_smem)同加序
 * (整行 stage, lane 各一整块, 32 lane 树归约), 只是逐 token。此前批路 out_b 走 q8_K 激活的 matmul_q4_K ——
 * 与解码不同轨(激活量化粒度 32→256), 是投机与纯解码输出分叉的一处根因(09-07)。 */
template <bool STAGE>
__global__ static void q4k_rows_q8_0_multi_kernel(float *out, const char *w, const int8_t *xq, const float *xs,
                                                  uint32_t kblocks, uint32_t out_dim, uint32_t n_tok) {
    DS4_PDL_WAIT();   /* 只等不触发: 77 KB shared ⇒ 每 SM 1 block, 后继核提前上 SM 占槽反而拖慢 */
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, n16 = kblocks * 9u;
    extern __shared__ uint4 q4k_stage[];
    uint4 *my = q4k_stage + (uint64_t)warp * n16;
    /* STAGE: 值 n_tok×kblocks×256 B + 尺度 n_tok×kblocks×8 f32 + Σq8 n_tok×kblocks×8 i32 进 shared(指针地址空间编译期可知) */
    uint32_t *dst = (uint32_t *)(q4k_stage + 8u * n16);
    const uint32_t nq = n_tok * kblocks * 64u, ns = n_tok * kblocks * 8u;
    const int8_t *xq_sh = (const int8_t *)dst;
    const float *xs_sh = (const float *)dst + nq;
    int32_t *s8 = (int32_t *)dst + nq + ns;
    if (STAGE) {
        for (uint32_t i = threadIdx.x; i < nq; i += blockDim.x) dst[i] = ((const uint32_t *)xq)[i];
        for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) ((float *)dst)[nq + i] = xs[i];
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) {   /* 子块 (tok,块,j) 的 32 值和: 8 个 dp4a */
            const int8_t *q8 = xq_sh + (uint64_t)i * 32u;
            int32_t sm = 0;
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k += 4u) sm = __dp4a(0x01010101, *(const int32_t *)(q8 + k), sm);
            s8[i] = sm;
        }
        __syncthreads();
    }
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src16 = (const uint4 *)(w + (uint64_t)row * kblocks * sizeof(cuda_block_q4_K));
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        if (STAGE) {   /* lane 各一整块(kblocks ≤ 32, 发射方校验): 尺度/nibble 解码一次, 逐 token 只剩 dp4a */
            q4k_wblk wb; q4k_wnib nb; const uint32_t b = lane;
            if (b < kblocks) { q4k_wblk_decode(wr + b, &wb); q4k_wnib_unpack(wr + b, &nb); }
            for (uint32_t tk = 0; tk < n_tok; tk++) {
                float acc = 0.0f;
                if (b < kblocks) {
                    const uint64_t bi = (uint64_t)tk * kblocks + b;
                    acc = dev_dot_q4_K_q8_0x8_pre_nib_v(wb, nb, xq_sh + bi * 256u, xs_sh + bi * 8u, s8 + bi * 8u);
                }
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
            }
        } else {
            q4k_act_q8_0 xbase; xbase.xq = xq; xbase.xs = xs;
            for (uint32_t tk = 0; tk < n_tok; tk++) {
                const q4k_act_q8_0 x = xbase.at((uint64_t)tk * kblocks);
                float acc = 0.0f;
                for (uint32_t b = lane; b < kblocks; b += 32u) acc += x.dot(wr + b, b);
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
            }
        }
        __syncwarp();
    }
}
static int q4k_rows_q8_0_multi_launch(float *out, const char *w, const int8_t *xq, const float *xs, uint32_t kblocks,
                                      uint32_t out_dim, uint32_t n_tok) {
    if (kblocks > 32u) { fprintf(stderr, "ds4: q4k rows q8_0 multi: 行 %u 块超过 32(整行 stage 上限)\n", kblocks); return 0; }
    unsigned g = (out_dim + 7u) / 8u;
    if (g > ds4_grid_cap()) g = ds4_grid_cap();
    size_t shm = 0;
    const int sx = q4k_shm_plan(q4k_rows_q8_0_multi_kernel<true>, (size_t)8u * kblocks * 9u * sizeof(uint4),
                                (size_t)n_tok * kblocks * (256u + 32u + 32u), &shm);   /* 值 + 尺度 + Σq8 */
    if (sx) g = q4k_grid_wave(shm, g, 2u);
    if (sx) ds4_launch_pdl(q4k_rows_q8_0_multi_kernel<true>, g, 256, shm, g_cur_stream, out, w, xq, xs, kblocks, out_dim, n_tok);
    else    ds4_launch_pdl(q4k_rows_q8_0_multi_kernel<false>, g, 256, shm, g_cur_stream, out, w, xq, xs, kblocks, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "q4k rows q8_0 multi launch");
}
static int q4k_rows_multi_launch(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t blocks,
                                 uint32_t out_dim, uint32_t n_tok) {
    if (blocks > 32u) { fprintf(stderr, "ds4: q4k rows multi: 行 %u 块超过 32(整行 stage 上限)\n", blocks); return 0; }
    unsigned g = (out_dim + 7u) / 8u;
    if (g > ds4_grid_cap()) g = ds4_grid_cap();
    size_t shm = 0;
    const int sx = q4k_shm_plan(q4k_rows_multi_kernel<true>, (size_t)8u * blocks * 9u * sizeof(uint4),
                                (size_t)n_tok * blocks * sizeof(cuda_block_q8_K), &shm);
    if (sx) g = q4k_grid_wave(shm, g, 2u);
    if (sx) ds4_launch_pdl(q4k_rows_multi_kernel<true>, g, 256, shm, g_cur_stream, out, w, xq, blocks, out_dim, n_tok);
    else    ds4_launch_pdl(q4k_rows_multi_kernel<false>, g, 256, shm, g_cur_stream, out, w, xq, blocks, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "q4k rows multi launch");
}
/* 小批 tile: 权重一遍, 逐 token(与单 token 核逐位同); 激活进 shared(装得下时, 见 q4k_shm_plan) */
static int q4k_tile_multi_launch(float *out, const char *w, const cuda_block_q8_K *xq, uint32_t blocks,
                                 uint32_t out_dim, uint32_t n_tok) {
    unsigned gx = q4k_tile_grid(out_dim / (32u / blocks));
    const size_t xb = (size_t)n_tok * blocks * sizeof(cuda_block_q8_K);
    size_t shm = 0; int sx = 0;
#define DS4_Q4K_TM_LAUNCH(B) do { sx = q4k_shm_plan(q4k_tile_multi_kernel<B, true>, Q4K_TILE_SHM, xb, &shm); \
    if (sx) { gx = q4k_grid_wave(shm, gx, 4u); ds4_launch_pdl(q4k_tile_multi_kernel<B, true>, gx, 256, shm, g_cur_stream, out, w, xq, out_dim, n_tok); } \
    else    ds4_launch_pdl(q4k_tile_multi_kernel<B, false>, gx, 256, shm, g_cur_stream, out, w, xq, out_dim, n_tok); } while (0)
    switch (blocks) {
        case 4u:  DS4_Q4K_TM_LAUNCH(4u); break;
        case 8u:  DS4_Q4K_TM_LAUNCH(8u); break;
        default:  DS4_Q4K_TM_LAUNCH(16u); break;
    }
#undef DS4_Q4K_TM_LAUNCH
    return cuda_ok(cudaGetLastError(), "q4k tile multi launch");
}
static int q4k_tile_grouped_launch(float *low, const char *w, const int8_t *xq, const float *xs, uint32_t blocks,
                                   uint32_t rank, uint32_t n_groups, uint32_t n_tok) {
    const uint32_t R = 32u / blocks;
    if (rank % R != 0u) { fprintf(stderr, "ds4: q4k grouped tile: rank %u 不是 %u 的倍数\n", rank, R); return 0; }
    const unsigned gx = q4k_tile_grid(n_groups * rank / R);
    switch (blocks) {
        case 4u:  ds4_launch_pdl(q4k_tile_grouped_multi_kernel<4u>, gx, 256, Q4K_TILE_SHM, g_cur_stream, low, w, xq, xs, rank, n_groups, n_tok); break;
        case 8u:  ds4_launch_pdl(q4k_tile_grouped_multi_kernel<8u>, gx, 256, Q4K_TILE_SHM, g_cur_stream, low, w, xq, xs, rank, n_groups, n_tok); break;
        default:  ds4_launch_pdl(q4k_tile_grouped_multi_kernel<16u>, gx, 256, Q4K_TILE_SHM, g_cur_stream, low, w, xq, xs, rank, n_groups, n_tok); break;
    }
    return cuda_ok(cudaGetLastError(), "q4k grouped tile launch");
}
