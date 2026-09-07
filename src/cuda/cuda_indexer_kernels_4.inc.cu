/* cuda_indexer_kernels_4.inc.cu — indexer 打分的 tensor-core 版(2026-09-06, 1M 上下文战役)。
 *
 * ---- 解码核 idx1 ----
 * 为什么: 老核 indexer_score_one_direct 每个压缩 key 一个 block, 块内 16 轮"4 头点积 + 两次
 * __syncthreads", 全靠延迟堆: 1M 上下文每层 262144 个 block, 21 个 indexer 层 ≈ 72 ms/token, 就是
 * 1M decode 8.4 t/s(118 ms/token, 短上下文底 33 ms)的主因。
 * 这里 block = 128 key: q(64 头×128) 转 f16 进 shared 一次, K 块 f32→f16 进 shared, 每 warp 16 key ×
 * 64 头用 wmma(M=key, N=头, K=128 维), 尾声 ReLU × w[h] × scale 按头求和。字节只剩 K 本身(f32,
 * 1M 时 134 MB/层 ⇒ ~0.6 ms/层)。与 prefill 的 wmma128 路同一 f16 口径 —— 老解码核是 f32 点积, 两路
 * 本就不逐位同; 近平局的 top-k 会翻, 判五指标同带(kernel_parity decode + prefill 两模式)。
 * 改了会怎样: 把 block 缩到 16 key 会让 q 的 shared 装载(16 KB)摊不薄; 把 K 留 f32 走 CUDA 核就回到
 * 每 key 64 次 warp 归约的老账。
 * GB10(sm_121) 每 block 动态 shared 上限是 99 KB(消费级 Blackwell 口径, 不是 H100 的 227 KB): 两个核都按
 * 这个上限设计, 发射前 cudaFuncSetAttribute 抬一次。 */
#define DS4_IDX1_KEYS 128u
/* q f16 16 KB + K 块 f16 32 KB + 8 warp 的 C 块 8 KB + w 256 B = 57,600 B */
#define DS4_IDX1_SMEM ((size_t)(64u * 128u * 2u + DS4_IDX1_KEYS * 128u * 2u + 8u * 256u * 4u + 64u * 4u))

__global__ static void indexer_score_one_wmma_kernel(
        float *scores,
        const float *q,             /* [64 头][128] */
        const float *weights,       /* [64] */
        const __half *k16,          /* [n_comp][128] f16 缓存行(core_gpu_graph.h: indexer 缓存恒 f16) */
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    extern __shared__ __align__(32) unsigned char idx1_smem[];
    __half *q_sh = (__half *)idx1_smem;                    /* [head][dim] */
    __half *k_sh = q_sh + 64u * 128u;                      /* [key][dim] */
    float *c_sh = (float *)(k_sh + DS4_IDX1_KEYS * 128u);  /* 每 warp 一块 16 key × 16 头 */
    float *w_sh = c_sh + 8u * 256u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    const uint32_t key0 = blockIdx.x * DS4_IDX1_KEYS;
    if (key0 >= n_comp || tid >= 256u) return;
    const uint32_t visible = causal ? (ratio ? (pos0 + 1u) / ratio : n_comp) : n_comp;
    if (key0 >= visible) {   /* 整块不可见: 只写 -INF */
        for (uint32_t i = tid; i < DS4_IDX1_KEYS; i += 256u)
            if (key0 + i < n_comp) scores[key0 + i] = -INFINITY;
        return;
    }
    for (uint32_t i = tid; i < 64u * 128u; i += 256u) q_sh[i] = __float2half(q[i]);
    if (tid < 64u) w_sh[tid] = weights[tid] * scale;
    for (uint32_t i = tid; i < DS4_IDX1_KEYS * 16u; i += 256u) {   /* f16 缓存行: 16 B 一搬 */
        const uint32_t r = i >> 4u, ch = i & 15u, c = key0 + r;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (c < n_comp) v = *(const uint4 *)(k16 + (uint64_t)c * 128u + ch * 8u);
        *(uint4 *)(k_sh + r * 128u + ch * 8u) = v;
    }
    __syncthreads();

    /* warp w 管 key 行 [w·16, w·16+16); lane 持 c_sh 里 8 个格: i = lane + 32·s ⇒ key r = (lane>>4) + 2s,
     * 头 hh = lane & 15。四个头块(16 头一块)依次乘加进同一组寄存器。 */
    float acc[8];
#pragma unroll
    for (uint32_t s = 0; s < 8u; s++) acc[s] = 0.0f;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    float *cw = c_sh + warp * 256u;
    for (uint32_t hn = 0; hn < 4u; hn++) {
        wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, k_sh + warp * 16u * 128u + k0, 128);
            wmma::load_matrix_sync(b_frag, q_sh + hn * 16u * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(cw, c_frag, 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (uint32_t s = 0; s < 8u; s++) {
            const uint32_t i = lane + 32u * s;
            acc[s] += fmaxf(cw[i], 0.0f) * w_sh[hn * 16u + (i & 15u)];
        }
        __syncwarp();
    }
    /* 每 key 的 16 个头格分在同一半 warp(lane>>4 = r&1)的 16 条 lane 上: 半 warp 内 xor 归约 */
#pragma unroll
    for (uint32_t s = 0; s < 8u; s++) {
        float v = acc[s];
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if ((lane & 15u) == 0u) {
            const uint32_t c = key0 + warp * 16u + 2u * s + (lane >> 4u);
            if (c < n_comp) scores[c] = (c < visible) ? v : -INFINITY;
        }
    }
#else
    (void)scores; (void)q; (void)weights; (void)k16; (void)n_comp; (void)pos0; (void)ratio; (void)scale; (void)causal;
#endif
}

/* ---- prefill 打分核 idxp v3: mma.sync 寄存器尾声、q 驻留、K(f16 影子)流式 ----
 * 老核 indexer_scores_wmma128 与 v2(wmma + shared 往返)都只跑到 ~12 TFLOPS(128k 剖面 97 / 89 ms/层块): ReLU 在头维,
 * 不能在累加器里跨头合并, wmma 每头每 16×16 块都要 store 到 shared 再读回做 ReLU×w, 加上 A/B 片段每头每子块从
 * shared 重装 ⇒ 每 8 条 mma 搬 10 KB shared, shared 带宽顶死。
 * 这里用 mma.sync.m16n8k16(PTX ISA 有文档的片段布局): 每线程持 C 的 4 个 f32 于已知 (row,col), ReLU×w[t][h] 直接在
 * 寄存器里按头累加, 零 shared 往返; 每头的 A(q)片段装一次驻寄存器, 同一 warp 扫 2 个 8-key 子块(B 换 A 不换)。
 * block = 16 token × 一段 DS4_IDXP_RANGE key: 16 头的 q f16 驻 shared(64 KB), K 128 key 块流式(32 KB), w 1 KB
 * ⇒ 97 KB(< 99 KB 上限, 每 SM 1 块); 4 遍(每遍 16 头)累加进 scores(本 block 独占区域: 第一遍写, 后三遍加, 最后
 * 一遍按 token 因果掩码定稿, 粒度与老核逐字同)。shared 布局做 16 B 块 XOR 搅拌(chunk ^ (row&7)), A/B 片段装载零 bank
 * 冲突且不占额外 shared。数值: f16 输入 f32 累加, 与 v2/老核同口径(头内 mma 归约序不同, 容差级)。 */
#define DS4_IDXP_KEYS 128u
#define DS4_IDXP_HEADS 16u
#define DS4_IDXP_RANGE 2048u
#define DS4_IDXP_SMEM ((size_t)(DS4_IDXP_HEADS * 16u * 128u * 2u + DS4_IDXP_KEYS * 128u * 2u + 16u * DS4_IDXP_HEADS * 4u))

__device__ __forceinline__ static uint32_t idxp_swz(uint32_t row, uint32_t col) {   /* 行内 16 B 块 XOR 搅拌 */
    return row * 128u + ((((col >> 3u) ^ (row & 7u)) << 3u) | (col & 7u));
}
__device__ __forceinline__ static void idxp_mma16816(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ static void __launch_bounds__(256, 1) indexer_scores_prefill_wmma_kernel(
        float *scores,
        const float *q,             /* [n_tokens][64 头][128] */
        const float *weights,       /* [n_tokens][64] */
        const __half *k16,          /* [n_comp][128] f16 影子 */
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) unsigned char idxp_smem[];
    __half *q_sh = (__half *)idxp_smem;                          /* [16 head][16 tok][128] 搅拌 */
    __half *k_sh = q_sh + DS4_IDXP_HEADS * 16u * 128u;           /* [128 key][128] 搅拌 */
    float *w_sh = (float *)(k_sh + DS4_IDXP_KEYS * 128u);        /* [16 tok][16 head] */
    const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    const uint32_t g = lane >> 2u, t4 = lane & 3u;               /* mma 片段布局: 行 g/g+8, 列 t4*2 */
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t range0 = blockIdx.x * DS4_IDXP_RANGE;
    if (tile_t >= n_tokens || range0 >= n_comp || tid >= 256u) return;
    const uint32_t range_end = min(range0 + DS4_IDXP_RANGE, n_comp);
    const uint32_t last_token = min(tile_t + 16u, n_tokens);
    const uint32_t max_visible = causal ? min((pos0 + last_token) / ratio, n_comp) : n_comp;
    const uint32_t tok_lo = tile_t + g, tok_hi = tile_t + g + 8u;
    const uint32_t vis_lo = causal ? (pos0 + tok_lo + 1u) / ratio : n_comp;
    const uint32_t vis_hi = causal ? (pos0 + tok_hi + 1u) / ratio : n_comp;

    for (uint32_t pass = 0; pass < 64u / DS4_IDXP_HEADS; pass++) {
        const uint32_t h0 = pass * DS4_IDXP_HEADS;
        for (uint32_t i = tid; i < DS4_IDXP_HEADS * 16u * 128u; i += 256u) {
            const uint32_t h = i >> 11u, t = (i >> 7u) & 15u, d = i & 127u;
            const uint32_t token = tile_t + t;
            q_sh[h * 16u * 128u + idxp_swz(t, d)] =
                __float2half(token < n_tokens ? q[((uint64_t)token * 64u + h0 + h) * 128u + d] : 0.0f);
        }
        for (uint32_t i = tid; i < 16u * DS4_IDXP_HEADS; i += 256u) {
            const uint32_t t = i / DS4_IDXP_HEADS, h = i % DS4_IDXP_HEADS, token = tile_t + t;
            w_sh[i] = token < n_tokens ? weights[(uint64_t)token * 64u + h0 + h] : 0.0f;
        }
        __syncthreads();
        for (uint32_t c0 = range0; c0 < range_end; c0 += DS4_IDXP_KEYS) {
            if (c0 >= max_visible) break;
            for (uint32_t i = tid; i < DS4_IDXP_KEYS * 16u; i += 256u) {   /* 16 B 块搬入 + 搅拌 */
                const uint32_t r = i >> 4u, ch = i & 15u, c = c0 + r;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (c < n_comp) v = *(const uint4 *)(k16 + (uint64_t)c * 128u + ch * 8u);
                *(uint4 *)(k_sh + r * 128u + ((ch ^ (r & 7u)) << 3u)) = v;
            }
            __syncthreads();
            float acc[2][4];
#pragma unroll
            for (uint32_t nt = 0; nt < 2u; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
            for (uint32_t h = 0; h < DS4_IDXP_HEADS; h++) {
                float d[2][4];
#pragma unroll
                for (uint32_t nt = 0; nt < 2u; nt++) { d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.0f; }
                const __half *qh = q_sh + h * 16u * 128u;
#pragma unroll
                for (uint32_t ks = 0; ks < 8u; ks++) {
                    const uint32_t col = ks * 16u + t4 * 2u;
                    uint32_t a[4];
                    a[0] = *(const uint32_t *)(qh + idxp_swz(g, col));
                    a[1] = *(const uint32_t *)(qh + idxp_swz(g + 8u, col));
                    a[2] = *(const uint32_t *)(qh + idxp_swz(g, col + 8u));
                    a[3] = *(const uint32_t *)(qh + idxp_swz(g + 8u, col + 8u));
#pragma unroll
                    for (uint32_t nt = 0; nt < 2u; nt++) {
                        const uint32_t key = (warp * 2u + nt) * 8u + g;
                        uint32_t b[2];
                        b[0] = *(const uint32_t *)(k_sh + idxp_swz(key, col));
                        b[1] = *(const uint32_t *)(k_sh + idxp_swz(key, col + 8u));
                        idxp_mma16816(d[nt], a, b);
                    }
                }
                const float wlo = w_sh[g * DS4_IDXP_HEADS + h], whi = w_sh[(g + 8u) * DS4_IDXP_HEADS + h];
#pragma unroll
                for (uint32_t nt = 0; nt < 2u; nt++) {
                    acc[nt][0] += fmaxf(d[nt][0], 0.0f) * wlo;
                    acc[nt][1] += fmaxf(d[nt][1], 0.0f) * wlo;
                    acc[nt][2] += fmaxf(d[nt][2], 0.0f) * whi;
                    acc[nt][3] += fmaxf(d[nt][3], 0.0f) * whi;
                }
            }
            /* 落盘: 行 g/g+8, 列 (warp*2+nt)*8 + t4*2, +1(第一遍写, 后面加, 最后一遍掩码定稿) */
            const int last = (pass + 1u == 64u / DS4_IDXP_HEADS);
#pragma unroll
            for (uint32_t nt = 0; nt < 2u; nt++) {
                const uint32_t comp = c0 + (warp * 2u + nt) * 8u + t4 * 2u;
#pragma unroll
                for (uint32_t j = 0; j < 4u; j++) {
                    const uint32_t token = (j < 2u) ? tok_lo : tok_hi;
                    const uint32_t cc = comp + (j & 1u);
                    if (token < n_tokens && cc < n_comp) {
                        float *dst = scores + (uint64_t)token * n_comp + cc;
                        const float v = acc[nt][j] * scale;
                        if (pass == 0u) *dst = v;
                        else if (!last) *dst += v;
                        else *dst = (cc >= ((j < 2u) ? vis_lo : vis_hi)) ? -INFINITY : (*dst + v);
                    }
                }
            }
            __syncthreads();   /* 下一块覆盖 k_sh 前全 warp 用完 */
        }
        __syncthreads();       /* 下一遍覆盖 q_sh 前 */
    }
    /* 整块不可见的 key 块: -INF(与老核同粒度) */
    {
        uint32_t c_start = range0;
        while (c_start < max_visible && c_start < range_end) c_start += DS4_IDXP_KEYS;
        const uint32_t span = range_end > c_start ? range_end - c_start : 0u;
        for (uint32_t i = tid; i < 16u * span; i += 256u) {
            const uint32_t t = i / span, comp = c_start + i % span;
            const uint32_t token = tile_t + t;
            if (token < n_tokens) scores[(uint64_t)token * n_comp + comp] = -INFINITY;
        }
    }
#else
    (void)scores; (void)q; (void)weights; (void)k16; (void)n_comp; (void)n_tokens; (void)pos0; (void)ratio; (void)scale; (void)causal;
#endif
}

/* ---- 小批(投机 verify 2..4 query)打分核 idxt(09-07, 1M 投机战役): token 放 N=8, key 放 M, mma.sync 寄存器尾声 ----
 * 为什么: 4 个 query 此前走 prefill 的 16-token tile 核(算满 16 个 token, 1M 处 27 ms/轮), 而且它与解码 idx1 的头内归约序不同
 * ⇒ 稀疏区(n_comp > 1024)投机 verify 的 top-k 与纯解码不同集, 温 0 不再全同(6K 提示第 237 字节分叉实撞); 逐 query 重放 idx1
 * 的 wmma(shared 往返)又是 4×idx1 = 43 ms。这里 A = K 块(16 key × 16 维, 与 idx1 同为 M=key), B = q(16 维 × 8 token 位, 只装
 * 4 个 token 的行, 位 4..7 复用 0..3 结果丢弃), 每头一次 m16n8k16 得到 (key, token) 分数; ReLU×w 与跨头求和逐字复刻 idx1:
 * 每个 (key, token) 先按头块序(hn = 0..3)累进 16 个头槽 p[hl], 再按 idx1 半 warp xor 树(8,4,2,1)的括号顺序合并 ⇒ 与 idx1
 * 逐位同(wmma m16n16k16 在 sm_80+ 就是两条 HMMA.16816, 元素级同硬件累加序; 链 54 英文 7K spec == plain 逐字节验过)。
 * v2(链 54 后): 首版每 128 key 重装一遍全部 q(4 遍 × 32 KB, 比 K 本身多 4× 流量 + 转 half 的 ALU) ⇒ 1M 反慢; 现在 64 头 × 4 token
 * 的 q(64 KB)每块只装一次, 块内流式扫 DS4_IDXT_RANGE 个 key(K 子块 32 KB), 与 prefill v3 同一摊薄法。
 * shared: q 64 KB + K 子块 32 KB + w 1 KB = 99,328 B(< 99 KB 上限 101,376), 每 SM 一块。
 * 改了会怎样: 把 token 放回 M=16 就回到 4× 浪费; 跨头改成顺序累加会与 idx1 差最后几个 ulp, 近平局 top-k 翻转 ⇒ 投机分叉;
 * 5..8 个 query 装不下 q(128 KB), 发射方回落 prefill tile 核(不同轨, 投机 k=4 用不到)。 */
#define DS4_IDXT_KEYS 128u
#define DS4_IDXT_QTOK 4u
#define DS4_IDXT_RANGE 2048u
#define DS4_IDXT_SMEM ((size_t)(64u * DS4_IDXT_QTOK * 128u * 2u + DS4_IDXT_KEYS * 128u * 2u + DS4_IDXT_QTOK * 64u * 4u))

/* idx1 的半 warp xor 归约树(8,4,2,1)在 lane 0 的括号顺序, 逐字展开 */
__device__ __forceinline__ static float idx1_tree16(const float *p) {
    const float a0 = p[0] + p[8], a4 = p[4] + p[12], a2 = p[2] + p[10], a6 = p[6] + p[14];
    const float a1 = p[1] + p[9], a5 = p[5] + p[13], a3 = p[3] + p[11], a7 = p[7] + p[15];
    const float b0 = a0 + a4, b2 = a2 + a6, b1 = a1 + a5, b3 = a3 + a7;
    const float c0 = b0 + b2, c1 = b1 + b3;
    return c0 + c1;
}

__global__ static void __launch_bounds__(256, 1) indexer_score_tokn_mma_kernel(
        float *scores,              /* [n_tokens][n_comp] */
        const float *q,             /* [n_tokens][64 头][128] */
        const float *weights,       /* [n_tokens][64] */
        const __half *k16,          /* [n_comp][128] f16 缓存行 */
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) unsigned char idxt_smem[];
    __half *q_sh = (__half *)idxt_smem;                                    /* [64 head][4 tok][128] 搅拌(行 = head*4+tok) */
    __half *k_sh = q_sh + 64u * DS4_IDXT_QTOK * 128u;                      /* [128 key][128] 搅拌 */
    float *w_sh = (float *)(k_sh + DS4_IDXT_KEYS * 128u);                  /* [4 tok][64 head] ×scale */
    const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    const uint32_t g = lane >> 2u, t4 = lane & 3u;
    const uint32_t range0 = blockIdx.x * DS4_IDXT_RANGE;
    if (range0 >= n_comp || tid >= 256u || n_tokens > DS4_IDXT_QTOK) return;
    const uint32_t range_end = min(range0 + DS4_IDXT_RANGE, n_comp);
    /* 可见范围随 query 位置单调不减: 对最后一个 query 都不可见的子块只写 -INF(与 idx1 整块早退同粒度) */
    const uint32_t vis_last = causal ? (ratio ? (pos0 + n_tokens) / ratio : n_comp) : n_comp;
    for (uint32_t i = tid; i < 64u * DS4_IDXT_QTOK * 128u; i += 256u) {   /* q 一次装满: 64 头 × 4 token */
        const uint32_t h = i >> 9u, t = (i >> 7u) & 3u, d = i & 127u;
        q_sh[idxp_swz(h * DS4_IDXT_QTOK + t, d)] =
            __float2half(t < n_tokens ? q[((uint64_t)t * 64u + h) * 128u + d] : 0.0f);
    }
    for (uint32_t i = tid; i < DS4_IDXT_QTOK * 64u; i += 256u) {
        const uint32_t t = i / 64u, h = i % 64u;
        w_sh[i] = t < n_tokens ? weights[(uint64_t)t * 64u + h] * scale : 0.0f;
    }
    /* 本线程持 C 的 4 格: (key g, tok t4*2), (key g, tok t4*2+1), (key g+8, tok t4*2), (key g+8, tok t4*2+1); token 位 ≥ 4 丢弃 */
    const uint32_t tk0 = t4 * 2u, tk1 = tk0 + 1u;
    const uint32_t qrow = g & (DS4_IDXT_QTOK - 1u);   /* B 片段的 n = g ⇒ token 位 g, 行只有 4 个 token */
    for (uint32_t c0 = range0; c0 < range_end; c0 += DS4_IDXT_KEYS) {
        __syncthreads();   /* 上一子块的 k_sh 用完(首轮兼作 q/w 就位屏障) */
        if (c0 >= vis_last) {   /* 子块整块不可见 */
            for (uint32_t i = tid; i < n_tokens * DS4_IDXT_KEYS; i += 256u) {
                const uint32_t t = i / DS4_IDXT_KEYS, c = c0 + i % DS4_IDXT_KEYS;
                if (c < n_comp) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            }
            continue;
        }
        for (uint32_t i = tid; i < DS4_IDXT_KEYS * 16u; i += 256u) {   /* K 子块 16 B 一搬 + 搅拌 */
            const uint32_t r = i >> 4u, ch = i & 15u, c = c0 + r;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (c < n_comp) v = *(const uint4 *)(k16 + (uint64_t)c * 128u + ch * 8u);
            *(uint4 *)(k_sh + r * 128u + ((ch ^ (r & 7u)) << 3u)) = v;
        }
        __syncthreads();
        float p[16][4];
#pragma unroll
        for (uint32_t hl = 0; hl < 16u; hl++) { p[hl][0] = 0.0f; p[hl][1] = 0.0f; p[hl][2] = 0.0f; p[hl][3] = 0.0f; }
        const uint32_t kr = warp * 16u + g;
        for (uint32_t hn = 0; hn < 4u; hn++) {
#pragma unroll
            for (uint32_t hl = 0; hl < 16u; hl++) {
                const uint32_t head = hn * 16u + hl;
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const uint32_t qr = head * DS4_IDXT_QTOK + qrow;   /* q_sh 里 (head, token 位 g) 的行号 */
#pragma unroll
                for (uint32_t ks = 0; ks < 8u; ks++) {
                    const uint32_t col = ks * 16u + t4 * 2u;
                    uint32_t a[4], b[2];
                    a[0] = *(const uint32_t *)(k_sh + idxp_swz(kr, col));
                    a[1] = *(const uint32_t *)(k_sh + idxp_swz(kr + 8u, col));
                    a[2] = *(const uint32_t *)(k_sh + idxp_swz(kr, col + 8u));
                    a[3] = *(const uint32_t *)(k_sh + idxp_swz(kr + 8u, col + 8u));
                    b[0] = *(const uint32_t *)(q_sh + idxp_swz(qr, col));
                    b[1] = *(const uint32_t *)(q_sh + idxp_swz(qr, col + 8u));
                    idxp_mma16816(d, a, b);
                }
                const float w0 = w_sh[(tk0 & 3u) * 64u + head], w1 = w_sh[(tk1 & 3u) * 64u + head];
                p[hl][0] += fmaxf(d[0], 0.0f) * w0;
                p[hl][1] += fmaxf(d[1], 0.0f) * w1;
                p[hl][2] += fmaxf(d[2], 0.0f) * w0;
                p[hl][3] += fmaxf(d[3], 0.0f) * w1;
            }
        }
        /* 跨 16 个头槽按 idx1 的树合并, 写分数(因果掩码逐 token) */
        float pe[16];
        const uint32_t ck0 = c0 + warp * 16u + g, ck1 = ck0 + 8u;
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
#pragma unroll
            for (uint32_t hl = 0; hl < 16u; hl++) pe[hl] = p[hl][e];
            const float v = idx1_tree16(pe);
            const uint32_t t = (e & 1u) ? tk1 : tk0, c = (e < 2u) ? ck0 : ck1;
            if (t < n_tokens && c < n_comp) {
                const uint32_t visible = causal ? (ratio ? (pos0 + t + 1u) / ratio : n_comp) : n_comp;
                scores[(uint64_t)t * n_comp + c] = (c < visible) ? v : -INFINITY;
            }
        }
    }
#else
    (void)scores; (void)q; (void)weights; (void)k16; (void)n_comp; (void)n_tokens; (void)pos0; (void)ratio; (void)scale; (void)causal;
#endif
}
