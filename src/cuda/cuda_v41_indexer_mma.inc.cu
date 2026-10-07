/* cuda_v41_indexer_mma.inc.cu — ds4_cuda.cu 分片: V4.1 indexer 打分核的张量核版(2026-09-30, 1M×3 路战役)。
 *
 * 为什么: 打分核是 1M 上下文的唯一大头 —— 每路每步要对源层全部组打分(L2/8/14 各 ctx/2 组, L20 ctx 组; C2 后 L24~36 只 16384),
 * 106k 实测 4.6 ns/(行,组)(CUDA 核, 09-23 已到发射上限), 1M 一路一步 +32 ms、三路 +97 ms。同一份算术上张量核, 每 (行,组) 的
 * 4096 次乘加(32 头 × 128 维)从 ~600 条 warp 指令降到 ~60 条。
 *
 * ★数值★ 索引 q 与 k 都是"fp4(e2m1) 尾数 × 2 的幂块缩放"(q: act_quant mode 1 = e2m1 + ue8m0, 32 维一块; k: cuda_kv_pack 的 MXFP4,
 * 32 维一块 E8M0)。把两边的尾数各乘 2 变成 |m| ≤ 12 的整数走 s8 mma(m16n8k32, 一条指令正好一块 32 维), int32 累加**精确**
 * (|Σ| ≤ 32 × 144), 再乘块缩放 2^(eq + ek − 2)(2 的幂, 精确) ⇒ 每块的点积 T_b 与"逐元素 f32 乘加"完全相同;
 * 四块相加 (T0+T1)+(T2+T3) 是唯一可能舍入的地方。★与现役 CUDA 核的差别只在这一步的加法顺序★: 现役核是 lane 内 4 维 → xor 16/8/4/2/1
 * 的蝶形树(先跨块后块内), 两边只在"块间缩放差 > 2^9 且刚好落在 bf16 舍入边界"时出不同的 bf16 点积 —— 逐字节门(12k/106k 温 0)说了算。
 * 之后 relu / ×w / bf16 舍点 / 各头**按头号顺序**的 f32 累加, 一个字不变(头序通过 shared 转置后由每组一个 lane 串行加)。
 *
 * 形状: 头数按元数据走模板(V4.1 Flash 是 32 × 128; 64 头也编一份), 维度只做 128; 别的形状拒绝(不是 fallback: 元数据不同就是另一个模型)。 */

/* 每路一份的 q 整数尾数暂存: [rows][NH][128] s8 + 指数 [rows][NH][4] s8。graph 路捕获前由 ds4_gpu_v41_indexer_scratch_prepare 长够。 */
static v41_scratch g_v41_idxq[DS4_GPU_MAX_LANES];
static inline uint64_t v41_idxq_row_bytes(uint32_t n_head) { return (uint64_t)n_head * 128u + (uint64_t)n_head * 4u; }
/* 开关 g_v41_idx_mma 与 setter 住 cuda_v41_indexer.inc.cu(分发入口那边) */
int ds4_gpu_v41_indexer_scratch_prepare(uint32_t n_tok, uint32_t n_head) {
    if (!g_v41_idx_mma || n_tok == 0u) return 1;
    return v41_grow(&g_v41_idxq[g_cur_lane], (uint64_t)n_tok * v41_idxq_row_bytes(n_head), "v41 indexer q s8") ? 1 : 0;
}

/* q(f32, 值 = e2m1 × 2^s) → 每 (头, 32 维块) 一个单位指数 e 与整数尾数 m2 = q × 2^(1−e)(|m2| ≤ 12): 一 warp 一个头, lane 管 4 维(= mma A 片的 k 排布)。
 * 块内最大值 amax = mmax × 2^s(mmax ∈ {0.5..6}), frexp 给 amax = f × 2^ex ⇒ e = ex − 3 ⇒ m2 ∈ {2m, 4m, 8m, 16m} 里保证 ≤ 12 的那档, 全是整数。 */
__global__ static void v41_idxq_prep_kernel(int8_t *qi, int8_t *qe, const float *q, uint32_t n_rows, uint32_t n_head) {
    v41_pdl_wait();
    const uint32_t lane = threadIdx.x & 31u, wid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);   /* wid = 行 × 头数 + 头 */
    if (wid >= n_rows * n_head) return;
    const float *qh = q + (uint64_t)wid * 128u + lane * 4u;
    float v[4];
    #pragma unroll
    for (uint32_t e = 0; e < 4u; e++) v[e] = qh[e];
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 1; o < 8; o <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));   /* 8 个 lane = 一块 32 维 */
    int ex = 0; (void)frexpf(a, &ex);
    const int e = a > 0.f ? ex - 3 : 0;
    uint32_t packed = 0;
    #pragma unroll
    for (uint32_t kk = 0; kk < 4u; kk++) {
        const int m2 = __float2int_rn(ldexpf(v[kk], 1 - e));   /* 精确整数 */
        packed |= ((uint32_t)(uint8_t)(int8_t)m2) << (8u * kk);
    }
    *(uint32_t *)(qi + (uint64_t)wid * 128u + lane * 4u) = packed;
    if ((lane & 7u) == 0u) qe[(uint64_t)wid * 4u + (lane >> 3)] = (int8_t)e;
}

/* 紧凑项 c → 真组号 g 与活性(与 v41_indexer_score_kernel 同一套规则): 列表外 / g ≥ ng / g ≥ vis 都是死组 */
__device__ __forceinline__ static bool v41_cand_live(const int32_t *cl, uint32_t nc, uint32_t c, uint32_t ns, uint32_t bs, uint32_t ng, uint32_t vis, uint32_t *g) {
    uint32_t gg = c;
    if (cl) { const uint32_t b = c / bs; gg = b < nc ? (uint32_t)cl[1u + b] * bs + c % bs : ng; }
    *g = gg;
    return c < ns && gg < ng && gg < vis;
}
__device__ __forceinline__ static void v41_mma_s8(int32_t *d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    const int32_t z = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}
/* 2^n 的 f32 位型(块缩放 e 来自正规的 amax, 不会出 f32 正规范围) */
__device__ __forceinline__ static float v41_pow2i(int n) { uint32_t u = (uint32_t)(n + 127) << 23; float f; memcpy(&f, &u, 4); return f; }

#define V41_IMMA_QS_STRIDE 144u   /* shared 里 q 尾数的头行距(128 + 16 补位): A 片读 8 行 × 4 word 正好落 32 个 bank */
/* NH = 头数(模板, 16 的倍数): m-tile 数 MT = NH/16; 一 block 8 warp, 每 warp 一次 8 组 */
template <uint32_t NH>
__global__ static void __launch_bounds__(256) v41_indexer_score_mma_kernel(float *score, const int8_t *qi, const int8_t *qe, const uint8_t *k,
                                                                          const float *w, const int32_t *cand, uint32_t pos0, uint32_t ng,
                                                                          uint32_t ratio, const int32_t *posd, uint32_t cand_bs, uint32_t cand_cap,
                                                                          uint32_t tiles) {
    constexpr uint32_t MT = NH / 16u, XS = NH + 1u;   /* XS: shared 转置行距(头数 + 1 补位: 写 [组][头] 时 4 个组不撞同一 bank) */
    v41_pdl_wait();
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    const int32_t *cl = cand ? cand + (uint64_t)i * (1u + cand_cap) : NULL;
    const uint32_t nc = cl ? (uint32_t)cl[0] : 0u;
    const uint32_t ns = cl ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    __shared__ uint16_t lut[256];                             /* 字节(两 nibble) → 两个 s8 尾数×2(低 nibble 在低字节) */
    __shared__ __align__(16) int8_t qs[NH * V41_IMMA_QS_STRIDE];   /* 本行 NH 头的 q 整数尾数 */
    __shared__ float xs[8][8][XS];                            /* [warp][组][头] relu(点积)·w 的 bf16 值, 转置给头序串行加 */
    {
        const uint32_t v = threadIdx.x, lo = v & 15u, hi = v >> 4;
        const int ml = ((lo >> 1) & 3u) ? ((2 + (int)(lo & 1u)) << (((lo >> 1) & 3u) - 1u)) : (int)(lo & 1u);
        const int mh = ((hi >> 1) & 3u) ? ((2 + (int)(hi & 1u)) << (((hi >> 1) & 3u) - 1u)) : (int)(hi & 1u);
        const int sl = (lo & 8u) ? -ml : ml, sh = (hi & 8u) ? -mh : mh;
        lut[v] = (uint16_t)((uint8_t)(int8_t)sl | ((uint16_t)(uint8_t)(int8_t)sh << 8));
        const uint32_t *src = (const uint32_t *)(qi + (uint64_t)i * NH * 128u);
        #pragma unroll
        for (uint32_t t = 0; t < NH / 8u; t++) {   /* NH × 128 B = NH·32 word, 每线程 NH/8 个: word x → 头 x/32, 维 (x%32)·4 */
            const uint32_t x = threadIdx.x + t * 256u;
            *(uint32_t *)(qs + (x >> 5) * V41_IMMA_QS_STRIDE + (x & 31u) * 4u) = src[x];
        }
    }
    __syncthreads();
    const uint32_t g4 = lane >> 2, t4 = lane & 3u;
    /* 本 lane 在 C 片里管的头: mt·16 + g4 (+8) —— 它们的块指数(4 块打包成 u32)与 w */
    uint32_t eqp[MT][2]; float wh[MT][2];
    #pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (uint32_t hh = 0; hh < 2u; hh++) {
            const uint32_t h = mt * 16u + g4 + hh * 8u;
            wh[mt][hh] = w[(uint64_t)i * NH + h];
            eqp[mt][hh] = *(const uint32_t *)(qe + ((uint64_t)i * NH + h) * 4u);
        }
    }
    const uint32_t c_base = (blockIdx.y * 8u + warp) * tiles * 8u;
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t c0 = c_base + t * 8u;
        if (c0 >= ns) break;
        /* B 片: 本 lane 管组 n = lane/4 的 k = b·32 + (lane%4)·4 (+16) 四维 = 两字节 nibble */
        uint32_t gB; const bool liveB = v41_cand_live(cl, nc, c0 + g4, ns, cand_bs, ng, vis, &gB);
        const uint8_t *kr = k + (uint64_t)(liveB ? gB : 0u) * DS4_V41_IDXK_BYTES;
        /* C 片的两组 (lane%4)·2 + {0,1}: 它们的 E8M0 块缩放(行尾 4 字节, 行基址 8 对齐 ⇒ 一次 u32) */
        uint32_t gC[2]; bool liveC[2]; uint32_t ek[2];
        #pragma unroll
        for (uint32_t j = 0; j < 2u; j++) {
            liveC[j] = v41_cand_live(cl, nc, c0 + t4 * 2u + j, ns, cand_bs, ng, vis, &gC[j]);
            ek[j] = *(const uint32_t *)(k + (uint64_t)(liveC[j] ? gC[j] : 0u) * DS4_V41_IDXK_BYTES + DS4_V41_IDXK_NIB);
        }
        float s01[MT][4], S[MT][4];
        #pragma unroll
        for (uint32_t b = 0; b < 4u; b++) {
            const uint32_t x0 = *(const uint16_t *)(kr + b * 16u + t4 * 2u), x1 = *(const uint16_t *)(kr + b * 16u + 8u + t4 * 2u);
            const uint32_t b0 = liveB ? ((uint32_t)lut[x0 & 0xFFu] | ((uint32_t)lut[x0 >> 8] << 16)) : 0u;
            const uint32_t b1 = liveB ? ((uint32_t)lut[x1 & 0xFFu] | ((uint32_t)lut[x1 >> 8] << 16)) : 0u;
            const float fk0 = ds4_e8m0_to_f32((uint8_t)(ek[0] >> (8u * b))), fk1 = ds4_e8m0_to_f32((uint8_t)(ek[1] >> (8u * b)));
            #pragma unroll
            for (uint32_t mt = 0; mt < MT; mt++) {
                const int8_t *q0 = qs + (mt * 16u + g4) * V41_IMMA_QS_STRIDE + b * 32u + t4 * 4u, *q1 = q0 + 8u * V41_IMMA_QS_STRIDE;
                int32_t d[4];
                v41_mma_s8(d, *(const uint32_t *)q0, *(const uint32_t *)q1, *(const uint32_t *)(q0 + 16u), *(const uint32_t *)(q1 + 16u), b0, b1);
                /* c0,c1: 头 g4 组 t4·2+{0,1}; c2,c3: 头 g4+8。缩放 = 2^(eq−2) × 2^ek(两次乘都精确) */
                const float fq0 = v41_pow2i((int)(int8_t)(eqp[mt][0] >> (8u * b)) - 2), fq1 = v41_pow2i((int)(int8_t)(eqp[mt][1] >> (8u * b)) - 2);
                float T[4];
                T[0] = ((float)d[0] * fq0) * fk0; T[1] = ((float)d[1] * fq0) * fk1;
                T[2] = ((float)d[2] * fq1) * fk0; T[3] = ((float)d[3] * fq1) * fk1;
                #pragma unroll
                for (uint32_t c = 0; c < 4u; c++) {   /* S = (T0 + T1) + (T2 + T3) */
                    if (b == 0u) s01[mt][c] = T[c];
                    else if (b == 1u) s01[mt][c] = s01[mt][c] + T[c];
                    else if (b == 2u) S[mt][c] = T[c];
                    else S[mt][c] = s01[mt][c] + (S[mt][c] + T[c]);
                }
            }
        }
        /* 尾段(官方逐式): bf16(点积) → relu → bf16(× w_h) → 转置到 shared, 每组一个 lane 按头序 f32 累加 → bf16 */
        #pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++) {
            #pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                const uint32_t hh = c >> 1, h = mt * 16u + g4 + hh * 8u, gl = t4 * 2u + (c & 1u);
                const float dd = fmaxf(v41_bf16r(S[mt][c]), 0.0f);
                xs[warp][gl][h] = v41_bf16r(dd * wh[mt][hh]);
            }
        }
        __syncwarp();
        if (lane < 8u) {
            uint32_t gW; const bool liveW = v41_cand_live(cl, nc, c0 + lane, ns, cand_bs, ng, vis, &gW);
            float acc = 0.f;
            #pragma unroll 8
            for (uint32_t h = 0; h < NH; h++) acc += xs[warp][lane][h];
            if (c0 + lane < ns) score[(uint64_t)i * ns + c0 + lane] = liveW ? v41_bf16r(acc) : -INFINITY;
        }
        __syncwarp();
    }
}

/* 打分核的张量核入口(与 ds4_gpu_v41_indexer_score_tensor 同一份契约, 由它按 --idx-mma 分发)。
 * grid: (行, 组块); 每 warp 连做 tiles 个 8 组的片(q 尾数每 block 装一次 shared 多用), tiles 按"至少铺满 2 波 SM"从 ns 与 SM 数算, 不写死。 */
static int v41_indexer_score_mma_launch(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights,
                                        const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    if ((n_head != 32u && n_head != 64u) || dk != 128u) {
        fprintf(stderr, "ds4: [v41] indexer 张量核打分只编了 32/64 头 × 128 维(元数据 %u × %u)\n", n_head, dk); return 0; }
    int8_t *qi = (int8_t *)v41_grow(&g_v41_idxq[g_cur_lane], (uint64_t)n_tok * v41_idxq_row_bytes(n_head), "v41 indexer q s8");
    if (!qi) return 0;
    int8_t *qe = qi + (uint64_t)n_tok * n_head * 128u;
    v41_idxq_prep_kernel<<<(unsigned)((n_tok * n_head + 7u) / 8u), 256, 0, g_cur_stream>>>(qi, qe, (const float *)q->ptr, n_tok, n_head);
    if (!cuda_ok(cudaGetLastError(), "v41 indexer q prep")) return 0;
    static int s_sm = 0;
    if (!s_sm && cudaDeviceGetAttribute(&s_sm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return cuda_ok(cudaGetLastError(), "v41 indexer mma");
    const uint32_t ns = cand_list ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    const uint32_t warps = (ns + 7u) / 8u;
    uint32_t tiles = (warps + 16u * (uint32_t)s_sm - 1u) / (16u * (uint32_t)s_sm);   /* 每 SM 2 个 block × 8 warp 为一波 */
    if (tiles < 1u) tiles = 1u;
    if (tiles > 32u) tiles = 32u;
    const uint32_t gblocks = (warps + 8u * tiles - 1u) / (8u * tiles);
    static int s_once = 0;
    if (!s_once) { s_once = 1; fprintf(stderr, "ds4: [v41] indexer 打分走张量核: 首发 n_tok %u ns %u 头 %u × %u 维, tiles %u, grid (%u, %u), SM %d\n", n_tok, ns, n_head, dk, tiles, n_tok, gblocks, s_sm); }
    if (gblocks > 65535u) { fprintf(stderr, "ds4: [v41] indexer 张量核 grid.y %u 超 CUDA 硬顶(ns %u)\n", gblocks, ns); return 0; }   /* 1M 上下文 tiles=32 时 512 个 block; 真撞到就是形状变了 */
    const dim3 grid(n_tok, gblocks);
    float *sp = (float *)score->ptr; const uint8_t *kp = (const uint8_t *)k->ptr; const float *wp = (const float *)weights->ptr;
    const int32_t *clp = cand_list ? (const int32_t *)cand_list->ptr : NULL, *pp = posd ? (const int32_t *)posd->ptr : NULL;
    if (n_head == 32u) v41_indexer_score_mma_kernel<32u><<<grid, 256, 0, g_cur_stream>>>(sp, qi, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    else               v41_indexer_score_mma_kernel<64u><<<grid, 256, 0, g_cur_stream>>>(sp, qi, qe, kp, wp, clp, pos0, ng, ratio, pp, cand_bs, cand_cap, tiles);
    return cuda_ok(cudaGetLastError(), "v41 indexer score mma");
}
