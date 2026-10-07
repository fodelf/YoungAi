/* cuda_bwd_attn.inc.cu — ds4_cuda.cu 分片: 后训练反传的注意力与 RoPE(契约 ds4_gpu_bwd.h, 2026-10-01; 10-02 重写注意力两核)。
 *
 * 只接后训练要的形态: 一题一块、从位置 0 起(pos0 = 0), 窗口键全在本批(窗口缓冲第 window+i 行 = 第 i 个位置的 kvn),
 * 压缩行来自 kv 源层: 源层在训练段里时传 gcomp, 对压缩行的梯度累加进去, 源层在训练段之下就是常量、传 NULL;
 * 选哪些压缩行(indexer topk)是离散的, 照前向存档的 idx 用。
 * 前向算式(v41_sparse_attn_kernel): s_t = scale·<q, k_t>, o = Σ_t e^{s_t−m}·k_t / (Σ_t e^{s_t−m} + e^{sink−m})(值 = 键, sink 只进分母)。
 * 反向: p_t = e^{s_t−lse}, D = <g_o, o>, g_s_t = p_t·(<g_o, k_t> − D), g_q = scale·Σ_t g_s_t·k_t,
 *       g_k_t = Σ_{i,h} [scale·g_s·q_ih + p·g_o_ih](键与值两种身份合计; 窗口键进 gkv, 压缩行进 gcomp[组号], 跨 query/读取层原子累加)。
 * 前向在概率分子上舍的 bf16、kvn 的 fp8 量化, 反向都当直通。
 *
 * 【10-02 为什么重写】旧版三核(一 warp 一个 (query, 头) 的 g_q 核 + 窗口键核 + 压缩行核)每个都对 (query, 头, 键) 重做 512 维点积,
 * 64 个头各读各解同一份键(MQA: 键与头无关), 压缩行核每个槽把本 query 64 个头的 q/g_o 全读一遍 —— 逐核表 1.82 s/题(整题 39%)。
 * 新版两核:
 *   ① bwd_attn_qm_kernel(10-02 夜起张量核, 见核前注释): 一 block = 同一 query 的 16 个头, 键 16 个一片(与前向 flash 核同一取键/打分),
 *      两趟(先 lse, 再 g_s → g_q); 每个键的两样权重 scale·g_s、p 按 bf16 落进 wb[i][键][2·头] —— 第二核的 B 操作数, 点积不再重算。
 *   ② bwd_attn_kg_kernel: 每个 query 的键梯度是一个小 GEMM —— G_K(键 × 512) = W(键 × 2·头) · [Q_i; G_i](2·头 × 512),
 *      bf16 张量核(A = [e][维] 存储用 ldmatrix.trans 取成 [维][e] 片段, B = wb 的 [键][e]), 结果原子加进 gkv / gcomp。
 *      q/g_o 每个 query 只读一遍。精度: 权重与 q/g_o 转 bf16 乘、f32 累加, 每 k16 从零起算再 FADD(同专家转置核 vqst)。
 * 出错会怎样: wb 的行宽 2·头 必须是 64 的倍数(第二核一轮吃 64 个 e), 头数不是 32 的倍数发射端直接拦; A 片交织写入与 .trans 取址
 * 同一个式子(vqm_swz256), 写错不报错只出垃圾梯度 —— 门 = 同题新旧二进制逐层梯度范数 + 有限差分比值(kdprof gradcheck=2)。 */

#define BWD_ATT_MAXK 1024u   /* 一个 query 最多几个键(窗口 128 + topk ≤ 512 + 余量): wb 的行宽按它估, 超了发射端拦 */
#define BWD_KG_BN 128u       /* ② 一块管几个键(16 个 n8 片) */
#define BWD_KG_BM 128u       /* ② 一块管几维(8 个 m16 片) */

/* ① g_q + 两样权重(10-02 夜换张量核): 与前向 flash 核(ds4_sparse_attn_mma_kernel)同一形状 —— 一 block = 一个 query 的 16 个头、256 线程,
 * 键 16 个一片(ds4_fa_gather: 窗口键 f32 → bf16, 压缩行 fp4 解, 同前向逐位), S = Q·Kᵀ 用 mma.m16n8k16(q 的 A 片段常驻寄存器, 8 个 warp 分 512 维,
 * partial 按固定序相加)。第一遍: 在线 max/sum 得 lse —— 打分与相加次序与前向逐位同, 反传用的 p 就是前向那一份(sink 只进分母, 同前向)。
 * 第二遍: 重算 S, 同形状再算 dP = G_o·Kᵀ; g_s = p·(dP − D); 两样权重 scale·g_s / p 落 wb(第二核的 B 操作数, 看不见的键写 0);
 * g_q += (scale·g_s)·K(A = g_s 片 bf16, B = 键片 .trans 取, 同前向的 P·V)。D = <g_o, o> 用 f32。
 * 为什么: 旧核是 FP32 标量(一 warp 一头、每键 3 个 512 维点积、键读两遍 × 4 个头组), 10-02 逐核表 15.8%、长题单发 55 ms。
 * 精度: q/g_o/键/g_s 转 bf16 乘、f32 累加(与 kg 核、vqt 同口径), 旧核全 f32 ⇒ 不逐位同; 门 = packcheck 有限差分 + 同题新旧梯度范数。 */
__global__ __launch_bounds__(256) static void bwd_attn_qm_kernel(float *gq, uint16_t *wb, uint32_t nkmax, const float *go, const float *o,
                                                                const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                                const float *sink, uint32_t n_head, uint32_t window, uint32_t ng, uint32_t topk,
                                                                float scale) {
    constexpr uint32_t KT = DS4_ATTN_MMA_KT, HB = DS4_ATTN_MMA_HEADS, HD = DS4_ATTN_MMA_HD;
    extern __shared__ __align__(16) char bwd_qm_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)bwd_qm_smem;                            /* [16 键][520] */
    float *spart = (float *)(ks + (size_t)KT * DS4_ATTN_FA_LD);                   /* [8 warp][16 头][16 键] */
    float *stile = spart + 8u * HB * KT, *dtile = stile + HB * KT;                /* [16][16]: 缩放并屏蔽后的 S / dP */
    __nv_bfloat16 *gtile = (__nv_bfloat16 *)(dtile + HB * KT);                    /* [16][16]: bf16(scale·g_s), g_q 那发的 A */
    float *rmax = (float *)(gtile + HB * KT), *rsum = rmax + HB, *lse = rsum + HB, *Dh = lse + HB, *dpart = Dh + HB;   /* dpart [8 warp][16 头] */
    int *valid = (int *)(dpart + 8u * HB);
    const uint32_t i = blockIdx.x, h0 = blockIdx.y * HB, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t lo = i + 1u > window ? i + 1u - window : 0u, nwin = i - lo + 1u, nkeys = nwin + topk;
    uint32_t qa[4][4], ga[4][4];   /* q / g_o 的 A 片段: warp w 管维 [64w, 64w+64) 的 4 个 k16(布局同前向) */
    {
        const uint32_t r = lane >> 2, c = (lane & 3u) * 2u;
        float d0 = 0.f, d1 = 0.f;   /* D 的部分和: 头 r / 头 r+8 */
        #pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            const uint64_t b0 = ((uint64_t)i * n_head + h0 + r) * HD + warp * 64u + s * 16u + c, b1 = b0 + 8u * HD;
            qa[s][0] = ds4_fa_bf16x2(q[b0], q[b0 + 1]); qa[s][1] = ds4_fa_bf16x2(q[b1], q[b1 + 1]);
            qa[s][2] = ds4_fa_bf16x2(q[b0 + 8], q[b0 + 9]); qa[s][3] = ds4_fa_bf16x2(q[b1 + 8], q[b1 + 9]);
            ga[s][0] = ds4_fa_bf16x2(go[b0], go[b0 + 1]); ga[s][1] = ds4_fa_bf16x2(go[b1], go[b1 + 1]);
            ga[s][2] = ds4_fa_bf16x2(go[b0 + 8], go[b0 + 9]); ga[s][3] = ds4_fa_bf16x2(go[b1 + 8], go[b1 + 9]);
            d0 += go[b0] * o[b0] + go[b0 + 1] * o[b0 + 1] + go[b0 + 8] * o[b0 + 8] + go[b0 + 9] * o[b0 + 9];
            d1 += go[b1] * o[b1] + go[b1 + 1] * o[b1 + 1] + go[b1 + 8] * o[b1 + 8] + go[b1 + 9] * o[b1 + 9];
        }
        d0 += __shfl_xor_sync(0xffffffffu, d0, 1); d0 += __shfl_xor_sync(0xffffffffu, d0, 2);
        d1 += __shfl_xor_sync(0xffffffffu, d1, 1); d1 += __shfl_xor_sync(0xffffffffu, d1, 2);
        if ((lane & 3u) == 0u) { dpart[warp * HB + r] = d0; dpart[warp * HB + r + 8u] = d1; }
    }
    if (threadIdx.x < HB) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    __syncthreads();
    if (threadIdx.x < HB) { float v = 0.f; for (uint32_t w = 0; w < 8u; w++) v += dpart[w * HB + threadIdx.x]; Dh[threadIdx.x] = v; }
    /* 一片 16 个键的 S(或 dP)部分和: 本 warp 的 64 维 × 16 头 × 16 键 → spart, 再按 warp 固定序求和进 out(S 缩放并屏蔽) */
    auto tile_mma = [&](const uint32_t (*fa)[4], float *out, uint32_t nt, bool is_s) {
        float sp[2][4] = { {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f} };
        #pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            uint32_t b[4];
            const uint32_t mt = lane >> 3, key = (mt >> 1) * 8u + (lane & 7u), col = warp * 64u + s * 16u + (mt & 1u) * 8u;
            ds4_fa_ldsm4(b, ks + (size_t)key * DS4_ATTN_FA_LD + col);
            ds4_fa_mma(sp[0], fa[s], b[0], b[1]); ds4_fa_mma(sp[1], fa[s], b[2], b[3]);
        }
        #pragma unroll
        for (uint32_t nb = 0; nb < 2u; nb++) {
            float *sw = spart + (size_t)warp * HB * KT + (lane >> 2) * KT + nb * 8u + (lane & 3u) * 2u;
            sw[0] = sp[nb][0]; sw[1] = sp[nb][1]; sw[8u * KT] = sp[nb][2]; sw[8u * KT + 1u] = sp[nb][3];
        }
        __syncthreads();
        {
            const uint32_t e = threadIdx.x, k = e % KT;
            float v = 0.f;
            #pragma unroll
            for (uint32_t w = 0; w < 8u; w++) v += spart[(size_t)w * HB * KT + e];
            out[e] = is_s ? ((k < nt && valid[k]) ? v * scale : -1e30f) : v;
        }
        __syncthreads();
    };
    for (uint32_t base = 0; base < nkeys; base += KT) {   /* 第一遍: lse(在线 max/sum, 一线程一头, 键序串行 —— 与前向同序) */
        const uint32_t nt = (nkeys - base) < KT ? (nkeys - base) : KT;
        __syncthreads();
        ds4_fa_gather(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, 0u, window, ng, topk);
        __syncthreads();
        tile_mma(qa, stile, nt, true);
        if (threadIdx.x < HB) {
            const uint32_t h = threadIdx.x; const float *sr = stile + h * KT; const float m = rmax[h];
            float tm = m;
            for (uint32_t k = 0; k < KT; k++) tm = fmaxf(tm, sr[k]);
            float sm = rsum[h] * expf(m - tm);
            for (uint32_t k = 0; k < KT; k++) sm += expf(sr[k] - tm);
            rmax[h] = tm; rsum[h] = sm;
        }
    }
    __syncthreads();
    if (threadIdx.x < HB) lse[threadIdx.x] = rmax[threadIdx.x] + logf(rsum[threadIdx.x] + expf(sink[h0 + threadIdx.x] - rmax[threadIdx.x]));
    float gacc[8][4];   /* g_q[16 头][本 warp 64 维] = 8 个 n8 片 */
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) { gacc[j][0] = gacc[j][1] = gacc[j][2] = gacc[j][3] = 0.f; }
    for (uint32_t base = 0; base < nkeys; base += KT) {   /* 第二遍: g_s → wb, g_q */
        const uint32_t nt = (nkeys - base) < KT ? (nkeys - base) : KT;
        __syncthreads();
        ds4_fa_gather(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, 0u, window, ng, topk);
        __syncthreads();
        tile_mma(qa, stile, nt, true);
        tile_mma(ga, dtile, nt, false);
        {   /* 一线程一个 (头, 键) */
            const uint32_t e = threadIdx.x, h = e / KT, k = e % KT;
            const float s = stile[e];
            const float p = s > -1e29f ? expf(s - lse[h]) : 0.f, gs = p * (dtile[e] - Dh[h]);
            const __nv_bfloat16 w = __float2bfloat16_rn(scale * gs);
            gtile[e] = w;
            if (k < nt) {
                uint16_t *wr = wb + ((uint64_t)i * nkmax + base + k) * (2u * n_head);
                wr[h0 + h] = __bfloat16_as_ushort(w);
                wr[n_head + h0 + h] = __bfloat16_as_ushort(__float2bfloat16_rn(p));
            }
        }
        __syncthreads();
        {   /* g_q += gtile[16 头][16 键] · ks[16 键][本 warp 64 维](同前向 P·V 的取法) */
            uint32_t pa[4];
            {   const uint32_t mt = lane >> 3, row = (lane & 7u) + (mt & 1u) * 8u, col = (mt >> 1) * 8u;
                ds4_fa_ldsm4(pa, gtile + (size_t)row * KT + col); }
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j += 2u) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt & 1u) * 8u + (lane & 7u), col = warp * 64u + j * 8u + (mt >> 1) * 8u;
                ds4_fa_ldsm4t(b, ks + (size_t)key * DS4_ATTN_FA_LD + col);
                ds4_fa_mma(gacc[j], pa, b[0], b[1]); ds4_fa_mma(gacc[j + 1], pa, b[2], b[3]);
            }
        }
    }
    {   /* 出口: c0,c1 = 头 lane/4、维 (lane&3)*2+{0,1}; c2,c3 = 头 +8 */
        const uint32_t hA = lane >> 2, hB = hA + 8u;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t d = warp * 64u + j * 8u + (lane & 3u) * 2u;
            float *gA = gq + ((uint64_t)i * n_head + h0 + hA) * HD + d, *gB = gq + ((uint64_t)i * n_head + h0 + hB) * HD + d;
            gA[0] = gacc[j][0]; gA[1] = gacc[j][1]; gB[0] = gacc[j][2]; gB[1] = gacc[j][3];
        }
    }
}
static size_t bwd_qm_smem_bytes(void) {
    return (size_t)DS4_ATTN_MMA_KT * DS4_ATTN_FA_LD * 2u + 8u * DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * 4u + 2u * DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * 4u
         + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * 2u + 4u * DS4_ATTN_MMA_HEADS * 4u + 8u * DS4_ATTN_MMA_HEADS * 4u + DS4_ATTN_MMA_KT * 4u;
}

/* ② 键梯度: block (query i, 键块, 维块) 算 G_K[键][维] = Σ_e W[键][e]·V[e][维], e ∈ [0, 2·头):
 * e < 头 → V = q_ie(权重 scale·g_s), 否则 V = g_o(权重 p)。一轮吃 64 个 e: A 片 = V 的 [64 个 e][128 维](f32 → bf16, 按 [K][M] 存, .trans 取);
 * B 片 = wb 的 [128 个键][64 个 e](已是 bf16, 直接搬; 交织/取法同 vqm 的 B 片)。出口: 键 < nwin → gkv[lo+键], 否则 → gcomp[组号]。 */
__global__ __launch_bounds__(512, 1) static void bwd_attn_kg_kernel(float *gkv, float *gcomp, const uint16_t *wb, uint32_t nkmax, const float *q,
                                                                    const float *go, const int32_t *idx, uint32_t n_head, uint32_t window,
                                                                    uint32_t ng, uint32_t topk) {
    __shared__ __align__(16) uint8_t As[64u * BWD_KG_BM * 2u];   /* 16 KB */
    __shared__ __align__(16) uint8_t Bs[BWD_KG_BN * 64u * 2u];   /* 16 KB */
    constexpr uint32_t NFW = (BWD_KG_BN / 8u) / 2u;               /* 每 warp 8 个 n8 片(键分两半给 th = 0/1) */
    const uint32_t i = blockIdx.x, kc0 = blockIdx.y * BWD_KG_BN, d0 = blockIdx.z * BWD_KG_BM;
    const uint32_t lo = i + 1u > window ? i + 1u - window : 0u, nwin = i - lo + 1u, nk = nwin + topk;
    if (kc0 >= nk) return;   /* 整个 block 一起退, 不会卡在屏障上 */
    const uint32_t nt = nk - kc0 < BWD_KG_BN ? nk - kc0 : BWD_KG_BN;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint64_t wrow = 2u * (uint64_t)n_head;
    float acc[NFW][4];
    #pragma unroll
    for (uint32_t f = 0; f < NFW; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }
    for (uint32_t rd = 0; rd < (2u * n_head) / 64u; rd++) {
        __syncthreads();
        {   /* A: 线程 → 第 er 个 e、维 dc..dc+15(两个 16 B 块) */
            const uint32_t er = tid >> 3, dc = (tid & 7u) * 16u, e = rd * 64u + er;
            const float *src = (e < n_head ? q + ((uint64_t)i * n_head + e) * 512u : go + ((uint64_t)i * n_head + (e - n_head)) * 512u) + d0 + dc;
            uint32_t pk[8];
            #pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                const float4 f = *(const float4 *)(src + c * 4u);
                __nv_bfloat162 b0 = __floats2bfloat162_rn(f.x, f.y), b1 = __floats2bfloat162_rn(f.z, f.w);
                memcpy(&pk[2u * c], &b0, 4); memcpy(&pk[2u * c + 1u], &b1, 4);
            }
            *(uint4 *)(As + vqm_swz256(er, dc / 8u)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            *(uint4 *)(As + vqm_swz256(er, dc / 8u + 1u)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        }
        #pragma unroll
        for (uint32_t c = 0; c < 2u; c++) {   /* B: 128 个键 × 8 个 16 B 块, 每线程 2 块 */
            const uint32_t ch = tid + c * 512u, bt = ch >> 3, bc = ch & 7u;
            uint4 pk = make_uint4(0, 0, 0, 0);
            if (bt < nt) pk = *(const uint4 *)(wb + ((uint64_t)i * nkmax + kc0 + bt) * wrow + rd * 64u + bc * 8u);
            *(uint4 *)(Bs + vqm_swz(bt, bc)) = pk;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < 4u; kk++) {
            uint32_t a[4];
            {   const uint32_t mt = lane >> 3, krow = kk * 16u + (lane & 7u) + (mt >> 1) * 8u;
                vqm_ldsm4t(a, As + vqm_swz256(krow, wr * 2u + (mt & 1u))); }
            #pragma unroll
            for (uint32_t np = 0; np < NFW / 2u; np++) {
                const uint32_t f0 = th * NFW + 2u * np;
                if (f0 * 8u >= nt) break;
                uint32_t b[4];
                const uint32_t mt = lane >> 3, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                vqm_ldsm4(b, Bs + vqm_swz(tok, kk * 2u + (mt & 1u)));
                float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                vqm_mma(t0, a, b[0], b[1]);
                #pragma unroll
                for (int x = 0; x < 4; x++) acc[2 * np][x] += t0[x];
                if ((f0 + 1u) * 8u < nt) { vqm_mma(t1, a, b[2], b[3]);
                    #pragma unroll
                    for (int x = 0; x < 4; x++) acc[2 * np + 1][x] += t1[x]; }
            }
        }
    }
    /* 出口: c0,c1 = 维 lane/4、键 (lane&3)*2+{0,1}; c2,c3 = 维 +8 */
    #pragma unroll
    for (uint32_t h2 = 0; h2 < 2u; h2++) {
        const uint32_t dcol = d0 + wr * 16u + (lane >> 2) + h2 * 8u;
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) {
            const uint32_t gf = th * NFW + f;
            if (gf * 8u >= nt) break;
            #pragma unroll
            for (uint32_t e2 = 0; e2 < 2u; e2++) {
                const uint32_t t = gf * 8u + (lane & 3u) * 2u + e2;
                if (t >= nt) continue;
                const uint32_t kk = kc0 + t;
                float *dst;
                if (kk < nwin) dst = gkv + (uint64_t)(lo + kk) * 512u;
                else {
                    if (!gcomp) continue;   /* 源层在训练段之下: 压缩行是常量 */
                    const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                    if (g < 0 || (uint32_t)g >= ng) continue;
                    dst = gcomp + (uint64_t)g * 512u;
                }
                atomicAdd(dst + dcol, acc[f][h2 * 2u + e2]);   /* 同一键被多个 query 看见、同一组被多个 query 选中: 必须原子加 */
            }
        }
    }
}

static v41_scratch g_bwd_wb;
int ds4_gpu_bwd_sparse_attn_tensor(ds4_gpu_tensor *gq, ds4_gpu_tensor *gkv, ds4_gpu_tensor *gcomp, const ds4_gpu_tensor *go, const ds4_gpu_tensor *o,
                                   const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win, const ds4_gpu_tensor *kv_comp,
                                   const ds4_gpu_tensor *idx, const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                   uint32_t n_tok, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale) {
    (void)model_size;
    if (!gq || !gkv || !go || !o || !q || !kv_win || head_dim != 512u || n_tok == 0) return 0;
    if (n_head % 32u) { fprintf(stderr, "ds4: [bwd] 注意力反向: 头数 %u 不是 32 的倍数(键梯度核一轮吃 64 个 e = 2·头)\n", n_head); return 0; }
    if (window + topk > BWD_ATT_MAXK) { fprintf(stderr, "ds4: [bwd] 注意力反向: 窗口 %u + topk %u 超核内上限 %u\n", window, topk, BWD_ATT_MAXK); return 0; }
    if ((kv_comp == NULL) != (idx == NULL)) return 0;
    if (!kv_comp) topk = 0;
    const uint32_t nkmax = window + topk;
    const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "bwd sink");
    uint16_t *wb = (uint16_t *)v41_grow(&g_bwd_wb, (uint64_t)n_tok * nkmax * 2u * n_head * 2u, "bwd attn wb");
    if (!sink || !wb) return 0;
    bwd_attn_qm_kernel<<<dim3(n_tok, n_head / DS4_ATTN_MMA_HEADS), 256, bwd_qm_smem_bytes(), g_cur_stream>>>((float *)gq->ptr, wb, nkmax, (const float *)go->ptr,
        (const float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr, kv_comp ? (const uint8_t *)kv_comp->ptr : NULL,
        idx ? (const int32_t *)idx->ptr : NULL, sink, n_head, window, ng, topk, scale);
    if (!cuda_ok(cudaGetLastError(), "bwd attn q")) return 0;
    if (cudaMemsetAsync(gkv->ptr, 0, (size_t)n_tok * 512u * 4u, g_cur_stream) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    bwd_attn_kg_kernel<<<dim3(n_tok, (nkmax + BWD_KG_BN - 1u) / BWD_KG_BN, 512u / BWD_KG_BM), 512, 0, g_cur_stream>>>((float *)gkv->ptr,
        (gcomp && kv_comp) ? (float *)gcomp->ptr : NULL, wb, nkmax, (const float *)q->ptr, (const float *)go->ptr,
        idx ? (const int32_t *)idx->ptr : NULL, n_head, window, ng, topk);
    return cuda_ok(cudaGetLastError(), "bwd attn kg");
}

/* RoPE 的反向 = 反方向旋转(正交阵的转置), 不舍 bf16。调用方传"前向那一发的 inverse 取反"。算式与 v41_rope_kernel 同源(v41_rope_freq)。 */
__global__ static void bwd_rope_kernel(float *x, const int32_t *pos, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                                       float theta, uint32_t osl, float factor, float bf, float bs, int inverse) {
    const uint32_t t = blockIdx.y, h = blockIdx.x, i = threadIdx.x;
    if (i >= n_rot / 2u) return;
    float *xr = x + ((uint64_t)t * n_head + h) * head_dim + (head_dim - n_rot) + 2u * i;
    const float ang = (float)pos[t] * v41_rope_freq(i, n_rot, theta, osl, factor, bf, bs);
    float c = cosf(ang), s = sinf(ang);
    if (inverse) s = -s;
    const float a = xr[0], b = xr[1];
    xr[0] = a * c - b * s;
    xr[1] = a * s + b * c;
}
int ds4_gpu_bwd_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                            uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast, float beta_slow, bool inverse) {
    if (!x || !pos || (n_rot & 1u) || n_rot > 128u || n_tok == 0) return 0;
    bwd_rope_kernel<<<dim3(n_head, n_tok), n_rot / 2u, 0, g_cur_stream>>>((float *)x->ptr, (const int32_t *)pos->ptr, n_head, head_dim, n_rot,
                                                                         theta, original_seq_len, factor, beta_fast, beta_slow, inverse ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "bwd rope");
}
