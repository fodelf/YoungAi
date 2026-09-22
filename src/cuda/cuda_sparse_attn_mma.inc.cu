/* cuda_sparse_attn_mma.inc.cu — 稀疏注意力的张量核版(speed.md 段 4, 2026-09-15)。
 *
 * 为什么要这一份: 标量版(cuda_v41_2.inc.cu 的 v41_sparse_attn_kernel)在预填里占 43%, 实测只有 259 GFLOPS,
 * 而这块板子的 BF16 稠密张量核实测 81~91 TFLOPS(S0, gemm_fp4_ceiling)。标量版慢的根子不是访存 ——
 * 09-15 把"扇区效率 / 占用率 / 激活重读 / 键重读"四个访存假设逐个改过, 全部判负(详见 fable5.md);
 * 唯一有效的那次改的是指令数。标量版每处理一个键、一个头, 要付一次 5 条 shuffle 的 warp 归约 + 一次 expf
 * + 两次 bf16 舍入, 才换 16 个有效 FMA。张量核把"一次点积一次 warp 归约"整个换掉。
 *
 * 形状(MLA): 64 个头**共用同一份 512 维 KV**, 所以对一个 query 来说
 *   S[64 头][nkeys] = Q[64][512] · Kᵀ[512][nkeys]        ← 是一个真 GEMM
 *   O[64][512]      = P[64][nkeys] · V[nkeys][512]       ← 同一份 KV 当 V 用(官方就是这么写的)
 * 一 block 管 1 个 query × 16 个头(= 正好一个 wmma 的 M 块), grid (n_tok, 4)。
 * 为什么不是 64 头一 block: q 进 shared 要 64×512×2 B = 64 KB, 再加键块就超 GB10 的 99 KB 上限。
 * 键被 4 个头组各读一遍无所谓 —— 整段键只有 1.3 MB, 全在 24 MB L2 里(09-15 实测: 把这个"重读"消掉反而慢 2%)。
 *
 * 两遍扫键, 不存整张 S: 第一遍算 S 只为拿每个头的 max/sum(在线 softmax 的分母), 第二遍**重算一次 S** 再出 P 与 O。
 * 代价是 S 算两遍(总算力 2 倍变 3 倍), 换来不用给 S 留 [16][640] f32 = 40 KB 的 shared。
 * 为什么不用标准的单遍 flash(边扫边把 O 按新 max 重缩放): wmma 的累加器 fragment 里"哪个线程持有哪一行"
 * 是不透明的, 按行(头)缩放拿不到那个映射; 重算一遍 S 比跟 fragment 布局较劲可靠得多。
 *
 * ★数值★: q/kv 的值本来就落在 bf16 格点上(调用方 rms_norm / act_quant 舍过), 所以转 bf16 是**无损的**;
 * 累加在 f32。与标量版只差累加序 ⇒ 判据是 NLL/五指标, 不是逐位(与 09-15 第二轮按键分块同一口径)。
 * P 在乘 V 之前舍 bf16 —— 这一步不是为了省, 是官方 acc_s_cast 就这么做, 标量版也一样。
 *
 * 出错会怎样: 无效 topk 槽(idx<0)的键行必须**写 0 并且把分数压成 -inf 两件都做**。只写 0 的话点积是 0 不是 -inf,
 * softmax 会给它一个 exp(0) 的权重, 表现是长上下文答案慢慢跑偏而不报错; 只压 -inf 不清零的话,
 * shared 里上一块的残留会被当成键参与 P·V —— 那正是 09-15 段 0 定罪的 0×NaN 那条 bug。 */

#define DS4_ATTN_MMA_HEADS 16u   /* 一 block 管几个头 = 一个 wmma M 块 */
#define DS4_ATTN_MMA_KT    16u   /* 一次进 shared 的键数 = 一个 wmma N 块 */
#define DS4_ATTN_MMA_WARPS 8u    /* 一 block 8 个 warp: 第一/二遍分 K 段, 出 O 时分输出维 */
#define DS4_ATTN_MMA_HD    512u  /* 头维, 与 v41 形状绑死(调用方已校验) */

/* 把 16 个键行搬进 shared 并转 bf16; 无效槽整行清零, valid[] 记状态(给后面压 -inf 用)。
 * 键序 = 窗口行(升序)后接 topk 压缩行, 与标量版一字不差。 */
__device__ __forceinline__ static void ds4_attn_mma_gather_keys(
        __nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
        uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
        uint32_t window, uint32_t ng, uint32_t topk) {
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t lane = threadIdx.x & 31u, kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;   /* 窗口行 f32 / 压缩行打包 FP4, 见 cuda_kv_pack */
        if (t < nt) {
            /* ring=1 恒成立: mma 版只在主路(!full_block)被调, 主路的历史段是环(见 v41_win_row) */
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u)
            ks[(size_t)t * DS4_ATTN_MMA_HD + d] = krow ? __float2bfloat16(krow[d])
                                                : (cpk ? __float2bfloat16(v41_ckv_get(cpk, d)) : (__nv_bfloat16)0.0f);
    }
}

/* 一个键块的 S 片 [16 头][16 键]: 8 个 warp 各算 K 维的 1/8(4 个 wmma k 步), partial 落 shared 后按固定序相加。
 * 固定序 = 每次都一样 ⇒ 结果确定(温 0 复现是硬要求, 见 09-15 段 0)。 */
__device__ __forceinline__ static void ds4_attn_mma_scores(
        float *stile, float *spart, const __nv_bfloat16 *qs, const __nv_bfloat16 *ks,
        const int *valid, uint32_t nt, float scale) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;   /* B[d][key] = ks[key][d] ⇒ 列主序, ldm=512 */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    const uint32_t ksteps = DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS;   /* 512/16/8 = 4 */
    for (uint32_t s = 0; s < ksteps; s++) {
        const uint32_t d0 = (warp * ksteps + s) * 16u;
        wmma::load_matrix_sync(a, qs + d0, DS4_ATTN_MMA_HD);
        wmma::load_matrix_sync(b, ks + d0, DS4_ATTN_MMA_HD);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(spart + (size_t)warp * 256u, c, 16, wmma::mem_row_major);
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) {
        float v = 0.f;
        for (uint32_t w = 0; w < DS4_ATTN_MMA_WARPS; w++) v += spart[(size_t)w * 256u + e];
        const uint32_t k = e & 15u;   /* stile 行主序 [头][键] */
        stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
    }
    __syncthreads();
}

__global__ static void ds4_sparse_attn_mma_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                                  const int32_t *idx, const float *sink, uint32_t pos0, uint32_t window,
                                                  uint32_t ng, uint32_t topk, uint32_t n_head, float scale, uint32_t win_lo) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;                       /* [16][512] */
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;                /* [16][512] */
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);             /* [8][16][16] */
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;                             /* [16][16] */
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);                       /* [16][16] */
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;      /* 每头的 max / 分母 */
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);                              /* [16] */

    const uint32_t i = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS;
    for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
        qs[e] = __float2bfloat16(q[((uint64_t)i * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }

    const uint32_t p = pos0 + i;
    uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    if (lo < win_lo) lo = win_lo;   /* 环里 win_lo 之前的槽没写过(CED), 不读(官方 -1 屏蔽同义); 见 ds4_gpu_v41.h */
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    __syncthreads();

    /* 第一遍: 只为拿每头的 max 与 exp 和 */
    for (uint32_t base = 0; base < nkeys; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (nkeys - base) < DS4_ATTN_MMA_KT ? (nkeys - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) {   /* 一线程一头: 在线 max/sum */
            const uint32_t h = threadIdx.x;
            float m = rmax[h], sm = rsum[h], tm = m;
            for (uint32_t k = 0; k < DS4_ATTN_MMA_KT; k++) tm = fmaxf(tm, stile[h * 16u + k]);
            sm *= expf(m - tm);
            for (uint32_t k = 0; k < DS4_ATTN_MMA_KT; k++) sm += expf(stile[h * 16u + k] - tm);
            rmax[h] = tm; rsum[h] = sm;
        }
    }
    __syncthreads();

    /* 第二遍: 重算 S → P(bf16) → O 累加。warp w 负责输出维 [w·64, w·64+64) 的 4 个 n 块。 */
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = 0; base < nkeys; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (nkeys - base) < DS4_ATTN_MMA_KT ? (nkeys - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x)   /* 官方 acc_s_cast: P 先舍 bf16 再乘 V */
            ptile[e] = __float2bfloat16(expf(stile[e] - rmax[e >> 4]));
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;   /* B[key][dim] = ks 行主序 */
        wmma::load_matrix_sync(pa, ptile, 16);
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(vb, ks + warp * 64u + (uint32_t)j * 16u, DS4_ATTN_MMA_HD);
            wmma::mma_sync(oacc[j], pa, vb, oacc[j]);
        }
    }

    /* 出口: 除以分母(sink 只进分母, 与标量版同), 舍 bf16 写回。
     * ★这里踩过一次坑(09-15)★: 一开始 8 个 warp 各把自己的 4 个累加器存到 `otile + j*256` ——
     * **没带 warp 偏移, 八个 warp 写同一块 shared 互相踩**。后果不是报错, 是 PPL 从 15.46 变成 28 万
     * 而且两跑不同(谁最后写赢由调度决定)。存放整 8 个 warp × 4 个片要 32 KB, shared 里没有;
     * 所以按 j 分四轮, 每轮只存 8 个 warp 各一片(8 KB, 正好借 spart), 存完写出去再进下一轮。 */
    float *otile = spart;   /* [8 warp][16 头 × 16 维] f32 = 8 KB */
    for (int j = 0; j < 4; j++) {
        __syncthreads();
        wmma::store_matrix_sync(otile + (size_t)warp * 256u, oacc[j], 16, wmma::mem_row_major);
        __syncthreads();
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_WARPS * 256u; e += blockDim.x) {
            const uint32_t w = e >> 8, r = e & 255u, h = r >> 4, d16 = r & 15u;
            const float den = rsum[h] + expf(sink[h0 + h] - rmax[h]);
            o[((uint64_t)i * n_head + h0 + h) * DS4_ATTN_MMA_HD + w * 64u + (uint32_t)j * 16u + d16] =
                v41_bf16r(otile[e] / den);
        }
    }
}

/* shared 用量: qs 16 KB + ks 16 KB + spart 8 KB + stile 1 KB + ptile 0.5 KB + rmax/rsum/valid < 0.2 KB ≈ 42 KB。
 * 超 48 KB 静态上限, 所以走动态 shared + opt-in(GB10 上限 99 KB)。 */
static size_t ds4_attn_mma_smem_bytes(void) {
    return (size_t)(DS4_ATTN_MMA_HEADS + DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_HD * sizeof(__nv_bfloat16)
         + (size_t)DS4_ATTN_MMA_WARPS * 256u * sizeof(float) + 256u * sizeof(float)
         + 256u * sizeof(__nv_bfloat16) + 2u * DS4_ATTN_MMA_HEADS * sizeof(float)
         + DS4_ATTN_MMA_KT * sizeof(int);
}

/* 能不能用: 形状对得上(64 头 × 512 维, 头数能被 16 整除)+ 块够大(n_tok 小的时候 grid 只有 n_tok×4 个 block,
 * 填不满 48 个 SM, 标量版反而好) + shared 抬得上去。返回 0 = 调用方回标量版。 */
static int ds4_sparse_attn_mma_launch(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                      const int32_t *idx, const float *sink, uint32_t n_tok, uint32_t pos0,
                                      uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head,
                                      uint32_t head_dim, float scale, uint32_t win_lo) {
    if (head_dim != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS) || n_tok < 64u) return 0;
    static int s_ok = 0;   /* 0 未试 / 1 可用 / -1 抬不上去 */
    const size_t smem = ds4_attn_mma_smem_bytes();
    if (s_ok == 0) {
        s_ok = cudaFuncSetAttribute(ds4_sparse_attn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess ? 1 : -1;
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [attn] 张量核版 shared %zu KB %s\n", smem >> 10, s_ok == 1 ? "已开" : "★抬不上去, 回标量版★");
    }
    if (s_ok != 1) return 0;
    ds4_sparse_attn_mma_kernel<<<dim3(n_tok, n_head / DS4_ATTN_MMA_HEADS), DS4_ATTN_MMA_WARPS * 32u, smem, g_cur_stream>>>(
        o, q, kvw, kvc, idx, sink, pos0, window, ng, topk, n_head, scale, win_lo);
    return cuda_ok(cudaGetLastError(), "sparse attn mma");
}
