/* cuda_v41_attn_mma_decode.inc.cu — ds4_cuda.cu 分片: 解码路的稀疏注意力上张量核(decode.md D2, 2026-09-16)。
 *
 * 病(12k 真场景逐核实测): 解码的标量 split 版 **12.12 ms/步 = 每层 303 µs**, 而一层只做
 * 640 键 × 64 头 × 512 维 = 8.4 亿次 FMA —— 合 69 GFLOP/s, 是这块板子标量峰值的 **0.4%**。
 * 字节也不是: 每层读 5 MB, 按 240 GB/s 只值 0.02 ms。两个假设已经实测排除:
 *   ①占用率: 段数 6 → 36(block 48 → 288, 每 SM 4 → 24 个 warp)只换来 1.6%, **判负**(见 attn_split 存档);
 *   ②字节: 见上。
 * 剩下的就是**指令数**: 标量版每算一个 (键, 头) 要付一次 5 条 shfl 的 warp 归约才换 16 个 FMA,
 * 外加一次 expf、两次 bf16 舍入。这正是 09-15 段 4 给**预填**写张量核版时诊断出来的同一件事
 * (那次预填 1.76×)。解码这条路当时没跟上, 因为 mma 版的 grid 是 (n_tok, 头组), n_tok=1 时只有 4 个 block。
 *
 * 这一片补的就是那一块: **mma 版 + 按键分段**, 与标量 split 版同一个骨架 ——
 *   grid (键段, 头组 16 个一组), 每个 block 在自己那段键上做完整的两遍在线 softmax, 出局部
 *   (max, sum, acc) 三件, 再交给 **现成的** v41_sparse_attn_merge_kernel 按段号固定序合并。
 * 对一个 query 来说 S = Q[64×512] · Kᵀ[512×nkeys] 本来就是一个真 GEMM, 这才是它该有的形态。
 *
 * ★数值★: 与标量版**不是逐字节同** —— 张量核的累加序、分段的在线 softmax 分组都不同
 * (与 09-15 段 4 落地预填 mma 版时同一条规矩, 那次 NLL 反而好 1.3%)。判据是质量尺, 不是 cmp。
 * 合并仍按段号固定序(不是原子加) ⇒ 同一输入两跑仍逐位可复现, 温 0 的复现性不丢。
 *
 * ★出错会怎样★: 无效 topk 槽(idx<0)必须"键行清零"且"分数压 -inf"两件都做 —— 只做一件的后果
 * 见 cuda_sparse_attn_mma.inc.cu 头注(0×NaN / softmax 给了它权重), 都是静默走偏。
 * 这里两件都由复用的 ds4_attn_mma_gather_keys / ds4_attn_mma_scores 承担, 所以**不要在这里
 * 另写一份 gather** —— 写第二份就是迟早与预填那份漂开。 */

/* 段长公式 v41_attn_seg_keys / 段数 v41_attn_nseg_at 的正本在 cuda_v41_attn_split.inc.cu(合并核也要用, 那片排在前面)。 */

/* 一段键上的注意力, 出局部 (acc, max, sum)。结构逐段照抄 ds4_sparse_attn_mma_kernel 的两遍扫,
 * 只有三处不同: ①键范围限定在 [k0, k1) ②出口不除分母、不加 sink(留给合并核) ③写 pacc/pmax/psum。
 * ★posd(graph 路, 2026-09-18)★: 位置从设备槽读, ng/topk 按它自算(ds4_gpu_v41.h "设备位置"口径); nseg 是桶上限,
 * 超出真段数的段直接返回不写 —— 合并核按同一公式只读真段, 所以不写也不会读到残留。 */
__global__ static void v41_attn_mma_seg_kernel(float *pacc, float *pmax, float *psum,
                                               const float *q, const float *kvw, const uint8_t *kvc,
                                               const int32_t *idx, uint32_t pos0, uint32_t window,
                                               uint32_t ng, uint32_t topk, uint32_t n_head,
                                               float scale, uint32_t ratio, uint32_t nseg, const int32_t *posd) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    /* graph: 位置在设备槽; n 行(gridDim.z, 投机验证批进图)时源层组数 = (pos0 + n)/ratio, 与直发路主机传的 ng_src 同式 */
    if (posd) { pos0 = (uint32_t)posd[0]; ng = ratio ? (pos0 + gridDim.z) / ratio : 0u; if (ng < topk) topk = ng; }
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);

    /* i = 本批第几个 query(解码恒 0; 投机验证批 0..k)。★每个 query 的可见键范围不同★ ——
     * 段数 nseg 是按最后一个 query(键最多)定的, 所以靠前的 query 会有"这一段整段都在可见范围之外"的
     * 空段: 必须照样把 max=-inf / sum=0 / acc=0 写出去, 合并核才不会读到上一轮的残留。 */
    const uint32_t seg = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS, i = blockIdx.z;
    const uint64_t pbase = ((uint64_t)i * nseg + seg) * n_head + h0;
    const uint32_t p = pos0 + i;
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    /* ★段长按这个 query 自己的位置算★(见文件头): 靠前的 query 段长可能与末位不同, 于是它的段数
     * 也可能少于 grid 给的 nseg —— 多出来的那些段走下面的"空段"分支, 对合并是精确中性的。 */
    const uint32_t seg_keys = v41_attn_seg_keys(p, window, ratio, topk);
    const uint32_t k0 = seg * seg_keys;
    if (k0 >= nkeys) {   /* 空段: 写中性值就走(exp(-1e30 - m) = 0 ⇒ 对合并没有贡献) */
        if (posd) return;   /* graph 路: 合并核只读真段, 空段不写(省 32 KB/段的白写) */
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
            pacc[(pbase + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD] = 0.f;
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = -1e30f; psum[pbase + threadIdx.x] = 0.f; }
        return;
    }
    const uint32_t k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
        qs[e] = __float2bfloat16(q[((uint64_t)i * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    __syncthreads();

    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {   /* 第一遍: 本段的 max 与 exp 和 */
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) {
            const uint32_t h = threadIdx.x;
            float m = rmax[h], sm = rsum[h], tm = m;
            for (uint32_t k = 0; k < DS4_ATTN_MMA_KT; k++) tm = fmaxf(tm, stile[h * 16u + k]);
            sm *= expf(m - tm);
            for (uint32_t k = 0; k < DS4_ATTN_MMA_KT; k++) sm += expf(stile[h * 16u + k] - tm);
            rmax[h] = tm; rsum[h] = sm;
        }
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {   /* 第二遍: 重算 S → P(bf16) → O */
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x)
            ptile[e] = __float2bfloat16(expf(stile[e] - rmax[e >> 4]));
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
        wmma::load_matrix_sync(pa, ptile, 16);
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(vb, ks + warp * 64u + (uint32_t)j * 16u, DS4_ATTN_MMA_HD);
            wmma::mma_sync(oacc[j], pa, vb, oacc[j]);
        }
    }

    /* 出口: 只落局部 acc/max/sum。★按 j 分四轮存★ —— 8 个 warp × 4 片一次要 32 KB, shared 装不下,
     * 而且不带 warp 偏移地共用一块就是 09-15 踩过的那个"互相踩、PPL 变 28 万"的坑。 */
    float *otile = spart;
    for (int j = 0; j < 4; j++) {
        __syncthreads();
        wmma::store_matrix_sync(otile + (size_t)warp * 256u, oacc[j], 16, wmma::mem_row_major);
        __syncthreads();
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_WARPS * 256u; e += blockDim.x) {
            const uint32_t w = e >> 8, r = e & 255u, h = r >> 4, d16 = r & 15u;
            pacc[(pbase + h) * DS4_ATTN_MMA_HD + w * 64u + (uint32_t)j * 16u + d16] = otile[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) {
        pmax[pbase + threadIdx.x] = rmax[threadIdx.x];
        psum[pbase + threadIdx.x] = rsum[threadIdx.x];
    }
}

/* 返回 1 = 这一发由解码张量核接管; 0 = 形状不合/shared 抬不上去, 调用方回标量 split 版。
 * ★n_tok 1..8 全走这一条★: 纯解码与投机验证批必须是同一个核、同一套分段, 否则同轨不成立。 */
/* graph 路要的暂存预建(2026-09-18): v41_grow 里有 cudaMalloc + 全设备同步, 捕获态下二者都作废捕获 ——
 * 所以开捕获之前把局部件按段数上限(V41_ATTN_SPLIT_MAX_SEG)一次长够。8 MB, 只长不缩。 */
int ds4_gpu_v41_attn_scratch_prepare(uint32_t n_head, uint32_t head_dim) {
    const uint64_t na = (uint64_t)V41_ATTN_SPLIT_MAX_SEG * n_head;
    return v41_grow(&g_v41_attn_pacc, na * head_dim * 4, "v41 attn mma acc") &&
           v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn mma max") &&
           v41_grow(&g_v41_attn_psum, na * 4, "v41 attn mma sum") ? 1 : 0;
}

/* posd / pos_cap: graph 路(见 ds4_gpu_v41.h): pos0..pos_cap 是图的有效位置区间, 段数按区间内最大值开 grid
 * (段数随位置不单调 —— 段长跳档时段数会掉, 所以得逐个位置扫, 不能只看两端)。 */
static int v41_attn_mma_decode(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                               const float *sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                               uint32_t topk, uint32_t ratio, uint32_t n_head, uint32_t hd, float scale,
                               const int32_t *posd, uint32_t pos_cap) {
    /* n_tok 1 = 纯解码; 2..8 = 投机验证批 —— ★两者走同一族核是"同轨"的前提★(mtp.md M1):
     * 温 0 下投机输出要与纯解码逐字节同, 而同一个 token 走两套不同累加序的核就不可能同。 */
    if (n_tok == 0u || n_tok > 8u || hd != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS)) return 0;
    const uint32_t plast = pos0 + n_tok - 1u;        /* 段数按键最多的那个 query 算(靠前的 query 多出来的段是空段) */
    const uint32_t nwin = plast + 1u > window ? window : plast + 1u;
    const uint32_t nkeys = nwin + topk;
    /* ★键少也走这里, 不再回落标量版(2026-09-16 M1′)★: 原来 nkeys < 64 就交给 split 版, 而 split 只收
     * n_tok==1 —— 于是会话开头那几步"纯解码走 split、验证批走预填核", 连核都不是同一个, 必然不同轨。
     * 键少时这个核只有一段, 开销就是一次 q 载入, 不值得为它留第二条路(铁律: 不留兜底路)。 */
    static int s_ok = 0;                             /* 0 未试 / 1 可用 / -1 抬不上去 */
    const size_t smem = ds4_attn_mma_smem_bytes();
    if (s_ok == 0) {
        s_ok = cudaFuncSetAttribute(v41_attn_mma_seg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess ? 1 : -1;
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [attn] 解码张量核版 shared %zu KB %s\n", smem >> 10, s_ok == 1 ? "已开" : "★抬不上去, 回标量版★");
    }
    if (s_ok != 1) return 0;
    /* grid 的段数 = 批里各 query 各自算出来的段数的**最大值**(它们的段长可能不同, 见 v41_attn_seg_keys)。
     * 段长本身由核里按各自的位置再算一遍 —— 主机只负责把 grid 开够。 */
    uint32_t nseg = 1u;
    if (posd) {   /* graph: 桶内每个位置的段数取最大(核里按真位置算真段数, 多出的段空跑); n 行时末行位置到 pos_cap + n − 1 */
        if (pos_cap < pos0) return 0;
        for (uint32_t p = pos0; p <= pos_cap + n_tok - 1u; p++) {
            const uint32_t ns = v41_attn_nseg_at(p, window, ratio, topk);
            if (ns > nseg) nseg = ns;
        }
    } else
    for (uint32_t i = 0; i < n_tok; i++) {
        const uint32_t p = pos0 + i;
        const uint32_t nw = p + 1u > window ? window : p + 1u;
        const uint32_t nk = nw + topk;
        const uint32_t sk = v41_attn_seg_keys(p, window, ratio, topk);
        const uint32_t ns = (nk + sk - 1u) / sk;
        if (ns > nseg) nseg = ns;
    }
    if (nseg > V41_ATTN_SPLIT_MAX_SEG) {   /* 键数上限 = 窗口 + indexer top-k 上限, 到不了这里 */
        fprintf(stderr, "ds4: ★[attn] 键 %u 需要 %u 段, 超上限 %u —— 走了另一条核, 同轨不再成立★\n",
                nkeys, nseg, (unsigned)V41_ATTN_SPLIT_MAX_SEG);
        return 0;
    }
    const uint64_t na = (uint64_t)nseg * n_tok * n_head;
    float *pacc = (float *)v41_grow(&g_v41_attn_pacc, na * hd * 4, "v41 attn mma acc");
    float *pmax = (float *)v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn mma max");
    float *psum = (float *)v41_grow(&g_v41_attn_psum, na * 4, "v41 attn mma sum");
    if (!pacc || !pmax || !psum) return 0;
    v41_attn_mma_seg_kernel<<<dim3(nseg, n_head / DS4_ATTN_MMA_HEADS, n_tok), DS4_ATTN_MMA_WARPS * 32u, smem, g_cur_stream>>>(
        pacc, pmax, psum, q, kvw, kvc, idx, pos0, window, ng, topk, n_head, scale, ratio, nseg, posd);
    if (!cuda_ok(cudaGetLastError(), "v41 attn mma seg")) return 0;
    /* 合并复用标量 split 版那一发(按段号固定序 + sink 进分母 + 除 + 舍 bf16), 语义完全一样 */
    v41_sparse_attn_merge_kernel<<<dim3(n_head, n_tok), 256, 0, g_cur_stream>>>(o, pacc, pmax, psum, sink, nseg, n_head, hd,
                                                                                posd, window, ratio, topk);
    return cuda_ok(cudaGetLastError(), "v41 attn mma merge");
}
