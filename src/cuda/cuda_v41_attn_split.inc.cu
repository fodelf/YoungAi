/* cuda_v41_attn_split.inc.cu — ds4_cuda.cu 分片: 解码路的稀疏注意力 split-K(single.md S4, 2026-09-15)。
 *
 * 病(逐核时间线实测): 解码时 `v41_sparse_attn_kernel` 的 grid 是 (n_tok=1, 8 个头组) = **只有 8 个 block**,
 * 48 个 SM 里 40 个干等; 键只能在这 8 个 block 里串着啃, 实测 **每个键 5.3 µs**, 32 步时已经 5~8 ms/步,
 * 而且随可见键数线性涨 —— 32k 上下文处 640 键 × 40 层 ≈ 136 ms/步, 比现在整步还长。
 *
 * 修: 把键切成段, grid 变成 (键段, 头组), 每个 block 只啃自己那段的键, 出**局部**的
 * (max, sum, acc) 三件(flash-attention 的在线 softmax 是可合并的), 再一发把各段合起来。
 * 48 个 SM 全上, 且键多了就多切几段 —— 时间不再随上下文线性涨。
 *
 * ★数值★: 分段改变了在线 softmax 的分组(与段 4 的 mma 版同一性质, 那次 NLL 反而好 1.3%),
 * 所以门是 NLL 尺不是逐位同。合并按**段号固定序**, 不用原子加 —— 否则同一输入两跑结果不同(E0 那种病)。
 *
 * 只给解码(n_tok == 1)用: n 大时原核的 grid 本来就够宽, 且预填走的是 mma 版。 */

/* 这两个常量的"正本"在 cuda_v41_2.inc.cu(主注意力核)。本分片按 include 顺序排在它**前面**(它要调 split),
 * 所以在这里先定义, 那边用 #ifndef 守着 —— 两处写的必须是同一个值, 改一处就要改另一处。 */
#define V41_ATTN_HEADS_PER_BLOCK 8u
#define V41_ATTN_KTILE 8u

#define V41_ATTN_SPLIT_MIN_KEYS 128u   /* 键少于这个数就别切了: 段太短, 合并那一发的开销反而占大头 */
#define V41_ATTN_SPLIT_SEG_KEYS 64u    /* 一段多少个键(8 的倍数; 与 KTILE 8 对齐) */
#define V41_ATTN_SPLIT_MAX_SEG  64u    /* 段数上限 ⇒ 暂存 64 × 64 头 × 512 × 4 B = 8 MB */

/* 一段的局部注意力: 与 v41_sparse_attn_kernel 的内层同一口径(8 键一块进 shared, bf16 舍概率),
 * 只是键范围限定在 [k0, k1) 且出口不除分母 —— 分母留给合并核。
 * pacc[seg][head][hd] / pmax[seg][head] / psum[seg][head] */
__global__ static void v41_sparse_attn_split_kernel(float *pacc, float *pmax, float *psum,
                                                    const float *q, const float *kvw, const float *kvc, const int32_t *idx,
                                                    uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                                    uint32_t n_head, uint32_t hd, float scale, uint32_t seg_keys) {
    const uint32_t seg = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;
    __shared__ float ks[V41_ATTN_KTILE][512];
    __shared__ int   kok[V41_ATTN_KTILE];
    float qa[2][16], acc[2][16], mx[2], sum[2];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) { qa[hh][e] = q[(uint64_t)h * hd + lane * per + e]; acc[hh][e] = 0.f; }
        mx[hh] = -1e30f; sum[hh] = 0.f;
    }
    const uint32_t p = pos0;                       /* 解码: 只有一个 query, 绝对位置就是 pos0 */
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    const uint32_t k0 = seg * seg_keys, k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t base = k0; base < k1; base += V41_ATTN_KTILE) {
        const uint32_t nt = (k1 - base) < V41_ATTN_KTILE ? (k1 - base) : V41_ATTN_KTILE;
        __syncthreads();
        for (uint32_t t = threadIdx.x / 32u; t < nt; t += blockDim.x / 32u) {
            const uint32_t kk = base + t;
            const float *krow = NULL;
            if (kk < nwin) krow = kvw + (uint64_t)((int64_t)lo + kk - ((int64_t)pos0 - (int64_t)window)) * hd;
            else { const int32_t g = idx[kk - nwin];
                   if (g >= 0 && (uint32_t)g < ng) krow = kvc + (uint64_t)g * hd; }
            if (lane == 0) kok[t] = krow != NULL;
            for (uint32_t d = lane; d < hd; d += 32u) ks[t][d] = krow ? krow[d] : 0.f;   /* 无效槽必须写 0, 见主核注释 */
        }
        __syncthreads();
        for (int hh = 0; hh < 2; hh++) {
            float s[V41_ATTN_KTILE];
            float tm = -1e30f;
            for (uint32_t t = 0; t < nt; t++) {
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * per + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);
            float acc_add[16];
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
            for (uint32_t t = 0; t < nt; t++) {
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = v41_bf16r(pv);
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * per + e];
            }
            sum[hh] = sum[hh] * rs + ps;
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        float *pa = pacc + ((uint64_t)seg * n_head + h) * hd;
        for (uint32_t e = 0; e < per; e++) pa[lane * per + e] = acc[hh][e];
        if (lane == 0) { pmax[(uint64_t)seg * n_head + h] = mx[hh]; psum[(uint64_t)seg * n_head + h] = sum[hh]; }
    }
}

/* 合并: 一 block 一个头, 按**段号固定序**做 flash 合并(不是原子加 —— 那会让同一输入两跑结果不同),
 * 再加 sink 进分母、除、舍 bf16。o[head][hd] */
__global__ static void v41_sparse_attn_merge_kernel(float *o, const float *pacc, const float *pmax, const float *psum,
                                                    const float *sink, uint32_t nseg, uint32_t n_head, uint32_t hd) {
    const uint32_t h = blockIdx.x;
    float m = -1e30f;
    for (uint32_t s = 0; s < nseg; s++) m = fmaxf(m, pmax[(uint64_t)s * n_head + h]);
    float den = 0.f;
    for (uint32_t s = 0; s < nseg; s++) den += psum[(uint64_t)s * n_head + h] * expf(pmax[(uint64_t)s * n_head + h] - m);
    den += expf(sink[h] - m);
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t s = 0; s < nseg; s++)
            v += pacc[((uint64_t)s * n_head + h) * hd + d] * expf(pmax[(uint64_t)s * n_head + h] - m);
        o[(uint64_t)h * hd + d] = v41_bf16r(v / den);
    }
}

static v41_scratch g_v41_attn_pacc, g_v41_attn_pmax, g_v41_attn_psum;

/* 返回 1 = 这一发由 split-K 路接管; 0 = 形状不适用, 调用方回落到原核。 */
static int v41_sparse_attn_split(float *o, const float *q, const float *kvw, const float *kvc, const int32_t *idx,
                                 const float *sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                                 uint32_t topk, uint32_t n_head, uint32_t hd, float scale) {
    if (n_tok != 1u || hd != 512u || (n_head % V41_ATTN_HEADS_PER_BLOCK)) return 0;
    const uint32_t nwin = pos0 + 1u > window ? window : pos0 + 1u;
    const uint32_t nkeys = nwin + topk;
    if (nkeys < V41_ATTN_SPLIT_MIN_KEYS) return 0;
    uint32_t seg_keys = V41_ATTN_SPLIT_SEG_KEYS;
    uint32_t nseg = (nkeys + seg_keys - 1u) / seg_keys;
    while (nseg > V41_ATTN_SPLIT_MAX_SEG) { seg_keys *= 2u; nseg = (nkeys + seg_keys - 1u) / seg_keys; }
    const uint64_t na = (uint64_t)nseg * n_head;
    float *pacc = (float *)v41_grow(&g_v41_attn_pacc, na * hd * 4, "v41 attn split acc");
    float *pmax = (float *)v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn split max");
    float *psum = (float *)v41_grow(&g_v41_attn_psum, na * 4, "v41 attn split sum");
    if (!pacc || !pmax || !psum) return 0;
    v41_sparse_attn_split_kernel<<<dim3(nseg, n_head / V41_ATTN_HEADS_PER_BLOCK), V41_ATTN_HEADS_PER_BLOCK * 16u, 0, g_cur_stream>>>(
        pacc, pmax, psum, q, kvw, kvc, idx, pos0, window, ng, topk, n_head, hd, scale, seg_keys);
    if (!cuda_ok(cudaGetLastError(), "v41 sparse attn split")) return 0;
    v41_sparse_attn_merge_kernel<<<n_head, 256, 0, g_cur_stream>>>(o, pacc, pmax, psum, sink, nseg, n_head, hd);
    return cuda_ok(cudaGetLastError(), "v41 sparse attn merge");
}
