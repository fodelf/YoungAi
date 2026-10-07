/* cuda_draft_attn.inc.cu — ds4_cuda.cu 分片: 草稿器蒸馏(src/core/core_draft_kd*.c, 2026-10-07)要的四样 GPU 原语。
 *
 * 为什么要单独一片: 训练是"一批几百个草稿块同时教师强制", 每个块的键集合 = 块前面 window 个历史位置(main_x 的投影)+ 块内 B 位,
 * 块内全可见(官方 get_dspark_topk_idxs)。部署的草稿块注意力(cuda_v41_attn_mma_decode.inc.cu, full_block)一次只算一个块、
 * 窗口是一段线性缓冲; 训练要几百个块各自滑动的窗口, 不能把几百份 133 行的缓冲拷来拷去。这里的前向把"历史行从一张
 * [所有位置][512] 的大表上按块首位切、块行从 [R][512] 取"做进 gather, 其余(分段公式 v41_attn_fb_seg_keys / 打分 / 在线
 * max-sum / P·V / 合并 / sink / bf16 出口)原样复用部署那串核的积木 ⇒ 同一个块、同一份键, 两条路算出的 o 逐位同。
 * 反向只接训练要的形态: 历史行是常量(main_x 冻结, 塔的 kv 投影冻结)不出梯度, 只出 q 与块内 kv 的梯度。
 *
 * 出错会怎样: 历史行下标算错不报错, 只是接受率训不上去(块看的是错位的上下文); 门 = 第 0 轮(件为零)的陪审团首位
 * Σmin 必须复现 d1 accjury 在同一份 ids 上的数(core_draft_kd.c)。 */

/* 行 r 属于块 b = r / B; 块首位绝对位置 ib = bpos[b], 历史键数 nh = min(window, ib)(位置 0 之前没有历史);
 * 键 kk < nh 取 hist 第 (ib − nh + kk − hbase) 行(hbase = hist 第 0 行的绝对位置), 否则取块内第 kk − nh 行。 */
__device__ __forceinline__ static const float *dk_key_row(const float *hist, uint32_t hbase, const float *blk, uint32_t b, uint32_t ib,
                                                          uint32_t nh, uint32_t B, uint32_t kk) {
    return kk < nh ? hist + (uint64_t)(ib - nh + kk - hbase) * DS4_ATTN_MMA_HD
                   : blk + ((uint64_t)b * B + (kk - nh)) * DS4_ATTN_MMA_HD;
}

/* 16 个键进 shared 并转 bf16(与 ds4_attn_mma_gather_keys 的窗口行分支同一句转换); 看不见的槽(t ≥ nt)整行清零并标无效 */
__device__ __forceinline__ static void dk_gather(__nv_bfloat16 *ks, int *valid, const float *hist, uint32_t hbase, const float *blk,
                                                 uint32_t b, uint32_t ib, uint32_t nh, uint32_t B, uint32_t base, uint32_t nt) {
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const float *krow = t < nt ? dk_key_row(hist, hbase, blk, b, ib, nh, B, base + t) : NULL;
        if (lane == 0) valid[t] = krow ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_MMA_HD;
        if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
        else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}

/* 一段键上的前向(逐段照抄 v41_attn_mma_seg_kernel 的 full_block 分支, 只换 gather): grid (段, 头组 16, 本片行数), 行 = r0 + blockIdx.z */
__global__ static void dk_attn_seg_kernel(float *pacc, float *pmax, float *psum, const float *q, const float *hist, uint32_t hbase,
                                          const float *blk, const int32_t *bpos, uint32_t r0, uint32_t B, uint32_t window,
                                          uint32_t n_head, float scale, uint32_t nseg) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);
    const uint32_t seg = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS, i = blockIdx.z, r = r0 + i, b = r / B;
    const uint32_t ib = (uint32_t)bpos[b], nh = ib < window ? ib : window, nkeys = nh + B;
    const uint64_t pbase = ((uint64_t)i * nseg + seg) * n_head + h0;
    const uint32_t seg_keys = v41_attn_fb_seg_keys(nkeys), k0 = seg * seg_keys;
    if (k0 >= nkeys) {   /* 空段(历史不满的块): 写中性值, 合并核读到就是 0 贡献 */
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
            pacc[(pbase + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD] = 0.f;
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = -1e30f; psum[pbase + threadIdx.x] = 0.f; }
        return;
    }
    const uint32_t k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
        qs[e] = __float2bfloat16(q[((uint64_t)r * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    __syncthreads();
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {   /* 第一遍: 本段 max 与 exp 和 */
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        dk_gather(ks, valid, hist, hbase, blk, b, ib, nh, B, base, nt);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        ds4_attn_mma_stats(stile, rmax, rsum);
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {   /* 第二遍: 重算 S → P(bf16) → O */
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        dk_gather(ks, valid, hist, hbase, blk, b, ib, nh, B, base, nt);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) ptile[e] = __float2bfloat16(expf(stile[e] - rmax[e >> 4]));
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
        wmma::load_matrix_sync(pa, ptile, 16);
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(vb, ks + warp * 64u + (uint32_t)j * 16u, DS4_ATTN_MMA_HD);
            wmma::mma_sync(oacc[j], pa, vb, oacc[j]);
        }
    }
    float *otile = (float *)ks;   /* 出口两轮各存两片(与部署核同, 见那里的注释) */
    for (int rr = 0; rr < 2; rr++) {
        __syncthreads();
        wmma::store_matrix_sync(otile + (size_t)warp * 512u, oacc[2 * rr], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(otile + (size_t)warp * 512u + 256u, oacc[2 * rr + 1], 16, wmma::mem_row_major);
        __syncthreads();
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_WARPS * 512u; e += blockDim.x) {
            const uint32_t w = e >> 9, x = e & 511u, jj = x >> 8, t = x & 255u, h = t >> 4, d16 = t & 15u;
            pacc[(pbase + h) * DS4_ATTN_MMA_HD + w * 64u + (uint32_t)(2 * rr) * 16u + jj * 16u + d16] = otile[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = rmax[threadIdx.x]; psum[pbase + threadIdx.x] = rsum[threadIdx.x]; }
}

/* 合并(逐式照抄 v41_sparse_attn_merge_kernel 的直发分支), 另出 lse = m + log(den) 给反向。grid (头, 本片行数) */
__global__ static void dk_attn_merge_kernel(float *o, float *lse, const float *pacc, const float *pmax, const float *psum, const float *sink,
                                            uint32_t nseg, uint32_t n_head, uint32_t hd, uint32_t r0) {
    const uint32_t h = blockIdx.x, i = blockIdx.y, r = r0 + i;
    const uint64_t b0 = (uint64_t)i * nseg * n_head;
    float m = -1e30f;
    for (uint32_t s = 0; s < nseg; s++) m = fmaxf(m, pmax[b0 + (uint64_t)s * n_head + h]);
    float den = 0.f;
    for (uint32_t s = 0; s < nseg; s++) den += psum[b0 + (uint64_t)s * n_head + h] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
    den += expf(sink[h] - m);
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t s = 0; s < nseg; s++) v += pacc[(b0 + (uint64_t)s * n_head + h) * hd + d] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
        o[((uint64_t)r * n_head + h) * hd + d] = v41_bf16r(v / den);
    }
    if (threadIdx.x == 0) lse[(uint64_t)r * n_head + h] = m + logf(den);
}

static v41_scratch g_dk_pacc, g_dk_pmax, g_dk_psum;
#define DK_ATTN_ROWS 64u   /* 一片几行: 局部件 64 行 × 9 段 × 64 头 × 2 KB = 75 MB */
int ds4_gpu_draft_attn_fwd_tensor(ds4_gpu_tensor *o, ds4_gpu_tensor *lse, const ds4_gpu_tensor *q, const ds4_gpu_tensor *hist, uint32_t hbase,
                                  const ds4_gpu_tensor *blk, const ds4_gpu_tensor *bpos, uint32_t nb, uint32_t B, uint32_t window,
                                  const void *model_map, uint64_t model_size, uint64_t sink_offset, uint32_t n_head, uint32_t head_dim, float scale) {
    (void)model_size;
    if (!o || !lse || !q || !hist || !blk || !bpos || !nb || !B || head_dim != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS)) return 0;
    const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "dk sink");
    if (!sink) return 0;
    static int s_ok = 0;
    const size_t smem = ds4_attn_mma_seg_smem_bytes();
    if (s_ok == 0) {
        s_ok = cudaFuncSetAttribute(dk_attn_seg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess ? 1 : -1;
        (void)cudaGetLastError();
        if (s_ok != 1) fprintf(stderr, "ds4: [dk] 块注意力核 shared %zu KB 抬不上去\n", smem >> 10);
    }
    if (s_ok != 1) return 0;
    const uint32_t nkmax = window + B, nseg = (nkmax + v41_attn_fb_seg_keys(nkmax) - 1u) / v41_attn_fb_seg_keys(nkmax);   /* 键少的块是空段 */
    const uint64_t na = (uint64_t)nseg * DK_ATTN_ROWS * n_head;
    float *pacc = (float *)v41_grow(&g_dk_pacc, na * head_dim * 4, "dk attn acc");
    float *pmax = (float *)v41_grow(&g_dk_pmax, na * 4, "dk attn max");
    float *psum = (float *)v41_grow(&g_dk_psum, na * 4, "dk attn sum");
    if (!pacc || !pmax || !psum) return 0;
    const uint32_t R = nb * B;
    for (uint32_t r0 = 0; r0 < R; r0 += DK_ATTN_ROWS) {
        const uint32_t nr = R - r0 < DK_ATTN_ROWS ? R - r0 : DK_ATTN_ROWS;
        dk_attn_seg_kernel<<<dim3(nseg, n_head / DS4_ATTN_MMA_HEADS, nr), DS4_ATTN_MMA_WARPS * 32u, smem, g_cur_stream>>>(
            pacc, pmax, psum, (const float *)q->ptr, (const float *)hist->ptr, hbase, (const float *)blk->ptr, (const int32_t *)bpos->ptr,
            r0, B, window, n_head, scale, nseg);
        if (!cuda_ok(cudaGetLastError(), "dk attn seg")) return 0;
        dk_attn_merge_kernel<<<dim3(n_head, nr), 256, 0, g_cur_stream>>>((float *)o->ptr, (float *)lse->ptr, pacc, pmax, psum, sink, nseg, n_head, head_dim, r0);
        if (!cuda_ok(cudaGetLastError(), "dk attn merge")) return 0;
    }
    return 1;
}

/* 反向(标量): 一 block = 一行 × 16 头(8 warp × 2 头), lane 持每头 16 维(与 v41_sparse_attn_kernel 同分工); 键 8 个一片进 shared, 值按 bf16 舍
 * (= 前向看到的键)。对每键 t: s = scale·<q̃,k>(q̃ = bf16 舍的 q, 同前向), p = exp(s − lse), dp = <g_o,k>, g_s = p·(dp − D), D = <g_o,o>;
 *   g_q += scale·g_s·k; 块内键另有 g_k += scale·g_s·q̃ + p·g_o(键与值两种身份; 同一键被块内 B 行 × 64 头看见 ⇒ 原子加)。
 * 前向在 P 上舍的 bf16、kv 的 fp8 格点、q 的 bf16 格点一律当直通。历史键不出梯度(main_x 与塔 kv 投影都冻结)。 */
__global__ __launch_bounds__(256) static void dk_attn_bwd_kernel(float *gq, float *gblk, const float *go, const float *o, const float *q,
                                                                 const float *lse, const float *hist, uint32_t hbase, const float *blk,
                                                                 const int32_t *bpos, uint32_t B, uint32_t window, uint32_t n_head, float scale) {
    constexpr uint32_t HD = DS4_ATTN_MMA_HD, KT = 8u, PER = HD / 32u;
    __shared__ float ks[KT][HD];
    const uint32_t r = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, b = r / B;
    const uint32_t ib = (uint32_t)bpos[b], nh = ib < window ? ib : window, nkeys = nh + B;
    float qa[2][PER], ga[2][PER], gacc[2][PER], Dh[2], lh[2];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * DS4_ATTN_MMA_HEADS + warp * 2u + hh;
        const uint64_t base = ((uint64_t)r * n_head + h) * HD + lane * PER;
        float d = 0.f;
        for (uint32_t e = 0; e < PER; e++) { qa[hh][e] = v41_bf16r(q[base + e]); ga[hh][e] = go[base + e]; gacc[hh][e] = 0.f; d += go[base + e] * o[base + e]; }
        for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
        Dh[hh] = d; lh[hh] = lse[(uint64_t)r * n_head + h];
    }
    for (uint32_t base = 0; base < nkeys; base += KT) {
        const uint32_t nt = (nkeys - base) < KT ? (nkeys - base) : KT;
        __syncthreads();
        for (uint32_t t = warp; t < nt; t += 8u) {
            const float *krow = dk_key_row(hist, hbase, blk, b, ib, nh, B, base + t);
            for (uint32_t d = lane; d < HD; d += 32u) ks[t][d] = v41_bf16r(krow[d]);
        }
        __syncthreads();
        for (uint32_t t = 0; t < nt; t++) {
            const uint32_t kk = base + t;
            const float *kr = ks[t] + lane * PER;
            float *gk = kk >= nh ? gblk + ((uint64_t)b * B + (kk - nh)) * HD + lane * PER : NULL;
            for (int hh = 0; hh < 2; hh++) {
                float s = 0.f, dp = 0.f;
                for (uint32_t e = 0; e < PER; e++) { s += qa[hh][e] * kr[e]; dp += ga[hh][e] * kr[e]; }
                for (int off = 16; off > 0; off >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, off); dp += __shfl_xor_sync(0xffffffffu, dp, off); }
                const float p = expf(s * scale - lh[hh]), c = scale * p * (dp - Dh[hh]);
                for (uint32_t e = 0; e < PER; e++) gacc[hh][e] += c * kr[e];
                if (gk) for (uint32_t e = 0; e < PER; e++) atomicAdd(gk + e, c * qa[hh][e] + p * ga[hh][e]);
            }
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * DS4_ATTN_MMA_HEADS + warp * 2u + hh;
        float *dst = gq + ((uint64_t)r * n_head + h) * HD + lane * PER;
        for (uint32_t e = 0; e < PER; e++) dst[e] = gacc[hh][e];
    }
}
int ds4_gpu_draft_attn_bwd_tensor(ds4_gpu_tensor *gq, ds4_gpu_tensor *gblk, const ds4_gpu_tensor *go, const ds4_gpu_tensor *o, const ds4_gpu_tensor *q,
                                  const ds4_gpu_tensor *lse, const ds4_gpu_tensor *hist, uint32_t hbase, const ds4_gpu_tensor *blk, const ds4_gpu_tensor *bpos,
                                  uint32_t nb, uint32_t B, uint32_t window, uint32_t n_head, uint32_t head_dim, float scale) {
    if (!gq || !gblk || !go || !o || !q || !lse || !hist || !blk || !bpos || !nb || !B || head_dim != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS)) return 0;
    const uint32_t R = nb * B;
    if (cudaMemsetAsync(gblk->ptr, 0, (size_t)R * head_dim * 4u, g_cur_stream) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    dk_attn_bwd_kernel<<<dim3(R, n_head / DS4_ATTN_MMA_HEADS), 256, 0, g_cur_stream>>>((float *)gq->ptr, (float *)gblk->ptr, (const float *)go->ptr,
        (const float *)o->ptr, (const float *)q->ptr, (const float *)lse->ptr, (const float *)hist->ptr, hbase, (const float *)blk->ptr,
        (const int32_t *)bpos->ptr, B, window, n_head, scale);
    return cuda_ok(cudaGetLastError(), "dk attn bwd");
}

/* 陪审团(与 core_v41_dcap.c v41_dcap_jury 同式, 搬上设备按行并行): 一 block 一行, 1024 线程。
 * 三遍扫词表: ①两边的 max 与 argmax(同值取小下标, 与 ds4_gpu_v41_argmax_tensor 同) ②两边的 Σexp(double) ③Σmin(p,q)(double)。 */
__device__ __forceinline__ static double dk_block_sum(double v, double *sh) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    double t = 0.0;
    for (uint32_t w = 0; w < (blockDim.x >> 5); w++) t += sh[w];
    __syncthreads();
    return t;
}
__device__ __forceinline__ static void dk_block_argmax(float v, uint32_t i, float *shv, uint32_t *shi, float *mv, uint32_t *mi) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off); const uint32_t oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) { shv[warp] = v; shi[warp] = i; }
    __syncthreads();
    float bv = shv[0]; uint32_t bi = shi[0];
    for (uint32_t w = 1; w < (blockDim.x >> 5); w++) if (shv[w] > bv || (shv[w] == bv && shi[w] < bi)) { bv = shv[w]; bi = shi[w]; }
    __syncthreads();
    *mv = bv; *mi = bi;
}
__global__ __launch_bounds__(1024) static void dk_jury_kernel(float *out, const float *ls, const float *lt, uint32_t V, float T) {
    __shared__ double shd[32]; __shared__ float shv[32]; __shared__ uint32_t shi[32];
    const uint32_t row = blockIdx.x;
    const float *s = ls + (uint64_t)row * V, *t = lt + (uint64_t)row * V;
    float vs = -INFINITY, vt = -INFINITY; uint32_t is = 0xffffffffu, it = 0xffffffffu;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        if (s[i] > vs || (s[i] == vs && i < is)) { vs = s[i]; is = i; }
        if (t[i] > vt || (t[i] == vt && i < it)) { vt = t[i]; it = i; }
    }
    float ms, mt; uint32_t as, at;
    dk_block_argmax(vs, is, shv, shi, &ms, &as);
    dk_block_argmax(vt, it, shv, shi, &mt, &at);
    double zs = 0.0, zt = 0.0;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        zs += isfinite(s[i]) ? exp(((double)s[i] - ms) / T) : 0.0;
        zt += isfinite(t[i]) ? exp(((double)t[i] - mt) / T) : 0.0;
    }
    zs = dk_block_sum(zs, shd); zt = dk_block_sum(zt, shd);
    double smin = 0.0;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        const double p = isfinite(t[i]) ? exp(((double)t[i] - mt) / T) / zt : 0.0, qq = isfinite(s[i]) ? exp(((double)s[i] - ms) / T) / zs : 0.0;
        smin += p < qq ? p : qq;
    }
    smin = dk_block_sum(smin, shd);
    if (threadIdx.x == 0) {
        out[(uint64_t)row * 4u] = (float)smin;
        out[(uint64_t)row * 4u + 1u] = (float)(exp(((double)t[as] - mt) / T) / zt);   /* 点质量草稿: 教师给草稿 argmax 的概率 */
        out[(uint64_t)row * 4u + 2u] = as == at ? 1.f : 0.f;
        out[(uint64_t)row * 4u + 3u] = 0.f;
    }
}
int ds4_gpu_draft_jury_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *ls, const ds4_gpu_tensor *lt, uint32_t m, uint32_t n_vocab, float T) {
    if (!out || !ls || !lt || !m || !n_vocab || !(T > 0.f)) return 0;
    dk_jury_kernel<<<m, 1024, 0, g_cur_stream>>>((float *)out->ptr, (const float *)ls->ptr, (const float *)lt->ptr, n_vocab, T);
    return cuda_ok(cudaGetLastError(), "dk jury");
}

/* 批量取行(markov embed): out[r][d] = tab[ids[r]][d], 表 bf16(2)/f32(4) 由调用方按 GGUF 登记类型给 */
__global__ static void dk_rows_gather_kernel(float *out, const uint8_t *tab, const int32_t *ids, uint32_t dim, uint32_t is_f32, uint64_t n_rows_tab) {
    const uint32_t r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const int32_t id = ids[r];
    float v = 0.f;
    if (id >= 0 && (uint64_t)id < n_rows_tab) {
        if (is_f32) memcpy(&v, tab + ((uint64_t)id * dim + d) * 4u, 4);
        else { __nv_bfloat16 h; memcpy(&h, tab + ((uint64_t)id * dim + d) * 2u, 2); v = __bfloat162float(h); }
    }
    out[(uint64_t)r * dim + d] = v;
}
int ds4_gpu_draft_rows_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t tab_offset, uint64_t n_rows_tab,
                                     uint32_t dim, uint32_t elem_bytes, const ds4_gpu_tensor *ids, uint32_t n) {
    if (!out || !ids || !n || (elem_bytes != 2u && elem_bytes != 4u)) return 0;
    const uint64_t bytes = n_rows_tab * dim * elem_bytes;
    if (tab_offset > model_size || bytes > model_size - tab_offset) return 0;
    const uint8_t *tab = (const uint8_t *)cuda_model_range_ptr(model_map, tab_offset, bytes, "dk markov embed");
    if (!tab) return 0;
    dk_rows_gather_kernel<<<dim3((dim + 255u) / 256u, n), 256, 0, g_cur_stream>>>((float *)out->ptr, tab, (const int32_t *)ids->ptr, dim, elem_bytes == 4u, n_rows_tab);
    return cuda_ok(cudaGetLastError(), "dk rows gather");
}

/* ★总变差损失(2026-10-07 下午, 替换前向 KL 当训练目标)★: 投机采样的期望接受率恰是 Σmin(p,q) = 1 − TV(p,q), 而前向 KL 是"覆盖式"的 ——
 * 最小端到端探针实撞: 两份 CFO 文本训 1 轮, 留出 KL 1.887 → 1.849(降), 留出首位 Σmin 0.761 → 0.737(掉), 五位全掉。例: p=(.9,.1), q 从 (.99,.01) 改成 (.7,.3)
 * KL 0.144 → 0.116 降而 Σmin 0.91 → 0.80 掉 —— KL 奖励把 q 摊平去盖住 p 的支撑, 接受率吃的是重合。所以目标就按尺来: L = TV = ½Σ|p−q|。
 * 教师只给 top-K(K 到 4096, 位图判榜上/榜外)+ 余量 r: 榜外的 p_j 当 0(只知道总量 r, 不知道形状), 其 TV 贡献 ½(q_j + p_j) 里的 p 部分记常数 ½r。
 * 梯度(次梯度, 过温度 T 的 softmax): ∂TV/∂z_i = (1/T)·q_i·½·(s_i − Σ_j q_j s_j), s_j = sign(q_j − p_j)(榜外 +1; 相等 0)。
 * 一 block 一行 1024 线程; glogits 可与 logits 同址(榜外 pass 只改榜外格, 榜上 pass 按榜单改榜上格, 两者不相交)。 */
#define DK_TV_MASK_WORDS 4096u   /* 词表位图: 131072 个词以内(现役 129280) */
__global__ __launch_bounds__(1024) static void dk_tv_kernel(float *g, float *loss, const float *z, uint32_t V, const int32_t *tid, const float *tp,
                                                           const float *trest, uint32_t K, float T, float scale) {
    __shared__ uint32_t mask[DK_TV_MASK_WORDS];
    __shared__ double shd[32]; __shared__ float shf[32]; __shared__ uint32_t shu[32];
    const uint32_t row = blockIdx.x;
    const float *zr = z + (uint64_t)row * V; float *gr = g + (uint64_t)row * V;
    const int32_t *ti = tid + (uint64_t)row * K; const float *tq = tp + (uint64_t)row * K;
    for (uint32_t w = threadIdx.x; w < DK_TV_MASK_WORDS; w += blockDim.x) mask[w] = 0u;
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) { const int32_t id = ti[k]; if (id >= 0 && (uint32_t)id < V) atomicOr(&mask[(uint32_t)id >> 5], 1u << ((uint32_t)id & 31u)); }
    __syncthreads();
    float mx = -INFINITY; uint32_t dummy = 0;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, zr[j]);
    { float m2; uint32_t i2; dk_block_argmax(mx, threadIdx.x, shf, shu, &m2, &i2); mx = m2; (void)dummy; }
    double Z = 0.0;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) Z += isfinite(zr[j]) ? exp(((double)zr[j] - mx) / T) : 0.0;
    Z = dk_block_sum(Z, shd);
    double qoff = 0.0;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x)
        if (!((mask[j >> 5] >> (j & 31u)) & 1u)) qoff += isfinite(zr[j]) ? exp(((double)zr[j] - mx) / T) / Z : 0.0;
    qoff = dk_block_sum(qoff, shd);
    double tvon = 0.0, qson = 0.0;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const int32_t id = ti[k];
        if (id < 0 || (uint32_t)id >= V) continue;
        const double q = isfinite(zr[id]) ? exp(((double)zr[id] - mx) / T) / Z : 0.0, d = q - (double)tq[k];
        tvon += fabs(d); qson += d > 0.0 ? q : (d < 0.0 ? -q : 0.0);
    }
    tvon = dk_block_sum(tvon, shd); qson = dk_block_sum(qson, shd);
    const double A = qoff + qson;   /* Σ_j q_j s_j */
    const float c = scale / T;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) {
        if ((mask[j >> 5] >> (j & 31u)) & 1u) continue;   /* 榜上格留给下面按榜单写 */
        const double q = isfinite(zr[j]) ? exp(((double)zr[j] - mx) / T) / Z : 0.0;
        gr[j] = (float)(c * q * 0.5 * (1.0 - A));
    }
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const int32_t id = ti[k];
        if (id < 0 || (uint32_t)id >= V) continue;
        const double q = isfinite(zr[id]) ? exp(((double)zr[id] - mx) / T) / Z : 0.0, d = q - (double)tq[k], s = d > 0.0 ? 1.0 : (d < 0.0 ? -1.0 : 0.0);
        gr[id] = (float)(c * q * 0.5 * (s - A));
    }
    if (threadIdx.x == 0) loss[row] = (float)(0.5 * (tvon + qoff + (double)trest[row]));
}
int ds4_gpu_draft_tv_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t m, uint32_t n_vocab,
                            const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp, const ds4_gpu_tensor *trest, uint32_t k, float T, float scale) {
    if (!glogits || !loss || !logits || !tid || !tp || !trest || !m || !k || !(T > 0.f) || n_vocab > DK_TV_MASK_WORDS * 32u) return 0;
    dk_tv_kernel<<<m, 1024, 0, g_cur_stream>>>((float *)glogits->ptr, (float *)loss->ptr, (const float *)logits->ptr, n_vocab,
                                               (const int32_t *)tid->ptr, (const float *)tp->ptr, (const float *)trest->ptr, k, T, scale);
    return cuda_ok(cudaGetLastError(), "dk tv");
}

/* ---- markov 偏置表(训练变量 2, 10-07 下午): bias[n][V] = e[n][R]·Wᵀ, W = markov_head [V][R], e = markov_embd 的行(按"前一个真 token"取) ----
 * 为什么训它: 块内第 2~5 位吃的是 noise 嵌入, 它们的分布形状大半来自这张按前一个 token 查的偏置表; 它与上下文无关, 学到的是
 * "部署底座的下一词条件分布对 FP 的偏移", 跨请求可迁移。Sgemm 写法照 ds4_gpu_v41_amp_apply_tensor(行主序 ↔ 列主序转置的账见那里)。 */
int ds4_gpu_draft_bias_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *e, const ds4_gpu_tensor *W, uint32_t n, uint32_t R, uint32_t V) {
    if (!out || !e || !W || !g_cublas_ready || !n || !R || !V) return 0;
    /* 解码(草稿块逐位 n=1)走自家 f32 GEMV —— 原件那条路(ds4_gpu_v41_matmul_f32_tensor)就是它; cuBLAS 在 n=1 时拆成 200 多个小核,
     * 10-07 部署门实撞: 偏置表版贪心投机 45.09 → 42.93 t/s(每轮五发各慢 ~0.5 ms), 换回同一个核才兑现接受率的收益 */
    if (n <= V41_GEMV_MAX_TOK && (R % 128u) == 0u)
        return v41_f32_gemv((const float *)W->ptr, R, V, (const float *)e->ptr, (float *)out->ptr, n, "dk bias gemv");
    const float a1 = 1.0f, b0 = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    /* outᵀ(V×n) = W(V×R, 存成 [V][R] 行主序 = 列主序 R×V 的转置) · eᵀ(R×n) */
    return cublas_ok(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)V, (int)n, (int)R, &a1, (const float *)W->ptr, (int)R,
                                 (const float *)e->ptr, (int)R, &b0, (float *)out->ptr, (int)V), "dk bias e·Wᵀ");
}
/* 反向: gW[V][R] += gᵀ·e(累加, 跨批攒); ge[n][R] = g·W(覆盖) */
int ds4_gpu_draft_bias_bwd_tensor(ds4_gpu_tensor *gW, ds4_gpu_tensor *ge, const ds4_gpu_tensor *g, const ds4_gpu_tensor *e, const ds4_gpu_tensor *W,
                                  uint32_t n, uint32_t R, uint32_t V) {
    if (!gW || !ge || !g || !e || !W || !g_cublas_ready || !n || !R || !V) return 0;
    const float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    /* gWᵀ(R×V) += eᵀ(R×n) · g(n×V)(g 存 [n][V] 行主序 = 列主序 V×n 的转置 ⇒ OP_T) */
    if (!cublas_ok(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, (int)R, (int)V, (int)n, &a1, (const float *)e->ptr, (int)R,
                               (const float *)g->ptr, (int)V, &b1, (float *)gW->ptr, (int)R), "dk bias gW+=gᵀe")) return 0;
    /* geᵀ(R×n) = Wᵀ(R×V) · gᵀ(V×n) */
    return cublas_ok(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)n, (int)V, &a1, (const float *)W->ptr, (int)R,
                                 (const float *)g->ptr, (int)V, &b0, (float *)ge->ptr, (int)R), "dk bias ge=gW");
}
/* 设备表的取行 / 梯度按行散加(同一个 token 在一批里出现多次 ⇒ 原子加) */
__global__ static void dk_rows_dev_kernel(float *out, const float *tab, const int32_t *ids, uint32_t ids_off, uint32_t R, uint32_t Vm) {
    const uint32_t r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= R) return;
    const int32_t id = ids[ids_off + r];
    out[(uint64_t)r * R + d] = (id >= 0 && (uint32_t)id < Vm) ? tab[(uint64_t)id * R + d] : 0.f;
}
__global__ static void dk_rows_scatter_kernel(float *gtab, const float *g, const int32_t *ids, uint32_t R, uint32_t Vm) {
    const uint32_t r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= R) return;
    const int32_t id = ids[r];
    if (id >= 0 && (uint32_t)id < Vm) atomicAdd(gtab + (uint64_t)id * R + d, g[(uint64_t)r * R + d]);
}
int ds4_gpu_draft_rows_dev_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tab, const ds4_gpu_tensor *ids, uint32_t ids_off, uint32_t n, uint32_t R, uint32_t Vm) {
    if (!out || !tab || !ids || !n || !R) return 0;
    dk_rows_dev_kernel<<<dim3((R + 255u) / 256u, n), 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)tab->ptr, (const int32_t *)ids->ptr, ids_off, R, Vm);
    return cuda_ok(cudaGetLastError(), "dk rows dev");
}
int ds4_gpu_draft_rows_scatter_tensor(ds4_gpu_tensor *gtab, const ds4_gpu_tensor *g, const ds4_gpu_tensor *ids, uint32_t n, uint32_t R, uint32_t Vm) {
    if (!gtab || !g || !ids || !n || !R) return 0;
    dk_rows_scatter_kernel<<<dim3((R + 255u) / 256u, n), 256, 0, g_cur_stream>>>((float *)gtab->ptr, (const float *)g->ptr, (const int32_t *)ids->ptr, R, Vm);
    return cuda_ok(cudaGetLastError(), "dk rows scatter");
}

/* ★小批低秩件应用(10-07 傍晚)★: y[n][D] += x[n][D]·(B·A), n ≤ 8。ds4_gpu_v41_amp_apply_tensor 走两发 cuBLAS Sgemm, n=5 时每发 ~0.2 ms
 * (cuBLAS 小 n 拆成一串小核): 草稿一轮 3 塔 + 出口 = 8 发 ⇒ 草稿 8.4 → 10.0 ms(dkspeed 0908 分账), 把接受率涨出来的 0.15 token/轮吃光。
 * 两发小核: T[n][K] = x·Bᵀ(一 warp 一个 (行, k) 点积), y += T·A(一线程一个 (行, d), k 内循环)。累加序与 cuBLAS 不同, 不逐位同 ——
 * 只接草稿态(草稿只提议, 不受逐字节门约束), 训练(n=640)与主干 ③ 仍走 cuBLAS。 */
__global__ static void dk_lowrank_t_kernel(float *T, const float *x, const float *B, uint32_t D, uint32_t K, uint32_t nw) {
    const uint32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = gw / K, k = gw % K;
    if (gw >= nw) return;
    const float *xr = x + (uint64_t)r * D, *br = B + (uint64_t)k * D;
    float s = 0.f;
    for (uint32_t d = lane; d < D; d += 32u) s += xr[d] * br[d];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) T[(uint64_t)r * K + k] = s;
}
__global__ static void dk_lowrank_y_kernel(float *y, const float *T, const float *A, uint32_t D, uint32_t K) {
    const uint32_t r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float *tr = T + (uint64_t)r * K;
    float s = 0.f;
    for (uint32_t k = 0; k < K; k++) s += tr[k] * A[(uint64_t)k * D + d];
    y[(uint64_t)r * D + d] += s;
}
int ds4_gpu_draft_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                   ds4_gpu_tensor *T, uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!y || !x || !A || !B || !T || n_tok == 0 || K == 0) return 0;
    if (n_tok > V41_GEMV_MAX_TOK) return ds4_gpu_v41_amp_apply_tensor(y, x, A, B, T, n_tok, D, K);   /* 大批照旧 cuBLAS */
    const uint32_t nw = n_tok * K;   /* 一 warp 一个 (行, k) */
    dk_lowrank_t_kernel<<<(nw + 7u) / 8u, 256, 0, g_cur_stream>>>((float *)T->ptr, (const float *)x->ptr, (const float *)B->ptr, D, K, nw);
    if (!cuda_ok(cudaGetLastError(), "dk lowrank T")) return 0;
    dk_lowrank_y_kernel<<<dim3((D + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>((float *)y->ptr, (const float *)T->ptr, (const float *)A->ptr, D, K);
    return cuda_ok(cudaGetLastError(), "dk lowrank y");
}
