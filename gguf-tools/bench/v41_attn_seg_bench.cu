/* v41_attn_seg_bench.cu — V4.1 解码稀疏注意力(张量核 seg 核 + 合并核)的形态微基准(spark 本机跑; 2026-09-29 立)。
 *
 * 【为什么要它】09-29 逐核表: 投机验证批 n=4 时这两发 8.7 ms/步, n=1 只 2.0 —— 每多一行 2.2 ms, 而它不读多少字节
 * (grid.z 按行, 每行各自 gather、各自 mma 同一批键)。换核形态先在这里过(引擎一趟要装 113 GB 模型 + 几分钟), 过了才进引擎。
 *
 * 【比什么】V0 = 引擎现核(cuda_v41_attn_mma_decode.inc.cu / cuda_sparse_attn_mma.inc.cu 逐字抄本, 去掉 PDL);
 *   V1 = gather 里压缩行的 32 个缩放每行只解一次(lane i 解第 i 个, shfl 分发), 元素循环不再逐元素解 e4m3 —— 值一模一样;
 *   V2 = 一次 gather 32 个键(两个 16 键子块), 同步次数减半; 两遍照旧;
 *   V3 = 一次 gather 整段(≤64 键), S 片与键片留在 shared, 第二遍不再 gather/不再算 S(只做 P·V);
 *   V4 = V1 + fp4 查表走 shfl(16 个值 lane 各持一个, 引擎的 static const 表是 LDC 按 lane 下标重放)+ 乘完直接 cvt.rn;
 *   V5 = V3 + 同样的查表; V6 = 每层先一发预解核把 n 行的键解成 bf16, seg 核只搬(同一批键原来被 4 头组 × 2 遍解 8 次);
 *   V7 = V4 + 在线 max/sum 并行版 + 出口两轮; V8 = V7 但一 block 32 头; V9/V10 = 目标段数 12(诊断, 不逐位同引擎)。
 * 09-29 实测(12k 形状, ms/步 seg+merge, n=1/4/6): V0 2.4/7.8/10.6 → V4 1.5/4.8/7.1 → V7 1.5/4.5/6.9(引擎取 V7);
 *   V6 5.1(解码已不是大头) / V8 5.7(59 KB shared ⇒ 每 SM 只挂 1 个 block; 板子 shared/SM 100 KB) / V10 1.95/4.3/5.6(段减半:
 *   n=1 填不满 48 个 SM 反而慢, 且分段变了不逐位同 —— 段数按位置定是"投机 == 纯解码"的前提, 不能随 n 变)。合并核占 ~0.4~0.5。
 * ★判据(缺一不可)★: ①各变体输出与 V0 逐位同 ②n 行批里第 i 行 == 只拿第 i 行在它自己位置跑 n=1 的结果, 逐位同 ——
 *   这就是引擎"投机 == 纯解码逐字节"那条门在核层面的样子。哪一条不同 = 分段/子块序/累加序被改了, 不许进引擎。
 * 逐位同的构造依据: S[头][键] 每个元素 = 8 个 warp 各 4 个 k 步的 mma 部分和按固定序相加, 与它落在哪个 16 键列块无关;
 *   在线 max/sum 仍按 16 键子块、按键序串行累加; P·V 仍按 16 键子块顺序 mma; 合并核按段号固定序。
 * 形状 = 12k 真场景: 64 头 × 512, 窗口 128, indexer top-k 512, 压缩比 4 ⇒ 每行 640 键、段长 32、20 段。
 * 用法: nvcc -O3 -arch=native -o v41_attn_seg_bench v41_attn_seg_bench.cu && ./v41_attn_seg_bench [层数=40] [遍数=5] [位置=12000] [只计时变体=-1] [只计时 n=0]
 *   后两个参数给 ncu 用(只发一种核一种 n, -k 正则才对得上): 例 `ncu -k regex:attn_seg_v_kernel --launch-skip 40 --launch-count 40 ./v41_attn_seg_bench 40 1 12000 4 4` */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "../../src/common/ds4_fp8.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "v41_attn_seg_bench_kernels.cuh"

/* ---- 变体 ---- */
/* V1 的 gather: 压缩行的 32 个缩放 lane i 解第 i 个, 元素循环 shfl 取 —— 同一个 scale 同一个 nibble 同一次乘法, 值逐位同引擎。
 * 一次搬 GT 个键(GT/8 个一 warp)。 */
template <uint32_t GT, int DEC>
__device__ __forceinline__ static void gather_v1(__nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                 uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
                                                 uint32_t window, uint32_t ng, uint32_t topk) {
    const uint32_t lane = threadIdx.x & 31u;
    /* DEC=1: fp4 的 16 个值 lane l 各持一个(表在寄存器里), 查表 = 一条 shfl; 引擎版是 static const 表 = LDC 按 lane 下标重放 */
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));
    for (uint32_t t = threadIdx.x / 32u; t < GT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;
        if (t < nt) {
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_MMA_HD;
        if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
        else if (cpk) {
            const float sc = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);   /* 32 个缩放 = 32 个 lane */
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ATTN_MMA_HD / 32u; j++) {
                const uint32_t d = lane + 32u * j;
                const uint8_t by = cpk[d >> 1];
                const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
                const float s = __shfl_sync(0xffffffffu, sc, (int)(d >> 4));
                if (DEC) kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int)nib) * s);   /* cvt.rn 本身就是 RNE, 与 v41_bf16r 同值 */
                else     kt[d] = __float2bfloat16(v41_bf16r(ds4_fp4_nibble_to_f32(nib) * s));
            }
        } else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}
/* V6: 每层先一发预解核把 n 行各自的键清单解成 bf16(每 (行, 键) 只解一次 —— seg 核里同一批键被 4 个头组 × 2 遍解 8 次),
 * seg 核的 gather 只按 16 B 整块搬。kbuf[(i·nkmax + kk)·512] bf16, kvalid[i·nkmax + kk]; 值与 V4 逐位同(同一条解码式)。 */
__global__ static void attn_prep_kernel(__nv_bfloat16 *kbuf, int *kvalid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                        uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk, uint32_t nkmax) {
    const uint32_t lane = threadIdx.x & 31u, i = blockIdx.y, kk = blockIdx.x * (blockDim.x / 32u) + (threadIdx.x >> 5);
    const uint32_t p = pos0 + i, lo = p + 1u > window ? p + 1u - window : 0u, nwin = p - lo + 1u, nkeys = nwin + topk;
    if (kk >= nkeys) return;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));
    const float *krow = NULL; const uint8_t *cpk = NULL;
    if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * DS4_ATTN_MMA_HD;
    else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)]; if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
    __nv_bfloat16 *kt = kbuf + ((size_t)i * nkmax + kk) * DS4_ATTN_MMA_HD;
    if (lane == 0) kvalid[(size_t)i * nkmax + kk] = (krow || cpk) ? 1 : 0;
    if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
    else if (cpk) {
        const float sc = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ATTN_MMA_HD / 32u; j++) {
            const uint32_t d = lane + 32u * j;
            const uint8_t by = cpk[d >> 1];
            const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
            const float s = __shfl_sync(0xffffffffu, sc, (int)(d >> 4));
            kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int)nib) * s);
        }
    } else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
}
template <uint32_t GT>
__device__ __forceinline__ static void gather_pre(__nv_bfloat16 *ks, int *valid, const __nv_bfloat16 *kbuf, const int *kvalid,
                                                  uint32_t i, uint32_t base, uint32_t nt, uint32_t nkmax) {
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t t = threadIdx.x / 32u; t < GT; t += blockDim.x / 32u) {
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_MMA_HD;
        if (t < nt) {
            const size_t row = (size_t)i * nkmax + base + t;
            if (lane == 0) valid[t] = kvalid[row];
            const uint4 *src = (const uint4 *)(kbuf + row * DS4_ATTN_MMA_HD);   /* 一行 1 KB = 64 个 uint4, 每 lane 两个 */
            ((uint4 *)kt)[lane] = src[lane]; ((uint4 *)kt)[lane + 32u] = src[lane + 32u];
        } else { if (lane == 0) valid[t] = 0; for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}
/* GT = 一次 gather 几个键(16/32/64); KEEP = 整段一次搬进来, 第二遍复用键片与 S 片(要求 GT ≥ 段长上限 64)。
 * shared: qs 16 KB + ks GT KB + spart 8 KB + stile 16×GT×4 + ptile 0.5 KB + rmax/rsum + valid[GT] */
template <uint32_t GT, int KEEP, int DEC>
__global__ static void attn_seg_v_kernel(float *pacc, float *pmax, float *psum, const float *q, const float *kvw,
                                         const uint8_t *kvc, const int32_t *idx, uint32_t pos0, uint32_t window,
                                         uint32_t ng, uint32_t topk, uint32_t n_head, float scale, uint32_t ratio, uint32_t nseg,
                                         const __nv_bfloat16 *kbuf, const int *kvalid, uint32_t nkmax) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;
    float *spart = (float *)(ks + GT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;                       /* [16][GT] */
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + DS4_ATTN_MMA_HEADS * GT);
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);
    const uint32_t seg = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS, i = blockIdx.z;
    const uint64_t pbase = ((uint64_t)i * nseg + seg) * n_head + h0;
    const uint32_t p = pos0 + i;
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    const uint32_t seg_keys = v41_attn_seg_keys(p, window, ratio, topk);
    const uint32_t k0 = seg * seg_keys;
    if (k0 >= nkeys) {
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
    /* 第一遍: 按 GT 键一批 gather, 批内按 16 键子块算 S 与在线 max/sum(子块序 = 引擎的 tile 序) */
    for (uint32_t base = k0; base < k1; base += GT) {
        const uint32_t nt = (k1 - base) < GT ? (k1 - base) : GT;
        __syncthreads();
        if (DEC == 2) gather_pre<GT>(ks, valid, kbuf, kvalid, i, base, nt, nkmax);
        else gather_v1<GT, DEC>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        for (uint32_t c = 0; c * 16u < nt; c++) {
            const uint32_t ntc = (nt - c * 16u) < 16u ? (nt - c * 16u) : 16u;
            ds4_attn_mma_scores<GT>(stile + c * 16u, spart, qs, ks + (size_t)c * 16u * DS4_ATTN_MMA_HD, valid + c * 16u, ntc, scale);
            if (threadIdx.x < DS4_ATTN_MMA_HEADS) attn_stats16(stile + threadIdx.x * GT + c * 16u, rmax, rsum);
        }
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = k0; base < k1; base += GT) {
        const uint32_t nt = (k1 - base) < GT ? (k1 - base) : GT;
        if (!KEEP) {   /* 两遍: 重 gather + 重算 S(与引擎同) */
            __syncthreads();
            if (DEC == 2) gather_pre<GT>(ks, valid, kbuf, kvalid, i, base, nt, nkmax);
            else gather_v1<GT, DEC>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
            __syncthreads();
            for (uint32_t c = 0; c * 16u < nt; c++) {
                const uint32_t ntc = (nt - c * 16u) < 16u ? (nt - c * 16u) : 16u;
                ds4_attn_mma_scores<GT>(stile + c * 16u, spart, qs, ks + (size_t)c * 16u * DS4_ATTN_MMA_HD, valid + c * 16u, ntc, scale);
            }
        }
        for (uint32_t c = 0; c * 16u < nt; c++) {
            for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x)
                ptile[e] = __float2bfloat16(expf(stile[(e >> 4) * GT + c * 16u + (e & 15u)] - rmax[e >> 4]));
            __syncthreads();
            attn_pv16(oacc, ptile, ks + (size_t)c * 16u * DS4_ATTN_MMA_HD);
            __syncthreads();   /* ptile 下一子块要覆盖 */
        }
    }
    attn_epilogue(oacc, spart, pacc, pmax, psum, rmax, rsum, pbase);
}

/* V7~V10: HB = 一 block 管几个头(16 = 现役; 32 = 两个 M 块共用同一次 gather 与同一批同步, block 数减半)。同时两处同值改法:
 *   ①在线 max/sum 并行版: 16 个 lane 各算一个 expf, 再按键序 shfl 串行相加 —— 加法序与逐字版一样, 值逐位同;
 *   ②出口两轮各存两片(借 ks 那 16 KB, 最后一次 P·V 之后它就空了), 原来四轮各一片。
 * tseg = 目标段数(引擎 24; 12 = 诊断"块数减半"值多少 —— 分段变了 ⇒ 与 V0 不同, 但批内各行仍须 == 单行自跑)。 */
__host__ __device__ __forceinline__ static uint32_t seg_keys_t(uint32_t p, uint32_t window, uint32_t ratio, uint32_t topk, uint32_t tseg) {
    const uint32_t nwin = p + 1u > window ? window : p + 1u;
    uint32_t tref = 0;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; tref = topk < vis ? topk : vis; }
    uint32_t seg = (nwin + tref + tseg - 1u) / tseg;
    seg = ((seg + DS4_ATTN_MMA_KT - 1u) / DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_KT;
    if (seg < DS4_ATTN_MMA_KT) seg = DS4_ATTN_MMA_KT;
    return seg > 64u ? 64u : seg;
}
__host__ __device__ __forceinline__ static uint32_t nseg_at_t(uint32_t p, uint32_t window, uint32_t ratio, uint32_t topk, uint32_t tseg) {
    const uint32_t nw = p + 1u > window ? window : p + 1u;
    uint32_t tk = topk;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; if (vis < tk) tk = vis; }
    const uint32_t sk = seg_keys_t(p, window, ratio, topk, tseg);
    return (nw + tk + sk - 1u) / sk;
}
/* 并行版在线 max/sum: warp w 管头 2w(lane 0..15)与 2w+1(lane 16..31), 每 lane 一个键; HB > 16 时每 warp 再管 +16 的头。
 * max 是精确运算(序无关); 和 = 先 rsum·expf(m−tm), 再按 k = 0..15 顺序加 expf(s_k−tm) —— 与逐字版同一序同一值。 */
template <uint32_t HB>
__device__ __forceinline__ static void attn_stats_par(const float *stile, float *rmax, float *rsum) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, k = lane & 15u;
    for (uint32_t h = warp * 2u + (lane >> 4); h < HB; h += 16u) {
        const float s = stile[h * 16u + k], m = rmax[h];
        float tm = fmaxf(m, s);
        tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 8)); tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
        tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2)); tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
        const float e = expf(s - tm);
        float sm = rsum[h] * expf(m - tm);
        const int b = (int)(lane & 16u);
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) sm += __shfl_sync(0xffffffffu, e, b + kk);
        if (k == 0u) { rmax[h] = tm; rsum[h] = sm; }
    }
}
template <uint32_t HB>
__global__ static void attn_seg_hb_kernel(float *pacc, float *pmax, float *psum, const float *q, const float *kvw, const uint8_t *kvc,
                                          const int32_t *idx, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head,
                                          float scale, uint32_t ratio, uint32_t nseg, uint32_t tseg) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;                       /* [HB][512] */
    __nv_bfloat16 *ks = qs + HB * DS4_ATTN_MMA_HD;                                  /* [16][512] */
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;                               /* [HB][16] */
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + HB * 16u);                     /* [16][16] */
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + HB;
    int *valid = (int *)(rsum + HB);
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t seg = blockIdx.x, h0 = blockIdx.y * HB, i = blockIdx.z;
    const uint64_t pbase = ((uint64_t)i * nseg + seg) * n_head + h0;
    const uint32_t p = pos0 + i;
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    const uint32_t seg_keys = seg_keys_t(p, window, ratio, topk, tseg);
    const uint32_t k0 = seg * seg_keys;
    if (k0 >= nkeys) {
        for (uint32_t e = threadIdx.x; e < HB * DS4_ATTN_MMA_HD; e += blockDim.x)
            pacc[(pbase + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD] = 0.f;
        if (threadIdx.x < HB) { pmax[pbase + threadIdx.x] = -1e30f; psum[pbase + threadIdx.x] = 0.f; }
        return;
    }
    const uint32_t k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t e = threadIdx.x; e < HB * DS4_ATTN_MMA_HD; e += blockDim.x)
        qs[e] = __float2bfloat16(q[((uint64_t)i * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < HB) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    __syncthreads();
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        gather_v1<DS4_ATTN_MMA_KT, 1>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        for (uint32_t m = 0; m < HB / 16u; m++)
            ds4_attn_mma_scores<16u>(stile + m * 256u, spart, qs + (size_t)m * 16u * DS4_ATTN_MMA_HD, ks, valid, nt, scale);
        attn_stats_par<HB>(stile, rmax, rsum);
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[HB / 16u][4];
    for (uint32_t m = 0; m < HB / 16u; m++) for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[m][j], 0.0f);
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        gather_v1<DS4_ATTN_MMA_KT, 1>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        for (uint32_t m = 0; m < HB / 16u; m++)
            ds4_attn_mma_scores<16u>(stile + m * 256u, spart, qs + (size_t)m * 16u * DS4_ATTN_MMA_HD, ks, valid, nt, scale);
        for (uint32_t m = 0; m < HB / 16u; m++) {
            for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x)
                ptile[e] = __float2bfloat16(expf(stile[m * 256u + e] - rmax[m * 16u + (e >> 4)]));
            __syncthreads();
            attn_pv16(oacc[m], ptile, ks);
            __syncthreads();
        }
    }
    float *otile = (float *)ks;   /* 出口: 借 ks(16 KB = 8 warp × 2 片), 两轮存完 4 片 */
    for (uint32_t m = 0; m < HB / 16u; m++) {
        for (int r = 0; r < 2; r++) {
            __syncthreads();
            wmma::store_matrix_sync(otile + (size_t)warp * 512u, oacc[m][2 * r], 16, wmma::mem_row_major);
            wmma::store_matrix_sync(otile + (size_t)warp * 512u + 256u, oacc[m][2 * r + 1], 16, wmma::mem_row_major);
            __syncthreads();
            for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_WARPS * 512u; e += blockDim.x) {
                const uint32_t w = e >> 9, rr = e & 511u, jj = rr >> 8, t = rr & 255u, h = t >> 4, d16 = t & 15u;
                pacc[(pbase + m * 16u + h) * DS4_ATTN_MMA_HD + w * 64u + (uint32_t)(2 * r) * 16u + jj * 16u + d16] = otile[e];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < HB) { pmax[pbase + threadIdx.x] = rmax[threadIdx.x]; psum[pbase + threadIdx.x] = rsum[threadIdx.x]; }
}

/* ---- 数据: 按 (层, 位置/组, 维) 哈希生成, 同一位置在任何排法里值相同(单行/多行对拍要靠这个) ---- */
static uint32_t h32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u ^ (d * 0x27D4EB2Fu);
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15; return x;
}
static float hf(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return ((float)(h32(a, b, c, d) >> 8) / 16777216.0f) * 2.0f - 1.0f; }

typedef struct { float *q, *kvw, *sink, *o; uint8_t *kvc; int32_t *idx; } layer_bufs;

/* 窗口环: 位置 a 在 v41_win_row(a, pos0, window, 1) 那一格(a ≥ pos0 = 本批行, 在 window + a − pos0) */
static void fill_kvw(float *dst, uint32_t L, uint32_t pos0, uint32_t n, uint32_t window) {
    for (int64_t a = (int64_t)pos0 - window; a < (int64_t)(pos0 + n); a++) {
        const uint64_t r = a >= (int64_t)pos0 ? (uint64_t)(window + a - pos0) : (uint64_t)(a % window);
        for (uint32_t d = 0; d < 512u; d++) dst[r * 512u + d] = hf(L, 1u, (uint32_t)a, d);
    }
}
/* top-k: 位置 p 的清单 = 可见组里按哈希挑 topk 个, 升序(引擎 topk 核的写法); 相邻位置清单高度重叠(同一哈希门槛) */
static void fill_idx(int32_t *dst, uint32_t L, uint32_t p, uint32_t ratio, uint32_t topk) {
    const uint32_t vis = (p + 1u) / ratio;
    /* 门槛: 挑 h < thr 的组; 位置每前进 1, 换掉 ~3% 的名额(用位置的高位扰动), 其余保持 ⇒ 相邻行清单重叠 ~97% */
    uint32_t cnt = 0;
    for (uint32_t g = 0; g < vis && cnt < topk; g++) {
        const uint32_t hv = h32(L, 2u, g, 0u) % 1000u, jit = h32(L, 3u, g, p / 4u) % 1000u;
        const uint32_t score = (hv * 97u + jit * 3u) / 100u;
        if (score < (topk * 1000u) / vis + 20u) dst[cnt++] = (int32_t)g;
    }
    for (uint32_t g = vis; cnt < topk && g > 0; g--) { bool dup = false; for (uint32_t k = 0; k < cnt; k++) if (dst[k] == (int32_t)(g - 1)) { dup = true; break; }
                                                     if (!dup) dst[cnt++] = (int32_t)(g - 1); }
    /* 补齐后重新升序 */
    for (uint32_t a = 1; a < cnt; a++) { int32_t v = dst[a]; uint32_t b = a; while (b > 0 && dst[b - 1] > v) { dst[b] = dst[b - 1]; b--; } dst[b] = v; }
    for (uint32_t k = cnt; k < topk; k++) dst[k] = -1;
}

int main(int argc, char **argv) {
    const uint32_t L = argc > 1 ? (uint32_t)atoi(argv[1]) : 40u, iters = argc > 2 ? (uint32_t)atoi(argv[2]) : 5u;
    const uint32_t pos0 = argc > 3 ? (uint32_t)atoi(argv[3]) : 12000u;
    const int only_v = argc > 4 ? atoi(argv[4]) : -1; const uint32_t only_n = argc > 5 ? (uint32_t)atoi(argv[5]) : 0u;
    const int no_merge = argc > 6 ? atoi(argv[6]) : 0;   /* 1 = 计时只发 seg 核(看合并核占几成); 逐位门仍带合并 */
    bool timing = false;
    { cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
      printf("板子: %s, SM %d, shared/SM %zu KB, 每 block 可批 %zu KB, 寄存器/SM %d, L2 %d MB\n", pr.name, pr.multiProcessorCount,
             pr.sharedMemPerMultiprocessor >> 10, pr.sharedMemPerBlockOptin >> 10, pr.regsPerMultiprocessor, pr.l2CacheSize >> 20); }
    const uint32_t window = 128u, ratio = 4u, topk = 512u, NH = 64u, HD = 512u, NMAX = 8u;
    const float scale = 0.044194174f;
    const uint32_t ng = (pos0 + NMAX) / ratio;
    /* 段数上限: 批里各行的最大值 */
    uint32_t nseg_max = 1u;
    for (uint32_t i = 0; i < NMAX; i++) { const uint32_t ns = v41_attn_nseg_at(pos0 + i, window, ratio, topk); if (ns > nseg_max) nseg_max = ns; }
    printf("形状: pos0 %u, 窗口 %u, top-k %u, 压缩比 %u(ng %u), 段长 %u, 段数 %u, 层 %u, 遍 %u\n", pos0, window, topk, ratio, ng,
           v41_attn_seg_keys(pos0, window, ratio, topk), nseg_max, L, iters);
    layer_bufs *lb = (layer_bufs *)calloc(L, sizeof(layer_bufs));
    float *hq = (float *)malloc((size_t)NMAX * NH * HD * 4), *hk = (float *)malloc((size_t)(window + NMAX) * HD * 4), hs[64];
    uint8_t *hc = (uint8_t *)malloc((size_t)ng * DS4_V41_CKV_BYTES);
    int32_t *hi = (int32_t *)malloc((size_t)NMAX * topk * 4);
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t i = 0; i < NMAX; i++) for (uint32_t e = 0; e < NH * HD; e++) hq[(size_t)i * NH * HD + e] = hf(l, 4u, pos0 + i, e) * 2.0f;
        fill_kvw(hk, l, pos0, NMAX, window);
        for (uint32_t g = 0; g < ng; g++) {
            uint8_t *row = hc + (size_t)g * DS4_V41_CKV_BYTES;
            for (uint32_t b = 0; b < DS4_V41_CKV_NIB; b++) row[b] = (uint8_t)(h32(l, 5u, g, b) & 0xFFu);
            for (uint32_t s = 0; s < 32u; s++) row[DS4_V41_CKV_NIB + s] = (uint8_t)(0x30u + (h32(l, 6u, g, s) % 12u));   /* e4m3 0.25..~2, 无 NaN */
        }
        for (uint32_t i = 0; i < NMAX; i++) fill_idx(hi + (size_t)i * topk, l, pos0 + i, ratio, topk);
        for (uint32_t h = 0; h < NH; h++) hs[h] = hf(l, 7u, h, 0u);
        CK(cudaMalloc(&lb[l].q, (size_t)NMAX * NH * HD * 4));   CK(cudaMemcpy(lb[l].q, hq, (size_t)NMAX * NH * HD * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&lb[l].kvw, (size_t)(window + NMAX) * HD * 4)); CK(cudaMemcpy(lb[l].kvw, hk, (size_t)(window + NMAX) * HD * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&lb[l].kvc, (size_t)ng * DS4_V41_CKV_BYTES)); CK(cudaMemcpy(lb[l].kvc, hc, (size_t)ng * DS4_V41_CKV_BYTES, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&lb[l].idx, (size_t)NMAX * topk * 4));       CK(cudaMemcpy(lb[l].idx, hi, (size_t)NMAX * topk * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&lb[l].sink, 64 * 4));                       CK(cudaMemcpy(lb[l].sink, hs, 64 * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&lb[l].o, (size_t)NMAX * NH * HD * 4));
    }
    float *pacc, *pmax, *psum;
    CK(cudaMalloc(&pacc, (size_t)nseg_max * NMAX * NH * HD * 4)); CK(cudaMalloc(&pmax, (size_t)nseg_max * NMAX * NH * 4)); CK(cudaMalloc(&psum, (size_t)nseg_max * NMAX * NH * 4));
    const size_t smem0 = (size_t)(16u + 16u) * HD * 2 + 8u * 256u * 4 + 256u * 4 + 256u * 2 + 2u * 16u * 4 + 16u * 4;
    auto smem_v = [&](uint32_t GT) { return (size_t)(16u + GT) * HD * 2 + 8u * 256u * 4 + (size_t)16u * GT * 4 + 256u * 2 + 2u * 16u * 4 + (size_t)GT * 4; };
    CK(cudaFuncSetAttribute(v41_attn_mma_seg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem0));
    CK(cudaFuncSetAttribute(attn_seg_v_kernel<16u, 0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_v(16u)));
    CK(cudaFuncSetAttribute(attn_seg_v_kernel<32u, 0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_v(32u)));
    const cudaError_t e64 = cudaFuncSetAttribute(attn_seg_v_kernel<64u, 1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_v(64u));
    if (e64 != cudaSuccess) { printf("V3(64 键整段) shared %zu KB 抬不上去: %s\n", smem_v(64u) >> 10, cudaGetErrorString(e64)); (void)cudaGetLastError(); }
    CK(cudaFuncSetAttribute(attn_seg_v_kernel<16u, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_v(16u)));
    CK(cudaFuncSetAttribute(attn_seg_v_kernel<64u, 1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_v(64u)));
    CK(cudaFuncSetAttribute(attn_seg_v_kernel<16u, 0, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_v(16u)));
    const uint32_t nkmax = window + topk;   /* 每行键数上限(窗口满 + top-k 满) */
    __nv_bfloat16 *kbuf; int *kvalid;
    CK(cudaMalloc(&kbuf, (size_t)NMAX * nkmax * HD * 2)); CK(cudaMalloc(&kvalid, (size_t)NMAX * nkmax * 4));
    auto smem_hb = [&](uint32_t HB) { return (size_t)(HB + 16u) * HD * 2 + 8u * 256u * 4 + (size_t)HB * 16u * 4 + 256u * 2 + 2u * HB * 4 + 16u * 4; };
    CK(cudaFuncSetAttribute(attn_seg_hb_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_hb(16u)));
    CK(cudaFuncSetAttribute(attn_seg_hb_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_hb(32u)));
    const char *vname[11] = { "V0 引擎现核", "V1 缩放一次解", "V2 一次 32 键", "V3 整段留片", "V4 V1+shfl查表", "V5 V3+shfl查表", "V6 预解一发+搬",
                              "V7 16头+并行和", "V8 32头+并行和", "V9 32头 段12", "V10 16头 段12" };
    const uint32_t vtseg[11] = { 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u, 12u, 12u };   /* 段 12 的两个是诊断, 与 V0 不同是预期 */
    /* 发一层: v = 变体; n 行, pos0 起; 局部件按 nseg 排 */
    auto run = [&](int v, const layer_bufs &b, uint32_t n, uint32_t p0, const float *q, const float *kvw, const int32_t *idx, float *o) {
        uint32_t nseg = 1u; const uint32_t tseg = vtseg[v];
        for (uint32_t i = 0; i < n; i++) { const uint32_t ns = nseg_at_t(p0 + i, window, ratio, topk, tseg); if (ns > nseg) nseg = ns; }
        const dim3 g(nseg, NH / 16u, n), g32(nseg, NH / 32u, n);
        switch (v) {
        case 0: v41_attn_mma_seg_kernel<<<g, 256, smem0>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg); break;
        case 1: attn_seg_v_kernel<16u, 0, 0><<<g, 256, smem_v(16u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, kbuf, kvalid, nkmax); break;
        case 2: attn_seg_v_kernel<32u, 0, 0><<<g, 256, smem_v(32u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, kbuf, kvalid, nkmax); break;
        case 3: attn_seg_v_kernel<64u, 1, 0><<<g, 256, smem_v(64u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, kbuf, kvalid, nkmax); break;
        case 4: attn_seg_v_kernel<16u, 0, 1><<<g, 256, smem_v(16u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, kbuf, kvalid, nkmax); break;
        case 5: attn_seg_v_kernel<64u, 1, 1><<<g, 256, smem_v(64u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, kbuf, kvalid, nkmax); break;
        case 6:
            attn_prep_kernel<<<dim3((nkmax + 7u) / 8u, n), 256>>>(kbuf, kvalid, kvw, b.kvc, idx, p0, window, ng, topk, nkmax);
            attn_seg_v_kernel<16u, 0, 2><<<g, 256, smem_v(16u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, kbuf, kvalid, nkmax); break;
        case 7: case 10: attn_seg_hb_kernel<16u><<<g, 256, smem_hb(16u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, tseg); break;
        default:         attn_seg_hb_kernel<32u><<<g32, 256, smem_hb(32u)>>>(pacc, pmax, psum, q, kvw, b.kvc, idx, p0, window, ng, topk, NH, scale, ratio, nseg, tseg); break;
        }
        if (!(timing && no_merge)) v41_sparse_attn_merge_kernel<<<dim3(NH, n), 256>>>(o, pacc, pmax, psum, b.sink, nseg, NH, HD);
    };
    const uint32_t ns_list[4] = { 1u, 2u, 4u, 6u };
    if (e64 != cudaSuccess) return 2;   /* 整段变体抬不上 shared: 这台板子不在讨论范围 */
    const int nv = 11;
    float *ref = (float *)malloc((size_t)NMAX * NH * HD * 4), *got = (float *)malloc((size_t)NMAX * NH * HD * 4);
    /* ① 逐位门(层 0): 变体 vs V0; n 行批第 i 行 vs 单行在自己位置(单行的窗口环按 pos0 = p_i 重排, 值同一位置同) */
    float *kvw1; CK(cudaMalloc(&kvw1, (size_t)(window + 1u) * HD * 4));
    int bad = 0;
    float *got1 = (float *)malloc((size_t)NH * HD * 4);
    for (uint32_t ni = 0; ni < 4; ni++) {
        const uint32_t n = ns_list[ni];
        run(0, lb[0], n, pos0, lb[0].q, lb[0].kvw, lb[0].idx, lb[0].o); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ref, lb[0].o, (size_t)n * NH * HD * 4, cudaMemcpyDeviceToHost));
        for (int v = 0; v < nv; v++) {
            CK(cudaMemset(lb[0].o, 0xFF, (size_t)n * NH * HD * 4));
            run(v, lb[0], n, pos0, lb[0].q, lb[0].kvw, lb[0].idx, lb[0].o); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got, lb[0].o, (size_t)n * NH * HD * 4, cudaMemcpyDeviceToHost));
            if (v) {
                const int same = memcmp(ref, got, (size_t)n * NH * HD * 4) == 0;
                if (vtseg[v] == 24u) { printf("  n=%u %-15s vs V0: %s\n", n, vname[v], same ? "逐位同 ✓" : "★不同★"); bad |= !same; }
                else printf("  n=%u %-15s vs V0: %s(段 12 诊断, 分段不同是预期)\n", n, vname[v], same ? "同" : "不同");
            }
            for (uint32_t i = 0; i < n; i++) {   /* 单行对拍: 同一变体只拿第 i 行的 q/idx, 窗口按 p_i 重排 */
                fill_kvw(hk, 0u, pos0 + i, 1u, window);
                CK(cudaMemcpy(kvw1, hk, (size_t)(window + 1u) * HD * 4, cudaMemcpyHostToDevice));
                run(v, lb[0], 1u, pos0 + i, lb[0].q + (size_t)i * NH * HD, kvw1, lb[0].idx + (size_t)i * topk, lb[0].o); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(got1, lb[0].o, (size_t)NH * HD * 4, cudaMemcpyDeviceToHost));
                if (memcmp(got + (size_t)i * NH * HD, got1, (size_t)NH * HD * 4) != 0) { printf("  ★%s n=%u 第 %u 行 ≠ 单行自跑 @%u★\n", vname[v], n, i, pos0 + i); bad = 1; }
            }
        }
        printf("  n=%u 各变体各行 == 单行自跑: %s\n", n, "见上(无★即全同)");
    }
    /* ② 计时: L 层各发一次 seg + merge = 一步 */
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    timing = true;
    printf("\n%-15s", no_merge ? "ms/步(只 seg)" : "ms/步(L 层)"); for (uint32_t ni = 0; ni < 4; ni++) printf("   n=%u", ns_list[ni]); printf("\n");
    for (int v = 0; v < nv; v++) {
        if (only_v >= 0 && v != only_v) continue;
        printf("%-15s", vname[v]);
        for (uint32_t ni = 0; ni < 4; ni++) {
            const uint32_t n = ns_list[ni];
            if (only_n && n != only_n) { printf("      -"); continue; }
            for (uint32_t l = 0; l < L; l++) run(v, lb[l], n, pos0, lb[l].q, lb[l].kvw, lb[l].idx, lb[l].o);   /* 暖身 */
            CK(cudaDeviceSynchronize());
            float best = 1e30f;
            for (uint32_t it = 0; it < iters; it++) {
                CK(cudaEventRecord(t0));
                for (uint32_t l = 0; l < L; l++) run(v, lb[l], n, pos0, lb[l].q, lb[l].kvw, lb[l].idx, lb[l].o);
                CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                float ms = 0.f; CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < best) best = ms;
            }
            printf(" %6.2f", best);
        }
        printf("\n");
    }
    printf("\n%s\n", bad ? "★★逐位门有红, 上面的数字不作数★★" : "逐位门全绿(变体 == V0, n 行批各行 == 单行自跑)");
    return bad ? 1 : 0;
}
