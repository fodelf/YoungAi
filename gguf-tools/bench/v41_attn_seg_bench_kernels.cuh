/* v41_attn_seg_bench_kernels.cuh — v41_attn_seg_bench.cu 的引擎同款部分(逐字抄本, 去掉 PDL): 常量、解码原语、分段公式、
 * gather / S 片 / 在线 max-sum / P·V / 出口 四个积木, V0 seg 核, 合并核。只被 v41_attn_seg_bench.cu include(拆出来是守 500 行)。 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "../../src/common/ds4_fp8.h"

/* ---- 引擎同款(逐字; ds4_gpu_v41.h / cuda_v41_1 / cuda_kv_pack / cuda_v41_attn_split / cuda_sparse_attn_mma) ---- */
#define DS4_V41_CKV_NIB   256u
#define DS4_V41_CKV_BYTES 288u
#define DS4_ATTN_MMA_HEADS 16u
#define DS4_ATTN_MMA_KT    16u
#define DS4_ATTN_MMA_WARPS 8u
#define DS4_ATTN_MMA_HD    512u
#define V41_ATTN_MMA_DEC_TARGET_SEG 24u
#define V41_ATTN_SPLIT_MAX_SEG 64u

__device__ __forceinline__ static float v41_bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}
__device__ __forceinline__ static uint64_t v41_win_row(int64_t a, uint32_t pos0, uint32_t window, uint32_t ring) {
    if (a >= (int64_t)pos0) return (uint64_t)((int64_t)window + a - (int64_t)pos0);
    return ring ? (uint64_t)(a % (int64_t)window) : (uint64_t)(a - ((int64_t)pos0 - (int64_t)window));
}
__device__ __forceinline__ static float v41_ckv_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
__host__ __device__ __forceinline__ static uint32_t v41_attn_seg_keys(uint32_t p, uint32_t window, uint32_t ratio, uint32_t topk) {
    const uint32_t nwin = p + 1u > window ? window : p + 1u;
    uint32_t tref = 0;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; tref = topk < vis ? topk : vis; }
    uint32_t seg = (nwin + tref + V41_ATTN_MMA_DEC_TARGET_SEG - 1u) / V41_ATTN_MMA_DEC_TARGET_SEG;
    seg = ((seg + DS4_ATTN_MMA_KT - 1u) / DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_KT;
    if (seg < DS4_ATTN_MMA_KT) seg = DS4_ATTN_MMA_KT;
    return seg > 64u ? 64u : seg;
}
__host__ __device__ __forceinline__ static uint32_t v41_attn_nseg_at(uint32_t p, uint32_t window, uint32_t ratio, uint32_t topk) {
    const uint32_t nw = p + 1u > window ? window : p + 1u;
    uint32_t tk = topk;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; if (vis < tk) tk = vis; }
    const uint32_t sk = v41_attn_seg_keys(p, window, ratio, topk);
    return (nw + tk + sk - 1u) / sk;
}
__device__ __forceinline__ static void ds4_attn_mma_gather_keys(
        __nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
        uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
        uint32_t window, uint32_t ng, uint32_t topk) {
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t lane = threadIdx.x & 31u, kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;
        if (t < nt) {
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
/* 一个 16 键列块的 S 片: 8 个 warp 各算 K 维的 1/8, partial 按固定序相加(与引擎逐字; stile 行步长 = SW 列) */
template <uint32_t SW>
__device__ __forceinline__ static void ds4_attn_mma_scores(
        float *stile, float *spart, const __nv_bfloat16 *qs, const __nv_bfloat16 *ks,
        const int *valid, uint32_t nt, float scale) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    const uint32_t ksteps = DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS;
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
        const uint32_t k = e & 15u, h = e >> 4;
        stile[h * SW + k] = (k < nt && valid[k]) ? v * scale : -1e30f;
    }
    __syncthreads();
}
/* 一个 16 键子块的在线 max/sum(一线程一头, 键序串行 —— 引擎逐字) */
__device__ __forceinline__ static void attn_stats16(const float *srow, float *rmax, float *rsum) {
    const uint32_t h = threadIdx.x;
    float m = rmax[h], sm = rsum[h], tm = m;
    for (uint32_t k = 0; k < DS4_ATTN_MMA_KT; k++) tm = fmaxf(tm, srow[k]);
    sm *= expf(m - tm);
    for (uint32_t k = 0; k < DS4_ATTN_MMA_KT; k++) sm += expf(srow[k] - tm);
    rmax[h] = tm; rsum[h] = sm;
}
/* P·V 一个 16 键子块(引擎逐字: warp w 管输出维 [64w, 64w+64) 的 4 个 n 块) */
__device__ __forceinline__ static void attn_pv16(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> *oacc,
                                                 const __nv_bfloat16 *ptile, const __nv_bfloat16 *kt) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
    wmma::load_matrix_sync(pa, ptile, 16);
    for (int j = 0; j < 4; j++) {
        wmma::load_matrix_sync(vb, kt + warp * 64u + (uint32_t)j * 16u, DS4_ATTN_MMA_HD);
        wmma::mma_sync(oacc[j], pa, vb, oacc[j]);
    }
}
__device__ __forceinline__ static void attn_epilogue(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> *oacc,
                                                     float *otile, float *pacc, float *pmax, float *psum, const float *rmax,
                                                     const float *rsum, uint64_t pbase) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
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
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = rmax[threadIdx.x]; psum[pbase + threadIdx.x] = rsum[threadIdx.x]; }
}

/* V0: 引擎现核逐字(直发路 posd=NULL, 去掉 PDL) */
__global__ static void v41_attn_mma_seg_kernel(float *pacc, float *pmax, float *psum, const float *q, const float *kvw,
                                               const uint8_t *kvc, const int32_t *idx, uint32_t pos0, uint32_t window,
                                               uint32_t ng, uint32_t topk, uint32_t n_head, float scale, uint32_t ratio, uint32_t nseg) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);
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
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        ds4_attn_mma_scores<16u>(stile, spart, qs, ks, valid, nt, scale);
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) attn_stats16(stile + threadIdx.x * 16u, rmax, rsum);
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        ds4_attn_mma_scores<16u>(stile, spart, qs, ks, valid, nt, scale);
        for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) ptile[e] = __float2bfloat16(expf(stile[e] - rmax[e >> 4]));
        __syncthreads();
        attn_pv16(oacc, ptile, ks);
    }
    attn_epilogue(oacc, spart, pacc, pmax, psum, rmax, rsum, pbase);
}
__global__ static void v41_sparse_attn_merge_kernel(float *o, const float *pacc, const float *pmax, const float *psum,
                                                    const float *sink, uint32_t nseg, uint32_t n_head, uint32_t hd) {
    const uint32_t h = blockIdx.x, i = blockIdx.y;
    const uint64_t b0 = (uint64_t)i * nseg * n_head;
    float m = -1e30f;
    for (uint32_t s = 0; s < nseg; s++) m = fmaxf(m, pmax[b0 + (uint64_t)s * n_head + h]);
    float den = 0.f;
    for (uint32_t s = 0; s < nseg; s++) den += psum[b0 + (uint64_t)s * n_head + h] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
    den += expf(sink[h] - m);
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t s = 0; s < nseg; s++) v += pacc[(b0 + (uint64_t)s * n_head + h) * hd + d] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
        o[((uint64_t)i * n_head + h) * hd + d] = v41_bf16r(v / den);
    }
}

