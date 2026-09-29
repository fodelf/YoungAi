/* v41_attn_prefill_bench_kernels.cuh — v41_attn_prefill_bench.cu 的引擎同款部分(逐字抄本, 2026-09-29 版 cuda_sparse_attn_mma.inc.cu):
 * 常量、bf16 舍入、窗口环行号、gather(shfl 版) / S 片(wmma, 8 warp 分 K) / 在线 max-sum(并行版) 三个积木, V0 = 预填稀疏注意力核。
 * 只被 v41_attn_prefill_bench.cu include(拆出来是守 500 行)。★每个函数都必须与引擎逐字同★: 变体的判据是"与 V0 同一量级地贴 f64 参考",
 * V0 抄错了整把尺就是假的。 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "../../src/common/ds4_fp8.h"

#define DS4_V41_CKV_NIB   256u
#define DS4_V41_CKV_BYTES 288u
#define DS4_ATTN_MMA_HEADS 16u
#define DS4_ATTN_MMA_KT    16u
#define DS4_ATTN_MMA_WARPS 8u
#define DS4_ATTN_MMA_HD    512u

__device__ __forceinline__ static float v41_bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}
__host__ __device__ __forceinline__ static uint64_t v41_win_row(int64_t a, uint32_t pos0, uint32_t window, uint32_t ring) {
    if (a >= (int64_t)pos0) return (uint64_t)((int64_t)window + a - (int64_t)pos0);
    return ring ? (uint64_t)(a % (int64_t)window) : (uint64_t)(a - ((int64_t)pos0 - (int64_t)window));
}

/* ---- 引擎逐字: gather(压缩行 32 个缩放 lane 各解一个, fp4 值表 lane 各持一个, shfl 取) ---- */
__device__ __forceinline__ static void ds4_attn_mma_gather_keys(
        __nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
        uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
        uint32_t window, uint32_t ng, uint32_t topk, uint32_t ring) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;
        if (t < nt) {
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, ring) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_MMA_HD;
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
}
/* ---- 引擎逐字: 一个键块的 S 片 [16 头][16 键], 8 个 warp 各算 K 的 1/8, partial 按固定序相加 ---- */
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
        const uint32_t k = e & 15u;
        stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
    }
    __syncthreads();
}
/* ---- 引擎逐字: 在线 max/sum 并行版(warp w 管头 2w / 2w+1, 每 lane 一个键) ---- */
__device__ __forceinline__ static void ds4_attn_mma_stats(const float *stile, float *rmax, float *rsum) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, k = lane & 15u;
    const uint32_t h = warp * 2u + (lane >> 4);
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

/* ---- V0: 引擎预填核逐字(grid (n_tok, 头组), 两遍扫键) ---- */
__global__ static void ds4_sparse_attn_mma_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                                  const int32_t *idx, const float *sink, uint32_t pos0, uint32_t window,
                                                  uint32_t ng, uint32_t topk, uint32_t n_head, float scale, uint32_t win_lo) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *qs = (__nv_bfloat16 *)ds4_attn_mma_smem;
    __nv_bfloat16 *ks = qs + DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD;
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);
    const uint32_t i = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS;
    for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
        qs[e] = __float2bfloat16(q[((uint64_t)i * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    const uint32_t p = pos0 + i;
    uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    if (lo < win_lo) lo = win_lo;
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    __syncthreads();
    for (uint32_t base = 0; base < nkeys; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (nkeys - base) < DS4_ATTN_MMA_KT ? (nkeys - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk, 1u);
        __syncthreads();
        ds4_attn_mma_scores(stile, spart, qs, ks, valid, nt, scale);
        ds4_attn_mma_stats(stile, rmax, rsum);
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = 0; base < nkeys; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (nkeys - base) < DS4_ATTN_MMA_KT ? (nkeys - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        ds4_attn_mma_gather_keys(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk, 1u);
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
    float *otile = spart;
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
static size_t ds4_attn_mma_smem_bytes(void) {
    return (size_t)(DS4_ATTN_MMA_HEADS + DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_HD * sizeof(__nv_bfloat16)
         + (size_t)DS4_ATTN_MMA_WARPS * 256u * sizeof(float) + 256u * sizeof(float)
         + 256u * sizeof(__nv_bfloat16) + 2u * DS4_ATTN_MMA_HEADS * sizeof(float)
         + DS4_ATTN_MMA_KT * sizeof(int);
}
