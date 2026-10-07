/* v41_vq_train_bench_ref.cuh — v41_vq_train_bench.cu 的引擎同款部分(逐字抄本, 2026-10-03 版引擎):
 * v3 载荷解析(cuda_vq_row.inc.cu v41_vq_open<1>)、bf16 舍入 / 码本搬运 / SwiGLU(cuda_v41_1 / cuda_vq_decode)、
 * 预填张量核 vqm_kernel<EXT, MODE, CB>(cuda_vq_prefill_mma.inc.cu)、转置张量核 vqt_prescale_kernel + vqt_kernel(cuda_bwd_vq.inc.cu)。
 * ★每个函数都必须与引擎逐字同★: 新核的判据是"与这里的输出逐位同", 抄错了整把尺就是假的。拆出来只为守 500 行。 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "../../vq_fmt.h"

__device__ __forceinline__ static float v41_bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}
__device__ __forceinline__ static void v41_vq_cb_to_shared(uint8_t *dst, const uint8_t *src, uint32_t bytes) {
    const uint2 *s = (const uint2 *)src; uint2 *d = (uint2 *)dst;
    const uint32_t n = bytes / 8u;
    uint32_t i = threadIdx.x;
    for (; i + 7u * blockDim.x < n; i += 8u * blockDim.x) {
        uint2 t[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) t[q] = s[i + (uint32_t)q * blockDim.x];
        #pragma unroll
        for (int q = 0; q < 8; q++) d[i + (uint32_t)q * blockDim.x] = t[q];
    }
    for (; i < n; i += blockDim.x) d[i] = s[i];
}
__device__ __forceinline__ static uint16_t v41_vq_swiglu(float gi, float ui, float clamp) {
    if (clamp > 0.f) { if (gi > clamp) gi = clamp; if (ui > clamp) ui = clamp; if (ui < -clamp) ui = -clamp; }
    const float sg = gi / (1.0f + expf(-gi));
    return (uint16_t)(__float_as_uint(v41_bf16r(sg * ui)) >> 16);
}
typedef struct { const uint8_t *cb, *gr, *ix, *ex; const float *gov; uint32_t nidx_row, nbit, imsk, nc; int ok; } v41_vq_mat;
template <int V3>
__device__ __forceinline__ static v41_vq_mat v41_vq_open(const uint8_t *blob, int e, int which, uint32_t rows, uint32_t cols,
                                                         const float *gov) {
    v41_vq_mat m; m.ok = 0; m.cb = m.gr = m.ix = m.ex = NULL; m.gov = gov; m.nidx_row = m.nbit = m.imsk = m.nc = 0;
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    if (!off) return m;
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != (V3 ? DS4VQ_MAT3_MAGIC : DS4VQ_MAT_MAGIC)) return m;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t r32, c32; memcpy(&r32, pay + 8, 4); memcpy(&c32, pay + 12, 4);
    if (r32 != rows || c32 != cols || d16 != 8u) return m;
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    m.nidx_row = cols / 8u; m.nbit = nbit; m.imsk = (1u << nbit) - 1u; m.nc = n16;
    if (V3) {
        uint32_t flags, mnb; memcpy(&flags, pay + 16, 4); memcpy(&mnb, pay + 20, 4);
        uint64_t cb_off; memcpy(&cb_off, pay + 24, 8);
        if (mnb != 12u || !(flags & 1u)) return m;
        m.cb = blob + cb_off;
        m.gr = pay + 32;
        m.ix = m.gr + (size_t)rows * 2u;
        const uint32_t mrow = (m.nidx_row * 12u + 7u) / 8u;
        m.ex = (flags & 2u) ? m.ix + (size_t)rows * mrow : NULL;
        if (((flags & 2u) != 0u) != (nbit > 12u)) return m;
    } else {
        m.cb = pay + 16; m.gr = m.cb + (size_t)n16 * 8u * 2u; m.ix = m.gr + (size_t)rows * 2u;
    }
    m.ok = 1;
    return m;
}
typedef struct { int32_t e, t0, nt; } vqp_item;

/* ---- cuda_vq_prefill_mma.inc.cu(10-03) ---- */
#define VQM_BM      128u
#define VQM_BN      128u
#define VQM_BK      64u
#define VQM_TW      2u
#define VQM_THREADS (256u * VQM_TW)
#define VQM_TILE_BYTES ((VQM_BM + VQM_BN) * VQM_BK * 2u)
__device__ __forceinline__ static uint32_t vqm_e4m3x2_to_bf16x2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    const uint32_t mag = h & 0x7fff7fffu, nz = __vcmpne2(mag, 0u);
    return (h & 0x80008000u) | ((((mag >> 3) & 0x0fff0fffu) + 0x38003800u) & nz);
}
__device__ __forceinline__ static void vqm_ldsm4(uint32_t *r, const uint8_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void vqm_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ static uint32_t vqm_swz(uint32_t row, uint32_t chunk) { return row * 128u + ((chunk ^ (row & 7u)) << 4); }
__device__ __forceinline__ static void vqm_ldsm4t(uint32_t *r, const uint8_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static uint32_t vqm_swz256(uint32_t row, uint32_t chunk) { return row * 256u + ((chunk ^ (row & 7u)) << 4); }
template <int CB>
__device__ __forceinline__ static uint4 vqm_cw(const uint8_t *cbs, uint32_t v) {
    if (CB == 1) return *(const uint4 *)(cbs + (size_t)v * 16u);
    const uint2 cw = *(const uint2 *)(cbs + (size_t)v * 8u);
    uint4 o;
    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
    return o;
}
template <int EXT, int MODE, int CB>
__global__ __launch_bounds__(VQM_THREADS, 1) static void vqm_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t CPR = VQM_BK / 8u, TPR = VQM_THREADS / VQM_BM, CPT = CPR / TPR;
    constexpr uint32_t NF = VQM_BN / 8u, NFW = NF / VQM_TW, CH = (VQM_BN * CPR) / VQM_THREADS;
    extern __shared__ __align__(16) uint8_t vqmsh[];
    uint8_t *cbs = vqmsh, *As = vqmsh + cb_bytes, *Bs = As + VQM_BM * VQM_BK * 2u;
    const int which = MODE;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile, nit = K / VQM_BK;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * (CB ? 16u : 8u) != cb_bytes) return;
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQM_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQM_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {
            if (MODE == 2)
                for (uint32_t i = tid; i < VQM_BM * nt; i += VQM_THREADS)
                    if (r0 + i % VQM_BM < M) ys[(uint64_t)(base + i / VQM_BM) * M + r0 + i % VQM_BM] = 0.f;
            continue;
        }
        const uint32_t grow = r0 + ar, mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3;
        const bool rv = grow < M;
        const uint8_t *rowp = m.ix + (size_t)(rv ? grow : 0u) * mrow;
        const uint8_t *ep = EXT ? m.ex + (size_t)(rv ? grow : 0u) * erow : NULL;
        float acc[NFW][4];
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }
        uint32_t w0 = 0, w1 = 0, eb = 0;
        uint4 bx[CH];
        #pragma unroll
        for (uint32_t c = 0; c < CH; c++) bx[c] = make_uint4(0, 0, 0, 0);
        auto load_round = [&](uint32_t i) {
            const uint32_t bitoff = i * 12u * CPR + sub * 12u * CPT, a = (bitoff >> 5) << 2;
            if (rv) { w0 = __ldg((const unsigned int *)(rowp + a)); w1 = __ldg((const unsigned int *)(rowp + a + 4u));
                      if (EXT) eb = __ldg(ep + i); }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * VQM_THREADS, bt = ch / CPR, bc = ch % CPR;
                if (bt < nt) bx[c] = __ldg((const uint4 *)(act + (uint64_t)(base + bt) * K + (uint64_t)i * VQM_BK + bc * 8u));
            }
        };
        load_round(0);
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t sh = (i * 12u * CPR + sub * 12u * CPT) & 31u;
            const uint64_t u = (((uint64_t)w1 << 32) | w0) >> sh;
            #pragma unroll
            for (uint32_t q = 0; q < CPT; q++) {
                uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> (sub * CPT + q)) & 1u) << 12;
                *(uint4 *)(As + vqm_swz(ar, sub * CPT + q)) = vqm_cw<CB>(cbs, v);
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) { const uint32_t ch = tid + c * VQM_THREADS; *(uint4 *)(Bs + vqm_swz(ch / CPR, ch % CPR)) = bx[c]; }
            __syncthreads();
            if (i + 1u < nit) load_round(i + 1u);
            #pragma unroll
            for (uint32_t kk = 0; kk < VQM_BK / 16u; kk++) {
                uint32_t a[4];
                {   const uint32_t mt = lane >> 3, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                    vqm_ldsm4(a, As + vqm_swz(row, kk * 2u + (mt >> 1))); }
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
                    for (int c = 0; c < 4; c++) acc[2 * np][c] += t0[c];
                    if ((f0 + 1u) * 8u < nt) { vqm_mma(t1, a, b[2], b[3]);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[2 * np + 1][c] += t1[c]; }
                }
            }
            __syncthreads();
        }
        const uint32_t rr[2] = { r0 + wr * 16u + (lane >> 2), r0 + wr * 16u + (lane >> 2) + 8u };
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t r = rr[h];
            if (r >= M) continue;
            __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
            const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
            #pragma unroll
            for (uint32_t f = 0; f < NFW; f++) {
                const uint32_t gf = th * NFW + f;
                if (gf * 8u >= nt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                    if (t >= nt) continue;
                    const uint64_t o = (uint64_t)(base + t) * M + r;
                    const float val = v41_bf16r(acc[f][h * 2u + e] * g);
                    if (MODE == 0) g32[o] = val;
                    else if (MODE == 1) { h16[o] = v41_vq_swiglu(g32[o], val, clamp); if (ys) ys[o] = val; }
                    else ys[o] = val;
                }
            }
        }
    }
}

/* ---- cuda_bwd_vq.inc.cu(10-03): 转置张量核 ---- */
__device__ __forceinline__ static float vqb_gain(const v41_vq_mat &m, uint32_t r) {
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    return __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
}
#define VQT_BM 128u
#define VQT_BK 64u
#define VQT_TILE_BYTES ((VQT_BK * VQT_BM + VQM_BN * VQT_BK) * 2u)
__global__ static void vqt_prescale_kernel(uint16_t *g16, const float *g, const uint8_t *blob, const vqp_item *items, const uint32_t *off,
                                           uint32_t which, uint32_t R, uint32_t C, const float *gr_all, uint32_t OUTd, int *bad) {
    const vqp_item it = items[blockIdx.x];
    const v41_vq_mat m = v41_vq_open<1>(blob, it.e, (int)which, R, C, (which == 2u && gr_all) ? gr_all + (size_t)it.e * OUTd : NULL);
    if (!m.ok) { if (threadIdx.x == 0) atomicExch(bad, 1); return; }
    const uint32_t base = off[it.e] + (uint32_t)it.t0, nt = (uint32_t)it.nt;
    for (uint32_t r = threadIdx.x; r < R; r += blockDim.x) {
        const float gn = vqb_gain(m, r);
        for (uint32_t t = 0; t < nt; t++) {
            const uint64_t o = (uint64_t)(base + t) * R + r;
            g16[o] = __bfloat16_as_ushort(__float2bfloat16_rn(g[o] * gn));
        }
    }
}
template <int EXT, int CB>
__global__ __launch_bounds__(VQM_THREADS, 1) static void vqt_kernel(
        float *out, const uint16_t *g16, const uint8_t *blob, const vqp_item *items, uint32_t nitems, const uint32_t *off,
        uint32_t which, uint32_t R, uint32_t C, const float *gr_all, uint32_t OUTd, int accumulate, uint32_t cb_bytes, int *bad) {
    constexpr uint32_t CPR = VQT_BM / 8u, TPR = VQM_THREADS / VQT_BK, CPT = CPR / TPR;
    constexpr uint32_t NF = VQM_BN / 8u, NFW = NF / VQM_TW;
    constexpr uint32_t BCPR = VQT_BK / 8u, CH = (VQM_BN * BCPR) / VQM_THREADS;
    extern __shared__ __align__(16) uint8_t vqtsh[];
    uint8_t *cbs = vqtsh, *As = vqtsh + cb_bytes, *Bs = As + VQT_BK * VQT_BM * 2u;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint32_t ntile = (C + VQT_BM - 1u) / VQT_BM, nwork = nitems * ntile, nit = R / VQT_BK;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, (int)which, R, C, NULL);
        if (!m0.ok || m0.nc * (CB ? 16u : 8u) != cb_bytes) { if (tid == 0) atomicExch(bad, 1); return; }
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQM_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t c0 = (w % ntile) * VQT_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, (int)which, R, C, (which == 2u && gr_all) ? gr_all + (size_t)it.e * OUTd : NULL);
        if (!m.ok) { if (tid == 0) atomicExch(bad, 1); continue; }
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3, j0 = c0 / 8u + sub * CPT;
        float acc[NFW][4];
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r0 = i * VQT_BK;
            {
                const uint32_t r = r0 + ar, bitoff = j0 * 12u, a = (bitoff >> 5) << 2, sh = bitoff & 31u;
                const uint8_t *rowp = m.ix + (size_t)r * mrow;
                const uint64_t u = (((uint64_t)__ldg((const unsigned int *)(rowp + a + 4u)) << 32) | __ldg((const unsigned int *)(rowp + a))) >> sh;
                const uint32_t eb = EXT ? (uint32_t)__ldg(m.ex + (size_t)r * erow + (j0 >> 3)) : 0u;
                #pragma unroll
                for (uint32_t q = 0; q < CPT; q++) {
                    uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                    if (EXT) v |= ((eb >> ((j0 + q) & 7u)) & 1u) << 12;
                    *(uint4 *)(As + vqm_swz256(ar, sub * CPT + q)) = vqm_cw<CB>(cbs, v);
                }
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * VQM_THREADS, bt = ch / BCPR, bc = ch % BCPR;
                uint4 pk = make_uint4(0, 0, 0, 0);
                if (bt < nt) pk = *(const uint4 *)(g16 + (uint64_t)(base + bt) * R + r0 + bc * 8u);
                *(uint4 *)(Bs + vqm_swz(bt, bc)) = pk;
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t kk = 0; kk < VQT_BK / 16u; kk++) {
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
                    for (int q = 0; q < 4; q++) acc[2 * np][q] += t0[q];
                    if ((f0 + 1u) * 8u < nt) { vqm_mma(t1, a, b[2], b[3]);
                        #pragma unroll
                        for (int q = 0; q < 4; q++) acc[2 * np + 1][q] += t1[q]; }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t cl = c0 + wr * 16u + (lane >> 2) + h * 8u;
            if (cl >= C) continue;
            #pragma unroll
            for (uint32_t f = 0; f < NFW; f++) {
                const uint32_t gf = th * NFW + f;
                if (gf * 8u >= nt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                    if (t >= nt) continue;
                    float *o = out + (uint64_t)(base + t) * C + cl;
                    *o = accumulate ? *o + acc[f][h * 2u + e] : acc[f][h * 2u + e];
                }
            }
        }
    }
}
