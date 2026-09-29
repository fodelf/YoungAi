/* v41_attn_prefill_bench_variants.cuh — v41_attn_prefill_bench.cu 的变体核(只被它 include; 拆出来是守 500 行)。
 *
 * 单遍 flash 形态的预填稀疏注意力(attn_fa_kernel<HB, KT>): 与引擎两遍扫键的差别 ——
 *   ①q 不进 shared: 每个 warp 把自己 K 段(64 维 = 4 个 k16)的 A 片段常驻寄存器(16 头 16 个寄存器, 32 头 32 个), 省 16~32 KB shared;
 *   ②S 用 mma.m16n8k16 直算(不走 wmma), 8 个 warp 分 K, partial 落 shared 按固定序相加(与引擎同一分工, 只是指令换了);
 *   ③在线 softmax 单遍: 每个键块算完 S 就更新 max/sum, O 累加器按 alpha = exp(m_old − m_new) 缩放后加 P·V —— 累加器布局是
 *     mma 文档定死的(c0,c1 = 行 lane/4, c2,c3 = 行 +8), 所以按头缩放拿得到那个映射(引擎注释说 wmma 的 fragment 不透明, 这正是换 mma 的理由);
 *     键只 gather 一次、S 只算一次(引擎两遍: gather 2 次 + S 2 次);
 *   ④键片行距补 8 个 bf16(1040 B): 行距 1024 B 时 ldmatrix 取 8 行同一列全落同一组 bank(8 路冲突), 引擎的 wmma load 同病。
 * ★数值★: 与引擎不逐位同(单遍 vs 两遍的舍入路径不同: P 在 bf16 的舍入点相同, 但 O 多了 alpha 缩放的 f32 乘法), 门 = 与 f64 参考的相对误差
 *   和引擎同一量级(bf16 出口 ~4e-3)。进引擎的判据是五指标/PPL, 与 09-15 mma 版落地时同一条规矩。 */
#pragma once
#include "v41_attn_prefill_bench_kernels.cuh"

#define FA_LD 520u   /* 键片行距(bf16 元素): 512 + 8, 避开 bank 冲突 */

__device__ __forceinline__ static void fa_ldsm4(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void fa_ldsm4t(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void fa_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ static uint32_t fa_bf16x2(float lo, float hi) {
    const uint32_t l = __float_as_uint(v41_bf16r(lo)) >> 16, h = __float_as_uint(v41_bf16r(hi)) >> 16;
    return l | (h << 16);
}

/* gather: 与引擎同一解码式, 只是行距 FA_LD; 一次 KT 个键(KT/8 个 warp 各一个键) */
template <uint32_t KT>
__device__ __forceinline__ static void fa_gather(__nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                 uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
                                                 uint32_t window, uint32_t ng, uint32_t topk) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));
    for (uint32_t t = threadIdx.x / 32u; t < KT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;
        if (t < nt) {
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * FA_LD;
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

/* HB = 一 block 管几个头(16 / 32 = 1 / 2 个 m16 片); KT = 一次进 shared 几个键(16 / 32)。8 个 warp: 算 S 时分 K(每 warp 64 维), 算 O 时分输出维(每 warp 64 维)。
 * shared: ks KT×FA_LD×2 + spart 8×HB×KT×4 + stile HB×KT×4 + ptile HB×KT×2 + rmax/rsum/alpha 3×HB×4 + valid KT×4 */
template <uint32_t HB, uint32_t KT>
__global__ __launch_bounds__(256) static void attn_fa_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                             const float *sink, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                                             uint32_t n_head, float scale, uint32_t win_lo) {
    constexpr uint32_t MT = HB / 16u, NB = KT / 8u;   /* m16 片数 / 键的 n8 片数 */
    extern __shared__ __align__(16) uint8_t fa_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)fa_smem;
    float *spart = (float *)(ks + (size_t)KT * FA_LD);
    float *stile = spart + 8u * HB * KT;
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + HB * KT);
    float *rmax = (float *)(ptile + HB * KT), *rsum = rmax + HB, *alpha = rsum + HB;
    int *valid = (int *)(alpha + HB);
    const uint32_t i = blockIdx.x, h0 = blockIdx.y * HB, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t p = pos0 + i;
    uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    if (lo < win_lo) lo = win_lo;
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    /* q 的 A 片段常驻寄存器: warp w 管维 [64w, 64w+64) 的 4 个 k16; 每片段 4 个寄存器(a0: 行 r 列 c..c+1, a1: 行 r+8, a2: 列 +8, a3: 行 +8 列 +8) */
    uint32_t qa[MT][4][4];
    {
        const uint32_t r = lane >> 2, c = (lane & 3u) * 2u;
        #pragma unroll
        for (uint32_t m = 0; m < MT; m++)
            #pragma unroll
            for (uint32_t s = 0; s < 4u; s++) {
                const uint32_t d0 = warp * 64u + s * 16u + c;
                const float *q0 = q + ((uint64_t)i * n_head + h0 + m * 16u + r) * DS4_ATTN_MMA_HD + d0;
                const float *q1 = q0 + 8u * DS4_ATTN_MMA_HD;
                qa[m][s][0] = fa_bf16x2(q0[0], q0[1]); qa[m][s][1] = fa_bf16x2(q1[0], q1[1]);
                qa[m][s][2] = fa_bf16x2(q0[8], q0[9]); qa[m][s][3] = fa_bf16x2(q1[8], q1[9]);
            }
    }
    if (threadIdx.x < HB) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    float oacc[MT][8][4];   /* O[HB 头][本 warp 64 维] = MT × 8 个 n8 片 */
    #pragma unroll
    for (uint32_t m = 0; m < MT; m++)
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) { oacc[m][j][0] = oacc[m][j][1] = oacc[m][j][2] = oacc[m][j][3] = 0.f; }
    __syncthreads();
    for (uint32_t base = 0; base < nkeys; base += KT) {
        const uint32_t nt = (nkeys - base) < KT ? (nkeys - base) : KT;
        __syncthreads();   /* 上一块的 ks/ptile 用完 */
        fa_gather<KT>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        /* ① S 部分和: 本 warp 的 64 维 × HB 头 × KT 键 */
        {
            float sp[MT][NB][4];
            #pragma unroll
            for (uint32_t m = 0; m < MT; m++)
                #pragma unroll
                for (uint32_t nb = 0; nb < NB; nb++) { sp[m][nb][0] = sp[m][nb][1] = sp[m][nb][2] = sp[m][nb][3] = 0.f; }
            #pragma unroll
            for (uint32_t s = 0; s < 4u; s++) {
                const uint32_t d0 = warp * 64u + s * 16u;
                #pragma unroll
                for (uint32_t nb = 0; nb < NB; nb += 2u) {   /* 一次 ldmatrix x4 取两个 n8 片(16 个键)的 k16 */
                    uint32_t b[4];
                    const uint32_t mt = lane >> 3, key = nb * 8u + (mt >> 1) * 8u + (lane & 7u), col = d0 + (mt & 1u) * 8u;
                    fa_ldsm4(b, ks + (size_t)key * FA_LD + col);
                    #pragma unroll
                    for (uint32_t m = 0; m < MT; m++) { fa_mma(sp[m][nb], qa[m][s], b[0], b[1]); fa_mma(sp[m][nb + 1], qa[m][s], b[2], b[3]); }
                }
            }
            /* partial → spart[warp][h][k](f32), 布局: c0,c1 = 行 lane/4 列 (lane&3)*2+{0,1}; c2,c3 = 行 +8 */
            #pragma unroll
            for (uint32_t m = 0; m < MT; m++)
                #pragma unroll
                for (uint32_t nb = 0; nb < NB; nb++) {
                    float *sw = spart + (size_t)warp * HB * KT + (m * 16u + (lane >> 2)) * KT + nb * 8u + (lane & 3u) * 2u;
                    sw[0] = sp[m][nb][0]; sw[1] = sp[m][nb][1]; sw[8u * KT] = sp[m][nb][2]; sw[8u * KT + 1u] = sp[m][nb][3];
                }
        }
        __syncthreads();
        /* ② 固定序求和 + 缩放 + 屏蔽 → stile; 每 (头, 键) 一个线程轮流 */
        for (uint32_t e = threadIdx.x; e < HB * KT; e += blockDim.x) {
            float v = 0.f;
            #pragma unroll
            for (uint32_t w = 0; w < 8u; w++) v += spart[(size_t)w * HB * KT + e];
            const uint32_t k = e % KT;
            stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
        }
        __syncthreads();
        /* ③ 在线 max/sum + P 片 + alpha: 每 warp 管 HB/8 个头(lane 组各 KT... 这里每头一条 lane 链: 线程 h 串行扫 KT 个键, 与引擎串行版同序) */
        if (threadIdx.x < HB) {
            const uint32_t h = threadIdx.x;
            const float *sr = stile + h * KT;
            const float m = rmax[h];
            float tm = m;
            for (uint32_t k = 0; k < KT; k++) tm = fmaxf(tm, sr[k]);
            const float al = expf(m - tm);
            float sm = rsum[h] * al;
            for (uint32_t k = 0; k < KT; k++) { const float pv = expf(sr[k] - tm); sm += pv; ptile[h * KT + k] = __float2bfloat16(pv); }
            rmax[h] = tm; rsum[h] = sm; alpha[h] = al;
        }
        __syncthreads();
        /* ④ O = O·alpha + P·V: A = ptile [HB][KT](行主序, k = 键), B = ks [键][维] 经 .trans 取; 本 warp 管维 [64w, 64w+64) 的 8 个 n8 片 */
        #pragma unroll
        for (uint32_t m = 0; m < MT; m++) {
            const float a0 = alpha[m * 16u + (lane >> 2)], a1 = alpha[m * 16u + (lane >> 2) + 8u];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) { oacc[m][j][0] *= a0; oacc[m][j][1] *= a0; oacc[m][j][2] *= a1; oacc[m][j][3] *= a1; }
        }
        #pragma unroll
        for (uint32_t kb = 0; kb < KT / 16u; kb++) {   /* 键按 k16 分块 */
            uint32_t pa[MT][4];
            #pragma unroll
            for (uint32_t m = 0; m < MT; m++) {
                const uint32_t mt = lane >> 3, row = m * 16u + (lane & 7u) + (mt & 1u) * 8u, col = kb * 16u + (mt >> 1) * 8u;
                fa_ldsm4(pa[m], ptile + (size_t)row * KT + col);
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j += 2u) {   /* 一次 .trans x4 取两个 n8 维片 × k16 键 */
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = kb * 16u + (mt & 1u) * 8u + (lane & 7u), col = warp * 64u + j * 8u + (mt >> 1) * 8u;
                fa_ldsm4t(b, ks + (size_t)key * FA_LD + col);
                #pragma unroll
                for (uint32_t m = 0; m < MT; m++) { fa_mma(oacc[m][j], pa[m], b[0], b[1]); fa_mma(oacc[m][j + 1], pa[m], b[2], b[3]); }
            }
        }
    }
    __syncthreads();
    /* 出口: 除以 (sum + exp(sink − max)), 舍 bf16 */
    #pragma unroll
    for (uint32_t m = 0; m < MT; m++) {
        const uint32_t hA = m * 16u + (lane >> 2), hB = hA + 8u;
        const float dA = rsum[hA] + expf(sink[h0 + hA] - rmax[hA]), dB = rsum[hB] + expf(sink[h0 + hB] - rmax[hB]);
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t d = warp * 64u + j * 8u + (lane & 3u) * 2u;
            float *oA = o + ((uint64_t)i * n_head + h0 + hA) * DS4_ATTN_MMA_HD + d, *oB = o + ((uint64_t)i * n_head + h0 + hB) * DS4_ATTN_MMA_HD + d;
            oA[0] = v41_bf16r(oacc[m][j][0] / dA); oA[1] = v41_bf16r(oacc[m][j][1] / dA);
            oB[0] = v41_bf16r(oacc[m][j][2] / dB); oB[1] = v41_bf16r(oacc[m][j][3] / dB);
        }
    }
}
template <uint32_t HB, uint32_t KT>
static size_t fa_smem_bytes(void) {
    return (size_t)KT * FA_LD * 2u + (size_t)8u * HB * KT * 4u + (size_t)HB * KT * 4u + (size_t)HB * KT * 2u + 3u * HB * 4u + (size_t)KT * 4u;
}
