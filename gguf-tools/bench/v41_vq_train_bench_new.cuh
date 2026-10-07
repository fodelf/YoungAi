/* v41_vq_train_bench_new.cuh — 专家张量核的新形态(2026-10-03 立), 只被 v41_vq_train_bench.cu include。
 *
 * 【为什么】10-03 训练逐核表(batch 4, maxlen 880): 专家核占 GPU 时间 52.5%, vqm 每发 5.5 ms、vqt 7.6 ms, 读位流的墙 ~2.5~3.3 ms。
 * 微基准(12 位层, 真路由 880 token)量出来现核每轮 ~0.9 µs ≈ 一次 DRAM 延迟: 它只预取一轮, 在途字节太少, 卡在延迟上。
 *
 * 【vqs = 前向的"分段预取 + 寄存器直解"】一个 block(16 warp)管 (工作项, 256 行) × 全部 K:
 *   ① 位流与激活按 RS 轮(一轮 = 64 列)一段, cp.async 先搬进 shared, S 段在途(整段合并访存, 在途字节够盖延迟; 一段一个屏障);
 *   ② lane (g = lane/4, q = lane%4) 解第 g 行与第 8+g 行的码字, 查完表就是 mma 的 A 片段(权重当 A、16 行; 激活当 B、8 个 token 一片),
 *      不写 A/B 瓦片、不用 ldmatrix。k16 组内逻辑 k = 2q+{0,1} ↔ 物理列 16kk+4q+{0,1}, 2q+8+{0,1} ↔ 16kk+4q+{2,3}
 *      ⇒ 本 lane 每个 k16 要码字 2kk+q/2 的元素 4(q&1)..+3(半个码字): bf16 码本一条 LDS.64, E4M3 码本一条 LDS.32 + 现转;
 *      激活同一组列 = 一条 8 B 读。
 * 【逐位同】mma 的 k16 组仍是同一组 16 列, 只换组内谁拿哪一列, 每个 k16 从零起算再 FADD 回累加器的次序一个没变 ——
 *   微基准实测与现核逐位同(张量核组内求和与元素次序无关)。
 * 激活段里 token t 的 16 B 块 c 存在 (c ^ 2(t&3)): 一条 LDS.64 的半个 warp 是 4 个 token × 2 个块, 这样落在 8 个不同的 bank 组。 */
#pragma once
#include <type_traits>
#define VQR_WARPS   16u
#define VQR_THREADS (VQR_WARPS * 32u)
#define VQS_THREADS 512u
#define VQS_BM      256u                 /* 一个工作单元的行数: 16 warp × 16 行 */

/* 一行这一轮的第 kk 个 k16 要的码字: 轮内码字号 c = 2kk + h(h = q/2), 位偏移 12c; 本轮 96 位在 w0..w2 */
__device__ __forceinline__ static uint32_t vqr_code(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t kk, uint32_t h) {
    if (kk == 0u) return (w0 >> (12u * h)) & 0xFFFu;
    if (kk == 1u) return (h ? (w1 >> 4) : __funnelshift_r(w0, w1, 24u)) & 0xFFFu;
    if (kk == 2u) return __funnelshift_r(w1, w2, 16u + 12u * h) & 0xFFFu;
    return (w2 >> (8u + 12u * h)) & 0xFFFu;
}
template <uint32_t N>
__device__ __forceinline__ static void vqs_cp(uint8_t *dst, const void *src) {   /* N = 16 走 .cg(只进 L2), 8/4 走 .ca */
    if (N == 16u) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
    else asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "n"(N) : "memory");
}
__device__ __forceinline__ static void vqs_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ static void vqs_pf_l2(const void *p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
template <int N> __device__ __forceinline__ static void vqs_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
/* E4M3x2 → bf16x2: 硬件转 f16x2(精确), 两半转 f32(精确), 打包 bf16x2(值只有 4 位有效位, 舍入不动它)。
 * 与 vqm_e4m3x2_to_bf16x2 在全部非 NaN 输入上逐位同(含 ±0 与非规格化数); NaN 那两个码两边不同, 但码本里没有(转换器只产有限值)。 */
__device__ __forceinline__ static uint32_t vqs_e4m3x2_bf16x2(uint32_t two) {
    uint32_t hh, o;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hh) : "h"((unsigned short)(two & 0xffffu)));
    const float lo = __half2float(__ushort_as_half((unsigned short)(hh & 0xffffu))), hi = __half2float(__ushort_as_half((unsigned short)(hh >> 16)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o) : "f"(hi), "f"(lo));
    return o;
}
/* D = A·B(C 恒 0, 输出单独一组寄存器)。为什么不用 vqm_mma(t, …) 配 t = {0}: 那是 "+f" 读写约束, ptxas 把输出 D 排进 A 那组寄存器,
 * A 片段在多片 token 间复用时每条 HMMA 前要 4 条 MOV 重拼 A(10-03 SASS: 一段 32 条 HMMA 配 128 条 MOV)。数值与 vqm_mma(t={0}) 逐位同(C = +0)。 */
__device__ __forceinline__ static void vqs_mma0(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
/* 码字 v 的第 4·hf..4·hf+3 个元素 → (e0, e1)。CBF 0 = 码本 E4M3(8 B/词)现转; 1 = 进 shared 时已转 bf16(16 B/词) */
template <int CBF>
__device__ __forceinline__ static void vqs_half(const uint8_t *cbs, uint32_t v, uint32_t hf, uint32_t &e0, uint32_t &e1) {
    if (CBF == 1) { const uint2 e = *(const uint2 *)(cbs + (size_t)v * 16u + hf * 8u); e0 = e.x; e1 = e.y; }
    else { const uint32_t e = *(const uint32_t *)(cbs + (size_t)v * 8u + hf * 4u); e0 = vqs_e4m3x2_bf16x2(e); e1 = vqs_e4m3x2_bf16x2(e >> 16); }
}
/* shared 字节: 码本 + S 段 × (位流 256 行 × 12·RS B + 位平面 256 × 4 B + 激活 NTM × 128·RS B) */
__host__ __device__ constexpr uint32_t vqs_smem(uint32_t cbb, int ext, uint32_t ntm, uint32_t rs, uint32_t s) {
    return cbb + s * (VQS_BM * 12u * rs + (ext ? VQS_BM * 4u : 0u) + ntm * 128u * rs);
}
template <int EXT, int MODE, int CBF, uint32_t NTM, uint32_t RS, uint32_t S, int PF = 0>
__global__ __launch_bounds__(VQS_THREADS, 1) static void vqs_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t NTN = NTM / 8u, RB = 12u * RS, CP = RS == 4u ? 16u : 8u;   /* 每行每段 RB 字节, 拆 3 块 CP 字节 */
    constexpr uint32_t BSB = VQS_BM * RB, EXB = EXT ? VQS_BM * 4u : 0u, ATB = NTM * 128u * RS, STB = BSB + EXB + ATB;
    static_assert(RS == 2u || RS == 4u, "一段 2 或 4 轮");
    extern __shared__ __align__(16) uint8_t vqssh[];
    uint8_t *cbs = vqssh, *ring = vqssh + cb_bytes;
    const int which = MODE;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u, h = q >> 1, hf = q & 1u;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * (CBF ? 16u : 8u) != cb_bytes) return;
        if (CBF == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQS_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t ntile = M / VQS_BM, nwork = nitems * ntile, nst = K / (64u * RS);
    const uint32_t la = warp * 16u + g, lb = la + 8u;
    /* 本线程搬位流的两块(k = tid, tid+512 < 768): 行 k/3 的第 k%3 块 */
    const uint32_t k0 = tid, k1 = tid + VQS_THREADS, rw0 = k0 / 3u, pt0 = k0 - rw0 * 3u, rw1 = k1 / 3u, pt1 = k1 - rw1 * 3u;
    const uint32_t cofs[4] = { ((0u + h) ^ (2u * (g & 3u))) << 4, ((2u + h) ^ (2u * (g & 3u))) << 4,
                               ((4u + h) ^ (2u * (g & 3u))) << 4, ((6u + h) ^ (2u * (g & 3u))) << 4 };
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQS_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {   /* 整个 block 同一个工作项 ⇒ 同一判断, 不会有人卡在屏障上 */
            if (MODE == 2)
                for (uint32_t i = tid; i < VQS_BM * nt; i += VQS_THREADS) ys[(uint64_t)(base + i / VQS_BM) * M + r0 + i % VQS_BM] = 0.f;
            continue;
        }
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3, nach = nt * 8u * RS;
        if (PF == 1 && tid == 0) {   /* 本单元 256 行的位流是一整段连续字节: 一条 bulk 预取进 L2, DRAM 看到的是整段顺序读(段内 cp.async 只读 L2) */
            const uint8_t *p = m.ix + (size_t)r0 * mrow; const uint32_t nb = (VQS_BM * mrow) & ~15u;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(nb) : "memory");
            if (EXT) { const uint8_t *pe = m.ex + (size_t)r0 * erow; asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pe), "r"((VQS_BM * erow) & ~15u) : "memory"); }
        }
        /* PF 2 = 逐行分块预取: 每行按 384 B(3 条整线)一块, 线程 tid 管第 tid 行, 提前 PD 块 —— DRAM 请求是整线连续的, L2 里每行只压 PD 块 */
        constexpr uint32_t PCH = 384u, PD = 2u, SPC = PCH / RB;   /* SPC = 一块够几段 */
        const uint8_t *prow = m.ix + (size_t)(r0 + (tid & 255u)) * mrow;
        auto pf_chunk = [&](uint32_t c) {
            if (PF == 2 && tid < VQS_BM && c * PCH < mrow) {
                const uint32_t nb = mrow - c * PCH < PCH ? mrow - c * PCH : PCH;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(prow + c * PCH), "r"(nb) : "memory");
            }
        };
        if (PF == 2) {
            #pragma unroll
            for (uint32_t c = 0; c < PD; c++) pf_chunk(c);
            if (EXT && tid == 0) { const uint8_t *pe = m.ex + (size_t)r0 * erow; asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pe), "r"((VQS_BM * erow) & ~15u) : "memory"); }
        }
        const uint8_t *bsrc0 = m.ix + (size_t)(r0 + rw0) * mrow + CP * pt0, *bsrc1 = m.ix + (size_t)(r0 + (k1 < 768u ? rw1 : 0u)) * mrow + CP * pt1;
        const uint8_t *esrc = EXT ? m.ex + (size_t)(r0 + (tid & 255u)) * erow : NULL;
        const uint16_t *xa = act + (uint64_t)base * K;
        auto issue = [&](uint32_t s) {   /* 第 s 段(轮 RS·s ..)→ 槽 s % S */
            uint8_t *st = ring + (s % S) * STB;
            vqs_cp<CP>(st + rw0 * RB + CP * pt0, bsrc0 + RB * s);
            if (k1 < 768u) vqs_cp<CP>(st + rw1 * RB + CP * pt1, bsrc1 + RB * s);
            if (EXT && tid < VQS_BM) vqs_cp<4>(st + BSB + tid * 4u, esrc + 4u * ((RS * s) >> 2));   /* 位平面按 4 轮一字搬, RS=2 时相邻两段搬同一字 */
            uint8_t *ad = st + BSB + EXB;
            for (uint32_t k = tid; k < nach; k += VQS_THREADS) {
                const uint32_t t = k / (8u * RS), rem = k - t * 8u * RS, r = rem >> 3, c = rem & 7u;
                vqs_cp<16>(ad + (r * NTM + t) * 128u + ((c ^ (2u * (t & 3u))) << 4), xa + (uint64_t)t * K + 64u * (RS * s + r) + 8u * c);
            }
        };
        #pragma unroll
        for (uint32_t s = 0; s + 1u < S; s++) { if (s < nst) issue(s); vqs_commit(); }
        const uint32_t nnt = (nt + 7u) >> 3;
        float acc[NTN][4];
        #pragma unroll
        for (uint32_t j = 0; j < NTN; j++) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
        /* 段循环按本项的 n8 片数 NNT 实例化(编译期常量): 片循环没有逐片分支, A 片段在片间复用不用重拼 */
        auto run = [&](auto nntc) {
        constexpr uint32_t NNT = decltype(nntc)::value;
        for (uint32_t s = 0; s < nst; s++) {
            vqs_wait<(int)S - 2>();
            __syncthreads();   /* 第 s 段全到了; 也保证大家都用完了第 s-1 段的槽 ⇒ 下面可以往里搬第 s+S-1 段 */
            if (s + S - 1u < nst) issue(s + S - 1u);
            vqs_commit();
            if (PF == 2 && s % SPC == 0u) pf_chunk(s / SPC + PD);
            const uint8_t *st = ring + (s % S) * STB;
            uint32_t wa[3 * RS], wb[3 * RS];
            if (RS == 4u) {
                #pragma unroll
                for (uint32_t k = 0; k < 3u; k++) { const uint4 u = *(const uint4 *)(st + la * RB + 16u * k), v = *(const uint4 *)(st + lb * RB + 16u * k);
                    wa[4 * k] = u.x; wa[4 * k + 1] = u.y; wa[4 * k + 2] = u.z; wa[4 * k + 3] = u.w; wb[4 * k] = v.x; wb[4 * k + 1] = v.y; wb[4 * k + 2] = v.z; wb[4 * k + 3] = v.w; }
            } else {
                #pragma unroll
                for (uint32_t k = 0; k < 3u; k++) { const uint2 u = *(const uint2 *)(st + la * RB + 8u * k), v = *(const uint2 *)(st + lb * RB + 8u * k);
                    wa[2 * k] = u.x; wa[2 * k + 1] = u.y; wb[2 * k] = v.x; wb[2 * k + 1] = v.y; }
            }
            uint32_t xa_ = 0u, xb_ = 0u;
            if (EXT) { xa_ = *(const uint32_t *)(st + BSB + la * 4u); xb_ = *(const uint32_t *)(st + BSB + lb * 4u); }
            const uint8_t *as = st + BSB + EXB + 8u * hf + g * 128u;
            #pragma unroll
            for (uint32_t r = 0; r < RS; r++) {
                const uint32_t eb = ((RS * s + r) & 3u) * 8u;   /* 本轮在位平面字里的字节 */
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; kk++) {
                    uint32_t va = vqr_code(wa[3 * r], wa[3 * r + 1], wa[3 * r + 2], kk, h), vb = vqr_code(wb[3 * r], wb[3 * r + 1], wb[3 * r + 2], kk, h);
                    if (EXT) { const uint32_t c = eb + 2u * kk + h; va |= ((xa_ >> c) & 1u) << 12; vb |= ((xb_ >> c) & 1u) << 12; }
                    uint32_t a[4];
                    vqs_half<CBF>(cbs, va, hf, a[0], a[2]); vqs_half<CBF>(cbs, vb, hf, a[1], a[3]);
                    const uint8_t *ap = as + r * NTM * 128u + cofs[kk];
                    #pragma unroll
                    for (uint32_t j = 0; j < NTN; j++) {
                        if (j >= NNT) break;
                        const uint2 x = *(const uint2 *)(ap + j * 8u * 128u);
                        float t[4];
                        vqs_mma0(t, a, x.x, x.y);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[j][c] += t[c];
                    }
                }
            }
        }
        };
#define VQS_NNT(n) std::integral_constant<uint32_t, ((n) < NTN ? (n) : NTN)>{}
        switch (nnt) {
        case 1: run(VQS_NNT(1)); break;  case 2: run(VQS_NNT(2)); break;  case 3: run(VQS_NNT(3)); break;  case 4: run(VQS_NNT(4)); break;
        case 5: run(VQS_NNT(5)); break;  case 6: run(VQS_NNT(6)); break;  case 7: run(VQS_NNT(7)); break;  default: run(VQS_NNT(8)); break;
        }
#undef VQS_NNT
        __syncthreads();   /* 下一个单元的序幕要覆盖这几个槽 */
        /* 出口: c0,c1 = 行 g、token 8j+2q+{0,1}; c2,c3 = 行 8+g。舍入点与现核同(乘行增益 → bf16r) */
        const uint32_t ra = r0 + la, rb = r0 + lb;
        float gna, gnb;
        { __half gh; memcpy(&gh, m.gr + (size_t)ra * 2u, 2); gna = __half2float(gh) * (m.gov ? m.gov[ra] : 1.0f);
          memcpy(&gh, m.gr + (size_t)rb * 2u, 2); gnb = __half2float(gh) * (m.gov ? m.gov[rb] : 1.0f); }
        #pragma unroll
        for (uint32_t j = 0; j < NTN; j++) {
            if (j >= nnt) break;
            #pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t t = 8u * j + 2u * q + e;
                if (t >= nt) continue;
                const uint64_t o = (uint64_t)(base + t) * M;
                const float va = v41_bf16r(acc[j][e] * gna), vb = v41_bf16r(acc[j][2u + e] * gnb);
                if (MODE == 0) { g32[o + ra] = va; g32[o + rb] = vb; }
                else if (MODE == 1) { h16[o + ra] = v41_vq_swiglu(g32[o + ra], va, clamp); h16[o + rb] = v41_vq_swiglu(g32[o + rb], vb, clamp);
                                      if (ys) { ys[o + ra] = va; ys[o + rb] = vb; } }
                else { ys[o + ra] = va; ys[o + rb] = vb; }
            }
        }
    }
}

/* 转置: 一个 warp = (工作项, 64 列) × 全部 R 行。B 片段 j(j=0..7)的第 n 列 = 码字列 n 的第 j 个元素 ⇒ lane (g,q) 解码字列 g 的 4 行 */
template <int CB>
__device__ __forceinline__ static void vqrt_pair(const uint4 &e0, const uint4 &e1, uint32_t j, uint32_t &o) {
    /* CB=1: e 是 8 个 bf16(.x = 元素 0,1 …); 取两行的第 j 个元素拼成 (行0, 行1) */
    const uint32_t w0 = j < 2u ? e0.x : j < 4u ? e0.y : j < 6u ? e0.z : e0.w;
    const uint32_t w1 = j < 2u ? e1.x : j < 4u ? e1.y : j < 6u ? e1.z : e1.w;
    o = __byte_perm(w0, w1, (j & 1u) ? 0x7632u : 0x5410u);
}
template <int EXT, int CB, uint32_t MTM>
__global__ __launch_bounds__(VQR_THREADS, 1) static void vqrt_kernel(
        float *out, const uint16_t *g16, const uint8_t *blob, const vqp_item *items, uint32_t nitems, const uint32_t *off,
        uint32_t which, uint32_t R, uint32_t C, const float *gr_all, uint32_t OUTd, int accumulate, uint32_t cb_bytes, int *bad) {
    extern __shared__ __align__(16) uint8_t vqrtsh[];
    uint8_t *cbs = vqrtsh;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, (int)which, R, C, NULL);
        if (!m0.ok || m0.nc * (CB ? 16u : 8u) != cb_bytes) { if (tid == 0) atomicExch(bad, 1); return; }
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQR_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t nst = C / 64u, nwork = nitems * nst, nit = R / 64u;
    for (uint32_t w = blockIdx.x * VQR_WARPS + warp; w < nwork; w += gridDim.x * VQR_WARPS) {
        const vqp_item it = items[w / nst];
        const uint32_t c0 = (w % nst) * 64u, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, (int)which, R, C, NULL);
        if (!m.ok) { if (lane == 0) atomicExch(bad, 1); continue; }
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3, cc = c0 / 8u + g, bit = 12u * cc, sh = bit & 31u;
        const uint8_t *col = m.ix + ((bit >> 5) << 2);
        const uint8_t *ecol = EXT ? m.ex + (cc >> 3) : NULL;
        const uint32_t nmt = (nt + 15u) >> 4;
        const uint16_t *gp[MTM][2];
        #pragma unroll
        for (uint32_t mt = 0; mt < MTM; mt++)
            #pragma unroll
            for (uint32_t s = 0; s < 2u; s++) {
                const uint32_t t = mt * 16u + g + 8u * s, tc = t < nt ? t : nt - 1u;
                gp[mt][s] = g16 + (uint64_t)(base + tc) * R + 4u * q;
            }
        float acc[MTM][8][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < MTM; mt++)
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) { acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f; }
        for (uint32_t i = 0; i < nit; i++) {
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; kk++) {
                const uint32_t rr = 64u * i + 16u * kk + 4u * q;
                uint4 E[4];
                #pragma unroll
                for (uint32_t s = 0; s < 4u; s++) {
                    const uint8_t *p = col + (size_t)(rr + s) * mrow;
                    uint32_t v = __funnelshift_r(__ldg((const unsigned int *)p), __ldg((const unsigned int *)(p + 4)), sh) & 0xFFFu;
                    if (EXT) v |= (((uint32_t)__ldg(ecol + (size_t)(rr + s) * erow) >> (cc & 7u)) & 1u) << 12;
                    E[s] = vqm_cw<CB>(cbs, v);
                }
                uint32_t a[MTM][4];
                #pragma unroll
                for (uint32_t mt = 0; mt < MTM; mt++) {
                    if (mt >= nmt) break;
                    const uint2 x0 = __ldg((const uint2 *)(gp[mt][0] + 64u * i + 16u * kk));
                    const uint2 x1 = __ldg((const uint2 *)(gp[mt][1] + 64u * i + 16u * kk));
                    a[mt][0] = x0.x; a[mt][1] = x1.x; a[mt][2] = x0.y; a[mt][3] = x1.y;
                }
                #pragma unroll
                for (uint32_t j = 0; j < 8u; j++) {
                    uint32_t b0, b1;
                    vqrt_pair<CB>(E[0], E[1], j, b0); vqrt_pair<CB>(E[2], E[3], j, b1);
                    #pragma unroll
                    for (uint32_t mt = 0; mt < MTM; mt++) {
                        if (mt >= nmt) break;
                        float t[4] = {0.f, 0.f, 0.f, 0.f};
                        vqm_mma(t, a[mt], b0, b1);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[mt][j][c] += t[c];
                    }
                }
            }
        }
        /* 出口: 片 j 的 c0 = (token g, 列 c0+16q+j), c1 = (token g, 列 c0+16q+8+j); c2/c3 = token +8 */
        #pragma unroll
        for (uint32_t mt = 0; mt < MTM; mt++) {
            if (mt >= nmt) break;
            #pragma unroll
            for (uint32_t u = 0; u < 2u; u++) {
                const uint32_t t = mt * 16u + g + 8u * u;
                if (t >= nt) continue;
                float *o = out + (uint64_t)(base + t) * C + c0 + 16u * q;
                #pragma unroll
                for (uint32_t j = 0; j < 8u; j++) {
                    const float v0 = acc[mt][j][2u * u], v1 = acc[mt][j][2u * u + 1u];
                    o[j] = accumulate ? o[j] + v0 : v0;
                    o[8u + j] = accumulate ? o[8u + j] + v1 : v1;
                }
            }
        }
    }
}
