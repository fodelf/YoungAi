/* v41_vq_train_bench_h.cuh — 热专家前向核 vqh(2026-10-03), 只被 v41_vq_train_bench.cu include(在 _new.cuh 之后, 借 vqs_* 原语)。
 *
 * 【为什么】vqs 拆账(12 位层, 训练包真路由 L25): 冷专家 169 项 6.2 ms(墙 4.8), 热专家 22 个 5.4~5.7 ms —— 热专家位流才 0.15 GB, 时间全在 mma:
 * ~3800 对 ≈ 270 GFLOP 只跑到 ~50 TFLOPS(GB10 bf16 mma 实测峰值 118 直接累加 / 230+ 每 k16 从零再加回)。vqs 一个 warp 只管 16 行 × ≤32 token,
 * 每个 k16 一条 A 片段只喂 4 条 HMMA, 激活片段每 HMMA 一条 LDS.64; 热专家按 32 切项, 863 个 token 的那个要解 27 遍。
 * 【形态】一个 warp 管 32 行(两组 A 片段: 行 g/8+g 与 16+g/24+g)× 64 个 token(8 个 n8 片); 一条激活片段喂两条 HMMA, 解码一次喂 16 条。
 * block = 8 warp(256 线程)= 256 行; 工作项按 64 个 token 切。段 = RS 轮(位流 + 激活), 两段在途, 逐行位流 384 B 一块提前两块预取进 L2(同 vqs)。
 * 数值与 vqs / vqm_kernel 逐位同(同一组 k16、同一个 FADD 次序、同一个出口)。 */
#pragma once
#define VQH_THREADS 256u
#define VQH_NTM     64u
static constexpr uint32_t vqh_smem(uint32_t cbb, int ext, uint32_t rs, uint32_t s) {
    return cbb + s * (VQS_BM * 12u * rs + (ext ? VQS_BM * 4u : 0u) + VQH_NTM * 128u * rs);
}
template <int EXT, int MODE, int CBF, uint32_t RS, uint32_t S>
__global__ __launch_bounds__(VQH_THREADS, 1) static void vqh_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t NTN = VQH_NTM / 8u, RB = 12u * RS, CP = RS == 4u ? 16u : (RS == 2u ? 8u : 4u);
    constexpr uint32_t BSB = VQS_BM * RB, EXB = EXT ? VQS_BM * 4u : 0u, ATB = VQH_NTM * 128u * RS, STB = BSB + EXB + ATB;
    constexpr uint32_t PCH = 384u, PD = 2u, SPC = PCH / RB;
    extern __shared__ __align__(16) uint8_t vqhsh[];
    uint8_t *cbs = vqhsh, *ring = vqhsh + cb_bytes;
    const int which = MODE;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u, h = q >> 1, hf = q & 1u;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * (CBF ? 16u : 8u) != cb_bytes) return;
        if (CBF == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQH_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t ntile = M / VQS_BM, nwork = nitems * ntile, nst = K / (64u * RS);
    uint32_t lr[4];   /* 本 lane 解的四行(单元内行号): 32w + g + {0, 8, 16, 24} */
    #pragma unroll
    for (uint32_t u = 0; u < 4u; u++) lr[u] = warp * 32u + g + 8u * u;
    const uint32_t cofs[4] = { ((0u + h) ^ (2u * (g & 3u))) << 4, ((2u + h) ^ (2u * (g & 3u))) << 4,
                               ((4u + h) ^ (2u * (g & 3u))) << 4, ((6u + h) ^ (2u * (g & 3u))) << 4 };
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQS_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {
            if (MODE == 2)
                for (uint32_t i = tid; i < VQS_BM * nt; i += VQH_THREADS) ys[(uint64_t)(base + i / VQS_BM) * M + r0 + i % VQS_BM] = 0.f;
            continue;
        }
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3, nach = nt * 8u * RS;
        const uint8_t *prow = m.ix + (size_t)(r0 + tid) * mrow;   /* 256 线程正好一行一个 */
        auto pf_chunk = [&](uint32_t c) { if (c * PCH < mrow) vqs_pf_l2(prow + c * PCH, mrow - c * PCH < PCH ? mrow - c * PCH : PCH); };
        #pragma unroll
        for (uint32_t c = 0; c < PD; c++) pf_chunk(c);
        if (EXT && tid == 0) vqs_pf_l2(m.ex + (size_t)r0 * erow, (VQS_BM * erow) & ~15u);
        const uint8_t *bsrc = m.ix + (size_t)(r0 + tid) * mrow, *esrc = EXT ? m.ex + (size_t)(r0 + tid) * erow : NULL;
        const uint16_t *xa = act + (uint64_t)base * K;
        auto issue = [&](uint32_t s) {   /* 位流: 线程 tid 搬第 tid 行的 RB 字节(3 块 CP 字节); 位平面 4 B; 激活 nt × RS 轮 */
            uint8_t *st = ring + (s % S) * STB;
            #pragma unroll
            for (uint32_t p = 0; p < 3u; p++) vqs_cp<CP>(st + tid * RB + CP * p, bsrc + RB * s + CP * p);
            if (EXT) vqs_cp<4>(st + BSB + tid * 4u, esrc + 4u * ((RS * s) >> 2));
            uint8_t *ad = st + BSB + EXB;
            for (uint32_t k = tid; k < nach; k += VQH_THREADS) {
                const uint32_t t = k / (8u * RS), rem = k - t * 8u * RS, r = rem >> 3, c = rem & 7u;
                vqs_cp<16>(ad + (r * VQH_NTM + t) * 128u + ((c ^ (2u * (t & 3u))) << 4), xa + (uint64_t)t * K + 64u * (RS * s + r) + 8u * c);
            }
        };
        #pragma unroll
        for (uint32_t s = 0; s + 1u < S; s++) { if (s < nst) issue(s); vqs_commit(); }
        const uint32_t nnt = (nt + 7u) >> 3;
        float acc[2][NTN][4];
        #pragma unroll
        for (uint32_t p = 0; p < 2u; p++)
            #pragma unroll
            for (uint32_t j = 0; j < NTN; j++) { acc[p][j][0] = acc[p][j][1] = acc[p][j][2] = acc[p][j][3] = 0.f; }
        auto run = [&](auto nntc) {
            constexpr uint32_t NNT = decltype(nntc)::value;
            for (uint32_t s = 0; s < nst; s++) {
                vqs_wait<(int)S - 2>();
                __syncthreads();
                if (s + S - 1u < nst) issue(s + S - 1u);
                vqs_commit();
                if (s % SPC == 0u) pf_chunk(s / SPC + PD);
                const uint8_t *st = ring + (s % S) * STB;
                const uint8_t *as = st + BSB + EXB + 8u * hf + g * 128u;
                #pragma unroll
                for (uint32_t r = 0; r < RS; r++) {
                    uint32_t wv[4][3], xe[4] = {0u, 0u, 0u, 0u};
                    #pragma unroll
                    for (uint32_t u = 0; u < 4u; u++) {
                        const uint32_t *pw = (const uint32_t *)(st + lr[u] * RB + 12u * r);
                        wv[u][0] = pw[0]; wv[u][1] = pw[1]; wv[u][2] = pw[2];
                        if (EXT) xe[u] = *(const uint32_t *)(st + BSB + lr[u] * 4u);
                    }
                    const uint32_t eb = ((RS * s + r) & 3u) * 8u;
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; kk++) {
                        uint32_t a[2][4];
                        #pragma unroll
                        for (uint32_t u = 0; u < 4u; u++) {
                            uint32_t v = vqr_code(wv[u][0], wv[u][1], wv[u][2], kk, h);
                            if (EXT) v |= ((xe[u] >> (eb + 2u * kk + h)) & 1u) << 12;
                            uint32_t e0, e1;
                            vqs_half<CBF>(cbs, v, hf, e0, e1);
                            a[u >> 1][(u & 1u)] = e0; a[u >> 1][2u + (u & 1u)] = e1;   /* 组 p = u/2: (行 g 低, 行 8+g 低, 行 g 高, 行 8+g 高) */
                        }
                        const uint8_t *ap = as + r * VQH_NTM * 128u + cofs[kk];
                        #pragma unroll
                        for (uint32_t j = 0; j < NNT; j++) {
                            const uint2 x = *(const uint2 *)(ap + j * 8u * 128u);
                            #pragma unroll
                            for (uint32_t p = 0; p < 2u; p++) {
                                float t[4];
                                vqs_mma0(t, a[p], x.x, x.y);
                                #pragma unroll
                                for (int c = 0; c < 4; c++) acc[p][j][c] += t[c];
                            }
                        }
                    }
                }
            }
        };
#define VQH_NNT(n) std::integral_constant<uint32_t, ((n) < NTN ? (n) : NTN)>{}
        switch (nnt) {
        case 1: run(VQH_NNT(1)); break;  case 2: run(VQH_NNT(2)); break;  case 3: run(VQH_NNT(3)); break;  case 4: run(VQH_NNT(4)); break;
        case 5: run(VQH_NNT(5)); break;  case 6: run(VQH_NNT(6)); break;  case 7: run(VQH_NNT(7)); break;  default: run(VQH_NNT(8)); break;
        }
#undef VQH_NNT
        __syncthreads();
        #pragma unroll
        for (uint32_t p = 0; p < 2u; p++) {
            const uint32_t ra = r0 + lr[2u * p], rb = r0 + lr[2u * p + 1u];
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
                    const float va = v41_bf16r(acc[p][j][e] * gna), vb = v41_bf16r(acc[p][j][2u + e] * gnb);
                    if (MODE == 0) { g32[o + ra] = va; g32[o + rb] = vb; }
                    else if (MODE == 1) { h16[o + ra] = v41_vq_swiglu(g32[o + ra], va, clamp); h16[o + rb] = v41_vq_swiglu(g32[o + rb], vb, clamp);
                                          if (ys) { ys[o + ra] = va; ys[o + rb] = vb; } }
                    else { ys[o + ra] = va; ys[o + rb] = vb; }
                }
            }
        }
    }
}
