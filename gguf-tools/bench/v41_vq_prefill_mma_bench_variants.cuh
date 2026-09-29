/* v41_vq_prefill_mma_bench_variants.cuh — v41_vq_prefill_mma_bench.cu 的变体核(只被它 include; 拆出来是守 500 行)。
 * vqm_v_kernel: 现核的参数化版(BN/BK/TW/CB, 见 .cu 文件头); vqm_pc_kernel: 生产/消费双组 warp + 双缓冲片(V10~V12)。 */
#pragma once
/* ---- 变体核 ---- */
template <uint32_t BK> __device__ __forceinline__ static uint32_t vswz(uint32_t row, uint32_t chunk) {
    /* 片一行 = BK 个 bf16 = BK/8 个 16 B 块; ldmatrix 一次取 8 行同一列块, 异或让它们落在 8 个不同 bank 组(BK=32 时一行只有 4 块, 用行号>>1) */
    return BK == 64u ? row * 128u + ((chunk ^ (row & 7u)) << 4) : row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}
/* 一层的码本转成 bf16 放全局(CB=2 用): 每码字 8 个 E4M3 → 8 个 bf16 = 16 B。转换式与现核逐字(vqm_e4m3x2_to_bf16x2)。 */
__global__ static void cb_bf16_kernel(uint4 *dst, const uint8_t *cb, uint32_t nc) {
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nc) return;
    const uint2 cw = *(const uint2 *)(cb + (size_t)v * 8u);
    uint4 o;
    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
    dst[v] = o;
}
template <int EXT, int MODE, uint32_t BN, uint32_t BK, uint32_t TW, int CB>
__global__ __launch_bounds__(256 * TW) static void vqm_v_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr, const uint4 *cbh) {
    constexpr uint32_t NT = 256u * TW, CPR = BK / 8u, TPR = NT / VQM_BM, CPT = CPR / TPR;   /* 每行每轮 CPR 个码字, TPR 个线程分, 每线程 CPT 个 */
    constexpr uint32_t NF = BN / 8u, NFW = NF / TW;                                        /* n8 片总数 / 每 warp 管几片 */
    constexpr uint32_t CH = (BN * CPR + NT - 1u) / NT;                                     /* B 片每线程搬几个 16 B 块 */
    static_assert(CPT >= 1u && CPT * TPR == CPR, "解码分工要整除");
    static_assert(NFW >= 2u && (NFW & 1u) == 0u, "每 warp 的 n8 片要成对(ldmatrix x4 一次取两片)");
    extern __shared__ __align__(16) uint8_t vqmsh[];
    const uint32_t nc = cb_bytes / 8u, cbsh = CB == 0 ? cb_bytes : (CB == 1 ? nc * 16u : 0u);
    uint8_t *cbs = vqmsh, *As = vqmsh + cbsh, *Bs = As + VQM_BM * BK * 2u;
    const int which = MODE;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile, nit = K / BK;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * 8u != cb_bytes) return;
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else if (CB == 1) for (uint32_t v = tid; v < nc; v += NT) {
            const uint2 cw = *(const uint2 *)(m0.cb + (size_t)v * 8u);
            uint4 o;
            o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
            o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
            *(uint4 *)(cbs + (size_t)v * 16u) = o;
        }
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQM_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {
            if (MODE == 2)
                for (uint32_t i = tid; i < VQM_BM * nt; i += NT)
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
        /* 本线程这一轮的码字在行内的位偏移 = 轮 × 每轮位数 + 本线程分到的码字 × 12; 读对齐的两个字再右移 —— 见文件头的分工表 */
        uint32_t w0 = 0, w1 = 0, eb = 0;
        uint4 bx[CH];
        #pragma unroll
        for (uint32_t c = 0; c < CH; c++) bx[c] = make_uint4(0, 0, 0, 0);
        auto load_round = [&](uint32_t i) {
            const uint32_t bitoff = i * 12u * CPR + sub * 12u * CPT, a = (bitoff >> 5) << 2;
            if (rv) { w0 = __ldg((const unsigned int *)(rowp + a)); w1 = __ldg((const unsigned int *)(rowp + a + 4u));
                      if (EXT) eb = __ldg(ep + ((i * CPR) >> 3)); }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * NT, bt = ch / CPR, bc = ch % CPR;
                if (ch < BN * CPR && bt < nt) bx[c] = __ldg((const uint4 *)(act + (uint64_t)(base + bt) * K + (uint64_t)i * BK + bc * 8u));
            }
        };
        load_round(0);
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t sh = (i * 12u * CPR + sub * 12u * CPT) & 31u;
            const uint64_t u = (((uint64_t)w1 << 32) | w0) >> sh;
            #pragma unroll
            for (uint32_t q = 0; q < CPT; q++) {
                uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> ((i * CPR + sub * CPT + q) & 7u)) & 1u) << 12;
                uint4 o;
                if (CB == 0) {
                    const uint2 cw = *(const uint2 *)(cbs + (size_t)v * 8u);
                    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
                    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
                } else if (CB == 1) o = *(const uint4 *)(cbs + (size_t)v * 16u);
                else o = __ldg(cbh + v);
                *(uint4 *)(As + vswz<BK>(ar, sub * CPT + q)) = o;
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * NT;
                if (ch < BN * CPR) *(uint4 *)(Bs + vswz<BK>(ch / CPR, ch % CPR)) = bx[c];
            }
            __syncthreads();
            if (i + 1u < nit) load_round(i + 1u);   /* 下一轮的位流/激活现在发出去, mma 期间在路上 */
            #pragma unroll
            for (uint32_t kk = 0; kk < BK / 16u; kk++) {
                uint32_t a[4];
                {   const uint32_t mt = lane >> 3, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                    vqm_ldsm4(a, As + vswz<BK>(row, kk * 2u + (mt >> 1))); }
                #pragma unroll
                for (uint32_t np = 0; np < NFW / 2u; np++) {
                    const uint32_t f0 = th * NFW + 2u * np;   /* 本 warp 的第 2np 片在全部片里的号 */
                    if (f0 * 8u >= nt) break;
                    uint32_t b[4];
                    const uint32_t mt = lane >> 3, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                    vqm_ldsm4(b, Bs + vswz<BK>(tok, kk * 2u + (mt & 1u)));
                    /* 每个 k16 从零起算再 FADD 提回(与引擎同: 张量核内部长 K 累加丢低位, 09-24 对拍定的) */
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
                    else if (MODE == 1) h16[o] = v41_vq_swiglu(g32[o], val, clamp);
                    else ys[o] = val;
                }
            }
        }
    }
}


/* ---- V10~V12: 生产/消费双组 warp + 双缓冲片(BK 32) ----
 * 为什么: 现核(与 vqm_v_kernel)一 block 的 8/16 个 warp 先一起解码再一起 mma, 两段用 __syncthreads 分相 —— 解码时张量管闲, mma 时 ALU 闲;
 * 13 位层码本 64 KB 占着 shared, 每 SM 只挂得下 1 个 block, 也没有第二个 block 来错相。这里把两件事分给两组 warp 同时做:
 * warp 0..7(生产)解下一轮的 A 片 + 搬下一轮的 B 片进备用缓冲, warp 8..15(消费)对当前缓冲做 mma; 每轮一次 __syncthreads 交换缓冲。
 * 一份 A+B 片(BK 32): (128 + BN) × 32 × 2 B, BN128 = 16 KB, 双缓冲 32 KB; 13 位码本 64 + 32 = 96 KB ≤ 99 ✓。
 * ★逐位同★: 同一组 bf16 乘积、同一个 k16 累加序(kk 升序, 每 k16 从零起算 FADD 提回)、同一个出口舍入点 —— 只换了哪些线程做哪件事。 */
template <int EXT, int MODE, uint32_t BN, int CB>
__global__ __launch_bounds__(512) static void vqm_pc_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr, const uint4 *cbh) {
    constexpr uint32_t BK = 32u, CPR = BK / 8u, NP = 256u, TPR = NP / VQM_BM, CPT = CPR / TPR;   /* 生产 256 线程: 每行 2 线程各 2 个码字 */
    constexpr uint32_t NF = BN / 8u, CH = (BN * CPR + NP - 1u) / NP, TILE = (VQM_BM + BN) * BK * 2u;
    static_assert(NF >= 2u && (NF & 1u) == 0u, "n8 片要成对");
    extern __shared__ __align__(16) uint8_t vqmsh[];
    const uint32_t nc = cb_bytes / 8u, cbsh = CB == 0 ? cb_bytes : (CB == 1 ? nc * 16u : 0u);
    uint8_t *cbs = vqmsh, *tiles = vqmsh + cbsh;
    const int which = MODE;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u;
    const bool producer = warp < 8u;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile, nit = K / BK;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * 8u != cb_bytes) return;
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else if (CB == 1) for (uint32_t v = tid; v < nc; v += 512u) {
            const uint2 cw = *(const uint2 *)(m0.cb + (size_t)v * 8u);
            uint4 o;
            o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
            o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
            *(uint4 *)(cbs + (size_t)v * 16u) = o;
        }
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;   /* 只对生产线程(tid < 256)有意义 */
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQM_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {
            if (MODE == 2)
                for (uint32_t i = tid; i < VQM_BM * nt; i += 512u)
                    if (r0 + i % VQM_BM < M) ys[(uint64_t)(base + i / VQM_BM) * M + r0 + i % VQM_BM] = 0.f;
            continue;
        }
        const uint32_t grow = r0 + ar, mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3;
        const bool rv = producer && grow < M;
        const uint8_t *rowp = m.ix + (size_t)(rv ? grow : 0u) * mrow;
        const uint8_t *ep = EXT ? m.ex + (size_t)(rv ? grow : 0u) * erow : NULL;
        float acc[NF][4];
        #pragma unroll
        for (uint32_t f = 0; f < NF; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }
        uint32_t w0 = 0, w1 = 0, eb = 0;
        uint4 bx[CH];
        #pragma unroll
        for (uint32_t c = 0; c < CH; c++) bx[c] = make_uint4(0, 0, 0, 0);
        auto load_round = [&](uint32_t i) {   /* 生产线程: 第 i 轮的位流字 + 位平面字节 + B 片块, 先发进寄存器 */
            const uint32_t bitoff = i * 12u * CPR + sub * 12u * CPT, a = (bitoff >> 5) << 2;
            if (rv) { w0 = __ldg((const unsigned int *)(rowp + a)); w1 = __ldg((const unsigned int *)(rowp + a + 4u));
                      if (EXT) eb = __ldg(ep + ((i * CPR) >> 3)); }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * NP, bt = ch / CPR, bc = ch % CPR;
                if (ch < BN * CPR && bt < nt) bx[c] = __ldg((const uint4 *)(act + (uint64_t)(base + bt) * K + (uint64_t)i * BK + bc * 8u));
            }
        };
        auto decode_into = [&](uint32_t b, uint32_t i) {   /* 生产线程: 把寄存器里第 i 轮的东西解进缓冲 b */
            uint8_t *As = tiles + (size_t)b * TILE, *Bs = As + VQM_BM * BK * 2u;
            const uint32_t sh = (i * 12u * CPR + sub * 12u * CPT) & 31u;
            const uint64_t u = (((uint64_t)w1 << 32) | w0) >> sh;
            #pragma unroll
            for (uint32_t q = 0; q < CPT; q++) {
                uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> ((i * CPR + sub * CPT + q) & 7u)) & 1u) << 12;
                uint4 o;
                if (CB == 0) {
                    const uint2 cw = *(const uint2 *)(cbs + (size_t)v * 8u);
                    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
                    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
                } else if (CB == 1) o = *(const uint4 *)(cbs + (size_t)v * 16u);
                else o = __ldg(cbh + v);
                *(uint4 *)(As + vswz<BK>(ar, sub * CPT + q)) = o;
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * NP;
                if (ch < BN * CPR) *(uint4 *)(Bs + vswz<BK>(ch / CPR, ch % CPR)) = bx[c];
            }
        };
        if (producer) { load_round(0); decode_into(0, 0); if (nit > 1u) load_round(1); }
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            if (producer) {
                if (i + 1u < nit) { decode_into((i + 1u) & 1u, i + 1u); if (i + 2u < nit) load_round(i + 2u); }
            } else {
                const uint8_t *As = tiles + (size_t)(i & 1u) * TILE, *Bs = As + VQM_BM * BK * 2u;
                #pragma unroll
                for (uint32_t kk = 0; kk < BK / 16u; kk++) {
                    uint32_t a[4];
                    {   const uint32_t mt = lane >> 3, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                        vqm_ldsm4(a, As + vswz<BK>(row, kk * 2u + (mt >> 1))); }
                    #pragma unroll
                    for (uint32_t np = 0; np < NF / 2u; np++) {
                        if (2u * np * 8u >= nt) break;
                        uint32_t bfr[4];
                        const uint32_t mt = lane >> 3, tok = (2u * np + (mt >> 1)) * 8u + (lane & 7u);
                        vqm_ldsm4(bfr, Bs + vswz<BK>(tok, kk * 2u + (mt & 1u)));
                        float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                        vqm_mma(t0, a, bfr[0], bfr[1]);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[2 * np][c] += t0[c];
                        if ((2u * np + 1u) * 8u < nt) { vqm_mma(t1, a, bfr[2], bfr[3]);
                            #pragma unroll
                            for (int c = 0; c < 4; c++) acc[2 * np + 1][c] += t1[c]; }
                    }
                }
            }
            __syncthreads();
        }
        if (producer) continue;
        const uint32_t rr[2] = { r0 + wr * 16u + (lane >> 2), r0 + wr * 16u + (lane >> 2) + 8u };
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t r = rr[h];
            if (r >= M) continue;
            __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
            const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
            #pragma unroll
            for (uint32_t f = 0; f < NF; f++) {
                if (f * 8u >= nt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = f * 8u + (lane & 3u) * 2u + e;
                    if (t >= nt) continue;
                    const uint64_t o = (uint64_t)(base + t) * M + r;
                    const float val = v41_bf16r(acc[f][h * 2u + e] * g);
                    if (MODE == 0) g32[o] = val;
                    else if (MODE == 1) h16[o] = v41_vq_swiglu(g32[o], val, clamp);
                    else ys[o] = val;
                }
            }
        }
    }
}

/* ---- V13/V14: 单组 warp 的双缓冲交错(BK 32): 每个 warp 既解下一轮的片(写备用缓冲)又对当前片做 mma, 每轮只一次 __syncthreads ----
 * 为什么: V10~V12 把两件事分给两组 warp, 结果 mma 只剩 8 个 warp 在发, 没赢。这里不分组: 同一 warp 在一轮里先把下一轮的码字解进备用缓冲
 * (查表/转换是 ALU 与 shared 存), 再对当前缓冲发 ldmatrix + mma —— 两串指令没有依赖, 编译器/硬件在 warp 内交错, 16 个 warp 再互相错相;
 * 而且一轮只剩一个 barrier(现核每轮两个)。缓冲 = 2 × (128 + BN) × 32 × 2 B(BN128 = 32 KB), 13 位码本 64 + 32 = 96 KB ≤ 99 ✓。
 * ★逐位同★: 乘积/累加序/舍入点与现核完全一样, 只换了先后与缓冲。 */
template <int EXT, int MODE, uint32_t BN, int CB>
__global__ __launch_bounds__(512) static void vqm_db_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr, const uint4 *cbh) {
    constexpr uint32_t BK = 32u, CPR = BK / 8u, NT = 512u, TPR = NT / VQM_BM, CPT = CPR / TPR;   /* 每行 4 线程各 1 个码字 */
    constexpr uint32_t NF = BN / 8u, NFW = NF / 2u, CH = (BN * CPR + NT - 1u) / NT, TILE = (VQM_BM + BN) * BK * 2u;
    static_assert(CPT >= 1u && NFW >= 2u && (NFW & 1u) == 0u, "分工要整除");
    extern __shared__ __align__(16) uint8_t vqmsh[];
    const uint32_t nc = cb_bytes / 8u, cbsh = CB == 0 ? cb_bytes : (CB == 1 ? nc * 16u : 0u);
    uint8_t *cbs = vqmsh, *tiles = vqmsh + cbsh;
    const int which = MODE;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile, nit = K / BK;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * 8u != cb_bytes) return;
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else if (CB == 1) for (uint32_t v = tid; v < nc; v += NT) {
            const uint2 cw = *(const uint2 *)(m0.cb + (size_t)v * 8u);
            uint4 o;
            o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
            o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
            *(uint4 *)(cbs + (size_t)v * 16u) = o;
        }
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQM_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {
            if (MODE == 2)
                for (uint32_t i = tid; i < VQM_BM * nt; i += NT)
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
                      if (EXT) eb = __ldg(ep + ((i * CPR) >> 3)); }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * NT, bt = ch / CPR, bc = ch % CPR;
                if (ch < BN * CPR && bt < nt) bx[c] = __ldg((const uint4 *)(act + (uint64_t)(base + bt) * K + (uint64_t)i * BK + bc * 8u));
            }
        };
        auto decode_into = [&](uint32_t b, uint32_t i) {
            uint8_t *As = tiles + (size_t)b * TILE, *Bs = As + VQM_BM * BK * 2u;
            const uint32_t sh = (i * 12u * CPR + sub * 12u * CPT) & 31u;
            const uint64_t u = (((uint64_t)w1 << 32) | w0) >> sh;
            #pragma unroll
            for (uint32_t q = 0; q < CPT; q++) {
                uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> ((i * CPR + sub * CPT + q) & 7u)) & 1u) << 12;
                uint4 o;
                if (CB == 0) {
                    const uint2 cw = *(const uint2 *)(cbs + (size_t)v * 8u);
                    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
                    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
                } else if (CB == 1) o = *(const uint4 *)(cbs + (size_t)v * 16u);
                else o = __ldg(cbh + v);
                *(uint4 *)(As + vswz<BK>(ar, sub * CPT + q)) = o;
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * NT;
                if (ch < BN * CPR) *(uint4 *)(Bs + vswz<BK>(ch / CPR, ch % CPR)) = bx[c];
            }
        };
        load_round(0); decode_into(0, 0); if (nit > 1u) load_round(1);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            if (i + 1u < nit) { decode_into((i + 1u) & 1u, i + 1u); if (i + 2u < nit) load_round(i + 2u); }   /* 下一轮的片, 写备用缓冲 */
            const uint8_t *As = tiles + (size_t)(i & 1u) * TILE, *Bs = As + VQM_BM * BK * 2u;             /* 当前片 */
            #pragma unroll
            for (uint32_t kk = 0; kk < BK / 16u; kk++) {
                uint32_t a[4];
                {   const uint32_t mt = lane >> 3, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                    vqm_ldsm4(a, As + vswz<BK>(row, kk * 2u + (mt >> 1))); }
                #pragma unroll
                for (uint32_t np = 0; np < NFW / 2u; np++) {
                    const uint32_t f0 = th * NFW + 2u * np;
                    if (f0 * 8u >= nt) break;
                    uint32_t bfr[4];
                    const uint32_t mt = lane >> 3, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                    vqm_ldsm4(bfr, Bs + vswz<BK>(tok, kk * 2u + (mt & 1u)));
                    float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                    vqm_mma(t0, a, bfr[0], bfr[1]);
                    #pragma unroll
                    for (int c = 0; c < 4; c++) acc[2 * np][c] += t0[c];
                    if ((f0 + 1u) * 8u < nt) { vqm_mma(t1, a, bfr[2], bfr[3]);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[2 * np + 1][c] += t1[c]; }
                }
            }
            __syncthreads();   /* 备用缓冲可读、当前缓冲可覆盖 */
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
                    else if (MODE == 1) h16[o] = v41_vq_swiglu(g32[o], val, clamp);
                    else ys[o] = val;
                }
            }
        }
    }
}
