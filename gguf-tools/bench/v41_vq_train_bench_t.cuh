/* v41_vq_train_bench_t.cuh — 转置专家核的"分段预取 + 寄存器直解"形态 vqst(2026-10-03), 只被 v41_vq_train_bench.cu include(在 _new.cuh 之后, 借它的 vqs_* 原语)。
 *
 * OUT[t][c] (+)= Σ_r G16[t][r]·W[r][c]   (G16 = bf16(g·行增益), vqt_prescale_kernel 出)
 * 【分工】一个 block(16 warp)管 (工作项, SW 个 64 列条) × 全部 R 行; warp w 管第 w 条(SW < 16 时多出来的 warp 只帮着搬)。
 *   mma 里权重当 A: 一个 warp 的 64 列 = 4 个 m16 片; lane (g, q) 解码字列 g 的第 16kk+4q+{0..3} 行(4 个码字),
 *   m16 片 i 的第 m 行 = 码字列 (m & 7) 的第 2i + (m >> 3) 个元素 ⇒ 4 个片正好用完这 4 个码字的 8 个元素(PRMT 拼行对)。
 *   G16 当 B: token 8 个一片, lane (g,q) 取 token g 的第 16kk+4q..+3 行 = 一条 8 B 读, 正好就是 (b0, b1), 不用拼。
 *   k16 组内逻辑 k = 2q+{0,1} ↔ 行 16kk+4q+{0,1}, 2q+8+{0,1} ↔ 16kk+4q+{2,3}: 组仍是同 16 行 ⇒ 与 vqt_kernel 逐位同的前提同 vqs。
 * 【分段】一段 = 64 行: 位流(每行 SW·12 B 的列窗)、位平面(每行 SW B)、G16(NTM 个 token × 64 行)cp.async 进 shared, 两段在途;
 *   每段提前 PD 段把整行(不只列窗)bulk 预取进 L2 —— 同一工作项的几个列块同时在读同几行, L2 去重, DRAM 看到的是整行连续请求。 */
#pragma once
#define VQST_PITCH 192u   /* shared 里位流一行(16 条 × 12 B)。垫到 200 B 消掉 4 组 lane 的 4 路 bank 冲突试过(10-03): 12 位转置三发 16.35 → 16.65 ms, 不是瓶颈, 判负 */
static constexpr uint32_t vqst_smem(uint32_t cbb, int ext, uint32_t ntm, uint32_t s) {
    return cbb + s * (64u * VQST_PITCH + (ext ? 64u * 16u : 0u) + ntm * 128u);
}
template <int EXT, int CBF, uint32_t NTM, uint32_t S>
__global__ __launch_bounds__(VQS_THREADS, 1) static void vqst_kernel(
        float *out, const uint16_t *g16, const uint8_t *blob, const vqp_item *items, uint32_t nitems, const uint32_t *off,
        uint32_t which, uint32_t R, uint32_t C, uint32_t SW, int accumulate, uint32_t cb_bytes, int *bad) {
    constexpr uint32_t NTN = NTM / 8u, BSB = 64u * VQST_PITCH, EXB = EXT ? 64u * 16u : 0u, GTB = NTM * 128u, STB = BSB + EXB + GTB, PD = 2u;
    extern __shared__ __align__(16) uint8_t vqstsh[];
    uint8_t *cbs = vqstsh, *ring = vqstsh + cb_bytes;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u;
    {
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, (int)which, R, C, NULL);
        if (!m0.ok || m0.nc * (CBF ? 16u : 8u) != cb_bytes) { if (tid == 0) atomicExch(bad, 1); return; }
        if (CBF == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQS_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t nu = C / (64u * SW), nwork = nitems * nu, nst = R / 64u, wb = SW * 12u, nwc = SW * 12u / 16u;   /* 每行列窗 wb 字节 = nwc 块 16 B */
    const uint32_t cofs[4] = { ((0u + (q >> 1)) ^ (2u * (g & 3u))) << 4, ((2u + (q >> 1)) ^ (2u * (g & 3u))) << 4,
                               ((4u + (q >> 1)) ^ (2u * (g & 3u))) << 4, ((6u + (q >> 1)) ^ (2u * (g & 3u))) << 4 };
    const bool act = warp < SW;
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / nu];
        const uint32_t c0 = (w % nu) * 64u * SW, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, (int)which, R, C, NULL);
        if (!m.ok) { if (tid == 0) atomicExch(bad, 1); continue; }   /* 整个 block 同一个工作项: 同一判断 */
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3;
        const uint8_t *win = m.ix + (c0 / 8u) * 12u / 8u, *ewin = EXT ? m.ex + c0 / 64u : NULL;   /* 本单元列窗在每行里的起点 */
        auto pf = [&](uint32_t s) {   /* 第 s 段的 64 行整行进 L2 */
            if (s < nst && tid < 64u) {
                vqs_pf_l2(m.ix + (size_t)(64u * s + tid) * mrow, mrow & ~15u);
                if (EXT && tid == 0) vqs_pf_l2(m.ex + (size_t)64u * s * erow, (64u * erow) & ~15u);
            }
        };
        auto issue = [&](uint32_t s) {   /* 第 s 段(行 64s..)→ 槽 s % S */
            uint8_t *st = ring + (s % S) * STB;
            for (uint32_t k = tid; k < 64u * nwc; k += VQS_THREADS) {
                const uint32_t row = k / nwc, ch = k - row * nwc;
                vqs_cp<16>(st + row * VQST_PITCH + ch * 16u, win + (size_t)(64u * s + row) * mrow + ch * 16u);
            }
            if (EXT) for (uint32_t k = tid; k < 64u * (SW / 4u); k += VQS_THREADS) {
                const uint32_t row = k / (SW / 4u), ch = k - row * (SW / 4u);
                vqs_cp<4>(st + BSB + row * 16u + ((ch ^ ((row >> 2) & 3u)) << 2), ewin + (size_t)(64u * s + row) * erow + ch * 4u);
            }
            uint8_t *gd = st + BSB + EXB;
            for (uint32_t k = tid; k < nt * 8u; k += VQS_THREADS) {
                const uint32_t t = k >> 3, c = k & 7u;
                vqs_cp<16>(gd + t * 128u + ((c ^ (2u * (t & 3u))) << 4), g16 + (uint64_t)(base + t) * R + 64u * s + 8u * c);
            }
        };
        #pragma unroll
        for (uint32_t s = 0; s < PD; s++) pf(s);
        #pragma unroll
        for (uint32_t s = 0; s + 1u < S; s++) { if (s < nst) issue(s); vqs_commit(); }
        const uint32_t nnt = (nt + 7u) >> 3;
        float acc[4][NTN][4];
        #pragma unroll
        for (uint32_t i = 0; i < 4u; i++)
            #pragma unroll
            for (uint32_t j = 0; j < NTN; j++) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f; }
        /* 本 lane 的码字列在列窗里的位偏移: 第 w 条的第 g 个码字 */
        const uint32_t cbit = 12u * (8u * warp + g), cwo = (cbit >> 5) << 2, csh = cbit & 31u;
        auto run = [&](auto nntc) {
            constexpr uint32_t NNT = decltype(nntc)::value;
            for (uint32_t s = 0; s < nst; s++) {
                vqs_wait<(int)S - 2>();
                __syncthreads();
                if (s + S - 1u < nst) issue(s + S - 1u);
                vqs_commit();
                pf(s + PD);
                if (!act) continue;
                const uint8_t *st = ring + (s % S) * STB, *gs = st + BSB + EXB + 4u * 0u + 8u * (q & 1u) + g * 128u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; kk++) {
                    uint4 E[4];
                    #pragma unroll
                    for (uint32_t u = 0; u < 4u; u++) {
                        const uint32_t row = 16u * kk + 4u * q + u;
                        const uint8_t *p = st + row * VQST_PITCH + cwo;
                        uint32_t v = __funnelshift_r(*(const uint32_t *)p, *(const uint32_t *)(p + 4), csh) & 0xFFFu;
                        if (EXT) v |= (((uint32_t)st[BSB + row * 16u + ((((warp >> 2) ^ q) << 2) | (warp & 3u))] >> g) & 1u) << 12;   /* 位平面行内按字异或 (行/4)&3 = q: 4 组 lane 落 4 个 bank */
                        E[u] = vqm_cw<CBF>(cbs, v);
                    }
                    const uint8_t *bp = gs + cofs[kk];
                    uint2 x[NTN];
                    #pragma unroll
                    for (uint32_t j = 0; j < NNT; j++) x[j] = *(const uint2 *)(bp + j * 8u * 128u);
                    #pragma unroll
                    for (uint32_t i = 0; i < 4u; i++) {
                        /* 片 i: 第 m 行 = 码字列 m&7 的第 2i + (m>>3) 个元素; lane 的 m = g 取元素 2i, m = g+8 取 2i+1 */
                        const uint32_t w0 = i == 0u ? E[0].x : i == 1u ? E[0].y : i == 2u ? E[0].z : E[0].w;
                        const uint32_t w1 = i == 0u ? E[1].x : i == 1u ? E[1].y : i == 2u ? E[1].z : E[1].w;
                        const uint32_t w2 = i == 0u ? E[2].x : i == 1u ? E[2].y : i == 2u ? E[2].z : E[2].w;
                        const uint32_t w3 = i == 0u ? E[3].x : i == 1u ? E[3].y : i == 2u ? E[3].z : E[3].w;
                        const uint32_t a[4] = { __byte_perm(w0, w1, 0x5410u), __byte_perm(w0, w1, 0x7632u),
                                                __byte_perm(w2, w3, 0x5410u), __byte_perm(w2, w3, 0x7632u) };
                        #pragma unroll
                        for (uint32_t j = 0; j < NNT; j++) {
                            float t[4];
                            vqs_mma0(t, a, x[j].x, x[j].y);
                            #pragma unroll
                            for (int c = 0; c < 4; c++) acc[i][j][c] += t[c];
                        }
                    }
                }
            }
        };
#define VQS_NNT(n) std::integral_constant<uint32_t, ((n) < NTN ? (n) : NTN)>{}
        switch (nnt) {
        case 1: run(VQS_NNT(1)); break;  case 2: run(VQS_NNT(2)); break;  case 3: run(VQS_NNT(3)); break;  default: run(VQS_NNT(4)); break;
        }
#undef VQS_NNT
        __syncthreads();
        /* 出口: 片 i 的 c0/c1 = (列 码字列 g 的元素 2i, token 8j+2q+{0,1}), c2/c3 = (元素 2i+1, 同 token) ⇒ 一个 token 拿码字列 g 的 8 个连续列 */
        if (act) {
            #pragma unroll
            for (uint32_t j = 0; j < NTN; j++) {
                if (j >= nnt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = 8u * j + 2u * q + e;
                    if (t >= nt) continue;
                    float *o = out + (uint64_t)(base + t) * C + c0 + 64u * warp + 8u * g;
                    #pragma unroll
                    for (uint32_t i = 0; i < 4u; i++) {
                        const float v0 = acc[i][j][e], v1 = acc[i][j][2u + e];
                        o[2u * i] = accumulate ? o[2u * i] + v0 : v0;
                        o[2u * i + 1u] = accumulate ? o[2u * i + 1u] + v1 : v1;
                    }
                }
            }
        }
    }
}
