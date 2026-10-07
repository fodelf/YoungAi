/* v41_vq_persist_v0.cuh — 引擎 V0 常驻核(v41_vq_gu/dn_persist_n_kernel + v41_vq_stream)与 n=1 参照核的逐式抄本(2026-10-07 从 v41_vq_persist_n_bench.cu 拆出守 500 行),
   被它在常量/打点头之后包含。内容未改, 只是搬家; 核体与 src/cuda/cuda_vq_persist.inc.cu 的对应关系见 bench 文件头。 */
/* ---- 引擎同款(逐字, 见 src/cuda/cuda_vq_row.inc.cu / cuda_vq_persist.inc.cu) ---- */
__device__ __forceinline__ static float bf16r(float x) {
    uint32_t u = __float_as_uint(x); u += 0x7fffu + ((u >> 16) & 1u); return __uint_as_float(u & 0xffff0000u);
}
__device__ __forceinline__ static __half2 e4m3x2_to_half2(uint32_t two) {
    uint32_t h; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    __half2 out; memcpy(&out, &h, 4); return out;
}
__device__ __forceinline__ static void cw(uint32_t v, const uint8_t *cbs, float *c) {   /* = v41_vq_cw<1> */
    const uint2 w = *(const uint2 *)(cbs + (size_t)v * 8u);
    const __half2 p0 = e4m3x2_to_half2(w.x), p1 = e4m3x2_to_half2(w.x >> 16), p2 = e4m3x2_to_half2(w.y), p3 = e4m3x2_to_half2(w.y >> 16);
    const float2 e0 = __half22float2(p0), e1 = __half22float2(p1), e2 = __half22float2(p2), e3 = __half22float2(p3);
    c[0] = e0.x; c[1] = e0.y; c[2] = e1.x; c[3] = e1.y; c[4] = e2.x; c[5] = e2.y; c[6] = e3.x; c[7] = e3.y;
}
__device__ __forceinline__ static float dot8_cw(const float *c, uint4 xw) {   /* = v41_vq_dot8_cw */
    return c[0] * __uint_as_float(xw.x << 16) + c[1] * __uint_as_float(xw.x & 0xffff0000u)
         + c[2] * __uint_as_float(xw.y << 16) + c[3] * __uint_as_float(xw.y & 0xffff0000u)
         + c[4] * __uint_as_float(xw.z << 16) + c[5] * __uint_as_float(xw.z & 0xffff0000u)
         + c[6] * __uint_as_float(xw.w << 16) + c[7] * __uint_as_float(xw.w & 0xffff0000u);
}
__device__ __forceinline__ static float dot8(uint32_t v, uint4 xw, const uint8_t *cbs) { float c[8]; cw(v, cbs, c); return dot8_cw(c, xw); }
__device__ __forceinline__ static uint32_t sel3(uint32_t r, uint32_t w0, uint32_t w1, uint32_t w2) { return r == 0u ? w0 : (r == 1u ? w1 : w2); }
typedef struct { uint32_t w0, w1, w2; } blk_t;
/* V0_LDCS(10-07 晚, 编译期开关): 位流块读改 ld.global.cs(流式、优先驱逐, 不占 L1) —— 验"13 位层 64 KB 码本把 L1 挤到 36 KB,
 * 位流流式读再把 M=2 的两条激活(20 KB)挤出 L1"这个假设: 引擎里 gu<13> 比 gu<12> 慢 32%(543 vs 411 µs), 指令只多 ~5%。 */
__device__ __forceinline__ static uint32_t blk_ld(const uint32_t *p) {
#ifdef V0_LDCS
    return __ldcs(p);
#else
    return *p;
#endif
}
__device__ __forceinline__ static blk_t blk_load(const uint32_t *blk, uint32_t nw) {
    const uint32_t lane = threadIdx.x & 31u;
    blk_t b; b.w0 = 0u; b.w1 = 0u; b.w2 = 0u;
    if (lane < nw) b.w0 = blk_ld(blk + lane);
    if (lane + 32u < nw) b.w1 = blk_ld(blk + lane + 32u);
    if (lane + 64u < nw) b.w2 = blk_ld(blk + lane + 64u);
    return b;
}
typedef struct { const uint8_t *gr, *ix; uint32_t rowb; } mat_t;   /* rowb: 一行位流字节(gate/up 960, down 432) */
__device__ __forceinline__ static const uint32_t *row_ptr(const mat_t &m, uint32_t r) { return (const uint32_t *)(m.ix + (size_t)r * m.rowb); }
__device__ __forceinline__ static blk_t row_first_blk(const mat_t &m, uint32_t r) { return blk_load(row_ptr(m, r), 8u * 12u); }

/* = v41_vq_stream<12, 0, M>: M=1 走 dot8(= cw + dot8_cw 内联, 与 n=1 核同一棵树); M>1 码字解一次、每 token 各一次 dot8_cw */
template <int M, uint32_t NI>   /* NI = 每行索引数(gate/up 640 = 20 轮, down 288 = 9 轮) */
__device__ __forceinline__ static void stream_m(const mat_t &m, uint32_t r0, uint32_t n, const uint32_t *const *xs, uint32_t nt,
                                                const uint8_t *cbs, blk_t *carry, const uint32_t *next, float *res) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = 12u;
    const uint32_t R = NI >> 5, G = n * R;
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *base = row_ptr(m, r0);
    float g1 = 0.f;
    if (lane < n) { __half gh; memcpy(&gh, m.gr + (size_t)(r0 + lane) * 2u, 2); g1 = __half2float(gh); }
    blk_t cur = *carry;
    float acc[M];
    #pragma unroll
    for (int j = 0; j < M; j++) { acc[j] = 0.f; res[j] = 0.f; }
    uint32_t kin = 0, row = 0;
    for (uint32_t g0 = 0; g0 < G; g0 += 8u) {
        blk_t nxt;
        if (g0 + 8u < G) { const uint32_t rem = G - g0 - 8u, nr = rem < 8u ? rem : 8u; nxt = blk_load(base + (size_t)(g0 + 8u) * MB, nr * MB); }
        else if (next) nxt = blk_load(next, 8u * MB);
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; }
        const uint32_t rounds = G - g0 < 8u ? G - g0 : 8u;
        #pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            if (k >= rounds) break;
            uint4 xa0;
            if (M == 1) xa0 = *(const uint4 *)(xs[0] + (size_t)(kin * 32u + lane) * 4u);
            const uint32_t f = k * MB;
            const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
            const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
            const uint32_t lo = __shfl_sync(0xffffffffu, sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
            const uint32_t hi = __shfl_sync(0xffffffffu, sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
            const uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
            if (M == 1) acc[0] += dot8(v, xa0, cbs);
            else {
                float c[8]; cw(v, cbs, c);
                #pragma unroll
                for (int j = 0; j < M; j++) if ((uint32_t)j < nt) acc[j] += dot8_cw(c, *(const uint4 *)(xs[j] + (size_t)(kin * 32u + lane) * 4u));
            }
            if (++kin == R) {
                #pragma unroll
                for (int j = 0; j < M; j++) if ((uint32_t)j < nt) for (int o = 16; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
                const float a1 = __shfl_sync(0xffffffffu, g1, (int)row);
                #pragma unroll
                for (int j = 0; j < M; j++) { const float val = acc[j] * a1; if (lane == row && (uint32_t)j < nt) res[j] = val; acc[j] = 0.f; }
                kin = 0u; row++;
            }
        }
        cur = nxt;
    }
    *carry = cur;
}
#define SEG(U, UEND, ROWS, P, R0, N) \
    const uint32_t P = (U) / (ROWS), R0 = (U) % (ROWS); \
    uint32_t N = (UEND) - (U); if (N > (ROWS) - R0) N = (ROWS) - R0; if (N > 32u) N = 32u
typedef struct { const uint8_t *g[NUMAX], *u[NUMAX]; } mats_t;   /* 这一层唯一专家的 gate/up 起点表(全局内存, 与引擎读槽表同) */
#define MAT(TAB, E, W) ({ mat_t m_; m_.gr = (W) ? (TAB)->u[(E)] : (TAB)->g[(E)]; m_.ix = m_.gr + MID * 2u; m_.rowb = ROWB; m_; })
/* down 矩阵(10-07 加): OUT=5120 行 × MID=2304 列 ⇒ 每行 288 个索引 = 9 轮 × 32, 位流 432 B; 行增益 f16[OUT] 在前 */
#define OUT 5120u
#define NIDX_DN (MID / 8u)
#define ROWB_DN (NIDX_DN * 12u / 8u)
typedef struct { const uint8_t *d[NUMAX]; } dmats_t;
#define DMAT(TAB, E) ({ mat_t m_; m_.gr = (TAB)->d[(E)]; m_.ix = m_.gr + OUT * 2u; m_.rowb = ROWB_DN; m_; })
__device__ __forceinline__ static uint16_t swiglu(float gs, float us) {
    const float gv = bf16r(gs), uv = bf16r(us);
    return (uint16_t)(__float_as_uint(bf16r(gv / (1.f + __expf(-gv)) * uv)) >> 16);
}
/* = v41_vqpn_groups: 按 order 把同一专家的对归组(≤ M 个) */
template <int M>
__device__ __forceinline__ static void groups(const int32_t *sel, const int32_t *order, uint32_t np, uint32_t *gq, uint32_t *gm, uint32_t *ng) {
    if (threadIdx.x == 0) {
        uint32_t g = 0, q = 0;
        while (q < np) {
            const int32_t e = sel[order[q]];
            uint32_t m = 1u;
            while (q + m < np && sel[order[q + m]] == e && m < (uint32_t)M) m++;
            gq[g] = q; gm[g] = m; g++; q += m;
        }
        *ng = g;
    }
}
/* = v41_vq_gu_persist_n_kernel<12,0,M,32>(去掉 PDL/格式校验). XSH: 激活先搬进 shared(码本之后), 否则走全局 */
template <int M, int XSH>
__global__ static void __launch_bounds__(1024, 1) gu_persist_n(uint16_t *h, const mats_t *__restrict__ mt, const int32_t *sel, const int32_t *order,
                                                              const uint32_t *x, uint32_t np, uint32_t n_tok, const uint8_t *cb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < CBB / 16u; i += blockDim.x) ((uint4 *)vqsh)[i] = ((const uint4 *)cb)[i];
    if (XSH) for (uint32_t i = threadIdx.x; i < n_tok * IN * 2u / 16u; i += blockDim.x) ((uint4 *)(vqsh + CBB))[i] = ((const uint4 *)x)[i];
    groups<M>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    ts_begin();
    const uint32_t *xb = XSH ? (const uint32_t *)(vqsh + CBB) : x;
    const uint32_t gw = blockIdx.x * WARPS + (threadIdx.x >> 5), nw = gridDim.x * WARPS;
    const uint32_t total = ng * MID, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        SEG(u, uend, MID, g, r0, n);
        u += n;
        const uint32_t q = gq[g], nt = gm[g];
        const int32_t e = sel[order[q]];
        const mat_t mg = MAT(mt, e, 0u), mu = MAT(mt, e, 1u);
        const uint32_t *xs[M]; uint32_t pr[M];
        #pragma unroll
        for (int j = 0; j < M; j++) { pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)]; xs[j] = xb + (uint64_t)(pr[j] / K) * (IN / 2u); }
        blk_t carry = row_first_blk(mg, r0);
        float gs[M], us[M];
        stream_m<M, NIDX>(mg, r0, n, xs, nt, vqsh, &carry, row_ptr(mu, r0), gs);
        stream_m<M, NIDX>(mu, r0, n, xs, nt, vqsh, &carry, NULL, us);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < nt) h[(uint64_t)pr[j] * MID + r0 + lane] = swiglu(gs[j], us[j]);
        }
    }
    ts_end();
}
/* = v41_vq_dn_persist_n_kernel<12,0,M,32>(去掉 PDL/格式校验): 工作项 (唯一专家组, OUT 行), h [np][MID] bf16 走全局, 出 partial[pr][OUT] f32 */
template <int M>
__global__ static void __launch_bounds__(1024, 1) dn_persist_n(float *partial, const dmats_t *__restrict__ mt, const int32_t *sel, const int32_t *order,
                                                              const uint32_t *h, uint32_t np, const uint8_t *cb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < CBB / 16u; i += blockDim.x) ((uint4 *)vqsh)[i] = ((const uint4 *)cb)[i];
    groups<M>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    ts_begin();
    const uint32_t gw = blockIdx.x * WARPS + (threadIdx.x >> 5), nw = gridDim.x * WARPS;
    const uint32_t total = ng * OUT, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        SEG(u, uend, OUT, g, r0, n);
        u += n;
        const uint32_t q = gq[g], nt = gm[g];
        const int32_t e = sel[order[q]];
        uint32_t pr[M]; const uint32_t *hs[M];
        #pragma unroll
        for (int j = 0; j < M; j++) { pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)]; hs[j] = h + (uint64_t)pr[j] * (MID / 2u); }
        const mat_t md = DMAT(mt, e);
        blk_t carry = row_first_blk(md, r0);
        float ys[M];
        stream_m<M, NIDX_DN>(md, r0, n, hs, nt, vqsh, &carry, NULL, ys);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < nt) partial[(uint64_t)pr[j] * OUT + r0 + lane] = bf16r(ys[j]);
        }
    }
    ts_end();
}
/* 参照: n=1 引擎核(= v41_vq_gu_persist_kernel<12,0>, 6 个专家各一对; 激活进 shared) —— 与 v41_vq_persist_bench 的抄本同 */
__global__ static void __launch_bounds__(1024, 1) gu_persist1(uint16_t *h, const mats_t *__restrict__ mt, const int32_t *sel, const uint32_t *xg,
                                                             uint32_t np, const uint8_t *cb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t gw = blockIdx.x * WARPS + (threadIdx.x >> 5), nw = gridDim.x * WARPS;
    const uint32_t total = np * MID, per = (total + nw - 1u) / nw, lane = threadIdx.x & 31u;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    for (uint32_t i = threadIdx.x; i < CBB / 16u; i += blockDim.x) ((uint4 *)vqsh)[i] = ((const uint4 *)cb)[i];
    for (uint32_t i = threadIdx.x; i < IN * 2u / 16u; i += blockDim.x) ((uint4 *)(vqsh + CBB))[i] = ((const uint4 *)xg)[i];
    __syncthreads();
    const uint32_t *x = (const uint32_t *)(vqsh + CBB);
    while (u < uend) {
        SEG(u, uend, MID, p, r0, n);
        u += n;
        const int32_t e = sel[p];
        const mat_t mg = MAT(mt, e, 0u), mu = MAT(mt, e, 1u);
        blk_t carry = row_first_blk(mg, r0);
        const uint32_t *xs[1] = { x };
        float gs, us;
        stream_m<1, NIDX>(mg, r0, n, xs, 1u, vqsh, &carry, row_ptr(mu, r0), &gs);
        stream_m<1, NIDX>(mu, r0, n, xs, 1u, vqsh, &carry, NULL, &us);
        if (lane < n) h[(uint64_t)p * MID + r0 + lane] = swiglu(gs, us);
    }
}

