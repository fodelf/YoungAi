/* v41_vq_persist_n_bench.cu — 投机验证批(n ≥ 2 行)的 VQ 专家常驻核(gateup)形态微基准(spark 本机跑; 2026-09-29 立)。
 *
 * 【为什么要它】钉 k 的账(09-29): 验证 2 行 43.9 ms / 4 行 61 ms, 纯解码走图 33.6 ⇒ 第一行多 10.3 ms、之后每行 8.7, 而字节账每行只该 3.9
 * (每多一行多 ~3.5 个唯一专家 = 0.9 GB)。差的 5~6 ms/行里专家核是最大嫌疑, 可它只有 n=1 的抄本(v41_vq_persist_bench), 验证批走的
 * 是另一个核(v41_vq_gu_persist_n_kernel: 工作项 = 唯一专家 × 行, 组 ≤ M=2 个 token, 激活走全局), 从没单独量过。
 *
 * 【比什么】核体 = 引擎 v41_vq_gu_persist_n_kernel<12,0,2,32> + v41_vq_stream<12,0,M> 的逐式抄本(组表/段切分/跨块流水/每 token 各一次
 * v41_vq_dot8_cw 都照抄, 去掉 PDL/格式校验); 选中表按真实重合度生成(每个新 token 的 6 个专家里 ~40% 与前面的 token 重复 ⇒ 唯一专家
 * n=1/2/4/6 ≈ 6/10/16.7/21.7, 与 [moe-uniq] 实测同档)。
 *   V0 = 引擎 n 行核; V1 = 激活先搬进 shared(n × 10 KB, 12 位码本 32 KB 之后; n=1 核就是这么做的); V2 = 不分组(M=1, 每对一个工作项,
 *   被几个 token 选中的专家读几遍 —— 回答"分组到底值多少")。参照 = n=1 引擎核(v41_vq_gu_persist_kernel)算 6 个专家。
 * ★判据★: 各变体每 (token, 行) 的 h 与 V0 逐位同; n 行核 n=1 时与 n=1 核逐位同(同一个 dot8 表达式树)。
 * 用法: nvcc -O3 -arch=native -o v41_vq_persist_n_bench v41_vq_persist_n_bench.cu && ./v41_vq_persist_n_bench [层数=80] [遍数=3] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define WARPS 32u
#define K 6u           /* top-k */
#define MID 2304u
#define IN 5120u
#define NIDX (IN / 8u) /* 一行 640 个索引 = 20 轮 × 32 */
#define ROWB (NIDX * 12u / 8u)   /* 一行位流 960 B */
#define CBB (4096u * 8u)         /* 12 位层码本: 4096 词 × 8 个 E4M3 */
#define NMAX 6u
#define NUMAX (NMAX * K)         /* 一层最多的唯一专家数(全不重合) */
#define MAXP 64u                 /* = V41_VQPN_MAXP */

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
__device__ __forceinline__ static blk_t blk_load(const uint32_t *blk, uint32_t nw) {
    const uint32_t lane = threadIdx.x & 31u;
    blk_t b; b.w0 = 0u; b.w1 = 0u; b.w2 = 0u;
    if (lane < nw) b.w0 = blk[lane];
    if (lane + 32u < nw) b.w1 = blk[lane + 32u];
    if (lane + 64u < nw) b.w2 = blk[lane + 64u];
    return b;
}
typedef struct { const uint8_t *gr, *ix; } mat_t;
__device__ __forceinline__ static const uint32_t *row_ptr(const mat_t &m, uint32_t r) { return (const uint32_t *)(m.ix + (size_t)r * ROWB); }
__device__ __forceinline__ static blk_t row_first_blk(const mat_t &m, uint32_t r) { return blk_load(row_ptr(m, r), 8u * 12u); }

/* = v41_vq_stream<12, 0, M>: M=1 走 dot8(= cw + dot8_cw 内联, 与 n=1 核同一棵树); M>1 码字解一次、每 token 各一次 dot8_cw */
template <int M>
__device__ __forceinline__ static void stream_m(const mat_t &m, uint32_t r0, uint32_t n, const uint32_t *const *xs, uint32_t nt,
                                                const uint8_t *cbs, blk_t *carry, const uint32_t *next, float *res) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = 12u;
    const uint32_t R = NIDX >> 5, G = n * R;
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
#define MAT(TAB, E, W) ({ mat_t m_; m_.gr = (W) ? (TAB)->u[(E)] : (TAB)->g[(E)]; m_.ix = m_.gr + MID * 2u; m_; })
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
        stream_m<M>(mg, r0, n, xs, nt, vqsh, &carry, row_ptr(mu, r0), gs);
        stream_m<M>(mu, r0, n, xs, nt, vqsh, &carry, NULL, us);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < nt) h[(uint64_t)pr[j] * MID + r0 + lane] = swiglu(gs[j], us[j]);
        }
    }
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
        stream_m<1>(mg, r0, n, xs, 1u, vqsh, &carry, row_ptr(mu, r0), &gs);
        stream_m<1>(mu, r0, n, xs, 1u, vqsh, &carry, NULL, &us);
        if (lane < n) h[(uint64_t)p * MID + r0 + lane] = swiglu(gs, us);
    }
}

static uint32_t g_rng = 0x9e3779b9u;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }

/* 一层的选中表: token 0 取 6 个新专家; 之后每个 token 的每一格 40% 重复前面某个 token 用过的(且本 token 没选过), 否则新专家。
 * 唯一专家数 n=1/2/4/6 ≈ 6/10/16/21, 与引擎 [moe-uniq] 实测(6/9.96/16.66/21.7)同档。返回唯一专家数; order = 对按专家号稳定排序。 */
static uint32_t make_sel(int32_t *sel, int32_t *order, uint32_t n) {
    uint32_t nu = 0;
    for (uint32_t t = 0; t < n; t++) for (uint32_t k = 0; k < K; k++) {
        int32_t e = -1;
        if (t > 0 && (rnd() % 100u) < 40u) {
            for (int tries = 0; tries < 16 && e < 0; tries++) {
                const int32_t cand = sel[rnd() % (t * K)];
                bool dup = false; for (uint32_t j = 0; j < k; j++) dup |= sel[t * K + j] == cand;
                if (!dup) e = cand;
            }
        }
        if (e < 0) e = (int32_t)nu++;
        sel[t * K + k] = e;
    }
    const uint32_t np = n * K;
    for (uint32_t i = 0; i < np; i++) order[i] = (int32_t)i;
    for (uint32_t a = 1; a < np; a++) { int32_t v = order[a]; uint32_t b = a; while (b > 0 && sel[order[b - 1]] > sel[v]) { order[b] = order[b - 1]; b--; } order[b] = v; }
    return nu;
}

int main(int argc, char **argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 80, reps = argc > 2 ? atoi(argv[2]) : 3;
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    const uint64_t mat_stride = (uint64_t)MID * 2u + (uint64_t)MID * ROWB;   /* 行增益 + 位流 */
    const uint64_t layer_bytes = mat_stride * 2u * NUMAX;                       /* 每层按最多唯一专家数预留 */
    const uint64_t total = layer_bytes * (uint64_t)layers + 4096u;
    uint8_t *buf = NULL, *cb = NULL; uint32_t *x = NULL; uint16_t *h = NULL, *h2 = NULL;
    CK(cudaMalloc(&buf, total)); CK(cudaMalloc(&cb, CBB)); CK(cudaMalloc(&x, (size_t)NMAX * IN * 2u));
    CK(cudaMalloc(&h, (size_t)NMAX * K * MID * 2u)); CK(cudaMalloc(&h2, (size_t)NMAX * K * MID * 2u));
    {   /* 随机位流/增益(f16 ~1)/码本(E4M3 去掉 NaN 码)/激活(bf16 ~±1) */
        const size_t chunk = 64u << 20; uint32_t *hb = (uint32_t *)malloc(chunk);
        for (uint64_t off = 0; off < total; off += chunk) {
            const size_t nb = total - off < chunk ? (size_t)(total - off) : chunk;
            for (size_t i = 0; i < nb / 4; i++) hb[i] = rnd();
            CK(cudaMemcpy(buf + off, hb, nb & ~(size_t)3, cudaMemcpyHostToDevice));
        }
        for (int L = 0; L < layers; L++) for (uint32_t m = 0; m < 2u * NUMAX; m++) {
            uint16_t g[MID]; for (uint32_t i = 0; i < MID; i++) g[i] = 0x3c00u;
            CK(cudaMemcpy(buf + (uint64_t)L * layer_bytes + m * mat_stride, g, sizeof g, cudaMemcpyHostToDevice));
        }
        uint8_t hc[CBB]; for (uint32_t i = 0; i < CBB; i++) { uint8_t b = (uint8_t)rnd(); if ((b & 0x7f) == 0x7f) b &= 0xfe; hc[i] = b; }
        CK(cudaMemcpy(cb, hc, CBB, cudaMemcpyHostToDevice));
        uint16_t *hx = (uint16_t *)malloc((size_t)NMAX * IN * 2u);
        for (uint32_t i = 0; i < NMAX * IN; i++) hx[i] = (uint16_t)(0x3f00u | (rnd() & 0x80ffu));
        CK(cudaMemcpy(x, hx, (size_t)NMAX * IN * 2u, cudaMemcpyHostToDevice));
        free(hb); free(hx);
    }
    const uint32_t shm1 = CBB + IN * 2u;
    CK(cudaFuncSetAttribute(gu_persist1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm1));
    CK(cudaFuncSetAttribute(gu_persist_n<2, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)CBB));
    CK(cudaFuncSetAttribute(gu_persist_n<1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)CBB));
    CK(cudaFuncSetAttribute(gu_persist_n<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(CBB + NMAX * IN * 2u)));
    /* 每层: 选中表 + 唯一专家的矩阵表(专家 e 的 gate = 层基 + 2e·stride, up = +(2e+1)·stride) */
    mats_t *dtab = NULL; int32_t *dsel = NULL, *dord = NULL;
    CK(cudaMalloc(&dtab, sizeof(mats_t) * (size_t)layers)); CK(cudaMalloc(&dsel, (size_t)layers * MAXP * 4)); CK(cudaMalloc(&dord, (size_t)layers * MAXP * 4));
    mats_t *tab = (mats_t *)malloc(sizeof(mats_t) * (size_t)layers);
    int32_t *hsel = (int32_t *)malloc((size_t)layers * MAXP * 4), *hord = (int32_t *)malloc((size_t)layers * MAXP * 4);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    uint16_t *ho = (uint16_t *)malloc((size_t)NMAX * K * MID * 2u), *ho2 = (uint16_t *)malloc((size_t)NMAX * K * MID * 2u);
    printf("v3 gateup 验证批常驻核抄本: %d SM × 1024 线程, 每层唯一专家 × 2 × %u 行 × %u B, %d 层, %d 遍\n", nsm, MID, ROWB, layers, reps);
    const uint32_t ns[4] = { 1u, 2u, 4u, 6u };
    for (uint32_t ni = 0; ni < 4; ni++) {
        const uint32_t n = ns[ni], np = n * K;
        double nu_sum = 0;
        for (int L = 0; L < layers; L++) {
            const uint32_t nu = make_sel(hsel + (size_t)L * MAXP, hord + (size_t)L * MAXP, n); nu_sum += nu;
            for (uint32_t e = 0; e < NUMAX; e++) { tab[L].g[e] = buf + (uint64_t)L * layer_bytes + (2u * e) * mat_stride; tab[L].u[e] = tab[L].g[e] + mat_stride; }
        }
        CK(cudaMemcpy(dtab, tab, sizeof(mats_t) * (size_t)layers, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsel, hsel, (size_t)layers * MAXP * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dord, hord, (size_t)layers * MAXP * 4, cudaMemcpyHostToDevice));
        const double nu_avg = nu_sum / layers, lay_mb = nu_avg * 2.0 * MID * ROWB / 1e6;
        auto launch = [&](int v, int L, uint16_t *dst) {
            const int32_t *s = dsel + (size_t)L * MAXP, *o = dord + (size_t)L * MAXP;
            switch (v) {
            case 0: gu_persist_n<2, 0><<<nsm, 1024, CBB>>>(dst, dtab + L, s, o, x, np, n, cb); break;
            case 1: gu_persist_n<2, 1><<<nsm, 1024, CBB + n * IN * 2u>>>(dst, dtab + L, s, o, x, np, n, cb); break;
            case 2: gu_persist_n<1, 0><<<nsm, 1024, CBB>>>(dst, dtab + L, s, o, x, np, n, cb); break;
            default: gu_persist1<<<nsm, 1024, shm1>>>(dst, dtab + L, s, x, np, cb); break;   /* 只在 n=1 有意义 */
            }
        };
        const char *vname[4] = { "V0 引擎 n 行核", "V1 激活进 shared", "V2 不分组(M=1)", "参照 n=1 核" };
        const int nv = n == 1u ? 4 : 3;
        printf("== n=%u: 每层唯一专家 %.2f / %u 对, 位流 %.1f MB/层\n", n, nu_avg, np, lay_mb);
        /* 逐位门(层 0): 各变体 vs V0 */
        launch(0, 0, h); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ho, h, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost));
        for (int v = 1; v < nv; v++) {
            CK(cudaMemset(h2, 0xFF, (size_t)np * MID * 2u)); launch(v, 0, h2); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ho2, h2, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost));
            printf("   %-16s vs V0: %s\n", vname[v], memcmp(ho, ho2, (size_t)np * MID * 2u) == 0 ? "逐位同 ✓" : "★不同★");
        }
        for (int rep = 0; rep < reps; rep++) {
            printf("   第 %d 遍:", rep + 1);
            for (int v = 0; v < nv; v++) {
                for (int L = 0; L < 4; L++) launch(v, L, h); CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0)); for (int L = 0; L < layers; L++) launch(v, L, h); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); const double us = (double)ms * 1e3 / layers;
                printf("  %s %6.1f µs/层 %5.0f GB/s(唯一字节)", vname[v], us, lay_mb / us * 1e3);
            }
            printf("\n");
        }
    }
    return 0;
}
