/* v41_vq_persist_bench.cu — V4.1 v3 专家"解码即乘"常驻核(gateup, n=1)的形态微基准(spark 本机跑; 2026-09-24 立)。
 *
 * 【为什么要它】专家核是投机一轮里最大的一项(验 4 行 28.4 ms / 纯解码 10.55 ms, 直发), 只到 ~152 GB/s(读墙 230)。
 * 引擎一趟要装 113 GB 模型 + 几分钟, 换形态先在这里过。mem_ceiling ⑧ 已排除"行分配/DRAM 行局部性"(三种分配 223~231),
 * 但同一把尺量出: 位流起点错开 8 B(没对齐 32 B 扇区)纯读 231 → 199 GB/s。真核的位流起点 = 载荷 + 32 + 行数×2,
 * 载荷只保证 8 B 对齐(首载荷偏移 9232 ≡ 16 mod 32) ⇒ 大部分位流是错位的。这把尺问: 真核的算力形态下, 对齐值多少。
 *
 * 【比什么】核体 = 引擎 v41_vq_gu_persist_kernel<12,0> + v41_vq_stream<12,0,1> 的逐式抄本(段切分/跨块流水/段首块/
 * gate 末块装 up 首块/行增益/SwiGLU 前的 bf16 舍入都照抄); 数据是随机的(位流/E4M3 码本/bf16 激活), 只量形态不量数值。
 * 一层 = 6 个专家的 gate(2304×5120)与 up, 各 13.27 MB, 层与层在 ~4 GB 里往后挪(L2 24 MB 装不下, 每层都是冷读)。
 * 变体 = 位流起点相对 128 B 线的错开量(0 / 8 / 16 / 24 / 64 B)。
 * 读法: 错开 8~24 B 那几档 ≈ 引擎实测(gateup 直发 7.10 ms/40 层 = 177 µs/层)⇒ 抄本是忠实的; 错开 0 那档比它快多少,
 * 就是"装载时把位流挪到 128 B 对齐"这一刀的上限。
 * 用法: nvcc -O3 -arch=native -o v41_vq_persist_bench v41_vq_persist_bench.cu && ./v41_vq_persist_bench [层数=120] [遍数=3] [散布档 GB=40, 0=不跑]
 *   散布档要一次 cudaMalloc 这么多显存(121 GB 统一内存的 spark 上 40 GB 安全; 引擎在跑时别开)。 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define WARPS 32u
#define NP 6u          /* 一层 6 个专家(纯解码 top-6) */
#define MID 2304u
#define IN 5120u
#define NIDX (IN / 8u) /* 一行 640 个索引 = 20 轮 × 32 */
#define ROWB (NIDX * 12u / 8u)   /* 一行位流 960 B */
#define CBB (4096u * 8u)         /* 12 位层码本: 4096 词 × 8 个 E4M3 */

/* ---- 引擎同款(逐字, 见 src/cuda/cuda_vq_row.inc.cu / cuda_vq_persist.inc.cu) ---- */
__device__ __forceinline__ static float bf16r(float x) {   /* = v41_bf16r: 就近偶舍到 bf16 */
    uint32_t u = __float_as_uint(x); u += 0x7fffu + ((u >> 16) & 1u); return __uint_as_float(u & 0xffff0000u);
}
__device__ __forceinline__ static __half2 e4m3x2_to_half2(uint32_t two) {
    uint32_t h; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    __half2 out; memcpy(&out, &h, 4); return out;
}
__device__ __forceinline__ static float dot8(uint32_t v, uint4 xw, const uint8_t *cbs) {
    const uint2 w = *(const uint2 *)(cbs + (size_t)v * 8u);
    const __half2 p0 = e4m3x2_to_half2(w.x), p1 = e4m3x2_to_half2(w.x >> 16), p2 = e4m3x2_to_half2(w.y), p3 = e4m3x2_to_half2(w.y >> 16);
    const float2 e0 = __half22float2(p0), e1 = __half22float2(p1), e2 = __half22float2(p2), e3 = __half22float2(p3);
    return e0.x * __uint_as_float(xw.x << 16) + e0.y * __uint_as_float(xw.x & 0xffff0000u)
         + e1.x * __uint_as_float(xw.y << 16) + e1.y * __uint_as_float(xw.y & 0xffff0000u)
         + e2.x * __uint_as_float(xw.z << 16) + e2.y * __uint_as_float(xw.z & 0xffff0000u)
         + e3.x * __uint_as_float(xw.w << 16) + e3.y * __uint_as_float(xw.w & 0xffff0000u);
}
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
typedef struct { const uint8_t *gr, *ix; } mat_t;   /* 一个矩阵: 行增益(f16) + 位流(行首尾相接) */
__device__ __forceinline__ static const uint32_t *row_ptr(const mat_t &m, uint32_t r) { return (const uint32_t *)(m.ix + (size_t)r * ROWB); }
__device__ __forceinline__ static blk_t row_first_blk(const mat_t &m, uint32_t r) { return blk_load(row_ptr(m, r), 8u * 12u); }

/* = v41_vq_stream<12, 0, 1> */
__device__ __forceinline__ static void stream(const mat_t &m, uint32_t r0, uint32_t n, const uint32_t *x, const uint8_t *cbs,
                                              blk_t *carry, const uint32_t *next, float *res) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = 12u;
    const uint32_t R = NIDX >> 5, G = n * R;
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *base = row_ptr(m, r0);
    float g1 = 0.f;
    if (lane < n) { __half gh; memcpy(&gh, m.gr + (size_t)(r0 + lane) * 2u, 2); g1 = __half2float(gh); }
    blk_t cur = *carry;
    float acc = 0.f; *res = 0.f;
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
            const uint4 xa0 = *(const uint4 *)(x + (size_t)(kin * 32u + lane) * 4u);
            const uint32_t f = k * MB;
            const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
            const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
            const uint32_t lo = __shfl_sync(0xffffffffu, sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
            const uint32_t hi = __shfl_sync(0xffffffffu, sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
            const uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
            acc += dot8(v, xa0, cbs);
            if (++kin == R) {
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                const float a1 = __shfl_sync(0xffffffffu, g1, (int)row);
                const float val = acc * a1;
                if (lane == row) *res = val;
                acc = 0.f; kin = 0u; row++;
            }
        }
        cur = nxt;
    }
    *carry = cur;
}
#define SEG(U, UEND, ROWS, P, R0, N) \
    const uint32_t P = (U) / (ROWS), R0 = (U) % (ROWS); \
    uint32_t N = (UEND) - (U); if (N > (ROWS) - R0) N = (ROWS) - R0; if (N > 32u) N = 32u
/* = v41_vq_gu_persist_kernel<12, 0>(去掉 PDL 与格式校验; gate/up 的专家 p 的矩阵直接按层内偏移给) */
/* 这一层 6 个专家的 gate/up 矩阵起点表(专家 k 的 gate = mt[2k], up = mt[2k+1]), 放全局内存 —— 引擎也是读 blob 槽表找载荷。
 * ★别按值传一个指针数组再用变量下标取★(2026-09-24 实撞): 编译器把参数表搬进本地内存, 整核慢 25%(124 → 155 µs/层), 量的就不是真核了。 */
typedef struct { const uint8_t *p[2u * NP]; } mats_t;
__global__ static void __launch_bounds__(1024, 1) gu_persist(uint16_t *h, const mats_t *__restrict__ mt, const uint8_t *cb, const uint32_t *xg) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t gw = blockIdx.x * WARPS + (threadIdx.x >> 5), nw = gridDim.x * WARPS;
    const uint32_t total = NP * MID, per = (total + nw - 1u) / nw, lane = threadIdx.x & 31u;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    for (uint32_t i = threadIdx.x; i < CBB / 16u; i += blockDim.x) ((uint4 *)vqsh)[i] = ((const uint4 *)cb)[i];
    /* 每个矩阵 = [行增益 MID×2 B][位流 MID×960 B]; 起点由主机按摆法给(紧凑 = 一层 12 个首尾相接; 散布 = 大块里随机挑专家) */
    #define MAT(P, W) ({ mat_t m_; m_.gr = mt->p[2u * (P) + (W)]; m_.ix = m_.gr + MID * 2u; m_; })
    blk_t carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u;
    if (u < uend) { const mat_t mg = MAT(u / MID, 0u); carry = row_first_blk(mg, u % MID); }
    for (uint32_t i = threadIdx.x; i < IN * 2u / 16u; i += blockDim.x) ((uint4 *)(vqsh + CBB))[i] = ((const uint4 *)xg)[i];
    __syncthreads();
    const uint32_t *x = (const uint32_t *)(vqsh + CBB);
    bool first = true;
    while (u < uend) {
        SEG(u, uend, MID, p, r0, n);
        u += n;
        const mat_t mg = MAT(p, 0u), mu = MAT(p, 1u);
        if (!first) carry = row_first_blk(mg, r0);
        first = false;
        float gs, us;
        stream(mg, r0, n, x, vqsh, &carry, row_ptr(mu, r0), &gs);
        stream(mu, r0, n, x, vqsh, &carry, NULL, &us);
        if (lane < n) { const float gv = bf16r(gs), uv = bf16r(us); h[(uint64_t)p * MID + r0 + lane] = (uint16_t)(__float_as_uint(bf16r(gv / (1.f + __expf(-gv)) * uv)) >> 16); }
    }
    #undef MAT
}

/* 大块随机填充(散布档用): 每个字一个整数散列, 只要"像随机位流"不要统计质量 */
__global__ static void fill_rand(uint32_t *p, uint64_t n, uint32_t seed) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t v = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32) ^ seed;
        v ^= v >> 15; v *= 0x2c1b3c6du; v ^= v >> 12; v *= 0x297a2d39u; v ^= v >> 15;
        p[i] = v;
    }
}

static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(int argc, char **argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 120;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    /* 矩阵跨度: 行增益 + 位流 + 8 B 垫, 向上取 128 B 整 ⇒ 每个矩阵的位流起点相对 128 B 线的错开量 = (起点 + MID×2) mod 128 = skew 决定 */
    const uint64_t mat_bytes = MID * 2u + (uint64_t)MID * ROWB + 8u;
    const uint64_t mat_stride = (mat_bytes + 127u) & ~(uint64_t)127u;
    const uint64_t layer_bytes = mat_stride * 2u * NP;
    const uint64_t total = layer_bytes * (uint64_t)layers + 4096u;
    uint8_t *buf = NULL, *cb = NULL; uint32_t *x = NULL; uint16_t *h = NULL;
    CK(cudaMalloc(&buf, total)); CK(cudaMalloc(&cb, CBB)); CK(cudaMalloc(&x, IN * 2u)); CK(cudaMalloc(&h, NP * MID * 2u));
    {   /* 随机位流/增益(f16 ~1)/码本(E4M3 去掉 NaN 码 0x7f/0xff)/激活(bf16 ~±1) */
        const size_t chunk = 64u << 20; uint32_t *hb = (uint32_t *)malloc(chunk);
        for (uint64_t off = 0; off < total; off += chunk) {
            const size_t n = total - off < chunk ? (size_t)(total - off) : chunk;
            for (size_t i = 0; i < n / 4; i++) hb[i] = rnd();
            CK(cudaMemcpy(buf + off, hb, n & ~(size_t)3, cudaMemcpyHostToDevice));
        }
        for (int L = 0; L < layers; L++) for (uint32_t m = 0; m < 2u * NP; m++) {   /* 行增益 f16 = 1.0 */
            uint16_t g[MID]; for (uint32_t i = 0; i < MID; i++) g[i] = 0x3c00u;
            CK(cudaMemcpy(buf + (uint64_t)L * layer_bytes + m * mat_stride, g, sizeof g, cudaMemcpyHostToDevice));
        }
        uint8_t hc[CBB]; for (uint32_t i = 0; i < CBB; i++) { uint8_t b = (uint8_t)rnd(); if ((b & 0x7f) == 0x7f) b &= 0xfe; hc[i] = b; }
        CK(cudaMemcpy(cb, hc, CBB, cudaMemcpyHostToDevice));
        uint16_t hx[IN]; for (uint32_t i = 0; i < IN; i++) hx[i] = (uint16_t)(0x3f00u | (rnd() & 0x80ffu));
        CK(cudaMemcpy(x, hx, sizeof hx, cudaMemcpyHostToDevice));
        free(hb);
    }
    const uint32_t shm = CBB + IN * 2u;
    CK(cudaFuncSetAttribute(gu_persist, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    printf("v3 gateup 常驻核抄本: %d SM × 1024 线程, 一层 %u 专家 × 2 × %u 行 × %u B 位流 = %.2f MB, %d 层\n",
           nsm, NP, MID, ROWB, 2.0 * NP * MID * ROWB / 1e6, layers);
    printf("  (引擎直发实测: gateup 7.10 ms / 40 层 = 177 µs/层)\n");
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    /* 按一张指针表发 n 层(热身 4 层不计), 返回 µs/层 */
    mats_t *dtab = NULL; CK(cudaMalloc(&dtab, sizeof(mats_t) * (size_t)layers));
    #define TIMED(TAB, N) ({ CK(cudaMemcpy(dtab, (TAB), sizeof(mats_t) * (size_t)(N), cudaMemcpyHostToDevice)); \
        for (int L_ = 0; L_ < 4; L_++) gu_persist<<<nsm, 1024, shm>>>(h, dtab + L_, cb, x); CK(cudaDeviceSynchronize()); \
        CK(cudaEventRecord(e0)); for (int L_ = 0; L_ < (N); L_++) gu_persist<<<nsm, 1024, shm>>>(h, dtab + L_, cb, x); \
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms_ = 0.f; CK(cudaEventElapsedTime(&ms_, e0, e1)); (double)ms_ * 1e3 / (N); })
    const double lay_mb = 2.0 * NP * MID * ROWB / 1e6;
    mats_t *tab = (mats_t *)malloc(sizeof(mats_t) * (size_t)layers);
    const uint32_t skews[] = { 0u, 8u, 16u, 24u, 64u, 72u };
    for (int rep = 0; rep < reps; rep++)
        for (size_t s = 0; s < sizeof skews / sizeof skews[0]; s++) {
            /* 位流起点 = 层基 + skew + m·stride + MID×2; MID×2 = 4608 ≡ 0 mod 128 ⇒ 错开量就是 skew */
            for (int L = 0; L < layers; L++) for (uint32_t m = 0; m < 2u * NP; m++)
                tab[L].p[m] = buf + skews[s] + (uint64_t)L * layer_bytes + m * mat_stride;
            const double us = TIMED(tab, layers);
            printf("  第 %d 遍 位流错开 %2u B: %6.1f µs/层  %6.1f GB/s  (40 层 %.2f ms)\n", rep + 1, skews[s], us, lay_mb / us * 1e3, us * 40 / 1e3);   /* MB/µs × 1000 = GB/s */
        }
    /* ★散布档(2026-09-24)★: 引擎里一层的专家 blob 是 384 个专家 × 3 矩阵 ≈ 2.5 GB, 40 层铺满 ~98 GB; 每层按路由随机挑 6 个 ——
     * 而上面的紧凑档把 6 个专家首尾相接摆在 26.5 MB 里, 40 层才 1 GB。引擎 12 位层实测 170 µs/层, 比紧凑档同错位慢 ~15%。
     * 这一档问: 同一个核、同样对齐(错开 0), 只把"去哪读"换成"大块里随机 6 个专家", 占地从 1 层 blob 涨到 SPREAD_GB,
     * 如果 µs/层随占地变大而涨 ⇒ 差在地址翻译(TLB)/DRAM 页, 该从"专家在显存里怎么摆"下手; 不涨 ⇒ 这个假设作废。
     * 大块用 GPU 核填随机数(40 GB 主机生成太慢); 行增益也是随机 f16, 可能是 NaN —— 只量时间, 不看值。 */
    const double sp_gb = argc > 3 ? atof(argv[3]) : 40.0;
    const uint64_t blob = mat_stride * 2u * 384u;   /* 一层: 384 个专家的 gate/up(down 不在这个核里读, 省掉) */
    const uint32_t nblob_max = (uint32_t)(sp_gb * 1e9 / (double)blob);
    if (nblob_max >= 1u) {
        uint8_t *big = NULL;
        CK(cudaMalloc(&big, blob * nblob_max));
        fill_rand<<<nsm * 8, 256>>>((uint32_t *)big, blob * nblob_max / 4u, 0x9e3779b9u); CK(cudaDeviceSynchronize());
        printf("  -- 散布档: 一层 blob %.2f GB(384 专家 gate/up), 每层随机挑 %u 个专家, 位流错开 0\n", blob / 1e9, NP);
        uint32_t fp[] = { 1u, 4u, nblob_max };
        for (int rep = 0; rep < reps; rep++)
            for (size_t f = 0; f < sizeof fp / sizeof fp[0]; f++) {
                const uint32_t nb = fp[f] < nblob_max ? fp[f] : nblob_max;
                for (int L = 0; L < layers; L++) {
                    uint32_t e[NP];
                    for (uint32_t k = 0; k < NP; k++) { uint32_t v, dup; do { v = rnd() % 384u; dup = 0; for (uint32_t j = 0; j < k; j++) dup |= e[j] == v; } while (dup); e[k] = v; }
                    const uint8_t *lb = big + (uint64_t)(L % nb) * blob;
                    for (uint32_t k = 0; k < NP; k++) { tab[L].p[2u * k] = lb + (uint64_t)(2u * e[k]) * mat_stride; tab[L].p[2u * k + 1u] = lb + (uint64_t)(2u * e[k] + 1u) * mat_stride; }
                }
                const double us = TIMED(tab, layers);
                printf("  第 %d 遍 占地 %2u 层 blob(%5.1f GB): %6.1f µs/层  %6.1f GB/s\n", rep + 1, nb, nb * blob / 1e9, us, lay_mb / us * 1e3);
            }
        CK(cudaFree(big));
    }
    #undef TIMED
    free(tab); CK(cudaFree(dtab));
    return 0;
}
