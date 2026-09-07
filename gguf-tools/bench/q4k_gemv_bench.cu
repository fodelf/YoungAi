/* q4k_gemv_bench.cu — 解码 Q4_K 稠密 gemv 的形态微基准(spark 本机跑, 2026-09-07)。
 *
 * 【为什么要它】骨架 Q4_K 六个矩阵单独跑只到 173~221 GB/s(墙 233~238), 每 token 差 2.4~3.2 ms。
 * 在引擎里试一版要 10 分钟(编译+parity+曲线), 09-06 四刀全盲试全空。这里按真实形状
 * (行数×256 值块数)造随机 Q4_K 矩阵 + q8_K 激活, 现核 vs 候选核逐个量 µs/GB/s, 并逐位比输出 ——
 * 候选核的数学必须与现核逐位同(同 dot 同归约序), 这里先验, 引擎 parity 终验。
 * 现核 = 引擎 09-07 前的 matmul_q4_K_warp_kernel(整行 staged, blocks≤16 半块拆分) 与 out_a/out_b 的 q8_0 dp4a 核;
 * 候选 = tile(整块/lane + 一 warp 32/BLOCKS 行, 已落地 cuda_q4k_tile.inc.cu) 与 out_b 的整块/整行变体。
 * 判死的变体(平面重排 q4r、协作 staging、预算 Σq8)结论记在 fable5 09-07, 代码已删。
 * 用法: q4k_gemv_bench [iters=20]   (nvcc -O3 -arch=native -o q4k_gemv_bench q4k_gemv_bench.cu)
 * 设备侧 dot/scale/f16 辅助函数逐字抄自 src/cuda/cuda_api_moe_corr_1/2.inc.cu —— 只为测速, 引擎才是真值。 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <time.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#define QK_K 256
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } blk_q4_K;   /* 144 B */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } blk_q8_K;             /* 292 B */

/* ---- 引擎同款 device 辅助(逐字) ---- */
__device__ static float dev_f16_to_f32(uint16_t v) { return __half2float(*reinterpret_cast<const __half *>(&v)); }
__device__ static void dev_q4_K_get_scale_min(uint32_t j, const uint8_t *scales, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) { *d_out = scales[j] & 63u; *m_out = scales[j + 4u] & 63u; }
    else { *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u); *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
}
__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}
__device__ static float dev_dot_q4_K_q8_K_block(const blk_q4_K *x, const blk_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d), xmin = dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0, summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0); dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        summs0 += (int)m0 * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)m1 * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        isum0 += (int)sc0 * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, 0);
        isum1 += (int)sc1 * dev_dot_q4_32(x->qs + byte_off, y->qs + (j + 1u) * 32u, 4);
    }
    return y->d * xd * (float)(isum0 + isum1) - y->d * xmin * (float)(summs0 + summs1);
}
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_block_half(const blk_q4_K *x, const blk_q8_K *y, uint32_t j0) {
    const float xd = dev_f16_to_f32(x->d), xmin = dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0, summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = j0; j < j0 + 4u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0); dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        summs0 += (int)m0 * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)m1 * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        isum0 += (int)sc0 * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, 0);
        isum1 += (int)sc1 * dev_dot_q4_32(x->qs + byte_off, y->qs + (j + 1u) * 32u, 4);
    }
    return y->d * xd * (float)(isum0 + isum1) - y->d * xmin * (float)(summs0 + summs1);
}
__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

/* ---- A: 现核(引擎 matmul_q4_K_warp_kernel 逐字, 单 token) ---- */
__global__ static void k_cur(float *out, const char *w_base, const blk_q8_K *xt, uint64_t row_bytes, uint32_t blocks, uint32_t out_dim) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        float acc = 0.0f;
        if (blocks <= 32u) {
            extern __shared__ uint4 dstage[];
            const uint32_t n16 = blocks * 9u;
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = dstage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const blk_q4_K *wr = (const blk_q4_K *)my;
            if (blocks <= 16u) {
                const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
                if (bi < blocks) acc += dev_dot_q4_K_q8_K_block_half(wr + bi, xt + bi, hj);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u) acc += dev_dot_q4_K_q8_K_block(wr + b, xt + b);
            }
            __syncwarp();
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[row] = acc;
    }
}


/* ---- q8_0 激活口径(out_a/out_b 现役 dp4a 核用: 每 32 值一个 f32 scale) ---- */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem(const blk_q4_K *x, const int8_t *xq, const float *xs) {
    const float d = dev_f16_to_f32(x->d), dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0); dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u, *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0), dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) { s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80); s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81); }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem_half(const blk_q4_K *x, const int8_t *xq, const float *xs, uint32_t j0) {
    const float d = dev_f16_to_f32(x->d), dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = j0; j < j0 + 4u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0); dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u, *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0), dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) { s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80); s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81); }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}
/* out_a 现役: grouped_q4_K_a_preq_warp8_dp4a_kernel 逐字(n_tokens=1, n_groups 组各自的 x 段) */
__global__ static void k_outa_cur(float *low, const unsigned char *w, const int8_t *xq, const float *xscale, uint64_t rank, uint32_t n_groups, uint64_t kblocks) {
    const uint32_t lane = threadIdx.x & 15u, slot = threadIdx.x >> 4u;
    const uint64_t row = (uint64_t)blockIdx.x * 16u + slot, low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim) return;
    const uint64_t group = row / rank;
    const int8_t *xqr = xq + group * kblocks * 256u; const float *xsr = xscale + group * kblocks * 8u;
    extern __shared__ uint4 dstage_a[];
    const uint4 *src16 = (const uint4 *)(w + row * kblocks * 144u);
    const uint32_t n16 = (uint32_t)kblocks * 9u;
    uint4 *my = dstage_a + (uint64_t)slot * n16;
    for (uint32_t i = lane; i < n16; i += 16u) my[i] = __ldcs(src16 + i);
    __syncwarp();
    const blk_q4_K *wr = (const blk_q4_K *)my;
    float acc = 0.0f;
    for (uint64_t b = lane; b < kblocks; b += 16u) acc += dev_dot_q4_K_q8_0x8_smem(wr + b, xqr + b * 256u, xsr + b * 8u);
    acc += __shfl_down_sync(0xffffffffu, acc, 8); acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2); acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) low[row] = acc;
}
/* out_b 现役: matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel 的 gemv 段逐字(epilogue 略去, 只存 block_out) */
__global__ static void k_outb_cur(float *out, const unsigned char *w, const int8_t *xq, const float *xscale, uint64_t out_dim, uint64_t kblocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    if (row >= out_dim) return;
    float acc = 0.0f;
    extern __shared__ uint4 dstage_b[];
    const uint32_t seg = (kblocks > 16u) ? 16u : (uint32_t)kblocks, n16 = seg * 9u;
    uint4 *my = dstage_b + (uint64_t)warp * n16;
    for (uint32_t b0 = 0; b0 < (uint32_t)kblocks; b0 += seg) {
        const uint32_t nb = ((uint32_t)kblocks - b0 < seg) ? ((uint32_t)kblocks - b0) : seg;
        const uint4 *src16 = (const uint4 *)(w + row * kblocks * 144u + (uint64_t)b0 * 144u);
        for (uint32_t i = lane; i < nb * 9u; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my;
        const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
        if (bi < nb) acc += dev_dot_q4_K_q8_0x8_smem_half(wr + bi, xq + (uint64_t)(b0 + bi) * 256u, xscale + (uint64_t)(b0 + bi) * 8u, hj);
        __syncwarp();
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}
/* ---- 整块/lane 假设: 半块 dot 的 j0 是运行期值 ⇒ 尺度解码带分支、偏移运行期算, ALU 翻倍; 整块 dot 全编译期展开。
 * out_b: 整行 stage + 每 lane 一整块(32 块=32 lane) + 封顶跨步。归约结合律与现役半块版不同 ⇒ 容差级, 报最大绝对差。 */
__global__ static void k_outb_full(float *out, const unsigned char *w, const int8_t *xq, const float *xscale, uint64_t out_dim, uint64_t kblocks) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 dstage_b[];
    const uint32_t n16 = (uint32_t)kblocks * 9u;
    uint4 *my = dstage_b + (uint64_t)warp * n16;
    for (uint64_t row = (uint64_t)blockIdx.x * 8u + warp; row < out_dim; row += (uint64_t)gridDim.x * 8u) {
        const uint4 *src16 = (const uint4 *)(w + row * kblocks * 144u);
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my;
        float acc = 0.0f;
        for (uint32_t b = lane; b < (uint32_t)kblocks; b += 32u) acc += dev_dot_q4_K_q8_0x8_smem(wr + b, xq + (uint64_t)b * 256u, xscale + (uint64_t)b * 8u);
        acc = warp_sum_f32(acc);
        if (lane == 0) out[row] = acc;
        __syncwarp();
    }
}
/* q8_K 口径短行: 一 warp stage R=32/BLOCKS 行(连续内存一段), lane = (行 l/BLOCKS, 块 l%BLOCKS) 整块 dot, 段内树归约。 */
template <uint32_t BLOCKS>
__global__ static void k_tile(float *out, const char *w_base, const blk_q8_K *xt, uint32_t out_dim) {
    constexpr uint32_t R = 32u / BLOCKS;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS;
    extern __shared__ uint4 dstage[];
    constexpr uint32_t n16 = R * BLOCKS * 9u;                 /* = 288 片 = 4608 B / warp */
    uint4 *my = dstage + (uint64_t)warp * n16;
    for (uint32_t row0 = (blockIdx.x * 8u + warp) * R; row0 < out_dim; row0 += gridDim.x * 8u * R) {
        const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row0 * BLOCKS * 144u);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my + rin * BLOCKS + b;
        float acc = dev_dot_q4_K_q8_K_block(wr, xt + b);
        #pragma unroll
        for (uint32_t off = BLOCKS / 2u; off > 0u; off >>= 1u) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (b == 0u && row0 + rin < out_dim) out[row0 + rin] = acc;
        __syncwarp();
    }
}

static void fill_rand(uint8_t *p, size_t n, uint32_t seed) { uint32_t s = seed; for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; p[i] = (uint8_t)(s >> 24); } }
static void make_q8(blk_q8_K *b, uint32_t blocks, uint32_t seed) {
    uint32_t s = seed;
    for (uint32_t k = 0; k < blocks; k++) {
        b[k].d = 0.01f + (float)(k % 7) * 0.001f;
        for (int i = 0; i < QK_K; i++) { s = s * 1664525u + 1013904223u; b[k].qs[i] = (int8_t)((int)(s >> 24) - 128); }
        for (int j = 0; j < 16; j++) { int sum = 0; for (int i = 0; i < 16; i++) sum += b[k].qs[j * 16 + i]; b[k].bsums[j] = (int16_t)sum; }
    }
}

typedef struct { const char *name; uint32_t rows, blocks; } shape_t;

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20;
    const int NCOPY = 8;   /* 轮换 8 份矩阵, 压过 L2 */
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("%s SMs=%d L2=%d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    const shape_t shapes[] = { {"q_b 32768x1024(4 blk)", 32768, 4}, {"shexp_down 4096x2048(8)", 4096, 8},
                               {"q_a+kv 1536x4096(16)", 1536, 16}, {"shexp_gate+up 4096x4096(16)", 4096, 16},
                               {"out_a 8192x4096(16)", 8192, 16}, {"out_b 4096x8192(32)", 4096, 32} };
    for (const shape_t &sh : shapes) {
        const uint32_t rows = sh.rows, blocks = sh.blocks;
        const uint64_t mbytes = (uint64_t)rows * blocks * 144u;
        const uint32_t U = blocks <= 16u ? 2u * blocks : blocks, R = 32u / U;
        uint8_t *h = (uint8_t *)malloc(mbytes); fill_rand(h, mbytes, 7u + rows);
        blk_q8_K *hx = (blk_q8_K *)malloc(blocks * sizeof(blk_q8_K)); make_q8(hx, blocks, 3u);
        uint8_t *dw[8]; float *o_cur, *o_new; blk_q8_K *dx;
        for (int c = 0; c < NCOPY; c++) { CK(cudaMalloc(&dw[c], mbytes)); CK(cudaMemcpy(dw[c], h, mbytes, cudaMemcpyHostToDevice)); }
        CK(cudaMalloc(&dx, blocks * sizeof(blk_q8_K))); CK(cudaMemcpy(dx, hx, blocks * sizeof(blk_q8_K), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&o_cur, rows * 4)); CK(cudaMalloc(&o_new, rows * 4)); CK(cudaDeviceSynchronize());
        const size_t shm = (size_t)8u * blocks * 9u * 16u;
        unsigned gx = (rows + 7u) / 8u; if (gx > 384u) gx = 384u;
        printf("== %s  %.1f MB  U=%u R=%u\n", sh.name, mbytes / 1e6, U, R);
        /* 现核 */
        for (int i = 0; i < 3; i++) k_cur<<<gx, 256, shm>>>(o_cur, (const char *)dw[i % NCOPY], dx, (uint64_t)blocks * 144u, blocks, rows);
        CK(cudaDeviceSynchronize()); double t0 = now_s();
        for (int i = 0; i < iters; i++) k_cur<<<gx, 256, shm>>>(o_cur, (const char *)dw[i % NCOPY], dx, (uint64_t)blocks * 144u, blocks, rows);
        CK(cudaDeviceSynchronize()); double dt = (now_s() - t0) / iters;
        printf("  现核 staged   grid %4u: %7.1f us  %6.1f GB/s\n", gx, dt * 1e6, mbytes / dt / 1e9);
        if (blocks <= 16u) {
            const size_t shm_t = (size_t)8u * 32u * 9u * 16u;   /* 每 warp 4608 B */
            const unsigned gt[] = { 96u, 192u, 384u };
            for (unsigned g : gt) {
                auto launch = [&](int i) { const char *w = (const char *)dw[i % NCOPY];
                    switch (blocks) { case 4: k_tile<4><<<g, 256, shm_t>>>(o_new, w, dx, rows); break; case 8: k_tile<8><<<g, 256, shm_t>>>(o_new, w, dx, rows); break; default: k_tile<16><<<g, 256, shm_t>>>(o_new, w, dx, rows); } };
                for (int i = 0; i < 3; i++) launch(i);
                CK(cudaDeviceSynchronize()); t0 = now_s();
                for (int i = 0; i < iters; i++) launch(i);
                CK(cudaDeviceSynchronize()); dt = (now_s() - t0) / iters;
                k_cur<<<gx, 256, shm>>>(o_cur, (const char *)dw[0], dx, (uint64_t)blocks * 144u, blocks, rows); launch(0); CK(cudaDeviceSynchronize());
                float *a = (float *)malloc(rows * 4), *bb = (float *)malloc(rows * 4);
                CK(cudaMemcpy(a, o_cur, rows * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bb, o_new, rows * 4, cudaMemcpyDeviceToHost));
                uint32_t bad = 0; double md = 0; for (uint32_t r = 0; r < rows; r++) { if (memcmp(&a[r], &bb[r], 4)) bad++; double d = fabs((double)a[r] - bb[r]); if (d > md) md = d; }
                printf("  tile 整块/lane grid %4u: %7.1f us  %6.1f GB/s  %s(不同 %u, 最大差 %.3g)\n", g, dt * 1e6, mbytes / dt / 1e9, bad ? "容差级" : "逐位同", bad, md);
                free(a); free(bb);
            }
        }
        for (int c = 0; c < NCOPY; c++) cudaFree(dw[c]);
        cudaFree(dx); cudaFree(o_cur); cudaFree(o_new); free(h); free(hx);
    }

    /* ---- out_a / out_b: 现役 dp4a 核 vs 结构变体(q8_0 激活口径) ---- */
    {
        const uint32_t kb = 16u, rank = 1024u, ngroups = 8u, rows_a = rank * ngroups;   /* out_a 8192×4096 */
        const uint64_t mb_a = (uint64_t)rows_a * kb * 144u;
        uint8_t *h = (uint8_t *)malloc(mb_a); fill_rand(h, mb_a, 99u);
        int8_t *hxq = (int8_t *)malloc(ngroups * kb * 256u); float *hxs = (float *)malloc(ngroups * kb * 8u * 4u);
        fill_rand((uint8_t *)hxq, ngroups * kb * 256u, 5u); for (uint32_t i = 0; i < ngroups * kb * 8u; i++) hxs[i] = 0.01f + (float)(i % 5) * 0.002f;
        uint8_t *dw[8]; int8_t *dxq; float *dxs, *oa, *ob;
        for (int c = 0; c < NCOPY; c++) { CK(cudaMalloc(&dw[c], mb_a)); CK(cudaMemcpy(dw[c], h, mb_a, cudaMemcpyHostToDevice)); }
        CK(cudaMalloc(&dxq, ngroups * kb * 256u)); CK(cudaMemcpy(dxq, hxq, ngroups * kb * 256u, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxs, ngroups * kb * 8u * 4u)); CK(cudaMemcpy(dxs, hxs, ngroups * kb * 8u * 4u, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&oa, rows_a * 4)); CK(cudaMalloc(&ob, rows_a * 4));
        const size_t shm_a = (size_t)16u * kb * 9u * 16u;
        printf("== out_a 8192x4096(16) q8_0 口径 %.1f MB\n", mb_a / 1e6);
        auto timeit = [&](const char *nm, auto fn) {
            for (int i = 0; i < 3; i++) fn(i, oa); CK(cudaDeviceSynchronize()); double t0 = now_s();
            for (int i = 0; i < iters; i++) fn(i, oa); CK(cudaDeviceSynchronize()); double dt = (now_s() - t0) / iters;
            printf("  %-28s %7.1f us  %6.1f GB/s", nm, dt * 1e6, mb_a / dt / 1e9);
        };
        timeit("现役 dp4a 16lane grid512", [&](int i, float *o){ k_outa_cur<<<rows_a / 16u, 256, shm_a>>>(o, dw[i % NCOPY], dxq, dxs, rank, ngroups, kb); }); printf("\n");
        float *ref = (float *)malloc(rows_a * 4), *got = (float *)malloc(rows_a * 4);
        k_outa_cur<<<rows_a / 16u, 256, shm_a>>>(oa, dw[0], dxq, dxs, rank, ngroups, kb); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref, oa, rows_a * 4, cudaMemcpyDeviceToHost));
        for (int c = 0; c < NCOPY; c++) cudaFree(dw[c]); cudaFree(dxq); cudaFree(dxs); cudaFree(oa); cudaFree(ob); free(h); free(hxq); free(hxs); free(ref); free(got);
    }
    {
        const uint32_t kb = 32u, rows_b = 4096u;   /* out_b 4096×8192 */
        const uint64_t mb_b = (uint64_t)rows_b * kb * 144u;
        uint8_t *h = (uint8_t *)malloc(mb_b); fill_rand(h, mb_b, 77u);
        int8_t *hxq = (int8_t *)malloc(kb * 256u); float *hxs = (float *)malloc(kb * 8u * 4u);
        fill_rand((uint8_t *)hxq, kb * 256u, 6u); for (uint32_t i = 0; i < kb * 8u; i++) hxs[i] = 0.01f + (float)(i % 5) * 0.002f;
        uint8_t *dw[8]; int8_t *dxq; float *dxs, *oa, *ob;
        for (int c = 0; c < NCOPY; c++) { CK(cudaMalloc(&dw[c], mb_b)); CK(cudaMemcpy(dw[c], h, mb_b, cudaMemcpyHostToDevice)); }
        CK(cudaMalloc(&dxq, kb * 256u)); CK(cudaMemcpy(dxq, hxq, kb * 256u, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxs, kb * 8u * 4u)); CK(cudaMemcpy(dxs, hxs, kb * 8u * 4u, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&oa, rows_b * 4)); CK(cudaMalloc(&ob, rows_b * 4));
        const size_t shm_cur = (size_t)8u * 16u * 9u * 16u, shm_v = (size_t)8u * kb * 9u * 16u;
        printf("== out_b 4096x8192(32) q8_0 口径 %.1f MB\n", mb_b / 1e6);
        auto timeit = [&](const char *nm, auto fn) {
            for (int i = 0; i < 3; i++) fn(i, oa); CK(cudaDeviceSynchronize()); double t0 = now_s();
            for (int i = 0; i < iters; i++) fn(i, oa); CK(cudaDeviceSynchronize()); double dt = (now_s() - t0) / iters;
            printf("  %-28s %7.1f us  %6.1f GB/s", nm, dt * 1e6, mb_b / dt / 1e9);
        };
        timeit("现役 dp4a 两段 grid512", [&](int i, float *o){ k_outb_cur<<<rows_b / 8u, 256, shm_cur>>>(o, dw[i % NCOPY], dxq, dxs, rows_b, kb); }); printf("\n");
        float *ref = (float *)malloc(rows_b * 4), *got = (float *)malloc(rows_b * 4);
        k_outb_cur<<<rows_b / 8u, 256, shm_cur>>>(oa, dw[0], dxq, dxs, rows_b, kb); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref, oa, rows_b * 4, cudaMemcpyDeviceToHost));
        const unsigned caps[] = { 144u, 192u, 384u, 512u };
        for (unsigned g : caps) {
            char nm[64]; snprintf(nm, sizeof nm, "整块/lane 整行stage grid%u", g);
            timeit(nm, [&](int i, float *o){ k_outb_full<<<g, 256, shm_v>>>(o, dw[i % NCOPY], dxq, dxs, rows_b, kb); });
            k_outb_full<<<g, 256, shm_v>>>(ob, dw[0], dxq, dxs, rows_b, kb); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(got, ob, rows_b * 4, cudaMemcpyDeviceToHost));
            uint32_t bad = 0; double md = 0; for (uint32_t r = 0; r < rows_b; r++) { if (memcmp(&ref[r], &got[r], 4)) bad++; double d = fabs((double)ref[r] - got[r]); if (d > md) md = d; }
            printf("  %s(不同 %u, 最大差 %.3g)\n", bad ? "容差级" : "逐位同", bad, md);
        }
        for (int c = 0; c < NCOPY; c++) cudaFree(dw[c]); cudaFree(dxq); cudaFree(dxs); cudaFree(oa); cudaFree(ob); free(h); free(hxq); free(hxs); free(ref); free(got);
    }
    return 0;
}
