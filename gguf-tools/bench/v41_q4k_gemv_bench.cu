/* v41_q4k_gemv_bench.cu — V4.1 骨架 q4_K 解码 GEMV 的形态微基准(spark 本机跑, 2026-09-23)。
 *
 * 【为什么要它】09-23 逐函数表: 骨架 q4_K GEMV 329 发/步 = 20.6 ms, 占一步 38 ms 的一半多, 各形状只到
 * 190~214 GB/s(墙 242)。sp.md 的规矩: 换结构先在这里过 200+ GB/s 才进引擎(引擎一趟要停服 + 编译 + 4 分钟)。
 * 老的 q4k_gemv_bench.cu 是 V4 的(q8 整数激活、V4 形状), 对不上这支。
 *
 * 【比什么】A = 引擎现核(src/cuda/cuda_v41_q4k.inc.cu 的 v41_q4k_gemv_kernel<1>, 逐字抄, 同发射配置);
 *          B = 同一个 lane 分工与累加序, 只是整个 CTA 先把自己那几行(内存里连续一段)按 16 B 整线读进 shared。
 * 判据: B 与 A 输出逐位同(不同就是字节取错了, 不许进引擎), 且 GB/s 更高。引擎的逐字节对拍是终验。
 * 形状 = v3 GGUF 的真实张量(09-23 从 GGUF 头读出): 见 main() 的 shapes 表。
 * 用法: nvcc -O3 -arch=native -o v41_q4k_gemv_bench v41_q4k_gemv_bench.cu && ./v41_q4k_gemv_bench [iters=50] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define BLK 256u
#define BYTES 144u
#define WARPS 8u

/* ---- 引擎同款(逐字) ---- */
__device__ __forceinline__ static uint32_t scb(const uint4 &h, uint32_t k) {
    const uint32_t w = k < 4u ? h.y : (k < 8u ? h.z : h.w);
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}
__device__ __forceinline__ static void sm_reg(const uint4 &h, uint32_t j, float *s, float *m) {
    uint32_t a, b;
    if (j < 4u) { a = scb(h, j) & 63u; b = scb(h, j + 4u) & 63u; }
    else { a = (scb(h, j + 4u) & 0xFu) | ((scb(h, j - 4u) >> 6) << 4);
           b = (scb(h, j + 4u) >> 4)   | ((scb(h, j) >> 6) << 4); }
    *s = (float)a; *m = (float)b;
}
__device__ __forceinline__ static void blk_acc(const uint4 &h, uint32_t qw, const float *x, uint32_t base, float &acc) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    const float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFFu)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    float s_lo, m_lo, s_hi, m_hi;
    sm_reg(h, gidx * 2u, &s_lo, &m_lo);
    sm_reg(h, gidx * 2u + 1u, &s_hi, &m_hi);
    const float dl = d * s_lo, ml = dmin * m_lo, dh = d * s_hi, mh = dmin * m_hi;
    float wl[4], wh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        wl[i] = dl * (float)(byte & 0xFu) - ml;
        wh[i] = dh * (float)(byte >> 4) - mh;
    }
    const uint32_t e_lo = base + gidx * 64u + q0, e_hi = e_lo + 32u;
    const float4 xl = *(const float4 *)(x + e_lo), xh = *(const float4 *)(x + e_hi);
    acc += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
         + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
}
/* 段间规约(与引擎同序) */
__device__ __forceinline__ static void finish(float acc, float *out, uint32_t r, uint32_t out_dim, uint32_t ksplit,
                                              uint32_t rloc, uint32_t kpart, float (*red)[1]) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) red[warp][0] = acc;
    __syncthreads();
    if (r < out_dim && kpart == 0 && lane == 0) {
        float sum = 0.f;
        for (uint32_t k = 0; k < ksplit; k++) sum += red[rloc * ksplit + k][0];
        out[r] = sum;
    }
}

/* A: 引擎现核(预取 4 块) */
__global__ static void __launch_bounds__(256, 4) k_cur(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
                                                       uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride,
                                                       uint32_t x_gstride, uint32_t out_gstride) {
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rpb = WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit, r = blockIdx.x * rpb + rloc;
    __shared__ float red[16][1];
    float acc = 0.f;
    if (r < out_dim) {
        const uint32_t nblk = in_dim / BLK;
        const uint8_t *wr = w + (uint64_t)r * nblk * BYTES;
        for (uint32_t b0 = kpart; b0 < nblk; b0 += 4u * ksplit) {
            uint4 hv[4]; uint32_t qv[4];
            #pragma unroll
            for (uint32_t u = 0; u < 4u; u++) {
                const uint32_t b = b0 + u * ksplit;
                hv[u] = make_uint4(0u, 0u, 0u, 0u); qv[u] = 0u;
                if (b < nblk) { const uint8_t *blk = wr + (uint64_t)b * BYTES; hv[u] = *(const uint4 *)blk; qv[u] = *(const uint32_t *)(blk + 16u + 4u * lane); }
            }
            #pragma unroll
            for (uint32_t u = 0; u < 4u; u++) { const uint32_t b = b0 + u * ksplit; if (b < nblk) blk_acc(hv[u], qv[u], x, b * BLK, acc); }
        }
    }
    finish(acc, out, r, out_dim, ksplit, rloc, kpart, red);
}

/* B: CTA 先把 [r0, r0+rpb) 这几行(连续一段)按 16 B 整线读进 shared, 再按 A 的分工与次序算。
 * __ldcs = 流式读(权重每步只读一遍, 别占 L2 挤掉激活)。 */
template <int MINB>
__global__ static void __launch_bounds__(256, MINB) k_stage(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
                                                         uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride,
                                                         uint32_t x_gstride, uint32_t out_gstride) {
    extern __shared__ uint4 st[];
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rpb = WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r0 = blockIdx.x * rpb, r = r0 + rloc, nblk = in_dim / BLK;
    const uint32_t nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb;
    const uint32_t n16 = nrows * nblk * (BYTES / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BYTES);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) st[i] = __ldcs(src + i);
    __syncthreads();
    __shared__ float red[16][1];
    float acc = 0.f;
    if (r < out_dim) {
        const uint8_t *wr = (const uint8_t *)st + (uint64_t)rloc * nblk * BYTES;
        for (uint32_t b = kpart; b < nblk; b += ksplit) {
            const uint8_t *blk = wr + b * BYTES;
            blk_acc(*(const uint4 *)blk, *(const uint32_t *)(blk + 16u + 4u * lane), x, b * BLK, acc);
        }
    }
    finish(acc, out, r, out_dim, ksplit, rloc, kpart, red);
}

/* C: 常驻 + 双缓冲。CTA 数 = 占满 SM 的数量, 每个 CTA 按 rg += gridDim.x 循环处理"行组"(一组 = rpb 行, 与 A/B 的
 * CTA 同一批行、同一分工), 算第 k 组时 cp.async 已在把第 k+1 组搬进另一半 shared ⇒ 读与算重叠, 且省掉每组一次 CTA 启停。
 * 行组内的累加序与 A 相同 ⇒ 逐位同。 */
__global__ static void __launch_bounds__(256, 2) k_pipe(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
                                                        uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride,
                                                        uint32_t x_gstride, uint32_t out_gstride) {
    extern __shared__ uint4 st[];
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rpb = WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit, nblk = in_dim / BLK;
    const uint32_t ngrp = (out_dim + rpb - 1u) / rpb, grp16 = rpb * nblk * (BYTES / 16u);
    __shared__ float red[16][1];
    auto issue = [&](uint32_t rg, uint32_t buf) {
        const uint32_t r0 = rg * rpb, nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb, n16 = nrows * nblk * (BYTES / 16u);
        const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BYTES);
        uint4 *dst = st + (uint64_t)buf * grp16;
        for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) __pipeline_memcpy_async(dst + i, src + i, 16);
    };
    uint32_t rg = blockIdx.x, buf = 0;
    if (rg < ngrp) issue(rg, 0);
    __pipeline_commit();
    for (; rg < ngrp; rg += gridDim.x, buf ^= 1u) {
        if (rg + gridDim.x < ngrp) issue(rg + gridDim.x, buf ^ 1u);
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        const uint32_t r = rg * rpb + rloc;
        float acc = 0.f;
        if (r < out_dim) {
            const uint8_t *wr = (const uint8_t *)(st + (uint64_t)buf * grp16) + (uint64_t)rloc * nblk * BYTES;
            for (uint32_t b = kpart; b < nblk; b += ksplit) {
                const uint8_t *blk = wr + b * BYTES;
                blk_acc(*(const uint4 *)blk, *(const uint32_t *)(blk + 16u + 4u * lane), x, b * BLK, acc);
            }
        }
        finish(acc, out, r, out_dim, ksplit, rloc, kpart, red);
        __syncthreads();   /* 这半 shared 下一轮要被覆盖: 全 CTA 算完再发下一次 issue */
    }
}

typedef struct { const char *name; uint32_t out_dim, in_dim, groups, per_step; } shape_t;

static uint32_t pick_ksplit(uint32_t out_dim, uint32_t groups, uint32_t nb) {   /* 引擎 v41_q4k_gemv 同式 */
    uint32_t k = 1;
    while (k < WARPS && out_dim * k * groups < 8192u) k <<= 1;
    while (k > 1u && k > nb) k >>= 1;
    return k;
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 50;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("%s SMs=%d L2=%d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    /* per_step = 每个解码步发几次(40 层; indexer q_b 只在 8 个源层; 出口头 1 次) */
    const shape_t shapes[] = {
        {"q_b 32768x1280", 32768, 1280, 1, 40}, {"wo_a 8x1024x4096", 1024, 4096, 8, 40},
        {"wo_b 5120x8192", 5120, 8192, 1, 40}, {"q_a 1280x5120", 1280, 5120, 1, 40},
        {"shexp_gate/up 2304x5120", 2304, 5120, 1, 80}, {"shexp_down 5120x2304", 5120, 2304, 1, 40},
        {"kv 512x5120", 512, 5120, 1, 40}, {"idx_q_b 4096x1280", 4096, 1280, 1, 8},
        {"output 129280x5120", 129280, 5120, 1, 1} };
    double tot_a = 0, tot_b = 0, tot_c = 0, tot_bytes = 0;
    for (const shape_t &sh : shapes) {
        const uint32_t nb = sh.in_dim / BLK, ks = pick_ksplit(sh.out_dim, sh.groups, nb), rpb = WARPS / ks;
        const uint64_t gbytes = (uint64_t)sh.out_dim * nb * BYTES, mbytes = gbytes * sh.groups;
        /* 轮换多份拷贝压过 L2: 总量 ≥ 256 MB */
        int ncopy = (int)((256ull << 20) / mbytes) + 1; if (ncopy > 64) ncopy = 64;
        uint8_t *h = (uint8_t *)malloc(mbytes);
        uint32_t s = 12345u + sh.out_dim;
        for (uint64_t i = 0; i < mbytes; i++) { s = s * 1664525u + 1013904223u; h[i] = (uint8_t)(s >> 24); }
        for (uint64_t b = 0; b < mbytes / BYTES; b++) {   /* d/dmin 写成正常 f16(0.01 量级), 别让随机字节出 NaN/Inf */
            const __half d = __float2half(0.001f + (float)(b % 13) * 1e-4f), m = __float2half(0.002f + (float)(b % 7) * 1e-4f);
            memcpy(h + b * BYTES, &d, 2); memcpy(h + b * BYTES + 2, &m, 2);
        }
        const uint64_t xn = (uint64_t)sh.in_dim * sh.groups;
        float *hx = (float *)malloc(xn * 4); for (uint64_t i = 0; i < xn; i++) hx[i] = (float)((int)(i % 17) - 8) * 0.0625f;
        uint8_t **dw = (uint8_t **)malloc(sizeof(uint8_t *) * ncopy);
        for (int c = 0; c < ncopy; c++) { CK(cudaMalloc(&dw[c], mbytes)); CK(cudaMemcpy(dw[c], h, mbytes, cudaMemcpyHostToDevice)); }
        float *dx, *oa, *ob; const uint64_t on = (uint64_t)sh.out_dim * sh.groups;
        CK(cudaMalloc(&dx, xn * 4)); CK(cudaMemcpy(dx, hx, xn * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&oa, on * 4)); CK(cudaMalloc(&ob, on * 4));
        const dim3 grid((sh.out_dim + rpb - 1u) / rpb, sh.groups);
        const size_t shm = (size_t)rpb * nb * BYTES;
        if (shm > 48u * 1024u) { CK(cudaFuncSetAttribute(k_stage<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
                                 CK(cudaFuncSetAttribute(k_stage<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); }
        CK(cudaFuncSetAttribute(k_pipe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2 * shm)));
        int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_pipe, 256, 2 * shm));
        const uint32_t ngrp_rows = (sh.out_dim + rpb - 1u) / rpb, want = (uint32_t)(occ * pr.multiProcessorCount) / sh.groups;
        const dim3 gpipe(want < ngrp_rows ? (want ? want : 1u) : ngrp_rows, sh.groups);
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        auto run = [&](int which, int i) {
            const uint8_t *wp = dw[i % ncopy];
            if (which == 0) k_cur<<<grid, 256>>>(oa, wp, dx, sh.in_dim, sh.out_dim, ks, gbytes, sh.in_dim, sh.out_dim);
            else if (which == 1) k_stage<4><<<grid, 256, shm>>>(ob, wp, dx, sh.in_dim, sh.out_dim, ks, gbytes, sh.in_dim, sh.out_dim);
            else k_pipe<<<gpipe, 256, 2 * shm>>>(ob, wp, dx, sh.in_dim, sh.out_dim, ks, gbytes, sh.in_dim, sh.out_dim);
        };
        float us[3];
        for (int which = 0; which < 3; which++) {
            for (int i = 0; i < 5; i++) run(which, i);
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) run(which, i);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us[which] = ms * 1000.f / iters;
        }
        run(0, 0); run(1, 0); CK(cudaDeviceSynchronize());
        float *rc = (float *)malloc(on * 4); run(2, 0); CK(cudaDeviceSynchronize());
        float *ra = (float *)malloc(on * 4), *rb = (float *)malloc(on * 4);
        CK(cudaMemcpy(ra, oa, on * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rb, ob, on * 4, cudaMemcpyDeviceToHost));
        uint64_t bad = 0; for (uint64_t i = 0; i < on; i++) if (memcmp(&ra[i], &rb[i], 4)) bad++;
        CK(cudaMemcpy(rc, ob, on * 4, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < on; i++) if (memcmp(&ra[i], &rc[i], 4)) bad++;
        free(rc);
        printf("%-26s %6.1f MB ks=%u rpb=%u | 现核 %7.1f us %4.0f | stage4 %7.1f us %4.0f | pipe(occ%d) %7.1f us %4.0f GB/s | %s\n",
               sh.name, mbytes / 1e6, ks, rpb, us[0], mbytes / (us[0] * 1e3), us[1], mbytes / (us[1] * 1e3),
               occ, us[2], mbytes / (us[2] * 1e3),
               bad ? "★不同★" : "逐位同");
        if (bad) printf("  ★%llu/%llu 个输出不同★\n", (unsigned long long)bad, (unsigned long long)on);
        tot_a += us[0] * sh.per_step; tot_b += us[1] * sh.per_step; tot_c += us[2] * sh.per_step; tot_bytes += (double)mbytes * sh.per_step;
        for (int c = 0; c < ncopy; c++) cudaFree(dw[c]);
        free(dw); cudaFree(dx); cudaFree(oa); cudaFree(ob); free(h); free(hx); free(ra); free(rb);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf("== 每步合计(按发射次数加权): 现核 %.2f / stage4 %.2f / pipe %.2f ms, 字节 %.2f GB ⇒ 墙(242 GB/s) %.2f ms\n",
           tot_a / 1e3, tot_b / 1e3, tot_c / 1e3, tot_bytes / 1e9, tot_bytes / 242e9 * 1e3);
    return 0;
}
