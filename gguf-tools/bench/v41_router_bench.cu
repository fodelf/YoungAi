/* v41_router_bench.cu — V4.1 路由核(v41_router_kernel)的孤立尺(2026-09-29 晚)。
 *
 * 【为什么要它】两条时间线(纯解码 / 4 行验证批)里 router 核一发都要 ~20 µs, 而它的活只是一个 warp 对 384 个专家做
 * softplus + 6 轮 warp argmax —— 按指令数该是 1~2 µs。一步 40 发 = 0.8~1.5 ms/步(2~2.5%)。引擎里的 ncu 今天被 engram 取行线程
 * 撞坏(application replay 下 io_uring 取行失败), 只能把核抄出来单独量: 它自己就要 20 µs(核体的事), 还是引擎环境的事(bias 在主机映射 / PDL 边)。
 * 【比什么】V0 = 引擎核逐字抄本(cuda_v41_3.inc.cu); 输入按引擎形状: logits [n][384] f32(设备), bias [384] f32(设备 / 主机映射两档)。
 * 判据: 各变体 sel/wts 与 V0 逐位同(路由结果一位不能差, 否则专家都不一样)。
 * 用法: nvcc -O3 -arch=native -o v41_router_bench v41_router_bench.cu && ./v41_router_bench [iters=200] [n_tok=1] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define V41_ROUTER_PER_LANE 12u

/* ---- V0: 引擎核逐字抄本(去掉 PDL wait) ---- */
__global__ static void k_router_v0(int32_t *sel, float *wts, const float *logits, const float *bias,
                                   uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    const uint32_t t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (t >= n_tok) return;
    const float *lg = logits + (uint64_t)t * n_expert;
    float pr[V41_ROUTER_PER_LANE], sc[V41_ROUTER_PER_LANE];
    #pragma unroll
    for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) {
        const uint32_t e = lane + 32u * k;
        if (e < n_expert) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : log1pf(expf(z));
            pr[k] = sqrtf(sp); sc[k] = pr[k] + bias[e];
        } else { pr[k] = 0.f; sc[k] = -INFINITY; }
    }
    uint32_t used = 0; float wsum = 0.f;
    for (uint32_t r = 0; r < topk; r++) {
        float bv = -INFINITY; int bk = -1;
        #pragma unroll
        for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) if (!((used >> k) & 1u) && sc[k] > bv) { bv = sc[k]; bk = (int)k; }
        int be = bk >= 0 ? (int)(lane + 32u * (uint32_t)bk) : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off); const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (oe >= 0 && (be < 0 || ov > bv || (ov == bv && oe < be))) { bv = ov; be = oe; }
        }
        float myp = 0.f;
        if (be >= 0 && (uint32_t)(be & 31) == lane) { const uint32_t k = (uint32_t)be >> 5; used |= 1u << k; myp = pr[k]; }
        for (int off = 16; off > 0; off >>= 1) myp += __shfl_xor_sync(0xffffffffu, myp, off);
        if (lane == 0) { sel[(uint64_t)t * topk + r] = be; wts[(uint64_t)t * topk + r] = myp; }
        wsum += myp;
    }
    if (lane == 0)
        for (uint32_t r = 0; r < topk; r++) wts[(uint64_t)t * topk + r] = wts[(uint64_t)t * topk + r] / (wsum + 1e-20f) * route_scale;
}
/* V1: 同一算式, 只把"myp = pr[k]"的运行期下标换成展开的选择(pr[] 不落 local memory); 输出逐位同(取的是同一个值) */
__global__ static void k_router_v1(int32_t *sel, float *wts, const float *logits, const float *bias,
                                   uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    const uint32_t t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (t >= n_tok) return;
    const float *lg = logits + (uint64_t)t * n_expert;
    float pr[V41_ROUTER_PER_LANE], sc[V41_ROUTER_PER_LANE];
    #pragma unroll
    for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) {
        const uint32_t e = lane + 32u * k;
        if (e < n_expert) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : log1pf(expf(z));
            pr[k] = sqrtf(sp); sc[k] = pr[k] + bias[e];
        } else { pr[k] = 0.f; sc[k] = -INFINITY; }
    }
    uint32_t used = 0; float wsum = 0.f;
    for (uint32_t r = 0; r < topk; r++) {
        float bv = -INFINITY; int bk = -1;
        #pragma unroll
        for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) if (!((used >> k) & 1u) && sc[k] > bv) { bv = sc[k]; bk = (int)k; }
        int be = bk >= 0 ? (int)(lane + 32u * (uint32_t)bk) : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off); const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (oe >= 0 && (be < 0 || ov > bv || (ov == bv && oe < be))) { bv = ov; be = oe; }
        }
        float myp = 0.f;
        if (be >= 0 && (uint32_t)(be & 31) == lane) {
            const uint32_t k = (uint32_t)be >> 5; used |= 1u << k;
            #pragma unroll
            for (uint32_t q = 0; q < V41_ROUTER_PER_LANE; q++) if (q == k) myp = pr[q];   /* 展开选择, 不动态下标 */
        }
        for (int off = 16; off > 0; off >>= 1) myp += __shfl_xor_sync(0xffffffffu, myp, off);
        if (lane == 0) { sel[(uint64_t)t * topk + r] = be; wts[(uint64_t)t * topk + r] = myp; }
        wsum += myp;
    }
    if (lane == 0)
        for (uint32_t r = 0; r < topk; r++) wts[(uint64_t)t * topk + r] = wts[(uint64_t)t * topk + r] / (wsum + 1e-20f) * route_scale;
}

typedef void (*rk_t)(int32_t *, float *, const float *, const float *, uint32_t, uint32_t, uint32_t, float);
static float time_k(rk_t k, int32_t *sel, float *wts, const float *lg, const float *bias, uint32_t n, uint32_t E, uint32_t K, int iters) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 10; i++) k<<<(n + 7u) / 8u, 256>>>(sel, wts, lg, bias, n, E, K, 2.5f);
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) k<<<(n + 7u) / 8u, 256>>>(sel, wts, lg, bias, n, E, K, 2.5f);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms * 1000.f / iters;
}
int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    const uint32_t n = argc > 2 ? (uint32_t)atoi(argv[2]) : 1u, E = 384u, K = 6u;
    float *hl = (float *)malloc(n * E * 4), *hb = (float *)malloc(E * 4);
    uint32_t s = 777u;
    for (uint32_t i = 0; i < n * E; i++) { s = s * 1664525u + 1013904223u; hl[i] = ((int)(s >> 8) % 20000 - 10000) * 1e-3f; }
    for (uint32_t i = 0; i < E; i++) { s = s * 1664525u + 1013904223u; hb[i] = ((int)(s >> 8) % 2000 - 1000) * 1e-3f; }
    float *dl, *db, *hbm, *wts0, *wts1; int32_t *sel0, *sel1;
    CK(cudaMalloc(&dl, n * E * 4)); CK(cudaMemcpy(dl, hl, n * E * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&db, E * 4)); CK(cudaMemcpy(db, hb, E * 4, cudaMemcpyHostToDevice));
    CK(cudaHostAlloc((void **)&hbm, E * 4, cudaHostAllocMapped)); memcpy(hbm, hb, E * 4);   /* 主机映射版 bias(引擎无 rb 侧车时 bias 走 mmap) */
    float *hbm_dev; CK(cudaHostGetDevicePointer((void **)&hbm_dev, hbm, 0));
    CK(cudaMalloc(&sel0, n * K * 4)); CK(cudaMalloc(&sel1, n * K * 4)); CK(cudaMalloc(&wts0, n * K * 4)); CK(cudaMalloc(&wts1, n * K * 4));
    const float t0 = time_k(k_router_v0, sel0, wts0, dl, db, n, E, K, iters);
    const float t0h = time_k(k_router_v0, sel0, wts0, dl, hbm_dev, n, E, K, iters);
    const float t1 = time_k(k_router_v1, sel1, wts1, dl, db, n, E, K, iters);
    /* 空核对照: 发射 + 启停的固定成本 */
    int32_t *hs0 = (int32_t *)malloc(n * K * 4), *hs1 = (int32_t *)malloc(n * K * 4); float *hw0 = (float *)malloc(n * K * 4), *hw1 = (float *)malloc(n * K * 4);
    CK(cudaMemcpy(hs0, sel0, n * K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs1, sel1, n * K * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hw0, wts0, n * K * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hw1, wts1, n * K * 4, cudaMemcpyDeviceToHost));
    const int same = memcmp(hs0, hs1, n * K * 4) == 0 && memcmp(hw0, hw1, n * K * 4) == 0;
    printf("router n_tok=%u E=%u topk=%u: V0(引擎抄本, bias 设备) %.2f us | V0 bias 主机映射 %.2f us | V1(去动态下标) %.2f us | %s\n",
           n, E, K, t0, t0h, t1, same ? "V1 == V0 逐位同" : "★V1 ≠ V0★");
    printf("  V0 sel[0..5] = %d %d %d %d %d %d, wts[0] = %.6f\n", hs0[0], hs0[1], hs0[2], hs0[3], hs0[4], hs0[5], hw0[0]);
    return 0;
}
