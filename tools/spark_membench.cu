// spark_membench.cu — GB10 kernel 读带宽微基准(判决 ds4 kernel 的 95GB/s 是墙还是没写好)
//   nvcc -O3 -o /tmp/membench tools/spark_membench.cu && /tmp/membench
// 三种模式: 1) 大跨度顺序流读  2) warp 行读(模拟 staging: 每 warp 一段 1344B 行)
//           3) 行读+shared staging(和 ds4 kernel 同构)
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define GB (1024ull*1024ull*1024ull)

__global__ void stream_read_kernel(const uint4 *src, uint64_t n16, float *sink) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint4 acc = {0,0,0,0};
    for (; i < n16; i += stride) { uint4 v = src[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if (acc.x == 0xdeadbeefu) sink[0] = (float)(acc.x + acc.y + acc.z + acc.w);
}

// 每 warp 处理一条 84-uint4 (1344B) 的"行", 行间跨度 row_stride16 — 模拟专家行访问
__global__ void row_read_kernel(const uint4 *src, uint64_t n_rows, uint64_t row16, float *sink) {
    uint64_t row = (uint64_t)blockIdx.x * (blockDim.x/32u) + (threadIdx.x>>5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= n_rows) return;
    const uint4 *r = src + row * row16;
    uint4 acc = {0,0,0,0};
    for (uint32_t i = lane; i < row16; i += 32u) { uint4 v = r[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if (acc.x == 0xdeadbeefu) sink[0] = 1.f;
}

// 行读 + shared staging + 假计算(同 ds4 gateup 结构)
__global__ void row_stage_kernel(const uint4 *src, uint64_t n_rows, uint64_t row16, float *sink) {
    extern __shared__ uint4 st[];
    uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    uint64_t row = (uint64_t)blockIdx.x * (blockDim.x/32u) + warp;
    if (row >= n_rows) return;
    const uint4 *r = src + row * row16;
    uint4 *my = st + (uint64_t)warp * row16;
    for (uint32_t i = lane; i < row16; i += 32u) my[i] = r[i];
    __syncwarp();
    uint32_t acc = 0;
    for (uint32_t i = lane; i < row16; i += 32u) acc ^= my[i].x ^ my[i].y ^ my[i].z ^ my[i].w;
    if (acc == 0xdeadbeefu) sink[0] = 1.f;
}


// ==== 真 q2_K dot 版(从 ds4_cuda.cu 同构抄): 判定 95GB/s 差距是否全在 dot 指令流 ====
struct blk_q2K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };   // 84B
struct blk_q8K { float d; int8_t qs[256]; int16_t bsums[16]; };
__device__ static float f16f32(uint16_t h){ __half x = __ushort_as_half(h); return __half2float(x); }
__device__ static int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}
__device__ static float dot_block(const blk_q2K *x, const blk_q8K *y) {
    const uint8_t *q2 = x->qs; const int8_t *q8 = y->qs; const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * f16f32(x->d), dmin = y->d * f16f32(x->dmin);
    int isum0 = 0, isum1 = 0, is = 0;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f, d1 = sc[is++] & 0x0f;
            isum0 += d0 * dot_q2_16(q2, q8, shift);
            isum1 += d1 * dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2; q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}
// 与 ds4 gateup 同构: 每 warp stage 2 行×84 uint4, 半 warp 各算 16 块(每 lane 1 块)
__global__ void row_stage_dot_kernel(const uint4 *src, uint64_t n_pairs, const blk_q8K *xq, float *sink) {
    extern __shared__ uint4 st[];
    uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    uint64_t pair = (uint64_t)blockIdx.x * (blockDim.x/32u) + warp;
    if (pair >= n_pairs) return;
    const uint4 *r = src + pair * 168u;          // 2 行共 168 uint4
    uint4 *my = st + (uint64_t)warp * 168u;
    for (uint32_t i = lane; i < 168u; i += 32u) my[i] = r[i];
    __syncwarp();
    const uint32_t half = lane >> 4u, l16 = lane & 15u;
    const blk_q2K *wr = (const blk_q2K *)((const uint8_t *)my + half * 1344u);
    float acc = dot_block(wr + l16, xq + l16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0 && acc == 12345.678f) sink[0] = acc;
}

static void run(const char *name, void(*launch)(const uint4*,uint64_t,float*,cudaStream_t), const uint4 *buf, uint64_t bytes, float *sink) {
    // 预热1次 + 计时5次
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(buf, bytes, sink, 0); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i=0;i<5;i++) launch(buf, bytes, sink, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    printf("%-28s %8.1f GB/s\n", name, 5.0*bytes/(ms*1e-3)/1e9);
}

int main() {
    const uint64_t bytes = 8ull*GB;
    uint4 *buf; float *sink;
    if (cudaMalloc(&buf, bytes) != cudaSuccess) { printf("alloc fail\n"); return 1; }
    cudaMalloc(&sink, 4);
    cudaMemset(buf, 0x5a, bytes);
    run("stream(grid-stride)", [](const uint4*b,uint64_t n,float*s,cudaStream_t){
        stream_read_kernel<<<48*16, 256>>>(b, n/16u, s); }, buf, bytes, sink);
    run("row_read(1344B/warp)", [](const uint4*b,uint64_t n,float*s,cudaStream_t){
        row_read_kernel<<<(unsigned)((n/1344u+7)/8), 256>>>(b, n/1344u, 84u, s); }, buf, bytes, sink);
    run("row+shared_stage(ds4同构)", [](const uint4*b,uint64_t n,float*s,cudaStream_t){
        row_stage_kernel<<<(unsigned)((n/1344u+7)/8), 256, 8*84*16>>>(b, n/1344u, 84u, s); }, buf, bytes, sink);
    run("row+stage+真q2Kdot", [](const uint4*b,uint64_t n,float*s,cudaStream_t){
        static blk_q8K *xq = NULL;
        if (!xq) { cudaMalloc(&xq, 16*sizeof(blk_q8K)); cudaMemset(xq, 1, 16*sizeof(blk_q8K)); }
        uint64_t pairs = n/2688u;
        row_stage_dot_kernel<<<(unsigned)((pairs+7)/8), 256, 8*168*16>>>(b, pairs, xq, s); }, buf, bytes, sink);
    return 0;
}
