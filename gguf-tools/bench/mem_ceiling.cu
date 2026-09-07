/* mem_ceiling.cu — 量本机 GPU 真实可达的 DRAM 读带宽(2026-09-05)。
 *
 * 【为什么要它】解码是带宽受限: 每 token 读 backbone 7.7 GB + VQ 专家 1.75 GB。
 * "273 GB/s" 是 GB10 的规格数, 不是 kernel 能拿到的数 —— kernel 到底能读多快只能量。
 * 量两种口径: ① cudaMemcpy D2D(拷贝: 读+写各 1 份) ② 只读归约 kernel(每线程 int4 向量读,
 * 网格跨步, 结果写 1 个数; 这是 gemv 类 kernel 的访存形态)。
 * 用法: mem_ceiling [GiB=4] [iters=10]   (nvcc -O3 -o mem_ceiling mem_ceiling.cu) */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

__global__ void read_reduce(const int4 *__restrict__ p, size_t n4, unsigned long long *out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        const int4 v = __ldg(p + i);
        acc ^= (unsigned)v.x ^ (unsigned)v.y ^ (unsigned)v.z ^ (unsigned)v.w;
    }
    if (acc == 0x12345678u) atomicAdd(out, 1ull);   /* 防止被优化掉, 几乎不触发 */
}
/* ③ 4 B/lane 窄读(VQ 位流暂存的形态): 同样网格跨步, 每线程每步只读 4 B */
__global__ void read_reduce_u32(const unsigned *__restrict__ p, size_t n, unsigned long long *out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc ^= __ldg(p + i);
    if (acc == 0x12345678u) atomicAdd(out, 1ull);
}
/* ④ 行式流: 每 warp 顺序读自己的 ROWS 行(每行 RB 字节, 4 B/lane × RB/128 条指令), 行首相邻;
 *    模拟 VQ gateup 每 warp 18 KB 私有区 + 数百 warp 并发的 DRAM 形态 */
__global__ void read_rows_u32(const unsigned *__restrict__ p, size_t rb_words, size_t rows_per_warp, unsigned long long *out) {
    const unsigned lane = threadIdx.x & 31u;
    const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    const unsigned *base = p + warp * rows_per_warp * rb_words;
    unsigned acc = 0;
    for (size_t r = 0; r < rows_per_warp; r++) {
        const unsigned *row = base + r * rb_words;
        for (size_t w = lane; w < rb_words; w += 32) acc ^= __ldg(row + w);
    }
    if (acc == 0x12345678u) atomicAdd(out, 1ull);
}

int main(int argc, char **argv) {
    const double gib = argc > 1 ? atof(argv[1]) : 4.0;
    const int iters = argc > 2 ? atoi(argv[2]) : 10;
    const size_t bytes = (size_t)(gib * 1024.0 * 1024.0 * 1024.0) & ~(size_t)4095;
    char *a = NULL, *b = NULL; unsigned long long *o = NULL;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&o, 8));
    CK(cudaMemset(a, 1, bytes)); CK(cudaMemset(b, 2, bytes)); CK(cudaMemset(o, 0, 8));
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    int clk = 0, mclk = 0;   /* CUDA 13 的 cudaDeviceProp 已删 clockRate 字段, 走属性查询 */
    (void)cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    (void)cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
    printf("%s SMs=%d clk=%.2fGHz memclk=%.2fGHz bus=%d bit  测试块 %.2f GiB\n",
           pr.name, pr.multiProcessorCount, clk / 1e6, mclk / 1e6, pr.memoryBusWidth, bytes / 1073741824.0);
    /* 占用率账要的硬件常量: kernel 每 SM 能驻几个 block 由这几项定 */
    printf("  shared/SM=%zu KB  shared/block(opt-in)=%zu KB  regs/SM=%d  threads/SM=%d  blocks/SM=%d  L2=%d MB\n",
           pr.sharedMemPerMultiprocessor / 1024, pr.sharedMemPerBlockOptin / 1024, pr.regsPerMultiprocessor,
           pr.maxThreadsPerMultiProcessor, pr.maxBlocksPerMultiProcessor, pr.l2CacheSize / (1024 * 1024));
    /* ① D2D 拷贝 */
    CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int i = 0; i < iters; i++) CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    double dt = (now_s() - t0) / iters;
    printf("① D2D memcpy: %.3f ms/次, 读+写 %.1f GB/s(单向 %.1f GB/s)\n", dt * 1e3, 2.0 * bytes / dt / 1e9, bytes / dt / 1e9);
    /* ② 只读归约, 几种网格规模 */
    const size_t n4 = bytes / 16;
    for (int blocks_per_sm = 2; blocks_per_sm <= 16; blocks_per_sm *= 2) {
        const int grid = pr.multiProcessorCount * blocks_per_sm, block = 256;
        read_reduce<<<grid, block>>>((const int4 *)a, n4, o); CK(cudaDeviceSynchronize());
        t0 = now_s();
        for (int i = 0; i < iters; i++) read_reduce<<<grid, block>>>((const int4 *)a, n4, o);
        CK(cudaDeviceSynchronize());
        dt = (now_s() - t0) / iters;
        printf("② 只读 int4 归约 grid=%d×%d: %.3f ms/次, %.1f GB/s\n", grid, block, dt * 1e3, bytes / dt / 1e9);
    }
    {   /* ③ 4 B/lane 窄读 */
        const int grid = pr.multiProcessorCount * 8, block = 256;
        read_reduce_u32<<<grid, block>>>((const unsigned *)a, bytes / 4, o); CK(cudaDeviceSynchronize());
        t0 = now_s();
        for (int i = 0; i < iters; i++) read_reduce_u32<<<grid, block>>>((const unsigned *)a, bytes / 4, o);
        CK(cudaDeviceSynchronize());
        dt = (now_s() - t0) / iters;
        printf("③ 只读 u32 归约 grid=%d×%d: %.3f ms/次, %.1f GB/s\n", grid, block, dt * 1e3, bytes / dt / 1e9);
    }
    {   /* ④ 行式流: 1152 B 行 × 16 行/warp, 并发 warp 数 = bytes / 18 KB */
        const size_t rbw = 288, rpw = 16, per_warp = rbw * rpw * 4;
        const size_t nwarps = bytes / per_warp;
        const int block = 128;
        const int grid = (int)(nwarps / (block / 32));
        /* 用动态 shared 占位压占用率: 100 KB/SM 下 34 KB ⇒ 2 block/SM(=VQ gateup 实况), 0 ⇒ 满占用 */
        const size_t shm_list[3] = {0, 34 * 1024, 17 * 1024};
        for (int rep = 0; rep < 3; rep++) {
            const size_t shm = shm_list[rep];
            read_rows_u32<<<grid, block, shm>>>((const unsigned *)a, rbw, rpw, o); CK(cudaDeviceSynchronize());
            t0 = now_s();
            for (int i = 0; i < iters; i++) read_rows_u32<<<grid, block, shm>>>((const unsigned *)a, rbw, rpw, o);
            CK(cudaDeviceSynchronize());
            dt = (now_s() - t0) / iters;
            printf("④ 行式 u32 流(1152 B 行×16/warp, %d block×4 warp, shared %zu KB): %.3f ms/次, %.1f GB/s\n",
                   grid, shm / 1024, dt * 1e3, (double)nwarps * per_warp / dt / 1e9);
        }
    }
    return 0;
}
