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
/* ⑤ 指针追逐延迟(2026-09-17): 一个 warp 顺着随机环链一跳一跳读, 每跳等上一跳回来 ⇒ 量的是**单次访存往返**。
 * 为什么要它: VQ 专家核判成"延迟受限"之后, 下一刀该把 load 提前多少轮发, 全看这块板子 L2 命中 / DRAM 各要
 * 多少纳秒 —— 仓里从来没量过, 全凭猜。链按 128 B 线随机排, 步长足够乱, 硬件预取器帮不上忙。 */
__global__ void chase_lat(const unsigned *__restrict__ p, int n, unsigned long long *out, long long *cyc) {
    unsigned idx = 0;
    const long long t0 = clock64();
    for (int i = 0; i < n; i++) idx = __ldg(p + idx);
    const long long t1 = clock64();
    if (threadIdx.x == 0) { *cyc = t1 - t0; if (idx == 0xfffffff0u) atomicAdd(out, 1ull); }
}
/* ⑥ VQ 位流形态的"在飞深度"尺(2026-09-17): 每 warp 私有 960 B 行, 每轮 12 个 lane 各读一个 32 位字(48 B),
 * 同一条流里下一轮的地址**依赖**上一轮读回的值(假依赖, 值恒 0 但编译器不知道) ⇒ 一条流一次只挂一个请求;
 * 一个 warp 同时走 D 条流 ⇒ 每 warp 恰好 D 轮在飞。48 SM × 32 warp(和专家核一样 1 block/SM, 64 KB shared 压占用率)。
 * 读法: D=1 的 GB/s 就是专家核现在的形态上限; 从哪个 D 起贴到 240, 专家核的寄存器环就得做多深。 */
/* 变体(同夜第二轮, 专家核加了行预取只到 133 GB/s = 这形态的上限之后): 一轮几个 lane 各读 4 B(LANES: 12 = 48 B 现役形态,
 * 16 = 64 B, 32 = 128 B 整线); HINT 0 = 普通读, 1 = `ld.global.L2::128B`(叫 L2 顺手把整线拉进来), 2 = `L2::256B`;
 * PF 1 = 每行开头 30 个 lane 把整行 30 个扇区 prefetch 进 L2(专家核步 0 第二版的做法)。
 * 读法: 哪个变体到 240, 专家核的位流读法就照它写 —— 请求粒度是这个核最后一堵墙, 别再猜。 */
template <int HINT>
__device__ __forceinline__ unsigned chase_ld(const unsigned char *a) {
    unsigned v;
    if (HINT == 1) asm("ld.global.nc.L2::128B.u32 %0, [%1];" : "=r"(v) : "l"(a));
    else if (HINT == 2) asm("ld.global.nc.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(a));
    else v = __ldg((const unsigned *)a);
    return v;
}
template <int D, int LANES, int HINT, int PF>
__global__ void chase_rows(const unsigned char *__restrict__ p, size_t rows_per_warp, unsigned long long *out) {
    const unsigned lane = threadIdx.x & 31u;
    const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    const unsigned char *rowbase = p + warp * rows_per_warp * 960u;
    const unsigned char *base = rowbase + (lane % LANES) * 4u;
    const int rounds = 960 / (LANES * 4);
    unsigned acc = 0;
    for (size_t r = 0; r + D <= rows_per_warp; r += D) {
        unsigned dep[D];
        #pragma unroll
        for (int d = 0; d < D; d++) {
            dep[d] = 0;
            if (PF) { const unsigned char *s = rowbase + (r + d) * 960u + lane * 32u; if (lane < 30u) asm volatile("prefetch.global.L2 [%0];" :: "l"(s)); }
        }
        for (int k = 0; k < rounds; k++) {
            #pragma unroll
            for (int d = 0; d < D; d++) {
                const unsigned v = chase_ld<HINT>(base + (r + d) * 960u + (unsigned)(LANES * 4) * k + (dep[d] & 4u));
                dep[d] = v; acc ^= v;
            }
        }
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
    {   /* ⑤ 指针追逐: 足迹 16 KB(L1) / 8 MB(L2) / 2 GiB(DRAM), 每档 20 万跳取 ns/跳(clock64 按最高时钟折算, 另给壁钟口径) */
        const size_t foot[3] = { 16u << 10, 8u << 20, (size_t)2 << 30 };
        const char *nm[3] = { "16 KB(L1)", "8 MB(L2)", "2 GiB(DRAM)" };
        long long *cyc = NULL; CK(cudaMalloc(&cyc, 8));
        unsigned *perm = (unsigned *)malloc(bytes);
        for (int f = 0; f < 3; f++) {
            if (foot[f] > bytes) break;
            const size_t nl = foot[f] / 128;                      /* 线数; 线 i 的首字存下一线的字下标 */
            unsigned *order = (unsigned *)malloc(nl * 4);
            for (size_t i = 0; i < nl; i++) order[i] = (unsigned)i;
            srand(12345);
            for (size_t i = nl - 1; i > 0; i--) { size_t j = ((size_t)rand() * 65536u + (size_t)rand()) % (i + 1); unsigned t = order[i]; order[i] = order[j]; order[j] = t; }
            for (size_t i = 0; i < nl; i++) perm[(size_t)order[i] * 32] = order[(i + 1) % nl] * 32u;   /* 随机环 */
            free(order);
            CK(cudaMemcpy(a, perm, foot[f], cudaMemcpyHostToDevice));
            const int n = 200000;
            chase_lat<<<1, 32>>>((const unsigned *)a, 20000, o, cyc); CK(cudaDeviceSynchronize());   /* 热身: 小足迹进缓存 */
            t0 = now_s();
            chase_lat<<<1, 32>>>((const unsigned *)a, n, o, cyc); CK(cudaDeviceSynchronize());
            dt = now_s() - t0;
            long long hc = 0; CK(cudaMemcpy(&hc, cyc, 8, cudaMemcpyDeviceToHost));
            printf("⑤ 指针追逐 %-12s: %.0f 周期/跳 = %.0f ns/跳(按 %.2f GHz), 壁钟口径 %.0f ns/跳\n",
                   nm[f], (double)hc / n, (double)hc / n / (clk / 1e6), clk / 1e6, dt / n * 1e9);
        }
        /* ⑦ 带载延迟(2026-09-18): 另一条流上用 ② 把带宽拉满(b 块), 同时在 a 上追逐 DRAM 足迹的链 ⇒ 队列排满时一跳多少 ns。
         * 为什么要它: 专家核提前几块发位流全看这个数 —— 空载 377 ns 若带载变 2~3 µs, 提前一块(8 轮)就盖不住。 */
        {
            cudaStream_t sa, sb; CK(cudaStreamCreate(&sa)); CK(cudaStreamCreate(&sb));
            const int n = 100000, grid = pr.multiProcessorCount * 4, block = 256;
            for (int rep = 0; rep < 2; rep++) {
                const int loaded = rep;
                t0 = now_s();
                if (loaded) for (int i = 0; i < 6; i++) read_reduce<<<grid, block, 0, sb>>>((const int4 *)b, n4, o);
                chase_lat<<<1, 32, 0, sa>>>((const unsigned *)a, n, o, cyc);
                CK(cudaStreamSynchronize(sa));
                const double tc = now_s() - t0;
                CK(cudaDeviceSynchronize());
                long long hc = 0; CK(cudaMemcpy(&hc, cyc, 8, cudaMemcpyDeviceToHost));
                printf("⑦ 2 GiB 足迹追逐, %s: %.0f 周期/跳 = %.0f ns/跳(壁钟 %.0f ns)\n", loaded ? "另一流满带宽流读中(带载)" : "空载",
                       (double)hc / n, (double)hc / n / (clk / 1e6), tc / n * 1e9);
            }
            CK(cudaStreamDestroy(sa)); CK(cudaStreamDestroy(sb));
        }
        free(perm); CK(cudaFree(cyc));
        CK(cudaMemset(a, 1, bytes));   /* 追逐把 a 写成了链表, ⑥ 要的是全 1 */
    }
    {   /* ⑥ 位流在飞深度: 48 SM × 1 block × 1024 线程, 64 KB shared 压成 1 block/SM(= 专家核实况) */
        const int grid = pr.multiProcessorCount, block = 1024;
        const size_t nwarps = (size_t)grid * (block / 32);
        const size_t rpw = bytes / nwarps / 960u;
        const size_t shm = 64u << 10;
        #define CHASE(D, L, H, PF) do { \
            CK(cudaFuncSetAttribute(chase_rows<D, L, H, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
            chase_rows<D, L, H, PF><<<grid, block, shm>>>((const unsigned char *)a, rpw, o); CK(cudaDeviceSynchronize()); \
            t0 = now_s(); \
            for (int i = 0; i < iters; i++) chase_rows<D, L, H, PF><<<grid, block, shm>>>((const unsigned char *)a, rpw, o); \
            CK(cudaDeviceSynchronize()); \
            dt = (now_s() - t0) / iters; \
            printf("⑥ 位流形态 %2d 轮在飞 × 每轮 %3d B(%s%s): %.3f ms/次, %.1f GB/s\n", \
                   D, L * 4, H == 0 ? "普通读" : H == 1 ? "L2::128B" : "L2::256B", PF ? " + 行首扇区预取" : "", \
                   dt * 1e3, (double)nwarps * (rpw / D * D) * (double)((960 / (L * 4)) * L * 4) / dt / 1e9); } while (0)
        CHASE(1, 12, 0, 0); CHASE(4, 12, 0, 0); CHASE(16, 12, 0, 0);
        CHASE(1, 12, 1, 0); CHASE(4, 12, 1, 0);
        CHASE(1, 12, 2, 0); CHASE(4, 12, 2, 0);
        CHASE(1, 12, 0, 1); CHASE(4, 12, 0, 1);
        CHASE(1, 16, 0, 0); CHASE(4, 16, 0, 0);
        CHASE(1, 32, 0, 0); CHASE(4, 32, 0, 0);
        #undef CHASE
    }
    return 0;
}
