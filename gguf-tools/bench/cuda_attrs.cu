/* cuda_attrs.cu — 打印本卡几个决定核形态的硬限(2026-10-07): 每 SM / 每 block 的 shared 上限、寄存器、SM 数、L2。
 * 用法: nvcc -o cuda_attrs cuda_attrs.cu && ./cuda_attrs */
#include <cstdio>
#include <unistd.h>
#include <cuda_runtime.h>
int main(void) {
    int v = 0;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s CC %d.%d SM %d L2 %d MB bus %d bit\n", p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize >> 20, p.memoryBusWidth);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0); printf("shared/SM        %d KB\n", v >> 10);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);     printf("shared/block max %d KB\n", v >> 10);
    cudaDeviceGetAttribute(&v, cudaDevAttrReservedSharedMemoryPerBlock, 0);     printf("shared reserved  %d B/block\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxRegistersPerMultiprocessor, 0);    printf("regs/SM          %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerMultiProcessor, 0);      printf("threads/SM       %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxBlocksPerMultiprocessor, 0);       printf("blocks/SM        %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, 0);                      printf("L2               %d KB\n", v >> 10);
    cudaDeviceGetAttribute(&v, cudaDevAttrGlobalL1CacheSupported, 0);           printf("global L1 cache  %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0);                  printf("mem clock        %d kHz\n", v);
    /* 内存模型(2026-10-07, 引擎 ds4_gpu_unified_memory_host 的判据就是下面这两个属性): ATS = GPU 经主机页表访问整机内存(GB10/Grace),
     * integrated = iGPU 共享内存(Jetson)。独显即使开了 HMM, pageable access 可能为 1 但 ATS 为 0。 */
    cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, 0);             printf("pageable access  %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0); printf("ATS(host PT)     %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, 0);                       printf("integrated       %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrHostRegisterReadOnlySupported, 0);    printf("hostreg readonly %d\n", v);
    { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
      printf("cudaMemGetInfo   free %.2f / total %.2f GiB\n", f / 1073741824.0, t / 1073741824.0); }
    { const double pg = (double)sysconf(_SC_PAGESIZE), np = (double)sysconf(_SC_PHYS_PAGES);
      printf("phys RAM         %.2f GiB\n", pg * np / 1073741824.0); }
    return 0;
}
