/* cuda_commit_probe.cu — 这台机器上 cudaMalloc 是不是"分了就占"(2026-09-29)。
 * 为什么要它: 服务端单条长请求的 MemAvailable 曲线里有一格 −162 MB 的台阶, 正好等于索引草稿 [2048][ng] 按 ng 翻倍到
 * 16384 的尺寸(134 + 34 MB) —— 而解码时那块只写第 1 行。若 cudaMalloc 延迟提交, 这块的物理占用应当只有 0.5 MB;
 * 若立即提交(驱动清零/统一内存直接映射), 则"按 cap_tok 行分配"就是实打实的浪费。同样决定 v41_grow 那种 free+malloc
 * 的碎片会不会真吃余量(cudaFree 回不回系统)。
 * 用法: cuda_commit_probe [MB=1024]   按顺序打印: 分配前 → cudaMalloc 后 → cudaMemset 后 → cudaFree 后 的 MemAvailable。
 * 只碰它自己分的那块, 对同机其他进程零影响(别在余量 < 2×MB 时跑)。 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static long avail_mb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f); return kb < 0 ? -1 : kb / 1024;
}

int main(int argc, char **argv) {
    const size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 1024u;
    const size_t bytes = mb << 20;
    void *p = NULL;
    (void)cudaFree(0);   /* 先把上下文建好, 别把上下文本身的几百 MB 算进分配账 */
    usleep(300000);
    const long a0 = avail_mb();
    if (cudaMalloc(&p, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc %zu MB 失败\n", mb); return 1; }
    usleep(300000);
    const long a1 = avail_mb();
    if (cudaMemset(p, 1, bytes) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "memset 失败\n"); return 1; }
    usleep(300000);
    const long a2 = avail_mb();
    (void)cudaFree(p);
    usleep(300000);
    const long a3 = avail_mb();
    printf("MemAvailable MB: 上下文就绪 %ld → cudaMalloc(%zu MB) 后 %ld (占 %ld) → memset 后 %ld (占 %ld) → cudaFree 后 %ld (还回 %ld)\n",
           a0, mb, a1, a0 - a1, a2, a0 - a2, a3, a3 - a2);
    return 0;
}
