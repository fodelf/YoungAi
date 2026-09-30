/* cuda_graph_churn_probe.cu — CUDA graph 反复"实例化 → 销毁"到底还不还内存(2026-09-29)。
 * 为什么要它: 服务端单条长请求余量曲线扣掉索引草稿的翻倍台阶后, 还剩 ~23 KB/token 的爬升; 解码整步 graph 每 1024 token
 * 一个位置桶, 桶里 1~6 行各重捕一张 ~1459 节点的图(cudaGraphExecDestroy 旧的 + cudaGraphInstantiate 新的), 折合每张
 * 3~4 MB 才对得上。cudaMalloc 的大块 cudaFree 会还回系统(cuda_commit_probe 实测), exec 的驱动内存不一定。
 * 用法: cuda_graph_churn_probe [节点数=1459] [轮数=120]
 *   模式 A: 每轮 捕获 → 实例化新 exec → 发一次 → 销毁旧 exec        (= 现役 core_decode_graph.c 的做法)
 *   模式 B: 第一轮实例化, 之后每轮 捕获 → cudaGraphExecUpdate 原地更新 → 发一次(= 候选改法)
 *   各打印 每 20 轮的 MemAvailable 与总跌幅。核是空核, 不占 GPU。 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void nop_kernel(int *p, int v) { if (p && threadIdx.x == 0) p[blockIdx.x] = v; }

static long avail_mb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f); return kb < 0 ? -1 : kb / 1024;
}

static cudaGraph_t capture(cudaStream_t s, int *buf, int nodes, int salt) {
    cudaGraph_t g = NULL;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return NULL;
    for (int i = 0; i < nodes; i++) nop_kernel<<<4 + (salt % 8), 32, 0, s>>>(buf, i + salt);   /* grid 随轮变, 像换桶 */
    if (cudaStreamEndCapture(s, &g) != cudaSuccess) return NULL;
    return g;
}

static int run(int mode, int nodes, int rounds, int *buf, cudaStream_t s) {
    cudaGraphExec_t exec = NULL;
    const long a0 = avail_mb();
    long amin = a0;
    for (int r = 0; r < rounds; r++) {
        cudaGraph_t g = capture(s, buf, nodes, r);
        if (!g) { fprintf(stderr, "捕获失败 轮 %d\n", r); return 1; }
        if (mode == 0 || !exec) {
            cudaGraphExec_t ne = NULL;
            if (cudaGraphInstantiate(&ne, g, 0) != cudaSuccess) { fprintf(stderr, "实例化失败 轮 %d: %s\n", r, cudaGetErrorString(cudaGetLastError())); return 1; }
            if (exec) (void)cudaGraphExecDestroy(exec);
            exec = ne;
        } else {
            cudaGraphExecUpdateResultInfo info;
            const cudaError_t ue = cudaGraphExecUpdate(exec, g, &info);
            if (ue != cudaSuccess) { fprintf(stderr, "原地更新失败 轮 %d: %s (result %d)\n", r, cudaGetErrorString(ue), (int)info.result); (void)cudaGetLastError(); return 1; }
        }
        (void)cudaGraphDestroy(g);
        if (cudaGraphLaunch(exec, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) { fprintf(stderr, "发图失败 轮 %d\n", r); return 1; }
        if ((r + 1) % 20 == 0) { usleep(200000); const long a = avail_mb(); if (a < amin) amin = a; printf("  模式 %c 轮 %3d: MemAvailable %ld MB (较起点 %+ld)\n", mode ? 'B' : 'A', r + 1, a, a - a0); fflush(stdout); }
    }
    if (exec) (void)cudaGraphExecDestroy(exec);
    usleep(300000);
    const long a1 = avail_mb();
    printf("模式 %c 收工: 起点 %ld → 全部销毁后 %ld (净 %+ld MB, 谷底 %ld)\n", mode ? 'B' : 'A', a0, a1, a1 - a0, amin);
    return 0;
}

int main(int argc, char **argv) {
    const int nodes = argc > 1 ? atoi(argv[1]) : 1459, rounds = argc > 2 ? atoi(argv[2]) : 120;
    int *buf = NULL;
    if (cudaMalloc((void **)&buf, 4096 * sizeof(int)) != cudaSuccess) return 1;
    cudaStream_t s; if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return 1;
    printf("节点 %d × 轮 %d\n", nodes, rounds);
    if (run(0, nodes, rounds, buf, s)) return 1;
    if (run(1, nodes, rounds, buf, s)) return 1;
    return 0;
}
