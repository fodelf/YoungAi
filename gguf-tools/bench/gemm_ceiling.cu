/* gemm_ceiling.cu — 量本机 cuBLAS SGEMM 在【反修实际形状】上的天花板(2026-08-27)。
 *
 * 【为什么要它】反修前向实测 82 GFLOPS, 而 GB10 该是 TFLOPS 量级 —— 差 100 倍以上。
 * 但"该是多少"是我推的, 不是量的。先把天花板量出来, 才知道差距住在哪:
 *   · 天花板也低  ⇒ 是精度/硬件问题(fp32 非张量核), 该动 math mode
 *   · 天花板很高  ⇒ 是我们的 per-call 开销(拷贝/同步/小 GEMM 碎片), 该动调用结构
 * 三种口径都量: ①全在显存(纯算力) ②W 托管+X 主机(当前 bytes_moe 形态) ③每调都拷(改前形态)
 *
 * 形状取自 bytes_moe: 每专家 nt≈256 个 token, w1: [nt,4096]×[2048,4096]^T
 * 用法: gemm_ceiling [nt=256] [iters=50]
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define DIM 4096
#define MOEI 2048
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 256;
    int iters = argc > 2 ? atoi(argv[2]) : 50;
    double gflop = 2.0 * nt * DIM * MOEI * 1e-9;
    cublasHandle_t h; cublasCreate(&h);
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking); cublasSetStream(h, st);

    float *dX, *dW, *dO, *mW, *hX, *hO;
    CK(cudaMalloc(&dX,(size_t)nt*DIM*4)); CK(cudaMalloc(&dW,(size_t)MOEI*DIM*4)); CK(cudaMalloc(&dO,(size_t)nt*MOEI*4));
    CK(cudaMallocManaged(&mW,(size_t)MOEI*DIM*4));           /* 托管: 与 g_bmw_buf 同款 */
    hX=(float*)malloc((size_t)nt*DIM*4); hO=(float*)malloc((size_t)nt*MOEI*4);
    for(size_t i=0;i<(size_t)nt*DIM;i++) hX[i]=0.01f;
    for(size_t i=0;i<(size_t)MOEI*DIM;i++) mW[i]=0.01f;
    CK(cudaMemcpy(dX,hX,(size_t)nt*DIM*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW,mW,(size_t)MOEI*DIM*4,cudaMemcpyHostToDevice));
    const float one=1.f, zero=0.f;
    #define GEMM(Wp,Xp,Op) cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,MOEI,nt,DIM,&one,(Wp),DIM,(Xp),DIM,&zero,(Op),MOEI)

    printf("形状 nt=%d  单次 %.3f GFLOP  ×%d 次\n", nt, gflop, iters);
    for (int mode = 0; mode < 2; mode++) {
        cublasSetMathMode(h, mode ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
        const char *mn = mode ? "TF32张量核" : "FP32(现用)";
        /* ①纯算力: 全在显存, 无拷贝 */
        GEMM(dW,dX,dO); CK(cudaStreamSynchronize(st));
        double t0=now_s(); for(int i=0;i<iters;i++) GEMM(dW,dX,dO);
        CK(cudaStreamSynchronize(st)); double t1=now_s();
        printf("  [%s] ①全显存无拷贝     %7.2f GFLOPS  (%.2f ms/次)\n", mn, gflop*iters/(t1-t0), (t1-t0)*1000/iters);
        /* ②当前形态: W 托管直访, X/out 每次拷 */
        t0=now_s();
        for(int i=0;i<iters;i++){ cudaMemcpyAsync(dX,hX,(size_t)nt*DIM*4,cudaMemcpyHostToDevice,st);
            GEMM(mW,dX,dO); cudaMemcpyAsync(hO,dO,(size_t)nt*MOEI*4,cudaMemcpyDeviceToHost,st); CK(cudaStreamSynchronize(st)); }
        t1=now_s();
        printf("  [%s] ②W托管+X拷+逐次同步 %7.2f GFLOPS  (%.2f ms/次)\n", mn, gflop*iters/(t1-t0), (t1-t0)*1000/iters);
        /* ③改前形态: W 也每次拷 */
        t0=now_s();
        for(int i=0;i<iters;i++){ cudaMemcpyAsync(dX,hX,(size_t)nt*DIM*4,cudaMemcpyHostToDevice,st);
            cudaMemcpyAsync(dW,mW,(size_t)MOEI*DIM*4,cudaMemcpyHostToDevice,st);
            GEMM(dW,dX,dO); cudaMemcpyAsync(hO,dO,(size_t)nt*MOEI*4,cudaMemcpyDeviceToHost,st); CK(cudaStreamSynchronize(st)); }
        t1=now_s();
        printf("  [%s] ③W也每次拷(改前)    %7.2f GFLOPS  (%.2f ms/次)\n", mn, gflop*iters/(t1-t0), (t1-t0)*1000/iters);
    }
    return 0;
}
