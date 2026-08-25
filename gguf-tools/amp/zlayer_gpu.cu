/* zlayer_gpu.cu — zlayer.c 的 CUDA 热点卸载(cuBLAS 通用行主序 gemm, 2026-08-25 二期)。
 * 依据: "spark 重计算必须 GPU 化"铁律; CPU(openblas 20线程) ~585s/层, 热点全在
 * mm64/mm32 两个中央 gemm 入口(教师/学生前向、Gram、SVD 子空间迭代、k 曲线评估)。
 * 语义契约 = zlayer.py 自身 cupy/numpy 双路先例: 能力探测非行为开关, 数值同语义
 * (求和顺序重排容差); CUDA 构建的金标口径 = held 挽回打印精度(0.1pp), 非逐字符。
 * 行主序 gemm: C[M×N] = op(A)·op(B) 经列主序恒等式 C_cm = op(B)_cm·op(A)_cm 直传 cuBLAS。
 * 构建: nvcc -O3 -arch=native -c zlayer_gpu.cu -o zlayer_gpu.o
 *       gcc ... -DZL_CUDA zlayer.c zlayer_gpu.o -lcublas -lcudart -lstdc++ */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

static cublasHandle_t g_h = nullptr;
static int g_init = 0, g_ok = 0;
/* 设备缓冲复用(层内多次大 gemm, 避免每次 cudaMalloc) */
static void *g_dA = nullptr, *g_dB = nullptr, *g_dC = nullptr;
static size_t g_sA = 0, g_sB = 0, g_sC = 0;

extern "C" int zg_ready(void) {
    if (!g_init) {
        g_init = 1;
        int n = 0;
        if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0 && cublasCreate(&g_h) == CUBLAS_STATUS_SUCCESS)
            g_ok = 1;
    }
    return g_ok;
}
static int ensure(void **p, size_t *cur, size_t need) {
    if (*cur >= need) return 1;
    if (*p) cudaFree(*p);
    *p = nullptr; *cur = 0;
    if (cudaMalloc(p, need) != cudaSuccess) return 0;
    *cur = need;
    return 1;
}

/* 通用行主序 gemm(与 cblas_?gemm(RowMajor, ta, tb, M,N,K, 1, A,lda, B,ldb, 0, C,ldc) 同参型)。
 * 返回 1=完成, 0=不可用/失败(调用方走 CPU 原路)。 */
extern "C" int zg_dgemm(int ta, int tb, int M, int N, int K,
                        const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
    if (!zg_ready()) return 0;
    size_t ra = (size_t)(ta ? K : M) * lda * 8, rb = (size_t)(tb ? N : K) * ldb * 8, rc = (size_t)M * ldc * 8;
    if (!ensure(&g_dA, &g_sA, ra) || !ensure(&g_dB, &g_sB, rb) || !ensure(&g_dC, &g_sC, rc)) return 0;
    if (cudaMemcpy(g_dA, A, ra, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_dB, B, rb, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    const double one = 1.0, zero = 0.0;
    /* 行主序→列主序: C_cm[N×M] = op(B)_cm · op(A)_cm */
    cublasStatus_t st = cublasDgemm(g_h,
        tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K, &one, (const double *)g_dB, ldb, (const double *)g_dA, lda, &zero, (double *)g_dC, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    return cudaMemcpy(C, g_dC, rc, cudaMemcpyDeviceToHost) == cudaSuccess;
}
extern "C" int zg_sgemm(int ta, int tb, int M, int N, int K,
                        const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    if (!zg_ready()) return 0;
    size_t ra = (size_t)(ta ? K : M) * lda * 4, rb = (size_t)(tb ? N : K) * ldb * 4, rc = (size_t)M * ldc * 4;
    if (!ensure(&g_dA, &g_sA, ra) || !ensure(&g_dB, &g_sB, rb) || !ensure(&g_dC, &g_sC, rc)) return 0;
    if (cudaMemcpy(g_dA, A, ra, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_dB, B, rb, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t st = cublasSgemm(g_h,
        tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K, &one, (const float *)g_dB, ldb, (const float *)g_dA, lda, &zero, (float *)g_dC, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    return cudaMemcpy(C, g_dC, rc, cudaMemcpyDeviceToHost) == cudaSuccess;
}
