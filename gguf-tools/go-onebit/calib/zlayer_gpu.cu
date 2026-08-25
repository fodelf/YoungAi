/* zlayer_gpu.cu — zlayer.c 的 CUDA 热点卸载(cuBLAS 薄封装, 2026-08-25 二期)。
 * 依据: "spark 重计算必须 GPU 化"铁律; 一期纯 CPU ~570s/层, 热点全是 gemm 形
 * (教师/学生专家前向 255×3 gemm、Gram 4608²×4096、SVD 子空间迭代、k 曲线评估)。
 * 语义: 与 CPU 路"数值同语义(fp32/f64, 求和顺序重排容差)" —— 这正是 zlayer.py 自身
 * cupy/numpy 双路的既有契约(能力探测非行为开关), 金标口径=held 挽回打印精度一致。
 * 接口: C 可调 zg_* 族; zg_ready()==0 时调用方走 CPU 原路(能力探测, 无行为开关)。
 * 构建: nvcc -O3 -arch=native -c zlayer_gpu.cu -o zlayer_gpu.o
 *       gcc ... zlayer.c zlayer_gpu.o -DZL_CUDA -lcublas -lcudart -lstdc++ ... */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

static cublasHandle_t g_h = nullptr;
static int g_init = 0, g_ok = 0;

extern "C" int zg_ready(void) {
    if (!g_init) {
        g_init = 1;
        int n = 0;
        if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0 && cublasCreate(&g_h) == CUBLAS_STATUS_SUCCESS)
            g_ok = 1;
    }
    return g_ok;
}

/* 行主序 C = A(m×k) · B(k×n)ᵀ?  统一约定: 本封装做 行主序 C[m×n] = A[m×k]·B[n×k]ᵀ
 * (调用方矩阵都是行主序、右操作数按"行=输出列"存 —— zlayer 的 X@V / Xa@Xa.T / U 投影
 * 全是这个形。cuBLAS 列主序 ⇒ 等价调用 sgemm(N,T) 于转置视图。) */
extern "C" int zg_sgemm_nt(int m, int n, int k, const float *A, const float *B, float *C) {
    if (!zg_ready()) return 0;
    float *dA, *dB, *dC;
    size_t sa = (size_t)m * k * 4, sb = (size_t)n * k * 4, sc = (size_t)m * n * 4;
    if (cudaMalloc(&dA, sa) != cudaSuccess) return 0;
    if (cudaMalloc(&dB, sb) != cudaSuccess) { cudaFree(dA); return 0; }
    if (cudaMalloc(&dC, sc) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return 0; }
    cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice);
    const float one = 1.0f, zero = 0.0f;
    /* 列主序: C_cm[n×m] = B_cm[k×n]ᵀ · A_cm[k×m] ⇒ sgemm(T, N, n, m, k, B, k, A, k, C, n) */
    cublasStatus_t st = cublasSgemm(g_h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                    &one, dB, k, dA, k, &zero, dC, n);
    int ok = (st == CUBLAS_STATUS_SUCCESS);
    if (ok) cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

extern "C" int zg_dgemm_nt(int m, int n, int k, const double *A, const double *B, double *C) {
    if (!zg_ready()) return 0;
    double *dA, *dB, *dC;
    size_t sa = (size_t)m * k * 8, sb = (size_t)n * k * 8, sc = (size_t)m * n * 8;
    if (cudaMalloc(&dA, sa) != cudaSuccess) return 0;
    if (cudaMalloc(&dB, sb) != cudaSuccess) { cudaFree(dA); return 0; }
    if (cudaMalloc(&dC, sc) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return 0; }
    cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice);
    const double one = 1.0, zero = 0.0;
    cublasStatus_t st = cublasDgemm(g_h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                    &one, dB, k, dA, k, &zero, dC, n);
    int ok = (st == CUBLAS_STATUS_SUCCESS);
    if (ok) cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

/* 对称 Gram: C[m×m] = A[m×k]·Aᵀ (f64, ridge 由调用方加) — cublasDsyrk 半三角+镜像 */
extern "C" int zg_dgram(int m, int k, const double *A, double *C) {
    if (!zg_ready()) return 0;
    double *dA, *dC;
    size_t sa = (size_t)m * k * 8, sc = (size_t)m * m * 8;
    if (cudaMalloc(&dA, sa) != cudaSuccess) return 0;
    if (cudaMalloc(&dC, sc) != cudaSuccess) { cudaFree(dA); return 0; }
    cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice);
    const double one = 1.0, zero = 0.0;
    cublasStatus_t st = cublasDsyrk(g_h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                    m, k, &one, dA, k, &zero, dC, m);
    int ok = (st == CUBLAS_STATUS_SUCCESS);
    if (ok) {
        cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost);
        for (int i = 0; i < m; i++)   /* 下三角(列主序)=上三角(行主序) → 镜像补全 */
            for (int j = 0; j < i; j++) C[(size_t)i * m + j] = C[(size_t)j * m + i];
    }
    cudaFree(dA); cudaFree(dC);
    return ok;
}
