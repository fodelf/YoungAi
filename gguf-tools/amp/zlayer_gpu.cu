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

/* ---- cusolver 扩展(2026-08-26 zloss 提速令: potrf/potrs 与 QR 正交化上卡;
 * 12288 维 Cholesky(~10s) 与标量 mgs(~15s) 是 gemm 上卡后的 CPU 剩余热点)。
 * 同一能力探测契约: 任何失败返 0, 调用方走 CPU 原路。 */
#include <cusolverDn.h>
static cusolverDnHandle_t g_sh = nullptr;
static int g_sinit = 0, g_sok = 0;
static void *g_dW2 = nullptr; static size_t g_sW2 = 0;
static int *g_dinfo = nullptr;
extern "C" int zg_solver_ready(void) {
    if (!g_sinit) {
        g_sinit = 1;
        if (zg_ready() && cusolverDnCreate(&g_sh) == CUSOLVER_STATUS_SUCCESS &&
            cudaMalloc((void **)&g_dinfo, 4) == cudaSuccess)
            g_sok = 1;
    }
    return g_sok;
}
static int zg_info_ok(void) {
    int info = -1;
    return cudaMemcpy(&info, g_dinfo, 4, cudaMemcpyDeviceToHost) == cudaSuccess && info == 0;
}
/* V: 行主序 d×k 就地正交化(薄 QR 的 Q)。geam 转置进出列主序。
 * 与 mgs 产的基不同但张成同一子空间 —— 截断积 U·diag(z)·Vᵀ 对基不变,
 * 金标口径 = selftest 收回率与针表打印精度, 非逐位。 */
extern "C" int zg_sqr_orth(int d, int k, float *V) {
    if (!zg_solver_ready()) return 0;
    size_t sv = (size_t)d * k * 4;
    if (!ensure(&g_dA, &g_sA, sv) || !ensure(&g_dB, &g_sB, sv)) return 0;
    if (cudaMemcpy(g_dA, V, sv, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    const float one = 1.0f, zero = 0.0f;
    /* g_dA(行主序 d×k = 列主序 k×d) → g_dB(列主序 d×k) */
    if (cublasSgeam(g_h, CUBLAS_OP_T, CUBLAS_OP_N, d, k, &one, (const float *)g_dA, k,
                    &zero, (const float *)g_dB, d, (float *)g_dB, d) != CUBLAS_STATUS_SUCCESS)
        return 0;
    int lw1 = 0, lw2 = 0;
    if (cusolverDnSgeqrf_bufferSize(g_sh, d, k, (float *)g_dB, d, &lw1) != CUSOLVER_STATUS_SUCCESS)
        return 0;
    size_t need = ((size_t)(lw1 > k ? lw1 : k) + (size_t)k) * 4;
    if (!ensure(&g_dW2, &g_sW2, need)) return 0;
    float *dtau = (float *)g_dW2, *dwork = dtau + k;
    int lw = (int)((g_sW2 / 4) - k);
    if (cusolverDnSorgqr_bufferSize(g_sh, d, k, k, (float *)g_dB, d, dtau, &lw2)
        != CUSOLVER_STATUS_SUCCESS) return 0;
    if (lw2 > lw) {
        if (!ensure(&g_dW2, &g_sW2, ((size_t)lw2 + k) * 4)) return 0;
        dtau = (float *)g_dW2; dwork = dtau + k; lw = lw2;
    }
    if (cusolverDnSgeqrf(g_sh, d, k, (float *)g_dB, d, dtau, dwork, lw, g_dinfo)
        != CUSOLVER_STATUS_SUCCESS || !zg_info_ok()) return 0;
    if (cusolverDnSorgqr(g_sh, d, k, k, (float *)g_dB, d, dtau, dwork, lw, g_dinfo)
        != CUSOLVER_STATUS_SUCCESS || !zg_info_ok()) return 0;
    /* g_dB(列主序 d×k) → g_dA(行主序 d×k) */
    if (cublasSgeam(g_h, CUBLAS_OP_T, CUBLAS_OP_N, k, d, &one, (const float *)g_dB, d,
                    &zero, (const float *)g_dA, k, (float *)g_dA, k) != CUBLAS_STATUS_SUCCESS)
        return 0;
    return cudaMemcpy(V, g_dA, sv, cudaMemcpyDeviceToHost) == cudaSuccess;
}

/* f32 因子化+解(2026-08-26 计时定罪: GB10 FP64=1:64 阉割, f64 potrf/potrs 在卡上
 * 跟 CPU 一样慢, ftA 12288 维 6 次因子化吃掉 47s/53s。f32 + 无量纲 ridge(条件数
 * ≤1/λ≈333)精度富余, 产物载荷本就是 fp16 —— zlayer ftA f32 Gram 冠军先例同族)。
 * A: 行主序 n×n 全对称 f64(入参不动, 卡上转 f32); B: 列主序 n×nrhs f64 进出。 */
static __global__ void zg_d2s(const double *s, float *d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (float)s[i];
}
static __global__ void zg_s2d(const float *s, double *d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (double)s[i];
}
extern "C" int zg_spotrf_potrs_f64io(int n, int nrhs, const double *A, double *B) {
    if (!zg_solver_ready()) return 0;
    size_t sa8 = (size_t)n * n * 8, sb8 = (size_t)n * nrhs * 8;
    size_t sa4 = sa8 / 2, sb4 = sb8 / 2;
    if (!ensure(&g_dA, &g_sA, sa8) || !ensure(&g_dB, &g_sB, sb8) ||
        !ensure(&g_dC, &g_sC, sa4 > sb4 ? sa4 + sb4 : sb4 + sa4)) return 0;
    float *fA = (float *)g_dC, *fB = fA + (size_t)n * n;
    if (cudaMemcpy(g_dA, A, sa8, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_dB, B, sb8, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    size_t na = (size_t)n * n, nb = (size_t)n * nrhs;
    zg_d2s<<<(unsigned)((na + 255) / 256), 256>>>((const double *)g_dA, fA, na);
    zg_d2s<<<(unsigned)((nb + 255) / 256), 256>>>((const double *)g_dB, fB, nb);
    int lwork = 0;
    if (cusolverDnSpotrf_bufferSize(g_sh, CUBLAS_FILL_MODE_LOWER, n, fA, n, &lwork)
        != CUSOLVER_STATUS_SUCCESS) return 0;
    if (!ensure(&g_dW2, &g_sW2, (size_t)lwork * 4)) return 0;
    if (cusolverDnSpotrf(g_sh, CUBLAS_FILL_MODE_LOWER, n, fA, n, (float *)g_dW2, lwork,
                         g_dinfo) != CUSOLVER_STATUS_SUCCESS || !zg_info_ok()) return 0;
    if (cusolverDnSpotrs(g_sh, CUBLAS_FILL_MODE_LOWER, n, nrhs, fA, n, fB, n, g_dinfo)
        != CUSOLVER_STATUS_SUCCESS || !zg_info_ok()) return 0;
    zg_s2d<<<(unsigned)((nb + 255) / 256), 256>>>(fB, (double *)g_dB, nb);
    return cudaMemcpy(B, g_dB, sb8, cudaMemcpyDeviceToHost) == cudaSuccess;
}
