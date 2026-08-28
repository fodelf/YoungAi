/* ds4quant_fwd.c — DeepSeek V4 前向 (C 移植, 自 dsv4_fwd.py 300 行 numpy)。
 * 目的: 让动态量化方案(逐层动态 z + 4损失 + 向前向后 + 感知)全 C 快跑, 逐层量化测试。
 * 移植纪律: 每模块和 numpy 对拍 (bit-close) 才继续; 用 st_db 读 HF (deepseek4-quantize 里)。
 *
 * 增量 1 (本文件当前): 数值原语 rms/silu/sigmoid/softmax/freqs_cis(yarn rope)/apply_rope/sinkhorn
 *   + --selftest 打印已知输入的输出, 与 numpy 参考逐值对比。
 * 待续: attention(MLA/compressor) / hc_pre_post / moe(gate/expert) / 动态量化 / 逐层测试。
 *
 * 编译(自测): cc -O3 -DDS4QUANT_SELFTEST -lm ds4quant_fwd.c -o ds4quant_selftest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846  /* Linux -std=c11 严格模式 math.h 不给(Darwin 给) */
#endif
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>

/* Apple Accelerate (AMX) sgemm: 大 matmul 走 BLAS, 标量路径留作参考/非苹果平台.
 * 累加同为 fp32, 与标量只差求和顺序(~1e-6 rel); anchor/quant 两遍同 kernel, 对比自洽. */
#if defined(__APPLE__) && !defined(DS4QUANT_NO_BLAS)
#define DQ_BLAS 1
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   /* 新 CBLAS 头(避免 macOS13.3+ 弃用警告); 不开 ILP64, int 仍 32 位 */
#endif
#include <Accelerate/Accelerate.h>
#elif defined(DS4QUANT_OPENBLAS)
/* Linux/Spark(GB10 aarch64) 移植(2026-08-17): cblas 接口同名同义, 编译加
 * -DDS4QUANT_OPENBLAS -lopenblas。标量路径仍是无 BLAS 时的参考。 */
#define DQ_BLAS 1
#include <cblas.h>
#endif

/* ---- 数值原语 (逐一对应 dsv4_fwd.py) ---- */

/* rms(x,w): x*rsqrt(mean(x²)+eps)*w. x[n], w[n] (per-row, 这里单行). */
void dq_rms(const float *x, const float *w, float *out, int n, float eps) {
    double v = 0.0;
    for (int i = 0; i < n; i++) v += (double)x[i] * (double)x[i];
    v = v / (double)n;
    float r = (float)(1.0 / sqrt(v + (double)eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static inline float dq_silu(float z) { return z / (1.0f + expf(-z)); }
static inline float dq_sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

/* softmax(z, axis=last) over n, 数值稳定 (减 max). */
void dq_softmax(const float *z, float *out, int n) {
    float m = z[0];
    for (int i = 1; i < n; i++) if (z[i] > m) m = z[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { out[i] = expf(z[i] - m); s += out[i]; }
    float inv = (float)(1.0 / s);
    for (int i = 0; i < n; i++) out[i] *= inv;
}

/* freqs_cis(dim, seqlen, orig, base, factor, bfast, bslow) → cos/sin 表 [seqlen, dim/2].
 * yarn NTK-by-parts (对应 numpy). cos_out/sin_out 预分配 [seqlen*(dim/2)]. */
static double dq_cdim(double nr, int dim, double orig, double base) {
    return (double)dim * log(orig / (nr * 2.0 * M_PI)) / (2.0 * log(base));
}
void dq_freqs_cis(int dim, int seqlen, double orig, double base, double factor,
                  double bfast, double bslow, float *cos_out, float *sin_out) {
    int half = dim / 2;
    double *fr = (double *)malloc((size_t)half * sizeof(double));
    for (int i = 0; i < half; i++) fr[i] = 1.0 / pow(base, (double)(2 * i) / (double)dim);
    if (orig > 0) {
        int lo = (int)floor(dq_cdim(bfast, dim, orig, base));
        int hi = (int)ceil(dq_cdim(bslow, dim, orig, base));
        if (lo < 0) lo = 0; if (hi > dim - 1) hi = dim - 1;
        if (lo == hi) hi = lo; /* numpy: hi+=0.001, 整数化后相同 → 用浮点 hi */
        double hif = (lo == (int)ceil(dq_cdim(bslow, dim, orig, base))) ? (double)hi + 0.001 : (double)hi;
        for (int i = 0; i < half; i++) {
            double ramp = ((double)i - (double)lo) / (hif - (double)lo);
            if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
            double smooth = 1.0 - ramp;
            fr[i] = fr[i] / factor * (1.0 - smooth) + fr[i] * smooth;
        }
    }
    for (int t = 0; t < seqlen; t++)
        for (int i = 0; i < half; i++) {
            double ang = (double)t * fr[i];
            cos_out[t * half + i] = (float)cos(ang);
            sin_out[t * half + i] = (float)sin(ang);
        }
    free(fr);
}

/* apply_rope: xp[rd] (单 token, 单 head) 就地旋转. cos/sin[rd/2] (该 token 行).
 * numpy: z=xc[0]+i*xc[1] (交错对), z*=(cos+i*sin) [inverse→conj]. */
void dq_apply_rope(float *xp, const float *cosr, const float *sinr, int rd, int inverse) {
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float re = xp[2 * i], im = xp[2 * i + 1];
        float c = cosr[i], s = inverse ? -sinr[i] : sinr[i];
        xp[2 * i]     = re * c - im * s;
        xp[2 * i + 1] = re * s + im * c;
    }
}

/* out[S,M] = X[S,K] @ W[M,K]^T  (W 行主序, 对应 numpy x@W.T). fp32 累加(近 numpy). */
#ifdef DS4QUANT_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif
#ifdef DS4QUANT_CUDA
typedef struct { float *p; size_t n; } dq_dbuf;
static int dq_dbuf_need(dq_dbuf *b, size_t n) {
    if (b->n >= n) return 1;
    if (b->p) cudaFree(b->p);
    b->p = NULL; b->n = 0;
    if (cudaMalloc((void **)&b->p, n * sizeof(float)) != cudaSuccess) { b->p = NULL; return 0; }
    b->n = n; return 1;
}
/* ★指针已在统一/显存里就别再拷★(2026-08-27, 128GiB GB10 适配)
 * bytes_moe 的批 dequant 把整层 256 专家×3 矩阵(26GiB)放在 cudaMallocManaged 里 —— GPU 本就
 * 直访。但下面的 GEMM 一直无条件 cudaMemcpy 到自己的暂存: 每调 33.5MB × 768 调/层 × 43 层
 * ≈ 1.1TB 纯浪费的 H2D。判一下指针类型, managed/device 直接喂 cuBLAS。
 * 失败或非托管指针 → 保持原拷贝路(小机器/非批模式行为逐字节不变)。
 * cudaPointerGetAttributes 对普通 malloc 指针在新版 CUDA 返回 success+Unregistered,
 * 老版返回 error 并污染 last_error —— 两种都要吞掉, 否则后续 cuda 调用误判失败。 */
static int dq_dev_ptr(const void *p) {
    if (!p) return 0;
    struct cudaPointerAttributes at;   /* C 里必须带 struct: CUDA 头的 typedef 是 C++ 专属 */
    cudaError_t e = cudaPointerGetAttributes(&at, p);
    if (e != cudaSuccess) { cudaGetLastError(); return 0; }
    return at.type == cudaMemoryTypeManaged || at.type == cudaMemoryTypeDevice;
}
static __thread cublasHandle_t g_dqh = NULL;
static __thread cudaStream_t g_dqs = NULL;
/* strided 两个变体各自的句柄/流。★必须放文件作用域★: 原来它们是函数内的 static __thread,
 * dq_gpu_thread_release 根本看不见 ⇒ 收不掉(2026-08-28 补)。 */
static __thread cublasHandle_t g_dqh2 = NULL, g_dqh3 = NULL;
static __thread cudaStream_t g_dqs2 = NULL, g_dqs3 = NULL;
static __thread dq_dbuf dX = {0, 0}, dW = {0, 0}, dO = {0, 0};
#endif
/* ★退出前必须调用(每层新建的专家 worker 线程)★: __thread CUDA 资源(句柄/流/显存暂存)
 * 随线程退出不自动释放 — 32k 锚实锤 ~2-3GB/层泄漏, L11 处 available 118→10G,
 * 杀进程后全量回收。非 CUDA 构建为空操作。 */
static void dq_gpu_thread_release(void) {
#ifdef DS4QUANT_CUDA
    if (dX.p) { cudaFree(dX.p); dX.p = NULL; dX.n = 0; }
    if (dW.p) { cudaFree(dW.p); dW.p = NULL; dW.n = 0; }
    if (dO.p) { cudaFree(dO.p); dO.p = NULL; dO.n = 0; }
    if (g_dqh) { cublasDestroy(g_dqh); g_dqh = NULL; }
    if (g_dqs) { cudaStreamDestroy(g_dqs); g_dqs = NULL; }
    if (g_dqh2) { cublasDestroy(g_dqh2); g_dqh2 = NULL; }
    if (g_dqs2) { cudaStreamDestroy(g_dqs2); g_dqs2 = NULL; }
    if (g_dqh3) { cublasDestroy(g_dqh3); g_dqh3 = NULL; }
    if (g_dqs3) { cudaStreamDestroy(g_dqs3); g_dqs3 = NULL; }
#endif
}
/* strided 版(08-18 attention GPU 化): C[S,M]=A[S,K](lda)·B[M,K](ldb)^T, 行主序任意行距 */
void dq_matmul_strided(const float *A, int lda, const float *B, int ldb,
                       float *Cst, int ldc, int S, int K, int M, float alpha) {
#ifdef DS4QUANT_CUDA
    if ((double)S * K * (double)M * 2.0 >= 2.0e8) {
        cublasHandle_t h2 = g_dqh2; cudaStream_t s2 = g_dqs2;
        if (!h2) {
            if (cublasCreate(&h2) != CUBLAS_STATUS_SUCCESS) h2 = NULL;
            else { cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking); cublasSetStream(h2, s2); }
            g_dqh2 = h2; g_dqs2 = s2;
        }
        if (h2) {
            const float zero = 0.0f;
            if (cublasSgemm(h2, CUBLAS_OP_T, CUBLAS_OP_N, M, S, K,
                            &alpha, B, ldb, A, lda, &zero, Cst, ldc) == CUBLAS_STATUS_SUCCESS &&
                cudaStreamSynchronize(s2) == cudaSuccess)
                return;
        }
    }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, M, K,
                alpha, A, lda, B, ldb, 0.0f, Cst, ldc);
#else
    for (int s = 0; s < S; s++)
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[(size_t)s * lda + k] * B[(size_t)m * ldb + k];
            Cst[(size_t)s * ldc + m] = acc * alpha;
        }
#endif
}
/* NT 版: C[S,M]=A[S,K](lda)·B[K,M](B 行主序[N,HD]视为 K=N 行 M=HD 列→NoTrans) */
void dq_matmul_nt_strided(const float *A, int lda, const float *B, int ldb,
                          float *Cst, int ldc, int S, int K, int M) {
#ifdef DS4QUANT_CUDA
    if ((double)S * K * (double)M * 2.0 >= 2.0e8) {
        cublasHandle_t h3 = g_dqh3; cudaStream_t s3 = g_dqs3;
        if (!h3) {
            if (cublasCreate(&h3) != CUBLAS_STATUS_SUCCESS) h3 = NULL;
            else { cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking); cublasSetStream(h3, s3); }
            g_dqh3 = h3; g_dqs3 = s3;
        }
        if (h3) {
            const float one = 1.0f, zero = 0.0f;
            /* RowMajor C=A·B ⇔ ColMajor C'=B'·A' */
            if (cublasSgemm(h3, CUBLAS_OP_N, CUBLAS_OP_N, M, S, K,
                            &one, B, ldb, A, lda, &zero, Cst, ldc) == CUBLAS_STATUS_SUCCESS &&
                cudaStreamSynchronize(s3) == cudaSuccess)
                return;
        }
    }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, M, K,
                1.0f, A, lda, B, ldb, 0.0f, Cst, ldc);
#else
    for (int s = 0; s < S; s++)
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[(size_t)s * lda + k] * B[(size_t)k * ldb + m];
            Cst[(size_t)s * ldc + m] = acc;
        }
#endif
}
void dq_matmul(const float *X, const float *W, float *out, int S, int K, int M) {
#ifdef DS4QUANT_CUDA
    /* GB10 统一内存 cuBLAS 直传(2026-08-18 用户令"必须使用GPU"): malloc 指针 GPU 直访
     * (探针 relerr 1.6e-7, 热点尺寸 2906x4096x2048=3.67ms=13.3 TFLOPS vs CPU 单线程 ~3.5s)。
     * 小 GEMM 走 CPU(launch+sync ~100µs 不划算); per-thread handle+stream: 外层专家
     * pthread 并发提交 GPU 多流; 任一 CUDA 失败静默落 CPU 路径(数值语义同, fp32 同精度)。 */
    /* 阈=2e8(08-18 实测校准): 只让 g_r 级大 GEMM(24 GFLOP)上 GPU。5e6 低阈实测负收益
     * (259s vs 181s/层): 20 线程高频小 GEMM 并发提交, launch+sync 队列争用吃掉全部收益。
     * 高频小矩阵的正确姿势是批量结构改造(GPTQ 段 GPU 常驻), 不是逐调用换后端。 */
    if ((double)S * K * (double)M * 2.0 >= 2.0e8) {
        if (!g_dqh) {
            if (cublasCreate(&g_dqh) != CUBLAS_STATUS_SUCCESS) g_dqh = NULL;
            else { cudaStreamCreateWithFlags(&g_dqs, cudaStreamNonBlocking); cublasSetStream(g_dqh, g_dqs); }
        }
        /* ★连续操作数走显存暂存(2026-08-22)★
         * 专家 GEMM 每次换一块 33.5MB 的新反量化权重: 留在主机靠 ATS 直访 = 页粒度搬运,
         * 实测等效 3.2 GB/s, 每层 25.7GB ⇒ ~8s。这里 X/W/out 全是连续的(无行距),
         * 一次 cudaMemcpy 批量搬即可, 不会退化成跨步 DMA(那是 attention 那条路的坑,
         * 它已改走 vqg_attention 整层驻留)。暂存 per-thread 复用, 按需增长;
         * ★退出线程必须 dq_gpu_thread_release(2026-08-26 泄漏实锤见下)★ */
        /* 已在统一/显存的操作数免拷(见 dq_dev_ptr 头注)。W 是热点: 批 dequant 模式下它就是
         * managed 的 g_bmw_buf 切片。X/out 通常是普通 malloc, 仍走暂存。 */
        const int wdev = dq_dev_ptr(W), xdev = dq_dev_ptr(X), odev = dq_dev_ptr(out);
        if (g_dqh && (xdev || dq_dbuf_need(&dX, (size_t)S * K))
                  && (wdev || dq_dbuf_need(&dW, (size_t)M * K))
                  && (odev || dq_dbuf_need(&dO, (size_t)S * M))) {
            const float one = 1.0f, zero = 0.0f;
            const size_t f = sizeof(float);
            const float *Xd = xdev ? X : dX.p, *Wd = wdev ? W : dW.p;
            float *Od = odev ? out : dO.p;
            int ok = 1;
            if (!xdev && cudaMemcpyAsync(dX.p, X, (size_t)S*K*f, cudaMemcpyHostToDevice, g_dqs) != cudaSuccess) ok = 0;
            if (ok && !wdev && cudaMemcpyAsync(dW.p, W, (size_t)M*K*f, cudaMemcpyHostToDevice, g_dqs) != cudaSuccess) ok = 0;
            /* RowMajor C[S,M]=X·W^T ⇔ ColMajor C'[M,S]=W'^T·X' */
            if (ok && cublasSgemm(g_dqh, CUBLAS_OP_T, CUBLAS_OP_N, M, S, K,
                            &one, Wd, K, Xd, K, &zero, Od, M) != CUBLAS_STATUS_SUCCESS) ok = 0;
            if (ok && !odev && cudaMemcpyAsync(out, dO.p, (size_t)S*M*f, cudaMemcpyDeviceToHost, g_dqs) != cudaSuccess) ok = 0;
            if (ok && cudaStreamSynchronize(g_dqs) == cudaSuccess) return;
            cudaGetLastError();   /* 失败则落 CPU 路, 先清错免污染后续调用 */
        }
    }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, M, K,
                1.0f, X, K, W, K, 0.0f, out, M);
#else
    for (int s = 0; s < S; s++) {
        const float *xr = X + (size_t)s * K;
        for (int m = 0; m < M; m++) {
            const float *wr = W + (size_t)m * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xr[k] * wr[k];
            out[(size_t)s * M + m] = acc;
        }
    }
#endif
}

/* expert_fp: h = silu(clip(x@w1.T, ≤lim)) * clip(x@w3.T, [-lim,lim]); out = (weight*h) @ w2.T.
 * x[S,DIM], w1/w3[MOEI,DIM], w2[DIM,MOEI]. out[S,DIM] 累加进 acc (weight per-token 或 NULL). */
void dq_expert_fp(const float *x, const float *w1, const float *w3, const float *w2,
                  const float *weight, float *acc_out, int S, int DIM, int MOEI, float swlim) {
    float *g = (float *)malloc((size_t)S * MOEI * sizeof(float));
    float *u = (float *)malloc((size_t)S * MOEI * sizeof(float));
    dq_matmul(x, w1, g, S, DIM, MOEI);
    dq_matmul(x, w3, u, S, DIM, MOEI);
    for (size_t i = 0; i < (size_t)S * MOEI; i++) {
        float gg = g[i], uu = u[i];
        if (swlim > 0) { if (uu > swlim) uu = swlim; if (uu < -swlim) uu = -swlim; if (gg > swlim) gg = swlim; }
        g[i] = dq_silu(gg) * uu;   /* h */
    }
    if (weight) for (int s = 0; s < S; s++) for (int j = 0; j < MOEI; j++) g[(size_t)s*MOEI+j] *= weight[s];
    /* out += h @ w2.T  (w2[DIM,MOEI]) — 走 dq_matmul(大 GEMM 上 GPU 显存暂存路)再累加。
     * 2026-08-26 用户令 GPU 化: 原来这里直落 cblas β=1, 共享专家 550 GFLOP/层与 256 个
     * routed 第三矩阵全部单线程 CPU 磨(锚定遍 46% 墙钟的主凶)。β=0 乘积+显式加法与
     * β=1 sgemm 的乘积段同序 ⇒ CPU 路数值逐位不变; GPU 路精度与前两 GEMM 同级。 */
    {
        float *t2 = (float *)malloc((size_t)S * DIM * sizeof(float));
        dq_matmul(g, w2, t2, S, MOEI, DIM);
        for (size_t i = 0; i < (size_t)S * DIM; i++) acc_out[i] += t2[i];
        free(t2);
    }
    free(g); free(u);
}

/* hc_sinkhorn: mixes[S, HCM+HCM+HCM*HCM] → pre[S,HCM], post[S,HCM], comb[S,HCM,HCM]. */
void dq_hc_sinkhorn(const float *mixes, const float *scale, const float *base,
                    float *pre, float *post, float *comb, int S, int HCM, int HCIT, float HCEPS) {
    for (int s = 0; s < S; s++) {
        const float *mx = mixes + (size_t)s * (2*HCM + HCM*HCM);
        for (int j = 0; j < HCM; j++) pre[(size_t)s*HCM+j]  = dq_sigmoid(mx[j]*scale[0]+base[j]) + HCEPS;
        for (int j = 0; j < HCM; j++) post[(size_t)s*HCM+j] = 2.0f*dq_sigmoid(mx[HCM+j]*scale[1]+base[HCM+j]);
        /* comb[HCM,HCM] = softmax over axis2 (last) of mixes[2HCM:].reshape(HCM,HCM)*scale2+base2, +eps */
        float *cb = comb + (size_t)s*HCM*HCM;
        for (int a = 0; a < HCM; a++) {
            float row[64];
            for (int b = 0; b < HCM; b++) row[b] = mx[2*HCM + a*HCM + b]*scale[2] + base[2*HCM + a*HCM + b];
            float sm[64]; dq_softmax(row, sm, HCM);
            for (int b = 0; b < HCM; b++) cb[a*HCM+b] = sm[b] + HCEPS;
        }
        /* comb /= comb.sum(axis1=行内跨a?, keepdims) — numpy: comb.sum(1)=沿HCM第一维(a) */
        /* numpy comb[S,HCM,HCM]; sum(1)=对a求和→[S,1,HCM(b)]; comb/=that. 然后迭代 HCIT-1 次 sum(2) then sum(1). */
        float colsum[64];
        for (int b = 0; b < HCM; b++) { double s2=0; for (int a=0;a<HCM;a++) s2+=cb[a*HCM+b]; colsum[b]=(float)s2+HCEPS; }
        for (int a=0;a<HCM;a++) for (int b=0;b<HCM;b++) cb[a*HCM+b] /= colsum[b];
        for (int it=0; it<HCIT-1; it++) {
            for (int a=0;a<HCM;a++){ double r=0; for(int b=0;b<HCM;b++) r+=cb[a*HCM+b]; float rr=(float)r+HCEPS; for(int b=0;b<HCM;b++) cb[a*HCM+b]/=rr; }
            for (int b=0;b<HCM;b++){ double c=0; for(int a=0;a<HCM;a++) c+=cb[a*HCM+b]; float cc=(float)c+HCEPS; for(int a=0;a<HCM;a++) cb[a*HCM+b]/=cc; }
        }
    }
}

/* hc_pre: h[S,HCM,DIM] → y[S,DIM], post[S,HCM], comb[S,HCM,HCM].
 * x=h.reshape(S,HCM*DIM); mixes=(x@fn.T)*rsqrt(mean(x²)); sinkhorn; y=Σ_j pre[j]*h[j]. */
/* ---- hc 混合族 s 切片并行(2026-08-26 用户令提速: 32k 锚定遍相位计时定罪
 * hc7-8s+mix8s/层全在单线程) —— 各 s 行完全独立, 按行切线程 = 数值逐位不变
 * (dq_softmax_rows_par 同纪律)。mode: 0=hc_pre mixes  1=hc_pre y 混合  2=hc_post。 ---- */
typedef struct {
    int mode, HCM, DIM, HD, mixdim;
    const float *h, *fn, *pre, *a, *resid, *post, *comb;
    float *mixes, *y, *out; float EPS;
} dq_hcw;
static void dq_hc_range(dq_hcw *w, int a0, int b0) {
    if (w->mode == 0) {
        for (int s = a0; s < b0; s++) {
            const float *x = w->h + (size_t)s * w->HD;
            double v = 0.0; for (int i = 0; i < w->HD; i++) v += (double)x[i]*(double)x[i];
            float rsq = (float)(1.0/sqrt(v/(double)w->HD + (double)w->EPS));
            float *mx = w->mixes + (size_t)s * w->mixdim;
            for (int m = 0; m < w->mixdim; m++) {
                const float *fr = w->fn + (size_t)m * w->HD;
                float acc = 0.0f; for (int k = 0; k < w->HD; k++) acc += x[k]*fr[k];
                mx[m] = acc * rsq;
            }
        }
    } else if (w->mode == 1) {
        for (int s = a0; s < b0; s++)
            for (int d = 0; d < w->DIM; d++) {
                float acc = 0.0f;
                for (int j = 0; j < w->HCM; j++)
                    acc += w->pre[(size_t)s*w->HCM+j] * w->h[((size_t)s*w->HCM+j)*w->DIM+d];
                w->y[(size_t)s*w->DIM+d] = acc;
            }
    } else {
        for (int s = a0; s < b0; s++)
            for (int j = 0; j < w->HCM; j++) {
                float pj = w->post[(size_t)s*w->HCM+j];
                const float *cj = w->comb + ((size_t)s*w->HCM+j)*w->HCM;
                float *o = w->out + ((size_t)s*w->HCM+j)*w->DIM;
                const float *ar = w->a + (size_t)s*w->DIM;
                for (int d = 0; d < w->DIM; d++) {
                    float acc = pj * ar[d];
                    for (int k = 0; k < w->HCM; k++) acc += cj[k] * w->resid[((size_t)s*w->HCM+k)*w->DIM+d];
                    o[d] = acc;
                }
            }
    }
}
typedef struct { dq_hcw *w; int a, b; } dq_hcj;
static void *dq_hc_thr(void *arg) { dq_hcj *j = (dq_hcj *)arg; dq_hc_range(j->w, j->a, j->b); return NULL; }
static void dq_hc_par(dq_hcw *w, int S) {
    static int NTH = 0;   /* 线程数=在线核数(dq_softmax_rows_par 同式, 不读 env) */
    if (!NTH) { long nc = sysconf(_SC_NPROCESSORS_ONLN); NTH = (int)(nc < 1 ? 1 : (nc > 64 ? 64 : nc)); }
    if (NTH <= 1 || S < 256) { dq_hc_range(w, 0, S); return; }
    pthread_t th[64]; dq_hcj js[64];
    int per = (S + NTH - 1) / NTH, n = 0;
    for (int t = 0; t < NTH; t++) {
        int a = t * per, b = a + per > S ? S : a + per;
        if (a >= b) break;
        js[n] = (dq_hcj){w, a, b};
        if (pthread_create(&th[n], NULL, dq_hc_thr, &js[n]) != 0) dq_hc_range(w, a, b);
        else n++;
    }
    for (int t = 0; t < n; t++) pthread_join(th[t], NULL);
}

void dq_hc_pre(const float *h, const float *fn, const float *scale, const float *base,
               float *y, float *post, float *comb, int S, int HCM, int DIM,
               int mixdim, int HCIT, float EPS, float HCEPS) {
    int HD = HCM * DIM;
    float *mixes = (float *)malloc((size_t)S * mixdim * sizeof(float));
    float *pre = (float *)malloc((size_t)S * HCM * sizeof(float));
    dq_hcw w0 = { 0, HCM, DIM, HD, mixdim, h, fn, NULL, NULL, NULL, NULL, NULL,
                  mixes, NULL, NULL, EPS };
    dq_hc_par(&w0, S);
    dq_hc_sinkhorn(mixes, scale, base, pre, post, comb, S, HCM, HCIT, HCEPS);
    dq_hcw w1 = { 1, HCM, DIM, HD, mixdim, h, NULL, pre, NULL, NULL, NULL, NULL,
                  NULL, y, NULL, EPS };
    dq_hc_par(&w1, S);
    free(mixes); free(pre);
}

/* hc_post: out[S,HCM,DIM] = post[:,:,None]*a[:,None,:] + einsum('sjk,skd->sjd', comb, resid). */
void dq_hc_post(const float *a, const float *resid, const float *post, const float *comb,
                float *out, int S, int HCM, int DIM) {
    dq_hcw w2 = { 2, HCM, DIM, HCM * DIM, 0, NULL, NULL, NULL, a, resid, post, comb,
                  NULL, NULL, out, 0.0f };
    dq_hc_par(&w2, S);
}

