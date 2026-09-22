/* v41_route_solve.cu — 路由偏置侧车(路由反修)的 GPU 数值件(2026-09-20)。接口与口径见 v41_route_solve.h。 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "v41_route_solve.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "★CUDA %s @%s:%d: %s★\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "★cuBLAS %s @%s:%d: %d★\n", #x, __FILE__, __LINE__, (int)s_); return -1; } } while (0)
#define NB(n) ((unsigned)(((n) + 255) / 256))

typedef struct { void *p; size_t cap; } dbuf;
static int grow(dbuf *b, size_t n) {
    if (b->cap >= n) return 0;
    if (b->p) cudaFree(b->p);
    b->p = NULL; b->cap = 0;
    CK(cudaMalloc(&b->p, n));
    b->cap = n;
    return 0;
}
static cublasHandle_t g_bl = NULL;
static dbuf g_logits, g_p, g_raw;

__global__ static void k_bf16_to_f32(const uint16_t *in, float *out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = (uint32_t)in[i] << 16;
    float f; memcpy(&f, &u, 4); out[i] = f;
}
/* p = sqrt(softplus(z)), score = p + bias (+Δb): 逐元素 */
__global__ static void k_route_scores(const float *logits, const float *B, const float *Db, float *p, float *s, long long n, int E) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = (int)(i % E);
    const float z = logits[i];
    const float sp = z > 20.0f ? z : log1pf(expf(z));
    const float pv = sqrtf(sp);
    p[i] = pv; s[i] = pv + B[e] + (Db ? Db[e] : 0.0f);
}
/* 一线程一 token: K 轮 argmax(严格大于才换 ⇒ 同分取小号, 与引擎 warp 版语义同); 权重 = p/(Σp+1e-20)·scale。E ≤ 1024。 */
__global__ static void k_route_topk(const float *p, const float *s, int n, int E, int K, float scale, int *sel, float *w) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const float *pt = p + (size_t)t * E, *st = s + (size_t)t * E;
    int *so = sel + (size_t)t * K; float *wo = w + (size_t)t * K;
    uint32_t used[32]; for (int i = 0; i < 32; i++) used[i] = 0u;
    float wsum = 0.f;
    for (int k = 0; k < K; k++) {
        float bv = -INFINITY; int be = -1;
        for (int e = 0; e < E; e++) { if ((used[e >> 5] >> (e & 31)) & 1u) continue; const float v = st[e]; if (v > bv) { bv = v; be = e; } }
        so[k] = be; wo[k] = be >= 0 ? pt[be] : 0.f; wsum += wo[k];
        if (be >= 0) used[be >> 5] |= 1u << (be & 31);
    }
    for (int k = 0; k < K; k++) wo[k] = wo[k] / (wsum + 1e-20f) * scale;
}

extern "C" int v41_route_upload_gate_gpu(const uint16_t *hWg_bf16, const float *hB, int E, int D, float **dWg_out, float **dB_out) {
    const size_t nw = (size_t)E * D;
    if (grow(&g_raw, nw * 2)) return -1;
    float *dWg = NULL, *dB = NULL;
    CK(cudaMalloc((void **)&dWg, nw * 4));
    CK(cudaMalloc((void **)&dB, (size_t)E * 4));
    CK(cudaMemcpy(g_raw.p, hWg_bf16, nw * 2, cudaMemcpyHostToDevice));
    k_bf16_to_f32<<<NB((long long)nw), 256>>>((const uint16_t *)g_raw.p, dWg, (long long)nw);
    CK(cudaGetLastError());
    CK(cudaMemcpy(dB, hB, (size_t)E * 4, cudaMemcpyHostToDevice));
    *dWg_out = dWg; *dB_out = dB;
    return 0;
}

extern "C" int v41_route_recompute_gpu(const float *dX, const float *dWg, const float *dB, const float *dDb, int n, int D, int E, int K, float scale,
                                       int *dsel_out, float *dw_out, float *dscore_out) {
    if (!g_bl) CB(cublasCreate(&g_bl));
    if (E > 1024 || K > 16 || !dscore_out) { fprintf(stderr, "★路由重算: E=%d K=%d 或缺 score 缓冲★\n", E, K); return -1; }
    const size_t nE = (size_t)n * E;
    if (grow(&g_logits, nE * 4) || grow(&g_p, nE * 4)) return -1;
    /* logits[n][E] = X[n][D]·Wgᵀ: 列主序 L_cm(E×n) = Wg_cmᵀ(E×D)·X_cm(D×n), Wg 行主序 [E][D] 即 Wg_cm(D×E) ld=D */
    const float a = 1.f, b = 0.f;
    CB(cublasSgemm(g_bl, CUBLAS_OP_T, CUBLAS_OP_N, E, n, D, &a, dWg, D, dX, D, &b, (float *)g_logits.p, E));
    k_route_scores<<<NB((long long)nE), 256>>>((const float *)g_logits.p, dB, dDb, (float *)g_p.p, dscore_out, (long long)nE, E);
    CK(cudaGetLastError());
    k_route_topk<<<NB(n), 256>>>((const float *)g_p.p, dscore_out, n, E, K, scale, dsel_out, dw_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    return 0;
}
