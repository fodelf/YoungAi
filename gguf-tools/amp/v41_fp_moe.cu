/* v41_fp_moe.cu — 反修靶的 FP(出厂精度)MoE 块评估器(2026-09-13), 接口见 v41_fp_moe.h。
 *
 * 【为什么单独一份而不用引擎的核】引擎的 routed 路读 VQ blob、shared 路读 GGUF fp4x32, 都是"量化后的字节"; 靶要的是
 * HF 出厂字节(FP4 打包 + 独立 scale 张量 / E4M3 + 32×32 块 scale), 字节布局不同, 但算式必须一样。所以这里只有
 * "HF 字节 → f16" 是新写的, gather / SwiGLU / reduce / bf16 舍入四个核逐式抄引擎(cuda_vq_prefill.inc.cu 的
 * vqp_gather_kernel / vqp_swiglu_kernel / vqp_reduce_kernel, cuda_v41_1.inc.cu 的 v41_bf16r / v41_x_to_f16_kernel,
 * cuda_v41_3.inc.cu 的 v41_swiglu_kernel), 改了引擎那边的舍入点这里要跟着改 —— 否则靶里混进实现差, 放大器学的是"两种
 * 实现的差"而不是"量化误差"。
 * 【量级】每层 384 专家 × 3 矩阵 ≈ 7.2 GB 出厂字节从 mmap 上传 + 解码, 单流串行; 正确性优先, 不做引擎那套 4 lane 流水。 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "v41_fp_moe.h"
#include "../../src/common/ds4_fp8.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "★CUDA %s @%s:%d: %s★\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "★cuBLAS %s @%s:%d: %d★\n", #x, __FILE__, __LINE__, (int)s_); return -1; } } while (0)

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
static dbuf g_raw, g_wgu, g_wd, g_xs, g_ys, g_gu, g_h, g_perm, g_inv, g_rw, g_routed;
static dbuf g_w1, g_w3, g_w2, g_x16, g_sg, g_su, g_sh, g_h16, g_so;

__device__ __forceinline__ static float bf16r(float x) {   /* = 引擎 v41_bf16r: RNE 舍到 bf16 再回 f32 */
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

/* FP4 打包 [rows][cols/2] + ue8m0 [rows][cols/32] → f16 [rows][cols](= 引擎 v41_fp4x32_to_f16_kernel 的算式, 只换字节布局) */
__global__ static void k_fp4_to_f16(__half *out, const uint8_t *w, const uint8_t *s, int rows, int cols) {
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int packed = cols / 2;
    if (j >= (long long)rows * packed) return;
    const int r = (int)(j / packed), c0 = (int)(j % packed) * 2;
    const uint8_t q = w[j];
    const float sc = ds4_e8m0_to_f32(s[(long long)r * (cols / 32) + c0 / 32]);
    __half *o = out + (long long)r * cols + c0;
    o[0] = __float2half_rn(ds4_fp4_nibble_to_f32(q & 0x0F) * sc);
    o[1] = __float2half_rn(ds4_fp4_nibble_to_f32(q >> 4) * sc);
}
/* E4M3 [rows][cols] + ue8m0 块 [rows/sbr][cols/sbc] → f16(出厂 shared 专家; 448×2^e 全在 f16 内精确; 255 = NaN 槽当 0) */
__global__ static void k_fp8_to_f16(__half *out, const uint8_t *w, const uint8_t *s, int rows, int cols, int sbr, int sbc) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    const uint8_t sb = s[(long long)(r / sbr) * (cols / sbc) + c / sbc];
    const float sc = sb == 255 ? 0.f : ds4_e8m0_to_f32(sb);
    out[i] = __float2half_rn(ds4_e4m3fn_to_f32(w[i]) * sc);
}
/* 以下四核 = 引擎 routed prefill 路逐式抄写 */
__global__ static void k_gather(__half *xs, const float *x, const int *perm, int n_used, int IN) {
    const int i = blockIdx.x, t = perm[i] / n_used;
    const float *src = x + (long long)t * IN;
    __half *dst = xs + (long long)i * IN;
    for (int k = threadIdx.x; k < IN; k += blockDim.x) dst[k] = __float2half(src[k]);
}
__global__ static void k_swiglu_f16(__half *h, const float *gu, int MID, float clamp) {
    const int row = blockIdx.x;
    const float *g = gu + (long long)row * 2 * MID, *u = g + MID;
    __half *o = h + (long long)row * MID;
    for (int m = threadIdx.x; m < MID; m += blockDim.x) {
        float gv = g[m], uv = u[m];
        if (clamp > 0.0f) { if (gv > clamp) gv = clamp; if (uv > clamp) uv = clamp; if (uv < -clamp) uv = -clamp; }
        o[m] = __float2half((gv / (1.0f + __expf(-gv))) * uv);
    }
}
__global__ static void k_reduce(float *out, const float *ys, const int *inv, const float *rw, int n_used, int OUT) {
    const int t = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float s = 0.0f;
    for (int pk = 0; pk < n_used; pk++) {
        const int i = inv[(long long)t * n_used + pk];
        if (i < 0) continue;
        s += rw[(long long)t * n_used + pk] * ys[(long long)i * OUT + o];
    }
    out[(long long)t * OUT + o] = s;
}
/* 以下四核 = 引擎 shared 路(v41_moe 的 shared 段: fp4x32 matmul round_out=1 + v41_swiglu_kernel + add + round) */
__global__ static void k_x_to_f16(__half *o, const float *x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half_rn(bf16r(x[i]));
}
__global__ static void k_round_bf16(float *x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = bf16r(x[i]);
}
__global__ static void k_swiglu_bf16(float *h, const float *g, const float *u, long long n, float limit) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) { uv = fminf(fmaxf(uv, -limit), limit); gv = fminf(gv, limit); }
    h[i] = bf16r((gv / (1.0f + expf(-gv))) * uv);
}
__global__ static void k_add_round(float *y, const float *a, const float *b, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16r(a[i] + b[i]);
}

/* C[n][out] = X[n][in] · Wᵀ, f16 输入 f32 累加(= 引擎 vqp_gemm / v41_gemm_f16 的 cuBLAS 口径) */
static int gemm_f16(const __half *W, int in_dim, const __half *X, float *C, int out_dim, int n) {
    const float a = 1.f, b = 0.f;
    CB(cublasGemmEx(g_bl, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, n, in_dim, &a, W, CUDA_R_16F, in_dim, X, CUDA_R_16F, in_dim,
                    &b, C, CUDA_R_32F, out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    return 0;
}
/* 主机 HF 字节 → 设备 f16 [rows][cols] */
static int mat_to_f16(const v41_fp_mat *M, __half *dst) {
    const size_t wb = M->fp8 ? (size_t)M->rows * M->cols : (size_t)M->rows * M->cols / 2;
    const size_t sb = M->fp8 ? (size_t)(M->rows / M->sbr) * (M->cols / M->sbc) : (size_t)M->rows * (M->cols / 32);
    if (grow(&g_raw, wb + sb)) return -1;
    uint8_t *dw = (uint8_t *)g_raw.p, *ds = dw + wb;
    CK(cudaMemcpy(dw, M->w, wb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, M->s, sb, cudaMemcpyHostToDevice));
    if (M->fp8) k_fp8_to_f16<<<(unsigned)((wb + 255) / 256), 256>>>(dst, dw, ds, M->rows, M->cols, M->sbr, M->sbc);
    else k_fp4_to_f16<<<(unsigned)((wb + 255) / 256), 256>>>(dst, dw, ds, M->rows, M->cols);
    CK(cudaGetLastError());
    return 0;
}
static int shape_ok(const v41_fp_ffn *f, int D, int MID, const char *who) {
    if (f->w1.rows != MID || f->w1.cols != D || f->w3.rows != MID || f->w3.cols != D || f->w2.rows != D || f->w2.cols != MID) {
        fprintf(stderr, "★%s 形状不对: w1 %d×%d w3 %d×%d w2 %d×%d, 期 [%d×%d]/[%d×%d]★\n", who, f->w1.rows, f->w1.cols,
                f->w3.rows, f->w3.cols, f->w2.rows, f->w2.cols, MID, D, D, MID);
        return -1;
    }
    return 0;
}

extern "C" int v41_fp_moe_layer(const v41_fp_ffn *ex, int n_expert, const v41_fp_ffn *sh,
                                const float *dX, const int *hsel, const float *hrw, int n, int D, int MID, int n_used, float clamp,
                                float *dY) {
    if (!g_bl) CB(cublasCreate(&g_bl));
    if (shape_ok(sh, D, MID, "shared")) return -1;
    const long long npair = (long long)n * n_used, nD = (long long)n * D, nM = (long long)n * MID;
    /* (token,pick) 对按专家计数排序(= 引擎: 稳定, 同专家内保持 token 序; reduce 按 pick 序定序 ⇒ 可复现) */
    int *cnt = (int *)calloc((size_t)n_expert, sizeof(int)), *off = (int *)malloc(sizeof(int) * ((size_t)n_expert + 1));
    int *cur = (int *)malloc(sizeof(int) * (size_t)n_expert), *perm = (int *)malloc(sizeof(int) * (size_t)npair);
    int *inv = (int *)malloc(sizeof(int) * (size_t)npair);
    if (!cnt || !off || !cur || !perm || !inv) return -1;
    for (long long k = 0; k < npair; k++) { const int e = hsel[k]; if (e >= 0 && e < n_expert) cnt[e]++; }
    off[0] = 0; int ne_max = 0;
    for (int e = 0; e < n_expert; e++) { off[e + 1] = off[e] + cnt[e]; if (cnt[e] > ne_max) ne_max = cnt[e]; }
    const int nvalid = off[n_expert];
    memcpy(cur, off, sizeof(int) * (size_t)n_expert);
    for (long long k = 0; k < npair; k++) {
        const int e = hsel[k];
        if (e < 0 || e >= n_expert) { inv[k] = -1; continue; }
        const int pos = cur[e]++; perm[pos] = (int)k; inv[k] = pos;
    }
    int rc = -1;
    do {
        if (grow(&g_xs, (size_t)nvalid * D * 2) || grow(&g_ys, (size_t)nvalid * D * 4) || grow(&g_perm, (size_t)nvalid * 4) ||
            grow(&g_inv, (size_t)npair * 4) || grow(&g_rw, (size_t)npair * 4) || grow(&g_gu, (size_t)ne_max * 2 * MID * 4) ||
            grow(&g_h, (size_t)ne_max * MID * 2) || grow(&g_wgu, (size_t)2 * MID * D * 2) || grow(&g_wd, (size_t)D * MID * 2) ||
            grow(&g_routed, (size_t)nD * 4)) break;
        if (cudaMemcpy(g_perm.p, perm, (size_t)nvalid * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_inv.p, inv, (size_t)npair * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_rw.p, hrw, (size_t)npair * 4, cudaMemcpyHostToDevice) != cudaSuccess) { fprintf(stderr, "★路由上传失败★\n"); break; }
        __half *xs = (__half *)g_xs.p, *wgu = (__half *)g_wgu.p, *wd = (__half *)g_wd.p, *h = (__half *)g_h.p;
        float *ys = (float *)g_ys.p, *gu = (float *)g_gu.p;
        k_gather<<<nvalid, 256>>>(xs, dX, (const int *)g_perm.p, n_used, D);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "★gather 核失败★\n"); break; }
        int bad = 0;
        for (int e = 0; e < n_expert && !bad; e++) {
            const int ne = cnt[e];
            if (!ne) continue;
            char who[32]; snprintf(who, sizeof who, "expert %d", e);
            if (shape_ok(&ex[e], D, MID, who)) { bad = 1; break; }
            if (mat_to_f16(&ex[e].w1, wgu) || mat_to_f16(&ex[e].w3, wgu + (size_t)MID * D) || mat_to_f16(&ex[e].w2, wd)) { bad = 1; break; }
            if (gemm_f16(wgu, D, xs + (size_t)off[e] * D, gu, 2 * MID, ne)) { bad = 1; break; }
            k_swiglu_f16<<<ne, 256>>>(h, gu, MID, clamp);
            if (gemm_f16(wd, MID, h, ys + (size_t)off[e] * D, D, ne)) { bad = 1; break; }
        }
        if (bad) break;
        k_reduce<<<dim3((unsigned)((D + 255) / 256), (unsigned)n), 256>>>((float *)g_routed.p, ys, (const int *)g_inv.p, (const float *)g_rw.p, n_used, D);
        /* shared 专家(出厂 E4M3): 与引擎 v41_moe 的 shared 段逐式同 */
        if (grow(&g_w1, (size_t)MID * D * 2) || grow(&g_w3, (size_t)MID * D * 2) || grow(&g_w2, (size_t)D * MID * 2) ||
            grow(&g_x16, (size_t)nD * 2) || grow(&g_sg, (size_t)nM * 4) || grow(&g_su, (size_t)nM * 4) || grow(&g_sh, (size_t)nM * 4) ||
            grow(&g_h16, (size_t)nM * 2) || grow(&g_so, (size_t)nD * 4)) break;
        if (mat_to_f16(&sh->w1, (__half *)g_w1.p) || mat_to_f16(&sh->w3, (__half *)g_w3.p) || mat_to_f16(&sh->w2, (__half *)g_w2.p)) break;
        const unsigned bD = (unsigned)((nD + 255) / 256), bM = (unsigned)((nM + 255) / 256);
        k_x_to_f16<<<bD, 256>>>((__half *)g_x16.p, dX, nD);
        if (gemm_f16((const __half *)g_w1.p, D, (const __half *)g_x16.p, (float *)g_sg.p, MID, n)) break;
        k_round_bf16<<<bM, 256>>>((float *)g_sg.p, nM);
        if (gemm_f16((const __half *)g_w3.p, D, (const __half *)g_x16.p, (float *)g_su.p, MID, n)) break;
        k_round_bf16<<<bM, 256>>>((float *)g_su.p, nM);
        k_swiglu_bf16<<<bM, 256>>>((float *)g_sh.p, (const float *)g_sg.p, (const float *)g_su.p, nM, clamp);
        k_x_to_f16<<<bM, 256>>>((__half *)g_h16.p, (const float *)g_sh.p, nM);
        if (gemm_f16((const __half *)g_w2.p, MID, (const __half *)g_h16.p, (float *)g_so.p, D, n)) break;
        k_round_bf16<<<bD, 256>>>((float *)g_so.p, nD);
        k_add_round<<<bD, 256>>>(dY, (const float *)g_routed.p, (const float *)g_so.p, nD);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "★shared 段核失败★\n"); break; }
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "★FP MoE 同步失败: %s★\n", cudaGetErrorString(cudaGetLastError())); break; }
        rc = 0;
    } while (0);
    free(cnt); free(off); free(cur); free(perm); free(inv);
    return rc;
}
