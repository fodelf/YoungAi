/* v41_kl_target.cu — 蒸馏靶: 把"让最终 logits 更像教师"翻译成"末层 MoE 输出该往哪挪"(2026-09-13)。
 * 设计与三处近似写在 v41_kl_target.h 头注释里, 这里只讲实现。
 *
 * 【算什么】R[i][d] = −η · α[i] · g_onorm[d] · Σ_v (p_s[i][v] − p_t[i][v]) · W_head[v][d]
 * 【怎么算】两份 logits 文件按 512 行一块读进来, 主机算 softmax 差(f32) → f16 上设备 → 一发 cublasGemmEx
 * 乘出口头权重。为什么分块: logits 是 [8192][129280] f32 = 4.2 GB 一份, 两份整读会把 spark 的余量吃光。
 * 【为什么 head 权重要整份常驻】1.32 GB(f16), 16 块都要乘它一遍, 每块重解一次就是 16 遍 dequant。
 * 【η 不拍脑袋】按 --eta-rel r 缩放, 使 ‖R‖ = r·‖y_q‖ —— 与现役放大器的 ‖Δ‖/‖y_q‖(中位 0.23)同一把尺,
 * 所以 r 可以直接和它比大小。
 * 编译: nvcc -O3 -fmad=false(与解算器同旗标, 同输入两路必须出同一个数)。 */
#include "v41_kl_target.h"
/* 这两个头是纯 C 且没有 extern "C" 守卫, 而 .cu 是 C++ TU —— 不包住就是 C++ 修饰名, 链接找不到符号 */
extern "C" {
#include "../../src/common/ds4_gguf.h"
#include "../../src/common/ds4_quantfmt.h"
}
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define KLT_BLK 512   /* 每块行数: 132 MB 的 f16 dp 缓冲, 一发 GEMM 就能压满 GB10 */

struct v41_klt {
    ds4_gguf g;
    __half *dW;        /* [V][D] 行主序 = 列主序 [D][V] */
    float *dOn;        /* [D] output_norm 权重 */
    __half *dDP;       /* [KLT_BLK][V] */
    float *dG;         /* [KLT_BLK][D] */
    float *hOn;
    int V, D;
    cublasHandle_t cb;
};

#define CKC(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)

/* fp4x32: 17 B/32 元素(ds4_quantfmt 的 type 43)。整份 [V][D] 在主机解到 f16 再一次上设备 ——
 * 逐行上传 129280 次小拷贝比一次 1.3 GB 慢得多。 */
static int klt_load_head(v41_klt *k, const ds4_gguf_tensor *t) {
    const size_t V = (size_t)k->V, D = (size_t)k->D, nb = D / 32;
    uint64_t nby = 0;
    const uint8_t *src = ds4_gguf_tensor_data(&k->g, t, &nby);
    if (!src || nby != (uint64_t)V * nb * 17) { fprintf(stderr, "★output.weight 数据取不到或字节数 %llu ≠ %llu★\n", (unsigned long long)nby, (unsigned long long)((uint64_t)V * nb * 17)); return -1; }
    __half *h = (__half *)malloc(V * D * sizeof(__half));
    float *row = (float *)malloc(D * 4);
    if (!h || !row) { fprintf(stderr, "★head 主机缓冲 %.1f GB 分配失败★\n", V * D * 2.0 / 1e9); return -1; }
    for (size_t v = 0; v < V; v++) {
        ds4_deq_fp4x32(src + v * nb * 17, nb, row);
        for (size_t d = 0; d < D; d++) h[v * D + d] = __float2half(row[d]);
    }
    free(row);
    if (cudaMalloc((void **)&k->dW, V * D * sizeof(__half)) != cudaSuccess) { free(h); fprintf(stderr, "★head 设备 %.1f GB 分配失败★\n", V * D * 2.0 / 1e9); return -1; }
    const cudaError_t e = cudaMemcpy(k->dW, h, V * D * sizeof(__half), cudaMemcpyHostToDevice);
    free(h);
    if (e != cudaSuccess) { fprintf(stderr, "★head 上传失败: %s★\n", cudaGetErrorString(e)); return -1; }
    return 0;
}

int v41_klt_open(v41_klt **out, const char *gguf_path, int D) {
    v41_klt *k = (v41_klt *)calloc(1, sizeof *k);
    if (!k) return -1;
    char err[256];
    if (ds4_gguf_open(&k->g, gguf_path, err, sizeof err)) { fprintf(stderr, "★打不开 %s: %s★\n", gguf_path, err); free(k); return -1; }
    const ds4_gguf_tensor *W = ds4_gguf_find(&k->g, "output.weight"), *N = ds4_gguf_find(&k->g, "output_norm.weight");
    if (!W || !N) { fprintf(stderr, "★GGUF 缺 output.weight / output_norm.weight★\n"); return -1; }
    if (W->nd != 2 || (int)W->ne[0] != D || W->type != DS4_GGT_FP4X32) {
        fprintf(stderr, "★output.weight 形状/类型不对: nd %u ne0 %llu type %u(期 D=%d type=%d)★\n",
                W->nd, (unsigned long long)W->ne[0], W->type, D, (int)DS4_GGT_FP4X32);
        return -1;
    }
    if (N->type != DS4_GGT_F32 || (int)N->ne[0] != D) { fprintf(stderr, "★output_norm.weight 不是 f32[%d]★\n", D); return -1; }
    k->D = D; k->V = (int)W->ne[1];
    if (klt_load_head(k, W)) return -1;
    uint64_t nnb = 0;
    const uint8_t *nsrc = ds4_gguf_tensor_data(&k->g, N, &nnb);
    if (!nsrc || nnb != (uint64_t)D * 4) { fprintf(stderr, "★output_norm 数据取不到(字节 %llu)★\n", (unsigned long long)nnb); return -1; }
    k->hOn = (float *)malloc((size_t)D * 4);
    memcpy(k->hOn, nsrc, (size_t)D * 4);
    if (cudaMalloc((void **)&k->dOn, (size_t)D * 4) != cudaSuccess ||
        cudaMemcpy(k->dOn, k->hOn, (size_t)D * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMalloc((void **)&k->dDP, (size_t)KLT_BLK * k->V * sizeof(__half)) != cudaSuccess ||
        cudaMalloc((void **)&k->dG, (size_t)KLT_BLK * D * 4) != cudaSuccess) { fprintf(stderr, "★KL 靶设备缓冲分配失败★\n"); return -1; }
    if (cublasCreate(&k->cb) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "★cublas 建不了★\n"); return -1; }
    printf("[KL 靶] 出口头 %d×%d(fp4x32→f16 %.2f GB 常驻) + output_norm f32[%d]\n", k->V, D, (double)k->V * D * 2 / 1e9, D);
    *out = k;
    return 0;
}

/* 一块行的 softmax 差(主机, f32→f16): dp = softmax(学生) − softmax(教师)。
 * 逐行减最大值再 exp —— logits 的量级到 ±30, 不减会溢出。 */
static void klt_block_dp(const float *stu, const float *ref, __half *dp, int rows, int V) {
    for (int i = 0; i < rows; i++) {
        const float *s = stu + (size_t)i * V, *r = ref + (size_t)i * V;
        __half *o = dp + (size_t)i * V;
        float ms = -INFINITY, mr = -INFINITY;
        for (int v = 0; v < V; v++) { if (s[v] > ms) ms = s[v]; if (r[v] > mr) mr = r[v]; }
        double zs = 0.0, zr = 0.0;
        for (int v = 0; v < V; v++) { zs += exp((double)s[v] - ms); zr += exp((double)r[v] - mr); }
        const double is = 1.0 / zs, ir = 1.0 / zr;
        for (int v = 0; v < V; v++) o[v] = __float2half((float)(exp((double)s[v] - ms) * is - exp((double)r[v] - mr) * ir));
    }
}

/* G[i][d] *= onorm[d] —— RMSNorm 雅可比主项里唯一逐通道的那一项(rms 是逐 token 正标量, 被步长吸收)。 */
__global__ static void klt_scale_on(float *G, const float *on, int rows, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d < D) G[(size_t)i * D + d] *= on[d];
}
/* R[i][d] = s · α[r] · G[r][d], r = perm[i] —— ★行序在这里换★: 梯度按 logits 的原行序算出来, 而解算器吃的是
 * "拟合行在前、val 行在后"的置换序。两者错位不会报错, 只会让靶配错行, 所以置换只在这一处做, 别处不许再搬。
 * (R 与 G 是同一块缓冲, 置换是乱序读写 ⇒ 必须先拷一份 G, 见调用处。) */
__global__ static void klt_finish(float *R, const float *G, const float *alpha, const int *perm, float s, int n, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d >= D) return;
    const int r = perm ? perm[i] : i;
    R[(size_t)i * D + d] = s * alpha[r] * G[(size_t)r * D + d];
}
/* 出口: dYfp = dYq + R —— 解算器拿 dYfp−dYq 当靶, 于是靶就是 R */
__global__ static void klt_add_yq(float *R, const float *Yq, int n, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d < D) R[(size_t)i * D + d] += Yq[(size_t)i * D + d];
}

/* 打开一份 logits 文件并校验头; 返回文件指针(定位在数据起点), 失败 NULL。 */
static FILE *klt_open_logits(const char *path, int n, int V) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "★打不开 logits %s★\n", path); return NULL; }
    int32_t hd[2];
    if (fread(hd, 4, 2, f) != 2 || hd[0] < n || hd[1] != V) {
        fprintf(stderr, "★%s 头 S=%d V=%d 与 n=%d V=%d 不符★\n", path, hd[0], hd[1], n, V);
        fclose(f); return NULL;
    }
    return f;
}

int v41_klt_target(v41_klt *k, const char *ref_path, const char *stu_path, const float *alpha, const int *perm,
                   int n, int D, double qnorm, float eta_rel, const float *dYq, float *dR, float *out_stat) {
    if (D != k->D) { fprintf(stderr, "★KL 靶 D %d ≠ %d★\n", D, k->D); return -1; }
    const int V = k->V;
    FILE *fr = klt_open_logits(ref_path, n, V), *fs = klt_open_logits(stu_path, n, V);
    if (!fr || !fs) { if (fr) fclose(fr); if (fs) fclose(fs); return -1; }
    float *hb = (float *)malloc((size_t)KLT_BLK * V * 4), *hb2 = (float *)malloc((size_t)KLT_BLK * V * 4);
    __half *hdp = (__half *)malloc((size_t)KLT_BLK * V * sizeof(__half));
    float *dAl = NULL, *dGall = NULL; int *dPerm = NULL;
    if (!hb || !hb2 || !hdp) { fprintf(stderr, "★KL 靶主机缓冲失败★\n"); return -1; }
    CKC(cudaMalloc((void **)&dAl, (size_t)n * 4));
    CKC(cudaMemcpy(dAl, alpha, (size_t)n * 4, cudaMemcpyHostToDevice));
    if (perm) { CKC(cudaMalloc((void **)&dPerm, (size_t)n * 4)); CKC(cudaMemcpy(dPerm, perm, (size_t)n * 4, cudaMemcpyHostToDevice)); }
    CKC(cudaMalloc((void **)&dGall, (size_t)n * D * 4));   /* 梯度按原行序单独存: 置换是乱序读, 不能与输出同缓冲 */
    const float one = 1.f, zero = 0.f;
    for (int off = 0; off < n; off += KLT_BLK) {
        const int rows = (n - off < KLT_BLK) ? n - off : KLT_BLK;
        if (fread(hb, 4, (size_t)rows * V, fs) != (size_t)rows * V || fread(hb2, 4, (size_t)rows * V, fr) != (size_t)rows * V) {
            fprintf(stderr, "★logits 读到第 %d 行截断★\n", off); return -1;
        }
        klt_block_dp(hb, hb2, hdp, rows, V);
        CKC(cudaMemcpy(k->dDP, hdp, (size_t)rows * V * sizeof(__half), cudaMemcpyHostToDevice));
        /* 列主序: G_cm[D][rows] = W_cm[D][V] × dp_cm[V][rows](两边都无转置) */
        if (cublasGemmEx(k->cb, CUBLAS_OP_N, CUBLAS_OP_N, D, rows, V, &one,
                         k->dW, CUDA_R_16F, D, k->dDP, CUDA_R_16F, V, &zero,
                         k->dG, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "★出口头 GEMM 失败(off %d)★\n", off); return -1;
        }
        klt_scale_on<<<dim3((D + 255) / 256, rows), 256>>>(k->dG, k->dOn, rows, D);
        CKC(cudaMemcpy(dGall + (size_t)off * D, k->dG, (size_t)rows * D * 4, cudaMemcpyDeviceToDevice));
    }
    fclose(fr); fclose(fs);
    free(hb); free(hb2); free(hdp);
    /* ‖g‖ → 缩放到 ‖R‖ = eta_rel·‖y_q‖; α 的极值一并报(它若跨零, 靶方向在那些行上是反的, 必须看见) */
    float gn = 0.f;
    if ((long long)n * D > 2147483647LL) { fprintf(stderr, "★n×D 超 int(cublas 接口上限)★\n"); return -1; }
    if (cublasSnrm2(k->cb, n * D, dGall, 1, &gn) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "★‖g‖ 失败★\n"); return -1; }
    double amin = alpha[0], amax = alpha[0], asum = 0.0;
    for (int i = 0; i < n; i++) { if (alpha[i] < amin) amin = alpha[i]; if (alpha[i] > amax) amax = alpha[i]; asum += alpha[i]; }
    /* α 已经乘进最终核, 所以缩放系数要按 α 加权后的范数定 —— 先用 α 的均方根近似, 再在核后精确重标一次 */
    double arms = 0.0;
    for (int i = 0; i < n; i++) arms += (double)alpha[i] * alpha[i];
    arms = sqrt(arms / n);
    const double gan = (double)gn * arms;
    if (!(gan > 0.0)) { fprintf(stderr, "★梯度范数为 0, 靶无意义★\n"); return -1; }
    const float s = (float)(-(double)eta_rel * qnorm / gan);   /* 负号 = 沿 −∇ 走 */
    klt_finish<<<dim3((D + 255) / 256, n), 256>>>(dR, dGall, dAl, dPerm, s, n, D);
    float rn = 0.f;
    if (cublasSnrm2(k->cb, n * D, dR, 1, &rn) != CUBLAS_STATUS_SUCCESS) return -1;
    if (rn > 0.f) {   /* 精确重标: arms 只是近似, 这一步把 ‖R‖ 钉到 eta_rel·‖y_q‖ */
        const float fix = (float)(eta_rel * qnorm / rn);
        if (cublasSscal(k->cb, n * D, &fix, dR, 1) != CUBLAS_STATUS_SUCCESS) return -1;
    }
    klt_add_yq<<<dim3((D + 255) / 256, n), 256>>>(dR, dYq, n, D);
    cudaFree(dAl); cudaFree(dGall); if (dPerm) cudaFree(dPerm);
    if (out_stat) { out_stat[0] = (float)(gan / (qnorm > 0 ? qnorm : 1.0)); out_stat[1] = (float)amin; out_stat[2] = (float)amax; out_stat[3] = (float)(asum / n); }
    return (int)cudaDeviceSynchronize();
}

void v41_klt_close(v41_klt *k) {
    if (!k) return;
    if (k->dW) cudaFree(k->dW);
    if (k->dOn) cudaFree(k->dOn);
    if (k->dDP) cudaFree(k->dDP);
    if (k->dG) cudaFree(k->dG);
    free(k->hOn);
    if (k->cb) cublasDestroy(k->cb);
    ds4_gguf_close(&k->g);
    free(k);
}
