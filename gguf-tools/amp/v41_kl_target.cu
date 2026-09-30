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

/* 出口头按类型反量化(走 src/common 的 ds4_deq_bytes, 不在工具侧另抄一份): 09-22 现役换 v3 后 output.weight 是
 * q4_K(type 12), 之前只认 fp4x32(type 43), 后训练解算在出口初始化就停车(09-23 实撞)。行跨度 = 总字节 / 行数,
 * 除不尽 = 不是按行连续存的格式, 直接拒。整份 [V][D] 在主机解到 f16 再一次上设备 ——
 * 逐行上传 129280 次小拷贝比一次 1.3 GB 慢得多。 */
static int klt_load_head(v41_klt *k, const ds4_gguf_tensor *t) {
    const size_t V = (size_t)k->V, D = (size_t)k->D;
    uint64_t nby = 0;
    const uint8_t *src = ds4_gguf_tensor_data(&k->g, t, &nby);
    if (!src || nby == 0 || nby % V) { fprintf(stderr, "★output.weight 数据取不到或字节数 %llu 不能按 %zu 行均分★\n", (unsigned long long)nby, V); return -1; }
    const size_t rb = (size_t)(nby / V);
    __half *h = (__half *)malloc(V * D * sizeof(__half));
    float *row = (float *)malloc(D * 4);
    if (!h || !row) { fprintf(stderr, "★head 主机缓冲 %.1f GB 分配失败★\n", V * D * 2.0 / 1e9); return -1; }
    for (size_t v = 0; v < V; v++) {
        if (ds4_deq_bytes(t->type, src + v * rb, D, row)) { fprintf(stderr, "★output.weight type %u 反量化不支持★\n", t->type); free(row); free(h); return -1; }
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
    if (W->nd != 2 || (int)W->ne[0] != D) {
        fprintf(stderr, "★output.weight 形状不对: nd %u ne0 %llu(期 D=%d)★\n",
                W->nd, (unsigned long long)W->ne[0], D);
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
    printf("[KL 靶] 出口头 %d×%d(type %u→f16 %.2f GB 常驻) + output_norm f32[%d]\n", k->V, D, W->type, (double)k->V * D * 2 / 1e9, D);
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

/* 填一块 dp 的回调: 蒸馏靶读两份 logits, SFT 靶读 top-K 表 —— 只有这一步不同, 后面
 * (出口头 GEMM → output_norm → α → 置换 → 缩放到 eta_rel·‖y_q‖ → 加 y_q)两条路一个字不差,
 * 所以主循环只有一份。返回非 0 = 这一块取不到料, 停车。 */
typedef int (*klt_fill_fn)(void *ud, __half *dp, int off, int rows, int V);

static int klt_core(v41_klt *k, klt_fill_fn fill, void *ud, const float *alpha, const int *perm,
                    int n, int D, double qnorm, float eta_rel, const float *dYq, float *dR, float *out_stat) {
    const int V = k->V;
    __half *hdp = (__half *)malloc((size_t)KLT_BLK * V * sizeof(__half));
    float *dAl = NULL, *dGall = NULL; int *dPerm = NULL;
    if (!hdp) { fprintf(stderr, "★KL 靶主机缓冲失败★\n"); return -1; }
    CKC(cudaMalloc((void **)&dAl, (size_t)n * 4));
    CKC(cudaMemcpy(dAl, alpha, (size_t)n * 4, cudaMemcpyHostToDevice));
    if (perm) { CKC(cudaMalloc((void **)&dPerm, (size_t)n * 4)); CKC(cudaMemcpy(dPerm, perm, (size_t)n * 4, cudaMemcpyHostToDevice)); }
    CKC(cudaMalloc((void **)&dGall, (size_t)n * D * 4));   /* 梯度按原行序单独存: 置换是乱序读, 不能与输出同缓冲 */
    const float one = 1.f, zero = 0.f;
    for (int off = 0; off < n; off += KLT_BLK) {
        const int rows = (n - off < KLT_BLK) ? n - off : KLT_BLK;
        if (fill(ud, hdp, off, rows, V)) return -1;
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
    free(hdp);
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

/* ---- 蒸馏靶(教师−学生两份 logits) ---- */
typedef struct { FILE *fs, *fr; float *hb, *hb2; } kl_ud;
static int kl_fill(void *ud, __half *dp, int off, int rows, int V) {
    kl_ud *u = (kl_ud *)ud;
    if (fread(u->hb, 4, (size_t)rows * V, u->fs) != (size_t)rows * V ||
        fread(u->hb2, 4, (size_t)rows * V, u->fr) != (size_t)rows * V) {
        fprintf(stderr, "★logits 读到第 %d 行截断★\n", off); return -1;
    }
    klt_block_dp(u->hb, u->hb2, dp, rows, V);
    return 0;
}

int v41_klt_target(v41_klt *k, const char *ref_path, const char *stu_path, const float *alpha, const int *perm,
                   int n, int D, double qnorm, float eta_rel, const float *dYq, float *dR, float *out_stat) {
    if (D != k->D) { fprintf(stderr, "★KL 靶 D %d ≠ %d★\n", D, k->D); return -1; }
    kl_ud u;
    u.fs = klt_open_logits(stu_path, n, k->V);
    u.fr = klt_open_logits(ref_path, n, k->V);
    u.hb = (float *)malloc((size_t)KLT_BLK * k->V * 4);
    u.hb2 = (float *)malloc((size_t)KLT_BLK * k->V * 4);
    int rc = -1;
    if (u.fs && u.fr && u.hb && u.hb2)
        rc = klt_core(k, kl_fill, &u, alpha, perm, n, D, qnorm, eta_rel, dYq, dR, out_stat);
    if (u.fs) fclose(u.fs);
    if (u.fr) fclose(u.fr);
    free(u.hb); free(u.hb2);
    return rc;
}

/* ★第一版的 SFT 梯度靶(dp = p − onehot + 成对项)已删★(09-13 夜): 实测在 y 空间只挽回 0.20%,
 * 定罪见 v41_sft_run.inc.c 头注释。留着它只会让人以为还有第二条路可选。后训练现在只有决策差一条。 */

/* ---- 决策差方向表(后训练第二版) ---- */
/* C[i][d] = sc_i · γ[d] · (W[a_i][d] − W[b_i][d]), sc_i = α_i·inv_i。
 * 一行一个 blockIdx.y; W 是 f16 行主序 [V][D], 两行都是顺序读 ⇒ 合并访存。 */
__global__ static void klt_dirs_kernel(float *C, const __half *W, const float *on, const int *a, const int *b,
                                       const float *sc, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d >= D) return;
    const int ai = a[i], bi = b[i];
    float v = 0.f;
    if (ai >= 0 && bi >= 0 && ai != bi)
        v = sc[i] * on[d] * (__half2float(W[(size_t)ai * D + d]) - __half2float(W[(size_t)bi * D + d]));
    C[(size_t)i * D + d] = v;
}

int v41_klt_margin_dirs(v41_klt *k, const int *a, const int *b, const float *alpha, const float *inv,
                        int n, int D, float *dC) {
    if (D != k->D) { fprintf(stderr, "★方向表 D %d ≠ %d★\n", D, k->D); return -1; }
    float *hsc = (float *)malloc((size_t)n * 4);
    int *hab = (int *)malloc((size_t)n * 8);
    if (!hsc || !hab) { fprintf(stderr, "★方向表主机缓冲失败★\n"); free(hsc); free(hab); return -1; }
    int nbad = 0;
    for (int i = 0; i < n; i++) {
        /* inv 必须是正的: 它是 rsqrt 的结果。是 0 或负 = 读错了文件(行号错位/文件是别的趟的),
         * 那样解出来的方向表整行是零, 解算照样跑得出一个数 —— 所以这里数出来报给人看。 */
        if (!(inv[i] > 0.f)) { nbad++; hsc[i] = 0.f; }
        else hsc[i] = alpha[i] * inv[i];
        if (a[i] < 0 || b[i] < 0 || a[i] >= k->V || b[i] >= k->V) { hab[i] = -1; hab[n + i] = -1; }
        else { hab[i] = a[i]; hab[n + i] = b[i]; }
    }
    if (nbad) fprintf(stderr, "★方向表: %d/%d 行的 inv 不是正数(整行按 0 处理)★\n", nbad, n);
    float *dsc = NULL; int *dab = NULL;
    int rc = -1;
    if (cudaMalloc((void **)&dsc, (size_t)n * 4) == cudaSuccess &&
        cudaMalloc((void **)&dab, (size_t)n * 8) == cudaSuccess &&
        cudaMemcpy(dsc, hsc, (size_t)n * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dab, hab, (size_t)n * 8, cudaMemcpyHostToDevice) == cudaSuccess) {
        klt_dirs_kernel<<<dim3((D + 255) / 256, n), 256>>>(dC, k->dW, k->dOn, dab, dab + n, dsc, D);
        rc = (int)cudaDeviceSynchronize();
        if (rc) fprintf(stderr, "★方向表核失败: %s★\n", cudaGetErrorString((cudaError_t)rc));
    } else fprintf(stderr, "★方向表设备缓冲/上传失败★\n");
    free(hsc); free(hab);
    if (dsc) cudaFree(dsc);
    if (dab) cudaFree(dab);
    return rc;
}

/* ★第七版的方向表(2026-09-29, back.md §14.4)★: 自 token 减该位期望 ——
 *     C[i][d] = sc_i · γ[d] · (W[y_i][d] − Σ_q p_iq · W[ids_iq][d]),  sc_i = α_i·inv_i
 * 它就是 ∂log p(y_i)/∂(本层输出) 穿过出口 norm→head 的精确线性化(softmax 的梯度 = onehot − p, p 只取榜上
 * K 项, 榜外尾巴 1−mass 丢掉; K=64 实测覆盖 97.3%)。解算器拿它做行方向, 右端给 η·优势, 于是
 * (Σ c cᵀ + 岭) Δ = η Σ A·c 正是"Fisher 预条件的策略梯度一步"。一行一个 blockIdx.y, 线程管 d, 榜上 K 行 W 顺序读。 */
__global__ static void klt_owndirs_kernel(float *C, const __half *W, const float *on, const int *a, const int *ids,
                                          const float *ps, const float *sc, int K, int D, int V) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d >= D) return;
    const int ai = a[i];
    float v = 0.f;
    if (ai >= 0 && ai < V && sc[i] != 0.f) {
        float ex = 0.f;
        for (int q = 0; q < K; q++) {
            const int t = ids[(size_t)i * K + q];
            const float p = ps[(size_t)i * K + q];
            if (t < 0 || t >= V || !(p > 0.f)) continue;
            ex += p * __half2float(W[(size_t)t * D + d]);
        }
        v = sc[i] * on[d] * (__half2float(W[(size_t)ai * D + d]) - ex);
    }
    C[(size_t)i * D + d] = v;
}

int v41_klt_own_dirs(v41_klt *k, const int *a, const int *dIds, const float *dPs, const float *alpha, const float *inv,
                     int n, int K, int D, float *dC) {
    if (D != k->D) { fprintf(stderr, "★自 token 方向表 D %d ≠ %d★\n", D, k->D); return -1; }
    float *hsc = (float *)malloc((size_t)n * 4); int *ha = (int *)malloc((size_t)n * 4);
    if (!hsc || !ha) { free(hsc); free(ha); return -1; }
    int nbad = 0;
    for (int i = 0; i < n; i++) {
        if (!(inv[i] > 0.f)) { nbad++; hsc[i] = 0.f; } else hsc[i] = alpha[i] * inv[i];
        ha[i] = (a[i] >= 0 && a[i] < k->V) ? a[i] : -1;
    }
    if (nbad) fprintf(stderr, "★自 token 方向表: %d/%d 行的 inv 不是正数(整行按 0 处理)★\n", nbad, n);
    float *dsc = NULL; int *da = NULL; int rc = -1;
    if (cudaMalloc((void **)&dsc, (size_t)n * 4) == cudaSuccess && cudaMalloc((void **)&da, (size_t)n * 4) == cudaSuccess &&
        cudaMemcpy(dsc, hsc, (size_t)n * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(da, ha, (size_t)n * 4, cudaMemcpyHostToDevice) == cudaSuccess) {
        klt_owndirs_kernel<<<dim3((D + 255) / 256, n), 256>>>(dC, k->dW, k->dOn, da, dIds, dPs, dsc, K, D, k->V);
        rc = (int)cudaDeviceSynchronize();
        if (rc) fprintf(stderr, "★自 token 方向表核失败: %s★\n", cudaGetErrorString((cudaError_t)rc));
    } else fprintf(stderr, "★自 token 方向表缓冲/上传失败★\n");
    free(hsc); free(ha);
    if (dsc) cudaFree(dsc);
    if (da) cudaFree(da);
    return rc;
}

/* ★主动集扫榜★(2026-09-13 夜第三针): 修正把 Δy 挪出去之后, 榜上【每个】token 的 logit 各动了多少。
 *   Δℓ[i][q] = α_i·inv_i · Σ_d γ[d]·W[ids[i][q]][d] · Δy[i][d]
 * 为什么必须扫整张榜: 每行只钉一对时, 钉住的那一对分毫不差(自检 2 相关 1.0000), 而 37% 的位置
 * argmax 还是变了 —— 冒头的是第三个 token。有了这张表, 主动集就能在【解算器内部】找出冒头的是谁,
 * 不用为此跑一趟真前向(一趟 3 分钟, 而这个核是毫秒级)。
 * 一行一个 block, 每个线程管榜上一格; W 行是顺序读。 */
__global__ static void klt_scan_kernel(float *dl, const __half *W, const float *on, const float *dY,
                                       const int *ids, const float *sc, int K, int D, int V) {
    const int i = blockIdx.x, q = threadIdx.x;
    if (q >= K) return;
    const int v = ids[(size_t)i * K + q];
    /* ★v 必须在词表内★: 榜是主机侧拼的, 拼错了(行错位、槽没写过)在这里就是越界读 1.3 GB 的
     * 出口头 —— 实撞过一次(09-13 22:29, 压实行时漏搬了榜)。按 0 处理并让上面的自检去发现, 
     * 好过让整个 CUDA 上下文报废。 */
    if (v < 0 || v >= V) { dl[(size_t)i * K + q] = 0.f; return; }
    const __half *wv = W + (size_t)v * D;
    const float *y = dY + (size_t)i * D;
    double acc = 0.0;
    for (int d = 0; d < D; d++) acc += (double)on[d] * __half2float(wv[d]) * y[d];
    dl[(size_t)i * K + q] = (float)(acc * sc[i]);
}

int v41_klt_margin_scan(v41_klt *k, const float *dY, const float *alpha, const float *inv,
                        const int *dIds, int n, int K, int D, float *dLogit) {
    if (D != k->D) { fprintf(stderr, "★扫榜 D %d ≠ %d★\n", D, k->D); return -1; }
    if (K > 1024) { fprintf(stderr, "★扫榜 K=%d > 1024(一个 block 一行)★\n", K); return -1; }
    float *hsc = (float *)malloc((size_t)n * 4), *dsc = NULL;
    if (!hsc) return -1;
    for (int i = 0; i < n; i++) hsc[i] = (inv[i] > 0.f) ? alpha[i] * inv[i] : 0.f;
    int rc = -1;
    if (cudaMalloc((void **)&dsc, (size_t)n * 4) == cudaSuccess &&
        cudaMemcpy(dsc, hsc, (size_t)n * 4, cudaMemcpyHostToDevice) == cudaSuccess) {
        klt_scan_kernel<<<n, ((K + 31) / 32) * 32>>>(dLogit, k->dW, k->dOn, dY, dIds, dsc, K, D, k->V);
        rc = (int)cudaDeviceSynchronize();
        if (rc) fprintf(stderr, "★扫榜核失败: %s★\n", cudaGetErrorString((cudaError_t)rc));
    } else fprintf(stderr, "★扫榜缓冲失败★\n");
    free(hsc);
    if (dsc) cudaFree(dsc);
    return rc;
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
