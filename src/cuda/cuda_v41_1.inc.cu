/* cuda_v41_1.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 批前向原语 ①(2026-09-12 战役 P2)。
 * 稠密 fp4x32 GEMM / 分组投影 / 嵌入 / bf16 舍入 / hyper-connection 三件 / RMSNorm / 加 / 展开。
 * 契约见 ds4_gpu_v41.h。口径: f32 计算, 官方 bf16 模块边界处显式舍 bf16。
 * 暂存: 自管 grow-only 槽(cuda_tmp_alloc 全局只有一块, 同一发里要两块就撞)。 */
#include "src/common/ds4_fp8.h"

typedef struct { void *p; uint64_t cap; } v41_scratch;
static v41_scratch g_v41_w16, g_v41_x16, g_v41_w16b, g_v41_misc;
/* 解码小批(n ≤ 8)走 cuda_v41_4.inc.cu 的融合核(不落 f16); 原型先声明, 定义在后面的分片(同一 TU) */
#define V41_GEMV_MAX_TOK 8u
static int v41_fp4x32_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                           const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                           uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what);
static int v41_vq_fused_moe(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                            const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                            const float *gr);
/* cuBLAS 句柄的流: 我们的核都发在 PTDS(编译 -default-stream per-thread 下的"流 0"), 但 cuBLAS 库不是按这个
 * 开关编的, 递给它的 0 是 legacy 流 —— 两者的隐式同步靠文档一句话, 实测整批路偶发脏读(同一列全 token 偏/NaN), 与 cuBLAS
 * 参与的路一一对应。显式递 cudaStreamPerThread, 让 cuBLAS 与我们的核同一条流, 不赌隐式同步。 */
static inline cudaStream_t v41_cublas_stream(void) { return g_cur_stream ? g_cur_stream : cudaStreamPerThread; }
static void *v41_grow(v41_scratch *s, uint64_t bytes, const char *what) {
    if (bytes <= s->cap) return s->p;
    (void)cudaDeviceSynchronize();
    if (s->p) (void)cudaFree(s->p);
    s->p = NULL; s->cap = 0;
    if (cudaMalloc(&s->p, (size_t)bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] %s 暂存分配失败 (%.1f MB)\n", what, (double)bytes / 1048576.0);
        return NULL;
    }
    s->cap = bytes;
    return s->p;
}

__device__ __forceinline__ static float v41_bf16r(float x) {   /* RNE 舍到 bf16 再回 f32 */
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;            /* NaN/Inf 原样 */
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

/* ---- fp4x32 → f16: 一线程一块(32 元素 17 B) ---- */
__global__ static void v41_fp4x32_to_f16_kernel(__half *out, const uint8_t *w, uint64_t nblk) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const uint8_t *p = w + b * 17u;
    const float s = ds4_e8m0_to_f32(p[16]);
    __half *o = out + b * 32u;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        o[2 * j]     = __float2half_rn(ds4_fp4_nibble_to_f32(p[j] & 0x0F) * s);
        o[2 * j + 1] = __float2half_rn(ds4_fp4_nibble_to_f32(p[j] >> 4) * s);
    }
}
/* 激活 f32 → bf16 格点 → f16(bf16 值在 f16 范围内精确可表示; 模拟官方 bf16 激活 × 精确权重的 f32 GEMM) */
__global__ static void v41_x_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(v41_bf16r(x[i]));
}
__global__ static void v41_round_bf16_kernel(float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v41_bf16r(x[i]);
}
int ds4_gpu_v41_round_bf16_tensor(ds4_gpu_tensor *x, uint64_t n) {
    if (!x || x->bytes < n * 4) return 0;
    v41_round_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)x->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 round bf16");
}

/* C[n][out] = X[n][in] · Wᵀ, 半精度输入 f32 累加(与 vqp_gemm 同口径) */
static int v41_gemm_f16(const __half *W, int in_dim, const __half *X, float *C, int out_dim, int n, const char *what) {
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());   /* 别的路会把句柄挂到侧流; V4.1 全链一条流, 每次显式钉回 */
    cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, n, in_dim, &alpha,
                                     W, CUDA_R_16F, in_dim, X, CUDA_R_16F, in_dim, &beta,
                                     C, CUDA_R_32F, out_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, what);
}

static int v41_dequant_fp4x32(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                              v41_scratch *slot, __half **w16_out, const char *what) {
    if ((in_dim % 32u) != 0u) { fprintf(stderr, "ds4: [v41] %s in_dim %llu 非 32 倍\n", what, (unsigned long long)in_dim); return 0; }
    const uint64_t nblk = out_dim * (in_dim / 32u), wbytes = nblk * 17u;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    __half *w16 = (__half *)v41_grow(slot, nblk * 32u * sizeof(__half), what);
    if (!w16) return 0;
    v41_fp4x32_to_f16_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, g_cur_stream>>>(w16, w, nblk);
    if (!cuda_ok(cudaGetLastError(), "v41 fp4x32 dequant")) return 0;
    *w16_out = w16;
    return 1;
}

int ds4_gpu_v41_matmul_fp4x32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * 4 || out->bytes < (uint64_t)n_tok * out_dim * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK)
        return v41_fp4x32_gemv(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr, (uint32_t)in_dim,
                               (float *)out->ptr, (uint32_t)out_dim, n_tok, 1u, 0u, 0u, round_out, "v41 fp4x32 gemv");
    __half *w16 = NULL;
    if (!v41_dequant_fp4x32(model_map, model_size, weight_offset, in_dim, out_dim, &g_v41_w16, &w16, "v41 fp4x32")) return 0;
    const uint64_t xn = (uint64_t)n_tok * in_dim;
    __half *x16 = (__half *)v41_grow(&g_v41_x16, xn * sizeof(__half), "v41 x16");
    if (!x16) return 0;
    v41_x_to_f16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(x16, (const float *)x->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 x→f16")) return 0;
    if (!v41_gemm_f16(w16, (int)in_dim, x16, (float *)out->ptr, (int)out_dim, (int)n_tok, "v41 fp4x32 gemm")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;
}

int ds4_gpu_v41_grouped_matmul_fp4x32_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                             uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (heads->bytes < (uint64_t)n_tok * in_all * 4 || low->bytes < (uint64_t)n_tok * out_all * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK) {   /* 小批: 八组一发 GEMV(grid.y=组), 输入/输出按组段取(行步长 in_all/out_all) */
        if (group_dim % 32u) return 0;
        return v41_fp4x32_gemv(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr, (uint32_t)in_all,
                               (float *)low->ptr, (uint32_t)out_all, n_tok, n_groups, (uint32_t)group_dim, (uint32_t)rank, round_out, "v41 wo_a gemv");
    }
    __half *w16 = NULL;   /* W 是 [n_groups*rank][group_dim]: 第 g 组的行块 [g*rank, (g+1)*rank) */
    if (!v41_dequant_fp4x32(model_map, model_size, weight_offset, group_dim, out_all, &g_v41_w16, &w16, "v41 wo_a")) return 0;
    const uint64_t xn = (uint64_t)n_tok * in_all;
    __half *x16 = (__half *)v41_grow(&g_v41_x16, xn * sizeof(__half), "v41 heads16");
    if (!x16) return 0;
    v41_x_to_f16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(x16, (const float *)heads->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 heads→f16")) return 0;
    /* 每组一发 GEMM: 输入按行步长 in_all 取第 g 段, 输出按行步长 out_all 写第 g 段 */
    const float alpha = 1.0f, beta = 0.0f;
    for (uint32_t g = 0; g < n_groups; g++) {
        cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rank, (int)n_tok, (int)group_dim, &alpha,
                                         w16 + (uint64_t)g * rank * group_dim, CUDA_R_16F, (int)group_dim,
                                         x16 + (uint64_t)g * group_dim, CUDA_R_16F, (int)in_all, &beta,
                                         (float *)low->ptr + (uint64_t)g * rank, CUDA_R_32F, (int)out_all,
                                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "v41 grouped gemm")) return 0;
    }
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;
}

int ds4_gpu_v41_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t wbytes = in_dim * out_dim * 4;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const float *W = (const float *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 f32 w");
    if (!W) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                    W, (int)in_dim, (const float *)x->ptr, (int)in_dim, &beta, (float *)out->ptr, (int)out_dim);
    return cublas_ok(st, "v41 f32 gemm");
}

/* ---- 嵌入: 行 = token, 每行 in_dim/32 块 ---- */
__global__ static void v41_embed_kernel(float *out, const int32_t *tok, const uint8_t *w, uint32_t n_vocab, uint32_t n_embd) {
    const uint32_t t = blockIdx.x, nb = n_embd / 32u;
    int32_t id = tok[t]; if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        const uint8_t *p = w + ((uint64_t)id * nb + b) * 17u;
        const float s = ds4_e8m0_to_f32(p[16]);
        float *o = out + (uint64_t)t * n_embd + b * 32u;
        for (int j = 0; j < 16; j++) {
            o[2 * j] = v41_bf16r(ds4_fp4_nibble_to_f32(p[j] & 0x0F) * s);
            o[2 * j + 1] = v41_bf16r(ds4_fp4_nibble_to_f32(p[j] >> 4) * s);
        }
    }
}
int ds4_gpu_v41_embed_fp4x32_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                    uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab,
                                    uint32_t n_tok, uint32_t n_embd) {
    if (!out || !tokens || n_tok == 0 || (n_embd % 32u)) return 0;
    const uint64_t wbytes = (uint64_t)n_vocab * (n_embd / 32u) * 17u;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 embed");
    if (!w || out->bytes < (uint64_t)n_tok * n_embd * 4) return 0;
    v41_embed_kernel<<<n_tok, 256, 0, g_cur_stream>>>((float *)out->ptr, (const int32_t *)tokens->ptr, w, n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "v41 embed");
}

/* ---- hc_mix: rsqrt(mean(flat²)+eps) 逐行, 乘在 GEMM 结果上 ---- */
__global__ static void v41_row_rsqrt_kernel(float *inv, const float *x, uint32_t dim, float eps) {
    const uint32_t r = blockIdx.x; const float *xr = x + (uint64_t)r * dim;
    float s = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    if (threadIdx.x == 0) inv[r] = rsqrtf(sh[0] / (float)dim + eps);
}
__global__ static void v41_scale_rows_kernel(float *y, const float *inv, uint32_t cols, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= inv[i / cols];
}
int ds4_gpu_v41_hc_mix_tensor(ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size,
                              uint64_t fn_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!mix || !hc || hc->bytes < (uint64_t)n_tok * dim * 4 || mix->bytes < (uint64_t)n_tok * mix_hc * 4) return 0;
    float *inv = (float *)v41_grow(&g_v41_misc, (uint64_t)n_tok * 4, "v41 hc inv");
    if (!inv) return 0;
    v41_row_rsqrt_kernel<<<n_tok, 256, 0, g_cur_stream>>>(inv, (const float *)hc->ptr, dim, eps);
    if (!cuda_ok(cudaGetLastError(), "v41 hc rsqrt")) return 0;
    if (!ds4_gpu_v41_matmul_f32_tensor(mix, model_map, model_size, fn_offset, dim, mix_hc, hc, n_tok)) return 0;
    const uint64_t n = (uint64_t)n_tok * mix_hc;
    v41_scale_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)mix->ptr, inv, mix_hc, n);
    return cuda_ok(cudaGetLastError(), "v41 hc mix scale");
}

/* ---- sinkhorn 拆分(官方 hc_split_sinkhorn_kernel 逐式), 一 block 一行, hc ≤ 8 ---- */
__global__ static void v41_hc_split_kernel(float *pre, float *post, float *comb, const float *mix, const float *scale,
                                           const float *base, uint32_t hc, uint32_t iters, float eps) {
    const uint32_t n = blockIdx.x, mix_hc = 2u * hc + hc * hc;
    const float *m = mix + (uint64_t)n * mix_hc;
    __shared__ float c[64];
    if (threadIdx.x < hc) {
        pre[n * hc + threadIdx.x] = 1.f / (1.f + expf(-(m[threadIdx.x] * scale[0] + base[threadIdx.x]))) + eps;
        post[n * hc + threadIdx.x] = 2.f / (1.f + expf(-(m[hc + threadIdx.x] * scale[1] + base[hc + threadIdx.x])));
    }
    if (threadIdx.x < hc * hc) c[threadIdx.x] = m[2u * hc + threadIdx.x] * scale[2] + base[2u * hc + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {   /* 4×4 矩阵的 sinkhorn 串行算, 与参考逐式同序 */
        for (uint32_t j = 0; j < hc; j++) {            /* softmax(-1) + eps */
            float mx = -INFINITY; for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
            float s = 0.f; for (uint32_t k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
            for (uint32_t k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
        }
        for (uint32_t k = 0; k < hc; k++) {            /* / (sum(-2) + eps) */
            float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k];
            for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps);
        }
        for (uint32_t it = 1; it < iters; it++) {
            for (uint32_t j = 0; j < hc; j++) { float s = 0.f; for (uint32_t k = 0; k < hc; k++) s += c[j * hc + k]; for (uint32_t k = 0; k < hc; k++) c[j * hc + k] /= (s + eps); }
            for (uint32_t k = 0; k < hc; k++) { float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k]; for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps); }
        }
    }
    __syncthreads();
    if (threadIdx.x < hc * hc) comb[(uint64_t)n * hc * hc + threadIdx.x] = c[threadIdx.x];
}
int ds4_gpu_v41_hc_split_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size,
                                uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t iters,
                                float eps, uint32_t n_tok) {
    if (!pre || !post || !comb || !mix || n_hc > 8u) return 0;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "v41 hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mix_hc * 4, "v41 hc base");
    if (!sc || !bs) return 0;
    v41_hc_split_kernel<<<n_tok, 64, 0, g_cur_stream>>>((float *)pre->ptr, (float *)post->ptr, (float *)comb->ptr,
                                                         (const float *)mix->ptr, sc, bs, n_hc, iters, eps);
    return cuda_ok(cudaGetLastError(), "v41 hc split");
}

/* hc_pre: out[n][d] = Σ_c pre[n][c]·hc[n][c][d] → bf16 */
__global__ static void v41_hc_pre_kernel(float *out, const float *hc, const float *pre, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float s = 0.f;
    for (uint32_t c = 0; c < n_hc; c++) s += pre[n * n_hc + c] * hc[((uint64_t)n * n_hc + c) * n_embd + d];
    out[(uint64_t)n * n_embd + d] = v41_bf16r(s);
}
int ds4_gpu_v41_hc_pre_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre,
                              uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out || !hc || !pre) return 0;
    v41_hc_pre_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)hc->ptr, (const float *)pre->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 hc pre");
}

/* hc_post(官方 hc_post 逐式): out[k][d] = post[k]·y[d] + Σ_j comb[j][k]·res[j][d] → bf16 */
__global__ static void v41_hc_post_kernel(float *out, const float *y, const float *res, const float *post, const float *comb,
                                          uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float yv = y[(uint64_t)n * n_embd + d];
    for (uint32_t k = 0; k < n_hc; k++) {
        float s = post[n * n_hc + k] * yv;
        for (uint32_t j = 0; j < n_hc; j++) s += comb[(uint64_t)n * n_hc * n_hc + j * n_hc + k] * res[((uint64_t)n * n_hc + j) * n_embd + d];
        out[((uint64_t)n * n_hc + k) * n_embd + d] = v41_bf16r(s);
    }
}
int ds4_gpu_v41_hc_post_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res,
                               const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                               uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out_hc || !y || !res || !post || !comb || out_hc->ptr == res->ptr) return 0;   /* 不许原地: 每 k 读全部 j */
    v41_hc_post_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_cur_stream>>>((float *)out_hc->ptr, (const float *)y->ptr, (const float *)res->ptr,
                                                                                      (const float *)post->ptr, (const float *)comb->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 hc post");
}

/* RMSNorm 带权 → bf16(官方 RMSNorm: f32 算, weight 是 bf16 值, 结果 .to(bf16)) */
__global__ static void v41_rms_norm_kernel(float *out, const float *x, const float *w, uint32_t dim, float eps) {
    const uint32_t r = blockIdx.x; const float *xr = x + (uint64_t)r * dim; float *o = out + (uint64_t)r * dim;
    float s = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) o[i] = v41_bf16r(w[i] * (xr[i] * inv));
}
int ds4_gpu_v41_rms_norm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset, uint32_t dim, uint32_t n_tok, float eps) {
    if (!out || !x) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset, (uint64_t)dim * 4, "v41 norm w");
    if (!w) return 0;
    v41_rms_norm_kernel<<<n_tok, 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)x->ptr, w, dim, eps);
    return cuda_ok(cudaGetLastError(), "v41 rms norm");
}

__global__ static void v41_add_kernel(float *a, const float *b, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
int ds4_gpu_v41_add_tensor(ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n) {
    if (!a || !b) return 0;
    v41_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 add");
}
__global__ static void v41_expand_hc_kernel(float *hc, const float *x, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float v = x[(uint64_t)n * n_embd + d];
    for (uint32_t c = 0; c < n_hc; c++) hc[((uint64_t)n * n_hc + c) * n_embd + d] = v;
}
int ds4_gpu_v41_expand_hc_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!hc || !x) return 0;
    v41_expand_hc_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_cur_stream>>>((float *)hc->ptr, (const float *)x->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 expand hc");
}
