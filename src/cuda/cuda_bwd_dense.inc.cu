/* cuda_bwd_dense.inc.cu — ds4_cuda.cu 分片: 后训练反传的稠密原语(契约 ds4_gpu_bwd.h, 2026-10-01)。
 * 转置乘 / 教师 top-K / KL 梯度 / RMSNorm·hc_pre·hc_post 反向 / 低秩放大器反向 / 梯度范数 / Adam。
 * ★必须在 cuda_v41_q4k / cuda_v41_gemv_highprec 之后 include★: 借它们的 q4_K·fp4x32·fp8 → bf16 解码核与 bf16 暂存槽,
 * 反向读到的权重值与前向预填 GEMM 路逐位同一份(同一个解码核)。
 * 反向一律不舍 bf16(前向的舍入当直通, 见头文件)。速度不是这里的目标: 训练一步的大头是前向与逐专家解码, 这些核都是零头。 */
#include "src/common/ds4_quantfmt.h"

/* 骨架矩阵第 [r0, r0+rows) 行解成 bf16 [rows][in](bf16 权重不解码, 直接返回映射里的指针)。fp8 32×32 的缩放按 32 行一格, r0 必须是 32 的倍数。 */
static const __nv_bfloat16 *bwd_rows_bf16(__nv_bfloat16 *wb, const void *model_map, uint64_t model_size, uint32_t wtype,
                                          uint64_t off, uint64_t in_dim, uint64_t out_dim, uint64_t r0, uint32_t rows) {
    if (wtype == DS4_GGT_BF16) {
        const uint64_t bytes = in_dim * out_dim * 2u;
        if (off > model_size || bytes > model_size - off) return NULL;
        const __nv_bfloat16 *w = (const __nv_bfloat16 *)cuda_model_range_ptr(model_map, off, bytes, "bwd bf16 w");
        return w ? w + r0 * in_dim : NULL;
    }
    if (wtype == DS4_GGT_Q4_K) {
        if (in_dim % V41_Q4K_BLK) return NULL;
        /* 整块要且本层重算刚解过(层内 bf16 缓存, cuda_v41_q4k.inc.cu): 直接用那份, 同一个解码核的同一份值 */
        if (r0 == 0u && rows == out_dim) { const __nv_bfloat16 *c = v41_wc_find(off); if (c) return c; }
        const uint64_t bpr = in_dim / V41_Q4K_BLK, bytes = out_dim * bpr * V41_Q4K_BYTES, nblk = (uint64_t)rows * bpr;
        if (off > model_size || bytes > model_size - off) return NULL;
        const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, bytes, "bwd q4k w");
        if (!w) return NULL;
        v41_q4k_to_bf16_kernel<<<(unsigned)((nblk + 7) / 8), dim3(32, 8), 0, g_cur_stream>>>(wb, w + r0 * bpr * V41_Q4K_BYTES, nblk);
        return cuda_ok(cudaGetLastError(), "bwd q4k→bf16") ? wb : NULL;
    }
    if (wtype == DS4_GGT_FP4X32) {
        if (in_dim % 32u) return NULL;
        const uint64_t bpr = in_dim / 32u, bytes = out_dim * bpr * 17u, nblk = (uint64_t)rows * bpr;
        if (off > model_size || bytes > model_size - off) return NULL;
        const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, bytes, "bwd fp4 w");
        if (!w) return NULL;
        v41_fp4x32_to_bf16_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, g_cur_stream>>>(wb, w + r0 * bpr * 17u, nblk);
        return cuda_ok(cudaGetLastError(), "bwd fp4→bf16") ? wb : NULL;
    }
    if (wtype == DS4_GGT_FP8_32X32) {
        if (r0 % 32u) return NULL;
        const uint64_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u, bytes = in_dim * out_dim + sbr * sbc;
        if (off > model_size || bytes > model_size - off) return NULL;
        const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, bytes, "bwd fp8 w");
        if (!w) return NULL;
        return v41_fp8blk_to_bf16(wb, w + r0 * in_dim, w + in_dim * out_dim + (r0 / 32u) * sbc, in_dim, rows) ? wb : NULL;
    }
    fprintf(stderr, "ds4: [bwd] 转置乘不认权重类型 %u\n", wtype);
    return NULL;
}

int ds4_gpu_bwd_matmul_t_tensor(ds4_gpu_tensor *gx, const void *model_map, uint64_t model_size, uint32_t wtype,
                                uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                const ds4_gpu_tensor *gy, uint32_t n_tok, int accumulate) {
    if (!gx || !gy || !g_cublas_ready || n_tok == 0) return 0;
    if (gx->bytes < (uint64_t)n_tok * in_dim * 4 || gy->bytes < (uint64_t)n_tok * out_dim * 4) return 0;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    if (wtype == DS4_GGT_F32) {   /* hc 混合矩阵这类小 f32 权重: 直接 Sgemm, 梯度不降精度 */
        const uint64_t bytes = in_dim * out_dim * 4u;
        if (weight_offset > model_size || bytes > model_size - weight_offset) return 0;
        const float *W = (const float *)cuda_model_range_ptr(model_map, weight_offset, bytes, "bwd f32 w");
        if (!W) return 0;
        const float a1 = 1.f, b = accumulate ? 1.f : 0.f;
        return cublas_ok(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)in_dim, (int)n_tok, (int)out_dim, &a1,
                                     W, (int)in_dim, (const float *)gy->ptr, (int)out_dim, &b, (float *)gx->ptr, (int)in_dim), "bwd f32 gemm");
    }
    const uint64_t yn = (uint64_t)n_tok * out_dim;
    __nv_bfloat16 *yb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, yn * sizeof(__nv_bfloat16), "bwd gy bf16");
    if (!yb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((yn + 255) / 256), 256, 0, g_cur_stream>>>(yb, (const float *)gy->ptr, yn);
    if (!cuda_ok(cudaGetLastError(), "bwd gy→bf16")) return 0;
    /* 按输出维(权重行)分块: 与前向预填同一个 400 MB 暂存上限(出口头 129280 行分 4 块), 每块的贡献 beta=1 累加 */
    uint64_t tile = V41_BF16_STAGE_ELEMS / in_dim;
    tile = tile < 32u ? 32u : (tile & ~31ull);
    if (tile > out_dim) tile = out_dim;
    __nv_bfloat16 *wb = wtype == DS4_GGT_BF16 ? NULL :
        (__nv_bfloat16 *)v41_grow(&g_v41_wbf, tile * in_dim * sizeof(__nv_bfloat16), "bwd w bf16");
    if (wtype != DS4_GGT_BF16 && !wb) return 0;
    for (uint64_t r0 = 0; r0 < out_dim; r0 += tile) {
        const uint32_t rows = (uint32_t)(out_dim - r0 < tile ? out_dim - r0 : tile);
        const __nv_bfloat16 *w = bwd_rows_bf16(wb, model_map, model_size, wtype, weight_offset, in_dim, out_dim, r0, rows);
        if (!w) return 0;
        const float a1 = 1.f, b = (accumulate || r0 > 0) ? 1.f : 0.f;
        const cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)in_dim, (int)n_tok, (int)rows, &a1,
                                               w, CUDA_R_16BF, (int)in_dim, yb + r0, CUDA_R_16BF, (int)out_dim, &b,
                                               (float *)gx->ptr, CUDA_R_32F, (int)in_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "bwd bf16 gemm")) return 0;
    }
    return 1;
}

int ds4_gpu_bwd_grouped_matmul_t_tensor(ds4_gpu_tensor *gheads, const void *model_map, uint64_t model_size, uint32_t wtype,
                                        uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim, uint64_t rank,
                                        const ds4_gpu_tensor *glow, uint32_t n_tok, int accumulate) {
    if (!gheads || !glow || !g_cublas_ready || n_tok == 0 || wtype == DS4_GGT_F32) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (gheads->bytes < (uint64_t)n_tok * in_all * 4 || glow->bytes < (uint64_t)n_tok * out_all * 4) return 0;
    const uint64_t yn = (uint64_t)n_tok * out_all;
    __nv_bfloat16 *yb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, yn * sizeof(__nv_bfloat16), "bwd glow bf16");
    __nv_bfloat16 *wb = wtype == DS4_GGT_BF16 ? NULL :
        (__nv_bfloat16 *)v41_grow(&g_v41_wbf, out_all * group_dim * sizeof(__nv_bfloat16), "bwd wo_a bf16");
    if (!yb || (wtype != DS4_GGT_BF16 && !wb)) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((yn + 255) / 256), 256, 0, g_cur_stream>>>(yb, (const float *)glow->ptr, yn);
    if (!cuda_ok(cudaGetLastError(), "bwd glow→bf16")) return 0;
    /* 整块一次解(行 = G·rank, 列 = gd): wo_a 只有几千万元素, 用不着分块 */
    const __nv_bfloat16 *w = bwd_rows_bf16(wb, model_map, model_size, wtype, weight_offset, group_dim, out_all, 0, (uint32_t)out_all);
    if (!w) return 0;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    const float a1 = 1.f, b = accumulate ? 1.f : 0.f;
    for (uint32_t g = 0; g < n_groups; g++) {
        const cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)group_dim, (int)n_tok, (int)rank, &a1,
                                               w + (uint64_t)g * rank * group_dim, CUDA_R_16BF, (int)group_dim,
                                               yb + (uint64_t)g * rank, CUDA_R_16BF, (int)out_all, &b,
                                               (float *)gheads->ptr + (uint64_t)g * group_dim, CUDA_R_32F, (int)in_all,
                                               CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "bwd wo_a gemm")) return 0;
    }
    return 1;
}

/* ---- 块内归约小工具(1024 线程以内, blockDim 是 32 的倍数) ---- */
__device__ static float bwd_block_sum(float v, float *sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ static float bwd_block_max(float v, float *sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < nw ? sh[threadIdx.x] : -INFINITY;
    if (w == 0) for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

#define BWD_KMAX 128u
#define BWD_VMASK_WORDS 4096u   /* 词表位图(一 token 一位): 131072 个词以内; 现役 129280 */
/* ★余量必须直接累加榜外项, 不许 1 − Σ榜上★(2026-10-01 实撞): 学生把概率几乎全押在榜上时, 1 − Σp 在 f32 里抵消成 0 甚至负数,
 * 托底成 1e-20 后 r_T/r_S 炸到 1e16, 榜外每个 token 的梯度 = p_j·(1 − 1e16) —— 批梯度范数在 0.2 与 2.7e7 之间来回跳, 一轮就把模型
 * 推成只吐"2026303030…"。现在 r_S = Σ_{j∉榜} e^{z_j−m} / Σ_j e^{z_j−m}(位图标榜上, 一遍归约), 两数都是正项和, 没有相消。 */
__global__ static void bwd_kl_topk_kernel(float *g, float *loss, const float *z, uint32_t row0, uint32_t V,
                                          const int32_t *tid, const float *tp, const float *trest, const float *w, uint32_t K, float scale) {
    __shared__ float sh[32];
    __shared__ float ps_k[BWD_KMAX];
    __shared__ uint32_t mask[BWD_VMASK_WORDS];
    const uint32_t i = blockIdx.x;
    const float wi = w ? w[i] : 1.f;   /* 逐行权重(硬目标题: 数字位 × hard_num, 其余 × hard_txt); 损失与梯度同乘, 报出来的 KL 也是加权的 */
    scale *= wi;
    const float *zr = z + (uint64_t)(row0 + i) * V;
    float *gr = g + (uint64_t)i * V;
    const int32_t *ids = tid + (uint64_t)i * K;
    const float *pt = tp + (uint64_t)i * K;
    for (uint32_t w = threadIdx.x; w < BWD_VMASK_WORDS; w += blockDim.x) mask[w] = 0u;
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) { const int32_t id = ids[k]; if (id >= 0 && (uint32_t)id < V) atomicOr(&mask[id >> 5], 1u << (id & 31)); }
    float mx = -INFINITY;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, zr[j]);
    mx = bwd_block_max(mx, sh);   /* 内含 __syncthreads: 位图此后可读 */
    float s = 0.f, sr = 0.f;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) {
        const float ev = expf(zr[j] - mx);
        s += ev;
        if (!((mask[j >> 5] >> (j & 31)) & 1u)) sr += ev;
    }
    s = bwd_block_sum(s, sh);
    sr = bwd_block_sum(sr, sh);
    const float lse = mx + logf(s), rs = sr / s;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const int32_t id = ids[k];
        ps_k[k] = (id >= 0 && (uint32_t)id < V) ? expf(zr[id] - lse) : 0.f;
    }
    __syncthreads();
    __shared__ float s_off;
    if (threadIdx.x == 0) {
        float l = 0.f;
        for (uint32_t k = 0; k < K; k++) if (pt[k] > 0.f) l += pt[k] * (logf(pt[k]) - logf(fmaxf(ps_k[k], 1e-30f)));
        const float rt = fmaxf(trest[i], 0.f);
        if (rt > 1e-12f) l += rt * (logf(rt) - logf(fmaxf(rs, 1e-30f)));
        loss[i] = l * wi;
        /* 榜外: q_j = r_T·pS_j/r_S ⇒ g_j = pS_j·(1 − r_T/r_S); r_S 真为 0 时 pS_j 也全为 0, 这一项本来就没有梯度 */
        s_off = rs > 1e-30f ? 1.f - rt / rs : 0.f;
    }
    __syncthreads();
    const float off = s_off;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) gr[j] = scale * expf(zr[j] - lse) * off;   /* 同址时: 每个 j 先读后写, 只有本线程碰它 */
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const int32_t id = ids[k];
        if (id >= 0 && (uint32_t)id < V) gr[id] = scale * (ps_k[k] - pt[k]);
    }
}
int ds4_gpu_bwd_kl_topk_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t row0,
                               uint32_t m, uint32_t n_vocab, const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp,
                               const ds4_gpu_tensor *trest, const ds4_gpu_tensor *w, uint32_t k, float scale) {
    if (!glogits || !loss || !logits || !tid || !tp || !trest || m == 0 || k == 0 || k > BWD_KMAX || n_vocab > BWD_VMASK_WORDS * 32u) return 0;
    if (logits->bytes < (uint64_t)(row0 + m) * n_vocab * 4 || glogits->bytes < (uint64_t)m * n_vocab * 4) return 0;
    if (w && w->bytes < (uint64_t)m * 4) return 0;
    /* 同址只许"glogits 就是 logits 第 row0 行起的视图"(逐行自读自写); 同一块缓冲但行错开会读到别的块写过的行 */
    if ((const char *)glogits->ptr != (const char *)logits->ptr + (uint64_t)row0 * n_vocab * 4 &&
        (const char *)glogits->ptr < (const char *)logits->ptr + logits->bytes &&
        (const char *)glogits->ptr + (uint64_t)m * n_vocab * 4 > (const char *)logits->ptr) {
        fprintf(stderr, "ds4: [bwd] KL 梯度与 logits 部分重叠(只许整行同址)\n"); return 0;
    }
    bwd_kl_topk_kernel<<<m, 1024, 0, g_cur_stream>>>((float *)glogits->ptr, (float *)loss->ptr, (const float *)logits->ptr, row0, n_vocab,
                                                     (const int32_t *)tid->ptr, (const float *)tp->ptr, (const float *)trest->ptr,
                                                     w ? (const float *)w->ptr : NULL, k, scale);
    return cuda_ok(cudaGetLastError(), "bwd kl topk");
}

/* 教师 top-K: 先整行 log-sum-exp, 再 K 轮"全块 argmax(同值取小下标) → 置 −inf" */
__global__ static void bwd_topk_kernel(int32_t *tid, float *tp, float *trest, float *z, uint32_t row0, uint32_t V, uint32_t K) {
    __shared__ float sh[32], shv[32];
    __shared__ int32_t shi[32];
    const uint32_t i = blockIdx.x;
    float *zr = z + (uint64_t)(row0 + i) * V;
    float mx = -INFINITY;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, zr[j]);
    mx = bwd_block_max(mx, sh);
    float s = 0.f;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) s += expf(zr[j] - mx);
    s = bwd_block_sum(s, sh);
    float psum = 0.f;
    for (uint32_t k = 0; k < K; k++) {
        float bv = -INFINITY; int32_t bi = 0x7fffffff;
        for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) { const float v = zr[j]; if (v > bv) { bv = v; bi = (int32_t)j; } }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u, nw = blockDim.x >> 5;
        if (l == 0) { shv[w] = bv; shi[w] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v0 = shv[0]; int32_t i0 = shi[0];
            for (uint32_t q = 1; q < nw; q++) if (shv[q] > v0 || (shv[q] == v0 && shi[q] < i0)) { v0 = shv[q]; i0 = shi[q]; }
            const float p = v0 == -INFINITY ? 0.f : expf(v0 - mx) / s;
            tid[(uint64_t)i * K + k] = v0 == -INFINITY ? -1 : i0;
            tp[(uint64_t)i * K + k] = p;
            psum += p;
            if (v0 != -INFINITY) zr[i0] = -INFINITY;
        }
        __syncthreads();
    }
    (void)psum;
    /* 余量 = 榜外直接累加(选中的已置 −inf, e^{−inf} = 0); 不用 1 − Σ榜上: 那是两个接近的数相减(见 KL 核头的实撞记录) */
    float sr = 0.f;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) sr += expf(zr[j] - mx);
    sr = bwd_block_sum(sr, sh);
    if (threadIdx.x == 0) trest[i] = sr / s;
}
int ds4_gpu_bwd_topk_tensor(ds4_gpu_tensor *tid, ds4_gpu_tensor *tp, ds4_gpu_tensor *trest, ds4_gpu_tensor *logits,
                            uint32_t row0, uint32_t m, uint32_t n_vocab, uint32_t k) {
    if (!tid || !tp || !trest || !logits || m == 0 || k == 0) return 0;
    if (logits->bytes < (uint64_t)(row0 + m) * n_vocab * 4 || tid->bytes < (uint64_t)m * k * 4 || tp->bytes < (uint64_t)m * k * 4) return 0;
    bwd_topk_kernel<<<m, 1024, 0, g_cur_stream>>>((int32_t *)tid->ptr, (float *)tp->ptr, (float *)trest->ptr, (float *)logits->ptr, row0, n_vocab, k);
    return cuda_ok(cudaGetLastError(), "bwd topk");
}

__global__ static void bwd_rms_norm_kernel(float *gx, const float *gxn, const float *x, const float *w, uint32_t D, float eps, int acc) {
    __shared__ float sh[32];
    const uint64_t r = blockIdx.x;
    const float *xr = x + r * D, *gr = gxn + r * D;
    float ss = 0.f, dot = 0.f;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) { const float xv = xr[d]; ss += xv * xv; dot += w[d] * gr[d] * xv; }
    ss = bwd_block_sum(ss, sh);
    dot = bwd_block_sum(dot, sh);
    const float inv = rsqrtf(ss / (float)D + eps), c = inv * inv * inv * dot / (float)D;
    float *o = gx + r * D;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = inv * w[d] * gr[d] - xr[d] * c;
        o[d] = acc ? o[d] + v : v;
    }
}
int ds4_gpu_bwd_rms_norm_tensor(ds4_gpu_tensor *gx, const ds4_gpu_tensor *gxn, const ds4_gpu_tensor *x,
                                const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                uint32_t dim, uint32_t n_tok, float eps, int accumulate) {
    if (!gx || !gxn || !x || n_tok == 0) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset, (uint64_t)dim * 4, "bwd norm w");
    if (!w) return 0;
    bwd_rms_norm_kernel<<<n_tok, 512, 0, g_cur_stream>>>((float *)gx->ptr, (const float *)gxn->ptr, (const float *)x->ptr, w, dim, eps, accumulate);
    return cuda_ok(cudaGetLastError(), "bwd rms norm");
}

#define BWD_HC_MAX 4u   /* 现役 mHC 4 路; 改宽度要连同下面两个核的寄存器数组一起改 */
__global__ static void bwd_hc_pre_kernel(float *ghc, float *gpre, const float *gx, const float *hc, const float *pre, uint32_t E, uint32_t HC) {
    __shared__ float sh[32];
    const uint64_t n = blockIdx.x;
    float acc[BWD_HC_MAX] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) {
        const float g = gx[n * E + d];
        for (uint32_t c = 0; c < HC; c++) {
            const uint64_t o = (n * HC + c) * E + d;
            acc[c] += g * hc[o];
            ghc[o] += pre[n * HC + c] * g;
        }
    }
    for (uint32_t c = 0; c < HC; c++) { const float v = bwd_block_sum(acc[c], sh); if (threadIdx.x == 0) gpre[n * HC + c] = v; }
}
int ds4_gpu_bwd_hc_pre_tensor(ds4_gpu_tensor *ghc, ds4_gpu_tensor *gpre, const ds4_gpu_tensor *gx,
                              const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!ghc || !gpre || !gx || !hc || !pre || n_hc > BWD_HC_MAX || n_tok == 0) return 0;
    bwd_hc_pre_kernel<<<n_tok, 512, 0, g_cur_stream>>>((float *)ghc->ptr, (float *)gpre->ptr, (const float *)gx->ptr,
                                                       (const float *)hc->ptr, (const float *)pre->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "bwd hc pre");
}

__global__ static void bwd_hc_post_kernel(float *gy, float *gres, float *gpost, float *gcomb, const float *gout, const float *y,
                                          const float *res, const float *post, const float *comb, uint32_t E, uint32_t HC) {
    __shared__ float sh[32];
    const uint64_t n = blockIdx.x;
    const float *cb = comb + n * HC * HC, *po = post + n * HC;
    float ap[BWD_HC_MAX] = {0.f, 0.f, 0.f, 0.f}, ac[BWD_HC_MAX * BWD_HC_MAX];
    for (uint32_t q = 0; q < BWD_HC_MAX * BWD_HC_MAX; q++) ac[q] = 0.f;
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) {
        float go[BWD_HC_MAX], rv[BWD_HC_MAX];
        for (uint32_t k = 0; k < HC; k++) go[k] = gout[(n * HC + k) * E + d];
        for (uint32_t j = 0; j < HC; j++) rv[j] = res[(n * HC + j) * E + d];
        const float yv = y[n * E + d];
        float s = 0.f;
        for (uint32_t k = 0; k < HC; k++) { s += po[k] * go[k]; ap[k] += go[k] * yv; }
        gy[n * E + d] = s;
        for (uint32_t j = 0; j < HC; j++) {
            float r = 0.f;
            for (uint32_t k = 0; k < HC; k++) { r += cb[j * HC + k] * go[k]; ac[j * BWD_HC_MAX + k] += go[k] * rv[j]; }
            if (gres) gres[(n * HC + j) * E + d] += r;
        }
    }
    for (uint32_t k = 0; k < HC; k++) { const float v = bwd_block_sum(ap[k], sh); if (threadIdx.x == 0 && gpost) gpost[n * HC + k] = v; }
    for (uint32_t j = 0; j < HC; j++)
        for (uint32_t k = 0; k < HC; k++) { const float v = bwd_block_sum(ac[j * BWD_HC_MAX + k], sh); if (threadIdx.x == 0 && gcomb) gcomb[n * HC * HC + j * HC + k] = v; }
}
int ds4_gpu_bwd_hc_post_tensor(ds4_gpu_tensor *gy, ds4_gpu_tensor *gres, ds4_gpu_tensor *gpost, ds4_gpu_tensor *gcomb,
                               const ds4_gpu_tensor *gout, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res,
                               const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                               uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!gy || !gout || !y || !res || !post || !comb || n_hc > BWD_HC_MAX || n_tok == 0) return 0;
    bwd_hc_post_kernel<<<n_tok, 512, 0, g_cur_stream>>>((float *)gy->ptr, gres ? (float *)gres->ptr : NULL, gpost ? (float *)gpost->ptr : NULL,
                                                        gcomb ? (float *)gcomb->ptr : NULL, (const float *)gout->ptr, (const float *)y->ptr,
                                                        (const float *)res->ptr, (const float *)post->ptr, (const float *)comb->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "bwd hc post");
}

int ds4_gpu_bwd_amp_tensor(ds4_gpu_tensor *gA, ds4_gpu_tensor *gB, ds4_gpu_tensor *gx, const ds4_gpu_tensor *gy,
                           const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                           ds4_gpu_tensor *T, ds4_gpu_tensor *gT, uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!gA || !gB || !gy || !x || !A || !B || !T || !gT || !g_cublas_ready || n_tok == 0) return 0;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    const float one = 1.f, zero = 0.f;
    /* 列主序视角: x/gy = D×n, A/B = D×K(内存 [K][D]), T/gT = K×n */
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &one,
                                    (const float *)B->ptr, (int)D, (const float *)x->ptr, (int)D, &zero, (float *)T->ptr, (int)K);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &one,
                         (const float *)A->ptr, (int)D, (const float *)gy->ptr, (int)D, &zero, (float *)gT->ptr, (int)K);
    if (st == CUBLAS_STATUS_SUCCESS)   /* gA(D×K) += gy(D×n)·Tᵀ */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, (int)D, (int)K, (int)n_tok, &one,
                         (const float *)gy->ptr, (int)D, (const float *)T->ptr, (int)K, &one, (float *)gA->ptr, (int)D);
    if (st == CUBLAS_STATUS_SUCCESS)   /* gB(D×K) += x(D×n)·gTᵀ */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, (int)D, (int)K, (int)n_tok, &one,
                         (const float *)x->ptr, (int)D, (const float *)gT->ptr, (int)K, &one, (float *)gB->ptr, (int)D);
    if (st == CUBLAS_STATUS_SUCCESS && gx)   /* gx(D×n) += B(D×K)·gT(K×n) */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n_tok, (int)K, &one,
                         (const float *)B->ptr, (int)D, (const float *)gT->ptr, (int)K, &one, (float *)gx->ptr, (int)D);
    return cublas_ok(st, "bwd amp");
}

__global__ static void bwd_sumsq_kernel(double *out, const float *g, uint64_t n) {
    __shared__ float sh[32];
    float s = 0.f;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) s += g[i] * g[i];
    s = bwd_block_sum(s, sh);
    if (threadIdx.x == 0) atomicAdd(out, (double)s);
}
static v41_scratch g_bwd_dsum;
int ds4_gpu_bwd_sumsq_tensor(const ds4_gpu_tensor *g, uint64_t n, double *out) {
    if (!g || !out) return 0;
    double *d = (double *)v41_grow(&g_bwd_dsum, sizeof(double), "bwd sumsq");
    if (!d || !cuda_ok(cudaMemsetAsync(d, 0, sizeof(double), v41_cublas_stream()), "bwd sumsq zero")) return 0;
    bwd_sumsq_kernel<<<256, 512, 0, g_cur_stream>>>(d, (const float *)g->ptr, n);
    if (!cuda_ok(cudaGetLastError(), "bwd sumsq")) return 0;
    return cuda_ok(cudaMemcpyAsync(out, d, sizeof(double), cudaMemcpyDeviceToHost, v41_cublas_stream()), "bwd sumsq d2h") &&
           cuda_ok(cudaStreamSynchronize(v41_cublas_stream()), "bwd sumsq sync");
}

__global__ static void bwd_adam_kernel(float *p, float *g, float *m, float *v, uint64_t n, float lr, float b1, float b2, float eps,
                                       float gs, float c1, float c2) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        const float gi = g[i] * gs;
        const float mi = b1 * m[i] + (1.f - b1) * gi, vi = b2 * v[i] + (1.f - b2) * gi * gi;
        m[i] = mi; v[i] = vi;
        p[i] -= lr * (mi / c1) / (sqrtf(vi / c2) + eps);
        g[i] = 0.f;
    }
}
int ds4_gpu_bwd_adam_tensor(ds4_gpu_tensor *p, ds4_gpu_tensor *g, ds4_gpu_tensor *m, ds4_gpu_tensor *v, uint64_t n,
                            float lr, float beta1, float beta2, float eps, float gs, uint32_t step) {
    if (!p || !g || !m || !v || step == 0) return 0;
    const float c1 = 1.f - powf(beta1, (float)step), c2 = 1.f - powf(beta2, (float)step);
    bwd_adam_kernel<<<512, 256, 0, g_cur_stream>>>((float *)p->ptr, (float *)g->ptr, (float *)m->ptr, (float *)v->ptr, n, lr, beta1, beta2, eps, gs, c1, c2);
    return cuda_ok(cudaGetLastError(), "bwd adam");
}

/* 层入口 hc 存档: 值本来就在 bf16 格点上(hc_post/嵌入/engram 出口都舍过), 存高 16 位无损; v41_bf16r 兜底(万一上游漏舍也只差一次正确舍入) */
__global__ static void bwd_pack16_kernel(uint16_t *o, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (uint16_t)(__float_as_uint(v41_bf16r(x[i])) >> 16);
}
__global__ static void bwd_unpack16_kernel(float *o, const uint16_t *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __uint_as_float((uint32_t)x[i] << 16);
}
int ds4_gpu_bwd_pack_bf16_tensor(ds4_gpu_tensor *dst16, const ds4_gpu_tensor *src, uint64_t n) {
    if (!dst16 || !src || dst16->bytes < n * 2u || src->bytes < n * 4u) return 0;
    bwd_pack16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((uint16_t *)dst16->ptr, (const float *)src->ptr, n);
    return cuda_ok(cudaGetLastError(), "bwd pack16");
}
int ds4_gpu_bwd_unpack_bf16_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src16, uint64_t n) {
    if (!dst || !src16 || dst->bytes < n * 4u || src16->bytes < n * 2u) return 0;
    bwd_unpack16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)dst->ptr, (const uint16_t *)src16->ptr, n);
    return cuda_ok(cudaGetLastError(), "bwd unpack16");
}

__global__ static void bwd_axpy_kernel(float *y, const float *x, float a, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) y[i] += a * x[i];
}
int ds4_gpu_bwd_axpy_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, float a, uint64_t n) {
    if (!y || !x || n == 0) return 0;
    bwd_axpy_kernel<<<256, 256, 0, g_cur_stream>>>((float *)y->ptr, (const float *)x->ptr, a, n);
    return cuda_ok(cudaGetLastError(), "bwd axpy");
}
