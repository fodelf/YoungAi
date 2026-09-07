/* cuda_embed_norm_kernels_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * embed/corr/repeat/norm/matmul(f16/f32) kernel 族 + kv fp8 rope store。
 */
__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

/* go1b correction kernels (mirror metal/moe.metal kernel_dsv4_corr_*). */
__global__ static void corr_router_bias_kernel(
        float *logits, const float *delta, uint32_t n_expert, uint64_t total) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    logits[gid] += delta[(uint32_t)(gid % n_expert)];
}

__global__ static void corr_apply_kernel(
        float *out, const float *x, const float *U, const float *V,
        const float *C, const float *b, const float *beta, const int *selected,
        uint32_t d_model, uint32_t d_l, uint32_t n_expert, uint32_t n_sel) {
    extern __shared__ float vx[];   // [d_l] = V @ x[tok]
    uint32_t tok = blockIdx.x;
    const float *xt = x + (uint64_t)tok * d_model;
    const int *sel = selected + (uint64_t)tok * n_sel;
    for (uint32_t i = threadIdx.x; i < d_l; i += blockDim.x) {
        const float *Vrow = V + (uint64_t)i * d_model;
        float acc = 0.0f;
        for (uint32_t j = 0; j < d_model; j++) acc += Vrow[j] * xt[j];
        vx[i] = acc;
    }
    __syncthreads();
    float beta_sum = 0.0f; uint32_t n_valid = 0;
    for (uint32_t s = 0; s < n_sel; s++) {
        int e = sel[s];
        if (e >= 0 && (uint32_t)e < n_expert) { beta_sum += beta[e]; n_valid++; }
    }
    float *outt = out + (uint64_t)tok * d_model;
    for (uint32_t d = threadIdx.x; d < d_model; d += blockDim.x) {
        const float *Urow = U + (uint64_t)d * d_l;
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_sel; s++) {
            int e = sel[s];
            if (e < 0 || (uint32_t)e >= n_expert) continue;
            const float *Crow = C + (uint64_t)e * d_l;
            float p = 0.0f;
            for (uint32_t i = 0; i < d_l; i++) p += Urow[i] * Crow[i] * vx[i];
            acc += p;
        }
        acc += (float)n_valid * b[d] + beta_sum;
        outt[d] += acc;
    }
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

static float *g_f16sk_partial = NULL;  /* split-K f16 partial: [64][64] 上限 16KB */

/* split-K f16(2026-08-17 第六夜): 瘦高矩阵(16384→4 只发 1 个 block / 16384→24 发 3 个,
 * 45 个 SM 围观)按 K 维切 S 段并行, partial 后按 s 固定序 reduce(确定性)。
 * decode(n_tok==1) 且 out≤64 且 in≥4096 时启用。 */
__global__ static void matmul_f16_splitk_kernel(
        float *partial,               /* [n_tok][S][out_dim] */
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t chunk,               /* 每段元素数, 8 的倍数 */
        uint32_t n_s,
        uint32_t n_tok) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    /* splitk3(施工日2): grid-stride cell 循环 — (out,S)≤4096 超短命块(每块~2KB即死)
     * 改常驻块循环(pair3 同配方); 单 cell 计算序不变=逐位一致。 */
    /* 批扩展(2026-08-21 verify 数值对齐): cell 空间加 token 维。每个 (tok,row,s) cell 的
     * 内部计算序与 n_tok==1 完全一致 ⇒ verify 批与 decode 逐位相同, 投机才可能无损。 */
    for (uint32_t cell = blockIdx.x; cell < out_dim * n_s * n_tok; cell += gridDim.x) {
    const uint32_t row = cell % (uint32_t)out_dim;
    const uint32_t s = (cell / (uint32_t)out_dim) % n_s;
    const uint32_t tok = cell / ((uint32_t)out_dim * n_s);
    const float *xt = x + (uint64_t)tok * in_dim;
    const uint64_t pidx = ((uint64_t)tok * n_s + s) * out_dim + row;
    const uint64_t base = (uint64_t)s * chunk;
    if (base >= in_dim) { if (threadIdx.x == 0) partial[pidx] = 0.0f; continue; }
    uint64_t end = base + chunk; if (end > in_dim) end = in_dim;
    float sum = 0.0f;
    /* uint4 宽读(2026-08-20 ②③刀): 8 half/load, 访存队列×4(LPDDR 要深队列);
     * chunk 为 8 倍数 ⇒ 段界 8 对齐; 行字节 in_dim*2, in 为 8 倍数时 16B 对齐。 */
    if (((in_dim & 7u) == 0u)) {
        const uint4 *wr4 = (const uint4 *)(w + (uint64_t)row * in_dim);
        const uint64_t g0 = base >> 3, g1 = end >> 3;
        for (uint64_t i = g0 + threadIdx.x; i < g1; i += blockDim.x) {
            const uint4 v = wr4[i];
            const __half2 *h = (const __half2 *)&v;
            const float4 xa = *(const float4 *)(xt + 8u * i);
            const float4 xb = *(const float4 *)(xt + 8u * i + 4u);
            sum += __low2float(h[0]) * xa.x + __high2float(h[0]) * xa.y
                 + __low2float(h[1]) * xa.z + __high2float(h[1]) * xa.w
                 + __low2float(h[2]) * xb.x + __high2float(h[2]) * xb.y
                 + __low2float(h[3]) * xb.z + __high2float(h[3]) * xb.w;
        }
    } else {
        const __half2 *wr = (const __half2 *)(w + (uint64_t)row * in_dim);
        const uint64_t h0 = base >> 1, h1 = end >> 1;
        for (uint64_t i = h0 + threadIdx.x; i < h1; i += blockDim.x) {
            const __half2 h = wr[i];
            sum += __low2float(h) * xt[2u * i] + __high2float(h) * xt[2u * i + 1u];
        }
    }
    /* 块内确定性归约: warp shuffle 树 + shared 固定序 */
    __shared__ float wsum[8];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31u) == 0) wsum[threadIdx.x >> 5u] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int wgi = 0; wgi < 8; wgi++) t += wsum[wgi];
        partial[pidx] = t;
    }
    __syncthreads();   /* splitk3: 下轮复用 wsum 前全员读完 */
    }
}

__global__ static void matmul_f16_splitk_reduce_kernel(
        float *out, const float *partial, uint32_t out_dim, uint32_t S, uint32_t n_tok) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim * n_tok) return;
    const uint32_t row = idx % out_dim;
    const uint32_t tok = idx / out_dim;
    float t = 0.0f;   /* s 固定序求和: 与 n_tok==1 完全一致 */
    for (uint32_t s = 0; s < S; s++) t += partial[((uint64_t)tok * S + s) * out_dim + row];
    out[idx] = t;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    /* 重写(同 pair 版): 原"每线程连续 chunk"访存不合并且 32 线程/block 一行。
     * 现 8 行/block × lane 交错 __half2 合并读。 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const __half2 *wr = (const __half2 *)(w + row * in_dim);
    const float *xr = x + tok * in_dim;
    const uint64_t n2 = in_dim >> 1;
    float sum = 0.0f;
    for (uint64_t i = lane; i < n2; i += 32u) {
        const __half2 h = wr[i];
        sum += __low2float(h) * xr[2u * i] + __high2float(h) * xr[2u * i + 1u];
    }
    if ((in_dim & 1u) && lane == 0)
        sum += __half2float(w[row * in_dim + in_dim - 1u]) * xr[in_dim - 1u];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[tok * out_dim + row] = sum;
}

/* pair 一行一块版(2026-08-17): 原 8 行/块×双矩阵交错, out=256/512 只发 32/64 块
 * (1/6 GPU 干活)。现 grid=2*out, 前半 w0 后半 w1, 256 线程整行, 块内固定序归约。 */
__global__ static void matmul_f16_pair_rowblock_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t pair2) {
    /* 刀③二进(2026-08-20 夜): 8行×32lane 版败因=每行在飞载荷太浅(2.90→3.11 档案);
     * 现 pair2=2行/块×128线程(uint4 宽读保留), 行内并行度仅减半、行数×2=深流水。
     * pair2=0(DS4_F16_PAIR2=0) 回一行一块原版。2*out 恒偶 ⇒ 双行永同界, 无发散 sync 险。 */
    const uint32_t nrow_th = pair2 ? 128u : 256u;
    const uint32_t tid = threadIdx.x & (nrow_th - 1u);
    /* pair3(施工日2): grid-stride 行循环 — 2048短命块(每块2迭代即死, 7 waves 全ramp,
     * 151GB/s)改常驻块循环吃行(dense 成功配方); 读宽/归约树不变。 */
    for (uint32_t r = pair2 ? (blockIdx.x * 2u + (threadIdx.x >> 7u)) : blockIdx.x;
         r < 2u * out_dim;
         r += pair2 ? (gridDim.x * 2u) : gridDim.x) {
    const uint32_t which = (r >= out_dim) ? 1u : 0u;
    const uint64_t row = which ? (uint64_t)(r - out_dim) : (uint64_t)r;
    float sum = 0.0f;
    if ((in_dim & 7u) == 0u) {   /* uint4 宽读: 8 half/load(2026-08-20 ②③刀) */
        const uint4 *wr4 = (const uint4 *)((which ? w1 : w0) + row * in_dim);
        const uint64_t n8 = in_dim >> 3;
        for (uint64_t i = tid; i < n8; i += nrow_th) {
            const uint4 v = wr4[i];
            const __half2 *h = (const __half2 *)&v;
            const float4 xa = *(const float4 *)(x + 8u * i);
            const float4 xb = *(const float4 *)(x + 8u * i + 4u);
            sum += __low2float(h[0]) * xa.x + __high2float(h[0]) * xa.y
                 + __low2float(h[1]) * xa.z + __high2float(h[1]) * xa.w
                 + __low2float(h[2]) * xb.x + __high2float(h[2]) * xb.y
                 + __low2float(h[3]) * xb.z + __high2float(h[3]) * xb.w;
        }
    } else {
        const __half2 *wr = (const __half2 *)((which ? w1 : w0) + row * in_dim);
        const uint64_t n2 = in_dim >> 1;
        for (uint64_t i = tid; i < n2; i += nrow_th) {
            const __half2 h = wr[i];
            sum += __low2float(h) * x[2u * i] + __high2float(h) * x[2u * i + 1u];
        }
        if ((in_dim & 1u) && tid == 0)
            sum += __half2float((which ? w1 : w0)[row * in_dim + in_dim - 1u]) * x[in_dim - 1u];
    }
    __shared__ float wsum[8];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31u) == 0) wsum[threadIdx.x >> 5u] = sum;
    __syncthreads();
    if (tid == 0) {
        const uint32_t wbase = pair2 ? ((threadIdx.x >> 7u) * 4u) : 0u;
        const uint32_t wn = pair2 ? 4u : 8u;
        float t = 0.0f;
        for (uint32_t wgi = 0; wgi < wn; wgi++) t += wsum[wbase + wgi];
        (which ? out1 : out0)[row] = t;
    }
    __syncthreads();   /* pair3: 下轮复用 wsum 前全员必须读完 */
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    /* 重写(2026-08-17 调优): 原版每线程读一段连续 chunk ⇒ warp 内 32 线程地址相距
     * chunk 远, 访存完全不合并, 且 32 线程/block 一行 —— 实测仅 ~102GB/s(46%)。
     * 现在: 8 行/block × 32 lane 交错读(__half2 双宽), 合并访存; 行内求和顺序仍是
     * 运行间确定的(lane 交错 + 固定 shuffle 树)。 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= out0_dim && row >= out1_dim) return;
    const __half2 *wr0 = (const __half2 *)(w0 + (row < out0_dim ? row * in_dim : 0));
    const __half2 *wr1 = (const __half2 *)(w1 + (row < out1_dim ? row * in_dim : 0));
    const uint64_t n2 = in_dim >> 1;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint64_t i = lane; i < n2; i += 32u) {
        const float x0 = x[2u * i], x1 = x[2u * i + 1u];
        if (row < out0_dim) {
            const __half2 h = wr0[i];
            sum0 += __low2float(h) * x0 + __high2float(h) * x1;
        }
        if (row < out1_dim) {
            const __half2 h = wr1[i];
            sum1 += __low2float(h) * x0 + __high2float(h) * x1;
        }
    }
    if (in_dim & 1u) {   /* 奇数尾 */
        const uint64_t i = in_dim - 1u;
        if (lane == 0) {
            if (row < out0_dim) sum0 += __half2float(w0[row * in_dim + i]) * x[i];
            if (row < out1_dim) sum1 += __half2float(w1[row * in_dim + i]) * x[i];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, off);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, off);
    }
    if (lane == 0) {
        if (row < out0_dim) out0[row] = sum0;
        if (row < out1_dim) out1[row] = sum1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

