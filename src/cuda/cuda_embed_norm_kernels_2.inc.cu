/* cuda_embed_norm_kernels_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * embed/corr/repeat/norm/matmul(f16/f32) kernel 族 + kv fp8 rope store。
 */
__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
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

__device__ __forceinline__ static float q8r_row_dot(
        const __half *scales, const int8_t *qs, uint64_t row, uint64_t blocks,
        const int8_t *xq, const float *xscale, uint32_t lane) {
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int4 xv0 = *(const int4 *)(xq + b * 32u);
        const int4 xv1 = *(const int4 *)(xq + b * 32u + 16u);
        const int4 w0 = *(const int4 *)(qs + (row * blocks + b) * 32u);
        const int4 w1 = *(const int4 *)(qs + (row * blocks + b) * 32u + 16u);
        int32_t d = 0;
        d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
        d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
        d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
        d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
        acc += __half2float(scales[row * blocks + b]) * xscale[b] * (float)d;
    }
    return acc;
}

__global__ static void matmul_q8r_pair_warp8_kernel(
        float *out0, float *out1,
        const __half *sc0, const int8_t *q0,
        const __half *sc1, const int8_t *q1,
        const int8_t *xq, const float *xscale,
        uint64_t out0_dim, uint64_t out1_dim, uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float a0 = 0.0f, a1 = 0.0f;
    if (row < out0_dim) a0 = q8r_row_dot(sc0, q0, row, blocks, xq, xscale, lane);
    if (row < out1_dim) a1 = q8r_row_dot(sc1, q1, row, blocks, xq, xscale, lane);
    a0 = warp_sum_f32(a0);
    a1 = warp_sum_f32(a1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = a0;
        if (row < out1_dim) out1[row] = a1;
    }
}

__global__ static void matmul_q8r_hc_expand_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint32_t n_embd, uint32_t n_hc, uint64_t blocks, int has_add) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    float acc = q8r_row_dot(scales, qs, row, blocks, xq, xscale, lane);
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void grouped_q8r_a_kernel(
        float *low, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t rank, uint32_t n_groups, uint32_t n_tokens, uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    float acc = q8r_row_dot(scales, qs, row, blocks,
                            xq + xrow * blocks * 32u, xscale + xrow * blocks, lane);
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__device__ __forceinline__ static float q8r_seg_dot(
        const __half *scales, const int8_t *qs, uint64_t row, uint64_t blocks,
        const int8_t *xq, const float *xscale, uint64_t seg_beg, uint64_t seg_end, uint32_t lane) {
    float acc = 0.0f;
    for (uint64_t b = seg_beg + lane; b < seg_end; b += 32u) {
        const int4 xv0 = *(const int4 *)(xq + b * 32u);
        const int4 xv1 = *(const int4 *)(xq + b * 32u + 16u);
        const int4 w0 = *(const int4 *)(qs + (row * blocks + b) * 32u);
        const int4 w1 = *(const int4 *)(qs + (row * blocks + b) * 32u + 16u);
        int32_t d = 0;
        d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
        d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
        d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
        d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
        acc += __half2float(scales[row * blocks + b]) * xscale[b] * (float)d;
    }
    return acc;
}

/* splitK gemv: 8 warps = 2 行 × 4 段。行数少的 decode gemv 每行单 warp 时
 * scheduler 无候选可发(ncu: No Eligible 96%, 活跃 warp 9.7/调度器) —— 行内
 * 4 段并行把在飞 warp 数 ×4。段序固定 ⇒ 归约顺序恒定, 运行间确定。 */
__global__ static void matmul_q8r_sk_kernel(
        float *out, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale, uint64_t out_dim, uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part[2][4];
    float acc = 0.0f;
    if (row < out_dim) {
        const uint64_t sb = (blocks * seg) >> 2u;
        const uint64_t se = (blocks * (seg + 1u)) >> 2u;
        acc = q8r_seg_dot(scales, qs, row, blocks, xq, xscale, sb, se, lane);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) part[rsub][seg] = acc;
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < out_dim)
            out[r2] = part[lane][0] + part[lane][1] + part[lane][2] + part[lane][3];
    }
}

__global__ static void matmul_q8r_sk_pair_kernel(
        float *out0, float *out1,
        const __half *sc0, const int8_t *q0,
        const __half *sc1, const int8_t *q1,
        const int8_t *xq, const float *xscale,
        uint64_t out0_dim, uint64_t out1_dim, uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part0[2][4], part1[2][4];
    float a0 = 0.0f, a1 = 0.0f;
    const uint64_t sb = (blocks * seg) >> 2u;
    const uint64_t se = (blocks * (seg + 1u)) >> 2u;
    if (row < out0_dim) a0 = q8r_seg_dot(sc0, q0, row, blocks, xq, xscale, sb, se, lane);
    if (row < out1_dim) a1 = q8r_seg_dot(sc1, q1, row, blocks, xq, xscale, sb, se, lane);
    a0 = warp_sum_f32(a0);
    a1 = warp_sum_f32(a1);
    if (lane == 0) { part0[rsub][seg] = a0; part1[rsub][seg] = a1; }
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < out0_dim) out0[r2] = part0[lane][0] + part0[lane][1] + part0[lane][2] + part0[lane][3];
        if (r2 < out1_dim) out1[r2] = part1[lane][0] + part1[lane][1] + part1[lane][2] + part1[lane][3];
    }
}

__global__ static void matmul_q8r_sk_hc_expand_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint32_t n_embd, uint32_t n_hc, uint64_t blocks, int has_add) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part[2][4];
    float acc = 0.0f;
    if (row < out_dim) {
        const uint64_t sb = (blocks * seg) >> 2u;
        const uint64_t se = (blocks * (seg + 1u)) >> 2u;
        acc = q8r_seg_dot(scales, qs, row, blocks, xq, xscale, sb, se, lane);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) part[rsub][seg] = acc;
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < out_dim) {
            const float a = part[lane][0] + part[lane][1] + part[lane][2] + part[lane][3];
            const uint32_t d = (uint32_t)r2;
            block_out[d] = a;
            float block_v = a;
            if (has_add) block_v += block_add[d];
            const float *post = split + n_hc;
            const float *comb = split + 2u * n_hc;
            for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
                float hc_acc = block_v * post[dst_hc];
                for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                    const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                    const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                    hc_acc += comb_v * res_v;
                }
                out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
            }
        }
    }
}

__global__ static void grouped_q8r_sk_a_kernel(
        float *low, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t rank, uint32_t n_groups, uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part[2][4];
    float acc = 0.0f;
    if (row < low_dim) {
        const uint64_t group = row / rank;
        const uint64_t sb = (blocks * seg) >> 2u;
        const uint64_t se = (blocks * (seg + 1u)) >> 2u;
        acc = q8r_seg_dot(scales, qs, row, blocks,
                          xq + group * blocks * 32u, xscale + group * blocks, sb, se, lane);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) part[rsub][seg] = acc;
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < low_dim) low[r2] = part[lane][0] + part[lane][1] + part[lane][2] + part[lane][3];
    }
}

__global__ static void matmul_q8r_warp8_kernel(
        float *out, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint64_t blocks) {
    /* repacked q8 gemv: scale/qs 平面分离, 每块 2 次 int4(128bit) 对齐读代替
     * 8 次非对齐 4B 读; 双行互填延迟。dp4a 求和顺序与原 kernel 相同 ⇒ bit 级不变。 */
    const uint64_t row0 = (uint64_t)blockIdx.x * 16u + (uint64_t)(threadIdx.x >> 5u) * 2u;
    const uint64_t row1 = row0 + 1u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const int has1 = row1 < out_dim;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int4 xv0 = *(const int4 *)(xq + b * 32u);
        const int4 xv1 = *(const int4 *)(xq + b * 32u + 16u);
        const float xs = xscale[b];
        {
            const int4 w0 = *(const int4 *)(qs + (row0 * blocks + b) * 32u);
            const int4 w1 = *(const int4 *)(qs + (row0 * blocks + b) * 32u + 16u);
            int32_t d = 0;
            d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
            d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
            d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
            d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
            acc0 += __half2float(scales[row0 * blocks + b]) * xs * (float)d;
        }
        if (has1) {
            const int4 w0 = *(const int4 *)(qs + (row1 * blocks + b) * 32u);
            const int4 w1 = *(const int4 *)(qs + (row1 * blocks + b) * 32u + 16u);
            int32_t d = 0;
            d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
            d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
            d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
            d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
            acc1 += __half2float(scales[row1 * blocks + b]) * xs * (float)d;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row0] = acc0;
        if (has1) out[row1] = acc1;
    }
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    /* 每 warp 双行: 两条 dp4a 累加链交替发射互填 L1 延迟(单链版 ncu SM 利用仅
     * 9-17%), xq/xscale 读两行共享。每行的求和顺序与单行版完全相同 ⇒ bit 级不变。 */
    const uint64_t row0 = (uint64_t)blockIdx.x * 16u + (uint64_t)(threadIdx.x >> 5u) * 2u;
    const uint64_t row1 = row0 + 1u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const unsigned char *wr0 = w + row0 * blocks * 34;
    const unsigned char *wr1 = (row1 < out_dim) ? w + row1 * blocks * 34 : NULL;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        const __half *s0 = (const __half *)(wr0 + b * 34);
        const int dot0 = dot_i8_block((const int8_t *)(wr0 + b * 34 + 2), xqb, bn, use_dp4a);
        acc0 += __half2float(*s0) * xs * (float)dot0;
        if (wr1) {
            const __half *s1 = (const __half *)(wr1 + b * 34);
            const int dot1 = dot_i8_block((const int8_t *)(wr1 + b * 34 + 2), xqb, bn, use_dp4a);
            acc1 += __half2float(*s1) * xs * (float)dot1;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row0] = acc0;
        if (wr1) out[row1] = acc1;
    }
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

