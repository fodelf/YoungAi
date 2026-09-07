/* q4k_batch_bench_kernels.cuh — q4k_batch_bench.cu 的设备侧核族(单文件 ≤500 行规矩拆出, 2026-09-07)。
 * 只被 q4k_batch_bench.cu 包含; 引擎核的逐字抄本, 只为测速, 引擎(src/cuda/cuda_q4k_*.inc.cu)才是真值。 */
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#define QK_K 256
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } blk_q4_K;   /* 144 B */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } blk_q8_K;             /* 292 B */

__device__ static float dev_f16_to_f32(uint16_t v) { return __half2float(*reinterpret_cast<const __half *>(&v)); }
__device__ static void dev_q4_K_get_scale_min(uint32_t j, const uint8_t *scales, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) { *d_out = scales[j] & 63u; *m_out = scales[j + 4u] & 63u; }
    else { *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u); *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
}
__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}
__device__ static float dev_dot_q4_K_q8_K_block(const blk_q4_K *x, const blk_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d), xmin = dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0, summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0); dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        summs0 += (int)m0 * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)m1 * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        isum0 += (int)sc0 * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, 0);
        isum1 += (int)sc1 * dev_dot_q4_32(x->qs + byte_off, y->qs + (j + 1u) * 32u, 4);
    }
    return y->d * xd * (float)(isum0 + isum1) - y->d * xmin * (float)(summs0 + summs1);
}
template <uint32_t BLOCKS>
__device__ __forceinline__ static float seg_sum(float acc) {
    #pragma unroll
    for (uint32_t off = BLOCKS / 2u; off > 0u; off >>= 1u) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

/* 参考: 引擎单 token tile 核(逐字), 逐 token 各发一次 */
template <uint32_t BLOCKS>
__global__ static void k_tile1(float *out, const char *w, const blk_q8_K *xq, uint32_t out_dim) {
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 st[];
    uint4 *my = st + (uint64_t)warp * n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)t * R * BLOCKS * 144u);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        float acc = dev_dot_q4_K_q8_K_block((const blk_q4_K *)my + rin * BLOCKS + b, xq + b);
        acc = seg_sum<BLOCKS>(acc);
        if (b == 0u) out[t * R + rin] = acc;
        __syncwarp();
    }
}
/* 现役多 token 核(引擎 q4k_tile_multi_kernel 逐字) */
template <uint32_t BLOCKS>
__global__ static void k_multi(float *out, const char *w, const blk_q8_K *xq, uint32_t out_dim, uint32_t n_tok) {
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 st[];
    uint4 *my = st + (uint64_t)warp * n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)t * R * BLOCKS * 144u);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my + rin * BLOCKS + b;
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = dev_dot_q4_K_q8_K_block(wr, xq + (uint64_t)tk * BLOCKS + b);
            acc = seg_sum<BLOCKS>(acc);
            if (b == 0u) out[(uint64_t)tk * out_dim + t * R + rin] = acc;
        }
        __syncwarp();
    }
}
/* 候选①: cp.async 双缓冲 —— 本 tile 算 n_tok 趟时, 下一 tile 已在往备用段搬 */
template <uint32_t BLOCKS>
__global__ static void k_multi_db(float *out, const char *w, const blk_q8_K *xq, uint32_t out_dim, uint32_t n_tok) {
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 st[];
    uint4 *buf0 = st + (uint64_t)warp * 2u * n16, *buf1 = buf0 + n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R, stride = gridDim.x * 8u;
    uint32_t t = blockIdx.x * 8u + warp;
    if (t >= ntiles) return;
    auto issue = [&](uint4 *dst, uint32_t tile) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)tile * R * BLOCKS * 144u);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) __pipeline_memcpy_async(dst + i, src + i, 16);
        __pipeline_commit();
    };
    issue(buf0, t);
    uint4 *cur = buf0, *nxt = buf1;
    for (; t < ntiles; t += stride) {
        const uint32_t tn = t + stride;
        if (tn < ntiles) { issue(nxt, tn); __pipeline_wait_prior(1); } else { __pipeline_wait_prior(0); }
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)cur + rin * BLOCKS + b;
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = dev_dot_q4_K_q8_K_block(wr, xq + (uint64_t)tk * BLOCKS + b);
            acc = seg_sum<BLOCKS>(acc);
            if (b == 0u) out[(uint64_t)tk * out_dim + t * R + rin] = acc;
        }
        __syncwarp();
        uint4 *tmp = cur; cur = nxt; nxt = tmp;
    }
}
/* 候选②: 激活先进 shared(n_tok×BLOCKS 块 292 B), 免每趟从 L1/L2 取; 其余同现役 */
template <uint32_t BLOCKS>
__global__ static void k_multi_xs(float *out, const char *w, const blk_q8_K *xq, uint32_t out_dim, uint32_t n_tok) {
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 st[];
    blk_q8_K *xs = (blk_q8_K *)(st + 8u * n16);
    for (uint32_t i = threadIdx.x; i < n_tok * BLOCKS * (uint32_t)sizeof(blk_q8_K) / 4u; i += blockDim.x)
        ((uint32_t *)xs)[i] = ((const uint32_t *)xq)[i];
    __syncthreads();
    uint4 *my = st + (uint64_t)warp * n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)t * R * BLOCKS * 144u);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my + rin * BLOCKS + b;
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = dev_dot_q4_K_q8_K_block(wr, xs + (uint64_t)tk * BLOCKS + b);
            acc = seg_sum<BLOCKS>(acc);
            if (b == 0u) out[(uint64_t)tk * out_dim + t * R + rin] = acc;
        }
        __syncwarp();
    }
}

/* 引擎 q2k_fused_quantize 逐字(cuda_qk_warp_2): block 内并行 q8_K 量化, 与 q8_K_quantize_kernel 逐位同义 */
__device__ static void fused_quantize(blk_q8_K *xqsh, const float *x, uint32_t blocks) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (uint32_t b = warp; b < blocks; b += 8u) {
        const float *xr = x + (uint64_t)b * 256u;
        blk_q8_K *yb = xqsh + b;
        float amax = 0.0f, maxv = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) { const float v = xr[lane * 8u + i]; const float a = fabsf(v); if (a > amax) { amax = a; maxv = v; } }
        for (int o = 16; o; o >>= 1) {
            const float oa = __shfl_down_sync(0xffffffffu, amax, o);
            const float ov = __shfl_down_sync(0xffffffffu, maxv, o);
            if (oa > amax) { amax = oa; maxv = ov; }
        }
        amax = __shfl_sync(0xffffffffu, amax, 0); maxv = __shfl_sync(0xffffffffu, maxv, 0);
        if (amax == 0.0f) {
            if (lane == 0) yb->d = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; i++) yb->qs[lane * 8u + i] = 0;
            if (lane < 16u) yb->bsums[lane] = 0;
            continue;
        }
        const float iscale = -127.0f / maxv;
        int psum = 0;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            int qv = (int)lrintf(iscale * xr[lane * 8u + i]);
            if (qv > 127) qv = 127; if (qv < -128) qv = -128;
            yb->qs[lane * 8u + i] = (int8_t)qv; psum += qv;
        }
        const int osum = __shfl_down_sync(0xffffffffu, psum, 1);
        if ((lane & 1u) == 0u) yb->bsums[lane >> 1] = (int16_t)(psum + osum);
        if (lane == 0) yb->d = 1.0f / iscale;
    }
}
/* 候选③: 激活量化融进核(每 block 自己把 n_tok×BLOCKS 块 f32 激活量成 q8_K 进 shared), 省掉 q8_K_quantize 小核一发 */
template <uint32_t BLOCKS>
__global__ static void k_multi_fq(float *out, const char *w, const float *x, uint32_t out_dim, uint32_t n_tok) {
    constexpr uint32_t R = 32u / BLOCKS, n16 = R * BLOCKS * 9u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    extern __shared__ uint4 st[];
    blk_q8_K *xs = (blk_q8_K *)(st + 8u * n16);
    fused_quantize(xs, x, n_tok * BLOCKS);
    __syncthreads();
    uint4 *my = st + (uint64_t)warp * n16;
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS, ntiles = out_dim / R;
    for (uint32_t t = blockIdx.x * 8u + warp; t < ntiles; t += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)t * R * BLOCKS * 144u);
        #pragma unroll
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my + rin * BLOCKS + b;
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = dev_dot_q4_K_q8_K_block(wr, xs + (uint64_t)tk * BLOCKS + b);
            acc = seg_sum<BLOCKS>(acc);
            if (b == 0u) out[(uint64_t)tk * out_dim + t * R + rin] = acc;
        }
        __syncwarp();
    }
}
/* 参考量化核(引擎 q8_K_quantize_kernel 逐字) */
__global__ static void k_q8_quant(blk_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x, row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * QK_K;
    blk_q8_K *yb = out + (uint64_t)row * (in_dim / QK_K) + b;
    __shared__ float abs_part[256]; __shared__ float val_part[256];
    uint32_t tid = threadIdx.x;
    float v = tid < QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < QK_K ? fabsf(v) : 0.0f; val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) { abs_part[tid] = abs_part[tid + stride]; val_part[tid] = val_part[tid + stride]; }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) { if (tid == 0) yb->d = 0.0f; if (tid < QK_K) yb->qs[tid] = 0; if (tid < QK_K / 16) yb->bsums[tid] = 0; return; }
    const float maxv = val_part[0], iscale = -127.0f / maxv;
    __shared__ int qsh[256];
    int qv = 0;
    if (tid < QK_K) { qv = (int)lrintf(iscale * v); if (qv > 127) qv = 127; if (qv < -128) qv = -128; yb->qs[tid] = (int8_t)qv; }
    qsh[tid] = qv; __syncthreads();
    if (tid < QK_K / 16) { int s = 0; for (int i = 0; i < 16; i++) s += qsh[tid * 16 + i]; yb->bsums[tid] = (int16_t)s; }
    if (tid == 0) yb->d = 1.0f / iscale;
}

static void fill_rand(uint8_t *p, size_t n, uint32_t seed) { uint32_t s = seed; for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; p[i] = (uint8_t)(s >> 24); } }
/* ---- out_b 形状(4096 行 × 32 块, q8_0 激活口径): 现役 q8_0 dot(引擎 dev_dot_q4_K_q8_0x8_smem 逐字) vs 预解码版 ---- */
__device__ __forceinline__ static float dot_q8_0_cur(const blk_q4_K *x, const int8_t *xq, const float *xs) {
    const float d = dev_f16_to_f32(x->d), dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0); dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u, *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0), dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) { s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80); s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81); }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}
struct wblk { float d, dmin; uint8_t sc[8], mn[8]; };
__device__ __forceinline__ static float dot_q8_0_pre(const blk_q4_K *x, const wblk &wb, const int8_t *xq, const float *xs, const int32_t *s8) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, xq + j * 32u, 0), dot1 = dev_dot_q4_32(x->qs + byte_off, xq + (j + 1u) * 32u, 4);
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s8[j]);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s8[j + 1u]);
    }
    return acc0 + acc1;
}
/* 现役: 整行 stage, lane 各一整块, 逐 token 现役 dot(激活从全局) */
__global__ static void k_rows8_cur(float *out, const char *w, const int8_t *xq, const float *xs, uint32_t kb, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, n16 = kb * 9u;
    extern __shared__ uint4 st[];
    uint4 *my = st + (uint64_t)warp * n16;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)row * kb * 144u);
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my;
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = 0.0f;
            for (uint32_t b = lane; b < kb; b += 32u) acc += dot_q8_0_cur(wr + b, xq + ((uint64_t)tk * kb + b) * 256u, xs + ((uint64_t)tk * kb + b) * 8u);
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
        }
        __syncwarp();
    }
}
/* 候选: 激活+尺度+Σq8 进 shared, 权重块尺度解码一次, 逐 token 只剩 dp4a */
__global__ static void k_rows8_pre(float *out, const char *w, const int8_t *xq, const float *xs, uint32_t kb, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, n16 = kb * 9u;
    extern __shared__ uint4 st[];
    uint4 *my = st + (uint64_t)warp * n16;
    uint32_t *dst = (uint32_t *)(st + 8u * n16);
    const uint32_t nq = n_tok * kb * 64u, ns = n_tok * kb * 8u;
    for (uint32_t i = threadIdx.x; i < nq; i += blockDim.x) dst[i] = ((const uint32_t *)xq)[i];
    for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) ((float *)dst)[nq + i] = xs[i];
    int32_t *s8 = (int32_t *)dst + nq + ns;
    __syncthreads();
    const int8_t *xqa = (const int8_t *)dst; const float *xsa = (const float *)dst + nq;
    for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) {
        const int8_t *q8 = xqa + (uint64_t)i * 32u; int32_t s = 0;
        #pragma unroll
        for (uint32_t k = 0; k < 32u; k += 4u) s = __dp4a(0x01010101, *(const int32_t *)(q8 + k), s);
        s8[i] = s;
    }
    __syncthreads();
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)row * kb * 144u);
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my;
        wblk wb; const uint32_t b = lane;
        if (b < kb) { wb.d = dev_f16_to_f32(wr[b].d); wb.dmin = dev_f16_to_f32(wr[b].dmin);
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) dev_q4_K_get_scale_min(j, wr[b].scales, &wb.sc[j], &wb.mn[j]); }
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = 0.0f;
            if (b < kb) { const uint64_t bi = (uint64_t)tk * kb + b; acc = dot_q8_0_pre(wr + b, wb, xqa + bi * 256u, xsa + bi * 8u, s8 + bi * 8u); }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
        }
        __syncwarp();
    }
}
/* 引擎发射形态复现: PDL 属性发射 + 核首 cudaGridDependencySynchronize(), 前面接一个小核当前序 */
__global__ static void k_dummy(float *o) { if (threadIdx.x == 0 && blockIdx.x == 0) o[0] += 1.0f; }
__global__ static void k_rows8_pre_pdl(float *out, const char *w, const int8_t *xq, const float *xs, uint32_t kb, uint32_t out_dim, uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, n16 = kb * 9u;
    extern __shared__ uint4 st[];
    uint4 *my = st + (uint64_t)warp * n16;
    uint32_t *dst = (uint32_t *)(st + 8u * n16);
    const uint32_t nq = n_tok * kb * 64u, ns = n_tok * kb * 8u;
    for (uint32_t i = threadIdx.x; i < nq; i += blockDim.x) dst[i] = ((const uint32_t *)xq)[i];
    for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) ((float *)dst)[nq + i] = xs[i];
    int32_t *s8 = (int32_t *)dst + nq + ns;
    __syncthreads();
    const int8_t *xqa = (const int8_t *)dst; const float *xsa = (const float *)dst + nq;
    for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) {
        const int8_t *q8 = xqa + (uint64_t)i * 32u; int32_t s = 0;
        #pragma unroll
        for (uint32_t k = 0; k < 32u; k += 4u) s = __dp4a(0x01010101, *(const int32_t *)(q8 + k), s);
        s8[i] = s;
    }
    __syncthreads();
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src = (const uint4 *)(w + (uint64_t)row * kb * 144u);
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my;
        wblk wb; const uint32_t b = lane;
        if (b < kb) { wb.d = dev_f16_to_f32(wr[b].d); wb.dmin = dev_f16_to_f32(wr[b].dmin);
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) dev_q4_K_get_scale_min(j, wr[b].scales, &wb.sc[j], &wb.mn[j]); }
        for (uint32_t tk = 0; tk < n_tok; tk++) {
            float acc = 0.0f;
            if (b < kb) { const uint64_t bi = (uint64_t)tk * kb + b; acc = dot_q8_0_pre(wr + b, wb, xqa + bi * 256u, xsa + bi * 8u, s8 + bi * 8u); }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
        }
        __syncwarp();
    }
}
static void launch_pdl_rows8(float *out, const char *w, const int8_t *xq, const float *xs, uint32_t kb, uint32_t out_dim, uint32_t n_tok, unsigned gx, size_t shm) {
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(gx); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = shm; cfg.stream = 0;
    cudaLaunchAttribute attr[1]; attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, k_rows8_pre_pdl, out, w, xq, xs, kb, out_dim, n_tok));
}
/* ---- 引擎核逐字搬进来(改名): 排查"同核微基准 164 µs / 引擎 383 µs" ---- */
struct act_q8_0_e {
    const int8_t *xq; const float *xs;
    __device__ __forceinline__ act_q8_0_e at(uint64_t blk_off) const { act_q8_0_e a; a.xq = xq + blk_off * 256u; a.xs = xs + blk_off * 8u; return a; }
    __device__ __forceinline__ float dot(const blk_q4_K *w, uint32_t b) const { return dot_q8_0_cur(w, xq + (uint64_t)b * 256u, xs + (uint64_t)b * 8u); }
};
__device__ __forceinline__ static void wblk_decode_e(const blk_q4_K *x, wblk *o) {
    o->d = dev_f16_to_f32(x->d); o->dmin = dev_f16_to_f32(x->dmin);
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) dev_q4_K_get_scale_min(j, x->scales, &o->sc[j], &o->mn[j]);
}
__global__ static void k_rows8_engine(float *out, const char *w, const int8_t *xq, const float *xs,
                                                  uint32_t kblocks, uint32_t out_dim, uint32_t n_tok, int stage_x) {
    /* 只等不触发(09-07): 本核 77 KB shared ⇒ 每 SM 只驻 1 block, 若一开场就触发, 后继核的 block 提前上 SM 停在等待里
     * 占住线程槽/调度, 8 波串行的本核被拖到 383 µs(微基准同核无后继 163 µs)。不触发 = 后继核等本核整体完成再发。 */
    
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, n16 = kblocks * 9u;
    extern __shared__ uint4 st[];
    uint4 *my = st + (uint64_t)warp * n16;
    const int8_t *xqa = xq; const float *xsa = xs;
    /* Σq8 表 s8[(tok*kblocks+b)*8+j](stage_x 时与激活一起进 shared; 否则算在寄存器里逐 token 现算) */
    int32_t *s8 = NULL;
    if (stage_x) {   /* q8_0 激活: 值 n_tok×kblocks×256 B + 尺度 n_tok×kblocks×8 f32 + Σq8 n_tok×kblocks×8 i32 */
        uint32_t *dst = (uint32_t *)(st + 8u * n16);
        const uint32_t nq = n_tok * kblocks * 64u, ns = n_tok * kblocks * 8u;
        for (uint32_t i = threadIdx.x; i < nq; i += blockDim.x) dst[i] = ((const uint32_t *)xq)[i];
        for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) ((float *)dst)[nq + i] = xs[i];
        s8 = (int32_t *)dst + nq + ns;
        __syncthreads();
        xqa = (const int8_t *)dst; xsa = (const float *)dst + nq;
        for (uint32_t i = threadIdx.x; i < ns; i += blockDim.x) {   /* 子块 (tok,块,j) 的 32 值和: 8 个 dp4a */
            const int8_t *q8 = xqa + (uint64_t)i * 32u;
            int32_t s = 0;
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k += 4u) s = __dp4a(0x01010101, *(const int32_t *)(q8 + k), s);
            s8[i] = s;
        }
        __syncthreads();
    }
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint4 *src16 = (const uint4 *)(w + (uint64_t)row * kblocks * sizeof(blk_q4_K));
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const blk_q4_K *wr = (const blk_q4_K *)my;
        if (stage_x && kblocks <= 32u) {   /* lane 各一整块: 尺度解码一次, 逐 token 只剩 dp4a */
            wblk wb; const uint32_t b = lane;
            if (b < kblocks) wblk_decode_e(wr + b, &wb);
            for (uint32_t tk = 0; tk < n_tok; tk++) {
                float acc = 0.0f;
                if (b < kblocks) {
                    const uint64_t bi = (uint64_t)tk * kblocks + b;
                    acc = dot_q8_0_pre(wr + b, wb, xqa + bi * 256u, xsa + bi * 8u, s8 + bi * 8u);
                }
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
            }
        } else {
            act_q8_0_e xbase; xbase.xq = xqa; xbase.xs = xsa;
            for (uint32_t tk = 0; tk < n_tok; tk++) {
                const act_q8_0_e x = xbase.at((uint64_t)tk * kblocks);
                float acc = 0.0f;
                for (uint32_t b = lane; b < kblocks; b += 32u) acc += x.dot(wr + b, b);
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)tk * out_dim + row] = acc;
            }
        }
        __syncwarp();
    }
}
