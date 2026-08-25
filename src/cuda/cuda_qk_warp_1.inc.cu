/* cuda_qk_warp_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * q4_K/q2_K warp gemv kernel 族与 dense 入口。
 */
__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    /* 双累加链: 原 isum 单链 16 组 dp4a 序列首尾相接; 拆偶/奇两链交错访存与运算。
     * 整数和无结合律问题, 结果逐位同义。 */
    int isum0 = 0, isum1 = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f;
            const int d1 = sc[is++] & 0x0f;
            isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
            isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 多 token 复用解量化(2026-08-21): 一组 16 个 2-bit 权重只解一次(4 次移位+掩码+4 次
 * shared 读), 批内 NT 个 token 各自 dp4a。批路径此前每个 token 把同一份权重重解一遍 ——
 * verify/drafter 的批就是靠这条路吃掉 ALU 的。整数点积的运算顺序与单 token 版逐位一致,
 * 浮点收尾也保持 (yd*xd)*isum 的左结合, 所以结果与 decode 路完全相同。 */
template <int NT>
__device__ static void dev_dot_q2_K_q8_K_block_smem_multi(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y0, uint64_t ystride, float *acc) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int summs[NT], isum0[NT], isum1[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) { summs[t] = 0; isum0[t] = 0; isum1[t] = 0; }
    #pragma unroll 4
    for (int j = 0; j < 16; j++) {
        const int m = sc[j] >> 4;
        #pragma unroll
        for (int t = 0; t < NT; t++) summs[t] += (y0 + (uint64_t)t * ystride)->bsums[j] * m;
    }
    const uint8_t *q2 = x->qs;
    int is = 0;
    #pragma unroll 1
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        #pragma unroll 1
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f;
            const int d1 = sc[is++] & 0x0f;
            int32_t w0[4], w1[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                w0[i] = (*(const int32_t *)(q2 + i * 4) >> shift) & 0x03030303;
                w1[i] = (*(const int32_t *)(q2 + 16 + i * 4) >> shift) & 0x03030303;
            }
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                const int8_t *q8 = (y0 + (uint64_t)t * ystride)->qs + k * 128 + j * 32;
                int a = 0, b = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    a = __dp4a(w0[i], *(const int32_t *)(q8 + i * 4), a);
                    b = __dp4a(w1[i], *(const int32_t *)(q8 + 16 + i * 4), b);
                }
                isum0[t] += d0 * a;
                isum1[t] += d1 * b;
            }
            shift += 2;
        }
        q2 += 32;
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        const float yd = (y0 + (uint64_t)t * ystride)->d;
        acc[t] += yd * xd * (float)(isum0[t] + isum1[t]) - yd * xmin * (float)summs[t];
    }
}

/* shared 专用低寄存器变体: 与 dev_dot_q2_K_q8_K_block 逐位同义, 仅禁循环展开。
 * 片上读延迟低无需深展开藏延迟; 全展开版把 gateup kernel 顶到 REG:128(occupancy 33%墙)。 */
__device__ static float dev_dot_q2_K_q8_K_block_smem(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    #pragma unroll 4
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int is = 0;
    #pragma unroll 1
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        #pragma unroll 1
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f;
            const int d1 = sc[is++] & 0x0f;
            isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
            isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 半块变体(2026-08-20 dense decode 占用率手术): 一 lane 算 128 权重(h=0/1 取块的前/后半)。
 * q2_K 块 = 2 个 k 迭代 × 128 权重, bsums/scales 按 8 组界干净切分; 整数域逐位同义,
 * 浮点只差 dall/dmin 合帐次序(与 q4 half 变体同级容差)。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_half(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y, uint32_t h) {
    const uint8_t *q2 = x->qs + h * 32u;
    const int8_t *q8 = y->qs + h * 128u;
    const uint8_t *sc = x->scales + h * 8u;
    int summs = 0;
    for (int j = 0; j < 8; j++) summs += y->bsums[h * 8u + j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int is = 0, shift = 0;
    for (int j = 0; j < 4; j++) {
        const int d0 = sc[is++] & 0x0f;
        const int d1 = sc[is++] & 0x0f;
        isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
        isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
        shift += 2;
        q8 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 四分之一块变体(2026-08-20 刀⑤): 一 lane 64 权重(qi∈0..3 = k半×k内前后64)。
 * blocks=8 时 8块×4=32 lane 全活(半块拆分只活16)。scale/bsums 按4组界干净切分。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_quarter(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y, uint32_t qi) {
    const uint32_t k = qi >> 1u, jh = qi & 1u;
    const uint8_t *q2 = x->qs + k * 32u;
    const int8_t *q8 = y->qs + k * 128u + jh * 64u;
    const uint8_t *sc = x->scales + k * 8u + jh * 4u;
    int summs = 0;
    for (int j = 0; j < 4; j++) summs += y->bsums[k * 8u + jh * 4u + j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int shift = (int)(jh * 4u);
    for (int j = 0; j < 2; j++) {
        const int d0 = sc[j * 2] & 0x0f;
        const int d1 = sc[j * 2 + 1] & 0x0f;
        isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
        isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
        shift += 2;
        q8 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 八分之一块变体(2026-08-20 刀⑦): 一 lane 32 权重 = 全量循环固定(k=qi>>2, j=qi&3)一步。
 * blocks=4 时 4块×8=32 lane 全活(半块拆分只活8)。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_eighth(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y, uint32_t qi) {
    const uint32_t k = qi >> 2u, j = qi & 3u;
    const uint8_t *q2 = x->qs + k * 32u;
    const int8_t *q8 = y->qs + k * 128u + j * 32u;
    const uint8_t *sc = x->scales + k * 8u + j * 2u;
    const int summs = y->bsums[k * 8u + j * 2u] * (sc[0] >> 4)
                    + y->bsums[k * 8u + j * 2u + 1u] * (sc[1] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    const int shift = (int)(j * 2u);
    const int isum = (sc[0] & 0x0f) * dev_dot_q2_16(q2, q8, shift)
                   + (sc[1] & 0x0f) * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

/* ---- dense Q4_K matmul (backbone-q4k 配方) ----
 * 引擎的 dense 路径此前只有 q8_0/f16/f32; Q4_K 只在 routed 专家里被支持。
 * backbone(q/kv 投影 + shared + 输出头)换 Q4_K 是把每 token 读取量从 10.3GB 压到
 * ~7.9GB 的关键(decode 带宽墙 26→34 t/s)。核心零件全现成: dev_dot_q4_K_q8_K_block
 * (routed q4k 路在用) + q8_K_quantize_kernel。激活 scratch 用专属 grow-only 显存,
 * 不碰共享 tmp арena(那东西已在 ntok>1 路径的嫌疑名单上)。 */
static void *g_q4k_xq_sc = NULL;
static float *g_down_partial = NULL;   /* [6][out_dim] per-expert 部分和 */
static uint64_t g_down_partial_bytes = 0;
static uint64_t g_q4k_xq_bytes = 0;

/* 向量化取块版 q4_K×q8_K dot: q4_K 块 144B = 9×16B 且行内偏移恒为 16 的倍数
 * (144, 2304 均整除 16) —— 用 uint4 整取进寄存器再算, 取代原 dot 的逐字节 load。
 * 数学与 dev_dot_q4_K_q8_K_block 逐位相同。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_block_vec(
        const cuda_block_q4_K *xg, const cuda_block_q8_K *y) {
    uint4 v[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) v[i] = ((const uint4 *)xg)[i];
    const cuda_block_q4_K *x = (const cuda_block_q4_K *)v;
    return dev_dot_q4_K_q8_K_block(x, y);
}

/* 16-lane 变体: blocks<=16 的小矩阵(attn q/kv/shared, L2 驻留)一 lane 一块零空转;
 * 大矩阵(logits 等)仍走 32-lane 版(A/B: 32-lane 对 DRAM 大矩阵最优)。 */
__global__ static DS4_CUDA_UNUSED void matmul_q4_K_warp16_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w_base + (uint64_t)row * row_bytes);
    const cuda_block_q8_K *xt = xq + (uint64_t)tok * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 16u)
        acc += dev_dot_q4_K_q8_K_block_vec(wr + b, xt + b);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) out[(uint64_t)tok * out_dim + row] = acc;
}

/* q4_K 同输入矩阵对(2026-08-17 第八夜): q_a+kv / shared gate+up 各共读同一激活行,
 * 原两次发射=2 kernel+2 quantize 节点+双份小 grid 调度气泡。合并: 行号跨两矩阵
 * 连续编址, 前段 w0 后段 w1, staging/dot 与单矩阵版同构。 */
__global__ static void matmul_q4_K_pair_warp_kernel(
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out0_dim, uint32_t out1_dim) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t total = out0_dim + out1_dim;
    for (uint32_t rg = blockIdx.x * 8u + warp; rg < total; rg += gridDim.x * 8u) {
        const uint32_t which = (rg >= out0_dim) ? 1u : 0u;
        const uint32_t row = which ? (rg - out0_dim) : rg;
        const char *wb = which ? w1 : w0;
        float acc = 0.0f;
        if (blocks <= 32u) {
            extern __shared__ uint4 dstage[];
            const uint32_t n16 = blocks * 9u;
            const uint4 *src16 = (const uint4 *)(wb + (uint64_t)row * row_bytes);
            uint4 *my = dstage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
            if (blocks <= 16u) {
                const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
                if (bi < blocks) acc += dev_dot_q4_K_q8_K_block_half(wr + bi, xq + bi, hj);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u)
                    acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
            }
            __syncwarp();
        } else {
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(wb + (uint64_t)row * row_bytes);
            for (uint32_t b = lane; b < blocks; b += 32u)
                acc += dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) (which ? out1 : out0)[row] = acc;
    }
}

/* cp.async 双缓冲 staging(2026-08-20 dot微架构刀①): 老 staging 同步搬完才算, 每行
 * 吃一次 LPDDR 全延迟(warp 数藏不住); 现算第 N 行时异步引擎流水搬 N+1 行, 延迟折叠。
 * dot 数学/lane 映射与 staged half/eighth 路同式同序 ⇒ 与现路逐位一致。 */
__global__ static void matmul_q2_K_warp_ca_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t split_maxblk) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t tok = blockIdx.y;
    const cuda_block_q8_K *xt = xq + (uint64_t)tok * blocks;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 q2ca[];                  /* [2][8 warps][n16] */
    uint4 *bufA = q2ca + (uint64_t)warp * n16;
    uint4 *bufB = q2ca + (uint64_t)(8u + warp) * n16;
    const uint32_t rstep = gridDim.x * 8u;
    uint32_t row = blockIdx.x * 8u + warp;
    if (row < out_dim) {
        const uint4 *src = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
        for (uint32_t i = lane; i < n16; i += 32u)
            __pipeline_memcpy_async(bufA + i, src + i, 16);
    }
    __pipeline_commit();
    uint32_t cur = 0u;
    while (row < out_dim) {
        const uint32_t nrow = row + rstep;
        uint4 *bn = cur ? bufA : bufB;
        if (nrow < out_dim) {
            const uint4 *nsrc = (const uint4 *)(w_base + (uint64_t)nrow * row_bytes);
            for (uint32_t i = lane; i < n16; i += 32u)
                __pipeline_memcpy_async(bn + i, nsrc + i, 16);
        }
        __pipeline_commit();
        __pipeline_wait_prior(1);   /* 只剩 1 组在飞 = 本行已就位 */
        __syncwarp();
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(cur ? bufB : bufA);
        float acc = 0.0f;
        if (blocks == 4u && split_maxblk >= 4u) {
            const uint32_t bi = lane >> 3u, qi = lane & 7u;
            acc = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
        } else if (blocks <= 16u && blocks <= split_maxblk) {
            const uint32_t bi = lane >> 1u, h = lane & 1u;
            if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
        } else {
            for (uint32_t b = lane; b < blocks; b += 32u)
                acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(uint64_t)tok * out_dim + row] = acc;
        __syncwarp();               /* 复写本缓冲前全 lane 必须算完 */
        row = nrow; cur ^= 1u;
    }
}

