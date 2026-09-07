/* cuda_api_attention_5.inc.cu — 批路静态注意力的 GEMM 版(2026-09-06, 1M 上下文战役)。
 *
 * 用途: prefill 批里非 top-k 的层(ratio-128 层, 以及 top-k 未启用的 ratio-4 层)对全部可见压缩行做注意力。
 * 老核 attention_decode_mixed_heads8_online 每 token 一个 block, 把全部压缩行(1M 时 ~8192 行 × 2 KB f32)从 L2/显存
 * 重搬一遍: 2048 token 一层搬 34 GB ⇒ 420 ms/层块, 5.9 TFLOPS(1M 点剖面 29%, 第一大项)。
 * 这里压缩行部分做成两发 cuBLAS f16 GEMM: 每片 128 token × 64 头 = 8192 行, S = scale·Q·Kᵀ(对全部 n_comp 行),
 * 行 softmax 核按 token 因果可见界掩、出未归一化 P = exp(S−m)(f16)与 (m_c, l_c), O_c = P·V(V 就是同一份压缩行);
 * 原始窗口 + sink 部分沿用老核的行选择与在线 softmax, 单独出未归一化 (o_r, m_r, l_r); 合并核按 online-softmax
 * 规则归一。数学与老核同式; 数值上压缩行部分 Q/K/P 走 f16(老核 f32 点积), 判五指标同带(prefill 模式)。
 * 改了会怎样: 片做大 S 暂存按 n_comp 线性长(8192 × 8192 f32 = 268 MB); 掩码若不按 token 逐行算会把未来压缩行
 * 算进去; 合并若忘了 sink, 与老核的 sum_s 差一项。 */
#define DS4_ASG_TILE_TOKENS 128u
#define DS4_ASG_RAW_STRIDE 516u   /* o_r[512] + m_r + l_r + 2 pad: 行距 2064 B 保 float4 对齐(514 会 misaligned address, 09-06 实撞) */

static struct {
    __half *q16; uint64_t q16_cap;   /* [n_tokens·64][512] */
    __half *k16; uint64_t k16_cap;   /* [n_comp][512] f16 影子(K 与 V 同一份): 缓存行是 [448 f16][64 f32], cuBLAS 要整行 f16 */
    float  *s;   uint64_t s_cap;     /* [8192][n_comp] */
    __half *p;   uint64_t p_cap;     /* [8192][n_comp] */
    float  *oc;  uint64_t oc_cap;    /* [8192][512] */
    float  *ml;  uint64_t ml_cap;    /* [8192][2] */
    float  *raw; uint64_t raw_cap;   /* [n_tokens·64][514] */
} g_asg;

static int asg_grow(void **p, uint64_t *cap, uint64_t need, size_t elem, const char *what) {
    if (need <= *cap) return 1;
    (void)cudaDeviceSynchronize();   /* 旧块可能仍被在飞 kernel 读 */
    if (*p) (void)cudaFree(*p);
    *p = NULL; *cap = 0;
    if (cudaMalloc(p, need * elem) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [attn-gemm] %s 分配失败 (%.1f MB)\n", what, (double)need * elem / 1048576.0);
        return 0;
    }
    *cap = need;
    return 1;
}

/* 行 softmax: 行 r ↔ (token t0 + r/64, 头 r%64); 可见界 = (pos0+t+1)/ratio 截 n_comp(同老核 comp_count) */
__global__ static void asg_softmax_kernel(float *ml, __half *p, const float *s, uint32_t n_comp,
                                          uint32_t t0, uint32_t pos0, uint32_t ratio) {
    const uint32_t r = blockIdx.x, tid = threadIdx.x;
    const uint32_t t = t0 + (r >> 6u);
    uint32_t visible = (pos0 + t + 1u) / ratio;
    if (visible > n_comp) visible = n_comp;
    const float *row = s + (uint64_t)r * n_comp;
    __half *prow = p + (uint64_t)r * n_comp;
    __shared__ float red[256];
    float m = -INFINITY;
    for (uint32_t c = tid; c < visible; c += 256u) m = fmaxf(m, row[c]);
    red[tid] = m; __syncthreads();
    for (uint32_t st = 128u; st > 0u; st >>= 1u) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float l = 0.0f;
    for (uint32_t c = tid; c < n_comp; c += 256u) {
        float e = 0.0f;
        if (c < visible) { e = expf(row[c] - m); l += e; }
        prow[c] = __float2half(e);
    }
    red[tid] = l; __syncthreads();
    for (uint32_t st = 128u; st > 0u; st >>= 1u) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    if (tid == 0) { ml[(uint64_t)r * 2u] = m; ml[(uint64_t)r * 2u + 1u] = red[0]; }
}

/* 原始窗口 + sink 的未归一化部分: block = token × 8 头组, warp = 头; 行选择与老核逐字同 */
__global__ static void asg_raw_partial_kernel(
        float *raw_out, const float *sinks, const float *q, const float *raw_kv,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t window, uint32_t n_head, float scale) {
    const uint32_t t = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * 8u + warp;
    if (t >= n_tokens) return;
    const bool valid_head = head < n_head;
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ const float4 *row_src[4];
    __shared__ float4 kv_shared[4 * 128];
    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0, raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) { const uint32_t wlo = qpos + 1u - window; if (wlo > lo) lo = wlo; }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) { raw_first_idx = lo - first_raw_pos; raw_count = hi - lo + 1u; if (raw_count > 256u) raw_count = 256u; }
            }
        }
        raw_count_s = raw_count;
        for (uint32_t r = 0; r < raw_count; r++) raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const float4 *q4 = valid_head ? (const float4 *)(q + ((uint64_t)t * n_head + head) * 512u) : NULL;
    float4 q0 = make_float4(0.f, 0.f, 0.f, 0.f), q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) { q0 = q4[lane]; q1 = q4[lane + 32u]; q2 = q4[lane + 64u]; q3 = q4[lane + 96u]; }
    float max_s = -INFINITY, sum_s = 0.0f;
    float4 o0 = make_float4(0.f, 0.f, 0.f, 0.f), o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < raw_count; row0 += 4u) {
        const uint32_t nr = raw_count - row0 < 4u ? raw_count - row0 : 4u;
        if (threadIdx.x < nr) row_src[threadIdx.x] = (const float4 *)(raw_kv + (uint64_t)raw_rows[row0 + threadIdx.x] * 512u);
        __syncthreads();
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) kv_shared[off] = row_src[off >> 7u][off & 127u];
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane], k1 = kv4[lane + 32u], k2 = kv4[lane + 64u], k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) + dot4_f32(q1, k1) + dot4_f32(q2, k2) + dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m), row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale; o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale; o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale; o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale; o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale; o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale; o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale; o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale; o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }
    if (!valid_head) return;
    const float sink = sinks[head];
    const float new_m = fmaxf(max_s, sink);
    const float old_scale = expf(max_s - new_m);   /* raw_count==0 时 max_s=-inf ⇒ 0 */
    sum_s = sum_s * old_scale + expf(sink - new_m);
    float4 *dst = (float4 *)(raw_out + ((uint64_t)t * n_head + head) * DS4_ASG_RAW_STRIDE);
    dst[lane] = make_float4(o0.x * old_scale, o0.y * old_scale, o0.z * old_scale, o0.w * old_scale);
    dst[lane + 32u] = make_float4(o1.x * old_scale, o1.y * old_scale, o1.z * old_scale, o1.w * old_scale);
    dst[lane + 64u] = make_float4(o2.x * old_scale, o2.y * old_scale, o2.z * old_scale, o2.w * old_scale);
    dst[lane + 96u] = make_float4(o3.x * old_scale, o3.y * old_scale, o3.z * old_scale, o3.w * old_scale);
    if (lane == 0) { raw_out[((uint64_t)t * n_head + head) * DS4_ASG_RAW_STRIDE + 512u] = new_m;
                     raw_out[((uint64_t)t * n_head + head) * DS4_ASG_RAW_STRIDE + 513u] = sum_s; }
}

/* 合并: heads[R] = (o_c·a + o_r·b) / (l_c·a + l_r·b), a = e^{m_c−m}, b = e^{m_r−m}, m = max */
__global__ static void asg_combine_kernel(float *heads, const float *oc, const float *ml, const float *raw, uint32_t r0) {
    const uint32_t r = blockIdx.x, d = threadIdx.x * 4u;   /* 128 线程 × 4 = 512 */
    const uint64_t R = (uint64_t)r0 + r;
    const float mc = ml[(uint64_t)r * 2u], lc = ml[(uint64_t)r * 2u + 1u];
    const float *rr = raw + R * DS4_ASG_RAW_STRIDE;
    const float mr = rr[512], lr = rr[513];
    const float m = fmaxf(mc, mr);
    const float a = (mc == -INFINITY) ? 0.0f : expf(mc - m), b = expf(mr - m);
    const float l = lc * a + lr * b;
    const float inv = l == 0.0f ? 0.0f : 1.0f / l;
    const float4 vc = *(const float4 *)(oc + (uint64_t)r * 512u + d);
    const float4 vr = *(const float4 *)(rr + d);
    *(float4 *)(heads + R * 512u + d) = make_float4((vc.x * a + vr.x * b) * inv, (vc.y * a + vr.y * b) * inv,
                                                   (vc.z * a + vr.z * b) * inv, (vc.w * a + vr.w * b) * inv);
}

/* 压缩缓存行([448 f16][64 f32]) → 整行 f16 影子: RoPE 段在此转 f16, 这是 asg 路(f16 GEMM)自带的舍入, 09-06 已验收 */
__global__ static void asg_comp_rows_to_f16_kernel(__half *dst, const uint8_t *rows, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(ld_comp1(COMP_ROW(rows, i / 512u), (uint32_t)(i % 512u)));
}

/* 返回 <0 = 本路不适用(cuBLAS 未就绪 / 捕获态里要增长 scratch), 调用方退回老核 */
static int attention_static_gemm_launch(
        float *heads, const float *sinks, const float *q, const float *raw_kv, const uint8_t *comp_kv,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head) {
    if (!g_cublas_ready || n_head != 64u || ratio == 0u) return -1;
    const uint64_t rows_total = (uint64_t)n_tokens * 64u;
    const uint64_t tile_rows = (uint64_t)DS4_ASG_TILE_TOKENS * 64u;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(0, &cs);
    if (cs != cudaStreamCaptureStatusNone) return -1;
    if (!asg_grow((void **)&g_asg.q16, &g_asg.q16_cap, rows_total * 512u, sizeof(__half), "q16") ||
        !asg_grow((void **)&g_asg.k16, &g_asg.k16_cap, (uint64_t)n_comp * 512u, sizeof(__half), "k16") ||
        !asg_grow((void **)&g_asg.s, &g_asg.s_cap, tile_rows * n_comp, sizeof(float), "S") ||
        !asg_grow((void **)&g_asg.p, &g_asg.p_cap, tile_rows * n_comp, sizeof(__half), "P") ||
        !asg_grow((void **)&g_asg.oc, &g_asg.oc_cap, tile_rows * 512u, sizeof(float), "Oc") ||
        !asg_grow((void **)&g_asg.ml, &g_asg.ml_cap, tile_rows * 2u, sizeof(float), "ml") ||
        !asg_grow((void **)&g_asg.raw, &g_asg.raw_cap, rows_total * DS4_ASG_RAW_STRIDE, sizeof(float), "raw")) return 0;
    const uint64_t nq = rows_total * 512u, nk = (uint64_t)n_comp * 512u;
    ds4_launch_pdl(f32_to_f16_kernel, (unsigned)((nq + 255u) / 256u), 256, 0, g_cur_stream, g_asg.q16, q, nq);
    asg_comp_rows_to_f16_kernel<<<(unsigned)((nk + 255u) / 256u), 256, 0, g_cur_stream>>>(g_asg.k16, comp_kv, nk);
    const float scale = rsqrtf(512.0f);
    asg_raw_partial_kernel<<<dim3(n_tokens, 8u, 1), 256, 0, g_cur_stream>>>(
        g_asg.raw, sinks, q, raw_kv, n_tokens, pos0, n_raw, raw_cap, raw_start, window, n_head, scale);
    if (!cuda_ok(cudaGetLastError(), "attn gemm prep launch")) return 0;
    (void)cublasSetStream(g_cublas, g_cur_stream);
    const float one = 1.0f, zero = 0.0f;
    for (uint32_t t0 = 0; t0 < n_tokens; t0 += DS4_ASG_TILE_TOKENS) {
        const uint32_t nt = (n_tokens - t0 < DS4_ASG_TILE_TOKENS) ? (n_tokens - t0) : DS4_ASG_TILE_TOKENS;
        const uint32_t rows = nt * 64u;
        cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n_comp, (int)rows, 512, &scale,
                                         g_asg.k16, CUDA_R_16F, 512, g_asg.q16 + (uint64_t)t0 * 64u * 512u, CUDA_R_16F, 512,
                                         &zero, g_asg.s, CUDA_R_32F, (int)n_comp, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attn gemm S")) return 0;
        asg_softmax_kernel<<<rows, 256, 0, g_cur_stream>>>(g_asg.ml, g_asg.p, g_asg.s, n_comp, t0, pos0, ratio);
        if (!cuda_ok(cudaGetLastError(), "attn gemm softmax launch")) return 0;
        st = cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, 512, (int)rows, (int)n_comp, &one,
                          g_asg.k16, CUDA_R_16F, 512, g_asg.p, CUDA_R_16F, (int)n_comp,
                          &zero, g_asg.oc, CUDA_R_32F, 512, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attn gemm PV")) return 0;
        asg_combine_kernel<<<rows, 128, 0, g_cur_stream>>>(heads, g_asg.oc, g_asg.ml, g_asg.raw, t0 * 64u);
        if (!cuda_ok(cudaGetLastError(), "attn gemm combine launch")) return 0;
    }
    return 1;
}
