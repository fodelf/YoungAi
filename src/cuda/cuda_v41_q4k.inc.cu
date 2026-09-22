/* cuda_v41_q4k.inc.cu — ds4_cuda.cu 分片: 骨架 q4_K 的解码 GEMV / 预填解量化 / 嵌入取行(2026-09-19)。
 *
 * 【为什么有这一族】骨架此前只有 fp4x32(4.25 bpw) 一种。09-19 的 100 GB 配方把骨架换成 q4_K
 * (4.5 bpw, 144 B/256 元素): 每块一个 f16 主 scale + 一个 f16 主 min + 8 组 6-bit 子 scale/min,
 * 比"一个 2 的幂 × 32 列"多了 min 项 ⇒ 能表示非零均值的块, 同三张量的权重残差 1.37% → 0.51%。
 * 盘上两种格式并存, 引擎按登记类型分发(core_v41_attn.c 的 v41_tproj), 老 GGUF 一行不用改。
 *
 * 【解码值的定义只有一处】`src/common/ds4_quantfmt.c` 的 ds4_deq_q4_K 是金标(ds4_unit 有逐字节夹具),
 * 本文件的核与它逐式同源。★改这里必须回去对拍那个夹具★ —— q4_K 的 12 B 里塞了 8 组 6-bit
 * scale + 8 组 6-bit min, 打包错一位不报错, 只让整组 32 个元素偏一个常数。
 *
 * 【lane 排布 = 一次 uint32 拿 4 个字节】09-18 量到的硬约束: GB10 的 DRAM 按 64 B 取, 一条 load
 * 读不满 128 B 整线就只有 142 GB/s(整线 219)。所以 lane l 读 qs[4l..4l+3] —— 一个 warp 一轮
 * 128 B 连续。qs 的字节 p 同时给两个元素: 低 nibble → 子块 2(p/32), 高 nibble → 子块 2(p/32)+1,
 * 两者相距 32 个元素(布局见 ds4_deq_q4_K 的 j += 64 那个循环)。 */

#define V41_Q4K_BLK 256u          /* 每块元素数 */
#define V41_Q4K_BYTES 144u        /* 每块字节数: 2(d) + 2(dmin) + 12(scales) + 128(qs) */
static v41_scratch g_v41_q4k_wb;  /* 预填: 解出来的 bf16 权重暂存(与 fp4x32 的 g_v41_wbf 同角色) */

/* 块头解出 8 组 (scale, min) 的第 j 组。与 ds4_quantfmt.c 的 q4k_scale_min 逐式同。 */
__device__ __forceinline__ static void v41_q4k_sm(const uint8_t *sc, int j, float *s, float *m) {
    uint8_t a, b;
    if (j < 4) { a = sc[j] & 63u; b = sc[j + 4] & 63u; }
    else { a = (uint8_t)((sc[j + 4] & 0xFu) | ((sc[j - 4] >> 6) << 4));
           b = (uint8_t)((sc[j + 4] >> 4)  | ((sc[j - 0] >> 6) << 4)); }
    *s = (float)a; *m = (float)b;
}

/* 一个 warp 对一个 q4_K 块与 NT 条激活做点积, 结果累加进 acc[NT]。
 * lane l: gidx = l>>3 选 64 元素组, q0 = (l&7)*4 选组内 4 个连续元素。
 * 低半 → 元素 gidx*64 + q0 + i(子块 2·gidx); 高半 → 再 +32(子块 2·gidx+1)。 */
template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_blk_acc(const uint8_t *blk, const float *x, uint32_t x_stride,
                                                       uint32_t base, float *acc) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    /* d/dmin/scales 是块级的, 32 个 lane 读同一地址 ⇒ 硬件广播, 不是 32 次访存 */
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const float dl = d * s_lo, ml = dmin * m_lo, dh = d * s_hi, mh = dmin * m_hi;
    /* qs 从块偏移 16 起; blk 是 144 的整数倍 + 张量起点, 16 字节对齐 ⇒ uint32 读安全(发射器已校对齐) */
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    float wl[4], wh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        wl[i] = dl * (float)(byte & 0xFu) - ml;
        wh[i] = dh * (float)(byte >> 4) - mh;
    }
    const uint32_t e_lo = base + gidx * 64u + q0, e_hi = e_lo + 32u;
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        const float *xt = x + (uint64_t)t * x_stride;
        const float4 xl = *(const float4 *)(xt + e_lo), xh = *(const float4 *)(xt + e_hi);
        acc[t] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
                + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
    }
}

/* 解码 GEMV: 一 block nwarp 个 warp, ksplit 段分 K, 每 warp 管 nwarp/ksplit 行中的一行一段。
 * 结构与 fp4x32 那支同(cuda_v41_4.inc.cu), 所以那边判负过的几条(激活进 shared / 一 warp 多行 /
 * 整段搬 shared)在这里同样不用再试 —— 病因是 L1 本来就接住了激活, 与格式无关。 */
template <uint32_t NT>
__global__ static void v41_q4k_gemv_kernel(float *out, const uint8_t *w, const float *x,
                                           uint32_t in_dim, uint32_t out_dim, uint32_t x_stride,
                                           uint32_t out_stride, uint32_t ksplit, uint64_t w_gstride,
                                           uint32_t x_gstride, uint32_t out_gstride, int round_out) {
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, nwarp = blockDim.x >> 5;
    const uint32_t rows_per_block = nwarp / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[16][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const uint32_t nblk = in_dim / V41_Q4K_BLK;
        const uint8_t *wr = w + (uint64_t)r * nblk * V41_Q4K_BYTES;
        for (uint32_t b = kpart; b < nblk; b += ksplit)
            v41_q4k_blk_acc<NT>(wr + (uint64_t)b * V41_Q4K_BYTES, x, x_stride, b * V41_Q4K_BLK, acc);
    }
    /* warp 内规约 → 段间规约(段和按 red[] 的固定序相加 ⇒ 同 ksplit 下确定) */
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++)
        for (int o = 16; o; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
    if (lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
    }
    __syncthreads();
    if (r < out_dim && kpart == 0 && lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float sum = 0.f;
            for (uint32_t k = 0; k < ksplit; k++) sum += red[rloc * ksplit + k][t];
            /* round_out: 与 fp4x32 路同口径 —— 官方在这几处把激活舍到 bf16 格点 */
            out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(sum) : sum;
        }
    }
}

/* q4_K → bf16(预填: 解成稠密再走 cuBLAS, 与 fp4x32 的 v41_fp4x32_to_bf16_kernel 同一套路)。
 * 一个 warp 一块, lane 排布与 GEMV 完全相同 ⇒ 两条路解出的值逐位同。 */
__global__ static void v41_q4k_to_bf16_kernel(__nv_bfloat16 *o, const uint8_t *w, uint64_t nblk) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const uint8_t *blk = w + b * V41_Q4K_BYTES;
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    __nv_bfloat16 *ob = o + b * V41_Q4K_BLK;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        ob[gidx * 64u + q0 + i]       = __float2bfloat16(d * s_lo * (float)(byte & 0xFu) - dmin * m_lo);
        ob[gidx * 64u + 32u + q0 + i] = __float2bfloat16(d * s_hi * (float)(byte >> 4)  - dmin * m_hi);
    }
}

/* 嵌入取行: token id → 该行 f32(词表 × 5120 的 q4_K 张量)。一个 warp 一块。 */
__global__ static void v41_q4k_embed_kernel(float *out, const int32_t *tok, const uint8_t *w,
                                            uint32_t n_vocab, uint32_t dim) {
    const uint32_t t = blockIdx.y, nblk = dim / V41_Q4K_BLK;
    const uint32_t b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const int32_t id = tok[t];
    if (id < 0 || (uint32_t)id >= n_vocab) return;
    const uint8_t *blk = w + ((uint64_t)id * nblk + b) * V41_Q4K_BYTES;
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    float *o = out + (uint64_t)t * dim + b * V41_Q4K_BLK;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        o[gidx * 64u + q0 + i]       = d * s_lo * (float)(byte & 0xFu) - dmin * m_lo;
        o[gidx * 64u + 32u + q0 + i] = d * s_hi * (float)(byte >> 4)  - dmin * m_hi;
    }
}

/* ---- 发射器 ---- */
/* 对齐前提: uint32 读 qs 要求每块 4 字节对齐。块长 144 是 16 的倍数, 所以只要张量起点对齐就都对齐;
 * GGUF 数据区按 32 B 对齐 ⇒ 正常成立。不成立时硬停车(返回 0), 不悄悄走一条慢路。 */
static int v41_q4k_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                        const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                        uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what) {
    if ((in_dim % V41_Q4K_BLK) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || n_groups == 0) return 0;
    const uint64_t wg = out_dim * (in_dim / V41_Q4K_BLK) * V41_Q4K_BYTES, wbytes = wg * n_groups;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    if (((uintptr_t)w & 3u) != 0u) { fprintf(stderr, "ds4: %s q4_K 张量起点未 4 字节对齐\n", what); return 0; }
    /* 并行度目标沿用 fp4x32 那支实测出来的 8192(见 cuda_v41_4.inc.cu 的长注释); K 的分段单位是一个
     * 256 元素块, 分不出 ksplit 段就收回来 —— 否则多出来的 warp 一轮都跑不到。 */
    uint32_t ksplit = 1;
    while (ksplit < V41_GEMV_WARPS && out_dim * ksplit * n_groups < 8192u) ksplit <<= 1;
    const uint32_t nb = (uint32_t)(in_dim / V41_Q4K_BLK);
    while (ksplit > 1u && ksplit > nb) ksplit >>= 1;
    const uint32_t warps = (n_tok > 1u) ? 16u : V41_GEMV_WARPS;
    const uint32_t rpb = warps / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), n_groups);
    #define V41_Q4K_LAUNCH(NT) v41_q4k_gemv_kernel<NT><<<grid, warps * 32u, 0, g_cur_stream>>>( \
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, x_stride, out_stride, ksplit, wg, x_gstride, out_gstride, round_out)
    switch (n_tok) {
        case 1: V41_Q4K_LAUNCH(1u); break;  case 2: V41_Q4K_LAUNCH(2u); break;
        case 3: V41_Q4K_LAUNCH(3u); break;  case 4: V41_Q4K_LAUNCH(4u); break;
        case 5: V41_Q4K_LAUNCH(5u); break;  case 6: V41_Q4K_LAUNCH(6u); break;
        case 7: V41_Q4K_LAUNCH(7u); break;  default: V41_Q4K_LAUNCH(8u); break;
    }
    #undef V41_Q4K_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

/* 预填(n_tok > V41_GEMV_MAX_TOK): 权重解成 bf16 暂存, 激活转 bf16, 每组一发 cuBLAS。
 * 与 fp4x32 的 wo_a 预填路同一形态(见 cuda_v41_1.inc.cu 里那段"为什么是 bf16 不是 f16")。
 * ★不走 NVFP4 张量核★: 那条路要把权重摆成 e2m1 nibble, q4_K 的值不在 FP4 格点上, 转过去是二次量化。 */
static int v41_q4k_gemm(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim,
                        uint64_t out_dim, const float *x, float *out, uint32_t n_tok, uint32_t n_groups,
                        uint32_t x_stride, uint32_t out_stride, int round_out, const char *what) {
    const uint64_t nblk_g = out_dim * (in_dim / V41_Q4K_BLK), nblk = nblk_g * n_groups;
    if (off > model_size || nblk * V41_Q4K_BYTES > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, nblk * V41_Q4K_BYTES, what);
    if (!w) return 0;
    if (((uintptr_t)w & 3u) != 0u) { fprintf(stderr, "ds4: %s q4_K 张量起点未 4 字节对齐\n", what); return 0; }
    __nv_bfloat16 *wb = (__nv_bfloat16 *)v41_grow(&g_v41_q4k_wb, nblk * V41_Q4K_BLK * sizeof(__nv_bfloat16), "v41 q4k w bf16");
    if (!wb) return 0;
    const dim3 dblk(32, 8);
    v41_q4k_to_bf16_kernel<<<(unsigned)((nblk + 7) / 8), dblk, 0, g_cur_stream>>>(wb, w, nblk);
    if (!cuda_ok(cudaGetLastError(), "v41 q4k→bf16")) return 0;
    const uint64_t xn = (uint64_t)n_tok * x_stride;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, xn * sizeof(__nv_bfloat16), "v41 q4k x bf16");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(xb, x, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 q4k x→bf16")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    for (uint32_t g = 0; g < n_groups; g++) {
        cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                         wb + (uint64_t)g * nblk_g * V41_Q4K_BLK, CUDA_R_16BF, (int)in_dim,
                                         xb + (uint64_t)g * in_dim, CUDA_R_16BF, (int)x_stride, &beta,
                                         out + (uint64_t)g * out_dim, CUDA_R_32F, (int)out_stride,
                                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, what)) return 0;
    }
    return 1;
}

/* ---- ds4_gpu_v41.h 契约的 q4_K 三支(命名沿用 v41_* 只为与同一伞头里的既有 API 一致;
 * 按铁律 09-15 新代码不该带版本名, 这是既有欠账, 整族改名时一起处理) ---- */
int ds4_gpu_v41_matmul_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * 4 || out->bytes < (uint64_t)n_tok * out_dim * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK)
        return v41_q4k_gemv(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr, (uint32_t)in_dim,
                            (float *)out->ptr, (uint32_t)out_dim, n_tok, 1u, 0u, 0u, round_out, "v41 q4k gemv");
    if (!v41_q4k_gemm(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr,
                      (float *)out->ptr, n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim, round_out, "v41 q4k gemm")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;
}

int ds4_gpu_v41_grouped_matmul_q4k_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                          uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                          uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (heads->bytes < (uint64_t)n_tok * in_all * 4 || low->bytes < (uint64_t)n_tok * out_all * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK) {
        if (group_dim % V41_Q4K_BLK) return 0;
        return v41_q4k_gemv(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr, (uint32_t)in_all,
                            (float *)low->ptr, (uint32_t)out_all, n_tok, n_groups, (uint32_t)group_dim, (uint32_t)rank, round_out, "v41 wo_a q4k gemv");
    }
    if (!v41_q4k_gemm(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr,
                      (float *)low->ptr, n_tok, n_groups, (uint32_t)in_all, (uint32_t)out_all, round_out, "v41 wo_a q4k gemm")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;
}

int ds4_gpu_v41_embed_q4k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                 uint64_t model_size, uint64_t weight_offset, uint64_t n_vocab,
                                 uint32_t n_tok, uint64_t dim) {
    if (!out || !tokens || n_tok == 0 || (dim % V41_Q4K_BLK)) return 0;
    const uint64_t nblk = n_vocab * (dim / V41_Q4K_BLK);
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * V41_Q4K_BYTES, "v41 q4k embed");
    if (!w || ((uintptr_t)w & 3u) != 0u) return 0;
    const uint32_t nb = (uint32_t)(dim / V41_Q4K_BLK);
    const dim3 dblk(32, 8), grid((nb + 7u) / 8u, n_tok);
    v41_q4k_embed_kernel<<<grid, dblk, 0, g_cur_stream>>>((float *)out->ptr, (const int32_t *)tokens->ptr, w,
                                                          (uint32_t)n_vocab, (uint32_t)dim);
    return cuda_ok(cudaGetLastError(), "v41 q4k embed");
}

/* 出口头列平方和 Σ_v W[v][d]²(dcap 的出口度量; fp4x32 版与所以然在 cuda_v41_gemv_highprec.inc.cu)。
 * 一 block 一列, 256 线程沿 v 跨步。q4_K 里元素 (v, d) 的位置: 块 v·nblk + d/256, 块内 e = d%256,
 * 组 g = e/64, 组内 o = e%64: o<32 取低 nibble(子块 2g), 否则高 nibble(子块 2g+1), 字节 = qs[g·32 + o%32]
 * —— 与上面 GEMV 的 lane 映射是同一张表, 只是这里按列取。 */
__global__ static void v41_q4k_colnorm_kernel(float *out, const uint8_t *w, uint32_t V, uint32_t D) {
    const uint32_t d = blockIdx.x, nblk = D / V41_Q4K_BLK, bi = d / V41_Q4K_BLK, e = d % V41_Q4K_BLK;
    const uint32_t g = e >> 6, o = e & 63u, hi = o >> 5;
    __shared__ float red[256];
    float s = 0.f;
    for (uint32_t v = threadIdx.x; v < V; v += blockDim.x) {
        const uint8_t *blk = w + ((uint64_t)v * nblk + bi) * V41_Q4K_BYTES;
        const float dd = __half2float(__ushort_as_half((uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8))));
        const float dm = __half2float(__ushort_as_half((uint16_t)((uint16_t)blk[2] | ((uint16_t)blk[3] << 8))));
        float sc, mn;
        v41_q4k_sm(blk + 4, (int)(g * 2u + hi), &sc, &mn);
        const uint8_t by = blk[16u + g * 32u + (o & 31u)];
        const float wv = dd * sc * (float)(hi ? (by >> 4) : (by & 0xFu)) - dm * mn;
        s += wv * wv;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (uint32_t k = blockDim.x >> 1; k; k >>= 1) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[d] = red[0];
}
int ds4_gpu_v41_head_colnorm_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                        uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd) {
    if (!out || (n_embd % V41_Q4K_BLK) || out->bytes < (uint64_t)n_embd * 4) return 0;
    const uint64_t nblk = (uint64_t)n_vocab * (n_embd / V41_Q4K_BLK);
    if (weight_offset > model_size || nblk * V41_Q4K_BYTES > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * V41_Q4K_BYTES, "v41 q4k head colnorm");
    if (!w) return 0;
    v41_q4k_colnorm_kernel<<<n_embd, 256, 0, g_cur_stream>>>((float *)out->ptr, w, n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "v41 q4k head colnorm");
}
