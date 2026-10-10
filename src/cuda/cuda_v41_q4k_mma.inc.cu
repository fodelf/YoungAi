/* cuda_v41_q4k_mma.inc.cu — ds4_cuda.cu 分片: 合批解码 9~16 行的 q4_K 骨架乘法(张量核, 2026-10-10)。在 cuda_v41_q4k 之后(借块常量与 v41_bf16r)。
 *
 * 【为什么】合批一步超过 8 行时, 稠密段原来掉进预填路(整层解成 bf16 暂存 + cuBLAS), 16 路一步 298 ms 比 8 路 87 还慢; 而解码 GEMV 核
 *   (stage/pipe)每个 warp 每行都从 L1 重读同段激活, 8 行起每行成本不降(微基准每步骨架 8 行 35.2 / 16 行 59.9 ms, 墙 19.8)。
 *   gguf-tools/bench/v41_q4k_gemv_bench_mma.cuh 第一版: 16 行 34.8 ms(扣掉表里重复的 wo_a 一项 ≈ 29), 与现 GEMV 相对差 1~2e-7。
 * 【怎么算】q4_K 一块 256 = 8 子块 × 32, w = d·s_j·q − dmin·m_j(q 是 0..15 整数)。按子块 Σ w x = d·s_j·Σ q x − dmin·m_j·Σ x:
 *   Σ q x 用 mma.m16n8k16(q 在 bf16 里精确, 激活本来在 bf16 格点上 —— 与解码多行 GEMV 的 bf16 激活同一前提; 积在 fp32 里精确, fp32 累加),
 *   每子块两发 k16 从零起算再在 fp32 里乘 d·s_j; Σ x 每 (行, 子块) 一个数由 v41_xsum_kernel 先算。与解码 GEMV 只差加法次序(不像预填路把权重舍到 bf16)。
 * 【只给合批】ds4_gpu_v41_set_multi_rows(1) 期间、行数 9..16 才走这里(core_v41_multi.c 在合批步超 8 行时开、步末关);
 *   单请求路 / 验证批 / 预填小尾块 / 打分路照旧 —— 那些路的输出逐位不变。 */
#define V41_MMA_MAX_TOK 16u
static int g_v41_multi_rows = 0;
int ds4_gpu_v41_set_multi_rows(int on) { g_v41_multi_rows = on ? 1 : 0; return 1; }
static v41_scratch g_v41_xsum;

/* Σx: xs[t][sb] = Σ_{i<32} x[t][32·sb + i](f32, 顺序加); x 行距 x_stride, 每行 nsb 个子块 */
__global__ static void v41_xsum_kernel(float *xs, const float *x, uint32_t x_stride, uint32_t nsb, uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsb * n_tok) return;
    const uint32_t t = i / nsb, sb = i % nsb;
    const float *p = x + (uint64_t)t * x_stride + sb * 32u;
    float s = 0.f;
    for (uint32_t k = 0; k < 32u; k++) s += p[k];
    xs[i] = s;
}
__device__ __forceinline__ static uint32_t v41_q2bf(uint32_t v, uint32_t sh) {   /* 两字节各取一个 nibble → bf16x2(低半 = 第一个字节) */
    const __nv_bfloat162 r = __floats2bfloat162_rn((float)((v >> sh) & 0xFu), (float)((v >> (8u + sh)) & 0xFu));
    return *(const uint32_t *)&r;
}
__device__ __forceinline__ static uint32_t v41_f2bf2(const float *p, bool ok) {   /* 两个相邻激活 → bf16x2; 越过行数的 token 给 0 */
    if (!ok) return 0u;
    const __nv_bfloat162 r = __floats2bfloat162_rn(p[0], p[1]);
    return *(const uint32_t *)&r;
}
__device__ __forceinline__ static void v41_mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ static void v41_q4k_smf(const uint4 &h, uint32_t j, float *s, float *m) {   /* 第 j 个子块的 6-bit scale/min(与 GEMV 核同式) */
    const uint32_t *b = (const uint32_t *)&h;
    auto sc = [&](uint32_t k) -> uint32_t { return (b[1 + k / 4u] >> ((k & 3u) * 8u)) & 0xFFu; };
    uint32_t a, c;
    if (j < 4u) { a = sc(j) & 63u; c = sc(j + 4u) & 63u; }
    else { a = (sc(j + 4u) & 0xFu) | ((sc(j - 4u) >> 6) << 4); c = (sc(j + 4u) >> 4) | ((sc(j) >> 6) << 4); }
    *s = (float)a; *m = (float)c;
}

/* 一个 CTA 8 warp = RG 个 16 行组 × KS 段(按块交错分 K); 本 CTA 的 16·RG 行 × 整个 K 搬进 shared。blockIdx.y = 组(wo_a 8 组一发)。 */
template <uint32_t NTILE, uint32_t RG>
__global__ static void __launch_bounds__(256) v41_q4k_mma_kernel(float *out, const uint8_t *w, const float *x, const float *xs,
                                                                 uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, uint32_t x_stride, uint32_t out_stride,
                                                                 uint64_t wg, uint32_t x_gstride, uint32_t out_gstride, uint32_t nsb_row, int round_out) {
    constexpr uint32_t KS = 8u / RG, NT = NTILE * 8u, ROWS = 16u * RG, BY = V41_Q4K_BYTES;
    extern __shared__ uint4 v41_mma_st[];
    __shared__ float red[8][16][NT];
    const uint32_t g = blockIdx.y, nblk = in_dim / V41_Q4K_BLK, r0 = blockIdx.x * ROWS;
    w += (uint64_t)g * wg; x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride;
    const uint32_t sb0 = g * (x_gstride / 32u);   /* 本组在 Σx 行里的子块起点 */
    const uint32_t n16 = ROWS * nblk * (BY / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BY);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) v41_mma_st[i] = __ldcs(src + i);
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, rg = warp / KS, kp = warp % KS, gid = lane >> 2, tig = lane & 3u;
    const uint8_t *sw = (const uint8_t *)v41_mma_st;
    float acc[NTILE][4];
    #pragma unroll
    for (uint32_t n = 0; n < NTILE; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
    for (uint32_t b = kp; b < nblk; b += KS) {
        const uint8_t *bk0 = sw + ((uint64_t)(rg * 16u + gid) * nblk + b) * BY, *bk1 = bk0 + (uint64_t)8u * nblk * BY;
        const uint4 h0 = *(const uint4 *)bk0, h1 = *(const uint4 *)bk1;
        const float d0 = __half2float(__ushort_as_half((unsigned short)(h0.x & 0xFFFFu))), n0 = __half2float(__ushort_as_half((unsigned short)(h0.x >> 16)));
        const float d1 = __half2float(__ushort_as_half((unsigned short)(h1.x & 0xFFFFu))), n1 = __half2float(__ushort_as_half((unsigned short)(h1.x >> 16)));
        #pragma unroll
        for (uint32_t q = 0; q < 4u; q++) {
            float s, m, sl0, ml0, sh0, mh0, sl1, ml1, sh1, mh1;
            v41_q4k_smf(h0, 2u * q, &s, &m); sl0 = d0 * s; ml0 = n0 * m;
            v41_q4k_smf(h0, 2u * q + 1u, &s, &m); sh0 = d0 * s; mh0 = n0 * m;
            v41_q4k_smf(h1, 2u * q, &s, &m); sl1 = d1 * s; ml1 = n1 * m;
            v41_q4k_smf(h1, 2u * q + 1u, &s, &m); sh1 = d1 * s; mh1 = n1 * m;
            uint32_t alo[2][4], ahi[2][4];
            #pragma unroll
            for (uint32_t t = 0; t < 2u; t++) {
                const uint32_t o = 16u + 32u * q + 16u * t + tig * 2u;
                const uint32_t p0 = *(const uint16_t *)(bk0 + o), p1 = *(const uint16_t *)(bk1 + o);
                const uint32_t p0b = *(const uint16_t *)(bk0 + o + 8u), p1b = *(const uint16_t *)(bk1 + o + 8u);
                alo[t][0] = v41_q2bf(p0, 0); alo[t][1] = v41_q2bf(p1, 0); alo[t][2] = v41_q2bf(p0b, 0); alo[t][3] = v41_q2bf(p1b, 0);
                ahi[t][0] = v41_q2bf(p0, 4); ahi[t][1] = v41_q2bf(p1, 4); ahi[t][2] = v41_q2bf(p0b, 4); ahi[t][3] = v41_q2bf(p1b, 4);
            }
            const uint32_t klo = b * V41_Q4K_BLK + 64u * q, khi = klo + 32u, sblo = sb0 + b * 8u + 2u * q, sbhi = sblo + 1u;
            #pragma unroll
            for (uint32_t n = 0; n < NTILE; n++) {
                const uint32_t tb = n * 8u + gid;   /* B 片段的 token */
                const bool okb = tb < n_tok;
                const float *xr = x + (uint64_t)(okb ? tb : 0u) * x_stride;
                float cl[4] = {0.f, 0.f, 0.f, 0.f}, ch[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (uint32_t t = 0; t < 2u; t++) {
                    const uint32_t kl = klo + 16u * t + tig * 2u, kh = khi + 16u * t + tig * 2u;
                    v41_mma16816(cl, alo[t], v41_f2bf2(xr + kl, okb), v41_f2bf2(xr + kl + 8u, okb));
                    v41_mma16816(ch, ahi[t], v41_f2bf2(xr + kh, okb), v41_f2bf2(xr + kh + 8u, okb));
                }
                const uint32_t t0 = n * 8u + tig * 2u;   /* C 片段的 token: t0 与 t0+1 */
                const float xl0 = t0 < n_tok ? xs[(uint64_t)t0 * nsb_row + sblo] : 0.f, xl1 = t0 + 1u < n_tok ? xs[(uint64_t)(t0 + 1u) * nsb_row + sblo] : 0.f;
                const float xh0 = t0 < n_tok ? xs[(uint64_t)t0 * nsb_row + sbhi] : 0.f, xh1 = t0 + 1u < n_tok ? xs[(uint64_t)(t0 + 1u) * nsb_row + sbhi] : 0.f;
                acc[n][0] += sl0 * cl[0] - ml0 * xl0 + sh0 * ch[0] - mh0 * xh0;
                acc[n][1] += sl0 * cl[1] - ml0 * xl1 + sh0 * ch[1] - mh0 * xh1;
                acc[n][2] += sl1 * cl[2] - ml1 * xl0 + sh1 * ch[2] - mh1 * xh0;
                acc[n][3] += sl1 * cl[3] - ml1 * xl1 + sh1 * ch[3] - mh1 * xh1;
            }
        }
    }
    #pragma unroll
    for (uint32_t n = 0; n < NTILE; n++) {
        const uint32_t t0 = n * 8u + tig * 2u;
        red[warp][gid][t0] = acc[n][0]; red[warp][gid][t0 + 1u] = acc[n][1];
        red[warp][gid + 8u][t0] = acc[n][2]; red[warp][gid + 8u][t0 + 1u] = acc[n][3];
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < ROWS * NT; i += blockDim.x) {
        const uint32_t row = i / NT, t = i % NT, rgi = row / 16u, rr = row % 16u;
        if (t >= n_tok || r0 + row >= out_dim) continue;
        float s = 0.f;
        for (uint32_t k = 0; k < KS; k++) s += red[rgi * KS + k][rr][t];
        out[(uint64_t)t * out_stride + r0 + row] = round_out ? v41_bf16r(s) : s;
    }
}

/* 走得了就发并返回 1; 返回 -1 = 这个形状不归它(行数/开关/对齐/整除不满足), 调用方照旧; 0 = 发射失败 */
static int v41_q4k_mma(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                       const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                       uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what) {
    if (!g_v41_multi_rows || n_tok <= V41_GEMV_MAX_TOK || n_tok > V41_MMA_MAX_TOK || (in_dim % V41_Q4K_BLK) || n_groups == 0) return -1;
    const uint32_t nblk = (uint32_t)(in_dim / V41_Q4K_BLK), rg = nblk < 8u ? 2u : 1u, rows = 16u * rg;
    if (out_dim % rows) return -1;
    const uint64_t wg = out_dim * nblk * V41_Q4K_BYTES, wbytes = wg * n_groups;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w || ((uintptr_t)w & 15u) != 0u) return -1;
    const size_t shm = (size_t)rows * nblk * V41_Q4K_BYTES;
    static int optin = 0;
    if (!optin && cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    const int dyn_max = optin - (int)(8u * 16u * 16u * 4u);
    if ((int)shm > dyn_max) return -1;
    /* Σx 按整行(含各组)算: 组 g 的子块从 g·x_gstride/32 起 */
    const uint32_t row_len = n_groups > 1u ? x_stride : (uint32_t)in_dim, nsb_row = row_len / 32u;
    float *xs = (float *)v41_grow(&g_v41_xsum, (uint64_t)n_tok * nsb_row * 4u, "v41 q4k mma xsum");
    if (!xs) return 0;
    v41_xsum_kernel<<<(nsb_row * n_tok + 255u) / 256u, 256, 0, g_cur_stream>>>(xs, x, x_stride, nsb_row, n_tok);
    const dim3 grid((uint32_t)(out_dim / rows), n_groups);
    static bool set1 = false, set2 = false;
    if (rg == 2u) {
        if (!set2) { (void)cudaFuncSetAttribute(v41_q4k_mma_kernel<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_max); set2 = true; }
        v41_q4k_mma_kernel<2, 2><<<grid, 256, shm, g_cur_stream>>>(out, w, x, xs, (uint32_t)in_dim, (uint32_t)out_dim, n_tok, x_stride, out_stride,
                                                                   wg, n_groups > 1u ? x_gstride : 0u, n_groups > 1u ? out_gstride : 0u, nsb_row, round_out);
    } else {
        if (!set1) { (void)cudaFuncSetAttribute(v41_q4k_mma_kernel<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_max); set1 = true; }
        v41_q4k_mma_kernel<2, 1><<<grid, 256, shm, g_cur_stream>>>(out, w, x, xs, (uint32_t)in_dim, (uint32_t)out_dim, n_tok, x_stride, out_stride,
                                                                   wg, n_groups > 1u ? x_gstride : 0u, n_groups > 1u ? out_gstride : 0u, nsb_row, round_out);
    }
    if (!cuda_ok(cudaGetLastError(), what)) return 0;
#ifdef V41_MMA_CHECK   /* 临时对账(10-10 定位分岔): 同样的行拆成两次 ≤8 行走解码 GEMV, 逐矩阵比相对差, 打前 120 发 */
    {
        static int ncheck = 0;
        if (ncheck < 120) {
            const uint64_t on = (uint64_t)n_tok * out_stride;
            float *o2; cudaMalloc((void **)&o2, on * 4); cudaMemset(o2, 0, on * 4);
            for (uint32_t t0 = 0; t0 < n_tok; t0 += 8u) {
                const uint32_t nn = n_tok - t0 < 8u ? n_tok - t0 : 8u;
                v41_q4k_gemv(model_map, model_size, off, in_dim, out_dim, x + (uint64_t)t0 * x_stride, x_stride, o2 + (uint64_t)t0 * out_stride, out_stride,
                             nn, n_groups, x_gstride, out_gstride, round_out, "mma check gemv");
            }
            cudaDeviceSynchronize();
            float *a = (float *)malloc(on * 4), *b2 = (float *)malloc(on * 4);
            cudaMemcpy(a, out, on * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b2, o2, on * 4, cudaMemcpyDeviceToHost);
            double emax = 0, amax = 0; uint64_t at = 0;
            for (uint64_t i = 0; i < on; i++) { const double e = fabs((double)a[i] - b2[i]); if (e > emax) { emax = e; at = i; } if (fabs(b2[i]) > amax) amax = fabs(b2[i]); }
            fprintf(stderr, "ds4: [mma check %d] %s in %llu out %llu 组 %u 行 %u round %d: 最大差 %.3g(@%llu: %.6g vs %.6g)/ 幅值 %.3g = %.2e\n", ncheck, what,
                    (unsigned long long)in_dim, (unsigned long long)out_dim, n_groups, n_tok, round_out, emax, (unsigned long long)at, a[at], b2[at], amax, amax > 0 ? emax / amax : 0.0);
            free(a); free(b2); cudaFree(o2);
            ncheck++;
        }
    }
#endif
    return 1;
}
