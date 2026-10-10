/* v41_q4k_gemv_bench_mma.cuh — q4_K 骨架小批乘法的张量核形态(2026-10-10; 被 v41_q4k_gemv_bench.cu 包含)。
 *
 * 【为什么】合批解码想把一步从 8 行放到 16/32 行(骨架权重一步只读一遍, 行多了摊薄)。现核 stage/pipe 每个 warp 每行都从 L1 重读同段激活,
 *   8 行起就不再贴带宽墙: PDL + bf16 块排布口径下每步骨架 NT=1 22.8 / 8 35.2 / 16 59.9 / 32 141.2 ms(墙 19.8)—— 每行成本不降。
 * 【怎么算】q4_K 一块 256 权重 = 8 个子块 × 32, w = d·s_j·q − dmin·m_j(q 是 0..15 的整数)。按子块:
 *   Σ_k w_k x_k = d·s_j·(Σ q_k x_k) − dmin·m_j·(Σ x_k)。Σ q·x 用 mma.m16n8k16(bf16 输入 fp32 累加): q 在 bf16 里精确, q·x 的积在 fp32 里精确,
 *   每个子块两发 k16 从零起算, 再在 fp32 里乘 d·s_j 加进总和; Σx 每 (token, 子块) 一个数, 由 k_xsum 先算好。
 *   ⇒ 与现核(fp32 权重 × bf16 激活, fp32 累加)只差加法次序, 不像预填路那样把权重先舍到 bf16。
 * 【布局】一个 CTA 8 warp = RG 个行组(每组 16 行) × KS 段(按块交错分 K); 本 CTA 的 16·RG 行 × 整个 K 一次搬进 shared(权重在全局里本来就是
 *   行优先、一行一串块, 连续 16·RG 行是一整段连续字节)。warp 内: gid = lane/4 管第 gid 与 gid+8 行, tig = lane%4 管 k 的 tig·2 与 tig·2+8 两对。
 *   激活按 bf16 行优先 [NT][in] 读(B 片段: token = nt·8 + gid, k = tig·2 起两个), 段间部分和经 shared 归约。 */

/* Σx: xs[t][sb] = Σ_{i<32} x[t][32·sb + i](bf16 → f32, 顺序加) */
__global__ static void k_xsum(float *xs, const uint16_t *x, uint32_t in_dim, uint32_t nt_rows) {
    const uint32_t nsb = in_dim / 32u, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsb * nt_rows) return;
    const uint32_t t = i / nsb, sb = i % nsb;
    const uint16_t *p = x + (uint64_t)t * in_dim + sb * 32u;
    float s = 0.f;
    for (uint32_t k = 0; k < 32u; k++) s += __uint_as_float((uint32_t)p[k] << 16);
    xs[i] = s;
}

__device__ __forceinline__ static uint32_t q2bf(uint32_t v, uint32_t sh) {   /* 两个字节各取一个 nibble(sh = 0 低 / 4 高) → bf16x2(低半 = 第一个字节) */
    const __nv_bfloat162 r = __floats2bfloat162_rn((float)((v >> sh) & 0xFu), (float)((v >> (8u + sh)) & 0xFu));
    return *(const uint32_t *)&r;
}
__device__ __forceinline__ static void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <uint32_t NTILE, uint32_t RG>
__global__ static void __launch_bounds__(256) k_mma(float *out, const uint8_t *w, const uint16_t *x, const float *xs,
                                                     uint32_t in_dim, uint32_t out_dim, uint32_t out_stride) {
    constexpr uint32_t KS = 8u / RG, NT = NTILE * 8u, ROWS = 16u * RG;
    extern __shared__ uint4 st[];
    __shared__ float red[8][16][NT];
    const uint32_t nblk = in_dim / BLK, nsb = in_dim / 32u, r0 = blockIdx.x * ROWS;
    const uint32_t n16 = ROWS * nblk * (BYTES / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BYTES);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) st[i] = __ldcs(src + i);
    pdl_wait_trigger();
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, rg = warp / KS, kp = warp % KS;
    const uint32_t gid = lane >> 2, tig = lane & 3u;
    const uint8_t *sw = (const uint8_t *)st;
    float acc[NTILE][4];
    #pragma unroll
    for (uint32_t n = 0; n < NTILE; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }
    for (uint32_t b = kp; b < nblk; b += KS) {
        const uint8_t *bk0 = sw + ((uint64_t)(rg * 16u + gid) * nblk + b) * BYTES, *bk1 = bk0 + (uint64_t)8u * nblk * BYTES;
        const uint4 h0 = *(const uint4 *)bk0, h1 = *(const uint4 *)bk1;
        const float d0 = __half2float(__ushort_as_half((unsigned short)(h0.x & 0xFFFFu))), n0 = __half2float(__ushort_as_half((unsigned short)(h0.x >> 16)));
        const float d1 = __half2float(__ushort_as_half((unsigned short)(h1.x & 0xFFFFu))), n1 = __half2float(__ushort_as_half((unsigned short)(h1.x >> 16)));
        #pragma unroll
        for (uint32_t g = 0; g < 4u; g++) {
            float s, m, sl0, ml0, sh0, mh0, sl1, ml1, sh1, mh1;
            sm_reg(h0, 2u * g, &s, &m); sl0 = d0 * s; ml0 = n0 * m;
            sm_reg(h0, 2u * g + 1u, &s, &m); sh0 = d0 * s; mh0 = n0 * m;
            sm_reg(h1, 2u * g, &s, &m); sl1 = d1 * s; ml1 = n1 * m;
            sm_reg(h1, 2u * g + 1u, &s, &m); sh1 = d1 * s; mh1 = n1 * m;
            uint32_t alo[2][4], ahi[2][4];   /* [k16 半段][a0a1 / a2a3 / a4a5 / a6a7] */
            #pragma unroll
            for (uint32_t t = 0; t < 2u; t++) {
                const uint32_t o = 16u + 32u * g + 16u * t + tig * 2u;
                const uint32_t p0 = *(const uint16_t *)(bk0 + o), p1 = *(const uint16_t *)(bk1 + o);
                const uint32_t p0b = *(const uint16_t *)(bk0 + o + 8u), p1b = *(const uint16_t *)(bk1 + o + 8u);
                alo[t][0] = q2bf(p0, 0); alo[t][1] = q2bf(p1, 0); alo[t][2] = q2bf(p0b, 0); alo[t][3] = q2bf(p1b, 0);
                ahi[t][0] = q2bf(p0, 4); ahi[t][1] = q2bf(p1, 4); ahi[t][2] = q2bf(p0b, 4); ahi[t][3] = q2bf(p1b, 4);
            }
            const uint32_t klo = b * BLK + 64u * g, khi = klo + 32u, sblo = b * 8u + 2u * g, sbhi = sblo + 1u;
            #pragma unroll
            for (uint32_t n = 0; n < NTILE; n++) {
                const uint16_t *xr = x + (uint64_t)(n * 8u + gid) * in_dim;
                float cl[4] = {0.f, 0.f, 0.f, 0.f}, ch[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (uint32_t t = 0; t < 2u; t++) {
                    const uint32_t kl = klo + 16u * t + tig * 2u, kh = khi + 16u * t + tig * 2u;
                    mma16816(cl, alo[t], *(const uint32_t *)(xr + kl), *(const uint32_t *)(xr + kl + 8u));
                    mma16816(ch, ahi[t], *(const uint32_t *)(xr + kh), *(const uint32_t *)(xr + kh + 8u));
                }
                const uint32_t t0 = n * 8u + tig * 2u;
                const float xl0 = xs[(uint64_t)t0 * nsb + sblo], xl1 = xs[(uint64_t)(t0 + 1u) * nsb + sblo];
                const float xh0 = xs[(uint64_t)t0 * nsb + sbhi], xh1 = xs[(uint64_t)(t0 + 1u) * nsb + sbhi];
                acc[n][0] += sl0 * cl[0] - ml0 * xl0 + sh0 * ch[0] - mh0 * xh0;
                acc[n][1] += sl0 * cl[1] - ml0 * xl1 + sh0 * ch[1] - mh0 * xh1;
                acc[n][2] += sl1 * cl[2] - ml1 * xl0 + sh1 * ch[2] - mh1 * xh0;
                acc[n][3] += sl1 * cl[3] - ml1 * xl1 + sh1 * ch[3] - mh1 * xh1;
            }
        }
    }
    /* 段间归约: 每个 warp 的部分和写进 red[warp], 同行组的 KS 段按段号升序相加 */
    #pragma unroll
    for (uint32_t n = 0; n < NTILE; n++) {
        const uint32_t t0 = n * 8u + tig * 2u;
        red[warp][gid][t0] = acc[n][0]; red[warp][gid][t0 + 1u] = acc[n][1];
        red[warp][gid + 8u][t0] = acc[n][2]; red[warp][gid + 8u][t0 + 1u] = acc[n][3];
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < ROWS * NT; i += blockDim.x) {
        const uint32_t row = i / NT, t = i % NT, g = row / 16u, rr = row % 16u;
        float s = 0.f;
        for (uint32_t k = 0; k < KS; k++) s += red[g * KS + k][rr][t];
        if (r0 + row < out_dim) out[(uint64_t)t * out_stride + r0 + row] = s;
    }
}

/* 行组数: K 块少(q_b 1280 = 5 块)时 8 段分不满, 让一个 CTA 管两个行组(KS = 4); 其余 KS = 8 */
static uint32_t mma_rg(uint32_t in_dim) { return in_dim / BLK < 8u ? 2u : 1u; }
template <uint32_t NTILE>
static void launch_mma(float *o, const uint8_t *w, const uint16_t *x, float *xs, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t nsb = in_dim / 32u, NT = NTILE * 8u, rg = mma_rg(in_dim), rows = 16u * rg, nblk = in_dim / BLK;
    launch_k(k_xsum, dim3((nsb * NT + 255u) / 256u), dim3(256), 0, xs, x, in_dim, NT);
    const size_t shm = (size_t)rows * nblk * BYTES;
    /* 动态 shared 上限 = 设备可选上限 − 静态 red[8][16][NT](wo_b 8192 列一个 CTA 16 行 = 73.7 KB) */
    static int optin = 0; if (!optin) CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const int dyn_max = optin - (int)(8u * 16u * NT * 4u);
    if ((int)shm > dyn_max) { printf("★张量核形态: %zu B shared 超上限 %d★\n", shm, dyn_max); exit(1); }
    if (rg == 2u) {
        static bool set2 = false; if (!set2) { CK(cudaFuncSetAttribute(k_mma<NTILE, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_max)); set2 = true; }
        launch_k(k_mma<NTILE, 2>, dim3(out_dim / rows), dim3(256), shm, o, w, x, (const float *)xs, in_dim, out_dim, out_dim);
    } else {
        static bool set1 = false; if (!set1) { CK(cudaFuncSetAttribute(k_mma<NTILE, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_max)); set1 = true; }
        launch_k(k_mma<NTILE, 1>, dim3(out_dim / rows), dim3(256), shm, o, w, x, (const float *)xs, in_dim, out_dim, out_dim);
    }
}

/* ---- 第二版(10-10): 每个 warp 自己流式搬权重(cp.async 双缓冲, 搬下一块时算这一块), 一个 warp 管 RW 个 16 行组、共用同一份激活片段 ----
 * 第一版的账(NT=8 / 16 / 32 每步 23.5 / 34.8 / 57.5 ms, 墙 19.8): CTA 先把整段权重搬进 shared 再算, 搬与算串行, 行数一多算的那段变长、带宽掉到 100~160 GB/s。
 * 这里: CTA 8 warp = (8/KS) 个 warp 行组 × KS 段; 每个 warp 管 16·RW 行, 按块交错走自己那一段 K, 每块 16·RW 行 × 144 B 用 cp.async 搬进本 warp 的两格缓冲。
 * KS > 1 时段间部分和经 shared 归约(复用缓冲区, 算完才写)。(RW, KS) 按形状选: 先 RW=2 KS=1, CTA 不到 96 个(每 SM 两个)就加 KS, 再不够退 RW=1。 */
template <uint32_t NTILE, uint32_t RW>
__global__ static void __launch_bounds__(256) k_mma2(float *out, const uint8_t *w, const uint16_t *x, const float *xs,
                                                      uint32_t in_dim, uint32_t out_dim, uint32_t out_stride, uint32_t KS) {
    constexpr uint32_t NT = NTILE * 8u, WR = 16u * RW, SEGB = WR * BYTES;   /* 一个 warp 一块 = WR 行 × 144 B */
    extern __shared__ uint4 st[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3u;
    const uint32_t nblk = in_dim / BLK, nsb = in_dim / 32u, wg = warp / KS, kp = warp % KS, rows_cta = (8u / KS) * WR;
    const uint32_t rw0 = blockIdx.x * rows_cta + wg * WR;   /* 本 warp 第一行 */
    uint8_t *buf = (uint8_t *)st + (uint64_t)warp * 2u * SEGB;
    auto issue = [&](uint32_t b, uint32_t slot) {
        for (uint32_t i = lane; i < WR * (BYTES / 16u); i += 32u) {
            const uint32_t row = i / (BYTES / 16u), part = i % (BYTES / 16u);
            __pipeline_memcpy_async(buf + slot * SEGB + row * BYTES + part * 16u, w + ((uint64_t)(rw0 + row) * nblk + b) * BYTES + part * 16u, 16);
        }
    };
    float acc[RW][NTILE][4];
    #pragma unroll
    for (uint32_t r = 0; r < RW; r++)
        #pragma unroll
        for (uint32_t n = 0; n < NTILE; n++) { acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f; }
    uint32_t slot = 0;
    if (kp < nblk) issue(kp, 0);
    __pipeline_commit();
    pdl_wait_trigger();
    for (uint32_t b = kp; b < nblk; b += KS, slot ^= 1u) {
        if (b + KS < nblk) issue(b + KS, slot ^ 1u);
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncwarp();
        const uint8_t *sb = buf + slot * SEGB;
        #pragma unroll
        for (uint32_t g = 0; g < 4u; g++) {
            uint32_t alo[RW][2][4], ahi[RW][2][4];
            float sl0[RW], ml0[RW], sh0[RW], mh0[RW], sl1[RW], ml1[RW], sh1[RW], mh1[RW];
            #pragma unroll
            for (uint32_t r = 0; r < RW; r++) {
                const uint8_t *bk0 = sb + (r * 16u + gid) * BYTES, *bk1 = bk0 + 8u * BYTES;
                const uint4 h0 = *(const uint4 *)bk0, h1 = *(const uint4 *)bk1;
                const float d0 = __half2float(__ushort_as_half((unsigned short)(h0.x & 0xFFFFu))), n0 = __half2float(__ushort_as_half((unsigned short)(h0.x >> 16)));
                const float d1 = __half2float(__ushort_as_half((unsigned short)(h1.x & 0xFFFFu))), n1 = __half2float(__ushort_as_half((unsigned short)(h1.x >> 16)));
                float s, m;
                sm_reg(h0, 2u * g, &s, &m); sl0[r] = d0 * s; ml0[r] = n0 * m;
                sm_reg(h0, 2u * g + 1u, &s, &m); sh0[r] = d0 * s; mh0[r] = n0 * m;
                sm_reg(h1, 2u * g, &s, &m); sl1[r] = d1 * s; ml1[r] = n1 * m;
                sm_reg(h1, 2u * g + 1u, &s, &m); sh1[r] = d1 * s; mh1[r] = n1 * m;
                #pragma unroll
                for (uint32_t t = 0; t < 2u; t++) {
                    const uint32_t o = 16u + 32u * g + 16u * t + tig * 2u;
                    const uint32_t p0 = *(const uint16_t *)(bk0 + o), p1 = *(const uint16_t *)(bk1 + o);
                    const uint32_t p0b = *(const uint16_t *)(bk0 + o + 8u), p1b = *(const uint16_t *)(bk1 + o + 8u);
                    alo[r][t][0] = q2bf(p0, 0); alo[r][t][1] = q2bf(p1, 0); alo[r][t][2] = q2bf(p0b, 0); alo[r][t][3] = q2bf(p1b, 0);
                    ahi[r][t][0] = q2bf(p0, 4); ahi[r][t][1] = q2bf(p1, 4); ahi[r][t][2] = q2bf(p0b, 4); ahi[r][t][3] = q2bf(p1b, 4);
                }
            }
            const uint32_t klo = b * BLK + 64u * g, khi = klo + 32u, sblo = b * 8u + 2u * g, sbhi = sblo + 1u;
            #pragma unroll
            for (uint32_t n = 0; n < NTILE; n++) {
                const uint16_t *xr = x + (uint64_t)(n * 8u + gid) * in_dim;
                uint32_t bl[2][2], bh[2][2];
                #pragma unroll
                for (uint32_t t = 0; t < 2u; t++) {
                    const uint32_t kl = klo + 16u * t + tig * 2u, kh = khi + 16u * t + tig * 2u;
                    bl[t][0] = *(const uint32_t *)(xr + kl); bl[t][1] = *(const uint32_t *)(xr + kl + 8u);
                    bh[t][0] = *(const uint32_t *)(xr + kh); bh[t][1] = *(const uint32_t *)(xr + kh + 8u);
                }
                const uint32_t t0 = n * 8u + tig * 2u;
                const float xl0 = xs[(uint64_t)t0 * nsb + sblo], xl1 = xs[(uint64_t)(t0 + 1u) * nsb + sblo];
                const float xh0 = xs[(uint64_t)t0 * nsb + sbhi], xh1 = xs[(uint64_t)(t0 + 1u) * nsb + sbhi];
                #pragma unroll
                for (uint32_t r = 0; r < RW; r++) {
                    float cl[4] = {0.f, 0.f, 0.f, 0.f}, ch[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(cl, alo[r][0], bl[0][0], bl[0][1]); mma16816(cl, alo[r][1], bl[1][0], bl[1][1]);
                    mma16816(ch, ahi[r][0], bh[0][0], bh[0][1]); mma16816(ch, ahi[r][1], bh[1][0], bh[1][1]);
                    acc[r][n][0] += sl0[r] * cl[0] - ml0[r] * xl0 + sh0[r] * ch[0] - mh0[r] * xh0;
                    acc[r][n][1] += sl0[r] * cl[1] - ml0[r] * xl1 + sh0[r] * ch[1] - mh0[r] * xh1;
                    acc[r][n][2] += sl1[r] * cl[2] - ml1[r] * xl0 + sh1[r] * ch[2] - mh1[r] * xh0;
                    acc[r][n][3] += sl1[r] * cl[3] - ml1[r] * xl1 + sh1[r] * ch[3] - mh1[r] * xh1;
                }
            }
        }
        __syncwarp();
    }
    if (KS == 1u) {   /* 不分段: 直接写 */
        #pragma unroll
        for (uint32_t r = 0; r < RW; r++)
            #pragma unroll
            for (uint32_t n = 0; n < NTILE; n++) {
                const uint32_t t0 = n * 8u + tig * 2u, ra = rw0 + r * 16u + gid;
                out[(uint64_t)t0 * out_stride + ra] = acc[r][n][0]; out[(uint64_t)(t0 + 1u) * out_stride + ra] = acc[r][n][1];
                out[(uint64_t)t0 * out_stride + ra + 8u] = acc[r][n][2]; out[(uint64_t)(t0 + 1u) * out_stride + ra + 8u] = acc[r][n][3];
            }
        return;
    }
    /* 分段: 部分和进 shared(缓冲区复用; 先等所有 warp 算完), 每个 warp 行组的 KS 段按段号升序相加 */
    __syncthreads();
    float *red = (float *)st;   /* [warp][WR][NT] */
    #pragma unroll
    for (uint32_t r = 0; r < RW; r++)
        #pragma unroll
        for (uint32_t n = 0; n < NTILE; n++) {
            const uint32_t t0 = n * 8u + tig * 2u, rr = r * 16u + gid;
            float *p = red + ((uint64_t)warp * WR + rr) * NT;
            p[t0] = acc[r][n][0]; p[t0 + 1u] = acc[r][n][1]; p[8u * NT + t0] = acc[r][n][2]; p[8u * NT + t0 + 1u] = acc[r][n][3];
        }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < rows_cta * NT; i += blockDim.x) {
        const uint32_t row = i / NT, t = i % NT, g2 = row / WR, rr = row % WR;
        float s = 0.f;
        for (uint32_t k = 0; k < KS; k++) s += red[((uint64_t)(g2 * KS + k) * WR + rr) * NT + t];
        out[(uint64_t)t * out_stride + blockIdx.x * rows_cta + row] = s;
    }
}

template <uint32_t NTILE>
static void launch_mma2(float *o, const uint8_t *w, const uint16_t *x, float *xs, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t nsb = in_dim / 32u, NT = NTILE * 8u, nblk = in_dim / BLK;
    launch_k(k_xsum, dim3((nsb * NT + 255u) / 256u), dim3(256), 0, xs, x, in_dim, NT);
    uint32_t rw = 2u, ks = 1u;
    auto ncta = [&]() { return out_dim / ((8u / ks) * 16u * rw); };
    while (ncta() < 96u && ks < 8u && ks * 2u <= nblk) ks *= 2u;
    if (ncta() < 96u) { rw = 1u; ks = 1u; while (ncta() < 96u && ks < 8u && ks * 2u <= nblk) ks *= 2u; }
    const size_t buf = 8u * 2u * 16u * rw * BYTES, red = (size_t)8u * 16u * rw * NT * 4u, shm = buf > red ? buf : red;
    static bool set = false;
    if (!set) {
        CK(cudaFuncSetAttribute(k_mma2<NTILE, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CK(cudaFuncSetAttribute(k_mma2<NTILE, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        set = true;
    }
    if (rw == 2u) launch_k(k_mma2<NTILE, 2>, dim3(ncta()), dim3(256), shm, o, w, x, (const float *)xs, in_dim, out_dim, out_dim, ks);
    else          launch_k(k_mma2<NTILE, 1>, dim3(ncta()), dim3(256), shm, o, w, x, (const float *)xs, in_dim, out_dim, out_dim, ks);
}
