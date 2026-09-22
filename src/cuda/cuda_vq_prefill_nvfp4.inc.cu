/* cuda_vq_prefill_nvfp4.inc.cu — 预填专家走 spark 原生 FP4 张量核(speed.md 段 5, 2026-09-15)。
 *
 * 【为什么要这条路】现役的融合路(cuda_vq_prefill_fused.inc.cu)把 VQ 解码和乘法揉在一个核里, 好处是
 * 权重不落暂存, 坏处是**码本 64 KB 必须待在 shared**, 于是每个 SM 只驻得下 1 个 block, 块大小就是占用率,
 * 每线程只剩 64 个寄存器。09-15 实撞三次: 一个工作项只带 8 个 token(NT=8)已经把寄存器占满, NT=16/32、
 * 连把 NT 做成编译期常量都报 "too many resources requested for launch" 起不来。
 * 后果是**权重被重读 ⌈ne/8⌉ 遍**(ne = 这个专家摊到几个 token): 块 512 时 ne≈8 读一遍, 块 2048 时 ne≈32
 * 读四遍 —— 这就是"块开大了预填一点不省"(512→1024 只快 1%)的根子。实测 398 ms/层, 而读一遍 blob 只要 11 ms。
 *
 * 【这条路怎么绕开】把解码和乘法拆成两步, 码本就不用待在 shared 里了(它 64 KB, 正好落在每个 SM 128 KB 的 L1 里,
 * 一个 block 只处理一个专家 ⇒ 全是 L1 命中):
 *   ① 专家权重 VQ → **NVFP4**(E2M1 + 每 16 个元素一个 E4M3 缩放)写进暂存, 每个专家每个矩阵 5.9 MB;
 *   ② 走 `v41_gemm_nvfp4`(cuBLASLt 块缩放 FP4 matmul) —— 这是 S0 实测本机**唯一**能摸到 284~356 TFLOPS
 *      的路(gguf-tools/bench/gemm_fp4_ceiling.cu: BF16 只有 81~91, FP8 76~188, 我们盘上的 MXFP4 根本没算法)。
 * 账: 每层写 7.65 GB + 读 7.65 GB = 15.3 GB ⇒ 240 GB/s 下约 64 ms/层, 与 token 数**无关**(权重只解一遍),
 * 对现役的 398 ms/层 是 6 倍。token 越多这条路越划算, 所以它和"块开大"是一件事。
 *
 * 【数值】VQ 码本值是 f16, 乘行增益(含反修覆盖)后落 NVFP4 格点 —— 比现役融合路(f32 全程)多一次量化。
 * 这正是 speed.md §2 定的"计算态解到 FP4", 判据是五指标/NLL, 不是逐位。
 *
 * 【出错会怎样】缩放张量必须按 128×4 swizzle 补齐并清零(见 v41_nvfp4_scale_off)。少分配不报错, 只让张量核
 * 越界读, 表现是随机几行输出是垃圾; 补齐区不清零则读到脏字节, 可能是 NaN。 */

#define V41_VQN_PAD 2048u   /* n 对齐的上限, 也是调用方要在输出缓冲尾部多留的行数 */

/* 一个专家一个矩阵的 NVFP4 暂存(双缓冲: 解下一个专家的同时上一个还在算) */
typedef struct { v41_scratch nib, sc; } vqn_stage;
static vqn_stage g_vqn[2];
static v41_scratch g_vqn_xnib, g_vqn_xsc, g_vqn_hnib, g_vqn_hsc;

/* VQ 位流 → NVFP4。一线程一个 16 元素块(= 两个码本条目, 码本条目是 8 个 half)。
 * row/col 的含义与 v41_vq_open 一致: 矩阵是 [rows][cols], 位流按 (r·nidx_row + j) 取索引, j 是第 j 组 8 个元素。 */
__global__ static void vqn_to_nvfp4_kernel(uint8_t *nib, uint8_t *sc, const uint8_t *blob, int e, int which,
                                           uint32_t rows, uint32_t cols, const float *gov, uint32_t nkb,
                                           unsigned long long *clamped) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (uint64_t)rows * nkb) return;
    const uint32_t r = (uint32_t)(b / nkb), c = (uint32_t)(b % nkb);
    /* ★这条路只认 DQVL v2★(2026-09-21): 它自 09-20 判负后没有调用者(存档在 cuda_vq_prefill_fused.inc.cu 的注释里),
     * 所以固定 v2 实例。要重开它, 除了那笔 0.013 Σmin 的账, 还得先给 v3 补上(层码本 + E4M3 + 13 位位平面)。 */
    const v41_vq_mat m = v41_vq_open<0>(blob, e, which, rows, cols, gov);
    if (!m.ok) return;
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
    float v[V41_NVFP4_BLK];
    const uint64_t i0 = (uint64_t)r * m.nidx_row + (uint64_t)c * 2u;   /* 16 个元素 = 连着两组 */
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const uint64_t bit = (i0 + (uint64_t)half) * m.nbit, by = bit >> 3;
        uint32_t wv; memcpy(&wv, m.ix + by, 4);
        const uint32_t idx = (wv >> (bit & 7)) & m.imsk;
        /* ★码本在全局内存里只保证 8 字节对齐, 不能用 uint4★(仓里 v41_vq_row_dot 的注释早写明了)。
         * 用 uint4 读会报 "misaligned address" —— 而且是**异步**冒出来的, 现场看到的是后面某次
         * cudaMalloc 说"分配失败 0.0 MB", 跟真因差着十万八千里。两次 uint2 就对齐了。 */
        const uint2 c0 = *(const uint2 *)(m.cb + (size_t)idx * 16u);
        const uint2 c1 = *(const uint2 *)(m.cb + (size_t)idx * 16u + 8u);
        __half2 h[4]; memcpy(&h[0], &c0.x, 4); memcpy(&h[1], &c0.y, 4); memcpy(&h[2], &c1.x, 4); memcpy(&h[3], &c1.y, 4);
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            const float2 f = __half22float2(h[t]);
            v[half * 8 + t * 2] = f.x * g; v[half * 8 + t * 2 + 1] = f.y * g;
        }
    }
    float amax = 6.0f * exp2f(-9.0f);   /* 与 v41_x_to_nvfp4_kernel 同口径: 全零块的缩放也不许是 0 */
    #pragma unroll
    for (int j = 0; j < (int)V41_NVFP4_BLK; j++) amax = fmaxf(amax, fabsf(v[j]));
    const float s = ds4_e4m3fn_round(amax / 6.0f), inv = s > 0.0f ? 1.0f / s : 0.0f;
    uint8_t *dst = nib + (uint64_t)r * (cols / 2u) + (uint64_t)c * (V41_NVFP4_BLK / 2u);
    #pragma unroll
    for (int j = 0; j < (int)V41_NVFP4_BLK / 2; j++) {
        const float a = fminf(fmaxf(v[2 * j] * inv, -6.0f), 6.0f);
        const float bb = fminf(fmaxf(v[2 * j + 1] * inv, -6.0f), 6.0f);
        dst[j] = (uint8_t)(ds4_fp4_f32_to_nibble(a) | (ds4_fp4_f32_to_nibble(bb) << 4));
    }
    uint8_t sb = 0;
    for (int j = 1; j < 127; j++) if (ds4_e4m3fn_value(j) == s) { sb = (uint8_t)j; break; }
    sc[v41_nvfp4_scale_off(r, c, nkb)] = sb;
    (void)clamped;
}

/* 激活的一段(排序后 [t0, t0+nt) 行) → NVFP4。与 v41_x_to_nvfp4_kernel 同式, 只是带行偏移。 */
__global__ static void vqn_xslice_to_nvfp4_kernel(uint8_t *nib, uint8_t *sc, const float *xs,
                                                  uint32_t row0, uint32_t nt, uint32_t npad, uint32_t dim, uint32_t nkb) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (uint64_t)npad * nkb) return;
    const uint32_t r = (uint32_t)(b / nkb), c = (uint32_t)(b % nkb);
    if (r >= nt) {   /* 补齐行: 写零(它的输出落在下一个专家的行区, 随后被覆盖) */
        uint8_t *z = nib + (uint64_t)r * (dim / 2u) + (uint64_t)c * (V41_NVFP4_BLK / 2u);
        for (int j = 0; j < (int)V41_NVFP4_BLK / 2; j++) z[j] = 0;
        sc[v41_nvfp4_scale_off(r, c, nkb)] = 0;
        return;
    }
    const float *src = xs + (uint64_t)(row0 + r) * dim + (uint64_t)c * V41_NVFP4_BLK;
    float amax = 6.0f * exp2f(-9.0f);
    for (int j = 0; j < (int)V41_NVFP4_BLK; j++) amax = fmaxf(amax, fabsf(src[j]));
    const float s = ds4_e4m3fn_round(amax / 6.0f), inv = s > 0.0f ? 1.0f / s : 0.0f;
    uint8_t *dst = nib + (uint64_t)r * (dim / 2u) + (uint64_t)c * (V41_NVFP4_BLK / 2u);
    for (int j = 0; j < (int)V41_NVFP4_BLK / 2; j++) {
        const float a = fminf(fmaxf(src[2 * j] * inv, -6.0f), 6.0f);
        const float bb = fminf(fmaxf(src[2 * j + 1] * inv, -6.0f), 6.0f);
        dst[j] = (uint8_t)(ds4_fp4_f32_to_nibble(a) | (ds4_fp4_f32_to_nibble(bb) << 4));
    }
    uint8_t sb = 0;
    for (int j = 1; j < 127; j++) if (ds4_e4m3fn_value(j) == s) { sb = (uint8_t)j; break; }
    sc[v41_nvfp4_scale_off(r, c, nkb)] = sb;
}

/* 一个专家的一个矩阵: VQ → NVFP4 暂存(槽 slot), 然后 GEMM 到 dst 的第 row0 行起。
 * A = 权重 [rows][cols](m = rows), B = 激活片 [nt][cols](n = nt), k = cols ⇒ D = [nt][rows] 行主序。 */
static int vqn_mat_gemm(uint8_t *xnib, uint8_t *xsc, const uint8_t *blob, int e, int which,
                        uint32_t rows, uint32_t cols, const float *gov, uint32_t nt, float *dst,
                        int slot, const char *what) {
    const uint32_t nkb = cols / V41_NVFP4_BLK;
    const uint64_t sc_n = ((uint64_t)(rows + 127u) / 128u * 128u) * ((nkb + 3u) / 4u * 4u);
    uint8_t *wnib = (uint8_t *)v41_grow(&g_vqn[slot].nib, (uint64_t)rows * (cols / 2u), "vqn 权重");
    uint8_t *wsc  = (uint8_t *)v41_grow(&g_vqn[slot].sc, sc_n, "vqn 权重缩放");
    if (!wnib || !wsc) return 0;
    (void)cudaMemsetAsync(wsc, 0, sc_n, g_cur_stream);   /* 补齐区必须清零, 张量核会读到 */
    const uint64_t nb = (uint64_t)rows * nkb;
    vqn_to_nvfp4_kernel<<<(unsigned)((nb + 255) / 256), 256, 0, g_cur_stream>>>(
        wnib, wsc, blob, e, which, rows, cols, gov, nkb, g_v41_nvfp4_clamped);
    if (!cuda_ok(cudaGetLastError(), "vqn vq→nvfp4")) return 0;
    return v41_gemm_nvfp4(wnib, wsc, xnib, xsc, dst, (int)rows, (int)nt, (int)cols, (int)rows, what);
}

/* 段 5 主体: 按专家逐个走"解到 NVFP4 → 原生 FP4 张量核"。
 * 复用融合路的脚手架(perm/off/cnt 排序 + gather32 + reduce), 只换中间这一段。
 * 返回 0 = 本路不可用, 调用方回融合路。 */
static int vqn_prefill_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert,
                           uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nvalid, float clamp,
                           uint32_t layer_index, const float *xs32, float *g32, float *h32, float *ys) {
    if ((IN % V41_NVFP4_BLK) || (MID % V41_NVFP4_BLK) || !nvalid) return 0;
    if (!g_v41_nvfp4_clamped) (void)cudaMalloc(&g_v41_nvfp4_clamped, sizeof(unsigned long long));
    const float *gr_l = g_v41_gr[layer_index < 64u ? layer_index : 0];
    for (uint32_t e = 0, slot = 0; e < n_total_expert; e++) {
        const uint32_t nt = cnt[e], row0 = off_h[e];
        if (!nt) continue;
        /* ★n 对齐到 2 的幂★: 每个专家摊到的 token 数都不一样, 直接拿它当 GEMM 的 n 会出几百种形状,
         * 把 cuBLASLt 的算法缓存撑爆(实撞: "算法缓存满(24 种形状)")。对齐后只剩十几种。
         * 代价几乎为零 —— 这个 GEMM 是**权重带宽**受限的(权重 5.9 MB, 激活才 92 KB), n 翻倍不影响。
         * 补齐行的输出落在下一个专家的行区, 而专家按序处理, 下一个专家随后就把它覆盖掉;
         * 最后一个专家的补齐落在缓冲尾部预留的 V41_VQN_PAD 行里(调用方多分配了这么多)。 */
        uint32_t npad = 8u; while (npad < nt) npad <<= 1;
        if (npad > V41_VQN_PAD) npad = V41_VQN_PAD;
        /* 激活片按专家单独转一次: 缩放张量是 128 行一块的 swizzle 布局, 从中间某一行切片会错位 */
        const uint32_t xnkb = IN / V41_NVFP4_BLK, hnkb = MID / V41_NVFP4_BLK;
        const uint64_t xsc_n = ((uint64_t)(npad + 127u) / 128u * 128u) * ((xnkb + 3u) / 4u * 4u);
        uint8_t *xnib = (uint8_t *)v41_grow(&g_vqn_xnib, (uint64_t)npad * (IN / 2u), "vqn 激活");
        uint8_t *xsc  = (uint8_t *)v41_grow(&g_vqn_xsc, xsc_n, "vqn 激活缩放");
        if (!xnib || !xsc) return 0;
        (void)cudaMemsetAsync(xsc, 0, xsc_n, g_cur_stream);
        vqn_xslice_to_nvfp4_kernel<<<(unsigned)(((uint64_t)npad * xnkb + 255) / 256), 256, 0, g_cur_stream>>>(
            xnib, xsc, xs32, row0, nt, npad, IN, xnkb);
        if (!cuda_ok(cudaGetLastError(), "vqn x→nvfp4")) return 0;
        /* gate → g32, up → h32, swiglu 就地, down → ys */
        if (!vqn_mat_gemm(xnib, xsc, blob, (int)e, 0, MID, IN, NULL, npad, g32 + (uint64_t)row0 * MID,
                          (int)(slot & 1u), "vqn gate")) return 0;
        if (!vqn_mat_gemm(xnib, xsc, blob, (int)e, 1, MID, IN, NULL, npad, h32 + (uint64_t)row0 * MID,
                          (int)((slot + 1u) & 1u), "vqn up")) return 0;
        {   /* swiglu 就地: h = silu(clamp(g)) · clamp(u); 直接用 v41_3 里那个核, 不另造入口 */
            const uint64_t hn = (uint64_t)nt * MID;
            v41_swiglu_kernel<<<(unsigned)((hn + 255) / 256), 256, 0, g_cur_stream>>>(
                h32 + (uint64_t)row0 * MID, g32 + (uint64_t)row0 * MID,
                h32 + (uint64_t)row0 * MID, hn, clamp);
            if (!cuda_ok(cudaGetLastError(), "vqn swiglu")) return 0;
        }
        /* h 片已是连续 nt 行, 直接当激活转 NVFP4(借同一组暂存) */
        const uint64_t hsc_n = ((uint64_t)(npad + 127u) / 128u * 128u) * ((hnkb + 3u) / 4u * 4u);
        uint8_t *hnib = (uint8_t *)v41_grow(&g_vqn_hnib, (uint64_t)npad * (MID / 2u), "vqn h");
        uint8_t *hsc  = (uint8_t *)v41_grow(&g_vqn_hsc, hsc_n, "vqn h 缩放");
        if (!hnib || !hsc) return 0;
        (void)cudaMemsetAsync(hsc, 0, hsc_n, g_cur_stream);
        vqn_xslice_to_nvfp4_kernel<<<(unsigned)(((uint64_t)npad * hnkb + 255) / 256), 256, 0, g_cur_stream>>>(
            hnib, hsc, h32, row0, nt, npad, MID, hnkb);
        if (!cuda_ok(cudaGetLastError(), "vqn h→nvfp4")) return 0;
        if (!vqn_mat_gemm(hnib, hsc, blob, (int)e, 2, OUT, MID, gr_l ? gr_l + (size_t)e * OUT : NULL, npad,
                          ys + (uint64_t)row0 * OUT, (int)(slot & 1u), "vqn down")) return 0;
        slot++;
    }
    return 1;
}
