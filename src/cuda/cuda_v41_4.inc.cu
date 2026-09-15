/* cuda_v41_4.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 解码小批(n ≤ 8)融合核(2026-09-12 战役 P4)。
 *
 * 为什么: P2c 首跑解码 1.24 t/s, 账在"每次乘法先把整个权重矩阵 dequant 成 f16 再 cuBLAS":
 *   骨架 fp4x32 0.53 B/元素 → 写 2 B + 读 2 B, 流量放大 8.5×(≈32 GB/token); 专家更糟: 6 专家×40 层的 f16 落地
 *   约 34 GB/token, 而压缩态 blob 本身只有 ~1.6 GB/token。那份 f16 只用一次就丢, 根本不该存在。
 * 这里对 n ≤ 8(解码/小批)直接从压缩态算。数值口径: 权重精确值(fp4×2^e / 码本×增益 f32 乘), 激活 bf16 格点,
 * f32 累加 —— 与 cuBLAS 路只差累加序。专家内按官方 Expert 的 bf16 边界舍: gate/up 出 bf16, h 出 bf16, down 出 bf16,
 * 再按路由权重 f32 加。
 * 三版演进(nsys 定的): ①lane 隔 32 块取整块 → 每条字节 load 跨 17 扇区, 55 GB/s; ②整 warp 逐块、lane 取块内第 l 个元素
 * → 扇区对了但小矩阵(kv 512 行)只有 64 个 block 填不满 48 个 SM, 仍 57 GB/s; ③(本版)K 维切给 ksplit 个 warp 共算一行
 * + wo_a 八组一发 + bf16 舍入并进尾巴。VQ 侧: ①码本 L1 全局 gather(每层 1.7 千万次随机 16 B, 扇区流量 566 MB/层)
 * → ②(本版)码本整块搬进 shared(64 KB/矩阵, sm_121 动态 shared 上限 227 KB), 一 block 32 行摊薄搬运。 */

/* ---- fp4x32 GEMV ----
 * grid (ceil(out_dim/(8/ksplit)), n_groups), block 256 = 8 warp; 同一行由 ksplit 个 warp 分 K 段(块号 ≡ kpart mod ksplit),
 * 各自 lane 取块内第 l 个元素(半字节 p[l/2] + 共用 scale p[16]; x 读 128 B 连续, 权重读同扇区), 段和经 shared 按固定序相加。
 * n_groups>1 = 块对角(wo_a): 第 g 组权重/输入/输出各按步长偏移。round_out: 出口直接舍 bf16(省一次 round 核)。 */
__global__ static void v41_fp4x32_gemv_kernel(float *out, const uint8_t *w, const float *x, uint32_t in_dim, uint32_t out_dim,
                                              uint32_t n_tok, uint32_t x_stride, uint32_t out_stride, uint32_t ksplit,
                                              uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out) {
    const uint32_t g = blockIdx.y;
    w += (uint64_t)g * w_gstride; x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][V41_GEMV_MAX_TOK];
    float acc[V41_GEMV_MAX_TOK];
    #pragma unroll
    for (uint32_t t = 0; t < V41_GEMV_MAX_TOK; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const uint32_t nblk = in_dim / 32u;
        const uint8_t *wr = w + (uint64_t)r * nblk * 17u;
        const uint32_t half = lane >> 1, hi = lane & 1u;
        for (uint32_t b = kpart; b < nblk; b += ksplit) {
            const uint8_t *p = wr + (uint64_t)b * 17u;
            const uint8_t byte = p[half];
            const float wv = ds4_fp4_nibble_to_f32(hi ? (byte >> 4) : (byte & 0x0F)) * ds4_e8m0_to_f32(p[16]);
            const float *xb = x + b * 32u + lane;
            #pragma unroll
            for (uint32_t t = 0; t < V41_GEMV_MAX_TOK; t++)
                if (t < n_tok) acc[t] += wv * v41_bf16r(xb[(uint64_t)t * x_stride]);
        }
        #pragma unroll
        for (uint32_t t = 0; t < V41_GEMV_MAX_TOK; t++) {
            if (t >= n_tok) break;
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) { for (uint32_t t = 0; t < n_tok; t++) red[warp][t] = acc[t]; }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            for (uint32_t t = 0; t < n_tok; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(v) : v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        for (uint32_t t = 0; t < n_tok; t++) out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(acc[t]) : acc[t];
    }
}
static int v41_fp4x32_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                           const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                           uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what) {
    if ((in_dim % 32u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || n_groups == 0) return 0;
    const uint64_t wg = out_dim * (in_dim / 32u) * 17u, wbytes = wg * n_groups;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    uint32_t ksplit = 1;   /* 小矩阵填不满 GPU: 行数×ksplit ≥ 2048 个 warp(48 SM × ~40 warp) */
    while (ksplit < 8u && out_dim * ksplit * n_groups < 2048u) ksplit <<= 1;
    const uint32_t rpb = 8u / ksplit;
    v41_fp4x32_gemv_kernel<<<dim3((unsigned)((out_dim + rpb - 1u) / rpb), n_groups), 256, 0, g_cur_stream>>>(
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, n_tok, x_stride, out_stride, ksplit, wg, x_gstride, out_gstride, round_out);
    return cuda_ok(cudaGetLastError(), what);
}

/* ---- VQ 专家解码即乘(DQVL v2 blob: 16 B 头 + 384×3 槽表; 槽 = DQVQ 载荷: 16 B 头 + 码本 + 行增益 f16 + 位流) ---- */
typedef struct { const uint8_t *cb, *gr, *ix; const float *gov; uint32_t nidx_row, nbit, imsk, nc; int ok; } v41_vq_mat;
/* gov(2026-09-13): 该专家该矩阵的逐行增益【缩放因子】[rows] f32, NULL = 按载荷原样(权重侧反修没挂或不挂这个矩阵)。 */
__device__ __forceinline__ static v41_vq_mat v41_vq_open(const uint8_t *blob, int e, int which, uint32_t rows, uint32_t cols,
                                                         const float *gov) {
    v41_vq_mat m; m.ok = 0; m.cb = m.gr = m.ix = NULL; m.gov = gov; m.nidx_row = m.nbit = m.imsk = m.nc = 0;
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    if (!off) return m;
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return m;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t r32, c32; memcpy(&r32, pay + 8, 4); memcpy(&c32, pay + 12, 4);
    if (r32 != rows || c32 != cols || d16 != 8u) return m;   /* 本核只写了 dim 8(V4.1 配方 vq8x4096) */
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    m.cb = pay + 16; m.gr = m.cb + (size_t)n16 * 8u * 2u; m.ix = m.gr + (size_t)rows * 2u;
    m.nidx_row = cols / 8u; m.nbit = nbit; m.imsk = (1u << nbit) - 1u; m.nc = n16; m.ok = 1;
    return m;
}
/* 一 warp 算一行与激活 x 的点积: 向量 j 归 lane j%32(同一轮 32 个 lane 的位流读落在 48 B 连续段内); 码本条目 16 B
 * 从 cbs 取(shared 版 16 B 对齐一次 uint4; 全局版载荷只保证 8 B 对齐, 两次 uint2)。返回 增益×Σ(已 warp 规约)。 */
__device__ __forceinline__ static float v41_vq_row_dot(const v41_vq_mat &m, uint32_t r, const float *xs, const uint8_t *cbs, int cb_shared) {
    const uint64_t i0 = (uint64_t)r * m.nidx_row;
    float acc = 0.f;
    #pragma unroll 4
    for (uint32_t j = threadIdx.x & 31u; j < m.nidx_row; j += 32u) {
        const uint64_t bit = (i0 + j) * m.nbit, by = bit >> 3;
        uint32_t wv; memcpy(&wv, m.ix + by, 4);
        const uint32_t v = (wv >> (bit & 7)) & m.imsk;
        uint2 cw0, cw1;
        if (cb_shared) { const uint4 c = *(const uint4 *)(cbs + (size_t)v * 16u); cw0.x = c.x; cw0.y = c.y; cw1.x = c.z; cw1.y = c.w; }
        else { cw0 = *(const uint2 *)(cbs + (size_t)v * 16u); cw1 = *(const uint2 *)(cbs + (size_t)v * 16u + 8u); }
        const float4 xa = *(const float4 *)(xs + (size_t)j * 8u), xb = *(const float4 *)(xs + (size_t)j * 8u + 4u);
        __half2 h0, h1, h2, h3; memcpy(&h0, &cw0.x, 4); memcpy(&h1, &cw0.y, 4); memcpy(&h2, &cw1.x, 4); memcpy(&h3, &cw1.y, 4);
        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
        acc += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w + f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    return acc * __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
}
__device__ __forceinline__ static void v41_vq_cb_to_shared(uint8_t *dst, const uint8_t *src, uint32_t bytes) {   /* bytes 是 8 的倍数 */
    const uint2 *s = (const uint2 *)src; uint2 *d = (uint2 *)dst;
    for (uint32_t i = threadIdx.x; i < bytes / 8u; i += blockDim.x) d[i] = s[i];
}
/* ★shared 码本版的一 block 行数★(2026-09-14 二改)。
 * 一 block 32 warp, 每 warp 再循环 V41_VQ_ITERS 行 ⇒ 一 block 管 32×ITERS 行, 码本只搬一次。
 * 为什么必须循环而不是一 block 32 行: 码本 64 KB 占满设备每 block 动态 shared 上限(GB10 = 99 KB)的一大半,
 * 每个 SM 只塞得下 1 个 block; 一 block 只算 32 行就重搬一次 64 KB, 而这 32 行的索引流才 30 KB ——
 * 搬运是有效数据的 2 倍多, 40 层累计好几 GB。行数翻 8 倍, 码本搬运就摊薄 8 倍, 有效带宽利用直接上去。 */
#define V41_VQ_ITERS   8u
#define V41_VQ_SH_ROWS (32u * V41_VQ_ITERS)
/* gate/up 同核(官方 Expert: w1/w3 出 bf16 → f32 截断 → silu(g)·u → bf16); cb_bytes>0: 码本进 shared(grid.x 按 32 行),
 * 否则全局 gather(grid.x 按 8 行)。x 已是 bf16 格点(调用方 rms_norm 出口舍过)。 */
__global__ static void v41_vq_gateup_kernel(float *h, const uint8_t *blob, const int32_t *sel, const float *x,
                                            uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = blockIdx.y, t = pair / K, rows = cb_bytes ? V41_VQ_SH_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    const v41_vq_mat mg = v41_vq_open(blob, e, 0, MID, IN, NULL), mu = v41_vq_open(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS : 1u;
    const float *xs = x + (uint64_t)t * IN;
    float g[V41_VQ_ITERS], u[V41_VQ_ITERS];
    #pragma unroll
    for (uint32_t i = 0; i < V41_VQ_ITERS; i++) { g[i] = 0.f; u[i] = 0.f; }
    if (cb_bytes) {
        /* ★一块 shared 用两遍★(2026-09-14): gate 与 up 各有一本 64 KB 码本, 一次性放两本要 128 KB,
         * 超过 GB10 每 block 的上限(99 KB) ⇒ 整个核被打回全局 gather。改成先载 gate 算完这 block 的
         * 全部行, 同步后把同一块 shared 覆盖成 up 的码本再算 u: 峰值只要一本的量。
         * ★循环里不能 return★: 后面还有 __syncthreads, 少一个 warp 就死锁, 越界的行只跳过计算。 */
        if (mg.nc * 16u != cb_bytes || mu.nc * 16u != cb_bytes) return;
        v41_vq_cb_to_shared(vqsh, mg.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) { const uint32_t r = r0 + i * 32u; if (r < MID) g[i] = v41_bf16r(v41_vq_row_dot(mg, r, xs, vqsh, 1)); }
        __syncthreads();                            /* 等所有 warp 读完 gate 码本, 才能覆盖它 */
        v41_vq_cb_to_shared(vqsh, mu.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) { const uint32_t r = r0 + i * 32u; if (r < MID) u[i] = v41_bf16r(v41_vq_row_dot(mu, r, xs, vqsh, 1)); }
    } else if (r0 < MID) {
        g[0] = v41_bf16r(v41_vq_row_dot(mg, r0, xs, mg.cb, 0));
        u[0] = v41_bf16r(v41_vq_row_dot(mu, r0, xs, mu.cb, 0));
    }
    for (uint32_t i = 0; i < nit; i++) {
        const uint32_t r = r0 + i * 32u;
        if (r >= MID) break;
        float gi = g[i], ui = u[i];
        if (clamp > 0.f) { if (gi > clamp) gi = clamp; if (ui > clamp) ui = clamp; if (ui < -clamp) ui = -clamp; }
        const float sg = gi / (1.0f + expf(-gi));
        if ((threadIdx.x & 31u) == 0) h[(uint64_t)pair * MID + r] = v41_bf16r(sg * ui);
    }
}
/* down: partial[pair][OUT] = bf16(W2·h) */
__global__ static void v41_vq_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const float *h,
                                          uint32_t MID, uint32_t OUT, uint32_t K, uint32_t cb_bytes, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = blockIdx.y, rows = cb_bytes ? V41_VQ_SH_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    const v41_vq_mat md = v41_vq_open(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS : 1u;
    if (!md.ok) {   /* 载荷不对: 这 block 负责的行全写 0(不能只写一行, 下游 reduce 会读到脏值) */
        for (uint32_t i = 0; i < nit; i++) { const uint32_t r = r0 + i * 32u; if (r < OUT && (threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = 0.f; }
        return;
    }
    const uint8_t *cbd = md.cb;
    if (cb_bytes) {
        if (md.nc * 16u != cb_bytes) return;
        v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
        __syncthreads();
        cbd = vqsh;
    }
    const float *hs = h + (uint64_t)pair * MID;
    for (uint32_t i = 0; i < nit; i++) {   /* 一 block 管 32×ITERS 行, 码本只搬一次(见 V41_VQ_ITERS 注释) */
        const uint32_t r = r0 + i * 32u;
        if (r >= OUT) break;
        const float y = v41_bf16r(v41_vq_row_dot(md, r, hs, cbd, cb_bytes != 0));
        if ((threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = y;
    }
    (void)K;
}
/* out[t][o] = Σ_k w[t][k]·partial[t·K+k][o](f32, 官方 y += weights·expert_out) */
__global__ static void v41_vq_reduce_kernel(float *out, const float *partial, const float *w, uint32_t K, uint32_t OUT) {
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= OUT) return;
    float a = 0.f;
    for (uint32_t k = 0; k < K; k++) a += w[(uint64_t)t * K + k] * partial[((uint64_t)t * K + k) * OUT + o];
    out[(uint64_t)t * OUT + o] = a;
}
static v41_scratch g_v41_vq_h, g_v41_vq_part;
/* ★两个核分开判定★(2026-09-14 实撞): 码本 4096×16 B = 64 KB/本。gateup 要 gate+up 两本 = 128 KB,
 * 超过 GB10 每 block 的动态 shared 上限; down 只要一本 64 KB, 本来放得下。原先一个 ok 变量把两个核
 * 绑在一起, gateup 申请失败就把 down 一起打回全局 gather —— 解码实测只有 1.60 t/s(prefill 45 t/s 正常)。
 * 0 未判定, 1 可用, -1 不可用。 */
static int g_v41_vq_sh_gateup = 0, g_v41_vq_sh_down = 0;
static int v41_vq_fused_moe(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                            const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                            const float *gr) {
    if (n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || (IN % 8u) || (MID % 8u)) return 0;
    const uint64_t np = (uint64_t)n_tok * K;
    float *h = (float *)v41_grow(&g_v41_vq_h, np * MID * 4, "v41 vq h");
    float *part = (float *)v41_grow(&g_v41_vq_part, np * OUT * 4, "v41 vq partial");
    if (!h || !part) return 0;
    const uint32_t cbb = nc * 16u;
    if (!g_v41_vq_sh_gateup) {   /* 一次性, 两个核各问各的: gateup 要两本(gate+up), down 只要一本 */
        int cap = 0; (void)cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        const bool og = cudaFuncSetAttribute(v41_vq_gateup_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        const bool od = cudaFuncSetAttribute(v41_vq_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        (void)cudaGetLastError();
        g_v41_vq_sh_gateup = og ? 1 : -1;
        g_v41_vq_sh_down = od ? 1 : -1;
        fprintf(stderr, "ds4: [v41] VQ 码本 %u×16 B(%u KB/本); 设备每 block 动态 shared 上限 %d KB ⇒ gate+up %s / down %s\n",
                nc, cbb >> 10, cap >> 10, og ? "进 shared(一块用两遍)" : "回全局 gather", od ? "进 shared" : "回全局 gather");
    }
    const bool shg = g_v41_vq_sh_gateup == 1, shd = g_v41_vq_sh_down == 1;
    /* 线程数恒为 32 warp(shared 版一 warp 循环 ITERS 行)或 8 warp(全局 gather 版一 warp 一行);
     * ★不能写成 rows×32★: shared 版 rows 已是 256, 那会要 8192 个线程, 超过每 block 1024 的上限。 */
    const uint32_t rg = shg ? V41_VQ_SH_ROWS : 8u, rd = shd ? V41_VQ_SH_ROWS : 8u;
    const uint32_t tg = shg ? 32u * 32u : 8u * 32u, td = shd ? 32u * 32u : 8u * 32u;
    v41_vq_gateup_kernel<<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, g_cur_stream>>>(h, blob, sel, x, IN, MID, K, clamp, shg ? cbb : 0u);
    if (!cuda_ok(cudaGetLastError(), "v41 vq gateup")) return 0;
    v41_vq_down_kernel<<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, g_cur_stream>>>(part, blob, sel, h, MID, OUT, K, shd ? cbb : 0u, gr);
    if (!cuda_ok(cudaGetLastError(), "v41 vq down")) return 0;
    v41_vq_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>(out, part, w, K, OUT);
    return cuda_ok(cudaGetLastError(), "v41 vq reduce");
}

/* ---- engram 查表行 dequant(fp8 e4m3 × ue8m0/32 → bf16 格点): 一线程一元素 ---- */
__global__ static void v41_engram_rows_kernel(float *out, const uint8_t *raw, uint32_t n_rows, uint32_t hd) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_rows * hd) return;
    const uint32_t r = (uint32_t)(i / hd), d = (uint32_t)(i % hd), stride = hd + hd / 32u;
    const uint8_t *row = raw + (uint64_t)r * stride;
    out[i] = v41_bf16r(ds4_e4m3fn_to_f32(row[d]) * ds4_e8m0_to_f32(row[hd + d / 32u]));
}
int ds4_gpu_v41_engram_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *raw, uint32_t n_rows, uint32_t head_dim) {
    if (!out || !raw || (head_dim % 32u)) return 0;
    const uint64_t n = (uint64_t)n_rows * head_dim;
    if (out->bytes < n * 4 || raw->bytes < (uint64_t)n_rows * (head_dim + head_dim / 32u)) return 0;
    v41_engram_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)out->ptr, (const uint8_t *)raw->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "v41 engram rows");
}

/* ---- 反修放大器: T(n×K) = x·B, y += T·A ----
 * 列主序推导: x 内存行主序 n×D = 列主序 xᵀ(D×n, ld D); B 内存 [K][D] 行主序 = 列主序 B_math(D×K, ld D);
 * Tᵀ(K×n) = B_mathᵀ·xᵀ ⇒ Sgemm(T, N, K, n, D); A 内存 [K][D] 行主序 = 列主序 Aᵀ(D×K, ld D);
 * yᵀ(D×n) += Aᵀ·Tᵀ ⇒ Sgemm(N, N, D, n, K, beta=1)。 */
int ds4_gpu_v41_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                 ds4_gpu_tensor *T, uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!y || !x || !A || !B || !T || !g_cublas_ready || n_tok == 0 || K == 0) return 0;
    if (T->bytes < (uint64_t)n_tok * K * 4 || y->bytes < (uint64_t)n_tok * D * 4 || x->bytes < (uint64_t)n_tok * D * 4) return 0;
    const float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &a1,
                                    (const float *)B->ptr, (int)D, (const float *)x->ptr, (int)D, &b0, (float *)T->ptr, (int)K);
    if (!cublas_ok(st, "v41 amp T=xB")) return 0;
    st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n_tok, (int)K, &a1,
                     (const float *)A->ptr, (int)D, (const float *)T->ptr, (int)K, &b1, (float *)y->ptr, (int)D);
    return cublas_ok(st, "v41 amp y+=TA");
}

/* ---- 小件: 就地 x = bf16(x·s)(indexer weights 缩放, 原先主机往返) / argmax(原先读回 517 KB 主机扫) ---- */
__global__ static void v41_scale_round_kernel(float *x, uint64_t n, float s) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v41_bf16r(x[i] * s);
}
int ds4_gpu_v41_scale_round_tensor(ds4_gpu_tensor *x, uint64_t n, float s) {
    if (!x || x->bytes < n * 4) return 0;
    v41_scale_round_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)x->ptr, n, s);
    return cuda_ok(cudaGetLastError(), "v41 scale round");
}
__global__ static void v41_argmax_kernel(int32_t *idx, const float *row, uint32_t n) {
    __shared__ float bv[1024]; __shared__ int32_t bi[1024];
    float best = -INFINITY; int32_t besti = 0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = row[i]; if (v > best) { best = v; besti = (int32_t)i; } }
    bv[threadIdx.x] = best; bi[threadIdx.x] = besti; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {   /* 同值取小下标(与主机顺序扫一致) */
        if (threadIdx.x < k) {
            const float ov = bv[threadIdx.x + k]; const int32_t oi = bi[threadIdx.x + k];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = ov; bi[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) idx[0] = bi[0];
}
int ds4_gpu_v41_argmax_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *logits, uint32_t row, uint32_t n_vocab) {
    if (!idx || !logits || idx->bytes < 4 || logits->bytes < ((uint64_t)row + 1) * n_vocab * 4) return 0;
    v41_argmax_kernel<<<1, 1024, 0, g_cur_stream>>>((int32_t *)idx->ptr, (const float *)logits->ptr + (uint64_t)row * n_vocab, n_vocab);
    return cuda_ok(cudaGetLastError(), "v41 argmax");
}
