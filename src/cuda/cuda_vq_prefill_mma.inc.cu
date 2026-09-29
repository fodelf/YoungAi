/* cuda_vq_prefill_mma.inc.cu — 预填专家(DQVL v3)走 bf16 张量核: VQ 码字当场解成 bf16 瓦片, mma 乘(2026-09-24; 09-29 换形态)。
 *
 * 【为什么】09-24 12k 提示逐核表(216.5 t/s): 专家融合核 vqp_fused_gu/down 合计 42.1 s = 整段 GPU 时间的 76%。
 * 那条路是标量 f32 乘加: 一层 512 token × top-6 × 35.4M 权重 × 2 = 217 GFLOP, 实测 ~80 ms/层 ≈ 2.6 TFLOPS,
 * 而读一遍本层位流只要 ~11 ms。它慢在算, 不在读 —— 码本 64 KB 占满 shared, 每 SM 只驻 1 个 block,
 * 每线程 64 个寄存器, 一个工作项只能带 8 个 token(见 cuda_vq_prefill_fused.inc.cu 的判负存档)。
 * 张量核把"算"这一项拿掉: 同样的乘加换成 mma.m16n8k16, 每 warp 一条指令 4096 次乘加。
 *
 * 【为什么是 bf16, 而且与融合路同一组乘积】(09-19 的教训: NVFP4 路把激活压成 4 bit, 白掉 0.013 Σmin, 已退役)
 *   - 权重: v3 码本是 E4M3(3 位尾数), bf16 有 7 位尾数、指数范围更宽 ⇒ E4M3 → bf16 **逐位精确**。
 *   - 激活: 调用方 rms_norm / swiglu 出口都舍过 bf16(cuda_vq_row.inc.cu "激活以 bf16 存"那段), 存成 bf16 无损。
 *   - bf16×bf16 的乘积在 f32 里精确, 张量核按 f32 累加; 行增益(含反修覆盖)在求和之后乘, 与融合路同一个点。
 *   ⇒ 与融合路相比只有**K 维累加顺序**不同(融合路本来就是 32 lane 分段 + shuffle 树, 与任何 GEMM 都不逐位同)。
 *   判据是 PPL/五指标与融合路持平, 不是逐字节。v2(f16 码本)转 bf16 会丢 3 位尾数 ⇒ v2 不走这里, 仍走融合路。
 *
 * 【形状(2026-09-29 换, 微基准 gguf-tools/bench/v41_vq_prefill_mma_bench.cu 定形, 真载荷 + 真路由)】
 *   一个 block = 16 warp(512 线程) = 128 行权重 × 一个工作项(同一专家的 ≤128 个 token); 每轮沿 K 走 64 列(每行 8 个码字):
 *   ① 512 个线程各解本行 2 个码字(12 位主流 + 13 位层的位平面)→ 查 shared 码本 → 8 个 bf16 写进 A 瓦片; 同时把 128 个 token × 64 列的激活
 *      (bf16)搬进 B 瓦片; 下一轮的位流/激活在 mma 期间就发出去(寄存器预取)。
 *   ② warp w: 管第 (w&7) 个 16 行片, token 分两半(w>>3), 对本工作项有 token 的 n8 片各做一次 mma。
 *   block 常驻(grid = SM 数 × 占用率 API 算出的每 SM block 数), 循环吃 (工作项, 行块) —— 码本每个 block 只搬一次。
 *   09-24 版是 8 warp × ≤32 token: 真路由(09-29 探针)一块 2048 token 只有 200~350 个专家有 token, 最热的吃 8~16%,
 *   ≤32 就要把热专家拆成十几项**各解一遍位流**(590 项/层块); ≤128 降到 340 项。微基准(ms/层块, 512/2048/4096 token):
 *   13 位层 32.6/48.2/83.6 → 31.7/40.4/54.0; 12 位层(码本进 shared 时顺手转成 bf16, 64 KB, 13 位的 128 KB 放不下)29.7/42.9/73.8 → 26.6/34.5/49.6。
 *   判负存档(同尺): 码本放 L2(查表延迟盖不住, 慢 30~90%) / BK 32 换每 SM 2 个 block(13 位层慢 30%) / 生产-消费双组 warp(mma 只剩 8 个 warp 在发)。
 * shared = 码本(12 位 bf16 64 KB / 13 位 E4M3 64 KB) + A 16 KB + B 16 KB = 96 KB, 在 GB10 每 block 99 KB 以内。
 * ★逐位同★: 乘积/k16 累加序/出口舍入点与 09-24 版一个没变(微基准逐位门), 引擎门 = 温 0 输出 cmp。
 *
 * 【出错会怎样】A/B 瓦片按 16 B 块做 XOR 交织(块号 ^ 行号低 3 位), ldmatrix 地址与写入地址必须用同一个式子;
 * 写错不报错, 只是 mma 吃到别的列 —— 表现是 PPL 爆到几万(09-15 sparse_attn_mma 实撞过同类的 warp 偏移漏写)。
 * 位流按"行起点 8 B 对齐"读 32 位字(转换器 v41_to_gguf_vq3: 载荷 8 B 对齐, 行字节 960/432 都是 8 的倍数),
 * 哪天换了行宽不是 4 的倍数, 这里读出来的是错位的字。 */

#define VQM_BM      128u   /* 一 block 的权重行: 8 个 16 行片 */
#define VQM_BN      128u   /* 一个工作项最多几个 token: 16 个 n8 片 */
#define VQM_BK      64u    /* 一轮沿 K 走几列: 每行 8 个码字 */
#define VQM_TW      2u     /* token 分几组给不同 warp ⇒ 16 warp, 512 线程 */
#define VQM_THREADS (256u * VQM_TW)
#define VQM_TILE_BYTES ((VQM_BM + VQM_BN) * VQM_BK * 2u)   /* A 16 KB + B 16 KB, 码本之后 */

/* 两个 E4M3 → 两个 bf16(打包)。先走硬件 cvt 到 f16(E4M3 ⊂ f16 正规数, 含 E4M3 的非规格化数), 再按位改指数偏置:
 * f16 正规数右移 3 位 = 指数落到 bf16 的指数位、尾数高 7 位落到 bf16 尾数(E4M3 只有高 3 位非零, 丢的全是 0),
 * 再加 (127-15)<<7 = 0x3800 改偏置。零要单独保住(否则变成 2^-15)。 */
__device__ __forceinline__ static uint32_t vqm_e4m3x2_to_bf16x2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    const uint32_t mag = h & 0x7fff7fffu, nz = __vcmpne2(mag, 0u);
    return (h & 0x80008000u) | ((((mag >> 3) & 0x0fff0fffu) + 0x38003800u) & nz);
}
__device__ __forceinline__ static void vqm_ldsm4(uint32_t *r, const uint8_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void vqm_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/* 瓦片一行 = 64 个 bf16 = 8 个 16 B 块; 块号与行号低 3 位异或, ldmatrix 一次取 8 行同一列块时落在 8 个不同 bank 组 */
__device__ __forceinline__ static uint32_t vqm_swz(uint32_t row, uint32_t chunk) { return row * 128u + ((chunk ^ (row & 7u)) << 4); }
/* 一个码字 → 8 个 bf16(16 B): CB=0 码本是 E4M3(8 B/词)现场转; CB=1 码本进 shared 时已转成 bf16(16 B/词), 直接取 */
template <int CB>
__device__ __forceinline__ static uint4 vqm_cw(const uint8_t *cbs, uint32_t v) {
    if (CB == 1) return *(const uint4 *)(cbs + (size_t)v * 16u);
    const uint2 cw = *(const uint2 *)(cbs + (size_t)v * 8u);
    uint4 o;
    o.x = vqm_e4m3x2_to_bf16x2(cw.x); o.y = vqm_e4m3x2_to_bf16x2(cw.x >> 16);
    o.z = vqm_e4m3x2_to_bf16x2(cw.y); o.w = vqm_e4m3x2_to_bf16x2(cw.y >> 16);
    return o;
}

/* 排序后的激活, bf16。值本来就在 bf16 格点上, 这里用 bf16r(舍)而不是截断: 万一上游漏舍, 也只差一次正确舍入。 */
__global__ static void vqm_gather16_kernel(uint16_t *xs, const float *x, const int32_t *perm, uint32_t K, uint32_t IN) {
    const uint32_t i = blockIdx.x, t = (uint32_t)perm[i] / K;
    const float *src = x + (uint64_t)t * IN;
    uint16_t *dst = xs + (uint64_t)i * IN;
    for (uint32_t d = threadIdx.x; d < IN; d += blockDim.x) dst[d] = (uint16_t)(__float_as_uint(v41_bf16r(src[d])) >> 16);
}

/* MODE 0 = gate: g32 = bf16(W1·x·g) | 1 = up: 读 g32 做 clamp+SwiGLU 写 h16 | 2 = down: ys = bf16(W2·h·g)。
 * 出口舍入点与融合路 vqp_fused_gu/down 逐式相同(求和 → 乘行增益 → bf16r)。cb_bytes = 码本在 shared 里的字节(CB=1 是 nc×16)。 */
template <int EXT, int MODE, int CB>
__global__ __launch_bounds__(VQM_THREADS, 1) static void vqm_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t CPR = VQM_BK / 8u, TPR = VQM_THREADS / VQM_BM, CPT = CPR / TPR;   /* 每行每轮 8 个码字, 4 个线程各 2 个 */
    constexpr uint32_t NF = VQM_BN / 8u, NFW = NF / VQM_TW, CH = (VQM_BN * CPR) / VQM_THREADS;   /* n8 片 16, 每 warp 8; B 片每线程 2 个 16 B 块 */
    extern __shared__ __align__(16) uint8_t vqmsh[];
    uint8_t *cbs = vqmsh, *As = vqmsh + cb_bytes, *Bs = As + VQM_BM * VQM_BK * 2u;
    const int which = MODE;   /* 载荷槽: 0 w1(gate) / 1 w3(up) / 2 w2(down) */
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, wr = warp & 7u, th = warp >> 3;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile, nit = K / VQM_BK;
    {   /* v3 码本一层一本(三矩阵、全部专家共用) ⇒ 取第一个工作项的就是本层的 */
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * (CB ? 16u : 8u) != cb_bytes) return;   /* 整个 block 同一判断, 不会有人卡在后面的 barrier 上 */
        if (CB == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQM_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t ar = tid / TPR, sub = tid % TPR;   /* 解码分工: 线程管 A 瓦片第 ar 行的第 sub·CPT.. 个码字 */
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQM_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {   /* 主机侧已按槽表查过, 走到这里 = 载荷坏了; down 写 0 不给 reduce 留脏值(与融合路同) */
            if (MODE == 2)
                for (uint32_t i = tid; i < VQM_BM * nt; i += VQM_THREADS)
                    if (r0 + i % VQM_BM < M) ys[(uint64_t)(base + i / VQM_BM) * M + r0 + i % VQM_BM] = 0.f;
            continue;
        }
        const uint32_t grow = r0 + ar, mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3;
        const bool rv = grow < M;
        const uint8_t *rowp = m.ix + (size_t)(rv ? grow : 0u) * mrow;
        const uint8_t *ep = EXT ? m.ex + (size_t)(rv ? grow : 0u) * erow : NULL;
        float acc[NFW][4];
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f; }
        uint32_t w0 = 0, w1 = 0, eb = 0;
        uint4 bx[CH];
        #pragma unroll
        for (uint32_t c = 0; c < CH; c++) bx[c] = make_uint4(0, 0, 0, 0);
        /* 本线程这一轮的码字在行内的位偏移 = 轮 × 96 + sub × 24: 读对齐的两个字(位偏移 mod 32 ∈ {0, 24, 16, 8}, 24 位都在 64 位窗口里)再右移 */
        auto load_round = [&](uint32_t i) {
            const uint32_t bitoff = i * 12u * CPR + sub * 12u * CPT, a = (bitoff >> 5) << 2;
            if (rv) { w0 = __ldg((const unsigned int *)(rowp + a)); w1 = __ldg((const unsigned int *)(rowp + a + 4u));
                      if (EXT) eb = __ldg(ep + i); }   /* 位平面: 一轮 8 个码字 = 1 字节 */
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) {
                const uint32_t ch = tid + c * VQM_THREADS, bt = ch / CPR, bc = ch % CPR;
                if (bt < nt) bx[c] = __ldg((const uint4 *)(act + (uint64_t)(base + bt) * K + (uint64_t)i * VQM_BK + bc * 8u));
            }
        };
        load_round(0);
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t sh = (i * 12u * CPR + sub * 12u * CPT) & 31u;
            const uint64_t u = (((uint64_t)w1 << 32) | w0) >> sh;
            #pragma unroll
            for (uint32_t q = 0; q < CPT; q++) {
                uint32_t v = (uint32_t)(u >> (12u * q)) & 0xFFFu;
                if (EXT) v |= ((eb >> (sub * CPT + q)) & 1u) << 12;   /* 第 13 位: 位平面第 i 字节的第 (j&7) 位 */
                *(uint4 *)(As + vqm_swz(ar, sub * CPT + q)) = vqm_cw<CB>(cbs, v);
            }
            #pragma unroll
            for (uint32_t c = 0; c < CH; c++) { const uint32_t ch = tid + c * VQM_THREADS; *(uint4 *)(Bs + vqm_swz(ch / CPR, ch % CPR)) = bx[c]; }
            __syncthreads();
            if (i + 1u < nit) load_round(i + 1u);   /* 下一轮的位流/激活现在发出去, mma 期间在路上 */
            /* ② 本 warp 的 16 行 × 本半的 n8 片 */
            #pragma unroll
            for (uint32_t kk = 0; kk < VQM_BK / 16u; kk++) {
                uint32_t a[4];
                {   const uint32_t mt = lane >> 3, row = wr * 16u + (lane & 7u) + (mt & 1u) * 8u;
                    vqm_ldsm4(a, As + vqm_swz(row, kk * 2u + (mt >> 1))); }
                #pragma unroll
                for (uint32_t np = 0; np < NFW / 2u; np++) {
                    const uint32_t f0 = th * NFW + 2u * np;   /* 本 warp 的第 2np 片在全部片里的号 */
                    if (f0 * 8u >= nt) break;
                    uint32_t b[4];
                    const uint32_t mt = lane >> 3, tok = (f0 + (mt >> 1)) * 8u + (lane & 7u);
                    vqm_ldsm4(b, Bs + vqm_swz(tok, kk * 2u + (mt & 1u)));
                    /* ★每个 k16 从零起算, 结果用普通 FADD 加回累加器★(2026-09-24 对拍实撞): 直接在 acc 上连乘 K 维,
                     * 张量核内部的 f32 累加会丢低位(对齐时截断, 不是就近舍) —— 与融合路逐元素比, 3.5% 的输出差 1 个
                     * bf16 ulp、相对 L2 5e-4, 2048 尺 PPL 14.096 → 14.192(+0.68%, 单向偏)。提回后降到 0.5% / 1.7e-4。
                     * 这与 DeepSeek 官方 FP8 GEMM"每 128 元素提回 CUDA 核累加"是同一件事, 只是这里提得更勤。 */
                    float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
                    vqm_mma(t0, a, b[0], b[1]);
                    #pragma unroll
                    for (int c = 0; c < 4; c++) acc[2 * np][c] += t0[c];
                    if ((f0 + 1u) * 8u < nt) { vqm_mma(t1, a, b[2], b[3]);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) acc[2 * np + 1][c] += t1[c]; }
                }
            }
            __syncthreads();
        }
        /* ③ 出口: c0,c1 = 行 lane/4、token (lane&3)*2+{0,1}; c2,c3 = 行 +8 */
        const uint32_t rr[2] = { r0 + wr * 16u + (lane >> 2), r0 + wr * 16u + (lane >> 2) + 8u };
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t r = rr[h];
            if (r >= M) continue;
            __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
            const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
            #pragma unroll
            for (uint32_t f = 0; f < NFW; f++) {
                const uint32_t gf = th * NFW + f;
                if (gf * 8u >= nt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                    if (t >= nt) continue;
                    const uint64_t o = (uint64_t)(base + t) * M + r;
                    const float val = v41_bf16r(acc[f][h * 2u + e] * g);
                    if (MODE == 0) g32[o] = val;
                    else if (MODE == 1) h16[o] = v41_vq_swiglu(g32[o], val, clamp);
                    else ys[o] = val;
                }
            }
        }
    }
}

static struct { vqp_item *d; uint64_t cap; uint16_t *xs16, *h16; float *g32; uint32_t *doff;
                uint64_t xs16_cap, h16_cap, g32_cap, doff_cap; int ready, nsm; int occ[2][3]; } g_vqm;

/* 每个实例的 grid: SM 数 × 占用率 API 算出的每 SM block 数(不写死档位; 96 KB shared 下现在是 1) */
template <int EXT, int MODE, int CB>
static int vqm_grid_of(size_t shb, uint32_t nwork) {
    int *occ = &g_vqm.occ[EXT][MODE];
    if (*occ == 0) {
        if (cudaFuncSetAttribute(vqm_kernel<EXT, MODE, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, vqm_kernel<EXT, MODE, CB>, (int)VQM_THREADS, shb) != cudaSuccess || *occ <= 0) {
            (void)cudaGetLastError(); *occ = -1;
        }
    }
    if (*occ < 0) return 0;
    const uint32_t g = (uint32_t)g_vqm.nsm * (uint32_t)*occ;
    return (int)(g < nwork ? g : nwork);
}

/* 返回 0 = 失败(调用方硬失败)。ys = 排序后的 down 输出 [nvalid][OUT], 与融合路写的是同一个缓冲, reduce 一字不改。 */
static int vqm_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert, uint32_t nvalid,
                   uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp, const float *x, const int32_t *perm,
                   uint32_t n_expert, uint32_t layer_index, float *ys, const float *gr) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    if ((nbit != 12u && nbit != 13u) || IN % VQM_BK || MID % VQM_BK) {
        fprintf(stderr, "ds4: [vq-prefill] L%u 张量核路不认这个形状(码本 %u 词, IN %u MID %u)\n", layer_index, nc, IN, MID);
        return 0;
    }
    /* 12 位层码本进 shared 时转成 bf16(nc×16 B = 64 KB); 13 位层 8192 词转 bf16 要 128 KB 放不下, 留 E4M3(64 KB)现场转 */
    const int cb = nbit == 12u ? 1 : 0;
    const uint32_t cbb = nc * (cb ? 16u : 8u), shb = cbb + VQM_TILE_BYTES;
    if (!g_vqm.ready) {
        int nsm = 0;
        const bool ok = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess && nsm > 0;
        (void)cudaGetLastError();
        g_vqm.ready = ok ? 1 : -1; g_vqm.nsm = nsm;
        fprintf(stderr, "ds4: [vq-prefill] 张量核路(bf16 mma, 一工作项 ≤%u token, %u 线程) %s, %d 个 SM\n",
                VQM_BN, VQM_THREADS, ok ? "就绪" : "★不可用★", nsm);
    }
    if (g_vqm.ready != 1) return 0;
    uint32_t nit = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) nit += (cnt[e] + VQM_BN - 1u) / VQM_BN;
    if (!nit) return 0;
    vqp_item *ih = (vqp_item *)malloc((size_t)nit * sizeof(vqp_item));
    if (!ih) return 0;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_total_expert; e++)
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += VQM_BN) {
            ih[k].e = (int32_t)e; ih[k].t0 = (int32_t)t0; ih[k].nt = (int32_t)(cnt[e] - t0 < VQM_BN ? cnt[e] - t0 : VQM_BN); k++;
        }
    int ok = vqp_grow((void **)&g_vqm.d, &g_vqm.cap, nit, sizeof(vqp_item), "mma items") &&
             vqp_grow((void **)&g_vqm.doff, &g_vqm.doff_cap, n_total_expert + 1u, sizeof(uint32_t), "mma off") &&
             vqp_grow((void **)&g_vqm.xs16, &g_vqm.xs16_cap, (uint64_t)nvalid * IN, sizeof(uint16_t), "mma xs16") &&
             vqp_grow((void **)&g_vqm.h16, &g_vqm.h16_cap, (uint64_t)nvalid * MID, sizeof(uint16_t), "mma h16") &&
             vqp_grow((void **)&g_vqm.g32, &g_vqm.g32_cap, (uint64_t)nvalid * MID, sizeof(float), "mma g32");
    if (ok) ok = cudaMemcpyAsync(g_vqm.d, ih, (size_t)nit * sizeof(vqp_item), cudaMemcpyHostToDevice, g_cur_stream) == cudaSuccess &&
                 cudaMemcpyAsync(g_vqm.doff, off_h, (size_t)(n_total_expert + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice, g_cur_stream) == cudaSuccess;
    /* ih 是异步 H2D 的源, 主机内存可分页 ⇒ cudaMemcpyAsync 在返回前已拷进暂存, 这里释放是安全的(融合路同款) */
    free(ih);
    if (!ok) { (void)cudaGetLastError(); fprintf(stderr, "ds4: [vq-prefill] L%u 张量核路暂存/拷贝失败\n", layer_index); return 0; }
    vqm_gather16_kernel<<<nvalid, 256, 0, g_cur_stream>>>(g_vqm.xs16, x, perm, n_expert, IN);
    if (!cuda_ok(cudaGetLastError(), "vq prefill mma gather16")) return 0;
    const uint32_t tg = (MID + VQM_BM - 1u) / VQM_BM, td = (OUT + VQM_BM - 1u) / VQM_BM;
#define VQM_LAUNCH(E, CB) do { \
        const int bg = vqm_grid_of<E, 0, CB>(shb, nit * tg), bu = vqm_grid_of<E, 1, CB>(shb, nit * tg), bd = vqm_grid_of<E, 2, CB>(shb, nit * td); \
        if (bg <= 0 || bu <= 0 || bd <= 0) { fprintf(stderr, "ds4: [vq-prefill] L%u 张量核实例开不出 %u KB shared\n", layer_index, shb >> 10); return 0; } \
        vqm_kernel<E, 0, CB><<<bg, VQM_THREADS, shb, g_cur_stream>>>(g_vqm.g32, NULL, NULL, blob, g_vqm.d, nit, g_vqm.xs16, g_vqm.doff, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill mma gate")) return 0; \
        vqm_kernel<E, 1, CB><<<bu, VQM_THREADS, shb, g_cur_stream>>>(g_vqm.g32, g_vqm.h16, NULL, blob, g_vqm.d, nit, g_vqm.xs16, g_vqm.doff, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill mma up")) return 0; \
        vqm_kernel<E, 2, CB><<<bd, VQM_THREADS, shb, g_cur_stream>>>(NULL, NULL, ys, blob, g_vqm.d, nit, g_vqm.h16, g_vqm.doff, OUT, MID, clamp, cbb, gr); \
        return cuda_ok(cudaGetLastError(), "vq prefill mma down"); \
    } while (0)
    if (nbit == 13u) VQM_LAUNCH(1, 0);
    VQM_LAUNCH(0, 1);
#undef VQM_LAUNCH
}
