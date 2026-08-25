/* cuda_vq_fused2_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * fused2 特化高速 VQ 解码路。
 */
/* ===== fused2: 特化高速 VQ 解码路 (2026-08-18, 冲 30 t/s) =====
 * 依据(vqcyc/EXP 消融): 旧 fused 每 warp 每行重复 vtab+payload 头解析(串行 global
 * 依赖链)+位流 shared 搬运 2×syncwarp+码本 global gather ⇒ 8.2cy/idx。
 * 特化条件(host vq2_probe_layer 校验后启用): d16=4, w1/w3 nc=512(9bit),
 * w2 nc=256(8bit), payload 4B 对齐, 全 256 专家槽在位。
 * 结构: 一 block 一 (token,expert), 4 warps×R 行/warp; x/h 与码本进 shared;
 * 位流按 lane 36B(9bit)/32B(8bit) 对齐段合并读进寄存器, funnelshift 纯 ALU 抽位。
 * 乘加顺序与旧 kernel 不同(重排容差), 对拍口径=vq diag cos/score-ids。 */
#define DS4_VQ2_ROWS_PER_WARP 8u

typedef struct { uint64_t gr_off1, ix_off1, gr_off3, ix_off3; } ds4_vq2_hdr;

/* thread0 per block: vtab→payload→gr/ix 偏移(全 block 共用, 一次) */
__device__ __forceinline__ static void vq2_resolve(
        const uint8_t *blob, int e, int which, uint32_t nc,
        uint32_t rows, uint64_t *gr_off, uint64_t *ix_off) {
    const uint64_t *vtab = (const uint64_t *)(blob + 16);
    const uint64_t off = vtab[(size_t)e * 3 + which];
    *gr_off = off + 16 + (uint64_t)nc * 4 * 2;
    *ix_off = *gr_off + (uint64_t)rows * 2;
}

/* 一行 9bit 点积: lane 段=36B(9 u32), 32 idx funnelshift 抽位, 码本 shared gather */
__device__ __forceinline__ static float vq2_row_dot9(
        const uint8_t *ix, uint32_t row, const float *xs, const __half *cb,
        uint32_t lane, uint32_t cols) {
    const uint32_t nidx_row = cols >> 2;                   /* d16=4 */
    /* lane 段覆盖 32 idx×4 元素=128 列; cols<4096(如 down 的 MID=2048)时高 lane
     * 在行内无段 —— 越 shared/位流界, 必须先退出(warp_sum 汇 0)。 */
    if ((lane << 7u) >= cols) return 0.0f;
    const uint8_t *seg = ix + (size_t)row * ((nidx_row * 9u) >> 3) + (size_t)lane * 36u;
    /* payload 无对齐保证(vq_pack 紧凑): 对齐基址读 40B 窗口, 错位字节并入位偏移 */
    const uint32_t misal = (uint32_t)((uintptr_t)seg & 3u);
    const uint32_t *wp = (const uint32_t *)(seg - misal);
    uint32_t bw[10];
    #pragma unroll
    for (uint32_t k = 0; k < 10u; k++) bw[k] = wp[k];
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < 32u; k++) {
        const uint32_t bit = k * 9u + misal * 8u, wi = bit >> 5u, sh = bit & 31u;
        const uint32_t v = __funnelshift_r(bw[wi], bw[wi + 1u < 10u ? wi + 1u : 9u], sh) & 511u;
        const uint32_t col4 = (lane * 32u + k) << 2u;
        const float4 xv = *(const float4 *)&xs[col4];
        const uint32_t cq0 = ((const uint32_t *)cb)[v * 2u];
        const uint32_t cq1 = ((const uint32_t *)cb)[v * 2u + 1u];
        const float2 f01 = __half22float2(*(const half2 *)&cq0);
        const float2 f23 = __half22float2(*(const half2 *)&cq1);
        acc += f01.x * xv.x + f01.y * xv.y + f23.x * xv.z + f23.y * xv.w;
    }
    return acc;
}

/* 一行 8bit 点积(w2): lane 段=32B, idx 直取字节 */
__device__ __forceinline__ static float vq2_row_dot8(
        const uint8_t *ix, uint32_t row, const float *xs, const __half *cb,
        uint32_t lane, uint32_t cols) {
    const uint32_t nidx_row = cols >> 2u;
    const uint8_t *seg = ix + (size_t)row * nidx_row + (size_t)lane * (nidx_row >> 5u);
    const uint32_t per_lane = nidx_row >> 5u;              /* 2048/4/32 = 16 idx */
    const uint32_t misal = (uint32_t)((uintptr_t)seg & 3u);
    const uint32_t *wp = (const uint32_t *)(seg - misal);
    uint32_t bw[5];
    #pragma unroll
    for (uint32_t k = 0; k < 5u; k++) bw[k] = wp[k];
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        const uint32_t bj = misal + k;
        const uint32_t v = (bw[bj >> 2u] >> ((bj & 3u) * 8u)) & 255u;
        const uint32_t col4 = (lane * per_lane + k) << 2u;
        const float4 xv = *(const float4 *)&xs[col4];
        const uint32_t cq0 = ((const uint32_t *)cb)[v * 2u];
        const uint32_t cq1 = ((const uint32_t *)cb)[v * 2u + 1u];
        const float2 f01 = __half22float2(*(const half2 *)&cq0);
        const float2 f23 = __half22float2(*(const half2 *)&cq1);
        acc += f01.x * xv.x + f01.y * xv.y + f23.x * xv.z + f23.y * xv.w;
    }
    return acc;
}

__device__ __forceinline__ static float vq2_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* gateup: grid=(ntok, MID/(4*R), nexp), block=128。shared: x[IN] f32 + cb1/cb3 各 4KB */
__global__ static void vq_moe_gateup_fused2_kernel(
        float *mid_out, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *x, uint32_t n_expert, uint32_t IN, uint32_t MID, float clamp) {
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0 || rw[pair] == 0.0f) return;
    extern __shared__ float sh2[];
    float *xs = sh2;                                        /* IN floats */
    __half *cb1 = (__half *)(xs + IN);                      /* 512*4 halfs */
    __half *cb3 = cb1 + 2048;
    __shared__ ds4_vq2_hdr hdr;
    if (threadIdx.x == 0) {
        vq2_resolve(blob, e, 0, 512u, MID, &hdr.gr_off1, &hdr.ix_off1);
        vq2_resolve(blob, e, 1, 512u, MID, &hdr.gr_off3, &hdr.ix_off3);
    }
    const float *xt = x + (uint64_t)t * IN;
    for (uint32_t i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = xt[i];
    {   /* 码本: payload+16 起 512*4 halfs (payload 无对齐保证, u8 组装) */
        const uint64_t *vtab = (const uint64_t *)(blob + 16);
        const uint8_t *c1 = blob + vtab[(size_t)e * 3] + 16;
        const uint8_t *c3 = blob + vtab[(size_t)e * 3 + 1] + 16;
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint16_t h1 = (uint16_t)c1[i * 2u] | ((uint16_t)c1[i * 2u + 1u] << 8);
            uint16_t h3 = (uint16_t)c3[i * 2u] | ((uint16_t)c3[i * 2u + 1u] << 8);
            cb1[i] = *(const __half *)&h1; cb3[i] = *(const __half *)&h3;
        }
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint8_t *ix1 = blob + hdr.ix_off1, *ix3 = blob + hdr.ix_off3;
    const __half *gr1 = (const __half *)(blob + hdr.gr_off1);
    const __half *gr3 = (const __half *)(blob + hdr.gr_off3);
    const float w = rw[pair];
    #pragma unroll
    for (uint32_t r = 0; r < DS4_VQ2_ROWS_PER_WARP; r++) {
        const uint32_t m = (blockIdx.y * 4u + warp) * DS4_VQ2_ROWS_PER_WARP + r;
        if (m >= MID) return;
        float g = vq2_row_dot9(ix1, m, xs, cb1, lane, IN);
        float u = vq2_row_dot9(ix3, m, xs, cb3, lane, IN);
        g = vq2_warp_sum(g);
        u = vq2_warp_sum(u);
        if (lane == 0) {
            uint16_t g1h = (uint16_t)((const uint8_t *)gr1)[m * 2u] | ((uint16_t)((const uint8_t *)gr1)[m * 2u + 1u] << 8);
            uint16_t g3h = (uint16_t)((const uint8_t *)gr3)[m * 2u] | ((uint16_t)((const uint8_t *)gr3)[m * 2u + 1u] << 8);
            g *= __half2float(*(const __half *)&g1h);
            u *= __half2float(*(const __half *)&g3h);
            if (clamp > 0.0f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            mid_out[pair * MID + m] = (g / (1.0f + __expf(-g))) * u * w;
        }
    }
}

/* down: grid=(ntok, OUT/(4*R), nexp)。shared: h[MID] f32 + cb2 2KB。
 * 旧 fused 的 rw 乘在 down 段; fused2 把 rw 折进 gateup 的 mid(silu*u*w), down 直接累加。 */
template <uint32_t NCB>
__global__ static void vq_moe_down_fused2_kernel(
        float *partial, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *h, uint32_t n_expert, uint32_t MID, uint32_t OUT) {
    /* per-pick 并行(原设计), 但各 pick 直写独立 partial 平面代替 atomicAdd —— 汇总由
     * vq2_down_reduce_kernel 以固定 pk 序完成 ⇒ 温 0 逐 bit 可复现。
     * NCB = w2 码本词数(256=8bit 索引, 512=9bit)。 */
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0 || rw[pair] == 0.0f) return;   /* 未写平面由 reduce 按 sel/rw 跳过 */
    extern __shared__ float sh2[];
    float *hs = sh2;                                        /* MID floats */
    __half *cb2 = (__half *)(hs + MID);                     /* NCB*4 halfs */
    __shared__ uint64_t ix_off2, gr_off2;
    if (threadIdx.x == 0) {
        uint64_t g2, i2;
        vq2_resolve(blob, e, 2, NCB, OUT, &g2, &i2);
        gr_off2 = g2; ix_off2 = i2;
    }
    const float *ht = h + pair * MID;
    for (uint32_t i = threadIdx.x; i < MID; i += blockDim.x) hs[i] = ht[i];
    {
        const uint64_t *vtab = (const uint64_t *)(blob + 16);
        const uint8_t *c2 = blob + vtab[(size_t)e * 3 + 2] + 16;
        for (uint32_t i = threadIdx.x; i < NCB * 4u; i += blockDim.x) {
            uint16_t h2 = (uint16_t)c2[i * 2u] | ((uint16_t)c2[i * 2u + 1u] << 8);
            cb2[i] = *(const __half *)&h2;
        }
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint8_t *ix2 = blob + ix_off2;
    const __half *gr2 = (const __half *)(blob + gr_off2);
    #pragma unroll
    for (uint32_t r = 0; r < DS4_VQ2_ROWS_PER_WARP; r++) {
        const uint32_t o = (blockIdx.y * 4u + warp) * DS4_VQ2_ROWS_PER_WARP + r;
        if (o >= OUT) return;
        float acc = (NCB == 512u) ? vq2_row_dot9(ix2, o, hs, cb2, lane, MID)
                                  : vq2_row_dot8(ix2, o, hs, cb2, lane, MID);
        acc = vq2_warp_sum(acc);
        if (lane == 0) {
            uint16_t g2h = (uint16_t)((const uint8_t *)gr2)[o * 2u] | ((uint16_t)((const uint8_t *)gr2)[o * 2u + 1u] << 8);
            /* w 已由 gateup_fused2 乘进 h, 此处不乘 */
            partial[pair * OUT + o] = acc * __half2float(*(const __half *)&g2h);
        }
    }
}

/* 固定 pk 序汇总 partial 平面 → out (确定性加序) */
__global__ static void vq2_down_reduce_kernel(
        float *out, const float *partial, const int32_t *sel, const float *rw,
        uint32_t n_expert, uint32_t OUT) {
    const uint32_t t = blockIdx.y;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float s = 0.0f;
    for (uint32_t pk = 0; pk < n_expert; pk++) {
        const uint64_t pair = (uint64_t)t * n_expert + pk;
        if (sel[pair] >= 0 && rw[pair] != 0.0f) s += partial[pair * OUT + o];
    }
    out[(uint64_t)t * OUT + o] = s;
}

/* host 侧特化闸: 全 256 专家×3 槽 在位+4B 对齐+维度/码本匹配 → fused2 可用 */
/* device 版 probe: 对 arena 副本判定, 避免 host 去读已 madvise(DONTNEED) 的 mmap
 * 原件 —— 那会触发 SSD refault + readahead 把模型页重新灌进 page cache, 与 81GiB
 * arena 叠加造成内存压力, decode 稳态实测掉 ~25%。单线程, μs 级。 */
__global__ static void vq2_probe_kernel(const uint8_t *blob, uint32_t n_total_expert,
                                        uint32_t IN, uint32_t MID, uint32_t OUT,
                                        int32_t *out_w2n) {
    const uint64_t *vtab = (const uint64_t *)(blob + 16);
    uint32_t w2n = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (int w = 0; w < 3; w++) {
            const uint64_t off = vtab[(size_t)e * 3 + w];
            if (!off) { out_w2n[0] = 0; return; }
            const uint8_t *pay = blob + off;
            uint16_t d16, n16; uint32_t rows, cols;
            memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
            memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
            const uint32_t want_r = (w == 2) ? OUT : MID;
            const uint32_t want_c = (w == 2) ? MID : IN;
            if (d16 != 4u || rows != want_r || cols != want_c) { out_w2n[0] = 0; return; }
            if (w < 2) { if (n16 != 512u) { out_w2n[0] = 0; return; } }
            else {
                if (n16 != 256u && n16 != 512u) { out_w2n[0] = 0; return; }
                if (w2n == 0) w2n = n16; else if (w2n != n16) { out_w2n[0] = 0; return; }
            }
        }
    }
    out_w2n[0] = (int32_t)w2n;
}

/* 返回 w2 码本词数(256/512), 0 = fused2 不可用。w1/w3 固定 512。 */
static uint32_t vq2_probe_layer(const uint8_t *blob, uint32_t n_total_expert,
                                uint32_t IN, uint32_t MID, uint32_t OUT) {
    const uint64_t *vtab = (const uint64_t *)(blob + 16);
    uint32_t w2n = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (int w = 0; w < 3; w++) {
            const uint64_t off = vtab[(size_t)e * 3 + w];
            if (!off) return 0;   /* 对齐不要求: kernel 侧对齐窗口读+misal 位偏移 */
            const uint8_t *pay = blob + off;
            uint16_t d16, n16; uint32_t rows, cols;
            memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
            memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
            const uint32_t want_r = (w == 2) ? OUT : MID;
            const uint32_t want_c = (w == 2) ? MID : IN;
            if (d16 != 4u || rows != want_r || cols != want_c) return 0;
            if (w < 2) { if (n16 != 512u) return 0; }
            else {
                if (n16 != 256u && n16 != 512u) return 0;
                if (w2n == 0) w2n = n16; else if (w2n != n16) return 0;
            }
        }
    }
    return w2n;
}


/* bs: 本 warp 专属的位流暂存(至少 DS4_VQ_BITWORDS 个 uint32); NULL=直接读全局 */
__device__ __forceinline__ static float vq_row_dot_dev(
        const uint8_t *blob, int e, int which, uint32_t row,
        const float *vec, uint32_t lane,
        const uint8_t *down_base, uint64_t down_ebytes, uint32_t mid_for_cold,
        const __half *cb_sh, uint32_t *bs) {
    const uint64_t off = vq_slot_dev(blob, e, which);
    if (off == 0u) {
        /* 冷 w2: base go1b 34B/256el sign 字节 → ±d, 边解边点乘 */
        if (which != 2 || down_base == NULL || down_ebytes == 0u) return 0.0f;
        const uint32_t nblk_row = mid_for_cold / 256u;
        const uint8_t *rb = down_base + (uint64_t)e * down_ebytes + (size_t)row * nblk_row * 34u;
        float acc = 0.0f;
        for (uint32_t b = lane; b < nblk_row; b += 32u) {
            uint16_t dsc; memcpy(&dsc, rb + (size_t)b * 34u, 2);
            __half dh; memcpy(&dh, &dsc, 2);
            const float dv = __half2float(dh);
            const uint8_t *sg = rb + (size_t)b * 34u + 2;
            const float *vs = vec + (size_t)b * 256u;
            for (int k = 0; k < 256; k++)
                acc += (((sg[k >> 3] >> (k & 7)) & 1) ? dv : -dv) * vs[k];
        }
        return acc;
    }
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return 0.0f;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)n16 * d16 * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32u) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / d16;
    __half gh; memcpy(&gh, gr + (size_t)row * 2, 2);
    const float g = __half2float(gh);
    const size_t i0 = (size_t)row * nidx_row;
    /* ★warp 协作合并读位流★
     * 9-bit 索引让相邻 lane 的读地址只差 1.125 字节, 各自发 4 字节非对齐读时硬件
     * 无法合并 —— 一个 warp 为了 36 字节的数据发 32 个独立请求。实测位流单独就占
     * 了 70% 的 kernel 时间, 折合仅 14.5GB/s(带宽的 5.7%)。
     * 改成: 全 warp 先按 4 字节对齐把整行位流合并搬进 shared, 之后各 lane 从片上
     * 取位段。取值与直接读全局逐位相同。 */
    size_t bs_base = 0;
    const long long cyc_a = g_vq_cyc_on ? clock64() : 0;
    if (bs) {
        const size_t byte0 = (i0 * nbit) >> 3;
        bs_base = byte0 & ~(size_t)3;
        const size_t byte_end = (((i0 + nidx_row) * nbit + 7u) >> 3) + 4u;   /* +4: 尾部窗口 */
        uint32_t nw = (uint32_t)((byte_end - bs_base + 3u) >> 2);
        if (nw > DS4_VQ_BITWORDS) nw = DS4_VQ_BITWORDS;
        for (uint32_t w = lane; w < nw; w += 32u)
            memcpy(&bs[w], ix + bs_base + (size_t)w * 4u, 4);
        __syncwarp();
    }
    const long long cyc_b = g_vq_cyc_on ? clock64() : 0;
    /* 4 路独立累加链。单链版本每次迭代是"读位流→移位→查码本(L1 随机, ~30cy)→FMA"
     * 一条串行依赖, warp 全程在等访存返回, 实测只跑出 51GB/s(带宽的 20%)。四条链
     * 交错发射即可把彼此的延迟填掉; 求和顺序变化只影响浮点结合律层面的末位。 */
    /* ★两阶段解码, 打断依赖链★
     * 原写法每次迭代是 "读位流 → 解出 v → 用 v 算码本地址 → 读码本 → 乘加",
     * 后一次访存的地址依赖前一次的结果, 硬件无从预取 —— 两段 ~30cy 的 L1 延迟首尾
     * 相接, 多路 ILP 也只是并行几条同样长的链。ncu: 74% 时间卡在 L1TEX scoreboard,
     * 而 L1 命中率 99.5%(数据本就在片上), 说明纯粹是延迟没被掩盖。
     * 改成: 先一次解出 BATCH 个索引(彼此无依赖, 访存可完全流水), 再拿这批已知地址
     * 批量查码本+乘加(地址提前就绪, 同样可流水)。 */
    #define DS4_VQ_BATCH 8u
    float a0 = 0.0f;
    uint32_t j = lane;
    for (; j + (DS4_VQ_BATCH - 1u) * 32u < nidx_row; j += DS4_VQ_BATCH * 32u) {
        uint32_t vv[DS4_VQ_BATCH];
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {       /* 阶段一: 只解索引 */
            const uint32_t jj = j + k * 32u;
            if (nbit == 8u) {
                vv[k] = ix[i0 + jj];
            } else {
                const size_t bit = (i0 + jj) * nbit;
                uint32_t w;
                if (bs) memcpy(&w, (const uint8_t *)bs + ((bit >> 3) - bs_base), 4);
                else    memcpy(&w, ix + (bit >> 3), 4);
                vv[k] = (w >> (bit & 7)) & imsk;
            }
        }
        if (g_vq_exp_mode == 1) {
            #pragma unroll
            for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) a0 += (float)vv[k];
            continue;
        }
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {       /* 阶段二: 地址已就绪 */
            const float *vs = vec + (size_t)(j + k * 32u) * d16;
            const uint8_t *c = (cb_sh ? (const uint8_t *)cb_sh : cb) + (size_t)vv[k] * d16 * 2;
            if (d16 == 4u) {
                uint64_t q; memcpy(&q, c, 8);               /* 码本项正好 8 字节 */
                #pragma unroll
                for (uint32_t d = 0; d < 4u; d++) {
                    __half hh; const uint16_t u = (uint16_t)(q >> (d * 16));
                    memcpy(&hh, &u, 2);
                    a0 += __half2float(hh) * vs[d];
                }
            } else {
                for (uint32_t d = 0; d < d16; d++) {
                    __half hh; memcpy(&hh, c + (size_t)d * 2, 2);
                    a0 += __half2float(hh) * vs[d];
                }
            }
        }
    }
    for (; j < nidx_row; j += 32u) {            /* 尾巴 */
        uint32_t v;
        if (nbit == 8u) {
            v = ix[i0 + j];
        } else {
            const size_t bit = (i0 + j) * nbit;
            uint32_t w;
            if (bs) memcpy(&w, (const uint8_t *)bs + ((bit >> 3) - bs_base), 4);
            else    memcpy(&w, ix + (bit >> 3), 4);
            v = (w >> (bit & 7)) & imsk;
        }
        const float *vs = vec + (size_t)j * d16;
        if (cb_sh) {
            const __half *pc = cb_sh + (size_t)v * d16;
            for (uint32_t d = 0; d < d16; d++) a0 += __half2float(pc[d]) * vs[d];
        } else {
            const uint8_t *c = cb + (size_t)v * d16 * 2;
            for (uint32_t d = 0; d < d16; d++) {
                __half ch; memcpy(&ch, c + (size_t)d * 2, 2);
                a0 += __half2float(ch) * vs[d];
            }
        }
    }
    if (g_vq_cyc_on && lane == 0) {
        const long long cyc_c = clock64();
        atomicAdd(&g_vq_cyc[0], (unsigned long long)(cyc_b - cyc_a));   /* 位流预取 */
        atomicAdd(&g_vq_cyc[1], (unsigned long long)(cyc_c - cyc_b));   /* 解码+乘加 */
        atomicAdd(&g_vq_cyc[2], 1ull);
    }
    return a0 * g;   /* g 是整行共用的标量, 提到最后统一乘 */
}

