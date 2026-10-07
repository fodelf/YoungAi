/* cuda_bwd_vq.inc.cu — ds4_cuda.cu 分片: routed 专家反向的 VQ 核(2026-10-01)。
 * 10-02 起 v3 载荷不再走下面两种直读核: 重算 H_g/H_u/O 改用预填张量核(vqm_run_parts), 转置累加改用本文件后半的 vqst_kernel(张量核, 10-03 起分段预取 + 寄存器直解);
 * 直读两核只留给 v2(f16 码本)载荷。
 *
 * 为什么: 第一版(已删)把每个有 token 的专家的三块矩阵解成 bf16 稠密阵再交给 cuBLAS ——
 * 一层约 365 个专家 × 3 × 2304×5120 = 26 GB 写 + 26 GB 读, 按 240 GB/s 是 0.2 s/层, 19 层 4 s, 正好是一题反传的大头。
 * 这里两种核都直接读位流, 不落稠密阵, 一层只读两遍位流(约 5 GB):
 *   行点积  out[p][r] = g_r·Σ_c W[r][c]·x[p][c]      (前向方向: 重算 H_g/H_u/O)
 *   转置累加 out[p][c] = Σ_r g[p][r]·g_r·W[r][c]    (反向方向: G_A = G_O·W2, G_X = G_Hg·W1 + G_Hu·W3)
 * 一发覆盖本层全部有 token 的专家(grid.y = 专家), 每个专家的配对行在排序表里连续(off/cnt)。
 * 解码与前向同一份: v41_vq_open(载荷解析) + 12 位主流 LSB 先读 + 第 13 位平面 + v41_vq_cw(E4M3 码字), 增益 = 行增益 × 反修覆盖(down 才有)。 */

/* 一趟带几个 token: 专家的 token 多于它就分趟, 每趟把整块位流重读重解一遍 —— 一层每遍 ~2.6 GB(365 个专家 × 3 块), 是反传的大头
 * (10-01 全层 prof: routed 专家反向 3.2 s/题, 占反传七成; 一题 ~600 token × top-6 摊到 ~365 个专家, 平均每个 ~10 个 token, 8 一趟要两遍)。
 * 行点积核每 token 只占 lane 上 1 个累加器 ⇒ 给 32; 转置核每 token 占 8 个(一个码字 8 列)⇒ 给 16(128 个累加寄存器)。
 * 每个 token 自己的累加顺序与趟宽无关(行点积按码字、转置按行), 改趟宽结果逐位不变。 */
#define VQB_MT_ROW 32u
#define VQB_MT 16u
#define VQB_RT 64u    /* 转置核每次搬进 shared 的行数 */

template <int V3>
__device__ __forceinline__ static uint32_t vqb_code(const v41_vq_mat &m, uint32_t r, uint32_t j) {
    const uint32_t mb = V3 ? 12u : m.nbit;
    const uint8_t *rowp = m.ix + (((uint64_t)r * m.nidx_row * mb) >> 3);
    const uint32_t bit = j * mb, by = bit >> 3, sh = bit & 7u;
    const uint32_t w = (uint32_t)rowp[by] | ((uint32_t)rowp[by + 1u] << 8) | ((uint32_t)rowp[by + 2u] << 16);
    uint32_t v = (w >> sh) & ((1u << mb) - 1u);
    if (V3 && m.ex) v |= ((uint32_t)(m.ex[(size_t)r * ((m.nidx_row + 7u) >> 3) + (j >> 3)] >> (j & 7u)) & 1u) << 12;
    return v;
}
__device__ __forceinline__ static float vqb_gain(const v41_vq_mat &m, uint32_t r) {
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    return __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
}

/* 行点积: 一 warp 一行, lane 按码字列跨; x 是排序后的 bf16 激活 [nv][C] */
template <int V3>
__global__ static void vqb_rowdot_kernel(float *out, const __nv_bfloat16 *x, const uint8_t *blob, uint32_t which, uint32_t R, uint32_t C,
                                         const uint32_t *act, const uint32_t *off, const uint32_t *cnt, const float *gr_all, uint32_t OUTd, int *bad) {
    const uint32_t k = blockIdx.y, e = act[k], p0 = off[k], m = cnt[k];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= R) return;
    const v41_vq_mat mm = v41_vq_open<V3>(blob, (int)e, (int)which, R, C, (which == 2u && gr_all) ? gr_all + (size_t)e * OUTd : NULL);
    if (!mm.ok) { if (lane == 0) atomicExch(bad, 1); return; }
    const float gain = vqb_gain(mm, r);
    for (uint32_t t0 = 0; t0 < m; t0 += VQB_MT_ROW) {
        const uint32_t mt = m - t0 < VQB_MT_ROW ? m - t0 : VQB_MT_ROW;
        float acc[VQB_MT_ROW];
        #pragma unroll
        for (uint32_t t = 0; t < VQB_MT_ROW; t++) acc[t] = 0.f;
        for (uint32_t j = lane; j < mm.nidx_row; j += 32u) {
            float c[8];
            v41_vq_cw<V3>(vqb_code<V3>(mm, r, j), mm.cb, 0, c);
            #pragma unroll
            for (uint32_t t = 0; t < VQB_MT_ROW; t++)
                if (t < mt) acc[t] += v41_vq_dot8_cw(c, *(const uint4 *)(x + (uint64_t)(p0 + t0 + t) * C + (uint64_t)j * 8u));
        }
        #pragma unroll
        for (uint32_t t = 0; t < VQB_MT_ROW; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && t < mt) out[(uint64_t)(p0 + t0 + t) * R + r] = v * gain;
        }
    }
}

/* 转置累加: 一线程一码字列(8 个输出列), 沿行走完整块矩阵; g 的行块与行增益按 VQB_RT 行搬进 shared */
template <int V3>
__global__ static void vqb_tdot_kernel(float *out, const float *g, const uint8_t *blob, uint32_t which, uint32_t R, uint32_t C,
                                       const uint32_t *act, const uint32_t *off, const uint32_t *cnt, const float *gr_all, uint32_t OUTd,
                                       int accumulate, int *bad) {
    __shared__ float gs[VQB_MT][VQB_RT];
    __shared__ float gn[VQB_RT];
    const uint32_t k = blockIdx.y, e = act[k], p0 = off[k], m = cnt[k];
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const v41_vq_mat mm = v41_vq_open<V3>(blob, (int)e, (int)which, R, C, (which == 2u && gr_all) ? gr_all + (size_t)e * OUTd : NULL);
    if (!mm.ok) { if (threadIdx.x == 0) atomicExch(bad, 1); return; }   /* 整块同一个专家: 要么全 ok 要么全不 ok, 不会卡在屏障上 */
    const bool live = j < mm.nidx_row;
    for (uint32_t t0 = 0; t0 < m; t0 += VQB_MT) {
        const uint32_t mt = m - t0 < VQB_MT ? m - t0 : VQB_MT;
        float acc[VQB_MT][8];
        #pragma unroll
        for (uint32_t t = 0; t < VQB_MT; t++)
            #pragma unroll
            for (uint32_t q = 0; q < 8u; q++) acc[t][q] = 0.f;
        for (uint32_t r0 = 0; r0 < R; r0 += VQB_RT) {
            const uint32_t nr = R - r0 < VQB_RT ? R - r0 : VQB_RT;
            __syncthreads();
            for (uint32_t q = threadIdx.x; q < VQB_MT * VQB_RT; q += blockDim.x) {
                const uint32_t t = q / VQB_RT, rr = q % VQB_RT;
                gs[t][rr] = (t < mt && rr < nr) ? g[(uint64_t)(p0 + t0 + t) * R + r0 + rr] : 0.f;
            }
            for (uint32_t rr = threadIdx.x; rr < VQB_RT; rr += blockDim.x) gn[rr] = rr < nr ? vqb_gain(mm, r0 + rr) : 0.f;
            __syncthreads();
            if (!live) continue;
            for (uint32_t rr = 0; rr < nr; rr++) {
                float c[8];
                v41_vq_cw<V3>(vqb_code<V3>(mm, r0 + rr, j), mm.cb, 0, c);
                const float gsc = gn[rr];
                #pragma unroll
                for (uint32_t t = 0; t < VQB_MT; t++) {
                    const float w = gs[t][rr] * gsc;   /* t ≥ mt 的槽搬进来时已是 0 */
                    #pragma unroll
                    for (uint32_t q = 0; q < 8u; q++) acc[t][q] += w * c[q];
                }
            }
        }
        if (live)
            for (uint32_t t = 0; t < mt; t++) {
                float *o = out + (uint64_t)(p0 + t0 + t) * C + (uint64_t)j * 8u;
                #pragma unroll
                for (uint32_t q = 0; q < 8u; q++) o[q] = accumulate ? o[q] + acc[t][q] : acc[t][q];
            }
    }
}

/* ---- 转置累加: 分段预取 + 寄存器直解(v3 载荷; vqst, 2026-10-03, 替掉 10-02 的瓦片版 vqt_kernel) ----
 * out[p][c] (+)= Σ_r g[p][r]·gain_r·W[r][c]  即  OUT(P×C) = (G ⊙ gain)(P×R) · W(R×C)。G ⊙ gain 先由 vqt_prescale_kernel 舍成 bf16(g16)。
 * 为什么换: 瓦片版每轮(64 行)把码字解进 A 瓦片再 ldmatrix.trans 读回, 位流不预取 —— 10-03 逐核表单发 7.6 / 8.4 ms(12 / 13 位), 是读位流墙的 4 倍多,
 * 卡在延迟上(同前向 vqm_kernel, 见 cuda_vq_reg_mma.inc.cu)。微基准(gguf-tools/bench/v41_vq_train_bench.cu, 训练包真路由)转置三发含预缩放:
 * 12 位层 26.3 → 16.2 ms, 13 位层 28.0 → 18.4 ms。
 * 【分工】一个 block(16 warp)管 (工作项, SW 个 64 列条) × 全部 R 行(SW = C/64 的不超过 16 的最大因子: 5120 列 16 条、2304 列 12 条,
 * 多出来的 warp 只帮着搬); warp w 管第 w 条。mma 里权重当 A: 64 列 = 4 个 m16 片, lane (g, q) 解码字列 g 的第 16kk+4q+{0..3} 行(4 个码字),
 * m16 片 i 的第 m 行 = 码字列 (m & 7) 的第 2i + (m >> 3) 个元素 ⇒ 4 片正好用完这 4 个码字的 8 个元素(PRMT 拼行对)。
 * g16 当 B: token 8 个一片, lane 取 token g 的第 16kk+4q..+3 行 = 一条 8 B 读, 正好是 (b0, b1)。
 * 【分段】一段 64 行: 位流(每行 SW·12 B 的列窗)、位平面(每行 SW B)、g16(≤ NTM 个 token × 64 行)cp.async 进 shared, 两段在途;
 * 每段提前两段把整行(不只列窗)bulk 预取进 L2 —— 同一工作项的几个列块同时在读同几行, L2 去重, DRAM 看到的是整行连续请求。
 * ★与瓦片版逐位同★: k16 组仍是同 16 行(组内逻辑 k = 2q+{0,1} ↔ 行 16kk+4q+{0,1}, 2q+8+{0,1} ↔ 16kk+4q+{2,3}), 权重当 A、g16 当 B 的朝向
 * 与瓦片版一样, 每个 k16 从零起算再 FADD 回累加器(张量核内部累加对齐时截断, 直接连乘会单向丢低位); 微基准 G_A / G_X 逐字节全同。
 * 精度: 权重 E4M3 → bf16 精确; g 是 f32 梯度, 转 bf16 有 2^-9 量级的相对舍入(bf16 乘、f32 累加); 输出不舍 bf16(梯度留 f32)。
 * 出错会怎样: g16 段里 token t 的 16 B 块 c 存在 (c ^ 2(t&3)), 写入与取址(cofs)必须同一个式子; 写错不报错, 只是梯度变垃圾 ——
 * 门 = 微基准逐位门 + 引擎合批门(kdprof packcheck=1: 结构 + 有限差分)。 */
#define VQST_PITCH 192u   /* shared 里位流一行的列窗(最多 16 条 × 12 B) */
static constexpr uint32_t vqst_smem(uint32_t cbb, int ext, uint32_t ntm) {   /* 码本 + 两段 × (位流 + 位平面 + g16) */
    return cbb + 2u * (64u * VQST_PITCH + (ext ? 64u * 16u : 0u) + ntm * 128u);
}
/* 一个工作项最多几个 token: 12 位层码本 bf16 进 shared(64 KB, 查完不用转), 32 个 token 时 128 个寄存器放不下要溢出 ⇒ 24;
 * 13 位层码本 E4M3(64 KB)查完现转, 32 个 token 不溢出。微基准里这两个各自最快。 */
#define VQST_NTM12 24u
#define VQST_NTM13 32u
static uint32_t vqst_item_tokens(uint32_t nbit) { return nbit == 13u ? VQST_NTM13 : VQST_NTM12; }

/* B 片预处理(10-02 夜): bf16(g·行增益) 一次算好 [nv][R], 主核按 bf16 直接搬。
 * 为什么: 原来主核每个列块都把 f32 的 g 整读一遍、现乘增益、现转 bf16 —— down 转置 16 个列块 = g 读 16 遍 f32。
 * 一 block 一个工作项: 线程按行 r 跨(读 g 合并访存), 每行的增益只解一次。 */
__global__ static void vqt_prescale_kernel(uint16_t *g16, const float *g, const uint8_t *blob, const vqp_item *items, const uint32_t *off,
                                           uint32_t which, uint32_t R, uint32_t C, const float *gr_all, uint32_t OUTd, int *bad) {
    const vqp_item it = items[blockIdx.x];
    const v41_vq_mat m = v41_vq_open<1>(blob, it.e, (int)which, R, C, (which == 2u && gr_all) ? gr_all + (size_t)it.e * OUTd : NULL);
    if (!m.ok) { if (threadIdx.x == 0) atomicExch(bad, 1); return; }
    const uint32_t base = off[it.e] + (uint32_t)it.t0, nt = (uint32_t)it.nt;
    for (uint32_t r = threadIdx.x; r < R; r += blockDim.x) {
        const float gn = vqb_gain(m, r);
        for (uint32_t t = 0; t < nt; t++) {
            const uint64_t o = (uint64_t)(base + t) * R + r;
            g16[o] = __bfloat16_as_ushort(__float2bfloat16_rn(g[o] * gn));
        }
    }
}

template <int EXT, int CBF, uint32_t NTM>
__global__ __launch_bounds__(VQS_THREADS, 1) static void vqst_kernel(
        float *out, const uint16_t *g16, const uint8_t *blob, const vqp_item *items, uint32_t nitems, const uint32_t *off,
        uint32_t which, uint32_t R, uint32_t C, uint32_t SW, int accumulate, uint32_t cb_bytes, int *bad) {
    constexpr uint32_t NTN = NTM / 8u, BSB = 64u * VQST_PITCH, EXB = EXT ? 64u * 16u : 0u, GTB = NTM * 128u, STB = BSB + EXB + GTB, PD = 2u;
    extern __shared__ __align__(16) uint8_t vqstsh[];
    uint8_t *cbs = vqstsh, *ring = vqstsh + cb_bytes;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u;
    {   /* v3 码本一层一本: 12 位层进 shared 时转 bf16(CBF 1), 13 位层原样 E4M3(CBF 0, 查时 vqm_cw<0> 现转) */
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, (int)which, R, C, NULL);
        if (!m0.ok || m0.nc * (CBF ? 16u : 8u) != cb_bytes) { if (tid == 0) atomicExch(bad, 1); return; }
        if (CBF == 0) v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
        else for (uint32_t v = tid; v < m0.nc; v += VQS_THREADS) *(uint4 *)(cbs + (size_t)v * 16u) = vqm_cw<0>(m0.cb, v);
    }
    __syncthreads();
    const uint32_t nu = C / (64u * SW), nwork = nitems * nu, nst = R / 64u, nwc = SW * 12u / 16u;   /* 每行列窗 nwc 块 16 B */
    const uint32_t cofs[4] = { ((0u + (q >> 1)) ^ (2u * (g & 3u))) << 4, ((2u + (q >> 1)) ^ (2u * (g & 3u))) << 4,
                               ((4u + (q >> 1)) ^ (2u * (g & 3u))) << 4, ((6u + (q >> 1)) ^ (2u * (g & 3u))) << 4 };
    const bool act = warp < SW;
    const uint32_t cbit = 12u * (8u * warp + g), cwo = (cbit >> 5) << 2, csh = cbit & 31u;   /* 本 lane 的码字列在列窗里的位偏移 */
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / nu];
        const uint32_t c0 = (w % nu) * 64u * SW, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, (int)which, R, C, NULL);
        if (!m.ok) { if (tid == 0) atomicExch(bad, 1); continue; }   /* 主机侧已按槽表查过: 走到这里 = 载荷坏了, 整次反传判失败 */
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3;
        const uint8_t *win = m.ix + (c0 / 8u) * 12u / 8u, *ewin = EXT ? m.ex + c0 / 64u : NULL;   /* 本单元列窗在每行里的起点 */
        auto pf = [&](uint32_t s) {   /* 第 s 段的 64 行整行进 L2 */
            if (s < nst && tid < 64u) {
                vqs_pf_l2(m.ix + (size_t)(64u * s + tid) * mrow, mrow & ~15u);
                if (EXT && tid == 0) vqs_pf_l2(m.ex + (size_t)64u * s * erow, (64u * erow) & ~15u);
            }
        };
        auto issue = [&](uint32_t s) {   /* 第 s 段(行 64s..)→ 槽 s & 1 */
            uint8_t *st = ring + (s & 1u) * STB;
            for (uint32_t k = tid; k < 64u * nwc; k += VQS_THREADS) {
                const uint32_t row = k / nwc, ch = k - row * nwc;
                vqs_cp<16>(st + row * VQST_PITCH + ch * 16u, win + (size_t)(64u * s + row) * mrow + ch * 16u);
            }
            if (EXT) for (uint32_t k = tid; k < 64u * (SW / 4u); k += VQS_THREADS) {
                const uint32_t row = k / (SW / 4u), ch = k - row * (SW / 4u);
                vqs_cp<4>(st + BSB + row * 16u + ch * 4u, ewin + (size_t)(64u * s + row) * erow + ch * 4u);
            }
            uint8_t *gd = st + BSB + EXB;
            for (uint32_t k = tid; k < nt * 8u; k += VQS_THREADS) {
                const uint32_t t = k >> 3, c = k & 7u;
                vqs_cp<16>(gd + t * 128u + ((c ^ (2u * (t & 3u))) << 4), g16 + (uint64_t)(base + t) * R + 64u * s + 8u * c);
            }
        };
        #pragma unroll
        for (uint32_t s = 0; s < PD; s++) pf(s);
        issue(0); vqs_commit();
        const uint32_t nnt = (nt + 7u) >> 3;
        float acc[4][NTN][4];
        #pragma unroll
        for (uint32_t i = 0; i < 4u; i++)
            #pragma unroll
            for (uint32_t j = 0; j < NTN; j++) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f; }
        auto run = [&](auto nntc) {   /* 段循环按本项的 n8 片数 NNT 实例化(编译期常量) */
            constexpr uint32_t NNT = decltype(nntc)::value;
            for (uint32_t s = 0; s < nst; s++) {
                vqs_wait<0>();
                __syncthreads();   /* 第 s 段全到了; 也保证大家都用完了第 s-1 段的槽 ⇒ 下面可以往里搬第 s+1 段 */
                if (s + 1u < nst) issue(s + 1u);
                vqs_commit();
                pf(s + PD);
                if (!act) continue;
                const uint8_t *st = ring + (s & 1u) * STB, *gs = st + BSB + EXB + 8u * (q & 1u) + g * 128u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; kk++) {
                    uint4 E[4];
                    #pragma unroll
                    for (uint32_t u = 0; u < 4u; u++) {
                        const uint32_t row = 16u * kk + 4u * q + u;
                        const uint8_t *p = st + row * VQST_PITCH + cwo;
                        uint32_t v = __funnelshift_r(*(const uint32_t *)p, *(const uint32_t *)(p + 4), csh) & 0xFFFu;
                        if (EXT) v |= (((uint32_t)st[BSB + row * 16u + warp] >> g) & 1u) << 12;
                        E[u] = vqm_cw<CBF>(cbs, v);
                    }
                    const uint8_t *bp = gs + cofs[kk];
                    uint2 x[NTN];
                    #pragma unroll
                    for (uint32_t j = 0; j < NNT; j++) x[j] = *(const uint2 *)(bp + j * 8u * 128u);
                    #pragma unroll
                    for (uint32_t i = 0; i < 4u; i++) {
                        /* 片 i: lane 的 m = g 取元素 2i(字 i 的低半), m = g+8 取 2i+1(高半); 两行一拼 */
                        const uint32_t w0 = i == 0u ? E[0].x : i == 1u ? E[0].y : i == 2u ? E[0].z : E[0].w;
                        const uint32_t w1 = i == 0u ? E[1].x : i == 1u ? E[1].y : i == 2u ? E[1].z : E[1].w;
                        const uint32_t w2 = i == 0u ? E[2].x : i == 1u ? E[2].y : i == 2u ? E[2].z : E[2].w;
                        const uint32_t w3 = i == 0u ? E[3].x : i == 1u ? E[3].y : i == 2u ? E[3].z : E[3].w;
                        const uint32_t a[4] = { __byte_perm(w0, w1, 0x5410u), __byte_perm(w0, w1, 0x7632u),
                                                __byte_perm(w2, w3, 0x5410u), __byte_perm(w2, w3, 0x7632u) };
                        #pragma unroll
                        for (uint32_t j = 0; j < NNT; j++) {
                            float t[4];
                            vqs_mma0(t, a, x[j].x, x[j].y);
                            #pragma unroll
                            for (int c = 0; c < 4; c++) acc[i][j][c] += t[c];
                        }
                    }
                }
            }
        };
#define VQS_NNT(n) std::integral_constant<uint32_t, ((n) < NTN ? (n) : NTN)>{}
        switch (nnt) {
        case 1: run(VQS_NNT(1)); break;
        case 2: run(VQS_NNT(2)); break;
        case 3: run(VQS_NNT(3)); break;
        default: run(VQS_NNT(4)); break;
        }
#undef VQS_NNT
        __syncthreads();   /* 下一个单元的序幕要覆盖这两个槽 */
        /* 出口: 片 i 的 c0/c1 = (码字列 g 的元素 2i, token 8j+2q+{0,1}), c2/c3 = (元素 2i+1, 同 token) ⇒ 一个 token 拿码字列 g 的 8 个连续列 */
        if (act) {
            #pragma unroll
            for (uint32_t j = 0; j < NTN; j++) {
                if (j >= nnt) break;
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t t = 8u * j + 2u * q + e;
                    if (t >= nt) continue;
                    float *o = out + (uint64_t)(base + t) * C + c0 + 64u * warp + 8u * g;
                    #pragma unroll
                    for (uint32_t i = 0; i < 4u; i++) {
                        const float v0 = acc[i][j][e], v1 = acc[i][j][2u + e];
                        o[2u * i] = accumulate ? o[2u * i] + v0 : v0;
                        o[2u * i + 1u] = accumulate ? o[2u * i + 1u] + v1 : v1;
                    }
                }
            }
        }
    }
}

static struct { vqp_item *d; uint64_t cap; uint32_t *doff; uint64_t doff_cap; uint32_t nitems, nv; int nsm; int occ[2];
                uint16_t *g16; uint64_t g16_cap; } g_vqt;   /* g16 = B 片预处理出口 [nv][R] bf16, 三发转置依次复用 */

/* 本层工作项(每专家按 ≤ bn 个 token 切, bn = vqst_item_tokens)与专家偏移进设备; 一层三发转置共用 */
static int vqt_prepare(const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert, uint32_t bn) {
    uint32_t nit = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) nit += (cnt[e] + bn - 1u) / bn;
    if (!nit) return 0;
    vqp_item *ih = (vqp_item *)malloc((size_t)nit * sizeof(vqp_item));
    if (!ih) return 0;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_total_expert; e++)
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += bn) {
            ih[k].e = (int32_t)e; ih[k].t0 = (int32_t)t0; ih[k].nt = (int32_t)(cnt[e] - t0 < bn ? cnt[e] - t0 : bn); k++;
        }
    int ok = vqp_grow((void **)&g_vqt.d, &g_vqt.cap, nit, sizeof(vqp_item), "vqt items") &&
             vqp_grow((void **)&g_vqt.doff, &g_vqt.doff_cap, n_total_expert + 1u, sizeof(uint32_t), "vqt off") &&
             cudaMemcpyAsync(g_vqt.d, ih, (size_t)nit * sizeof(vqp_item), cudaMemcpyHostToDevice, g_cur_stream) == cudaSuccess &&
             cudaMemcpyAsync(g_vqt.doff, off_h, (size_t)(n_total_expert + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice, g_cur_stream) == cudaSuccess;
    free(ih);   /* 可分页源: cudaMemcpyAsync 返回前已拷进暂存(同 vqm_run) */
    if (!ok) { (void)cudaGetLastError(); return 0; }
    if (!g_vqt.nsm && cudaDeviceGetAttribute(&g_vqt.nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    g_vqt.nitems = nit; g_vqt.nv = off_h[n_total_expert];
    return 1;
}

/* 一发转置(vqt_prepare 之后): which 0/1/2 = gate/up/down, R×C = 该矩阵的行×列; nc = 本层码本词数(12 位 4096 / 13 位 8192) */
static int vqt_launch(float *out, const float *g, const uint8_t *blob, uint32_t which, uint32_t R, uint32_t C, uint32_t nc,
                      const float *gr_all, uint32_t OUTd, int accumulate, int *bad) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    const uint32_t ns = C / 64u; uint32_t sw = 16u; while (sw > 4u && ns % sw) sw--;   /* 一个单元几条: C/64 的不超过 16 的最大因子 */
    if ((nbit != 12u && nbit != 13u) || R % 64u || C % 64u || ns % sw || sw % 4u) {
        fprintf(stderr, "ds4: [bwd] 转置张量核不认这个形状(码本 %u 词, %u×%u)\n", nc, R, C); return 0;
    }
    if (!vqp_grow((void **)&g_vqt.g16, &g_vqt.g16_cap, (uint64_t)g_vqt.nv * R, sizeof(uint16_t), "vqt g16")) return 0;
    vqt_prescale_kernel<<<g_vqt.nitems, 256, 0, g_cur_stream>>>(g_vqt.g16, g, blob, g_vqt.d, g_vqt.doff, which, R, C, gr_all, OUTd, bad);
    if (!cuda_ok(cudaGetLastError(), "bwd vq prescale")) return 0;
    const uint32_t nwork = g_vqt.nitems * (ns / sw);
#define VQT_GO(E, CBF, NTM) do { \
        const uint32_t cbb = nc * (CBF ? 16u : 8u), shb = vqst_smem(cbb, E, NTM); \
        int *occ = &g_vqt.occ[E]; \
        if (*occ == 0) { \
            if (cudaFuncSetAttribute(vqst_kernel<E, CBF, NTM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) != cudaSuccess || \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, vqst_kernel<E, CBF, NTM>, (int)VQS_THREADS, shb) != cudaSuccess || *occ <= 0) { \
                (void)cudaGetLastError(); *occ = -1; } \
        } \
        if (*occ < 0) { fprintf(stderr, "ds4: [bwd] 转置张量核开不出 %u KB shared\n", shb >> 10); return 0; } \
        const uint32_t gsz = (uint32_t)g_vqt.nsm * (uint32_t)*occ; \
        vqst_kernel<E, CBF, NTM><<<gsz < nwork ? gsz : nwork, VQS_THREADS, shb, g_cur_stream>>>(out, g_vqt.g16, blob, g_vqt.d, g_vqt.nitems, g_vqt.doff, \
                                                                                           which, R, C, sw, accumulate, cbb, bad); \
        return cuda_ok(cudaGetLastError(), "bwd vq tdot reg"); \
    } while (0)
    if (nbit == 13u) VQT_GO(1, 0, VQST_NTM13);
    VQT_GO(0, 1, VQST_NTM12);
#undef VQT_GO
}

/* 发射器: which 0/1/2 = gate/up/down; R×C = 该矩阵的行×列; nact 个有 token 的专家(act/off/cnt 设备数组) */
static int vqb_rowdot(float *out, const __nv_bfloat16 *x, const uint8_t *blob, uint32_t ver, uint32_t which, uint32_t R, uint32_t C,
                      const uint32_t *act, const uint32_t *off, const uint32_t *cnt, uint32_t nact, const float *gr_all, uint32_t OUTd, int *bad) {
    const dim3 grid((R + 3u) / 4u, nact);
    if (ver == 3u) vqb_rowdot_kernel<1><<<grid, 128, 0, g_cur_stream>>>(out, x, blob, which, R, C, act, off, cnt, gr_all, OUTd, bad);
    else vqb_rowdot_kernel<0><<<grid, 128, 0, g_cur_stream>>>(out, x, blob, which, R, C, act, off, cnt, gr_all, OUTd, bad);
    return cuda_ok(cudaGetLastError(), "bwd vq rowdot");
}
static int vqb_tdot(float *out, const float *g, const uint8_t *blob, uint32_t ver, uint32_t which, uint32_t R, uint32_t C,
                    const uint32_t *act, const uint32_t *off, const uint32_t *cnt, uint32_t nact, const float *gr_all, uint32_t OUTd, int accumulate, int *bad) {
    const dim3 grid((C / 8u + 127u) / 128u, nact);
    if (ver == 3u) vqb_tdot_kernel<1><<<grid, 128, 0, g_cur_stream>>>(out, g, blob, which, R, C, act, off, cnt, gr_all, OUTd, accumulate, bad);
    else vqb_tdot_kernel<0><<<grid, 128, 0, g_cur_stream>>>(out, g, blob, which, R, C, act, off, cnt, gr_all, OUTd, accumulate, bad);
    return cuda_ok(cudaGetLastError(), "bwd vq tdot");
}
