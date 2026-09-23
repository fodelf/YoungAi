/* cuda_vq_persist.inc.cu — v3 专家"解码即乘"的常驻核(纯解码 n=1, 2026-09-23)。
 *
 * 【为什么】v3 的码本**一层一本**(cuda_vq_row.inc.cu 文件头: 同层所有专家的 gate/up/down 都指 blob 头后同一处),
 * 可 v41_vq_gateup/down_kernel 还是 v2 的写法: 每个 block 先把码本(32/64 KB)搬进 shared、全 block 同步再算,
 * gateup 还要把"同一本"再搬一遍给 up。这两个核 1024 线程 × 56 寄存器, 每 SM 只挂 1 个 block ⇒ 搬码本时整个 SM 干等。
 * 一层 gateup 216 个 block × 2 次、down 240 个 block × 1 次, 每 SM 一层要等 ~14 次; 外加每 block 三次全体屏障让
 * 32 个 warp 锁步(09-18 认定的剩余等待来源之一)。
 * 【怎么改】grid = SM 数, 每 SM 一个常驻 block: 码本一层只搬一次, 之后每个 warp 顺序做一段连续的"(专家, 行)"工作项,
 * 再没有全 block 屏障。gateup 同一个 warp 对同一行先算 gate 再算 up, 结果在寄存器里直接 SwiGLU(不经 h 暂存)。
 * 【逐字节同】每一行仍走同一个 v41_vq_row_dot(同 lane↔索引、同累加序、同规约树); gate 值原来经 h 以 bf16 暂存再读回,
 * 它本来就在 bf16 格点上(v41_bf16r 出口), 存取无损 ⇒ SwiGLU 的输入逐位同。门 = 与原核的输出 cmp。
 * 【出错会怎样】某个载荷的码本不指向本层那一本(格式违约)⇒ __trap() 硬停, 不许拿错码本算出一套"看着正常"的假权重。
 * ★必须在 cuda_vq_decode.inc.cu 之后 include★(用它的 v41_vq_cb_to_shared / v41_vq_swiglu / V41_VQ_WARPS)。 */

/* 本块第一个有效对的某个矩阵的码本 = 本层码本(全块同值, 所以下面的 return 是全块一致的, 不会有 warp 漏掉屏障) */
__device__ __forceinline__ static const uint8_t *v41_vq_layer_cb(const uint8_t *blob, const int32_t *sel, uint32_t np, int which,
                                                                 uint32_t rows, uint32_t cols) {
    for (uint32_t p = 0; p < np; p++) {
        const int32_t e = sel[p];
        if (e < 0) continue;
        const v41_vq_mat m = v41_vq_open<1>(blob, e, which, rows, cols, NULL);
        if (m.ok) return m.cb;
    }
    return NULL;
}

/* ★一段连续行按一条位流读(2026-09-23 第二刀)★
 * 病(ncu, 常驻核第一版): long_scoreboard ≈ 14, 激活挪进 shared 后 gateup 一点没动 ⇒ 等的是位流。v41_vq_row_dot 的流水是
 * "算这一块时读下一块"(一块 8 轮), 而一行 gateup 20 轮 = 8+8+**4**、down 9 轮 = 8+**1**: 每行末块只有 4 轮(1 轮)的活去盖下一行
 * 首块的 DRAM 延迟(带载 1.2~1.4 µs), 盖不住 ⇒ **每行开头停一次**; 行增益 gr[r](down 还有侧车 gov[r])又是行尾才发、立刻要用
 * 的全局读, 每行再停一次。
 * 改: 同一矩阵相邻行的位流首尾相接(一行 R 轮 × 12 字, 位平面一行 R 个字), 所以一个 warp 的连续 n 行就是一条连续位流 ——
 * 块按"段内全局轮号"每 8 轮一切, 不在行边界重新起块; 一行的轮走满就当场规约、乘增益、换行。行增益在段首预取(lane i 拿第 i 行)。
 * ★逐字节同★: 每行仍从 0 起、按轮序做同一个 v41_vq_dot8 累加, 规约树与增益乘法式同 v41_vq_row_dot; 一轮内 lane↔字的映射只看
 * 轮在块内的位置 k(每轮恰 12 个整字, 块内起字 = 12k), 与原来逐字相同。 */
/* ★判负存档(2026-09-23)★ "激活在 shared 里存 f32, 省每轮 8 条 bf16 拆包": 同二进制状态成对, gateup 6.46 → 6.67 ms(更慢),
 * 输出逐字节同。每轮多一条 LDS.128 比省下的 8 条整数指令贵 ⇒ 这个核不是被指令条数卡住的。 */
/* ★判负存档(2026-09-23)★ "激活在 shared 里存 f32, 省每轮 8 条 bf16 拆包": 同二进制状态成对, gateup 6.46 → 6.67 ms(更慢),
 * 输出逐字节同。每轮多一条 LDS.128 比省下的 8 条整数指令贵 ⇒ 这个核不是被指令条数卡住的。 */
/* ★判负存档(2026-09-23)★ "位流走 shared 环, cp.async 提前 S 块(32 KB 码本层 S=4 / 64 KB 层 S=2), 激活改回读全局": 依据是 ncu
 * long_scoreboard ≈ 9、位流 L2 命中 16%, 怀疑寄存器上限让编译器把"下一块"的 LDG 挪到用处前。纯墙钟成对 33.46 → 33.62 ms(噪声内偏慢),
 * 输出逐字节同 ⇒ 位流到得晚不是剩余等待的来源(与 09-18 "L2 预取提前两块"噪声内同一句话)。下一步只能上 ncu 逐指令采样定位。 */
template <int NBIT, int EXT>
__device__ __forceinline__ static float v41_vq_stream(const v41_vq_mat &m, uint32_t r0, uint32_t n, const uint32_t *xs, const uint8_t *cbs,
                                                      v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = 12u;                                         /* v3 主流恒 12 位 */
    const uint32_t R = m.nidx_row >> 5, G = n * R;                       /* 每行轮数 / 段内总轮数 */
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *base = v41_vq_row_ptr<NBIT, 1>(m, r0);
    const uint32_t *exb = EXT ? v41_vq_ext_ptr(m, r0) : NULL;
    float g1 = 0.f, g2 = 1.f;                                            /* lane i: 第 r0+i 行的载荷增益 / 侧车增益 */
    if (lane < n) { __half gh; memcpy(&gh, m.gr + (size_t)(r0 + lane) * 2u, 2); g1 = __half2float(gh); if (m.gov) g2 = m.gov[r0 + lane]; }
    v41_vq_blk cur = *carry;
    float acc = 0.f, res = 0.f;
    uint32_t kin = 0, row = 0;
    for (uint32_t g0 = 0; g0 < G; g0 += 8u) {
        v41_vq_blk nxt;
        if (g0 + 8u < G) { const uint32_t rem = G - g0 - 8u, nr = rem < 8u ? rem : 8u;
                           nxt = v41_vq_blk_load<EXT>(base + (size_t)(g0 + 8u) * MB, nr * MB, EXT ? exb + (g0 + 8u) : NULL, nr); }
        else if (next) nxt = v41_vq_blk_load<EXT>(next, 8u * MB, nextex, 8u);   /* 下一段首块(调用方保证那段 ≥ 8 轮) */
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; nxt.ex = 0u; }
        const uint32_t rounds = G - g0 < 8u ? G - g0 : 8u;
        #pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            if (k >= rounds) break;
            const uint4 xa = *(const uint4 *)(xs + (size_t)(kin * 32u + lane) * 4u);
            const uint32_t f = k * MB;                                   /* 以下四行与 v41_vq_blk_rounds 逐字同 */
            const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
            const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
            const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
            const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
            uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
            if (EXT) { const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k); v |= ((exw >> lane) & 1u) << 12; }
            acc += v41_vq_dot8<1>(v, xa, cbs, 1);
            if (++kin == R) {                                            /* 一行走满: 同 v41_vq_row_dot 的收尾 */
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                const float a1 = __shfl_sync(0xffffffffu, g1, (int)row), a2 = __shfl_sync(0xffffffffu, g2, (int)row);
                const float val = acc * a1 * (m.gov ? a2 : 1.0f);
                if (lane == row) res = val;
                acc = 0.f; kin = 0u; row++;
            }
        }
        cur = nxt;
    }
    *carry = cur;
    return res;
}

/* ★__launch_bounds__(1024, 1)★: 1024 线程/block ⇒ 每线程最多 64 寄存器, 超了 launch 直接 too many resources。
 * 13 位实例(带位平面)自然会超, 必须钉; 钉完看 cuobjdump 的 LOCAL, 有 spill 就不许进。
 * 工作划分: 全核 np×rows 个"(专家, 行)"按 warp 切成连续段, 段再按 ≤32 行、不跨专家切(lane i 收第 i 行的结果)。 */
#define V41_VQ_SEG(U, UEND, ROWS, P, R0, N) \
    const uint32_t P = (U) / (ROWS), R0 = (U) % (ROWS); \
    uint32_t N = (UEND) - (U); if (N > (ROWS) - R0) N = (ROWS) - R0; if (N > 32u) N = 32u
template <int NBIT, int EXT>
__global__ static void __launch_bounds__(1024, 1) v41_vq_gu_persist_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                                                uint32_t IN, uint32_t MID, uint32_t np, float clamp, uint32_t cbb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t gw = blockIdx.x * V41_VQ_WARPS + (threadIdx.x >> 5), nw = gridDim.x * V41_VQ_WARPS;
    const uint32_t total = np * MID, per = (total + nw - 1u) / nw, lane = threadIdx.x & 31u;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    /* PDL: 码本是常量, 用 0 号专家的载荷定位(v3 同层共用一本)先搬; sel/x 是上游产出, 等 v41_pdl_wait 之后才碰 */
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 0, MID, IN, NULL);
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 0, MID, IN); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (u < uend) {   /* 首段首块先发, 藏在激活搬运后面 */
        const int32_t e = sel[u / MID];
        if (e >= 0) { const v41_vq_mat mg = v41_vq_open<1>(blob, e, 0, MID, IN, NULL); if (mg.ok) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(mg, u % MID); }
    }
    v41_vq_cb_to_shared(vqsh + cbb, (const uint8_t *)x, IN * 2u);   /* 激活(bf16)也进 shared(对 down 有小收益, gateup 持平) */
    __syncthreads();   /* 全核唯一一次全体屏障 */
    x = (const uint32_t *)(vqsh + cbb);
    bool first = true;
    while (u < uend) {
        V41_VQ_SEG(u, uend, MID, p, r0, n);
        u += n;
        const int32_t e = sel[p];
        if (e < 0) { first = false; continue; }   /* 原核: 专家号无效 ⇒ 不写 h */
        const v41_vq_mat mg = v41_vq_open<1>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<1>(blob, e, 1, MID, IN, NULL);
        if (!mg.ok || !mu.ok) { first = false; continue; }   /* 原核: 载荷不对 ⇒ 这一对不写 h */
        if (mg.cb != cb || mu.cb != cb) __trap();   /* 格式违约: 不许拿别的码本算 */
        if (!first) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(mg, r0);
        first = false;
        /* gate 段 → up 段(gate 末块顺手装 up 首块); 两段都 ≥ 8 轮(一行 20 轮) */
        const float gv = v41_bf16r(v41_vq_stream<NBIT, EXT>(mg, r0, n, x, vqsh, &carry, v41_vq_row_ptr<NBIT, 1>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL));
        const float ui = v41_bf16r(v41_vq_stream<NBIT, EXT>(mu, r0, n, x, vqsh, &carry, NULL, NULL));
        if (lane < n) h[(uint64_t)p * MID + r0 + lane] = v41_vq_swiglu(gv, ui, clamp);
    }
}

template <int NBIT, int EXT>
__global__ static void __launch_bounds__(1024, 1) v41_vq_dn_persist_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *h,
                                                uint32_t MID, uint32_t OUT, uint32_t np, uint32_t cbb, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t gw = blockIdx.x * V41_VQ_WARPS + (threadIdx.x >> 5), nw = gridDim.x * V41_VQ_WARPS;
    const uint32_t total = np * OUT, per = (total + nw - 1u) / nw, lane = threadIdx.x & 31u;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 2, OUT, MID, NULL);   /* PDL: 同 gateup, 码本先搬 */
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 2, OUT, MID); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (u < uend) {
        const int32_t e = sel[u / OUT];
        if (e >= 0) { const v41_vq_mat md = v41_vq_open<1>(blob, e, 2, OUT, MID, NULL); if (md.ok) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(md, u % OUT); }
    }
    v41_vq_cb_to_shared(vqsh + cbb, (const uint8_t *)h, np * MID * 2u);   /* 全部对的 h(bf16)进 shared: 6 × 2304 × 2 B = 27 KB */
    __syncthreads();
    h = (const uint32_t *)(vqsh + cbb);
    bool first = true;
    while (u < uend) {
        V41_VQ_SEG(u, uend, OUT, p, r0, n);
        u += n;
        const int32_t e = sel[p];
        if (e < 0) { first = false; continue; }                                   /* 原核: 专家号无效 ⇒ 不写 */
        const v41_vq_mat md = v41_vq_open<1>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
        if (!md.ok) { if (lane < n) partial[(uint64_t)p * OUT + r0 + lane] = 0.f; first = false; continue; }   /* 原核: 载荷不对 ⇒ 写 0 */
        if (md.cb != cb) __trap();
        if (!first) carry = v41_vq_row_first_blk<NBIT, 1, EXT>(md, r0);
        first = false;
        const float y = v41_bf16r(v41_vq_stream<NBIT, EXT>(md, r0, n, h + (uint64_t)p * (MID / 2u), vqsh, &carry, NULL, NULL));
        if (lane < n) partial[(uint64_t)p * OUT + r0 + lane] = y;
    }
}
#undef V41_VQ_SEG

/* 发射: stage 0 = gateup, 1 = down。返回 1 = 发了; 0 = 启动失败(调用方按错误处理, 不回退旧核)。
 * opt-in 的量按"这个实例已批到多少"记(与 v41_vq_fused_moe_n 那段同一个坑: 两档码本 32/64 KB 并存)。 */
template <int NBIT, int EXT>
static int v41_vq_persist_launch(int stage, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                                 uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t np, float clamp, uint32_t cbb, const float *gr) {
    static uint32_t s_optin[2] = { 0u, 0u };   /* [gateup, down] 已批到的动态 shared */
    static int s_nsm = 0;
    const uint32_t need = cbb + (stage == 0 ? IN * 2u : np * MID * 2u);   /* 码本 + 激活(bf16) */
    if (!s_nsm && cudaDeviceGetAttribute(&s_nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        fprintf(stderr, "ds4: [v41] VQ 常驻核取不到 SM 数: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    if (s_optin[stage] < need) {
        const cudaError_t e = stage == 0
            ? cudaFuncSetAttribute(v41_vq_gu_persist_kernel<NBIT, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need)
            : cudaFuncSetAttribute(v41_vq_dn_persist_kernel<NBIT, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need);
        if (e != cudaSuccess) {   /* 例: 64 KB 码本 + 验证批的大 h 超 99 KB。常驻核只接 n=1, 现役最大 91 KB */
            fprintf(stderr, "ds4: [v41] VQ 常驻核批不到 %u B 动态 shared(码本 %u + 激活): %s\n", need, cbb, cudaGetErrorString(e));
            (void)cudaGetLastError();
            return 0;
        }
        s_optin[stage] = need;
    }
    v41_pdl_register((const void *)v41_vq_gu_persist_kernel<NBIT, EXT>);   /* 两核都在碰 sel/x/h 之前 v41_pdl_wait */
    v41_pdl_register((const void *)v41_vq_dn_persist_kernel<NBIT, EXT>);
    if (stage == 0) v41_vq_gu_persist_kernel<NBIT, EXT><<<(unsigned)s_nsm, V41_VQ_WARPS * 32u, need, g_cur_stream>>>(h, blob, sel, xb, IN, MID, np, clamp, cbb);
    else v41_vq_dn_persist_kernel<NBIT, EXT><<<(unsigned)s_nsm, V41_VQ_WARPS * 32u, need, g_cur_stream>>>(part, blob, sel, (const uint32_t *)h, MID, OUT, np, cbb, gr);
    return cuda_ok(cudaGetLastError(), stage == 0 ? "v41 vq gateup persist" : "v41 vq down persist");
}
