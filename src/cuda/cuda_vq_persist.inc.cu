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
/* ★M = 这一段同时乘几条激活(2026-09-24)★: 纯解码 M=1(xs[0] = 那一条); 投机验证批里同一专家被 nt ≤ M 个 token 选中时,
 * 位流读一遍、码字解一次(v41_vq_cw), 对每个 token 各做一次 v41_vq_dot8_cw —— 与 M=1 的 v41_vq_dot8(= cw + dot8_cw, 内联后
 * 同一棵表达式树)同式, 每个 token 自己的累加序/规约树/增益乘序一字不差 ⇒ 验证批第 t 行 == 纯解码那一位, 投机同轨门靠这一条。
 * M=1 仍走原来那一句 v41_vq_dot8 调用, 纯解码的代码生成不动。结果: lane i 的 res[j] = 第 r0+i 行对第 j 个 token 的值。 */
template <int NBIT, int EXT, int M>
__device__ __forceinline__ static void v41_vq_stream(const v41_vq_mat &m, uint32_t r0, uint32_t n, const uint32_t *const *xs, uint32_t nt,
                                                     const uint8_t *cbs, v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex,
                                                     float *res) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = 12u;                                         /* v3 主流恒 12 位 */
    const uint32_t R = m.nidx_row >> 5, G = n * R;                       /* 每行轮数 / 段内总轮数 */
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *base = v41_vq_row_ptr<NBIT, 1>(m, r0);
    const uint32_t *exb = EXT ? v41_vq_ext_ptr(m, r0) : NULL;
    float g1 = 0.f, g2 = 1.f;                                            /* lane i: 第 r0+i 行的载荷增益 / 侧车增益 */
    if (lane < n) { __half gh; memcpy(&gh, m.gr + (size_t)(r0 + lane) * 2u, 2); g1 = __half2float(gh); if (m.gov) g2 = m.gov[r0 + lane]; }
    v41_vq_blk cur = *carry;
    float acc[M];
    #pragma unroll
    for (int j = 0; j < M; j++) { acc[j] = 0.f; res[j] = 0.f; }
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
            uint4 xa0;   /* M=1: 进轮就读(原样); M>1: 每条激活用到时才读, 少占 4(M-1) 个寄存器(1024 线程下每线程只有 64 个) */
            if (M == 1) xa0 = *(const uint4 *)(xs[0] + (size_t)(kin * 32u + lane) * 4u);
            const uint32_t f = k * MB;                                   /* 以下四行与 v41_vq_blk_rounds 逐字同 */
            const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
            const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
            const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
            const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
            uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
            if (EXT) { const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k); v |= ((exw >> lane) & 1u) << 12; }
            if (M == 1) acc[0] += v41_vq_dot8<1>(v, xa0, cbs, 1);
            else {
                float c[8];
                v41_vq_cw<1>(v, cbs, 1, c);
                #pragma unroll
                for (int j = 0; j < M; j++)
                    if ((uint32_t)j < nt) acc[j] += v41_vq_dot8_cw(c, *(const uint4 *)(xs[j] + (size_t)(kin * 32u + lane) * 4u));
            }
            if (++kin == R) {                                            /* 一行走满: 同 v41_vq_row_dot 的收尾 */
                #pragma unroll
                for (int j = 0; j < M; j++)
                    if ((uint32_t)j < nt) for (int o = 16; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
                const float a1 = __shfl_sync(0xffffffffu, g1, (int)row), a2 = __shfl_sync(0xffffffffu, g2, (int)row);
                #pragma unroll
                for (int j = 0; j < M; j++) {
                    const float val = acc[j] * a1 * (m.gov ? a2 : 1.0f);
                    if (lane == row && (uint32_t)j < nt) res[j] = val;
                    acc[j] = 0.f;
                }
                kin = 0u; row++;
            }
        }
        cur = nxt;
    }
    *carry = cur;
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
        float gs, us;
        v41_vq_stream<NBIT, EXT, 1>(mg, r0, n, &x, 1u, vqsh, &carry, v41_vq_row_ptr<NBIT, 1>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL, &gs);
        v41_vq_stream<NBIT, EXT, 1>(mu, r0, n, &x, 1u, vqsh, &carry, NULL, NULL, &us);
        if (lane < n) h[(uint64_t)p * MID + r0 + lane] = v41_vq_swiglu(v41_bf16r(gs), v41_bf16r(us), clamp);
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
        const uint32_t *hp = h + (uint64_t)p * (MID / 2u);
        float ys;
        v41_vq_stream<NBIT, EXT, 1>(md, r0, n, &hp, 1u, vqsh, &carry, NULL, NULL, &ys);
        if (lane < n) partial[(uint64_t)p * OUT + r0 + lane] = v41_bf16r(ys);
    }
}

/* ==== 投机验证批(n_tok ≥ 2)的常驻核(2026-09-24) ====
 * 【为什么】验证批原来走"逐对核(只被一个 token 选中的专家) + 分组核(被多个 token 选中的)", 真实 CFO 请求上 4 行一次专家 ~36 ms
 * ≈ 每行 9 ms, 与单 token 常驻核(9.6 ms/6 个专家)同一档 —— 同一专家被几个 token 选中时省下的只是码本查表, 位流/每 block
 * 启停/三次全体屏障照付。而 4 行里唯一专家只有 16/24(09-24 d0a [moe-uniq]), 按常驻核 ~1.6 ms/专家该是 ~26 ms。
 * dspark_sim 在同请求取料上: 每多验一行 12.45 → 9.5 ms ⇒ 37.3 → 41.0 t/s, 到 7.3 ⇒ 44.6。
 * 【怎么做】工作项从"(对, 行)"换成"(唯一专家, 行)": 块头按 order(对按专家排好序)把同一专家的对归成一组, 位流读一遍、码字解一次,
 * 对组里 nt 个 token 各乘一遍(v41_vq_stream<M>)。码本一层一本、每 SM 一个常驻 block 只搬一次, 与 n=1 常驻核同。
 * 激活/h 不进 shared: 13 位层码本 64 KB + 4 行激活 40 KB 超过 99 KB 的 opt-in 上限, 走全局(L1 命中)。
 * 【寄存器】n=1 常驻核已顶在 64 个(1024 线程); 多 token 只多 M-1 个累加器 + 结果, 激活用到时才读 ⇒ M=2 仍放得进 32 warp(见 V41_VQPN_M)。
 * 【逐字节同】见 v41_vq_stream 头注释; 门 = d1_kv_ring_gate.sh 真实请求档的"同轨"(投机 == 纯解码逐字节)。 */
#define V41_VQPN_MAXP 64u   /* 对数上限: V41_GEMV_MAX_TOK(8) × top-k(6) = 48 */
/* ★一组最多 M=2 个 token, 多的切成几组(2026-09-24 实测定的)★: 首版按批大小分档 M=2/4/6, 为放下累加器把 warp 减到 32/24/16;
 * 真实请求 4 行验证 75.5 → 80.7 ms(更慢)—— 这个核是等位流的(long_scoreboard 为主), warp 少一档在飞的请求就少一档, 省下的专家字节
 * 全赔回去。所以一律 32 warp、M=2: 被 m 个 token 选中的专家读 ⌈m/2⌉ 遍位流(4 行里 m ≥ 3 的专家很少)。 */
#define V41_VQPN_M 2u
/* 块头: 按 order 把同一专家的对归组(q = 组在 order 里的起点, m = 组里几对, ≤ V41_VQPN_M); 线程 0 做, 其余等屏障 */
/* ★判负存档(2026-09-24)★ "组表挪进 v41_vq_order_kernel 一层算一次, 两个常驻核直接读": 线程 0 这段串行(每对两次前后依赖的全局读)
 * 看着像每发白等十几 µs。真实 CFO 请求成对: 投机 43.01 → 43.03 t/s, 一轮 70.3 → 70.3 ms, 验证 k1 46.1 → 45.8 / k3 65.3 → 65.2(噪声内),
 * 输出逐字节同 ⇒ 走图 + PDL 下这段已被藏住, 不是钱。代码已回退。 */
__device__ __forceinline__ static void v41_vqpn_groups(const int32_t *sel, const int32_t *order, uint32_t np,
                                                       uint32_t *gq, uint32_t *gm, uint32_t *ng) {
    if (threadIdx.x == 0) {
        uint32_t g = 0, q = 0;
        while (q < np) {
            const int32_t e = sel[order[q]];
            uint32_t m = 1u;
            while (q + m < np && sel[order[q + m]] == e && m < V41_VQPN_M) m++;
            gq[g] = q; gm[g] = m; g++; q += m;
        }
        *ng = g;
    }
}
template <int NBIT, int EXT, int M, int NW>
__global__ static void __launch_bounds__(NW * 32, 1) v41_vq_gu_persist_n_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel,
        const int32_t *order, const uint32_t *x, uint32_t IN, uint32_t MID, uint32_t K, uint32_t np, float clamp, uint32_t cbb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[V41_VQPN_MAXP], gm[V41_VQPN_MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 0, MID, IN, NULL);   /* PDL: 码本是常量先搬, sel/order/x 等 v41_pdl_wait 之后才碰 */
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 0, MID, IN); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vqpn_groups(sel, order, np, gq, gm, &ng);
    __syncthreads();   /* 全核唯一一次全体屏障(码本 + 组表) */
    const uint32_t gw = blockIdx.x * NW + (threadIdx.x >> 5), nw = gridDim.x * NW;
    const uint32_t total = ng * MID, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        V41_VQ_SEG(u, uend, MID, g, r0, n);
        u += n;
        const uint32_t q = gq[g], nt = gm[g];
        const int32_t e = sel[order[q]];
        if (e < 0) continue;                                       /* 原核: 专家号无效 ⇒ 不写 h */
        const v41_vq_mat mg = v41_vq_open<1>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<1>(blob, e, 1, MID, IN, NULL);
        if (!mg.ok || !mu.ok) continue;                            /* 原核: 载荷不对 ⇒ 这些对不写 h */
        if (mg.cb != cb || mu.cb != cb) __trap();                  /* 格式违约: 不许拿别的码本算 */
        const uint32_t *xs[M]; uint32_t pr[M];
        #pragma unroll
        for (int j = 0; j < M; j++) {
            pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)];   /* 空位指向组头: 指针合法但不读不写 */
            xs[j] = x + (uint64_t)(pr[j] / K) * (IN / 2u);
        }
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT, 1, EXT>(mg, r0);
        float gs[M], us[M];
        v41_vq_stream<NBIT, EXT, M>(mg, r0, n, xs, nt, vqsh, &carry, v41_vq_row_ptr<NBIT, 1>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL, gs);
        v41_vq_stream<NBIT, EXT, M>(mu, r0, n, xs, nt, vqsh, &carry, NULL, NULL, us);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++)
                if ((uint32_t)j < nt) h[(uint64_t)pr[j] * MID + r0 + lane] = v41_vq_swiglu(v41_bf16r(gs[j]), v41_bf16r(us[j]), clamp);
        }
    }
}
template <int NBIT, int EXT, int M, int NW>
__global__ static void __launch_bounds__(NW * 32, 1) v41_vq_dn_persist_n_kernel(float *partial, const uint8_t *blob, const int32_t *sel,
        const int32_t *order, const uint32_t *h, uint32_t MID, uint32_t OUT, uint32_t np, uint32_t cbb, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[V41_VQPN_MAXP], gm[V41_VQPN_MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    const v41_vq_mat m0 = v41_vq_open<1>(blob, 0, 2, OUT, MID, NULL);
    const uint8_t *cb = m0.ok ? m0.cb : NULL;
    if (cb) v41_vq_cb_to_shared(vqsh, cb, cbb);
    v41_pdl_wait();
    if (!cb) { cb = v41_vq_layer_cb(blob, sel, np, 2, OUT, MID); if (!cb) return; v41_vq_cb_to_shared(vqsh, cb, cbb); }
    v41_vqpn_groups(sel, order, np, gq, gm, &ng);
    __syncthreads();
    const uint32_t gw = blockIdx.x * NW + (threadIdx.x >> 5), nw = gridDim.x * NW;
    const uint32_t total = ng * OUT, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        V41_VQ_SEG(u, uend, OUT, g, r0, n);
        u += n;
        const uint32_t q = gq[g], nt = gm[g];
        const int32_t e = sel[order[q]];
        if (e < 0) continue;                                       /* 原核: 专家号无效 ⇒ 不写 */
        uint32_t pr[M]; const uint32_t *hs[M];
        #pragma unroll
        for (int j = 0; j < M; j++) {
            pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)];
            hs[j] = h + (uint64_t)pr[j] * (MID / 2u);
        }
        const v41_vq_mat md = v41_vq_open<1>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
        if (!md.ok) {                                              /* 原核: 载荷不对 ⇒ 写 0(下游 tail 会读) */
            if (lane < n) for (uint32_t j = 0; j < nt; j++) partial[(uint64_t)pr[j] * OUT + r0 + lane] = 0.f;
            continue;
        }
        if (md.cb != cb) __trap();
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT, 1, EXT>(md, r0);
        float ys[M];
        v41_vq_stream<NBIT, EXT, M>(md, r0, n, hs, nt, vqsh, &carry, NULL, NULL, ys);
        if (lane < n) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < nt) partial[(uint64_t)pr[j] * OUT + r0 + lane] = v41_bf16r(ys[j]);
        }
    }
}
/* 一档(M, NW)的 opt-in + 发射。opt-in 按实例记(函数模板的 static 每实例一份; 两档码本 32/64 KB 并存, 09-22 的教训)。 */
template <int NBIT, int EXT, int M, int NW>
static int v41_vq_persist_n_go(int stage, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const int32_t *ord,
                               const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K, uint32_t np, float clamp,
                               uint32_t cbb, const float *gr, int nsm) {
    static uint32_t s_optin[2] = { 0u, 0u };
    if (s_optin[stage] < cbb) {
        const cudaError_t e = stage == 0
            ? cudaFuncSetAttribute(v41_vq_gu_persist_n_kernel<NBIT, EXT, M, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb)
            : cudaFuncSetAttribute(v41_vq_dn_persist_n_kernel<NBIT, EXT, M, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb);
        if (e != cudaSuccess) {
            fprintf(stderr, "ds4: [v41] VQ 验证批常驻核(M=%d)批不到 %u B 动态 shared: %s\n", M, cbb, cudaGetErrorString(e));
            (void)cudaGetLastError();
            return 0;
        }
        s_optin[stage] = cbb;
    }
    if (stage == 0) {
        v41_pdl_register((const void *)v41_vq_gu_persist_n_kernel<NBIT, EXT, M, NW>);   /* 碰 sel/order/x 之前 v41_pdl_wait */
        v41_vq_gu_persist_n_kernel<NBIT, EXT, M, NW><<<(unsigned)nsm, NW * 32, cbb, g_cur_stream>>>(h, blob, sel, ord, xb, IN, MID, K, np, clamp, cbb);
    } else {
        v41_pdl_register((const void *)v41_vq_dn_persist_n_kernel<NBIT, EXT, M, NW>);
        v41_vq_dn_persist_n_kernel<NBIT, EXT, M, NW><<<(unsigned)nsm, NW * 32, cbb, g_cur_stream>>>(part, blob, sel, ord, (const uint32_t *)h, MID, OUT, np, cbb, gr);
    }
    return cuda_ok(cudaGetLastError(), stage == 0 ? "v41 vq gateup persist(验证批)" : "v41 vq down persist(验证批)");
}
/* 返回 1 = 发了; 0 = 不适用或批不到 shared(调用方回逐对 + 分组核)。组大小恒 ≤ V41_VQPN_M, 与批大小无关 ⇒ 一个实例。 */
template <int NBIT, int EXT>
static int v41_vq_persist_n_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel,
                                   const int32_t *ord, const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K,
                                   uint32_t np, float clamp, uint32_t cbb, const float *gr) {
    static int s_nsm = 0;
    if (np > V41_VQPN_MAXP || n_tok < 2u) return 0;
    if (!s_nsm && cudaDeviceGetAttribute(&s_nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    return v41_vq_persist_n_go<NBIT, EXT, (int)V41_VQPN_M, 32>(stage, h, part, blob, sel, ord, xb, IN, MID, OUT, K, np, clamp, cbb, gr, s_nsm);
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
