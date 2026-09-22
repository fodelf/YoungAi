/* cuda_vq_group.inc.cu — VQ 专家的多 token 分组"解码即乘"核(2026-09-22, 投机验证批与草稿塔的小批专用)。
 * 聚合根按序 #include: 在 cuda_vq_decode.inc.cu 之后(用它的 v41_vq_cb_to_shared / v41_vq_swiglu 与 cuda_vq_row 那一族),
 * 在 cuda_vq_decode_launch.inc.cu 之前(v41_vq_fused_moe_n 在那里实例化, 到那时本片的 v41_vq_grp_launch 必须已可见)。
 *
 * 治的病(fable5 09-19 / 09-22 的账): 验证批 n 行走"一 block 一 (token, 专家) 对"的核, n=1 专家核 0.356 ms/层、n=4 要
 * 1.283 ms/层 = 3.6 倍 —— 权重字节一点没摊薄, 投机"验 k+1 位约等于验 1 位"的前提在那个形态上不成立。ncu 早把这个核的
 * 产能单位量清了: L1TEX 波前 —— 一轮(32 个索引)码本查表 ~11.7 个 + 激活 4 个 + 位流 ~1.5 个; 逐对路上 m 个 token 选中
 * 同一个专家, 码本查表就付 m 遍。这里一 block 管一个专家 × 它在本批被几个 token 选中(m ≤ M = 批大小): 位流读一遍、每个
 * 索引的码字只解一次, 对 m 个 token 各做一次 8 元素乘加 —— 一轮 11.7 + 4m 个波前, 对比逐对的 15.7m。
 *
 * ★同轨(投机 == 纯解码逐字节同)靠什么★: 每个 token 每一行的乘加式与单 token 核是**同一个函数**(cuda_vq_row.inc.cu 的
 * v41_vq_dot8_cw), 每 lane 的累加序、跨 lane 的归约树、行增益的乘序一个字没动; 变的只是"哪个 block 在什么时候算哪一对",
 * 与 09-19 的 SORTED 实例同一类改动。门 = speed-bench/d1_kv_ring_gate.sh 的 spec 格(cmp)。
 * 09-16 那版分组核判负(m=2 慢 1.7% / m=4 慢 7.1%)是在 f32 激活(每 token 每轮 8 个波前)+ 48 B 位流读的旧形态下量的,
 * 那时 4 个 token 的激活 80 KB 装不进码本占掉一半之后的 L1; 现在激活 bf16(10 KB/token)、位流整块读, 账变了 ——
 * 09-19 陪审团把它排在"每行 ×0.65"那一档。A/B 开关: --no-vq-group(同一二进制回逐对核)。
 *
 * 出错会怎样: 分组表(order 按专家排序)错一位 = 某个 token 的输出写进别人的 pair 槽 —— 不报错, 只有温 0 逐字节门抓得到。 */

/* 一 block 16 warp = 512 线程 ⇒ 每线程最多 128 个寄存器: M 个累加器 + M 条激活 load 才放得下(逐对核 1024 线程只许 64 个,
 * 已顶到 56~64)。寄存器 ≤ 64 时每 SM 还能挂两个这样的 block(shared 2×64 KB / 线程 1024 都在 GB10 的额度内), 与逐对核同占用。
 * 每 warp 循环几行: gateup 2(一 block 32 行), down 4(64 行) —— 逐对核是 2/4 × 32 warp, 这里 block 少一半、每 block 行数少一半,
 * 靠 grid 多 wave 保负载均衡(09-18 实测"这个核靠多 wave 活着")。 */
#define V41_VQG_WARPS    16u
#define V41_VQG_THREADS  (V41_VQG_WARPS * 32u)
#define V41_VQG_ITERS_GU 2u
#define V41_VQG_ITERS_DN 4u

/* 一块的最多 8 轮, M 个 token 版: 索引与码字只解一次, 对每个 token 各做一次 v41_vq_dot8_cw。
 * xs[j] = 第 j 个 token 的激活行(bf16 打包), boff = 本块在行内的字偏移(32 索引 × 4 字 一轮); m 是 block 一致的, 分支不发散。 */
template <int NBIT, int V3, int EXT, int M>
__device__ __forceinline__ static void v41_vqg_blk_rounds(const v41_vq_blk &cur, uint32_t rounds, uint32_t a, uint32_t sh, uint32_t imsk,
                                                          const uint32_t *const *xs, uint32_t boff, uint32_t m, const uint8_t *cbs, float *acc) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint32_t f = k * MB;
        const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
        uint32_t v = __funnelshift_r(lo, hi, sh) & (EXT ? 0xFFFu : imsk);
        if (EXT) {
            const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k);
            v |= ((exw >> lane) & 1u) << 12;
        }
        float c[8];
        v41_vq_cw<V3>(v, cbs, 1, c);                    /* 码字只解一次(码本恒在 shared) */
        const uint32_t xo = boff + (k * 32u + lane) * 4u;
        #pragma unroll
        for (int j = 0; j < M; j++) {
            if ((uint32_t)j < m) {
                const uint4 xa = *(const uint4 *)(xs[j] + xo);   /* 8 个 bf16 = 16 B, L1 命中 */
                acc[j] += v41_vq_dot8_cw(c, xa);              /* 与单 token 核同一个乘加式 */
            }
        }
    }
}

/* 一 warp 算一行对 m 个 token 的点积(跨块流水与 v41_vq_row_dot 同套: carry 进来是本行首块, 出去是 next 行的首块)。
 * out[j] = 增益 × Σ(已 warp 规约), 乘序与单 token 核一字不差(acc × 行增益 × gov, 不预乘)。 */
template <int NBIT, int V3, int EXT, int M>
__device__ __forceinline__ static void v41_vqg_row_dot(const v41_vq_mat &mt, uint32_t r, const uint32_t *const *xs, uint32_t m, const uint8_t *cbs,
                                                       v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex, float *out) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t R = mt.nidx_row >> 5;
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;
    const uint32_t *row = v41_vq_row_ptr<NBIT, V3>(mt, r);
    const uint32_t *rex = EXT ? v41_vq_ext_ptr(mt, r) : NULL;
    v41_vq_blk cur = *carry;
    float acc[M];
    #pragma unroll
    for (int j = 0; j < M; j++) acc[j] = 0.f;
    for (uint32_t b = 0; b < R; b += 8u) {
        v41_vq_blk nxt;
        if (b + 8u < R) {
            const uint32_t rem = R - b - 8u, nr = (rem < 8u ? rem : 8u);
            nxt = v41_vq_blk_load<EXT>(row + (size_t)(b + 8u) * MB, nr * MB, EXT ? rex + (b + 8u) : NULL, nr);   /* 位平面按组推进(09-22 修) */
        }
        else if (next) { const uint32_t nr = (R < 8u ? R : 8u); nxt = v41_vq_blk_load<EXT>(next, nr * MB, nextex, nr); }
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; nxt.ex = 0u; }
        v41_vqg_blk_rounds<NBIT, V3, EXT, M>(cur, (R - b < 8u) ? R - b : 8u, a, sh, mt.imsk, xs, b * 32u * 4u, m, cbs, acc);
        cur = nxt;
    }
    *carry = cur;
    __half gh; memcpy(&gh, mt.gr + (size_t)r * 2u, 2);
    #pragma unroll
    for (int j = 0; j < M; j++) {
        if ((uint32_t)j < m) {
            float v = acc[j];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            out[j] = v * __half2float(gh) * (mt.gov ? mt.gov[r] : 1.0f);
        }
    }
}

/* 本 block 管哪个专家、哪几个 token: order[] 按 (专家, 对号) 升序(cuda_vq_decode 的 v41_vq_order_kernel), 同一专家的对相邻;
 * 只有那一段的头 block 干活(其余 block 立刻退, 整 block 一致 ⇒ 不影响后面的 __syncthreads), 段长 m(一个 token 选同一专家至多一次)。
 * ★混合派发(2026-09-22 第二版)★: 首版让所有组都走分组核, k=3 验证批实测 94.0 ms 对逐对核 87.5 —— 多数组只有 1 个 token
 * (n=4 时 24 对里约 17 个专家, 12 个只被选一次), 它们在 16 warp / 高寄存器的瘦形态里比逐对核的 32 warp 慢, 把 m≥2 那几组省下的钱
 * 又吐了回去。所以: m=1 的组照旧走逐对核(它的 SORTED 实例跳过 m≥2 的对), 分组核只收 m ∈ [lo, M] 的组; M=2/4/6 三档实例,
 * 寄存器随 M 涨(M=2/4 各 64 个, 每 SM 两个 block; M=6 顶到 128), 组大的走大档, 不让小组陪着大档吃低占用率。
 * 空位 j ≥ m 指向头对: 指针合法但从不读写(全部读写都在 j < m 之内)。返回 m; 0 = 本 block 不是段头或段长不在本实例的档里。 */
__device__ __forceinline__ static uint32_t v41_vqg_group(const int32_t *sel, const int32_t *order, uint32_t np, uint32_t q, int M, uint32_t lo,
                                                         int32_t *e_out, uint32_t *pairs) {
    const uint32_t p0 = (uint32_t)order[q];
    const int32_t e = sel[p0];
    *e_out = e;
    if (e < 0) return 0u;
    if (q > 0u && sel[order[q - 1u]] == e) return 0u;
    uint32_t m = 1u;
    while (q + m < np && sel[order[q + m]] == e) m++;
    if (m < lo || m > (uint32_t)M) return 0u;
    for (int j = 0; j < M; j++) pairs[j] = ((uint32_t)j < m) ? (uint32_t)order[q + (uint32_t)j] : p0;
    return m;
}

/* gate/up 同核(与 v41_vq_gateup_kernel 同式: w1/w3 出 bf16 → silu(g)·u → bf16), 一 block 一个专家 × m 个 token。
 * shared 一块用两遍(先 gate 码本后 up 码本), 与逐对核同一套做法; gate 的结果以 bf16 暂存在各 token 自己的 h 行里。 */
template <int NBIT, int V3, int EXT, int M>
__global__ static void __launch_bounds__(V41_VQG_THREADS)
v41_vqg_gateup_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                      uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes, const int32_t *order, uint32_t np, uint32_t mlo) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    constexpr uint32_t W = V41_VQG_WARPS, NIT = V41_VQG_ITERS_GU;
    int32_t e; uint32_t pairs[M];
    const uint32_t m = v41_vqg_group(sel, order, np, blockIdx.y, M, mlo, &e, pairs);
    if (!m) return;
    const uint32_t *xs[M]; uint16_t *hp[M];
    #pragma unroll
    for (int j = 0; j < M; j++) { xs[j] = x + (uint64_t)(pairs[j] / K) * (IN / 2u); hp[j] = h + (uint64_t)pairs[j] * MID; }
    const v41_vq_mat mg = v41_vq_open<V3>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<V3>(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    constexpr uint32_t CW = V3 ? 8u : 16u;
    if (mg.nc * CW != cb_bytes || mu.nc * CW != cb_bytes) return;
    const uint32_t r0 = blockIdx.x * (W * NIT) + (threadIdx.x >> 5);
    const bool lead = (threadIdx.x & 31u) == 0u;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (r0 < MID) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(mg, r0);   /* 首块先发, 藏在码本搬运后面 */
    v41_vq_cb_to_shared(vqsh, mg.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < NIT; i++) {
        const uint32_t r = r0 + i * W;
        if (r >= MID) break;
        const uint32_t rn = r + W;
        const int own = (i + 1u < NIT && rn < MID);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mg, rn) : v41_vq_row_ptr<NBIT, V3>(mu, r0);
        const uint32_t *nextex = EXT ? (own ? v41_vq_ext_ptr(mg, rn) : v41_vq_ext_ptr(mu, r0)) : NULL;
        float g[M];
        v41_vqg_row_dot<NBIT, V3, EXT, M>(mg, r, xs, m, vqsh, &carry, next, nextex, g);
        if (lead) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < m) hp[j][r] = (uint16_t)(__float_as_uint(v41_bf16r(g[j])) >> 16);
        }
    }
    __syncthreads();                            /* 等所有 warp 读完 gate 码本, 才能覆盖它 */
    v41_vq_cb_to_shared(vqsh, mu.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < NIT; i++) {
        const uint32_t r = r0 + i * W;
        if (r >= MID) break;
        const uint32_t rn = r + W;
        const int own = (i + 1u < NIT && rn < MID);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mu, rn) : NULL;
        float u[M];
        v41_vqg_row_dot<NBIT, V3, EXT, M>(mu, r, xs, m, vqsh, &carry, next, EXT && own ? v41_vq_ext_ptr(mu, rn) : NULL, u);
        if (lead) {
            #pragma unroll
            for (int j = 0; j < M; j++)
                if ((uint32_t)j < m) hp[j][r] = v41_vq_swiglu(__uint_as_float((uint32_t)hp[j][r] << 16), v41_bf16r(u[j]), clamp);
        }
    }
}

/* down: partial[pair][OUT] = bf16(W2·h_pair), 一 block 一个专家 × m 个 token */
template <int NBIT, int V3, int EXT, int M>
__global__ static void __launch_bounds__(V41_VQG_THREADS)
v41_vqg_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *hh,
                    uint32_t MID, uint32_t OUT, uint32_t K, uint32_t cb_bytes, const float *gr, const int32_t *order, uint32_t np, uint32_t mlo) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    constexpr uint32_t W = V41_VQG_WARPS, NIT = V41_VQG_ITERS_DN;
    int32_t e; uint32_t pairs[M];
    const uint32_t m = v41_vqg_group(sel, order, np, blockIdx.y, M, mlo, &e, pairs);
    if (!m) return;
    const uint32_t *hs[M];
    #pragma unroll
    for (int j = 0; j < M; j++) hs[j] = hh + (uint64_t)pairs[j] * (MID / 2u);
    const v41_vq_mat md = v41_vq_open<V3>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * (W * NIT) + (threadIdx.x >> 5);
    const bool lead = (threadIdx.x & 31u) == 0u;
    if (!md.ok) {   /* 载荷不对: 这 block 负责的行全写 0(下游 reduce 会读) */
        for (uint32_t i = 0; i < NIT; i++) {
            const uint32_t r = r0 + i * W;
            if (r < OUT && lead) for (int j = 0; j < M; j++) if ((uint32_t)j < m) partial[(uint64_t)pairs[j] * OUT + r] = 0.f;
        }
        return;
    }
    if (md.nc * (V3 ? 8u : 16u) != cb_bytes) return;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (r0 < OUT) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(md, r0);
    v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < NIT; i++) {
        const uint32_t r = r0 + i * W;
        if (r >= OUT) break;
        const uint32_t rn = r + W;
        const int own = (i + 1u < NIT && rn < OUT);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(md, rn) : NULL;
        float y[M];
        v41_vqg_row_dot<NBIT, V3, EXT, M>(md, r, hs, m, vqsh, &carry, next, EXT && own ? v41_vq_ext_ptr(md, rn) : NULL, y);
        if (lead) {
            #pragma unroll
            for (int j = 0; j < M; j++) if ((uint32_t)j < m) partial[(uint64_t)pairs[j] * OUT + r] = v41_bf16r(y[j]);
        }
    }
    (void)K;
}

/* 一个 M 档的 opt-in(动态 shared 按实例记: 函数模板的 static 局部变量每实例一份, 与逐对核那条 09-22 的教训同一条规矩)。
 * 返回 1 = 已批到 cbb; 0 = 批不上去(整个分组路回逐对核)。 */
template <int NBIT, int V3, int EXT, int M>
static int v41_vqg_optin(uint32_t cbb) {
    static uint32_t s_optin = 0u;
    if (s_optin >= cbb) return 1;
    const bool ok = cudaFuncSetAttribute(v41_vqg_gateup_kernel<NBIT, V3, EXT, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess &&
                    cudaFuncSetAttribute(v41_vqg_down_kernel<NBIT, V3, EXT, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
    (void)cudaGetLastError();
    if (!ok) return 0;
    s_optin = cbb;
    cudaFuncAttributes fg, fd; memset(&fg, 0, sizeof fg); memset(&fd, 0, sizeof fd);
    (void)cudaFuncGetAttributes(&fg, v41_vqg_gateup_kernel<NBIT, V3, EXT, M>);
    (void)cudaFuncGetAttributes(&fd, v41_vqg_down_kernel<NBIT, V3, EXT, M>);
    int bg = 0, bd = 0;
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bg, v41_vqg_gateup_kernel<NBIT, V3, EXT, M>, (int)V41_VQG_THREADS, (size_t)cbb);
    (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bd, v41_vqg_down_kernel<NBIT, V3, EXT, M>, (int)V41_VQG_THREADS, (size_t)cbb);
    (void)cudaGetLastError();
    fprintf(stderr, "ds4: [v41] VQ 分组核 M=%d(NBIT %d): %u 线程/block, 动态 shared %u KB; gateup %d 寄存器 ⇒ 每 SM %d block, down %d 寄存器 ⇒ %d block\n",
            M, NBIT, V41_VQG_THREADS, cbb >> 10, fg.numRegs, bg, fd.numRegs, bd);
    return 1;
}
/* 发一个 M 档的 gateup 或 down(收 m ∈ [mlo, M] 的组); grid.y = np, 不在档里的 block 一进来就退 */
template <int NBIT, int V3, int EXT, int M>
static int v41_vqg_launch_gu(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *xb, uint32_t IN, uint32_t MID, uint32_t K,
                             float clamp, uint32_t cbb, const int32_t *ord, uint32_t np, uint32_t mlo) {
    const uint32_t rg = V41_VQG_WARPS * V41_VQG_ITERS_GU;
    v41_vqg_gateup_kernel<NBIT, V3, EXT, M><<<dim3((MID + rg - 1u) / rg, np), V41_VQG_THREADS, cbb, g_cur_stream>>>(
        h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, mlo);
    return cuda_ok(cudaGetLastError(), "v41 vq gateup(分组)");
}
template <int NBIT, int V3, int EXT, int M>
static int v41_vqg_launch_dn(float *part, const uint8_t *blob, const int32_t *sel, const uint16_t *h, uint32_t MID, uint32_t OUT, uint32_t K,
                             uint32_t cbb, const float *gr, const int32_t *ord, uint32_t np, uint32_t mlo) {
    const uint32_t rd = V41_VQG_WARPS * V41_VQG_ITERS_DN;
    v41_vqg_down_kernel<NBIT, V3, EXT, M><<<dim3((OUT + rd - 1u) / rd, np), V41_VQG_THREADS, cbb, g_cur_stream>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, cbb, gr, ord, np, mlo);
    return cuda_ok(cudaGetLastError(), "v41 vq down(分组)");
}
/* ★分组路的两道(先 gateup 后 down)★, 由 cuda_vq_decode.inc.cu 的 v41_vq_fused_moe_n 调: 逐对核(SORTED 实例, 只算 m=1 的对)
 * 与分组核(m≥2 的组按 m 分 M=2 / 4 / 6 三档)各发各的 block, 写的是不相交的 pair 槽。批大小 n_tok 定要发哪几档(m ≤ n_tok):
 * n ≤ 2 只有 M=2; 3~4 再加 M=4(收 3~4); 5~6 再加 M=6(收 5~6)。
 * 返回 1 = 发完; 0 = 某档批不到 shared(整条分组路不用, 调用方回全逐对); -1 = 启动失败(硬错)。 */
template <int NBIT, int V3, int EXT>
static int v41_vq_grp_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                             uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K, float clamp, uint32_t cbb, const float *gr,
                             const int32_t *ord, uint32_t np) {
    if (n_tok < 2u || n_tok > 6u) return 0;
    if (!v41_vqg_optin<NBIT, V3, EXT, 2>(cbb)) return 0;
    if (n_tok >= 3u && !v41_vqg_optin<NBIT, V3, EXT, 4>(cbb)) return 0;
    if (n_tok >= 5u && !v41_vqg_optin<NBIT, V3, EXT, 6>(cbb)) return 0;
    if (stage == 0) {
        if (!v41_vqg_launch_gu<NBIT, V3, EXT, 2>(h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, 2u)) return -1;
        if (n_tok >= 3u && !v41_vqg_launch_gu<NBIT, V3, EXT, 4>(h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, 3u)) return -1;
        if (n_tok >= 5u && !v41_vqg_launch_gu<NBIT, V3, EXT, 6>(h, blob, sel, xb, IN, MID, K, clamp, cbb, ord, np, 5u)) return -1;
        return 1;
    }
    if (!v41_vqg_launch_dn<NBIT, V3, EXT, 2>(part, blob, sel, h, MID, OUT, K, cbb, gr, ord, np, 2u)) return -1;
    if (n_tok >= 3u && !v41_vqg_launch_dn<NBIT, V3, EXT, 4>(part, blob, sel, h, MID, OUT, K, cbb, gr, ord, np, 3u)) return -1;
    if (n_tok >= 5u && !v41_vqg_launch_dn<NBIT, V3, EXT, 6>(part, blob, sel, h, MID, OUT, K, cbb, gr, ord, np, 5u)) return -1;
    return 1;
}
