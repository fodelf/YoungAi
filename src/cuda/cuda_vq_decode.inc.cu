/* cuda_vq_decode.inc.cu — VQ 专家的"解码即乘"核(2026-09-16 从 cuda_v41_4.inc.cu 拆出)。
 *
 * 为什么单独一片: 原来它和骨架 fp4x32 GEMV 挤在一个文件里, 两条家族的演进史互不相干,
 * 加几条判负存档就破了 500 行守卫。拆开之后这一片只讲一件事: **压缩态专家权重怎么直接参与乘法**。
 *
 * 盘上形态(DQVL v2 blob): 16 B 头 + 384×3 个槽表; 每槽是一份 DQVQ 载荷 =
 * 16 B 头 + 码本(nc × 8 个 f16; nc4096 = 64 KB) + 逐行增益 f16 + 位流(每 8 个元素一个码本号)。
 * ★索引位宽由 nc 定, 现役两档★: nc4096 = 12 位(一行 640 索引 = 960 B), nc2048 = 11 位(880 B,
 * 2026-09-19 的 100 GB 配方)。下面的 v41_vq_row_dot 整族是 NBIT 模板, 几何天然通用 ——
 * 一块恒 8 轮 = 8×NBIT 个字(12 位 96 字 = 3 条整线; 11 位 88 字 = 2 条整线 + 24 lane 的 96 B),
 * 每 lane 恒 3 个寄存器(没触到 09-18 那个 64 寄存器硬顶), 行起点恒整字节(nidx_row 640/288 是 8 的倍数)。
 * ★11 位的第三条读不满整线 ⇒ 按 09-18 请求粒度账掉到 ~142 GB/s 档, 而字节少 8.3%; 净亏净赚未用 12k 尺量。★
 *
 * 09-16 体检与第一刀: 这两个核原本合计 14.23 ms/步, 离板子 240 GB/s 的墙很远。ncu 指出真凶不是权重 ——
 * 一次发射(一层 gateup)全局 load 扇区 42.0 M × 32 B = 1.34 GB, 而这一层的权重才 31.5 MB, **放大 43 倍**,
 * 那些扇区几乎全是**激活重读**。改激活存 bf16 后 **14.23 → 13.28 ms/步**, 且逐位同(见 v41_vq_dot8)。
 * 09-17/18 第二刀: 位流按 384 B 整块(3 条整线)读进寄存器再 shfl 分发(见 v41_vq_row_dot)。此前位流读法三次判负,
 * 真因不是"读法不重要", 是那三版一轮都只请求 48 B —— DRAM 按 64 B 取, 一半白取(mem_ceiling ⑥ 量死的)。
 *
 * ★必须排在 cuda_v41_4.inc.cu 之后★: 用它的 v41_bf16r / v41_grow / v41_scratch;
 * 又必须排在 cuda_v41_draft.inc.cu 之前(草稿塔借本片的 reduce 核与暂存槽)。 */
/* 载荷解析 + 一行点积(v41_vq_open / v41_vq_dot8 / v41_vq_row_dot 整族)住 cuda_vq_row.inc.cu —— 拆出去为守 500 行,
 * 也因为 v3 布局(层码本 + E4M3 + 13 位位平面)只动那一族。聚合根按序 include: 本片必须在它之后。 */

/* ★2026-09-15 single.md S2 判负存档: "码本条目改 20 B 跨距消 bank conflict"★
 * 假设: 条目 16 B = 4 个 bank, 32 个 lane 拿随机码本号 v, 地址 v*16 落在 bank 组 (v*4)%32 只有 8 个值
 * ⇒ 平均 4 路冲突。改 20 B(5 bank, gcd(5,32)=1)让它铺满 32 个 bank。
 * 实测 **58.7 → 59.9 ms(持平偏慢)**, 回退。两个原因: ①20 B 不是 8 的倍数, uint2 读直接
 * misaligned 崩(CUDA flush failed: misaligned address), 只能退成 4 次 uint 读, 多出来的指令
 * 吃掉了省下的冲突; ②8 的倍数的跨距做不到 bank 全覆盖(stride/4 必为偶数) —— 对齐与全覆盖互斥。
 * ⇒ 这个核的瓶颈也不是 shared bank。 */
/* 码本搬进 shared: 8 条 load 一起发再一起存(2026-09-18)。原来一条读一条存, 每条都等一次 L2(143 ns), 64 KB 要
 * 等 8 次; 这段是全 block 同步段, SM 里没别的活能盖住它。码本只保证 8 B 对齐(载荷偏移交替 8/16 对齐), 只能 uint2。 */
__device__ __forceinline__ static void v41_vq_cb_to_shared(uint8_t *dst, const uint8_t *src, uint32_t bytes) {   /* bytes 是 8 的倍数 */
    const uint2 *s = (const uint2 *)src; uint2 *d = (uint2 *)dst;
    const uint32_t n = bytes / 8u;
    uint32_t i = threadIdx.x;
    for (; i + 7u * blockDim.x < n; i += 8u * blockDim.x) {
        uint2 t[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) t[q] = s[i + (uint32_t)q * blockDim.x];
        #pragma unroll
        for (int q = 0; q < 8; q++) d[i + (uint32_t)q * blockDim.x] = t[q];
    }
    for (; i < n; i += blockDim.x) d[i] = s[i];
}
/* SwiGLU 出口, 与官方 Expert 同式: clamp → silu(g)·u → bf16。g/u 都已在 bf16 格点上。 */
__device__ __forceinline__ static uint16_t v41_vq_swiglu(float gi, float ui, float clamp) {
    if (clamp > 0.f) { if (gi > clamp) gi = clamp; if (ui > clamp) ui = clamp; if (ui < -clamp) ui = -clamp; }
    const float sg = gi / (1.0f + expf(-gi));
    return (uint16_t)(__float_as_uint(v41_bf16r(sg * ui)) >> 16);
}
/* ★shared 码本版的一 block 行数★(2026-09-14 二改)。
 * 一 block 32 warp, 每 warp 再循环 V41_VQ_ITERS 行 ⇒ 一 block 管 32×ITERS 行, 码本只搬一次。
 * 为什么必须循环而不是一 block 32 行: 码本 64 KB 占满设备每 block 动态 shared 上限(GB10 = 99 KB)的一大半,
 * 每个 SM 只塞得下 1 个 block; 一 block 只算 32 行就重搬一次 64 KB, 而这 32 行的索引流才 30 KB ——
 * 搬运是有效数据的 2 倍多, 40 层累计好几 GB。行数翻 8 倍, 码本搬运就摊薄 8 倍, 有效带宽利用直接上去。 */
/* ★2026-09-15 段 2 下调 8 → 2★: 上面那笔"摊薄码本搬运"的账只算了 L2 流量, 漏了**尾巴**。
 * 码本 64 KB ⇒ 每个 SM 只驻 1 个 block, 而 ITERS=8 时 grid = (2304/256, 6 专家) = **54 个 block**
 * 撒在 48 个 SM 上 = ncu 实测 1.12 waves/SM: 第二个 wave 只有 6 个 block 在跑、42 个 SM 干等,
 * 这个核一个人吃掉解码每步 21 ms。ITERS=2 ⇒ 一 block 64 行 ⇒ grid 216 = 4.5 waves, 尾巴损失从
 * 约一半降到约一成。多出来的码本搬运是 L2 命中(6 个专家的码本合计 384 KB, 远小于 24 MB L2),
 * 每层多几微秒, 换回来的是几十微秒。★口径没变, 数值逐位同 —— 只是行怎么分给 block。★
 * (09-15 长尾修掉后重扫 ITERS: 2 = 58.7 ms < 4 = 59.8 < 8 = 63.0, 维持 2。)
 * ★2026-09-18 两个核分开定★(位流整块读 + 跨块流水之后重扫, 12k 尺, ms/步):
 *   gateup: 2 行 = 6.72 < 4 行 = 7.19 < 8 行 = 8.77 —— **block 越少越慢, 单调**。8 行时 grid = 2048/256 × 6 = 48 block
 *   = 正好 1 wave, 账面上码本搬运少 4 倍、没尾巴, 实测最慢: 1 wave 没有任何负载均衡, 整发等最慢的那个 SM;
 *   4 wave(2 行)时块调度器一直在给先干完的 SM 派活。⇒ 这个核靠"多 wave"活着, 维持 2 行。
 *   down: 4 行(240 block = 5 wave)= 3.76 < 2 行 4.51(那次连同流水一起量, 未单独拆)。
 * gate 的结果不再占寄存器数组, 先以 bf16 暂存进 h 本行的位置(存取无损), 行数改动不再碰寄存器上限。 */
#define V41_VQ_ITERS_GU 2u
#define V41_VQ_ITERS_DN 4u
/* 每 block 几个 warp(shared 码本版)。32 = 1024 线程, 寄存器上限 64/线程; 16 = 512 线程, 上限 128, 但 1 block/SM 时占用率减半。 */
#define V41_VQ_WARPS    32u
#define V41_VQ_GU_ROWS  (V41_VQ_WARPS * V41_VQ_ITERS_GU)
#define V41_VQ_DN_ROWS  (V41_VQ_WARPS * V41_VQ_ITERS_DN)

/* ★★2026-09-16 判负存档: 多 token 小批"按专家去重"(mtp-1.md M4′)★★
 * 依据看着最硬的一版, 也是整条投机路上账面最大的一笔: 验证批 4 个 token × top-6 = 24 个 (token,专家) 对里,
 * 唯一专家只有 14 个左右 —— 四成的专家权重是重复读的; 而专家核占投机一轮(115 ms)的 53%。
 * 写了一份去重核(一 block 管同一专家的 2 个对, 位流读一遍、码本查一遍、点积做两次; 逐位同, 只换"哪个
 * block 算哪一对"), 算出来: 码本搬运 −42%、位流 −41%, 只有激活多 17%(单成员组白算的那一份)。
 * **实测: 专家核一轮 61.5 → 60.5 ms, 1.6%。** 同一把尺, 同机器状态。
 * ★这一步把话讲死了★: 位流字节砍四成没用, 码本搬运砍四成也没用 ⇒ 这个核的成本**既不是权重字节、
 * 也不是码本搬运, 而是每一个 (索引, token) 对上那点活本身**(一次 16 B 随机查表 + 8 个元素的乘加 +
 * 一条 16 B 激活读)。去重省的是"每个索引查几次表", 省不掉"每个 token 都得跟这一行乘一遍"。
 * ⇒ 想让验证 k 位便宜过验证 1 位, 只剩两条路: ①每个 token 读更少的权重(换量化格式, 那是质量线的决定);
 * ②把这个点积搬上张量核。**别再从"少读点字节"这个方向来了** —— 连同 09-16 那三次位流读法的判负,
 * 这是第四次同源。 */
/* gate/up 同核(官方 Expert: w1/w3 出 bf16 → f32 截断 → silu(g)·u → bf16); cb_bytes>0: 码本进 shared(grid.x 按 32 行),
 * 否则全局 gather(grid.x 按 8 行)。x 已是 bf16 格点(调用方 rms_norm 出口舍过)。 */
/* ★这个核一个字都别乱动★(2026-09-16 实撞): 1024 线程 × **64 个寄存器** × 64 KB 码本, 三样正好把一个 SM
 * 吃满(启动日志自报"每 SM 挂 1 个 block, 占用率 67%")。只是把 `pair = blockIdx.y` 改成
 * `order ? order[blockIdx.y] : blockIdx.y`(多一个"pair 可能来自全局内存"的可能性), **连 n=1 这条路都慢了 21%**
 * —— gateup 8.20 → 9.95 ms/步、down 4.88 → 5.92, 整步 47.2 → 50.1 ms, 输出却逐字节没变。
 * 多 token 的去重版另起一个核, 住 cuda_vq_union.inc.cu。 */
/* ★寄存器是这个核的硬墙★: 1024 线程/block ⇒ 每线程最多 64 个。没写 __launch_bounds__ —— 写了 ptxas 会把 64 填满,
 * 同一份代码 gateup 6.72 → 7.11 ms/步; 让它自然落在 56/48 更快。改动后看 cuobjdump --dump-resource-usage: REG 超 64 就装不下 SM,
 * launch 直接报 too many resources(第四刀初版 72 个就撞过)。 */
/* ★验证批(n≥2)按专家排序发 block(2026-09-19)★: 一个 block 仍管一个 (token, 专家) 对、内层一个字不动(每对累加序与 n=1 逐位同 ⇒ 同轨),
 * 只改"第几个 block 算哪一对": order[] 把同一专家的对排到相邻的 blockIdx.y。块按 blockIdx 顺序派发, 原来同一专家的两个对隔着 ~36 MB 别的
 * 专家流量, 第二次读早被 24 MB 的 L2 挤掉 ⇒ n 个 token 读 n 份 DRAM; 相邻则第二个对全是 L2 命中, DRAM 字节降到唯一专家份(09-17 实测
 * n=1..5 唯一专家 6/9.96/13.26/17.38/21.71 ⇒ n=3 省 26%)。SORTED 是模板参数: 09-16 实撞, 同一实例里加运行期 `order ? … : …` 让 n=1 慢 21%。 */
/* np / uniq_only(2026-09-22, 只在 SORTED=1 实例里用): uniq_only≠0 时只算"本批里只被一个 token 选中"的专家那些对 —— 同一专家被
 * 多个 token 选中的那几段交给分组核(cuda_vq_group.inc.cu)。判据从 order 的左右邻居读(同专家的对在 order 里相邻)。SORTED=0(n=1)
 * 那支这两个参数编译期就死了, 代码一个字不变。 */
template <int NBIT, int V3, int EXT, int SORTED>
__global__ static void v41_vq_gateup_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                                            uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes, const int32_t *order,
                                            uint32_t np, uint32_t uniq_only) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = SORTED ? (uint32_t)order[blockIdx.y] : blockIdx.y, t = pair / K, rows = cb_bytes ? V41_VQ_GU_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    if (SORTED && uniq_only &&
        ((blockIdx.y > 0u && sel[order[blockIdx.y - 1u]] == e) || (blockIdx.y + 1u < np && sel[order[blockIdx.y + 1u]] == e))) return;
    const v41_vq_mat mg = v41_vq_open<V3>(blob, e, 0, MID, IN, NULL), mu = v41_vq_open<V3>(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS_GU : 1u;
    const uint32_t *xs = x + (uint64_t)t * (IN / 2u);   /* 两个 bf16 一个字 */
    uint16_t *hp = h + (uint64_t)pair * MID;
    const bool lead = (threadIdx.x & 31u) == 0u;
    if (cb_bytes) {
        /* ★一块 shared 用两遍★(2026-09-14): gate 与 up 各有一本 64 KB 码本, 一次性放两本要 128 KB,
         * 超过 GB10 每 block 的上限(99 KB) ⇒ 整个核被打回全局 gather。改成先载 gate 算完这 block 的
         * 全部行, 同步后把同一块 shared 覆盖成 up 的码本再算 u: 峰值只要一本的量。
         * ★循环里不能 return★: 后面还有 __syncthreads, 少一个 warp 就死锁, 越界的行只跳过计算。
         * gate 的结果以 bf16 暂存在 h 本行的位置(它本来就在 bf16 格点上, 存取无损), 同一个 lane 写、同一个 lane 读回。 */
        constexpr uint32_t CW = V3 ? 8u : 16u;   /* 一个码字的字节数: v3 是 8 个 E4M3, v2 是 8 个 f16 */
        if (mg.nc * CW != cb_bytes || mu.nc * CW != cb_bytes) return;
        v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
        if (r0 < MID) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(mg, r0);   /* 首块先发, 藏在码本搬运后面 */
        v41_vq_cb_to_shared(vqsh, mg.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r = r0 + i * V41_VQ_WARPS;
            if (r >= MID) break;
            const uint32_t rn = r + V41_VQ_WARPS;   /* gate 末行时预装 up 的首行(它在码本换本之后才算) */
            const int own = (i + 1u < nit && rn < MID);   /* 下一块是本矩阵的下一行, 还是换本码本之后 up 的首行 */
            const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mg, rn) : v41_vq_row_ptr<NBIT, V3>(mu, r0);
            const uint32_t *nextex = EXT ? (own ? v41_vq_ext_ptr(mg, rn) : v41_vq_ext_ptr(mu, r0)) : NULL;
            const float gv = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mg, r, xs, vqsh, 1, &carry, next, nextex));
            if (lead) hp[r] = (uint16_t)(__float_as_uint(gv) >> 16);
        }
        __syncthreads();                            /* 等所有 warp 读完 gate 码本, 才能覆盖它 */
        v41_vq_cb_to_shared(vqsh, mu.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r = r0 + i * V41_VQ_WARPS;
            if (r >= MID) break;
            const uint32_t rn = r + V41_VQ_WARPS;
            const int own = (i + 1u < nit && rn < MID);
            const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(mu, rn) : NULL;
            const float ui = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mu, r, xs, vqsh, 1, &carry, next, EXT && own ? v41_vq_ext_ptr(mu, rn) : NULL));
            if (lead) hp[r] = v41_vq_swiglu(__uint_as_float((uint32_t)hp[r] << 16), ui, clamp);
        }
    } else if (r0 < MID) {
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT, V3, EXT>(mg, r0);
        const float gv = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mg, r0, xs, mg.cb, 0, &carry,
                                   v41_vq_row_ptr<NBIT, V3>(mu, r0), EXT ? v41_vq_ext_ptr(mu, r0) : NULL));
        const float ui = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(mu, r0, xs, mu.cb, 0, &carry, NULL, NULL));
        if (lead) hp[r0] = v41_vq_swiglu(gv, ui, clamp);
    }
}
/* down: partial[pair][OUT] = bf16(W2·h) */
template <int NBIT, int V3, int EXT, int SORTED>
__global__ static void v41_vq_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *h,
                                          uint32_t MID, uint32_t OUT, uint32_t K, uint32_t cb_bytes, const float *gr, const int32_t *order,
                                          uint32_t np, uint32_t uniq_only) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = SORTED ? (uint32_t)order[blockIdx.y] : blockIdx.y, rows = cb_bytes ? V41_VQ_DN_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    if (SORTED && uniq_only &&
        ((blockIdx.y > 0u && sel[order[blockIdx.y - 1u]] == e) || (blockIdx.y + 1u < np && sel[order[blockIdx.y + 1u]] == e))) return;
    const v41_vq_mat md = v41_vq_open<V3>(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS_DN : 1u;
    if (!md.ok) {   /* 载荷不对: 这 block 负责的行全写 0(不能只写一行, 下游 reduce 会读到脏值) */
        for (uint32_t i = 0; i < nit; i++) { const uint32_t r = r0 + i * V41_VQ_WARPS; if (r < OUT && (threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = 0.f; }
        return;
    }
    const uint8_t *cbd = md.cb;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u; carry.ex = 0u;
    if (r0 < OUT) carry = v41_vq_row_first_blk<NBIT, V3, EXT>(md, r0);       /* 首块先发, 藏在码本搬运后面 */
    if (cb_bytes) {
        if (md.nc * (V3 ? 8u : 16u) != cb_bytes) return;
        v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
        __syncthreads();
        cbd = vqsh;
    }
    const uint32_t *hs = h + (uint64_t)pair * (MID / 2u);
    for (uint32_t i = 0; i < nit; i++) {   /* 一 block 管 32×ITERS 行, 码本只搬一次(见 V41_VQ_ITERS 注释) */
        const uint32_t r = r0 + i * V41_VQ_WARPS;
        if (r >= OUT) break;
        const uint32_t rn = r + V41_VQ_WARPS;
        const int own = (i + 1u < nit && rn < OUT);
        const uint32_t *next = own ? v41_vq_row_ptr<NBIT, V3>(md, rn) : NULL;
        const float y = v41_bf16r(v41_vq_row_dot<NBIT, V3, EXT>(md, r, hs, cbd, cb_bytes != 0, &carry, next,
                                  EXT && own ? v41_vq_ext_ptr(md, rn) : NULL));
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
/* 验证批的 block 顺序表: order[q] = 第 q 个 block 算的对号, 按 (专家号, 对号) 升序; 一 block np(≤48) 线程各数"排我前面的有几个"。 */
__global__ static void v41_vq_order_kernel(int32_t *order, const int32_t *sel, uint32_t np) {
    const uint32_t p = threadIdx.x;
    if (p >= np) return;
    const int32_t e = sel[p];
    uint32_t rank = 0;
    for (uint32_t q = 0; q < np; q++) { const int32_t eq = sel[q]; if (eq < e || (eq == e && q < p)) rank++; }
    order[rank] = (int32_t)p;
}
static v41_scratch g_v41_vq_h, g_v41_vq_part, g_v41_vq_xb, g_v41_vq_ogc, g_v41_vq_ord;
/* ★多 token 分组核(2026-09-22, cuda_vq_group.inc.cu, 本 TU 后面的分片定义)★: n≥2 且码本在 shared 时, 同一专家被 ≥2 个 token 选中的
 * 那几段交给它(一 block 管一个专家 × m 个 token, 码字只解一次, 每 token 一次同式乘加), 只被一个 token 选中的对仍走下面的 SORTED 实例
 * (uniq_only=1)。stage 0 = gateup, 1 = down(down 必须在两条路的 gateup 都发完之后)。返回 1 = 发完; 0 = 不适用(全走逐对核); -1 = 启动失败。
 * --no-vq-group(g_ds4_v41_vq_group=0)钉回全逐对, 给同一二进制做 A/B。 */
template <int NBIT, int V3, int EXT>
static int v41_vq_grp_launch(int stage, uint32_t n_tok, uint16_t *h, float *part, const uint8_t *blob, const int32_t *sel, const uint32_t *xb,
                             uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t K, float clamp, uint32_t cbb, const float *gr,
                             const int32_t *ord, uint32_t np);
extern int g_ds4_v41_vq_group;
/* ★两个核分开判定★(2026-09-14 实撞): 码本 4096×16 B = 64 KB/本。gateup 要 gate+up 两本 = 128 KB,
 * 超过 GB10 每 block 的动态 shared 上限; down 只要一本 64 KB, 本来放得下。原先一个 ok 变量把两个核
 * 绑在一起, gateup 申请失败就把 down 一起打回全局 gather —— 解码实测只有 1.60 t/s(prefill 45 t/s 正常)。 */
template <int NBIT, int V3, int EXT>
static int v41_vq_fused_moe_n(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                              const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                              const float *gr) {
    if (n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || (IN % 8u) || (MID % 8u)) return 0;
    const uint64_t np = (uint64_t)n_tok * K;
    /* h 现在存 bf16(2 B/元素), 不是 f32 —— 它只被 down 核读, 而 down 读的就是打包形态。 */
    uint16_t *h = (uint16_t *)v41_grow(&g_v41_vq_h, np * MID * 2, "v41 vq h");
    float *part = (float *)v41_grow(&g_v41_vq_part, np * OUT * 4, "v41 vq partial");
    uint16_t *xb = (uint16_t *)v41_grow(&g_v41_vq_xb, (uint64_t)n_tok * IN * 2, "v41 vq x(bf16)");
    if (!h || !part || !xb) return 0;
    {   /* 激活打包成 bf16: 一层一次, n_tok×5120 个元素, 相对一层几百微秒的专家核可忽略 */
        const uint64_t nx = (uint64_t)n_tok * IN;
        v41_vq_xpack_kernel<<<(unsigned)((nx + 255) / 256), 256, 0, g_cur_stream>>>(xb, x, nx);
        if (!cuda_ok(cudaGetLastError(), "v41 vq xpack")) return 0;
        extern int g_ds4_v41_prof;
        if (g_ds4_v41_prof) {
            uint32_t *c = (uint32_t *)v41_grow(&g_v41_vq_ogc, 4, "v41 vq offgrid");
            uint32_t hc = 0;
            if (c && cudaMemsetAsync(c, 0, 4, g_cur_stream) == cudaSuccess) {
                v41_vq_offgrid_kernel<<<(unsigned)((nx + 255) / 256), 256, 0, g_cur_stream>>>(c, x, nx);
                if (cudaStreamSynchronize(g_cur_stream) == cudaSuccess &&
                    cudaMemcpy(&hc, c, 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    static uint64_t tot = 0, bad = 0;
                    tot += nx; bad += hc;
                    if ((tot / nx) % 40u == 0u)
                        fprintf(stderr, "[vq-grid] 激活不在 bf16 格点的元素: 累计 %llu / %llu\n",
                                (unsigned long long)bad, (unsigned long long)tot);
                }
            }
            v41_f16range_probe(x, NULL, nx, 0, "激活(x)");   /* mtp-2 §5.3: 定 f16 还是 TF32 */
        }
    }
    const uint32_t cbb = nc * (V3 ? 8u : 16u);   /* v3 码本每词 8 个 E4M3 ⇒ nc8192 也只 64 KB, 与今天 nc4096 f16 同大 */
    /* ★opt-in 的量要按"这个核实例 + 这一层的码本"记, 不能只记"判定过没有"★(2026-09-22 实撞, 投机路挂了一整天):
     * cudaFuncSetAttribute 批的是**某一个核函数能要多少动态 shared**, 批多少下次就只能用多少。v3 底座
     * (vq8sh14)两档码本并存 —— 浅 14 层 13 位是 8192 词(cbb 64 KB), 深层是 4096 词(32 KB) —— 原来这里
     * 是个文件级标志 `if (!g_v41_vq_sh_gateup)` 只跑一次: 谁先进来按谁的量批, 另一档启动时 shared 超过
     * 已批的量, cudaLaunch 直接回 invalid argument, 前向整个失败。
     * 症状为什么难认: 纯解码(n=1)走的是专家融合核, 根本不进这个函数, 只有投机验证批(n≥2)走这里 ⇒
     * 表面上是"投机一开就崩", 报错还挂在 gateup 头上, 与码本大小看不出关系。
     * 两件事一起修: ①static 局部变量在函数模板里是**每个实例一份**, 正好对上"每个 (NBIT,V3,EXT) 是不同的
     * 核函数, 各批各的"; ②判据从"判定过没有"换成"已批的量够不够这一层用", 不够就按新的量再批一次。
     * 批不上去(超设备上限)时 s_*_optin 不动 ⇒ 这一层自动回全局 gather, 而已经批好的小码本层不受牵连。 */
    static uint32_t s_gu_optin = 0u, s_dn_optin = 0u;   /* 这个实例已批到的动态 shared 字节; 0 = 还没批过 */
    if (s_gu_optin < cbb || s_dn_optin < cbb) {   /* 两个核各问各的: gateup 要两本(gate+up), down 只要一本 */
        int cap = 0; (void)cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        /* ★占用率是被谁卡住的, 要有实数才能判★(2026-09-16): 一个 SM 同时挂几个 block, 由"线程槽 /
         * shared / 寄存器"里最紧的那个定。per-block 的 shared 上限(cap)只说这一个 block 能要多少,
         * 真正决定并行度的是 **per-SM 的 shared 总量** 与 per-SM 线程上限 —— 缺这两个数就只能猜。 */
        int sh_sm = 0, thr_sm = 0, nsm = 0, regs_sm = 0;
        (void)cudaDeviceGetAttribute(&sh_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
        (void)cudaDeviceGetAttribute(&thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
        (void)cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        (void)cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
        const bool og = cudaFuncSetAttribute(v41_vq_gateup_kernel<NBIT, V3, EXT, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess &&
                        cudaFuncSetAttribute(v41_vq_gateup_kernel<NBIT, V3, EXT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        const bool od = cudaFuncSetAttribute(v41_vq_down_kernel<NBIT, V3, EXT, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess &&
                        cudaFuncSetAttribute(v41_vq_down_kernel<NBIT, V3, EXT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        /* ★占用率必须在开完 opt-in shared **之后**问★: 在 SetAttribute 之前调用, API 按默认 48 KB
         * 的限额算, 会判"一个都挂不上"(返回 0) —— 第一版就这么打出个 0%, 差点据此下结论。 */
        cudaFuncAttributes fa; memset(&fa, 0, sizeof fa);
        (void)cudaFuncGetAttributes(&fa, v41_vq_gateup_kernel<NBIT, V3, EXT, 0>);
        int blocks_sm = 0;
        const int thr_blk = (int)(V41_VQ_WARPS * 32u);
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm, v41_vq_gateup_kernel<NBIT, V3, EXT, 0>, thr_blk, (size_t)cbb);
        fprintf(stderr, "ds4: [v41] 设备: %d SM / 每 SM shared %d KB / 每 SM 线程 %d / 每 SM 寄存器 %d\n"
                        "ds4: [v41] VQ gateup 核: %d 线程/block, 每线程 %d 寄存器, 动态 shared %u KB"
                        " ⇒ 每 SM 挂 %d 个 block = %d 线程, **占用率 %.0f%%**\n"
                        "ds4: [v41]   谁卡的: shared %d 份 / 寄存器 %d 份 / 线程槽 %d 份(取最小)\n",
                nsm, sh_sm >> 10, thr_sm, regs_sm, thr_blk, fa.numRegs, cbb >> 10,
                blocks_sm, blocks_sm * thr_blk, thr_sm ? 100.0 * blocks_sm * thr_blk / thr_sm : 0.0,
                cbb ? sh_sm / (int)cbb : 0,
                fa.numRegs ? regs_sm / (fa.numRegs * thr_blk) : 0, thr_sm / thr_blk);
        (void)cudaGetLastError();
        if (og) s_gu_optin = cbb;   /* 批不上去就不动, 这一层自动回全局 gather(下一层码本小的照样走 shared) */
        if (od) s_dn_optin = cbb;
        fprintf(stderr, "ds4: [v41] VQ 码本 %u×16 B(%u KB/本, NBIT %d); 设备每 block 动态 shared 上限 %d KB ⇒ gate+up %s / down %s\n",
                nc, cbb >> 10, NBIT, cap >> 10, og ? "进 shared(一块用两遍)" : "回全局 gather", od ? "进 shared" : "回全局 gather");
    }
    const bool shg = cbb <= s_gu_optin, shd = cbb <= s_dn_optin;
    /* ★2026-09-16 判负存档: "多 token 小批按专家去重, 让重复的专家权重只读一份"(mtp.md M3)★
     * 依据看着很硬: 投机验证一次 4 行, 实测 4 行的 top-8 里唯一专家只有 **59%**(--v41-prof 的
     * [moe-uniq] 行), 四成的专家读是纯重复; 而专家是解码的大头。写了一版"一 block 管一个专家,
     * 位流与码本只过一遍, 对组里每个 token 各累一个点积"(逐位同, 只换了哪个 block 算哪一对)。
     * 同批 A/B(同一个 x/sel 背靠背发, 才不被接受率变化污染)实测: **一组收 2 个慢 1.7%, 收 4 个慢 7.1%**
     * —— 去重越多越慢, 单调。
     * 为什么: 这个核的大头不是从 DRAM 读权重, 是**激活在 L1 里的复用**。码本占掉 64 KB shared,
     * L1 只剩 ~64 KB; 逐对路一个 block 只碰一条激活(5120×4 B = 20 KB), 64 行里 63 行是 L1 命中。
     * 一组收 4 个 ⇒ 工作集 80 KB 装不下 ⇒ 每行都回 L2 重取, 省下的那点 DRAM 字节远抵不上。
     * (中途还踩了一个坑: 激活指针的 `pair/K` 写在了内层循环里, 运行期整数除法一行做上千次,
     *  第一版因此慢 23%, 差点据此把整条路判死 —— 提到循环外才露出真实的 1.7%。)
     * ★真正的账在这里★: 同一套尺量出 n=1 时专家核 0.356 ms/层, n=4 时 1.283 ms/层 = **3.6 倍**,
     * 权重一点没摊薄。投机解码的全部前提就是"验证 k+1 位约等于验证 1 位", 在这个形态上不成立。
     * 下次要动, 先回答: 怎么在保住激活 L1 复用的前提下省权重字节(两者在当前布局下互斥)。 */
    /* 线程数恒为 32 warp(shared 版一 warp 循环 ITERS 行)或 8 warp(全局 gather 版一 warp 一行);
     * ★不能写成 rows×32★: shared 版 rows 已是 256, 那会要 8192 个线程, 超过每 block 1024 的上限。 */
    const uint32_t rg = shg ? V41_VQ_GU_ROWS : 8u, rd = shd ? V41_VQ_DN_ROWS : 8u;
    const uint32_t tg = shg ? V41_VQ_WARPS * 32u : 8u * 32u, td = shd ? V41_VQ_WARPS * 32u : 8u * 32u;
    /* n≥2(投机验证批): 先出 block 顺序表, 两个核走 SORTED=1 的实例; n=1 走 SORTED=0, 与改前逐字相同(见 gateup 核头) */
    int32_t *ord = NULL;
    if (n_tok >= 2u) {
        ord = (int32_t *)v41_grow(&g_v41_vq_ord, np * 4, "v41 vq order");
        if (!ord) return 0;
        v41_vq_order_kernel<<<1, (unsigned)np, 0, g_cur_stream>>>(ord, sel, (uint32_t)np);
        if (!cuda_ok(cudaGetLastError(), "v41 vq order")) return 0;
    }
    /* ★发之前先把上游没检查的错误捡走★(2026-09-22 实撞): CUDA 的错误是粘的 —— 别的模块哪一发核没检查,
     * 下面这句 cudaGetLastError 就会把它算到 gateup 头上。这次投机路挂了, 报的是 "vq gateup failed",
     * 真凶在哪根本不知道, 白查了一轮。分开打, 就能一眼看出是"上游留的"还是"这一发自己的"。 */
    {
        const cudaError_t pre = cudaGetLastError();
        if (pre != cudaSuccess)
            fprintf(stderr, "ds4: ★[v41] 进 vq gateup 之前就有未清的 CUDA 错误: %s —— 真凶是上游某一发没检查的核★\n",
                    cudaGetErrorString(pre));
    }
    int grp = 0;   /* 1 = 多 token 的组走分组核, 逐对核只算单成员的对(两条路写不相交的 pair 槽) */
    if (ord && shg && shd && g_ds4_v41_vq_group) {   /* 分组核: 两本码本都进得了 shared 才走(它没有全局 gather 的形态) */
        grp = v41_vq_grp_launch<NBIT, V3, EXT>(0, n_tok, h, part, blob, sel, (const uint32_t *)xb, IN, MID, OUT, K, clamp, cbb, gr, ord, (uint32_t)np);
        if (grp < 0) return 0;
    }
    if (ord) v41_vq_gateup_kernel<NBIT, V3, EXT, 1><<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, g_cur_stream>>>(
        h, blob, sel, (const uint32_t *)xb, IN, MID, K, clamp, shg ? cbb : 0u, ord, (uint32_t)np, grp > 0 ? 1u : 0u);
    else v41_vq_gateup_kernel<NBIT, V3, EXT, 0><<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, g_cur_stream>>>(
        h, blob, sel, (const uint32_t *)xb, IN, MID, K, clamp, shg ? cbb : 0u, NULL, 0u, 0u);
    if (!cuda_ok(cudaGetLastError(), "v41 vq gateup")) {
        /* ★挂了必须连启动参数一起打★: "invalid argument" 只说"某个参数不对", 不说是哪个。把
         * grid/block/动态 shared 与模板实例(NBIT/V3/EXT/SORTED)一起打出来, 对着设备上限就能直接判:
         * shared 超过这个实例 opt-in 过的量 / grid 某一维是 0 / 线程数超 1024, 三者都一眼可见。
         * ★shared 那一项尤其要盯★: SetAttribute 只在第一次进这个函数时做一次, 而 cbb 是**逐层**算的
         * (码本大小 nc 随层变) —— 后面某层 cbb 比第一层大, 就是这个报错。 */
        fprintf(stderr, "ds4: [v41] gateup 启动参数: grid(%u,%u) block %u 动态 shared %u B; "
                        "NBIT %d V3 %d EXT %d SORTED %d; n_tok %u K %u np %llu IN %u MID %u nc %u cbb %u B 已批 %u B\n",
                (MID + rg - 1u) / rg, (unsigned)np, tg, shg ? cbb : 0u,
                NBIT, V3, EXT, ord ? 1 : 0, n_tok, K, (unsigned long long)np, IN, MID, nc, cbb, s_gu_optin);
        return 0;
    }
    {   /* h(swiglu 出口, bf16 格点)也要过 f16 范围: 它是 down 那一发 B 片的来源 */
        extern int g_ds4_v41_prof;
        if (g_ds4_v41_prof) v41_f16range_probe(NULL, h, np * MID, 1, "中间量(h)");
    }
    if (grp > 0 && v41_vq_grp_launch<NBIT, V3, EXT>(1, n_tok, h, part, blob, sel, (const uint32_t *)xb, IN, MID, OUT, K, clamp, cbb, gr, ord, (uint32_t)np) < 0) return 0;
    if (ord) v41_vq_down_kernel<NBIT, V3, EXT, 1><<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, g_cur_stream>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, shd ? cbb : 0u, gr, ord, (uint32_t)np, grp > 0 ? 1u : 0u);
    else v41_vq_down_kernel<NBIT, V3, EXT, 0><<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, g_cur_stream>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, shd ? cbb : 0u, gr, NULL, 0u, 0u);
    if (!cuda_ok(cudaGetLastError(), "v41 vq down")) return 0;
    if (!out) return 1;   /* 调用方稍后用 ds4_gpu_v41_moe_tail_tensor 把归约与 shared 专家的相加一发做完 */
    v41_vq_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>(out, part, w, K, OUT);
    return cuda_ok(cudaGetLastError(), "v41 vq reduce");
}
/* ★MoE 尾巴四发合一(2026-09-18, 小核合并)★: y = bf16(Σ_k w·partial + so)。原来是 reduce → copy(y←routed) → add(y+=so) → round 四发;
 * 算式一个字没动(同 k 序累加得 a, 再 a + so, 再舍 bf16) ⇒ 逐位同。partial 就是上面 down 核留在暂存里的那份。 */
__global__ static void v41_vq_tail_kernel(float *y, const float *partial, const float *w, const float *so, uint32_t K, uint32_t OUT) {
    const uint32_t t = blockIdx.y, o = blockIdx.x * 256u + threadIdx.x;
    if (o >= OUT) return;
    float a = 0.f;
    for (uint32_t k = 0; k < K; k++) a += w[(uint64_t)t * K + k] * partial[((uint64_t)t * K + k) * OUT + o];
    y[(uint64_t)t * OUT + o] = v41_bf16r(a + so[(uint64_t)t * OUT + o]);
}
int ds4_gpu_v41_moe_tail_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *so, const ds4_gpu_tensor *weights,
                                uint32_t n_tok, uint32_t n_used, uint32_t out_dim) {
    if (!y || !so || !weights || !g_v41_vq_part.p || g_v41_vq_part.cap < (uint64_t)n_tok * n_used * out_dim * 4) return 0;
    v41_vq_tail_kernel<<<dim3((out_dim + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>(
        (float *)y->ptr, (const float *)g_v41_vq_part.p, (const float *)weights->ptr, (const float *)so->ptr, n_used, out_dim);
    return cuda_ok(cudaGetLastError(), "v41 moe tail");
}
