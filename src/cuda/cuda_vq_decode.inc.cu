/* cuda_vq_decode.inc.cu — VQ 专家的"解码即乘"核(2026-09-16 从 cuda_v41_4.inc.cu 拆出)。
 *
 * 为什么单独一片: 原来它和骨架 fp4x32 GEMV 挤在一个文件里, 两条家族的演进史互不相干,
 * 加几条判负存档就破了 500 行守卫。拆开之后这一片只讲一件事: **压缩态专家权重怎么直接参与乘法**。
 *
 * 盘上形态(DQVL v2 blob): 16 B 头 + 384×3 个槽表; 每槽是一份 DQVQ 载荷 =
 * 16 B 头 + 码本(4096 × 8 个 f16 = 64 KB) + 逐行增益 f16 + 位流(每 8 个元素一个 12 位码本号)。
 * 一行 640 个索引 = 960 B; 一个专家三个矩阵约 5.9 MB; 一层 8 个专家约 47 MB。
 *
 * 09-16 体检与第一刀: 这两个核原本合计 14.23 ms/步, 离板子 240 GB/s 的墙很远。ncu 指出真凶不是权重 ——
 * 一次发射(一层 gateup)全局 load 扇区 42.0 M × 32 B = 1.34 GB, 而这一层的权重才 31.5 MB, **放大 43 倍**,
 * 那些扇区几乎全是**激活重读**。改激活存 bf16 后 **14.23 → 13.28 ms/步**, 且逐位同(见 v41_vq_dot8)。
 * 位流的读法试过三种花样, 三次全负(存档在 v41_vq_row_dot 里) —— 别再往那边使劲。
 *
 * ★必须排在 cuda_v41_4.inc.cu 之后★: 用它的 v41_bf16r / v41_grow / v41_scratch;
 * 又必须排在 cuda_v41_draft.inc.cu 之前(草稿塔借本片的 reduce 核与暂存槽)。 */
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
/* ★激活以 bf16 存(2026-09-16, 本核最大的一刀)★
 *
 * 为什么: L1 每周期只供得起一个 wavefront(128 B), 所以这个核的产能单位是**wavefront 数**, 不是字节数。
 * 数一轮(32 个 lane 取 32 个索引)要付多少:
 *   激活 16 个 —— 每 lane 8 个 f32 = 32 B, 没有 32 B 的读指令, 只能拆成两条 float4;
 *                每条的 32 个 lane 地址相隔 32 B, 一条就铺开 1024 B = 8 个 wavefront, 两条 16 个
 *   码本 ~4 个(随机查表的 bank 冲突) + 位流 4 个(4 条 LDG.E.U8)
 * **激活一个人占了三分之二**, 而它本来就在 bf16 格点上(调用方 rms_norm / 上一步的 bf16r 舍过) ——
 * 存 f32 等于把每个数的低 16 位零白搬一遍。
 * 改成 bf16 后 8 个元素正好 16 B **连续**, 一条 uint4 拿完, 32 个 lane 合起来 512 B = **4 个 wavefront**。
 *
 * ★逐位同★: bf16 → f32 就是低位补 16 个零, 而值本来就在格点上(低 16 位全零), 来回一趟精确无损,
 * 乘加的次序也一个字没变。所以它的门是**输出逐字节相同**, 不是质量尺。
 * ★出错会怎样★: 哪天上游忘了舍 bf16 就喂进来, 这里会把低 16 位直接丢掉 —— 不报错, 只是悄悄少几位精度。
 * 所以打包核用的是 v41_bf16r(舍), 不是截断: 真没舍过也只差一次正确的舍入, 不会是垃圾。 */
__device__ __forceinline__ static float v41_vq_dot8(uint32_t v, const uint32_t *x8b, const uint8_t *cbs, int cb_shared) {
    uint2 cw0, cw1;
    if (cb_shared) { const uint4 c = *(const uint4 *)(cbs + (size_t)v * 16u); cw0.x = c.x; cw0.y = c.y; cw1.x = c.z; cw1.y = c.w; }
    else { cw0 = *(const uint2 *)(cbs + (size_t)v * 16u); cw1 = *(const uint2 *)(cbs + (size_t)v * 16u + 8u); }
    const uint4 xw = *(const uint4 *)x8b;   /* 8 个 bf16 = 16 B, 一条读 */
    __half2 h0, h1, h2, h3; memcpy(&h0, &cw0.x, 4); memcpy(&h1, &cw0.y, 4); memcpy(&h2, &cw1.x, 4); memcpy(&h3, &cw1.y, 4);
    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
    /* 一个 uint32 装两个 bf16: 低半是第 2k 个元素, 高半是第 2k+1 个。补零还原成 f32。 */
    return f0.x * __uint_as_float(xw.x << 16) + f0.y * __uint_as_float(xw.x & 0xffff0000u)
         + f1.x * __uint_as_float(xw.y << 16) + f1.y * __uint_as_float(xw.y & 0xffff0000u)
         + f2.x * __uint_as_float(xw.z << 16) + f2.y * __uint_as_float(xw.z & 0xffff0000u)
         + f3.x * __uint_as_float(xw.w << 16) + f3.y * __uint_as_float(xw.w & 0xffff0000u);
}
/* f32(已在 bf16 格点) → 打包成 bf16。一线程一元素, 每层一次, 5120 个元素, 可忽略。 */
__global__ static void v41_vq_xpack_kernel(uint16_t *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (uint16_t)(__float_as_uint(v41_bf16r(src[i])) >> 16);
}
/* ★"激活已在 bf16 格点"这条不变量的看门狗(只在 --v41-prof 下跑)★
 * 上面那一刀的逐位同**完全建立在这条不变量上**: 值的低 16 位全是零, 打包/还原才是恒等变换。
 * 与其去重建一个改前的二进制来比输出哈希, 不如直接数一遍"低 16 位非零的元素有几个" ——
 * 是 0 就说明这个变换在数学上是恒等的, 比对哈希更硬(哈希相同只证明这一条提示没露馅)。
 * 哪天上游改了 rms_norm 的出口舍入, 这个数会立刻从 0 变成几千, 而速度和输出都"看着正常"。 */
__global__ static void v41_vq_offgrid_kernel(uint32_t *cnt, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (__float_as_uint(src[i]) & 0xffffu) != 0u) atomicAdd(cnt, 1u);
}
/* 一 warp 算一行与激活 x 的点积: 索引 j 归 lane j%32, 一轮 32 个 lane 覆盖 32 个索引。
 * 返回 增益 × Σ(已 warp 规约)。两条取索引的路见下, 数值上完全相同。 */
__device__ __forceinline__ static float v41_vq_row_dot(const v41_vq_mat &m, uint32_t r, const uint32_t *xs, const uint8_t *cbs, int cb_shared) {
    const uint64_t i0 = (uint64_t)r * m.nidx_row;
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.f;
    /* ★2026-09-16 判负存档(第三次撞同一堵墙): "一轮 32 个 lane 合作读位流, 再用 shfl 分发"★
     * 依据看着最硬的一版: 一轮要取的 32 个索引 = 32 × nbit 位 = **恰好 nbit 个对齐的 32 位字**(永远整除),
     * 所以让 lane 0..nbit-1 各读一个字(合成一次 48 B 合并请求)、广播全 warp、各自 funnelshift 取自己那 12 位,
     * 访存指令一轮 4 条 LDG.E.U8 → **1 条 LDG.E.32**, 而且逐位同。
     * 同一把尺(nsys 逐核表, 12k 真提示, 与 bf16 激活那一刀分开量):
     *   gateup 8.37 → 8.78(+4.9%) / down 4.98 → 6.03(**+21%**) —— 两个核都更慢, 已回退。
     * 为什么: 那 4 条字节读本来就落在同一个 32 B 扇区里, L1 一次就供上, 它们**根本不在等内存**;
     * 换来的却是"载入 → 两次 shfl → funnelshift"的依赖链, 而 shfl 要全 warp 对齐, 把原来各 lane
     * 各自独立取数的那点并行也一起掐掉了。down 更惨是因为它一行只有 8 轮(gateup 20 轮),
     * 每行摊到的固定开销本来就重。
     * ★这是这条路上第三次同源判负★(另两次: 09-16 "两条对齐读 + funnelshift" 慢 32%,
     * 09-16 "8 lane 分一个块 + 一条 float4" 慢 9%) ⇒ **位流的读法不是这个核的瓶颈, 别再动它**。
     * 真正有效的那一刀是下面的激活 bf16(见 v41_vq_dot8 的注释): 瓶颈在**激活**, 不在权重。 */
    #pragma unroll 4
    for (uint32_t j = lane; j < m.nidx_row; j += 32u) {
        const uint64_t bit = (i0 + j) * m.nbit, by = bit >> 3;
        /* ★2026-09-16 判负存档: "非对齐 4 字节 memcpy → 两条对齐 32 位读 + funnelshift"★
         * 依据(看着很硬): SASS 里这一句编出 **4 条 `LDG.E.U8`** —— 硬件没有非对齐 32 位读, 编译器
         * 只能拆成 4 次字节读。一行 640 个索引 = 2560 条访存指令, 有效字节才 960 B。改成
         * "按 4 字节对齐读两个字 + funnelshift 拼窗口"(fused2 核 ③ 的做法), 指令 4 → 2。
         * 实测 **gateup 9.13 → 12.07 ms / down 4.91 → 6.74 ms(慢 32~37%)**, 已回退。
         * 为什么反而慢: 末尾几个索引的第二个字会越出位流, 得加边界判断, 于是热循环里多了一个
         * 三岔分支 + 一次 64 位指针比较; 而那 4 条字节读本来就落在**同一个 32 B 扇区**里, L1 一次就供上,
         * 省下的 2 条指令买不回分支的钱。★教训★: "指令条数少了"不等于"快", 得看这些指令是不是真在等内存。
         * 下次要动这里, 先用 ncu 问 `smsp__warp_issue_stalled_*` 看 warp 到底停在哪一类原因上。 */
        uint32_t wv; memcpy(&wv, m.ix + by, 4);
        const uint32_t v = (wv >> (bit & 7)) & m.imsk;
        acc += v41_vq_dot8(v, xs + (size_t)j * 4u, cbs, cb_shared);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    return acc * __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
}
/* ★2026-09-15 single.md S2 判负存档: "码本条目改 20 B 跨距消 bank conflict"★
 * 假设: 条目 16 B = 4 个 bank, 32 个 lane 拿随机码本号 v, 地址 v*16 落在 bank 组 (v*4)%32 只有 8 个值
 * ⇒ 平均 4 路冲突。改 20 B(5 bank, gcd(5,32)=1)让它铺满 32 个 bank。
 * 实测 **58.7 → 59.9 ms(持平偏慢)**, 回退。两个原因: ①20 B 不是 8 的倍数, uint2 读直接
 * misaligned 崩(CUDA flush failed: misaligned address), 只能退成 4 次 uint 读, 多出来的指令
 * 吃掉了省下的冲突; ②8 的倍数的跨距做不到 bank 全覆盖(stride/4 必为偶数) —— 对齐与全覆盖互斥。
 * ⇒ 这个核的瓶颈也不是 shared bank。 */
__device__ __forceinline__ static void v41_vq_cb_to_shared(uint8_t *dst, const uint8_t *src, uint32_t bytes) {   /* bytes 是 8 的倍数 */
    const uint2 *s = (const uint2 *)src; uint2 *d = (uint2 *)dst;
    for (uint32_t i = threadIdx.x; i < bytes / 8u; i += blockDim.x) d[i] = s[i];
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
 * (09-15 长尾修掉后重扫 ITERS: 2 = 58.7 ms < 4 = 59.8 < 8 = 63.0, 维持 2。) */
#define V41_VQ_ITERS   2u
#define V41_VQ_SH_ROWS (32u * V41_VQ_ITERS)
/* gate/up 同核(官方 Expert: w1/w3 出 bf16 → f32 截断 → silu(g)·u → bf16); cb_bytes>0: 码本进 shared(grid.x 按 32 行),
 * 否则全局 gather(grid.x 按 8 行)。x 已是 bf16 格点(调用方 rms_norm 出口舍过)。 */
__global__ static void v41_vq_gateup_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                                            uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = blockIdx.y, t = pair / K, rows = cb_bytes ? V41_VQ_SH_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    const v41_vq_mat mg = v41_vq_open(blob, e, 0, MID, IN, NULL), mu = v41_vq_open(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS : 1u;
    const uint32_t *xs = x + (uint64_t)t * (IN / 2u);   /* 两个 bf16 一个字 */
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
        if ((threadIdx.x & 31u) == 0) h[(uint64_t)pair * MID + r] = (uint16_t)(__float_as_uint(v41_bf16r(sg * ui)) >> 16);
    }
}
/* down: partial[pair][OUT] = bf16(W2·h) */
__global__ static void v41_vq_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *h,
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
    const uint32_t *hs = h + (uint64_t)pair * (MID / 2u);
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
static v41_scratch g_v41_vq_h, g_v41_vq_part, g_v41_vq_xb, g_v41_vq_ogc;
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
        }
    }
    const uint32_t cbb = nc * 16u;
    if (!g_v41_vq_sh_gateup) {   /* 一次性, 两个核各问各的: gateup 要两本(gate+up), down 只要一本 */
        int cap = 0; (void)cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        /* ★占用率是被谁卡住的, 要有实数才能判★(2026-09-16): 一个 SM 同时挂几个 block, 由"线程槽 /
         * shared / 寄存器"里最紧的那个定。per-block 的 shared 上限(cap)只说这一个 block 能要多少,
         * 真正决定并行度的是 **per-SM 的 shared 总量** 与 per-SM 线程上限 —— 缺这两个数就只能猜。 */
        int sh_sm = 0, thr_sm = 0, nsm = 0, regs_sm = 0;
        (void)cudaDeviceGetAttribute(&sh_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
        (void)cudaDeviceGetAttribute(&thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
        (void)cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        (void)cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
        const bool og = cudaFuncSetAttribute(v41_vq_gateup_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        const bool od = cudaFuncSetAttribute(v41_vq_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        /* ★占用率必须在开完 opt-in shared **之后**问★: 在 SetAttribute 之前调用, API 按默认 48 KB
         * 的限额算, 会判"一个都挂不上"(返回 0) —— 第一版就这么打出个 0%, 差点据此下结论。 */
        cudaFuncAttributes fa; memset(&fa, 0, sizeof fa);
        (void)cudaFuncGetAttributes(&fa, v41_vq_gateup_kernel);
        int blocks_sm = 0;
        const int thr_blk = (int)(32u * 32u);
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm, v41_vq_gateup_kernel, thr_blk, (size_t)cbb);
        fprintf(stderr, "ds4: [v41] 设备: %d SM / 每 SM shared %d KB / 每 SM 线程 %d / 每 SM 寄存器 %d\n"
                        "ds4: [v41] VQ gateup 核: %d 线程/block, 每线程 %d 寄存器, 动态 shared %u KB"
                        " ⇒ 每 SM 挂 %d 个 block = %d 线程, **占用率 %.0f%%**\n"
                        "ds4: [v41]   谁卡的: shared %d 份 / 寄存器 %d 份 / 线程槽 %d 份(取最小)\n",
                nsm, sh_sm >> 10, thr_sm, regs_sm, thr_blk, fa.numRegs, cbb >> 10,
                blocks_sm, blocks_sm * thr_blk, thr_sm ? 100.0 * blocks_sm * thr_blk / thr_sm : 0.0,
                cbb ? sh_sm / (int)cbb : 0,
                fa.numRegs ? regs_sm / (fa.numRegs * thr_blk) : 0, thr_sm / thr_blk);
        (void)cudaGetLastError();
        g_v41_vq_sh_gateup = og ? 1 : -1;
        g_v41_vq_sh_down = od ? 1 : -1;
        fprintf(stderr, "ds4: [v41] VQ 码本 %u×16 B(%u KB/本); 设备每 block 动态 shared 上限 %d KB ⇒ gate+up %s / down %s\n",
                nc, cbb >> 10, cap >> 10, og ? "进 shared(一块用两遍)" : "回全局 gather", od ? "进 shared" : "回全局 gather");
    }
    const bool shg = g_v41_vq_sh_gateup == 1, shd = g_v41_vq_sh_down == 1;
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
    const uint32_t rg = shg ? V41_VQ_SH_ROWS : 8u, rd = shd ? V41_VQ_SH_ROWS : 8u;
    const uint32_t tg = shg ? 32u * 32u : 8u * 32u, td = shd ? 32u * 32u : 8u * 32u;
    v41_vq_gateup_kernel<<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, g_cur_stream>>>(h, blob, sel, (const uint32_t *)xb, IN, MID, K, clamp, shg ? cbb : 0u);
    if (!cuda_ok(cudaGetLastError(), "v41 vq gateup")) return 0;
    v41_vq_down_kernel<<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, g_cur_stream>>>(part, blob, sel, (const uint32_t *)h, MID, OUT, K, shd ? cbb : 0u, gr);
    if (!cuda_ok(cudaGetLastError(), "v41 vq down")) return 0;
    v41_vq_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>(out, part, w, K, OUT);
    return cuda_ok(cudaGetLastError(), "v41 vq reduce");
}
