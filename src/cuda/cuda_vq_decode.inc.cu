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
 * 09-17/18 第二刀: 位流按 384 B 整块(3 条整线)读进寄存器再 shfl 分发(见 v41_vq_row_dot)。此前位流读法三次判负,
 * 真因不是"读法不重要", 是那三版一轮都只请求 48 B —— DRAM 按 64 B 取, 一半白取(mem_ceiling ⑥ 量死的)。
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
/* 激活那 16 B 由调用方(v41_vq_blk_rounds)装好传进来(xw), 本函数只做查表 + 乘加。 */
__device__ __forceinline__ static float v41_vq_dot8(uint32_t v, uint4 xw, const uint8_t *cbs, int cb_shared) {
    uint2 cw0, cw1;
    if (cb_shared) { const uint4 c = *(const uint4 *)(cbs + (size_t)v * 16u); cw0.x = c.x; cw0.y = c.y; cw1.x = c.z; cw1.y = c.w; }
    else { cw0 = *(const uint2 *)(cbs + (size_t)v * 16u); cw1 = *(const uint2 *)(cbs + (size_t)v * 16u + 8u); }
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
/* ★位流按 384 B 整块读, 再用 shfl 分发(2026-09-18, fable5 "复审立案" 步 1 定版)★
 *
 * 为什么(全是 mem_ceiling.cu ⑤⑥ 量出来的, 不是猜的): 板上 L1/L2/DRAM 单跳 18/143/377 ns; 而"每 warp 私有行、
 * 每轮 48 B"这种流, **在飞 1 轮和 16 轮都只有 135~142 GB/s**, 加 `ld.global.L2::128B/256B` 提示也一样;
 * 每条指令读满 64 B ⇒ 187~207, 读满 **128 B 整线 ⇒ 219**(墙 237~241)。⇒ 这个核的最后一堵墙是**请求粒度**:
 * DRAM 按 64 B 取, 一轮只要 48 B 就有一半白取 —— 和延迟深度、指令条数、shared bank 都无关。
 * 步 0 的两版 prefetch(按线 / 按扇区, 提前一行)都只把 gateup 从 8.25 拉到 7.1 ms/步 = 133 GB/s, 正好贴在
 * 48 B 形态自己的上限上, 已撤。09-16 那次"12 个 lane 各读一字 + shfl"判负的真因也在这: 它一轮还是 48 B。
 *
 * 怎么读: 一块 = 3 条 128 B 线 = 96 个 32 位字 = 256 个索引 = 8 轮。lane j 用三条 LDG.32 拿字 j / j+32 / j+64,
 * 每条指令 32 个 lane 正好一条整线。第 k 轮 lane j 要的索引 j+32k 在块内第 12k+⌊3j/8⌋ 个字, 位移 (12j)%32;
 * 位移 > 20 的 lane(j%8 ∈ {2,5}, 固定 8 个)跨到下一个字 ⇒ 每轮两条 shfl(低字、高字)+ funnelshift + and。
 * 持有字 W 的是 lane W%32 的第 W/32 个寄存器; 一轮的 12 个字只在 k%8 ∈ {2,5} 时跨寄存器, 那两轮供给方
 * 按"自己被问的是哪个字"选寄存器, 其余轮寄存器号是编译期常量。
 * ★lane↔索引映射、每 lane 的累加次序、跨 lane 的归约树一个字没动 ⇒ 输出逐字节同, 门 = cmp。★
 * 尾块(gateup 一行 20 轮 = 2 块 + 4 轮)只装 48 个字, 越界的字不读(谓词), 不靠载荷尾巴的 8 B 零垫。 */
__device__ __forceinline__ static uint32_t v41_vq_sel3(uint32_t r, uint32_t w0, uint32_t w1, uint32_t w2) {
    return r == 0u ? w0 : (r == 1u ? w1 : w2);
}
/* 一块位流在寄存器里的样子: lane j 持有块内第 j / j+32 / j+64 个字。 */
typedef struct { uint32_t w0, w1, w2; } v41_vq_blk;
template <int NBIT>
__device__ __forceinline__ static const uint32_t *v41_vq_row_ptr(const v41_vq_mat &m, uint32_t r) {
    return (const uint32_t *)(m.ix + (((uint64_t)r * m.nidx_row * NBIT) >> 3));
}
/* 装一块: 只读本块真有的字(尾块 48 个), 越界的 lane 不发读 —— 不靠载荷尾巴的 8 B 零垫。 */
__device__ __forceinline__ static v41_vq_blk v41_vq_blk_load(const uint32_t *blk, uint32_t nw) {
    const uint32_t lane = threadIdx.x & 31u;
    v41_vq_blk b; b.w0 = 0u; b.w1 = 0u; b.w2 = 0u;
    if (lane < nw) b.w0 = blk[lane];
    if (lane + 32u < nw) b.w1 = blk[lane + 32u];
    if (lane + 64u < nw) b.w2 = blk[lane + 64u];
    return b;
}
/* ★跨块软件流水, 提前一块(2026-09-18 第三刀)★
 * 整块读落地后 ncu 说 long_scoreboard 仍 17~18 —— 每个 block 三次 __syncthreads 之后 32 个 warp **锁步**: 一起发块读、
 * 一起等 DRAM 900 周期、一起算 8 轮, SM 与 DRAM 轮流空转, 占用率再高也盖不住(大家停在同一拍上)。所以块的 3 个字在
 * 上一块开算之前就发出去(carry: 本行首块由上一行的末块顺手装好; 跨阶段时 gate 末行装 up 首行, 藏在码本搬运与同步后面)。
 * 12k 尺: gateup 7.06 → 6.72, down 4.51 → 3.76 ms/步(连同 down 4 行/warp + 码本搬运批发)。
 * ★同日两次加码都判负, 存档★(同尺, 逐字节同):
 *   ①"激活提前一轮装进寄存器": gateup 6.74 持平、down 4.07 退 —— 每轮那条激活 LDG 是 L1 命中(ncu 第五趟: L1 扇区命中 81%,
 *     未命中的 2.1 M 扇区里位流 0.83 M + 码本搬运 0.77 M 是必付的), 不是等待的来源; 多占 4 个寄存器顶到 64 硬上限反而伤。
 *   ②"两级队列, 提前两块(跨行跨阶段按块序列走)": gateup 7.40、down 3.90, 都退 —— 每块多 30~40 条簿记指令 + 6 个寄存器。
 *   ⇒ 这个核顶在 64 寄存器(1024 线程/block 的硬上限)上, **再加任何寄存器都是负的**; long_scoreboard 剩下的那份不是
 *   块读没提前够, 更像是 32 个 warp 同拍往 L1TEX 队列里塞 16 个波前/轮(码本查表 11.7 + 激活 4)的排队延迟。 */
template <int NBIT>
__device__ __forceinline__ static v41_vq_blk v41_vq_row_first_blk(const v41_vq_mat &m, uint32_t r) {
    const uint32_t R = m.nidx_row >> 5;
    return v41_vq_blk_load(v41_vq_row_ptr<NBIT>(m, r), (R < 8u ? R : 8u) * NBIT);
}
/* 一块的最多 8 轮: 第 k 轮 lane j 取索引 32(b+k)+j, 两条 shfl 拿字、funnelshift 取 12 位、查表乘加。乘加次序与旧核一字不差。
 * ★判负存档(2026-09-18)★ "轮数做模板参数变直线代码(1~7 轮尾块各一份实例 + switch)": 想让编译器把下一轮的 shfl/查表交错进
 * 本轮的乘加链。gateup 6.68 / down 3.92 —— 噪声内, 且 gateup 顶到 64 寄存器带 32 B spill。运行期轮数 + 每轮一条分支就够。 */
template <int NBIT>
__device__ __forceinline__ static void v41_vq_blk_rounds(const v41_vq_blk &cur, uint32_t rounds, uint32_t a, uint32_t sh, uint32_t imsk,
                                                         const uint32_t *xb, const uint8_t *cbs, int cb_shared, float &acc) {
    const uint32_t lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint4 xa = *(const uint4 *)(xb + (size_t)(k * 32u + lane) * 4u);   /* 8 个 bf16 = 16 B, L1 命中 */
        const uint32_t f = k * NBIT;                                     /* 本轮首字(编译期常量) */
        const uint32_t lo_r = ((f & 31u) > 32u - NBIT) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - NBIT) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
        const uint32_t v = __funnelshift_r(lo, hi, sh) & imsk;
        acc += v41_vq_dot8(v, xa, cbs, cb_shared);
    }
}
/* 一 warp 算一行与激活 x 的点积: 索引 j 归 lane j%32, 一轮 32 个 lane 覆盖 32 个索引。
 * 返回 增益 × Σ(已 warp 规约)。NBIT = 码本号位宽(4096 词 = 12), 也是一轮的字数。
 * carry 进来是本行首块, 出去是 next 行的首块(next 为 NULL 就是空块); 每块的字在上一块开算之前就发出去。 */
template <int NBIT>
__device__ __forceinline__ static float v41_vq_row_dot(const v41_vq_mat &m, uint32_t r, const uint32_t *xs, const uint8_t *cbs, int cb_shared,
                                                       v41_vq_blk *carry, const uint32_t *next) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t R = m.nidx_row >> 5;                                  /* 轮数: gateup 20, down 9 */
    const uint32_t a = (lane * NBIT) >> 5, sh = (lane * NBIT) & 31u;    /* 本 lane 在一轮 NBIT 个字里的字号与位移 */
    const uint32_t *row = v41_vq_row_ptr<NBIT>(m, r);
    v41_vq_blk cur = *carry;
    float acc = 0.f;
    for (uint32_t b = 0; b < R; b += 8u) {
        v41_vq_blk nxt;
        if (b + 8u < R) { const uint32_t rem = R - b - 8u; nxt = v41_vq_blk_load(row + (size_t)(b + 8u) * NBIT, (rem < 8u ? rem : 8u) * NBIT); }
        else if (next) nxt = v41_vq_blk_load(next, (R < 8u ? R : 8u) * NBIT);
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; }
        /* ★判负存档(2026-09-18)★ "再往后一块用 prefetch.global.L2 先送进 L2(12 个 lane 各一扇区, 零寄存器)": 依据是带载 DRAM
         * 一跳 1.2~1.4 µs(mem_ceiling ⑦; 空载 0.33), 怕寄存器提前一块(8 轮)盖不住。实测 gateup 6.93 / down 3.78, 噪声内 ⇒
         * 位流的到达不是剩余等待的来源。 */
        v41_vq_blk_rounds<NBIT>(cur, (R - b < 8u) ? R - b : 8u, a, sh, m.imsk, xs + (size_t)b * 32u * 4u, cbs, cb_shared, acc);
        cur = nxt;
    }
    *carry = cur;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    /* 行增益在行尾才取(试过提到行首: 多占寄存器, gateup 6.72 → 7.11, 退回) */
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
template <int NBIT>
__global__ static void v41_vq_gateup_kernel(uint16_t *h, const uint8_t *blob, const int32_t *sel, const uint32_t *x,
                                            uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = blockIdx.y, t = pair / K, rows = cb_bytes ? V41_VQ_GU_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    const v41_vq_mat mg = v41_vq_open(blob, e, 0, MID, IN, NULL), mu = v41_vq_open(blob, e, 1, MID, IN, NULL);
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
        if (mg.nc * 16u != cb_bytes || mu.nc * 16u != cb_bytes) return;
        v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u;
        if (r0 < MID) carry = v41_vq_row_first_blk<NBIT>(mg, r0);   /* 首块先发, 藏在码本搬运后面 */
        v41_vq_cb_to_shared(vqsh, mg.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r = r0 + i * V41_VQ_WARPS;
            if (r >= MID) break;
            const uint32_t rn = r + V41_VQ_WARPS;   /* gate 末行时预装 up 的首行(它在码本换本之后才算) */
            const uint32_t *next = (i + 1u < nit && rn < MID) ? v41_vq_row_ptr<NBIT>(mg, rn) : v41_vq_row_ptr<NBIT>(mu, r0);
            const float gv = v41_bf16r(v41_vq_row_dot<NBIT>(mg, r, xs, vqsh, 1, &carry, next));
            if (lead) hp[r] = (uint16_t)(__float_as_uint(gv) >> 16);
        }
        __syncthreads();                            /* 等所有 warp 读完 gate 码本, 才能覆盖它 */
        v41_vq_cb_to_shared(vqsh, mu.cb, cb_bytes);
        __syncthreads();
        for (uint32_t i = 0; i < nit; i++) {
            const uint32_t r = r0 + i * V41_VQ_WARPS;
            if (r >= MID) break;
            const uint32_t rn = r + V41_VQ_WARPS;
            const uint32_t *next = (i + 1u < nit && rn < MID) ? v41_vq_row_ptr<NBIT>(mu, rn) : NULL;
            const float ui = v41_bf16r(v41_vq_row_dot<NBIT>(mu, r, xs, vqsh, 1, &carry, next));
            if (lead) hp[r] = v41_vq_swiglu(__uint_as_float((uint32_t)hp[r] << 16), ui, clamp);
        }
    } else if (r0 < MID) {
        v41_vq_blk carry = v41_vq_row_first_blk<NBIT>(mg, r0);
        const float gv = v41_bf16r(v41_vq_row_dot<NBIT>(mg, r0, xs, mg.cb, 0, &carry, v41_vq_row_ptr<NBIT>(mu, r0)));
        const float ui = v41_bf16r(v41_vq_row_dot<NBIT>(mu, r0, xs, mu.cb, 0, &carry, NULL));
        if (lead) hp[r0] = v41_vq_swiglu(gv, ui, clamp);
    }
}
/* down: partial[pair][OUT] = bf16(W2·h) */
template <int NBIT>
__global__ static void v41_vq_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const uint32_t *h,
                                          uint32_t MID, uint32_t OUT, uint32_t K, uint32_t cb_bytes, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = blockIdx.y, rows = cb_bytes ? V41_VQ_DN_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    const v41_vq_mat md = v41_vq_open(blob, e, 2, OUT, MID, gr ? gr + (size_t)e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS_DN : 1u;
    if (!md.ok) {   /* 载荷不对: 这 block 负责的行全写 0(不能只写一行, 下游 reduce 会读到脏值) */
        for (uint32_t i = 0; i < nit; i++) { const uint32_t r = r0 + i * V41_VQ_WARPS; if (r < OUT && (threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = 0.f; }
        return;
    }
    const uint8_t *cbd = md.cb;
    v41_vq_blk carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u;
    if (r0 < OUT) carry = v41_vq_row_first_blk<NBIT>(md, r0);       /* 首块先发, 藏在码本搬运后面 */
    if (cb_bytes) {
        if (md.nc * 16u != cb_bytes) return;
        v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
        __syncthreads();
        cbd = vqsh;
    }
    const uint32_t *hs = h + (uint64_t)pair * (MID / 2u);
    for (uint32_t i = 0; i < nit; i++) {   /* 一 block 管 32×ITERS 行, 码本只搬一次(见 V41_VQ_ITERS 注释) */
        const uint32_t r = r0 + i * V41_VQ_WARPS;
        if (r >= OUT) break;
        const uint32_t rn = r + V41_VQ_WARPS;
        const uint32_t *next = (i + 1u < nit && rn < OUT) ? v41_vq_row_ptr<NBIT>(md, rn) : NULL;
        const float y = v41_bf16r(v41_vq_row_dot<NBIT>(md, r, hs, cbd, cb_bytes != 0, &carry, next));
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
template <int NBIT>
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
        const bool og = cudaFuncSetAttribute(v41_vq_gateup_kernel<NBIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        const bool od = cudaFuncSetAttribute(v41_vq_down_kernel<NBIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        /* ★占用率必须在开完 opt-in shared **之后**问★: 在 SetAttribute 之前调用, API 按默认 48 KB
         * 的限额算, 会判"一个都挂不上"(返回 0) —— 第一版就这么打出个 0%, 差点据此下结论。 */
        cudaFuncAttributes fa; memset(&fa, 0, sizeof fa);
        (void)cudaFuncGetAttributes(&fa, v41_vq_gateup_kernel<NBIT>);
        int blocks_sm = 0;
        const int thr_blk = (int)(V41_VQ_WARPS * 32u);
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_sm, v41_vq_gateup_kernel<NBIT>, thr_blk, (size_t)cbb);
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
    const uint32_t rg = shg ? V41_VQ_GU_ROWS : 8u, rd = shd ? V41_VQ_DN_ROWS : 8u;
    const uint32_t tg = shg ? V41_VQ_WARPS * 32u : 8u * 32u, td = shd ? V41_VQ_WARPS * 32u : 8u * 32u;
    v41_vq_gateup_kernel<NBIT><<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, g_cur_stream>>>(
        h, blob, sel, (const uint32_t *)xb, IN, MID, K, clamp, shg ? cbb : 0u);
    if (!cuda_ok(cudaGetLastError(), "v41 vq gateup")) return 0;
    {   /* h(swiglu 出口, bf16 格点)也要过 f16 范围: 它是 down 那一发 B 片的来源 */
        extern int g_ds4_v41_prof;
        if (g_ds4_v41_prof) v41_f16range_probe(NULL, h, np * MID, 1, "中间量(h)");
    }
    v41_vq_down_kernel<NBIT><<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, g_cur_stream>>>(
        part, blob, sel, (const uint32_t *)h, MID, OUT, K, shd ? cbb : 0u, gr);
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
/* 按码本词数分发: 位宽 = ⌈log2(词数)⌉ 决定一轮的字数, 核按它实例化(现役配方 vq8x4096 ⇒ 12 位)。
 * 一轮 = 32 个索引 ⇒ 行的索引数(cols/8)必须是 32 的倍数(IN/MID 是 256 的倍数); 不满足就硬错, 不留慢路。 */
static int v41_vq_fused_moe(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                            const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x, uint32_t n_tok, uint32_t nc,
                            const float *gr) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    /* 一轮 32 个索引 ⇒ IN/MID 是 256 的倍数(V4.1 Flash: 5120 → 20 轮, 2304 → 9 轮); 尾块几轮都行 */
    if ((IN % 256u) || (MID % 256u)) {
        fprintf(stderr, "ds4: [v41] VQ 解码核要求 IN/MID 是 256 的倍数(一轮 32 个索引), 现 %u/%u\n", IN, MID);
        return 0;
    }
    if (nbit == 12u) return v41_vq_fused_moe_n<12>(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr);
    fprintf(stderr, "ds4: [v41] VQ 码本 %u 词(%u 位)没有对应的解码核实例\n", nc, nbit);
    return 0;
}
