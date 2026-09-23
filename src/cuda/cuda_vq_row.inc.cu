/* cuda_vq_row.inc.cu — VQ 载荷解析 + 一行点积(2026-09-21 从 cuda_vq_decode.inc.cu 拆出)。
 *
 * 为什么单独一片: 这一族(v41_vq_open / v41_vq_dot8 / v41_vq_row_dot)是**盘上布局的唯一消费点**,
 * 而 113.md 方案 v3 只动布局 —— 拆开之后 v3 的三处改动全在本片, 上面那片的两个 kernel 只多传一个模板参数。
 * ★必须排在 cuda_vq_decode.inc.cu 之前★(它用本片的 v41_vq_mat / row_dot); 本片用 cuda_v41_4 的 v41_bf16r。
 *
 * 【两个盘上版本, 用模板参数 V3 选, 不用运行期分支】(09-16 实撞: 同一实例里多一个运行期 `order ? :`, n=1 就慢 21%)
 *   V3=0  DQVL v2: 载荷 = [头 16][码本 nc×8 f16][行增益 rows f16][NBIT 位直排位流][8 B 垫]; 码本每载荷各带一份。
 *   V3=1  DQVL v3: 载荷 = [头 32(带 flags/主流位宽/cb_off)][行增益][**12 位**主流位流][(13 位层)第 13 位平面][8 B 垫];
 *                  码本**一层一本**住 blob 头之后(cb_off 指过去), 只存 E4M3(1 B/元素)。
 *   ⇒ V3=1 时 NBIT 是"有效位宽"(12 或 13), 主流恒 12 位 ⇒ 块几何(3 条整线 = 96 字 = 256 索引 = 8 轮)一个字没变,
 *     这正是 09-18 用 mem_ceiling 量死"读不满整线就掉到 142 GB/s"之后必须守住的东西。
 *   ⇒ 13 位那一位走位平面: 一块 = 256 个索引 = 32 B = 8 个字, lane j<8 各读一个字(一条 LDG.32),
 *     第 k 轮要的是"索引 32k+lane 的第 13 位" = 块内第 k 个字的第 lane 位 ⇒ 一条 shfl(取 lane k 的字)+ 一次移位与。
 *     每 lane 只多 1 个寄存器(现 56/64; 09-18 记着"再加任何寄存器都是负的", 所以只肯加这一个)。 */

/* gov(2026-09-13): 该专家该矩阵的逐行增益【缩放因子】[rows] f32, NULL = 按载荷原样(权重侧反修没挂或不挂这个矩阵)。
 * ex(2026-09-21): v3 13 位层的第 13 位平面, NULL = 没有(12 位层与全部 v2 载荷)。 */
typedef struct { const uint8_t *cb, *gr, *ix, *ex; const float *gov; uint32_t nidx_row, nbit, imsk, nc; int ok; } v41_vq_mat;
template <int V3>
__device__ __forceinline__ static v41_vq_mat v41_vq_open(const uint8_t *blob, int e, int which, uint32_t rows, uint32_t cols,
                                                         const float *gov) {
    v41_vq_mat m; m.ok = 0; m.cb = m.gr = m.ix = m.ex = NULL; m.gov = gov; m.nidx_row = m.nbit = m.imsk = m.nc = 0;
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    if (!off) return m;
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != (V3 ? DS4VQ_MAT3_MAGIC : DS4VQ_MAT_MAGIC)) return m;   /* 版本不对 ⇒ ok=0(调用方写零或跳过), 不按本版布局硬解 */
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t r32, c32; memcpy(&r32, pay + 8, 4); memcpy(&c32, pay + 12, 4);
    if (r32 != rows || c32 != cols || d16 != 8u) return m;   /* 本核只写了 dim 8(V4.1 配方 vq8x4096) */
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    m.nidx_row = cols / 8u; m.nbit = nbit; m.imsk = (1u << nbit) - 1u; m.nc = n16;
    if (V3) {
        uint32_t flags, mnb; memcpy(&flags, pay + 16, 4); memcpy(&mnb, pay + 20, 4);
        uint64_t cb_off; memcpy(&cb_off, pay + 24, 8);
        if (mnb != 12u || !(flags & 1u)) return m;            /* 主流必须 12 位、码本必须 E4M3(转换器只产这一种) */
        m.cb = blob + cb_off;                                 /* 层码本: 同层三矩阵指同一处 */
        m.gr = pay + 32;
        m.ix = m.gr + (size_t)rows * 2u;
        const uint32_t mrow = (m.nidx_row * 12u + 7u) / 8u;
        m.ex = (flags & 2u) ? m.ix + (size_t)rows * mrow : NULL;
        if (((flags & 2u) != 0u) != (nbit > 12u)) return m;    /* 位平面在不在, 必须与码本词数推出的位宽一致 */
    } else {
        m.cb = pay + 16; m.gr = m.cb + (size_t)n16 * 8u * 2u; m.ix = m.gr + (size_t)rows * 2u;
    }
    m.ok = 1;
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
/* 激活那 16 B 由调用方(v41_vq_blk_rounds)装好传进来(xw), 本函数只做查表 + 乘加。
 * FP8=1(v3): 码字 8 个 E4M3 = 8 B, 一条 uint2 拿完; 解码走 ds4_e4m3fn_to_f32 的同一张表(全仓唯一实现, src/common/ds4_fp8.h)。
 * FP8=0(v2): 码字 8 个 f16 = 16 B, 与 09-16 定版逐字不差。 */
/* ★E4M3 → half2 必须走硬件指令★(2026-09-21 实撞, 这是本轮最贵的一个教训)
 * 首版用 `ds4_e4m3fn_to_f32`(src/common 的标量 C 实现: 位拆解 + 非规格化分支)逐个字节解码 ——
 * 数值完全正确, 但一个索引要解 8 个元素、一步有几百万个索引, 而 f16 那条路是**一条**硬件转换指令。
 * 实测代价: 12k 走图 42.65 → 57.4 ms/步(−26%), 预填 209 → 168 t/s(−20%); 12 位层(无位平面)同样慢
 * ⇒ 真因就在这个查表, 不在位平面。
 * sm_89+ 有 `cvt.rn.f16x2.e4m3x2`(GB10 = sm_121): 一条指令把 2 个 E4M3 字节变成 2 个 f16, 4 条覆盖一个码字。
 * 数值上是精确转换(E4M3 ⊂ f16), 与标量实现逐位同 —— 换的是指令数不是语义。 */
__device__ __forceinline__ static __half2 v41_e4m3x2_to_half2(uint32_t two) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    __half2 out; memcpy(&out, &h, 4);
    return out;
}
/* 码字 v → 8 个 f32(2026-09-22 从 dot8 拆出): 多 token 分组核(cuda_vq_group.inc.cu)一个索引只解一次码字, 对每个 token
 * 各做一次下面的 v41_vq_dot8_cw; 单 token 核仍走 v41_vq_dot8 = 解码 + dot8_cw, 内联后与拆分前同一棵表达式树。 */
template <int FP8>
__device__ __forceinline__ static void v41_vq_cw(uint32_t v, const uint8_t *cbs, int cb_shared, float *c) {
    if (FP8) {
        const uint2 w = *(const uint2 *)(cbs + (size_t)v * 8u);
        const __half2 p0 = v41_e4m3x2_to_half2(w.x), p1 = v41_e4m3x2_to_half2(w.x >> 16),
                      p2 = v41_e4m3x2_to_half2(w.y), p3 = v41_e4m3x2_to_half2(w.y >> 16);
        const float2 e0 = __half22float2(p0), e1 = __half22float2(p1), e2 = __half22float2(p2), e3 = __half22float2(p3);
        c[0] = e0.x; c[1] = e0.y; c[2] = e1.x; c[3] = e1.y; c[4] = e2.x; c[5] = e2.y; c[6] = e3.x; c[7] = e3.y;
    } else {
        uint2 cw0, cw1;
        if (cb_shared) { const uint4 q = *(const uint4 *)(cbs + (size_t)v * 16u); cw0.x = q.x; cw0.y = q.y; cw1.x = q.z; cw1.y = q.w; }
        else { cw0 = *(const uint2 *)(cbs + (size_t)v * 16u); cw1 = *(const uint2 *)(cbs + (size_t)v * 16u + 8u); }
        __half2 h0, h1, h2, h3; memcpy(&h0, &cw0.x, 4); memcpy(&h1, &cw0.y, 4); memcpy(&h2, &cw1.x, 4); memcpy(&h3, &cw1.y, 4);
        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
        c[0] = f0.x; c[1] = f0.y; c[2] = f1.x; c[3] = f1.y; c[4] = f2.x; c[5] = f2.y; c[6] = f3.x; c[7] = f3.y;
    }
}
/* 8 元素乘加。★这一个式子决定"与 09-16 定版逐位同"★: 单 token 核与分组核都只经这里, 谁也不许另抄一份
 * (fast-math 下同一组乘加换个写法, 编译器的 FMA 合并方式就变, 累加序跟着变 —— 09-16 在 GEMV 核上实撞过)。
 * 一个 uint32 装两个 bf16: 低半是第 2k 个元素, 高半是第 2k+1 个。补零还原成 f32。 */
__device__ __forceinline__ static float v41_vq_dot8_cw(const float *c, uint4 xw) {
    return c[0] * __uint_as_float(xw.x << 16) + c[1] * __uint_as_float(xw.x & 0xffff0000u)
         + c[2] * __uint_as_float(xw.y << 16) + c[3] * __uint_as_float(xw.y & 0xffff0000u)
         + c[4] * __uint_as_float(xw.z << 16) + c[5] * __uint_as_float(xw.z & 0xffff0000u)
         + c[6] * __uint_as_float(xw.w << 16) + c[7] * __uint_as_float(xw.w & 0xffff0000u);
}
template <int FP8>
__device__ __forceinline__ static float v41_vq_dot8(uint32_t v, uint4 xw, const uint8_t *cbs, int cb_shared) {
    float c[8];
    v41_vq_cw<FP8>(v, cbs, cb_shared, c);
    return v41_vq_dot8_cw(c, xw);
}
/* f32(已在 bf16 格点) → 打包成 bf16。一线程一元素, 每层一次, 5120 个元素, 可忽略。 */
__global__ static void v41_vq_xpack_kernel(uint16_t *dst, const float *src, uint64_t n) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
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
/* 一块位流在寄存器里的样子: lane j 持有块内第 j / j+32 / j+64 个字; ex = 位平面那一块的第 lane 个字(只 lane<8 有效)。 */
typedef struct { uint32_t w0, w1, w2, ex; } v41_vq_blk;
/* 主流位宽: v3 恒 12(高位走位平面), v2 就是 NBIT。编译期常量。 */
template <int NBIT, int V3> __device__ __forceinline__ static constexpr uint32_t v41_vq_mbit() { return V3 ? 12u : (uint32_t)NBIT; }
template <int NBIT, int V3>
__device__ __forceinline__ static const uint32_t *v41_vq_row_ptr(const v41_vq_mat &m, uint32_t r) {
    return (const uint32_t *)(m.ix + (((uint64_t)r * m.nidx_row * v41_vq_mbit<NBIT, V3>()) >> 3));
}
/* 位平面的行首(只 v3 13 位层): 每行 (nidx_row+7)/8 字节 */
__device__ __forceinline__ static const uint32_t *v41_vq_ext_ptr(const v41_vq_mat &m, uint32_t r) {
    return (const uint32_t *)(m.ex + (size_t)r * ((m.nidx_row + 7u) >> 3));
}
/* 装一块: 只读本块真有的字(尾块 48 个), 越界的 lane 不发读 —— 不靠载荷尾巴的 8 B 零垫。
 * EXT=1 再多一条 LDG.32: 一块 256 个索引的位平面 = 32 B = 8 个字, lane j<8 各拿一个(nwx = 本块真有的字数)。 */
template <int EXT>
__device__ __forceinline__ static v41_vq_blk v41_vq_blk_load(const uint32_t *blk, uint32_t nw, const uint32_t *bex, uint32_t nwx) {
    const uint32_t lane = threadIdx.x & 31u;
    v41_vq_blk b; b.w0 = 0u; b.w1 = 0u; b.w2 = 0u; b.ex = 0u;
    if (lane < nw) b.w0 = blk[lane];
    if (lane + 32u < nw) b.w1 = blk[lane + 32u];
    if (lane + 64u < nw) b.w2 = blk[lane + 64u];
    if (EXT && bex && lane < nwx) b.ex = bex[lane];
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
 *   块读没提前够, 更像是 32 个 warp 同拍往 L1TEX 队列里塞 16 个波前/轮(码本查表 11.7 + 激活 4)的排队延迟。
 *   ⇒ 所以 v3 的位平面只肯占 **1 个**寄存器(b.ex), 而且是与主流同一条块读路径上顺手带的。 */
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ static v41_vq_blk v41_vq_row_first_blk(const v41_vq_mat &m, uint32_t r) {
    const uint32_t R = m.nidx_row >> 5, nr = (R < 8u ? R : 8u);
    return v41_vq_blk_load<EXT>(v41_vq_row_ptr<NBIT, V3>(m, r), nr * v41_vq_mbit<NBIT, V3>(),
                                EXT ? v41_vq_ext_ptr(m, r) : NULL, nr);
}
/* 一块的最多 8 轮: 第 k 轮 lane j 取索引 32(b+k)+j, 两条 shfl 拿字、funnelshift 取 12 位、查表乘加。乘加次序与旧核一字不差。
 * ★判负存档(2026-09-18)★ "轮数做模板参数变直线代码(1~7 轮尾块各一份实例 + switch)": 想让编译器把下一轮的 shfl/查表交错进
 * 本轮的乘加链。gateup 6.68 / down 3.92 —— 噪声内, 且 gateup 顶到 64 寄存器带 32 B spill。运行期轮数 + 每轮一条分支就够。
 * EXT=1: 第 k 轮的第 13 位 = 块内第 k 个字(lane k 持有)的第 lane 位 ⇒ 一条 shfl + 一次移位与, 零额外访存。 */
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ static void v41_vq_blk_rounds(const v41_vq_blk &cur, uint32_t rounds, uint32_t a, uint32_t sh, uint32_t imsk,
                                                         const uint32_t *xb, const uint8_t *cbs, int cb_shared, float &acc) {
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint4 xa = *(const uint4 *)(xb + (size_t)(k * 32u + lane) * 4u);   /* 8 个 bf16 = 16 B, L1 命中 */
        const uint32_t f = k * MB;                                       /* 本轮首字(编译期常量) */
        const uint32_t lo_r = ((f & 31u) > 32u - MB) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - MB) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v41_vq_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int)(f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v41_vq_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int)(f + a + 1u));
        uint32_t v = __funnelshift_r(lo, hi, sh) & (EXT ? 0xFFFu : imsk);
        if (EXT) {
            const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int)k);
            v |= ((exw >> lane) & 1u) << 12;
        }
        acc += v41_vq_dot8<V3>(v, xa, cbs, cb_shared);
    }
}
/* 一 warp 算一行与激活 x 的点积: 索引 j 归 lane j%32, 一轮 32 个 lane 覆盖 32 个索引。
 * 返回 增益 × Σ(已 warp 规约)。NBIT = 码本号位宽(4096 词 = 12; v3 13 位层 = 13, 但主流仍 12 位 + 位平面)。
 * carry 进来是本行首块, 出去是 next 行的首块(next 为 NULL 就是空块); 每块的字在上一块开算之前就发出去。
 * nextex = next 行的位平面行首(EXT=1 时必须与 next 配对给, 不然下一行的第 13 位全是 0 —— 那是"不报错的假权重")。 */
template <int NBIT, int V3, int EXT>
__device__ __forceinline__ static float v41_vq_row_dot(const v41_vq_mat &m, uint32_t r, const uint32_t *xs, const uint8_t *cbs, int cb_shared,
                                                       v41_vq_blk *carry, const uint32_t *next, const uint32_t *nextex) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t R = m.nidx_row >> 5;                                  /* 轮数: gateup 20, down 9 */
    constexpr uint32_t MB = v41_vq_mbit<NBIT, V3>();
    const uint32_t a = (lane * MB) >> 5, sh = (lane * MB) & 31u;         /* 本 lane 在一轮 MB 个字里的字号与位移 */
    const uint32_t *row = v41_vq_row_ptr<NBIT, V3>(m, r);
    const uint32_t *rex = EXT ? v41_vq_ext_ptr(m, r) : NULL;
    v41_vq_blk cur = *carry;
    float acc = 0.f;
    for (uint32_t b = 0; b < R; b += 8u) {
        v41_vq_blk nxt;
        if (b + 8u < R) {
            const uint32_t rem = R - b - 8u, nr = (rem < 8u ? rem : 8u);
            /* ★位平面的行内推进是"每组一个字", 不是"每块一个字"★(2026-09-22 修, 这是 v3 解码路 PPL 高 42% 的真因):
             * b 是 32 索引一组的组号, 一块 = 8 组 = 256 个索引 = 位平面 8 个字 ⇒ 下一块的字从 rex + (b+8) 开始。
             * 原来写的是 rex + (b+8)/8(块号), 于是每行只有第一块的第 13 位是对的, 后面的块读到第 1、2 个字 ——
             * 索引错 ±4096 = 换了个码字, 权重整片错。主流指针同一行就是 * MB(组号 × 每组字数), 两处口径要一致。
             * 只打 v3 的 13 位层(EXT=1): 12 位层与全部 v2 载荷没有位平面, 所以老模型看不出毛病。
             * 实撞代价: 同一条 4000 token 序列, 预填路(块 16)PPL 5.75 / 解码路(块 8)8.19; 老 fp4 模型两路只差 1%。
             * 而判决尺全走预填路 ⇒ 指标一路全绿, 产品每一个 token 都吃着这份错权重。 */
            nxt = v41_vq_blk_load<EXT>(row + (size_t)(b + 8u) * MB, nr * MB, EXT ? rex + (b + 8u) : NULL, nr);
        }
        else if (next) { const uint32_t nr = (R < 8u ? R : 8u); nxt = v41_vq_blk_load<EXT>(next, nr * MB, nextex, nr); }
        else { nxt.w0 = 0u; nxt.w1 = 0u; nxt.w2 = 0u; nxt.ex = 0u; }
        /* ★判负存档(2026-09-18)★ "再往后一块用 prefetch.global.L2 先送进 L2(12 个 lane 各一扇区, 零寄存器)": 依据是带载 DRAM
         * 一跳 1.2~1.4 µs(mem_ceiling ⑦; 空载 0.33), 怕寄存器提前一块(8 轮)盖不住。实测 gateup 6.93 / down 3.78, 噪声内 ⇒
         * 位流的到达不是剩余等待的来源。 */
        v41_vq_blk_rounds<NBIT, V3, EXT>(cur, (R - b < 8u) ? R - b : 8u, a, sh, m.imsk, xs + (size_t)b * 32u * 4u, cbs, cb_shared, acc);
        cur = nxt;
    }
    *carry = cur;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    /* 行增益在行尾才取(试过提到行首: 多占寄存器, gateup 6.72 → 7.11, 退回) */
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    return acc * __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
}
