/* cuda_vq_fused2_0.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * fused2 特化高速 VQ 解码路: 两个 kernel 共用的 device 辅助函数(09-05 从 fused2_1 拆出, 行数闸)。
 *
 * ===== 09-05 重写账(冠军态 nsys 定罪: fused2 因 w2=512 词被闸回老 fused, 379+257 µs/层) =====
 * 数值逐位不变的前提下, 一共动了四刀, 每刀都有 nsys 读数:
 *  ① 激活 float4 读 xs[(lane*NIDX+k)*4]: lane 步长 512/256 B, 32 lane 全落同一组 bank ⇒ 128-bit
 *     读 32 个 wavefront(理想 4)。修: 装入时按 p=(g%NIDX)*32+g/NIDX 转置, 第 k 步 lane 连续。
 *  ② 码本两次 32-bit 读只落偶 bank ⇒ 一次 uint2 读。
 *  ③ 位流窗口 bw[wi] 的 wi 含运行期 misal ⇒ 数组进 local memory。修: 先整段 funnelshift 去错位,
 *     之后每个索引的字/位偏移全是编译期常量。
 *     ①②③ + 512 词 w2 放行 + down 全 warp 干活: 379+257 → 158+72 µs/层。
 *  ④ 真凶=访存指令形态: 每 lane 各读自己 36 B 段, 一条 warp 级 4 B load 横跨 36 个 sector,
 *     一行 10 条指令 = 360 个 L1 wavefront, LSU 吞吐成墙(转置/寄存器/shared 减流三刀 158→161→161
 *     纹丝不动的原因)。修: 整行按 lane 连续合并读(每条指令 128 B = 4 sector)进 warp 私有 shared
 *     暂存, 各 lane 再从暂存取段(步长 9 字 gcd(9,32)=1 ⇒ bank 无冲突)。
 *     —— 实测 ④ 与寄存器预取环(⑤, 每 warp 提前 4 行 9 KB 在飞)、16 行/warp 减半开场白、
 *     __launch_bounds__ 压寄存器、激活段进寄存器, 五种改法 gateup 全在 140~161 µs 抖动;
 *     消融(去掉点积只留访存)144 µs = 196 GB/s ⇒ 是这台机器该占用率下的访存形态上限
 *     (mem_ceiling ④ 同形态: 满占用 238 GB/s, 2 block/SM 223)。最终取最简形态(各 lane 直读
 *     自己段)+ 实测最优参数(8 行/warp, gateup 两行一趟/down 四行一趟): 140 + 70 µs/层。 */
/* 分发端 grid 用同一常量算(fused2_3): 每 block 4 warp × 8 行。 */
#define DS4_VQ2_ROWS_PER_WARP 8u
#define DS4_VQ2_ROWS_PER_BLOCK (4u * DS4_VQ2_ROWS_PER_WARP)

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

/* 激活向量装入 shared 并转置(见 ①): 组 g → 位置 (g%NIDX)*32 + g/NIDX。 */
template <uint32_t NIDX>
__device__ __forceinline__ static void vq2_load_vec_perm(float4 *dst4, const float *src, uint32_t cols) {
    const uint32_t ngrp = cols >> 2u;
    for (uint32_t g = threadIdx.x; g < ngrp; g += blockDim.x)
        dst4[(g % NIDX) * 32u + g / NIDX] = *(const float4 *)&src[g << 2u];
}

/* 码本(payload+16 起 n*4 halfs)按 8 B/词装入 shared。payload 由 vq_pack 紧凑排布、格式上无对齐
 * 保证, 但各段字节数(头 16/码本 4096/行尺度 rows*2/位流 rows*1152|576)全是 8 的倍数, 实际 8 B
 * 对齐 ⇒ 走 uint2 读; 万一不对齐退回逐字节拼。 */
__device__ __forceinline__ static void vq2_load_cb(uint2 *dst, const uint8_t *cbg, uint32_t nwords) {
    if (((uintptr_t)cbg & 7u) == 0u) {
        const uint2 *src = (const uint2 *)cbg;
        for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) dst[i] = __ldg(src + i);
        return;
    }
    for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
        const uint8_t *p = cbg + (size_t)i * 8u;
        uint2 q;
        q.x = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        q.y = (uint32_t)p[4] | ((uint32_t)p[5] << 8) | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        dst[i] = q;
    }
}

/* 位流行段: lane 覆盖 NIDX 个索引(9bit 段长 NIDX*9/8 B, 8bit NIDX B), 对齐窗口 NW 个 u32
 * (≥ 段长+3 B 错位), 读完整段右移 misal*8 位去错位(见 ③)。
 * 09-05 消融定谳: 这条"各 lane 直读自己段"的形态与"整行合并读进 warp 暂存再取段"(④)、
 * 寄存器预取环(⑤)三者实测同速(gateup 140~161 µs 内抖动), 且去掉点积只留访存仍 144 µs
 * ⇒ 该核在 2 block/SM 占用下的访存效率 ~200 GB/s 是这台 GB10 的形态上限(mem_ceiling ④:
 * 同形态满占用 238, 压到 2 block/SM 223)。取最简形态 + 实测最优参数(每 warp 8 行, 两行一趟)。 */
template <uint32_t NIDX, uint32_t NW>
__device__ __forceinline__ static void vq2_load9(
        const uint8_t *__restrict__ ix, uint32_t row, uint32_t lane, uint32_t cols, uint32_t *bw) {
    const uint32_t row_bytes = ((cols >> 2u) * 9u) >> 3u;
    const uint8_t *seg = ix + (size_t)row * row_bytes + (size_t)lane * (NIDX * 9u / 8u);
    const uint32_t misal = (uint32_t)((uintptr_t)seg & 3u);
    const uint32_t *wp = (const uint32_t *)(seg - misal);
    #pragma unroll
    for (uint32_t k = 0; k < NW; k++) bw[k] = __ldg(wp + k);
    #pragma unroll
    for (uint32_t k = 0; k + 1u < NW; k++) bw[k] = __funnelshift_r(bw[k], bw[k + 1u], misal * 8u);
}
template <uint32_t NIDX, uint32_t NW>
__device__ __forceinline__ static void vq2_load8(
        const uint8_t *__restrict__ ix, uint32_t row, uint32_t lane, uint32_t cols, uint32_t *bw) {
    const uint32_t nidx_row = cols >> 2u;
    const uint8_t *seg = ix + (size_t)row * nidx_row + (size_t)lane * NIDX;
    const uint32_t misal = (uint32_t)((uintptr_t)seg & 3u);
    const uint32_t *wp = (const uint32_t *)(seg - misal);
    #pragma unroll
    for (uint32_t k = 0; k < NW; k++) bw[k] = __ldg(wp + k);
    #pragma unroll
    for (uint32_t k = 0; k + 1u < NW; k++) bw[k] = __funnelshift_r(bw[k], bw[k + 1u], misal * 8u);
}

/* 索引抽取(已去错位窗口): 9bit 走 funnelshift, 8bit 直取字节; k 编译期 ⇒ 全在寄存器。 */
template <uint32_t NW, uint32_t NBIT>
__device__ __forceinline__ static uint32_t vq2_idx(const uint32_t *bw, uint32_t k) {
    if (NBIT == 9u) {
        const uint32_t bit = k * 9u, wi = bit >> 5u, sh = bit & 31u;
        return __funnelshift_r(bw[wi], bw[wi + 1u < NW ? wi + 1u : NW - 1u], sh) & 511u;
    }
    return (bw[k >> 2u] >> ((k & 3u) * 8u)) & 255u;
}

/* 一段乘加: 码本词 × 激活 float4。表达式与旧 dot9/dot8 逐字相同 ⇒ 每 lane 累加序不变。 */
__device__ __forceinline__ static float vq2_fma4(float acc, const uint2 cq, const float4 xv) {
    const float2 f01 = __half22float2(*(const half2 *)&cq.x);
    const float2 f23 = __half22float2(*(const half2 *)&cq.y);
    return acc + (f01.x * xv.x + f01.y * xv.y + f23.x * xv.z + f23.y * xv.w);
}

/* 一段点积(已去错位窗口): 索引 → 码本 uint2(shared) × 转置激活 float4(shared)。 */
template <uint32_t NIDX, uint32_t NW, uint32_t NBIT>
__device__ __forceinline__ static float vq2_dot(
        const uint32_t *bw, const float4 *xs4, const uint2 *cb2, uint32_t lane) {
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < NIDX; k++)
        acc = vq2_fma4(acc, cb2[vq2_idx<NW, NBIT>(bw, k)], xs4[k * 32u + lane]);
    return acc;
}

__device__ __forceinline__ static float vq2_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* 行尺度 × clamp × SwiGLU × 路由权重 → mid(与旧核逐字同式, 只是抽成函数) */
__device__ __forceinline__ static void vq2_store_mid(
        float *mid_out, uint64_t pair, uint32_t MID, uint32_t m, float g, float u,
        const __half *gr1, const __half *gr3, float clamp, float w) {
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
