/* cuda_q4k_dot.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * Q4_K 稠密 tile 核族的块点积/激活访问器/stage 助手(2026-09-07 从 cuda_q4k_tile 拆出, 供 cuda_q4k_multi 与 cuda_q4k_tile 共用)。
 *
 * 【依据: gguf-tools/bench/q4k_gemv_bench.cu 按真实形状的微基准, 现核 vs 候选逐位/µs 对拍】
 * 半块/lane(两 lane 分一块, j0 运行期)的 dot 把尺度解码变成运行期分支, ALU 翻倍, 短行更亏;
 * 整块/lane + 一 warp 同时 R=32/BLOCKS 行(4 块行 8 行/warp, 8 块行 4 行, 16 块行 2 行):
 * shexp_down 201→227 GB/s, q_a+kv 210→276, shexp gate/up 230→242, q_b 225→229~235。
 * 数值: 归约结合律变(半块两半相加 → 整块) = 容差级(ULP), 判官 = parity 五指标同带(09-06 K3/K4 先例)。
 * 激活口径两种, 与 09-07 前各矩阵一致(不改任何激活精度): q8_K(每 256 值一尺度+bsums: q_b/q_a/kv/shexp/
 * 输出头) 与 q8_0(每 32 值一尺度: attn_output a/b)。tile 结构相同, 只换块 dot(ACT 模板)。
 * [记录] out_b 的 q8_0 dot 只到 176~180 GB/s(q8_K 230); 换 q8_K 激活省 ~1 ms/token 但 Σmin −0.0014(链 q4kt), 精度归用户裁。
 * 布局: 一 warp 一 tile = R 行连续 4608 B, 整 tile 由 32 lane 连续 16 B 读进 shared(8 warp 块 36 KB,
 * 3 块/SM), lane = (行 lane/BLOCKS, 块 lane%BLOCKS) 各算一整块, 段内 shuffle 树归约。 */
/* q8_0 激活 × q4_K 整块(与 09-07 前 attn_output 路逐字同; 09-05 删掉非 dp4a 标量路径的 shared 源版) */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem(
        const cuda_block_q4_K *x, const int8_t *xq, const float *xs) {
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* 多 token 时把每 token 重复的两样东西拿出点积: 权重块的尺度/min 解码(8 次 get_scale_min, 与 token 无关)和激活
 * 子块和 Σq8(每 (token,块,子块) 64 个 dp4a, 与权重无关)。表达式与 dev_dot_q4_K_q8_0x8_smem 逐字同(两链 acc0/acc1,
 * 同乘同减同序), 只是操作数来自预解码 ⇒ 逐位同。 */
struct q4k_wblk { float d, dmin; uint8_t sc[8], mn[8]; };
__device__ __forceinline__ static void q4k_wblk_decode(const cuda_block_q4_K *x, q4k_wblk *o) {
    o->d = dev_f16_to_f32(x->d); o->dmin = dev_f16_to_f32(x->dmin);
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) dev_q4_K_get_scale_min(j, x->scales, &o->sc[j], &o->mn[j]);
}
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_pre(const cuda_block_q4_K *x, const q4k_wblk &wb,
                                                                const int8_t *xq, const float *xs, const int32_t *s8) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, xq + j * 32u, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, xq + (j + 1u) * 32u, 4);
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s8[j]);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s8[j + 1u]);
    }
    return acc0 + acc1;
}
/* 同上但 Σq8 在 lane 内现算(8 个 dp4a/子块): 给激活块在行间复用少的核(分组 out_a)用 —— 全表预算要每 block 读全部激活
 * (4 token × 8 组 × 32 KB = 128 KB/block, 实测 182 → 215 µs 反慢), 这里只省权重侧的尺度解码。表达式仍同式同序。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_wpre(const cuda_block_q4_K *x, const q4k_wblk &wb,
                                                                 const int8_t *xq, const float *xs) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u, *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s80);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s81);
    }
    return acc0 + acc1;
}
/* 激活访问器: at(blk_off) 平移到第 blk_off 个 256 值块, dot(w, b) 算本行第 b 块 */
struct q4k_act_q8K {
    const cuda_block_q8_K *x;
    __device__ __forceinline__ q4k_act_q8K at(uint64_t blk_off) const { q4k_act_q8K a; a.x = x + blk_off; return a; }
    __device__ __forceinline__ float dot(const cuda_block_q4_K *w, uint32_t b) const { return dev_dot_q4_K_q8_K_block(w, x + b); }
};
struct q4k_act_q8_0 {
    const int8_t *xq; const float *xs;
    __device__ __forceinline__ q4k_act_q8_0 at(uint64_t blk_off) const { q4k_act_q8_0 a; a.xq = xq + blk_off * 256u; a.xs = xs + blk_off * 8u; return a; }
    __device__ __forceinline__ float dot(const cuda_block_q4_K *w, uint32_t b) const { return dev_dot_q4_K_q8_0x8_smem(w, xq + (uint64_t)b * 256u, xs + (uint64_t)b * 8u); }
};

/* 段内树归约: 一行占 BLOCKS 个连续 lane(与 32 lane 补零树同形) */
template <uint32_t BLOCKS>
__device__ __forceinline__ static float q4k_seg_sum(float acc) {
    #pragma unroll
    for (uint32_t off = BLOCKS / 2u; off > 0u; off >>= 1u) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

/* 一 tile: stage R 行 → lane 各算一整块 → 段内归约; 返回本 lane 段首的行和 */
template <uint32_t BLOCKS, typename ACT>
__device__ __forceinline__ static float q4k_tile_dot(uint4 *my, const char *w_tile, const ACT &x, uint32_t lane) {
    constexpr uint32_t R = 32u / BLOCKS;
    constexpr uint32_t n16 = R * BLOCKS * 9u;
    const uint4 *src16 = (const uint4 *)w_tile;
    #pragma unroll
    for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
    __syncwarp();
    const uint32_t rin = lane / BLOCKS, b = lane % BLOCKS;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my + rin * BLOCKS + b;
    float acc = x.dot(wr, b);
    acc = q4k_seg_sum<BLOCKS>(acc);
    __syncwarp();   /* 下一 tile 复写 my 前全 lane 算完 */
    return acc;
}

/* 解码单 token 核的激活也进 shared(09-07 微基准 shexp_gate 203 → 233 GB/s): BLOCKS 块 ≤ 4.7 KB 放权重 stage 区之后, <48 KB 免属性 */
__device__ __forceinline__ static const cuda_block_q8_K *q4k_stage_x1(uint4 *area, const cuda_block_q8_K *xq, uint32_t blocks) {
    uint32_t *dst = (uint32_t *)area;
    const uint32_t *src = (const uint32_t *)xq;
    const uint32_t n_words = blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
    for (uint32_t i = threadIdx.x; i < n_words; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    return (const cuda_block_q8_K *)area;
}

/* ===== 多 token 复用: 权重块 nibble 只解一次(09-07) =====
 * 多 token 核每 lane 一整块权重、逐 token 串行点积, 剖面定罪是算力/ILP 界而非字节界(4 token 的 grouped/out_b 是单 token 的 2×,
 * 权重字节相同)。dev_dot_q4_32 每 token 都重做 64 次 (load, >>, &) 取 nibble —— 只依赖权重。这里把一块 256 个 nibble 一次
 * 展开成 64 个 int8x4 字(寄存器), 逐 token 只剩 dp4a。整数运算无舍入、dp4a 序与 dev_dot_q4_32 逐字同(i=0..28 步 4 同链),
 * 浮点表达式与原 dot 同式同序 ⇒ 逐位同(verify 与解码同轨不受影响)。 */
struct q4k_wnib { uint32_t v[64]; };   /* v[g*16 + i] = 第 g 组(32 B)第 i 字低 nibble; v[g*16 + 8 + i] = 高 nibble */
__device__ __forceinline__ static void q4k_wnib_unpack(const cuda_block_q4_K *x, q4k_wnib *o) {
    #pragma unroll
    for (uint32_t g = 0; g < 4u; g++) {
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const int32_t w = *(const int32_t *)(x->qs + g * 32u + i * 4u);
            o->v[g * 16u + i] = (uint32_t)(w & 0x0f0f0f0f);
            o->v[g * 16u + 8u + i] = (uint32_t)((w >> 4) & 0x0f0f0f0f);
        }
    }
}
__device__ __forceinline__ static int32_t dev_dot_q4_32_pre(const uint32_t *v, const int8_t *q8) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 8u; i++) sum = __dp4a((int32_t)v[i], *(const int32_t *)(q8 + i * 4u), sum);
    return sum;
}
/* = dev_dot_q4_K_q8_K_block(cuda_api_moe_corr_1) 同式同序: 双链 isum/summs, 尾式 y.d*xd*(isum) − y.d*xmin*(summs) */
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_block_nib(const q4k_wblk &wb, const q4k_wnib &nb, const cuda_block_q8_K *y) {
    int isum0 = 0, isum1 = 0, summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        summs0 += (int)wb.mn[j] * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)wb.mn[j + 1u] * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t *v0 = nb.v + (j >> 1u) * 16u;
        isum0 += (int)wb.sc[j] * dev_dot_q4_32_pre(v0, y->qs + j * 32u);
        isum1 += (int)wb.sc[j + 1u] * dev_dot_q4_32_pre(v0 + 8u, y->qs + (j + 1u) * 32u);
    }
    return y->d * wb.d * (float)(isum0 + isum1) - y->d * wb.dmin * (float)(summs0 + summs1);
}
/* = dev_dot_q4_K_q8_0x8_pre(Σq8 查表版)同式同序 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_pre_nib(const q4k_wblk &wb, const q4k_wnib &nb,
                                                                    const int8_t *xq, const float *xs, const int32_t *s8) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t *v0 = nb.v + (j >> 1u) * 16u;
        const int32_t dot0 = dev_dot_q4_32_pre(v0, xq + j * 32u);
        const int32_t dot1 = dev_dot_q4_32_pre(v0 + 8u, xq + (j + 1u) * 32u);
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s8[j]);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s8[j + 1u]);
    }
    return acc0 + acc1;
}
/* = dev_dot_q4_K_q8_0x8_wpre(Σq8 lane 内现算)同式同序 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_wpre_nib(const q4k_wblk &wb, const q4k_wnib &nb,
                                                                     const int8_t *xq, const float *xs) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t *v0 = nb.v + (j >> 1u) * 16u;
        const int8_t *q8a = xq + j * 32u, *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32_pre(v0, q8a);
        const int32_t dot1 = dev_dot_q4_32_pre(v0 + 8u, q8b);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s80);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s81);
    }
    return acc0 + acc1;
}

/* ---- 发射侧公共量: 块数只支持 4/8/16(R=8/4/2; 引擎形状 q_b 4, shexp_down 8, q_a/kv/shexp/out_a/输出头 16),
 * 其它块数由调用方走整块/lane 的 warp 核(cuda_qk_warp_2 matmul_q4_K_warp_kernel)。 ---- */
#define Q4K_TILE_SHM ((size_t)8u * 32u * 9u * sizeof(uint4))   /* 8 warp × 4608 B */
static unsigned q4k_tile_grid(uint32_t tiles) {
    unsigned g = (tiles + 7u) / 8u;
    if (g > ds4_grid_cap()) g = ds4_grid_cap();
    return g ? g : 1u;
}
static int q4k_tile_supported(uint32_t blocks, uint32_t rows) {
    return (blocks == 4u || blocks == 8u || blocks == 16u) && rows % (32u / blocks) == 0u;
}

/* ===== 激活 16 B 向量读(09-07 晚): 多 token 核每 token 每块 64 次 4 B 标量 load 读激活(grouped 走全局 = L2 延迟界;
 * STAGE 核走 shared 也是 64 次 LDS.32), 剖面定罪 4 token 是单 token 的 2× 而权重字节相同。改 int4 读: 同 8 个字同序喂 dp4a,
 * 整数结果逐位同。q8_K 块结构 {float d; qs[256]; bsums[16]} 的 qs 偏移 4 不对齐 ⇒ STAGE 时拆成 [qs 256 B 对齐][d][bsums] 三段。 */
__device__ __forceinline__ static int32_t dev_dot_q4_32_pre_v(const uint32_t *v, const int8_t *q8 /* 16 B 对齐 */) {
    const int4 a = *(const int4 *)q8, b = *(const int4 *)(q8 + 16);
    int32_t sum = 0;
    sum = __dp4a((int32_t)v[0], a.x, sum); sum = __dp4a((int32_t)v[1], a.y, sum);
    sum = __dp4a((int32_t)v[2], a.z, sum); sum = __dp4a((int32_t)v[3], a.w, sum);
    sum = __dp4a((int32_t)v[4], b.x, sum); sum = __dp4a((int32_t)v[5], b.y, sum);
    sum = __dp4a((int32_t)v[6], b.z, sum); sum = __dp4a((int32_t)v[7], b.w, sum);
    return sum;
}
__device__ __forceinline__ static int32_t dev_sum_q8_32_v(const int8_t *q8 /* 16 B 对齐 */) {
    const int4 a = *(const int4 *)q8, b = *(const int4 *)(q8 + 16);
    int32_t s = 0;
    s = __dp4a(0x01010101, a.x, s); s = __dp4a(0x01010101, a.y, s); s = __dp4a(0x01010101, a.z, s); s = __dp4a(0x01010101, a.w, s);
    s = __dp4a(0x01010101, b.x, s); s = __dp4a(0x01010101, b.y, s); s = __dp4a(0x01010101, b.z, s); s = __dp4a(0x01010101, b.w, s);
    return s;
}
/* = dev_dot_q4_K_q8_K_block 同式同序, 激活三段式(qs 对齐) */
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_split_nib(const q4k_wblk &wb, const q4k_wnib &nb,
                                                                    const int8_t *qs, float yd, const int16_t *bsums) {
    int isum0 = 0, isum1 = 0, summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        summs0 += (int)wb.mn[j] * (int)(bsums[2u * j] + bsums[2u * j + 1u]);
        summs1 += (int)wb.mn[j + 1u] * (int)(bsums[2u * j + 2u] + bsums[2u * j + 3u]);
        const uint32_t *v0 = nb.v + (j >> 1u) * 16u;
        isum0 += (int)wb.sc[j] * dev_dot_q4_32_pre_v(v0, qs + j * 32u);
        isum1 += (int)wb.sc[j + 1u] * dev_dot_q4_32_pre_v(v0 + 8u, qs + (j + 1u) * 32u);
    }
    return yd * wb.d * (float)(isum0 + isum1) - yd * wb.dmin * (float)(summs0 + summs1);
}
/* = dev_dot_q4_K_q8_0x8_pre(Σq8 查表)同式同序, 向量读 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_pre_nib_v(const q4k_wblk &wb, const q4k_wnib &nb,
                                                                      const int8_t *xq, const float *xs, const int32_t *s8) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t *v0 = nb.v + (j >> 1u) * 16u;
        const int32_t dot0 = dev_dot_q4_32_pre_v(v0, xq + j * 32u);
        const int32_t dot1 = dev_dot_q4_32_pre_v(v0 + 8u, xq + (j + 1u) * 32u);
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s8[j]);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s8[j + 1u]);
    }
    return acc0 + acc1;
}
/* = dev_dot_q4_K_q8_0x8_wpre(Σq8 lane 内现算)同式同序, 向量读 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_wpre_nib_v(const q4k_wblk &wb, const q4k_wnib &nb,
                                                                       const int8_t *xq, const float *xs) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        const uint32_t *v0 = nb.v + (j >> 1u) * 16u;
        const int8_t *q8a = xq + j * 32u, *q8b = xq + (j + 1u) * 32u;
        const int32_t dot0 = dev_dot_q4_32_pre_v(v0, q8a);
        const int32_t dot1 = dev_dot_q4_32_pre_v(v0 + 8u, q8b);
        const int32_t s80 = dev_sum_q8_32_v(q8a), s81 = dev_sum_q8_32_v(q8b);
        acc0 += xs[j] * (wb.d * (float)wb.sc[j] * (float)dot0 - wb.dmin * (float)wb.mn[j] * (float)s80);
        acc1 += xs[j + 1u] * (wb.d * (float)wb.sc[j + 1u] * (float)dot1 - wb.dmin * (float)wb.mn[j + 1u] * (float)s81);
    }
    return acc0 + acc1;
}
/* q8_K 激活拆三段进 shared: [nb×256 B qs(16 B 对齐)][nb×float d][nb×16 int16 bsums]; 占 nb×292 B(与结构体同量) */
struct q4k_q8K_split { const int8_t *qs; const float *d; const int16_t *bs; };
__device__ __forceinline__ static q4k_q8K_split q4k_stage_q8K_split(void *area, const cuda_block_q8_K *xq, uint32_t nb) {
    int8_t *qs = (int8_t *)area; float *d = (float *)(qs + (uint64_t)nb * 256u); int16_t *bs = (int16_t *)(d + nb);
    const uint32_t nw = nb * 64u;
    for (uint32_t i = threadIdx.x; i < nw; i += blockDim.x) {   /* qs: 块 i/64 的第 i%64 字(源偏移 4, 4 B 对齐读) */
        const uint32_t blk = i >> 6u, wi = i & 63u;
        ((uint32_t *)qs)[i] = *(const uint32_t *)(xq[blk].qs + wi * 4u);
    }
    for (uint32_t i = threadIdx.x; i < nb * 8u; i += blockDim.x) {   /* bsums: 块 i/8 的第 i%8 字(16 int16 = 8 字, 源偏移 260 对齐 4) */
        const uint32_t blk = i >> 3u, wi = i & 7u;
        ((uint32_t *)bs)[i] = *(const uint32_t *)((const char *)(xq + blk) + 260u + wi * 4u);
    }
    for (uint32_t i = threadIdx.x; i < nb; i += blockDim.x) d[i] = xq[i].d;
    __syncthreads();
    q4k_q8K_split r; r.qs = qs; r.d = d; r.bs = bs; return r;
}
/* 两波网格: 有 stage 前奏(激活搬 shared + Σq8)的核, 网格封顶到"驻留 block 数 × SM 数 × 2", 每 warp 多跑行把前奏摊薄
 * (rows_q8_0 41 KB 激活/块 × 512 块 = 19 MB 搬运 = 权重本身的字节量, 这就是 4 token 2× 的来源之一)。 */
static unsigned q4k_grid_wave(size_t shm_bytes, unsigned want, unsigned waves) {
    static int n_sm = 0, smem_sm = 0;
    if (!n_sm) {
        int dev = 0; (void)cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n_sm <= 0) n_sm = 48;
        if (cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess || smem_sm <= 0) smem_sm = 101376;
        (void)cudaGetLastError();
    }
    unsigned per = (unsigned)((size_t)smem_sm / (shm_bytes + 1024u));
    if (per < 1u) per = 1u; if (per > 4u) per = 4u;
    /* 封顶两波, 不压到一波: 一波时全部 block 同时做前奏、没有权重流在飞(链 30 实测 128 → 48 块的 shexp 29 → 43 µs 反退);
     * 两波让后一波的前奏叠在前一波的权重流里。小网格(≤ 两波)原样不动。 */
    const unsigned cap = waves * (unsigned)n_sm * per;   /* rows_q8_0 两波(512 → 96); tile_multi 只封 >4 波(输出头 384 → 192) */
    return want < cap ? want : cap;
}
