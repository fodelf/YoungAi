/* cuda_v41_4.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 解码小批(n ≤ 8)融合核(2026-09-12 战役 P4)。
 *
 * 为什么: P2c 首跑解码 1.24 t/s, 账在"每次乘法先把整个权重矩阵 dequant 成 f16 再 cuBLAS":
 *   骨架 fp4x32 0.53 B/元素 → 写 2 B + 读 2 B, 流量放大 8.5×(≈32 GB/token); 专家更糟: 6 专家×40 层的 f16 落地
 *   约 34 GB/token, 而压缩态 blob 本身只有 ~1.6 GB/token。那份 f16 只用一次就丢, 根本不该存在。
 * 这里对 n ≤ 8(解码/小批)直接从压缩态算。数值口径: 权重精确值(fp4×2^e / 码本×增益 f32 乘), 激活 bf16 格点,
 * f32 累加 —— 与 cuBLAS 路只差累加序。专家内按官方 Expert 的 bf16 边界舍: gate/up 出 bf16, h 出 bf16, down 出 bf16,
 * 再按路由权重 f32 加。
 * 三版演进(nsys 定的): ①lane 隔 32 块取整块 → 每条字节 load 跨 17 扇区, 55 GB/s; ②整 warp 逐块、lane 取块内第 l 个元素
 * → 扇区对了但小矩阵(kv 512 行)只有 64 个 block 填不满 48 个 SM, 仍 57 GB/s; ③(本版)K 维切给 ksplit 个 warp 共算一行
 * + wo_a 八组一发 + bf16 舍入并进尾巴。VQ 侧: ①码本 L1 全局 gather(每层 1.7 千万次随机 16 B, 扇区流量 566 MB/层)
 * → ②(本版)码本整块搬进 shared(64 KB/矩阵, sm_121 动态 shared 上限 227 KB), 一 block 32 行摊薄搬运。 */

/* ---- fp4x32 GEMV ----
 * grid (ceil(out_dim/(8/ksplit)), n_groups), block 256 = 8 warp; 同一行由 ksplit 个 warp 分 K 段(块号 ≡ kpart mod ksplit),
 * 各自 lane 取块内第 l 个元素(半字节 p[l/2] + 共用 scale p[16]; x 读 128 B 连续, 权重读同扇区), 段和经 shared 按固定序相加。
 * n_groups>1 = 块对角(wo_a): 第 g 组权重/输入/输出各按步长偏移。round_out: 出口直接舍 bf16(省一次 round 核)。 */
/* ★2026-09-15 段 2: 按 token 数 NT 模板特化★
 * 病因: 内层原来是 `#pragma unroll over V41_GEMV_MAX_TOK(=8)` 再用 `if (t < n_tok)` 挡掉多余的。
 * n_tok 是运行期值, 所以编译器老老实实展开 8 份带谓词的代码 —— **解码时 n_tok=1, 八分之七的指令
 * 是白跑的**。不能改成 `for (t < n_tok)` 动态循环, 因为 acc[t] 用动态下标会掉进 local memory。
 * 正解是把 token 数变成模板参数: NT=1 时内层就是一条 FFMA, acc 就是一个寄存器。
 * 实测背景: 提占用率(ksplit 目标 2048→32768)只换来 3%, 一 warp 管 4 行更是慢 2.8 倍 ⇒ 这个核
 * 既不是延迟受限也不是激活重读受限, 剩下的嫌疑就是指令数。★数值逐位同: 只是把死代码删掉。★ */
template <uint32_t NT>
__global__ static void v41_fp4x32_gemv_kernel(float *out, const uint8_t *w, const float *x, uint32_t in_dim, uint32_t out_dim,
                                              uint32_t x_stride, uint32_t out_stride, uint32_t ksplit,
                                              uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out) {
    const uint32_t g = blockIdx.y;
    w += (uint64_t)g * w_gstride; x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    /* ★09-15 判负存档: "把整条激活载进 shared 让全 block 共用"★
     * 想法: 每个 warp 管一行, 内层每个权重块都为 32 个 lane 读 128 B 激活并重新舍一遍 bf16, 而一条激活
     * 才 ≤32 KB 却被 block 里 8 行各读各舍一遍。实测 **44.7 → 74.3 ms/token(慢 66%)**, 回退。
     * 原因和 09-15 那四个"省访存"的假设同源: L1 本来就把这条激活接住了, 换成 shared 反而多付一次
     * 写入 + 一次 __syncthreads, 还把 32 KB shared 占掉压低了占用率。 */
    if (r < out_dim) {
        const uint32_t nblk = in_dim / 32u;
        const uint8_t *wr = w + (uint64_t)r * nblk * 17u;
        /* ★2026-09-15 段 2: 一个 lane 从"管 1 个元素"改成"管 4 个字节 = 8 个元素"★
         * 病因(接着模板特化那一针往下挖): 旧排布里 lane l 只负责块内第 l 个元素, 于是每算一次乘加要付
         * 权重字节读 + scale 字节读 + e8m0 解码 + nibble 解码 + 激活读 + bf16 舍 —— 6 条指令换 1 次 FFMA,
         * 而 scale 是整块共用的却被 32 个 lane 各读各解一遍。改成 4 个 lane 分一个块(每 lane 拿 4 个连续字节),
         * scale 的读与解码摊到 8 个元素上(省 8 倍), 权重字节读从每元素 2 次降到 0.625 次。
         * ★排布与旧版逐元素对得上★: 块内元素 e ↔ 字节 e/2, 偶数是低半字节、奇数是高半字节(旧版 half/hi 同义);
         * lane 只是换了负责哪几个 e, 权重与激活的配对一个都没动。
         * ★数值★: 段和的相加次序变了(warp 规约里各 lane 的分担变了), 与旧版不是逐位同 ⇒ 门是 NLL/PPL 尺, 不是逐位。
         * ★一 warp 一轮覆盖 8 个 k 块★, 所以 K 的分段单位从"块"变成"8 块一组", ksplit 语义不变。 */
        const uint32_t q = lane & 3u, sub = lane >> 2;
        const uint32_t ngrp = (nblk + 7u) / 8u;
        /* ★两组一轮, 先把两边的权重 load 都发出去再算(single.md S5)★
         * 病: 这个核是**访存延迟**受限(165 GB/s = 墙的 69%, 而算的部分只有 8 次 FFMA)。原来一轮只发
         * 2 条权重 load(4 B nibble + 1 B scale)就立刻用它们, 每个 warp 同时在飞的访存请求太少,
         * 延迟盖不住。展开成两组 ⇒ 4 条 load 并行在飞, ILP 翻倍。
         * ★数值逐位同★: 两组仍按 gi 升序先后累加进同一个 acc, 加法次序一个没变。
         * ★为什么不是"一 warp 管多行"★: 那条 09-15 试过, 慢 2.8 倍(每行的 scale/激活都要重读一遍)。 */
        #define V41_GEMV_BLK(BI) do {                                                        \
            const uint32_t b = (BI) * 8u + sub;                                              \
            if (b < nblk) {                                                                  \
                const uint8_t *p = wr + (uint64_t)b * 17u;                                   \
                const float sc = ds4_e8m0_to_f32(p[16]);                                     \
                /* 块跨距 17 B 天然不对齐, 4 B 一次 memcpy 让编译器发一条非对齐 32 位读      \
                 * (同一个 32 B 扇区内), 比 4 条 LDG.U8 少 3 条指令; 字节一个没变。 */        \
                uint32_t w4; memcpy(&w4, p + q * 4u, 4);                                     \
                float wv[8];                                                                 \
                _Pragma("unroll")                                                            \
                for (uint32_t j = 0; j < 4u; j++) {                                          \
                    const uint32_t byte = (w4 >> (8u * j)) & 0xFFu;                          \
                    wv[2u * j]      = ds4_fp4_nibble_to_f32(byte & 0x0Fu) * sc;              \
                    wv[2u * j + 1u] = ds4_fp4_nibble_to_f32(byte >> 4) * sc;                 \
                }                                                                            \
                const float *xb = x + b * 32u + q * 8u;                                      \
                /* ★这里不再舍 bf16★: 进这个核的激活**已经在 bf16 格点上** —— rms_norm / hc_fused /  \
                 * swiglu / sparse_attn / 上一发 GEMV 的 round_out, 每一条产出 x 的路都在出口舍过。  \
                 * v41_bf16r 对已在格点的值是恒等(RNE 的不动点), 所以去掉它**数值逐位同**, 省的是    \
                 * 每轮每 lane 8×4 条 ALU —— 这个核是延迟/指令受限的(165 GB/s = 墙的 69%), 指令值钱。\
                 * ★前提写死在这里★: 以后若有哪条路把非格点的 f32 直接喂进来, 结果会与官方差一个   \
                 * 舍入位, 不报错 —— 加新调用点时先确认输入是不是 bf16 格点。 */                     \
                _Pragma("unroll")                                                            \
                for (uint32_t t = 0; t < NT; t++) {                                          \
                    const float *xt = xb + (uint64_t)t * x_stride;                           \
                    _Pragma("unroll")                                                        \
                    for (uint32_t j = 0; j < 8u; j++) acc[t] += wv[j] * xt[j];               \
                }                                                                            \
            }                                                                                \
        } while (0)
        uint32_t gi = kpart;
        for (; gi + ksplit < ngrp; gi += 2u * ksplit) { V41_GEMV_BLK(gi); V41_GEMV_BLK(gi + ksplit); }
        for (; gi < ngrp; gi += ksplit) V41_GEMV_BLK(gi);
        #undef V41_GEMV_BLK
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(v) : v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(acc[t]) : acc[t];
    }
}
static int v41_fp4x32_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                           const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                           uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what) {
    if ((in_dim % 32u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || n_groups == 0) return 0;
    const uint64_t wg = out_dim * (in_dim / 32u) * 17u, wbytes = wg * n_groups;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    /* ★并行度目标: 2048(段 2 前) → 32768(段 2) → 8192(single.md S5, 09-16)★
     * 这个数字是"并行度"与"激活复用"的折中, 两边都实测过:
     *   ksplit 大 ⇒ 一 block 管的行少(rpb = 8/ksplit), 同一条激活被更多 block 各读一遍。
     *   wo_b 那种 [5120][8192]: ksplit=8 时 rpb=1, 32 KB 激活被 5120 个 block 读 = 164 MB,
     *   而权重才 22 MB —— ncu 实测这个核 24.92 个扇区/请求(理想 4), 一大半是激活重读。
     * 扫过三档(同机器状态中位): 32768 = 56.0 ms / **8192 = 55.7** / 4096 = 57.0。
     * 8192 让 wo_b/wq_b/head 这些大矩阵回到 rpb=8(激活复用 8 倍), 小矩阵(wkv/wq_a)仍吃满 ksplit=8。
     * ★别再往下调★: 4096 时小矩阵的 warp 数不够盖访存延迟, 反而退(2048 那一版更是只有 1.1 个 wave,
     *   gemv 只跑到 62 GB/s)。总 warp 数 = out_dim × ksplit × n_groups(一 block 恒 8 warp, 管 8/ksplit 行);
     *   ksplit 上限是 8 —— 再大 rows_per_block 就不足 1。
     * ★数值★: ksplit 变了 = K 维分段变了, 段和按 red[] 固定序相加(仍然确定), 但与别的 ksplit 不是逐位同,
     *   所以门是 speed-bench/prefill_2048_ruler.sh 的 NLL/PPL, 不是逐位。 */
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit * n_groups < 8192u) ksplit <<= 1;
    /* ★K 的分段单位现在是"8 个 k 块一组"(见核里的 lane 排布注释)★ 分不出这么多组就把 ksplit 收回来,
     * 否则多出来的 warp 一轮都跑不到(in_dim=1280 只有 5 组, ksplit=8 时 8 个 warp 里 3 个是空转)。 */
    const uint32_t ngrp = (uint32_t)((in_dim / 32u + 7u) / 8u);
    while (ksplit > 1u && ksplit > ngrp) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), n_groups);
    /* NT 是模板参数 ⇒ 按实际 token 数挑一份实例(解码恒走 NT=1 那份, 内层只有一条 FFMA) */
    #define V41_GEMV_LAUNCH(NT) v41_fp4x32_gemv_kernel<NT><<<grid, 256, 0, g_cur_stream>>>( \
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, x_stride, out_stride, ksplit, wg, x_gstride, out_gstride, round_out)
    switch (n_tok) {
        case 1: V41_GEMV_LAUNCH(1u); break;  case 2: V41_GEMV_LAUNCH(2u); break;
        case 3: V41_GEMV_LAUNCH(3u); break;  case 4: V41_GEMV_LAUNCH(4u); break;
        case 5: V41_GEMV_LAUNCH(5u); break;  case 6: V41_GEMV_LAUNCH(6u); break;
        case 7: V41_GEMV_LAUNCH(7u); break;  default: V41_GEMV_LAUNCH(8u); break;
    }
    #undef V41_GEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

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
/* 一 warp 算一行与激活 x 的点积: 向量 j 归 lane j%32(同一轮 32 个 lane 的位流读落在 48 B 连续段内); 码本条目 16 B
 * 从 cbs 取(shared 版 16 B 对齐一次 uint4; 全局版载荷只保证 8 B 对齐, 两次 uint2)。返回 增益×Σ(已 warp 规约)。 */
__device__ __forceinline__ static float v41_vq_row_dot(const v41_vq_mat &m, uint32_t r, const float *xs, const uint8_t *cbs, int cb_shared) {
    const uint64_t i0 = (uint64_t)r * m.nidx_row;
    float acc = 0.f;
    #pragma unroll 4
    for (uint32_t j = threadIdx.x & 31u; j < m.nidx_row; j += 32u) {
        const uint64_t bit = (i0 + j) * m.nbit, by = bit >> 3;
        uint32_t wv; memcpy(&wv, m.ix + by, 4);
        const uint32_t v = (wv >> (bit & 7)) & m.imsk;
        uint2 cw0, cw1;
        if (cb_shared) { const uint4 c = *(const uint4 *)(cbs + (size_t)v * 16u); cw0.x = c.x; cw0.y = c.y; cw1.x = c.z; cw1.y = c.w; }
        else { cw0 = *(const uint2 *)(cbs + (size_t)v * 16u); cw1 = *(const uint2 *)(cbs + (size_t)v * 16u + 8u); }
        const float4 xa = *(const float4 *)(xs + (size_t)j * 8u), xb = *(const float4 *)(xs + (size_t)j * 8u + 4u);
        __half2 h0, h1, h2, h3; memcpy(&h0, &cw0.x, 4); memcpy(&h1, &cw0.y, 4); memcpy(&h2, &cw1.x, 4); memcpy(&h3, &cw1.y, 4);
        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
        acc += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w + f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
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
__global__ static void v41_vq_gateup_kernel(float *h, const uint8_t *blob, const int32_t *sel, const float *x,
                                            uint32_t IN, uint32_t MID, uint32_t K, float clamp, uint32_t cb_bytes) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const uint32_t pair = blockIdx.y, t = pair / K, rows = cb_bytes ? V41_VQ_SH_ROWS : 8u;
    const int32_t e = sel[pair];
    if (e < 0) return;
    const v41_vq_mat mg = v41_vq_open(blob, e, 0, MID, IN, NULL), mu = v41_vq_open(blob, e, 1, MID, IN, NULL);
    if (!mg.ok || !mu.ok) return;
    const uint32_t r0 = blockIdx.x * rows + (threadIdx.x >> 5), nit = cb_bytes ? V41_VQ_ITERS : 1u;
    const float *xs = x + (uint64_t)t * IN;
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
        if ((threadIdx.x & 31u) == 0) h[(uint64_t)pair * MID + r] = v41_bf16r(sg * ui);
    }
}
/* down: partial[pair][OUT] = bf16(W2·h) */
__global__ static void v41_vq_down_kernel(float *partial, const uint8_t *blob, const int32_t *sel, const float *h,
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
    const float *hs = h + (uint64_t)pair * MID;
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
static v41_scratch g_v41_vq_h, g_v41_vq_part;
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
    float *h = (float *)v41_grow(&g_v41_vq_h, np * MID * 4, "v41 vq h");
    float *part = (float *)v41_grow(&g_v41_vq_part, np * OUT * 4, "v41 vq partial");
    if (!h || !part) return 0;
    const uint32_t cbb = nc * 16u;
    if (!g_v41_vq_sh_gateup) {   /* 一次性, 两个核各问各的: gateup 要两本(gate+up), down 只要一本 */
        int cap = 0; (void)cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        const bool og = cudaFuncSetAttribute(v41_vq_gateup_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        const bool od = cudaFuncSetAttribute(v41_vq_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess;
        (void)cudaGetLastError();
        g_v41_vq_sh_gateup = og ? 1 : -1;
        g_v41_vq_sh_down = od ? 1 : -1;
        fprintf(stderr, "ds4: [v41] VQ 码本 %u×16 B(%u KB/本); 设备每 block 动态 shared 上限 %d KB ⇒ gate+up %s / down %s\n",
                nc, cbb >> 10, cap >> 10, og ? "进 shared(一块用两遍)" : "回全局 gather", od ? "进 shared" : "回全局 gather");
    }
    const bool shg = g_v41_vq_sh_gateup == 1, shd = g_v41_vq_sh_down == 1;
    /* 线程数恒为 32 warp(shared 版一 warp 循环 ITERS 行)或 8 warp(全局 gather 版一 warp 一行);
     * ★不能写成 rows×32★: shared 版 rows 已是 256, 那会要 8192 个线程, 超过每 block 1024 的上限。 */
    const uint32_t rg = shg ? V41_VQ_SH_ROWS : 8u, rd = shd ? V41_VQ_SH_ROWS : 8u;
    const uint32_t tg = shg ? 32u * 32u : 8u * 32u, td = shd ? 32u * 32u : 8u * 32u;
    v41_vq_gateup_kernel<<<dim3((MID + rg - 1u) / rg, (unsigned)np), tg, shg ? cbb : 0u, g_cur_stream>>>(h, blob, sel, x, IN, MID, K, clamp, shg ? cbb : 0u);
    if (!cuda_ok(cudaGetLastError(), "v41 vq gateup")) return 0;
    v41_vq_down_kernel<<<dim3((OUT + rd - 1u) / rd, (unsigned)np), td, shd ? cbb : 0u, g_cur_stream>>>(part, blob, sel, h, MID, OUT, K, shd ? cbb : 0u, gr);
    if (!cuda_ok(cudaGetLastError(), "v41 vq down")) return 0;
    v41_vq_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>(out, part, w, K, OUT);
    return cuda_ok(cudaGetLastError(), "v41 vq reduce");
}

/* ---- engram 查表行 dequant(fp8 e4m3 × ue8m0/32 → bf16 格点): 一线程一元素 ---- */
__global__ static void v41_engram_rows_kernel(float *out, const uint8_t *raw, uint32_t n_rows, uint32_t hd) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_rows * hd) return;
    const uint32_t r = (uint32_t)(i / hd), d = (uint32_t)(i % hd), stride = hd + hd / 32u;
    const uint8_t *row = raw + (uint64_t)r * stride;
    out[i] = v41_bf16r(ds4_e4m3fn_to_f32(row[d]) * ds4_e8m0_to_f32(row[hd + d / 32u]));
}
int ds4_gpu_v41_engram_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *raw, uint32_t n_rows, uint32_t head_dim) {
    if (!out || !raw || (head_dim % 32u)) return 0;
    const uint64_t n = (uint64_t)n_rows * head_dim;
    if (out->bytes < n * 4 || raw->bytes < (uint64_t)n_rows * (head_dim + head_dim / 32u)) return 0;
    v41_engram_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)out->ptr, (const uint8_t *)raw->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "v41 engram rows");
}

/* ---- 反修放大器: T(n×K) = x·B, y += T·A ----
 * 列主序推导: x 内存行主序 n×D = 列主序 xᵀ(D×n, ld D); B 内存 [K][D] 行主序 = 列主序 B_math(D×K, ld D);
 * Tᵀ(K×n) = B_mathᵀ·xᵀ ⇒ Sgemm(T, N, K, n, D); A 内存 [K][D] 行主序 = 列主序 Aᵀ(D×K, ld D);
 * yᵀ(D×n) += Aᵀ·Tᵀ ⇒ Sgemm(N, N, D, n, K, beta=1)。 */
int ds4_gpu_v41_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                 ds4_gpu_tensor *T, uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!y || !x || !A || !B || !T || !g_cublas_ready || n_tok == 0 || K == 0) return 0;
    if (T->bytes < (uint64_t)n_tok * K * 4 || y->bytes < (uint64_t)n_tok * D * 4 || x->bytes < (uint64_t)n_tok * D * 4) return 0;
    const float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &a1,
                                    (const float *)B->ptr, (int)D, (const float *)x->ptr, (int)D, &b0, (float *)T->ptr, (int)K);
    if (!cublas_ok(st, "v41 amp T=xB")) return 0;
    st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n_tok, (int)K, &a1,
                     (const float *)A->ptr, (int)D, (const float *)T->ptr, (int)K, &b1, (float *)y->ptr, (int)D);
    return cublas_ok(st, "v41 amp y+=TA");
}

/* ---- 小件: 就地 x = bf16(x·s)(indexer weights 缩放, 原先主机往返) / argmax(原先读回 517 KB 主机扫) ---- */
__global__ static void v41_scale_round_kernel(float *x, uint64_t n, float s) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v41_bf16r(x[i] * s);
}
int ds4_gpu_v41_scale_round_tensor(ds4_gpu_tensor *x, uint64_t n, float s) {
    if (!x || x->bytes < n * 4) return 0;
    v41_scale_round_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)x->ptr, n, s);
    return cuda_ok(cudaGetLastError(), "v41 scale round");
}
__global__ static void v41_argmax_kernel(int32_t *idx, const float *row, uint32_t n) {
    __shared__ float bv[1024]; __shared__ int32_t bi[1024];
    float best = -INFINITY; int32_t besti = 0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { const float v = row[i]; if (v > best) { best = v; besti = (int32_t)i; } }
    bv[threadIdx.x] = best; bi[threadIdx.x] = besti; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {   /* 同值取小下标(与主机顺序扫一致) */
        if (threadIdx.x < k) {
            const float ov = bv[threadIdx.x + k]; const int32_t oi = bi[threadIdx.x + k];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = ov; bi[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) idx[0] = bi[0];
}
int ds4_gpu_v41_argmax_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *logits, uint32_t row, uint32_t n_vocab) {
    if (!idx || !logits || idx->bytes < 4 || logits->bytes < ((uint64_t)row + 1) * n_vocab * 4) return 0;
    v41_argmax_kernel<<<1, 1024, 0, g_cur_stream>>>((int32_t *)idx->ptr, (const float *)logits->ptr + (uint64_t)row * n_vocab, n_vocab);
    return cuda_ok(cudaGetLastError(), "v41 argmax");
}
