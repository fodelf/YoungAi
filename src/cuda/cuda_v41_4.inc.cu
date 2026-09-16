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

/* 一 block 多少个 warp。16 试过(激活复用翻倍): 55.4 vs 8 warp 的 55.2 —— 持平偏差, 维持 8。
 * 连同平面副本只值 4% 这件事一起说明: 这个核的大矩阵**已经贴着带宽墙**(wo_b 单发 22.3 MB/~110 µs),
 * 剩下的差距在小矩阵的固定开销上, 不在激活重读。 */
#define V41_GEMV_WARPS 8u
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
/* PLANAR=1: 权重来自平面副本(nibble 一片 16 B/块 + scale 一片 1 B/块, 见 cuda_v41_fp4_planar.inc.cu),
 * 一个 warp 一轮读的 8 个块正好是 128 B 连续。PLANAR=0: 盘上的交错布局(17 B 跨距)。
 * 两条路的字节与配对完全相同 ⇒ 数值逐位同。 */
template <uint32_t NT, uint32_t PLANAR>
__global__ static void v41_fp4x32_gemv_kernel(float *out, const uint8_t *w, const uint8_t *wsc, const float *x, uint32_t in_dim, uint32_t out_dim,
                                              uint32_t x_stride, uint32_t out_stride, uint32_t ksplit,
                                              uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out) {
    const uint32_t g = blockIdx.y;   /* 块对角(wo_a)的"第几组": 权重/输入/输出各按步长偏移 */
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride;
    if (PLANAR) { const uint64_t nb = (uint64_t)out_dim * (in_dim / 32u); w += (uint64_t)g * nb * 16u; wsc += (uint64_t)g * nb; }
    else w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    /* ★一 block 16 warp(single.md §2.6 后续): 治的是"激活重读"★
     * 一个 block 管 rows_per_block 行, 这些行共用同一条激活(在 L1 里)。warp 少 ⇒ 一 block 管的行少 ⇒
     * 同一条激活被更多 block 各读一遍。wo_b [5120][8192] 在 8 warp/ksplit=2 时激活读 41 MB, 权重才 22 MB;
     * 16 warp 把它减半。ncu 那 24.92 扇区/请求里, 权重只占小头(平面化后只省了 4%), 大头就是这个。 */
    /* ★warp 数从 blockDim 读, 不再是编译期常量(2026-09-16, mtp.md M2)★
     * 为什么: 投机验证批(NT>1)每个权重块要读 NT 条激活, 激活重读是 NT 倍 —— 一个 block 管的行越多,
     * 同一条激活被越少的 block 各读一遍。NT=1 时 16 warp 实测持平(09-15), 但 NT=6 时压力是 6 倍,
     * 所以只给 NT>1 开 16。★逐位同★: ksplit 没动 ⇒ 每一行的段划分与段和相加次序一个字没变,
     * 变的只是"哪些行归同一个 block"。 */
    const uint32_t nwarp = blockDim.x >> 5;
    const uint32_t rows_per_block = nwarp / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[16][NT];
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
        const uint8_t *wr = w + (uint64_t)r * nblk * (PLANAR ? 16u : 17u);
        const uint8_t *wrs = PLANAR ? wsc + (uint64_t)r * nblk : NULL;
        /* ★2026-09-15 段 2: 一个 lane 从"管 1 个元素"改成"管 4 个字节 = 8 个元素"★
         * 病因(接着模板特化那一针往下挖): 旧排布里 lane l 只负责块内第 l 个元素, 于是每算一次乘加要付
         * 权重字节读 + scale 字节读 + e8m0 解码 + nibble 解码 + 激活读 + bf16 舍 —— 6 条指令换 1 次 FFMA,
         * 而 scale 是整块共用的却被 32 个 lane 各读各解一遍。改成 4 个 lane 分一个块(每 lane 拿 4 个连续字节),
         * scale 的读与解码摊到 8 个元素上(省 8 倍), 权重字节读从每元素 2 次降到 0.625 次。
         * ★排布与旧版逐元素对得上★: 块内元素 e ↔ 字节 e/2, 偶数是低半字节、奇数是高半字节(旧版 half/hi 同义);
         * lane 只是换了负责哪几个 e, 权重与激活的配对一个都没动。
         * ★数值★: 段和的相加次序变了(warp 规约里各 lane 的分担变了), 与旧版不是逐位同 ⇒ 门是 NLL/PPL 尺, 不是逐位。
         * ★一 warp 一轮覆盖 8 个 k 块★, 所以 K 的分段单位从"块"变成"8 块一组", ksplit 语义不变。 */
        /* ★2026-09-16 判负存档: "8 lane 分一个块 + 激活改一条 float4"(single-1.md #3)★
         * 依据(SASS 实证, 不是猜的): 激活这一行编出 **8 条标量 `LDG.E.CONSTANT`**, 而一条指令里 32 个
         * lane 的地址是 x + BI·256 + sub·32 + q·8(float) ⇒ **铺开 1024 B, 每 lane 只取 4 B**, 一次请求
         * 碰 32 个扇区 —— 09-16 ncu 那个"24.92 扇区/请求"的大头在这儿, 不在权重的 17 B 跨距
         * (所以只动权重的平面副本才只值 4%)。
         * 改法: 8 个 lane 分一个块, 每 lane 4 个元素 = 一条 LDG.128, 32 个 lane 正好连续 512 B。
         * 实测 **22.75 ms vs 原来的 20.83(慢 9%)**, 已回退。
         * 为什么扇区少了反而慢: 一个 warp 一轮从 8 个块(256 个权重)掉到 4 个块(128 个), **循环轮数翻倍**,
         * 每轮的块号算术/边界判断/scale 解码都要再付一遍; 而省下的那些扇区本来**命中 L1**
         * (同一条激活被 block 内 8 行反复读, 早就在 L1 里了), 省的是 L1 的吞吐不是 DRAM 的字节。
         * ★教训与 VQ 那刀同源★: ncu 的"扇区/请求"是 L1 侧指标, 它高不等于在等 DRAM。
         * 下次动之前先用 `smsp__warp_issue_stalled_*` 看 warp 到底停在哪 —— 这正是 single-1.md §4
         * 第 1 步写的 ncu 那两趟, 我跳过它直接改码, 于是两刀全负。
         * 真要治"一轮 4 个块"的循环开销, 得在保持 8 lane/块的同时把展开从 2 组提到 4 组(未试)。 */
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
                const uint8_t *p = wr + (uint64_t)b * (PLANAR ? 16u : 17u);                  \
                const float sc = ds4_e8m0_to_f32(PLANAR ? wrs[b] : p[16]);                   \
                /* 块跨距 17 B 天然不对齐; 4 B 一次 memcpy 编出 4 条 LDG.U8, 但它们落在同一个    \
                 * 32 B 扇区里, L1 一次供上(09-16 试过改对齐读 + funnelshift, 慢, 见 VQ 那边的存档)。*/ \
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
                /* ★这 8 个 float 读成两条 float4(2026-09-16)★                                 \
                 * 病(09-16 ncu 实证, 25.72 扇区/请求): 写成 `for j<8: xt[j]` 时编译器证不出对齐,     \
                 * 编出 **8 条标量 LDG.E.CONSTANT**; 每条里 32 个 lane 的地址是 sub·128 + q·32,      \
                 * 铺开 **1024 B 却每 lane 只取 4 B** ⇒ 一条就碰 8 个 wavefront, 八条 **64 个**。     \
                 * 而一个 warp 一轮覆盖的 8 个块 = 256 个 float = 1024 B **本来就是连续的**,           \
                 * 每 lane 正好 32 B —— 形状是完美合并的, 只是读法把它拆散了。                        \
                 * 两条 float4: 每条 32 lane × 16 B(跨距 32 B), 8 个 wavefront, 两条 16 个 ⇒ **4 倍**。\
                 * ★逐位同★: 取的是同样 8 个值, 乘加次序 j=0..7 一个字没变。                          \
                 * ★对齐前提★: xt 的元素下标 = t·x_stride + b·32 + q·8, 三项都是 4 的倍数,            \
                 * 字节地址必是 16 的倍数(x_stride 是 in_dim, 32 的倍数) —— 破了这条会直接            \
                 * misaligned address 崩, 不会静默走偏。                                            \
                 * ★与 09-16 那条判负存档的区别★: 那次改的是"哪个 lane 管哪个块"(8 lane 分一块),      \
                 * 循环轮数翻倍, 慢 9%; 这里 lane 分工一个字没动, 只换读法。 */                       \
                _Pragma("unroll")                                                            \
                for (uint32_t t = 0; t < NT; t++) {                                          \
                    const float *xt = xb + (uint64_t)t * x_stride;                           \
                    const float4 xa = *(const float4 *)xt, xc = *(const float4 *)(xt + 4u);  \
                    /* ★必须写成八条独立的累加, 不能合成一个表达式★(2026-09-16 实撞):            \
                     * 合成一句 `acc += w0x0 + w1x1 + ... + w7x7` 时, --use_fast_math 允许编译器    \
                     * 把右边算成一棵加法树再并进 acc —— 累加次序就变了。实测输出在第 100 个字符    \
                     * 左右开始分叉, 而且**看着完全通顺**, 只有逐字节比才抓得到。 */               \
                    acc[t] += wv[0] * xa.x; acc[t] += wv[1] * xa.y;                           \
                    acc[t] += wv[2] * xa.z; acc[t] += wv[3] * xa.w;                           \
                    acc[t] += wv[4] * xc.x; acc[t] += wv[5] * xc.y;                           \
                    acc[t] += wv[6] * xc.z; acc[t] += wv[7] * xc.w;                           \
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
    /* ★平面副本(single.md §2.6 A 路)★: 盘上 17 B 跨距让每个块都不对齐(ncu: 24.92 扇区/请求);
     * 平面版一个 warp 一轮读的 8 个块是 128 B 连续。拿不到就照旧走交错, 不是必需品。
     * n_groups>1(wo_a 的块对角)那条路暂不建副本 —— 它的分组偏移语义要另算, 收益也小。 */
    const v41_fp4_planar pl = (n_groups == 1u)
        ? v41_fp4_planar_get(model_map, model_size, off, in_dim, out_dim, what)
        : (v41_fp4_planar){ NULL, NULL };
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
    while (ksplit < V41_GEMV_WARPS && out_dim * ksplit * n_groups < 8192u) ksplit <<= 1;
    /* ★K 的分段单位是"8 个 k 块一组"(见核里的 lane 排布注释)★ 分不出这么多组就把 ksplit 收回来,
     * 否则多出来的 warp 一轮都跑不到(in_dim=1280 只有 5 组, ksplit=8 时 8 个 warp 里 3 个是空转)。 */
    const uint32_t ngrp = (uint32_t)((in_dim / 32u + 7u) / 8u);
    while (ksplit > 1u && ksplit > ngrp) ksplit >>= 1;
    /* 验证批(NT>1)一 block 开 16 warp: 激活重读减半(见核里的注释); 解码恒 8(那边 16 实测持平) */
    const uint32_t warps = (n_tok > 1u) ? 16u : V41_GEMV_WARPS;
    const uint32_t rpb = warps / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), n_groups);
    /* NT 是模板参数 ⇒ 按实际 token 数挑一份实例(解码恒走 NT=1 那份, 内层只有一条 FFMA) */
    #define V41_GEMV_LAUNCH(NT) do {                                                                       \
        if (pl.nib) v41_fp4x32_gemv_kernel<NT, 1u><<<grid, warps * 32u, 0, g_cur_stream>>>(         \
            out, pl.nib, pl.sc, x, (uint32_t)in_dim, (uint32_t)out_dim, x_stride, out_stride, ksplit, wg,    \
            x_gstride, out_gstride, round_out);                                            \
        else v41_fp4x32_gemv_kernel<NT, 0u><<<grid, warps * 32u, 0, g_cur_stream>>>(                \
            out, w, NULL, x, (uint32_t)in_dim, (uint32_t)out_dim, x_stride, out_stride, ksplit, wg,          \
            x_gstride, out_gstride, round_out);                                            \
    } while (0)
    switch (n_tok) {
        case 1: V41_GEMV_LAUNCH(1u); break;  case 2: V41_GEMV_LAUNCH(2u); break;
        case 3: V41_GEMV_LAUNCH(3u); break;  case 4: V41_GEMV_LAUNCH(4u); break;
        case 5: V41_GEMV_LAUNCH(5u); break;  case 6: V41_GEMV_LAUNCH(6u); break;
        case 7: V41_GEMV_LAUNCH(7u); break;  default: V41_GEMV_LAUNCH(8u); break;
    }
    #undef V41_GEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
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
