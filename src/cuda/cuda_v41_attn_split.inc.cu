/* cuda_v41_attn_split.inc.cu — ds4_cuda.cu 分片: 解码路的稀疏注意力 split-K(single.md S4, 2026-09-15)。
 *
 * 病(逐核时间线实测): 解码时 `v41_sparse_attn_kernel` 的 grid 是 (n_tok=1, 8 个头组) = **只有 8 个 block**,
 * 48 个 SM 里 40 个干等; 键只能在这 8 个 block 里串着啃, 实测 **每个键 5.3 µs**, 32 步时已经 5~8 ms/步,
 * 而且随可见键数线性涨 —— 32k 上下文处 640 键 × 40 层 ≈ 136 ms/步, 比现在整步还长。
 *
 * 修: 把键切成段, grid 变成 (键段, 头组), 每个 block 只啃自己那段的键, 出**局部**的
 * (max, sum, acc) 三件(flash-attention 的在线 softmax 是可合并的), 再一发把各段合起来。
 * 48 个 SM 全上, 且键多了就多切几段 —— 时间不再随上下文线性涨。
 *
 * ★数值★: 分段改变了在线 softmax 的分组(与段 4 的 mma 版同一性质, 那次 NLL 反而好 1.3%),
 * 所以门是 NLL 尺不是逐位同。合并按**段号固定序**, 不用原子加 —— 否则同一输入两跑结果不同(E0 那种病)。
 *
 * 只给解码(n_tok == 1)用: n 大时原核的 grid 本来就够宽, 且预填走的是 mma 版。 */

/* 这两个常量的"正本"在 cuda_v41_2.inc.cu(主注意力核)。本分片按 include 顺序排在它**前面**(它要调 split),
 * 所以在这里先定义, 那边用 #ifndef 守着 —— 两处写的必须是同一个值, 改一处就要改另一处。 */
#define V41_ATTN_HEADS_PER_BLOCK 8u
#define V41_ATTN_KTILE 8u

/* ★2026-09-16 single-1.md #2: 阈值 128 → 16, 段长改成按键数自适应★
 * 病: 短上下文解码可见键只有几十个(3 token 提示跑 32 步 ⇒ 约 35 个), 低于 128 就整个走回原核 ——
 * 原核解码时 grid 只有 8 个 block, 48 个 SM 里 40 个干等, 实测 **130 µs/层 × 40 层 = 5.2 ms/步**,
 * 占 54.2 ms 的一成。single.md 把这一条记成"短上下文量不到它", 其实是"阈值把它挡在门外"。
 * 段长固定 64 也有同样的毛病: 70 个键只切出 2 段 × 8 头组 = 16 个 block, 还是填不满。
 * 改成**先定 block 数再回推段长**: 目标每头组 6 段 ⇒ 6 × 8 头组 = 48 block = 一个 SM 一个。
 * 段长上限仍是 64(键多时维持原样, 不动已验过的长上下文行为), 下限 8(= KTILE, 再小 shared 搬运划不来)。
 * 合并那一发的开销: grid 64 个 block × 256 线程, 实测量级 5 µs —— 与 130 µs 不是一个数量级,
 * 所以"段太短合并占大头"这个当初的顾虑在解码这一侧不成立。
 * ★数值★: 分段改变在线 softmax 的分组 ⇒ 不是逐位同, 门是 NLL/PPL 尺(与 S4 落地时同一条规矩)。 */
#define V41_ATTN_SPLIT_MIN_KEYS 16u    /* 键少于这个数就别切了: 一个 block 几微秒就啃完, 切了纯赔合并 */
#define V41_ATTN_SPLIT_SEG_KEYS 64u    /* 一段最多多少个键(8 的倍数; 与 KTILE 8 对齐) */
#define V41_ATTN_SPLIT_SEG_MIN  8u     /* 一段最少多少个键 = KTILE */
/* ★2026-09-16 decode.md D2 判负存档: 目标段数 6 → 36(提占用率)★
 * 假设: 这个核 12k 真场景实测 **12.12 ms/步 = 每层 303 µs**, 而一层只做 640 键 × 64 头 × 512 维
 * = 21M 次 FMA(合 69 GFLOP/s, 标量峰值的 0.4%); 字节更不是(每层读 5 MB, 按墙只值 0.02 ms)。
 * 于是怀疑 warp 不够: 6 段 × 8 头组 = 48 个 block × 128 线程 ⇒ 每个 SM 才 4 个 warp(能装 64 个)。
 * 改成 36 段 = 288 个 block(每 SM 6 个)。
 * **实测判负**: 12k 61.2 → 60.2 ms(只 −1.6%), 而温 0 输出**变了**(分段变 ⇒ 在线 softmax 分组变)。
 * 为 1.6% 换掉"输出不变"这条不划算 —— 铁律: 质量门不为速度让步。已回退到 6。
 * ⇒ 这个核慢的真因**不是占用率**。下一个嫌疑: 每个 (键, 头) 付一次 5 条 shfl 的 warp 归约才换 16 个 FMA
 * (与 09-15 写 mma 版时给标量版做的诊断同源); 要动就得换成"一 lane 一个键、512 维串着点积"的形态
 * (无 shuffle), 那是另一件事, 且同样不逐位同。 */
#define V41_ATTN_SPLIT_TARGET_SEG 6u   /* 每个头组切几段(判负存档见上, 别再往上调) */
#define V41_ATTN_SPLIT_MAX_SEG  64u    /* 段数上限 ⇒ 暂存 64 × 64 头 × 512 × 4 B = 8 MB */

/* 一段的局部注意力: 与 v41_sparse_attn_kernel 的内层同一口径(8 键一块进 shared, bf16 舍概率),
 * 只是键范围限定在 [k0, k1) 且出口不除分母 —— 分母留给合并核。
 * pacc[seg][head][hd] / pmax[seg][head] / psum[seg][head] */
__global__ static void v41_sparse_attn_split_kernel(float *pacc, float *pmax, float *psum,
                                                    const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                    uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                                    uint32_t n_head, uint32_t hd, float scale, uint32_t seg_keys) {
    const uint32_t seg = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;
    __shared__ float ks[V41_ATTN_KTILE][512];
    __shared__ int   kok[V41_ATTN_KTILE];
    __shared__ float ksc[V41_ATTN_KTILE][32];   /* 压缩行的 32 个缩放, 每行解一次(见 v41_ckv_get_s) */
    float qa[2][16], acc[2][16], mx[2], sum[2];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) { qa[hh][e] = q[(uint64_t)h * hd + lane * per + e]; acc[hh][e] = 0.f; }
        mx[hh] = -1e30f; sum[hh] = 0.f;
    }
    const uint32_t p = pos0;                       /* 解码: 只有一个 query, 绝对位置就是 pos0 */
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    const uint32_t k0 = seg * seg_keys, k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t base = k0; base < k1; base += V41_ATTN_KTILE) {
        const uint32_t nt = (k1 - base) < V41_ATTN_KTILE ? (k1 - base) : V41_ATTN_KTILE;
        __syncthreads();
        for (uint32_t t = threadIdx.x / 32u; t < nt; t += blockDim.x / 32u) {
            const uint32_t kk = base + t;
            const float *krow = NULL; const uint8_t *cpk = NULL;   /* 窗口行 f32 / 压缩行打包 FP4, 见 cuda_kv_pack */
            /* ring=1 恒成立: 这条路只在主路(!full_block)被调, 而主路的历史段就是环(见 v41_win_row) */
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * hd;
            else { const int32_t g = idx[kk - nwin];
                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
            if (lane == 0) kok[t] = (krow || cpk) ? 1 : 0;
            if (cpk) ksc[t][lane] = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);   /* 32 个缩放, 一 lane 一个 */
            __syncwarp();                              /* 本 warp 自己写自己读, 不用全 block 同步 */
            for (uint32_t d = lane; d < hd; d += 32u)   /* 无效槽必须写 0, 见主核注释 */
                ks[t][d] = krow ? krow[d] : (cpk ? v41_ckv_get_s(cpk, d, ksc[t]) : 0.f);
        }
        __syncthreads();
        for (int hh = 0; hh < 2; hh++) {
            float s[V41_ATTN_KTILE];
            float tm = -1e30f;
            for (uint32_t t = 0; t < nt; t++) {
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * ks[t][lane * per + e];
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                s[t] = kok[t] ? d * scale : -1e30f;
                tm = fmaxf(tm, s[t]);
            }
            const float nm = fmaxf(mx[hh], tm);
            const float rs = expf(mx[hh] - nm);
            float acc_add[16];
            for (uint32_t e = 0; e < per; e++) acc_add[e] = 0.f;
            float ps = 0.f;
            for (uint32_t t = 0; t < nt; t++) {
                const float pv = expf(s[t] - nm);
                ps += pv;
                const float pb = v41_bf16r(pv);
                for (uint32_t e = 0; e < per; e++) acc_add[e] += pb * ks[t][lane * per + e];
            }
            sum[hh] = sum[hh] * rs + ps;
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + acc_add[e];
            mx[hh] = nm;
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        float *pa = pacc + ((uint64_t)seg * n_head + h) * hd;
        for (uint32_t e = 0; e < per; e++) pa[lane * per + e] = acc[hh][e];
        if (lane == 0) { pmax[(uint64_t)seg * n_head + h] = mx[hh]; psum[(uint64_t)seg * n_head + h] = sum[hh]; }
    }
}

/* ★解码张量核版的段长(正本; 2026-09-18 从 cuda_v41_attn_mma_decode.inc.cu 挪到这里, 合并核也要用它)★
 * ★段长只许由"与批大小无关的参考键数"定(2026-09-16, mtp-1.md M1′)★
 *
 * 为什么必须是常数: 分段决定在线 softmax 的分组, 分组变 ⇒ 累加序变 ⇒ 近平局的 token 会翻面。
 * 原来是"把键平分成 24/n_tok 段", 于是同一个位置的同一个 query, 纯解码(n=1)切 20 段、
 * 投机验证批(n=4)切 10 段 —— **两条路算出来的 logits 不一样**, 温 0 下投机与纯解码第二句就分叉
 * (mtp-1.md §4 的盘上证据: 同一份 2K 提示, 纯解码说 "truncated README", 投机说 "native inference engine")。
 * 而投机的全部合法性就建立在"它只省时间、不改输出"上。
 *
 * ★写死一个常数不行, 三档都比改造前慢★(同机器状态纯解码 t/s, 基线 2K 20.00 / 12k 19.87):
 *   写死 32 → 2K 19.16 / 12k 18.95; 写死 16 → 2K 18.61。
 * 原因: 这个模型的层分两类 —— 压缩比大的层可见键只有一百多个, 要 16 才切得出足够的 block 铺满 SM;
 * 键多的层(2K ≈588、12k 640)要 32, 段再短就是白把 16 个头的 q(32 KB/block)重载一遍 + 合并核多几段要加。
 * 改造前那个自适应公式(平分 24 段 + 取到 16 的倍数 + 上限 64)恰好就给出这两个值, 它本身是对的;
 * 错的只是**喂给它的键数是一个两条路不一样的量**。
 * ⇒ 正解: 公式原样保留, 改喂"按**这个 query 自己的绝对位置**算出来的参考键数" ——
 * 于是纯解码的分段与改造前逐个相同(对基线逐字节), 验证批里每个 query 又与纯解码在同一位置时同段。
 * ★"按本批第一个 query 算"也不行★(实撞, 2K 第 270 字节分叉): 批里第 2..k 个 query 就用了别人的段长。
 * ★空槽是精确中性的★: 分数压 -inf ⇒ p=0; 整段空时 max=-1e30/acc=0, 合并核里乘 exp(-1e30-m)=0,
 * 所以"验证批多几个空槽、多一段"不影响结果 —— 只要它不去改段长。 */
#define V41_ATTN_MMA_DEC_TARGET_SEG 24u   /* 目标段数(照抄改造前的公式; 24 段 × 4 头组 ≈ 96 block) */
/* 某个 query 的段长: 只由它**自己的绝对位置** p 决定(外加层常量 window/ratio 与批级上限 topk)。
 * 主机与核里都调它 —— 两处各写一份迟早漂开, 而漂开的症状是"投机偶尔与纯解码差一个 token"。
 * topk 是批级的槽数上限: 取 min 之后, 纯解码在位置 p 算出来的与验证批里那个 p 算出来的逐个相同
 * (ng ≥ 任何一个 query 的可见组数, 所以 min 的结果由 (p+1)/ratio 决定)。 */
__host__ __device__ __forceinline__ static uint32_t v41_attn_seg_keys(uint32_t p, uint32_t window,
                                                                     uint32_t ratio, uint32_t topk) {
    const uint32_t nwin = p + 1u > window ? window : p + 1u;
    uint32_t tref = 0;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; tref = topk < vis ? topk : vis; }
    uint32_t seg = (nwin + tref + V41_ATTN_MMA_DEC_TARGET_SEG - 1u) / V41_ATTN_MMA_DEC_TARGET_SEG;
    seg = ((seg + DS4_ATTN_MMA_KT - 1u) / DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_KT;
    if (seg < DS4_ATTN_MMA_KT) seg = DS4_ATTN_MMA_KT;
    return seg > 64u ? 64u : seg;
}
/* 位置 p 的解码 query 有几段(= 主机直发路给 grid 的段数; graph 路的合并核按设备位置自算同一个数)。
 * topk 给批级上限(index_topk), 里面取 min(topk, 可见组数) —— 与直发路主机传 min(index_topk, ng) 再算逐个相同。 */
__host__ __device__ __forceinline__ static uint32_t v41_attn_nseg_at(uint32_t p, uint32_t window, uint32_t ratio, uint32_t topk) {
    const uint32_t nw = p + 1u > window ? window : p + 1u;
    uint32_t tk = topk;
    if (ratio) { const uint32_t vis = (p + 1u) / ratio; if (vis < tk) tk = vis; }
    const uint32_t sk = v41_attn_seg_keys(p, window, ratio, topk);
    return (nw + tk + sk - 1u) / sk;
}

/* 合并: 一 block 一个头, 按**段号固定序**做 flash 合并(不是原子加 —— 那会让同一输入两跑结果不同),
 * 再加 sink 进分母、除、舍 bf16。o[head][hd] */
/* ★2026-09-16: 加 token 维(grid.y), 给投机验证批的张量核版用(decode.md D2 / mtp.md M2)★
 * 局部件的排法是 [(token·nseg + 段)·头 + h]; n_tok=1 时就是原来的 [段·头 + h], 一个字节没挪 ——
 * 所以 split 那条路(恒 n_tok=1)的行为完全不变, 只是多传一个 grid.y=1。 */
/* ★posd(graph 路, 2026-09-18)★: nseg 传的是桶上限(grid 按它开), 真段数按设备位置自算 —— 只读真段。
 * 与直发路逐位同: 直发路的段数就是 v41_attn_nseg_at(p) 那个数, 多出来的段本来就不存在, 少读它们不改任何加法。 */
__global__ static void v41_sparse_attn_merge_kernel(float *o, const float *pacc, const float *pmax, const float *psum,
                                                    const float *sink, uint32_t nseg, uint32_t n_head, uint32_t hd,
                                                    const int32_t *posd, uint32_t window, uint32_t ratio, uint32_t topk) {
    const uint32_t h = blockIdx.x, i = blockIdx.y;
    const uint64_t b0 = (uint64_t)i * nseg * n_head;   /* 局部件的行步长按 grid 的 nseg(上限)排 */
    if (posd) nseg = v41_attn_nseg_at((uint32_t)posd[0], window, ratio, topk);
    float m = -1e30f;
    for (uint32_t s = 0; s < nseg; s++) m = fmaxf(m, pmax[b0 + (uint64_t)s * n_head + h]);
    float den = 0.f;
    for (uint32_t s = 0; s < nseg; s++) den += psum[b0 + (uint64_t)s * n_head + h] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
    den += expf(sink[h] - m);
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t s = 0; s < nseg; s++)
            v += pacc[(b0 + (uint64_t)s * n_head + h) * hd + d] * expf(pmax[b0 + (uint64_t)s * n_head + h] - m);
        o[((uint64_t)i * n_head + h) * hd + d] = v41_bf16r(v / den);
    }
}

static v41_scratch g_v41_attn_pacc, g_v41_attn_pmax, g_v41_attn_psum;

/* 返回 1 = 这一发由 split-K 路接管; 0 = 形状不适用, 调用方回落到原核。 */
static int v41_sparse_attn_split(float *o, const float *q, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                 const float *sink, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                                 uint32_t topk, uint32_t n_head, uint32_t hd, float scale) {
    if (n_tok != 1u || hd != 512u || (n_head % V41_ATTN_HEADS_PER_BLOCK)) return 0;
    const uint32_t nwin = pos0 + 1u > window ? window : pos0 + 1u;
    const uint32_t nkeys = nwin + topk;
    if (nkeys < V41_ATTN_SPLIT_MIN_KEYS) return 0;
    /* 段长 = min(64, 把键平分成 TARGET_SEG 段的长度), 8 的倍数, 至少 KTILE。
     * 键多时(≥384)取 64, 与 09-15 验过的行为完全一样; 键少时段变短, block 数补上来。 */
    uint32_t seg_keys = (nkeys + V41_ATTN_SPLIT_TARGET_SEG - 1u) / V41_ATTN_SPLIT_TARGET_SEG;
    seg_keys = ((seg_keys + V41_ATTN_SPLIT_SEG_MIN - 1u) / V41_ATTN_SPLIT_SEG_MIN) * V41_ATTN_SPLIT_SEG_MIN;
    if (seg_keys < V41_ATTN_SPLIT_SEG_MIN) seg_keys = V41_ATTN_SPLIT_SEG_MIN;
    if (seg_keys > V41_ATTN_SPLIT_SEG_KEYS) seg_keys = V41_ATTN_SPLIT_SEG_KEYS;
    uint32_t nseg = (nkeys + seg_keys - 1u) / seg_keys;
    while (nseg > V41_ATTN_SPLIT_MAX_SEG) { seg_keys *= 2u; nseg = (nkeys + seg_keys - 1u) / seg_keys; }
    const uint64_t na = (uint64_t)nseg * n_head;
    float *pacc = (float *)v41_grow(&g_v41_attn_pacc, na * hd * 4, "v41 attn split acc");
    float *pmax = (float *)v41_grow(&g_v41_attn_pmax, na * 4, "v41 attn split max");
    float *psum = (float *)v41_grow(&g_v41_attn_psum, na * 4, "v41 attn split sum");
    if (!pacc || !pmax || !psum) return 0;
    v41_sparse_attn_split_kernel<<<dim3(nseg, n_head / V41_ATTN_HEADS_PER_BLOCK), V41_ATTN_HEADS_PER_BLOCK * 16u, 0, g_cur_stream>>>(
        pacc, pmax, psum, q, kvw, kvc, idx, pos0, window, ng, topk, n_head, hd, scale, seg_keys);
    if (!cuda_ok(cudaGetLastError(), "v41 sparse attn split")) return 0;
    v41_sparse_attn_merge_kernel<<<dim3(n_head, 1), 256, 0, g_cur_stream>>>(o, pacc, pmax, psum, sink, nseg, n_head, hd,
                                                                            NULL, window, 0u, topk);
    return cuda_ok(cudaGetLastError(), "v41 sparse attn merge");
}
