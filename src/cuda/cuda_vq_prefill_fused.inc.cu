/* cuda_vq_prefill_fused.inc.cu — 预填专家路: VQ 解码即乘, 不落 f16 暂存(2026-09-15, speed.md 第三轮)。
 *
 * 【为什么】nsys(09-15, 512 token 一块)把预填第一大户定在 vq_dequant_kernel 31.9% + 它喂的 f16 GEMM ~24%。
 * 算清楚才知道病不在格式: 512 token × top-6 ÷ 384 专家 ⇒ **每个专家平均只摊到 ne ≈ 8 个 token**, 而为这 8 个
 * token 要把该专家 35.4M 个权重整份解成 f16(写 27 GB + GEMM 再读 27 GB/层 ≈ 225 ms), 真正有用的算力只有
 * 2.7 ms —— 算术强度只有 8。换暂存格式(f16→NVFP4)只能把 54 GB 减到 14 GB(2.8×), 减不掉"搬运"本身。
 * 融合之后每层只读一遍 blob 2.63 GB ⇒ ~11 ms, 这才是 15× 的那条路。
 *
 * 【做法】沿用老路的"按专家计数排序"脚手架(perm/off/cnt 不动, reduce 不动), 只把中间那段
 * 「逐专家 dequant → cuBLAS」换成: 一个 block 载一次码本, 把它负责的那些权重行各解一次,
 * 每行当场与该专家名下的 ≤8 个 token 激活做点积。权重行只解一次、只读一次, 激活被复用 ne 遍。
 *
 * 【与解码融合核的关系】解码路(cuda_v41_4.inc.cu)的 grid.y 是 (token,pick) 对 —— 一个 pair 一份权重,
 * 拿到预填上会把同一专家的权重读 ne 遍。这里的 grid.y 是「工作项」= (专家, token 段, 段内个数),
 * 差别只在内层多了一重 token 循环, 点积原语 v41_vq_row_dot 的解码部分逐式照搬(同一个 blob 布局,
 * 同一套码本/行增益/位流语义), 所以两路数值口径一致。
 *
 * 【数值】老路激活/中间 h 走 f16(cuBLAS 半精度), 本路全程 f32 读 + 官方 bf16 边界舍入(与解码融合核同口径)
 * ⇒ 与老路**不是逐位同**, 判据是 NLL/五指标。另: 专家内的 K 维求和顺序由"一 warp 32 lane 分组 + shuffle 规约"
 * 决定, 与老路的 cuBLAS 不同, 这本身就不可能逐位同。 */

/* ★第四轮(R 行分块)判负, 已回退到这一版★ —— 判负理由写在文件末尾的注释里, 别再重走。 */
/* ★09-15 段 5 判负存档: NT 加不上去★
 * 想法: 一个工作项只带 NT 个 token, 专家有 ne 个 token 就要 ⌈ne/NT⌉ 个工作项, 每个都把这个专家的整份权重
 * 重解一遍 —— 块 512 时 ne≈8 正好一遍, 块 2048 时 ne≈32 就**重读 4 遍**。这也是"块开大了预填一点不省"
 * (512→1024 只快 1%)的根子。
 * 实撞: NT=32 / NT=16 / 连把 NT 做成模板参数的 NT=8 **全部报 "too many resources requested for launch"**
 * —— 不是跑慢, 是起不来。一 block 32 个 warp = 1024 线程, 每线程寄存器上限 64 个, 现版把 acc[] 放在
 * local memory 才刚好塞下; 一旦模板化把它提进寄存器就超预算。要腾寄存器就得把块降到 512 线程, 而码本
 * 64 KB 让每个 SM 只驻 1 个 block, **块大小就是占用率**, 降块 = SM 的 warp 数砍半(第四轮量过, 亏)。
 * ⇒ 这 4 倍在"码本进 shared"的形态里吃不掉。正路是 speed.md 段 5: 专家解到 **NVFP4 暂存 + 板子原生
 * FP4 张量核 GEMM**(S0 实测 284~356 TFLOPS), 那条路根本不需要码本待在 shared 里。 */
#define V41_VQP_NT     8u          /* 一个工作项最多带几个 token(acc 在 local memory, 见上) */
#define V41_VQP_ITERS  8u          /* 一 warp 循环几行: 码本只搬一次, 摊薄到 32×8 = 256 行/block */
#define V41_VQP_ROWS   (32u * V41_VQP_ITERS)

/* 排序后的激活, f32(老路是 f16; 融合核直接读 f32, 少一次精度往返) */
__global__ static void vqp_gather32_kernel(float *xs, const float *x, const int32_t *perm, uint32_t K, uint32_t IN) {
    const uint32_t i = blockIdx.x;
    const uint32_t t = (uint32_t)perm[i] / K;
    const float *src = x + (uint64_t)t * IN;
    float *dst = xs + (uint64_t)i * IN;
    for (uint32_t d = threadIdx.x; d < IN; d += blockDim.x) dst[d] = src[d];
}

/* R 行权重 × nt 个激活向量。解码部分(位流 → 码本 16 B → 8 个 half)与 v41_vq_row_dot 同式;
 * 差别只有一处: 激活按组读一次, 内层对 R 行复用 —— 这就是本文件第四轮的全部内容。
 * acc 出来时已经过 warp 规约并乘上行增益(含反修覆盖), 只有 lane 0 的值有效。 */
/* 一行权重 × nt 个激活向量。解码部分与 v41_vq_row_dot 同式(位流 → 码本 16 B → 8 个 half),
 * 只是把"一个 x"换成"nt 个 x", 权重解一次用 nt 遍 —— 这就是本文件存在的全部理由。 */
/* V3/EXT 与解码路同义(见 cuda_vq_row.inc.cu 文件头): v3 的位流恒 12 位主流, 第 13 位在位平面; 码本 8 个 E4M3。
 * ★这条路是预填专家的唯一实现★ —— 漏改它的后果不是变慢而是预填出假权重(解码对、预填错, 两条路的五指标会分叉)。 */
template <int V3, int EXT>
__device__ __forceinline__ static void v41_vq_row_dotN(const v41_vq_mat &m, uint32_t r, const float *xs,
                                                       uint32_t xstride, uint32_t nt, const uint8_t *cbs, float *acc) {
    const uint64_t i0 = (uint64_t)r * m.nidx_row;
    const uint32_t MB = V3 ? 12u : m.nbit;                       /* 主流位宽 */
    const uint8_t *ex = EXT ? m.ex + (size_t)r * ((m.nidx_row + 7u) >> 3) : NULL;
    for (uint32_t t = 0; t < V41_VQP_NT; t++) acc[t] = 0.f;
    for (uint32_t j = threadIdx.x & 31u; j < m.nidx_row; j += 32u) {
        const uint64_t bit = (V3 ? (uint64_t)j : (i0 + j)) * MB + (V3 ? (uint64_t)r * ((m.nidx_row * MB + 7u) / 8u) * 8u : 0u), by = bit >> 3;
        uint32_t wv; memcpy(&wv, m.ix + by, 4);
        uint32_t v = (wv >> (bit & 7)) & (EXT ? 0xFFFu : m.imsk);
        if (EXT) v |= (uint32_t)((ex[j >> 3] >> (j & 7)) & 1u) << 12;
        float f0x, f0y, f1x, f1y, f2x, f2y, f3x, f3y;
        if (V3) {   /* E4M3 走硬件指令, 不走标量 C 实现 —— 见 cuda_vq_row.inc.cu 里 v41_e4m3x2_to_half2 的教训 */
            const uint2 w = *(const uint2 *)(cbs + (size_t)v * 8u);
            const float2 e0 = __half22float2(v41_e4m3x2_to_half2(w.x)), e1 = __half22float2(v41_e4m3x2_to_half2(w.x >> 16)),
                         e2 = __half22float2(v41_e4m3x2_to_half2(w.y)), e3 = __half22float2(v41_e4m3x2_to_half2(w.y >> 16));
            f0x = e0.x; f0y = e0.y; f1x = e1.x; f1y = e1.y; f2x = e2.x; f2y = e2.y; f3x = e3.x; f3y = e3.y;
        } else {
            const uint4 c = *(const uint4 *)(cbs + (size_t)v * 16u);
            __half2 h0, h1, h2, h3; memcpy(&h0, &c.x, 4); memcpy(&h1, &c.y, 4); memcpy(&h2, &c.z, 4); memcpy(&h3, &c.w, 4);
            const float2 g0 = __half22float2(h0), g1 = __half22float2(h1), g2 = __half22float2(h2), g3 = __half22float2(h3);
            f0x = g0.x; f0y = g0.y; f1x = g1.x; f1y = g1.y; f2x = g2.x; f2y = g2.y; f3x = g3.x; f3y = g3.y;
        }
        const float2 f0 = make_float2(f0x, f0y), f1 = make_float2(f1x, f1y), f2 = make_float2(f2x, f2y), f3 = make_float2(f3x, f3y);
        for (uint32_t t = 0; t < nt; t++) {
            const float *xt = xs + (uint64_t)t * xstride + (size_t)j * 8u;
            const float4 xa = *(const float4 *)xt, xb = *(const float4 *)(xt + 4u);
            acc[t] += f0.x * xa.x + f0.y * xa.y + f1.x * xa.z + f1.y * xa.w +
                      f2.x * xb.x + f2.y * xb.y + f3.x * xb.z + f3.y * xb.w;
        }
    }
    __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2);
    const float g = __half2float(gh) * (m.gov ? m.gov[r] : 1.0f);
    for (uint32_t t = 0; t < nt; t++) {
        for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
        acc[t] *= g;
    }
}

/* 工作项: 专家 e 的第 [t0, t0+nt) 个 token(排序后的下标 = off[e]+t0 起) */
typedef struct { int32_t e, t0, nt; } vqp_item;

/* gate 与 up ★分两发★(2026-09-15 实撞): 原来一个核里同时留 g[ITERS][NT] 与 u[ITERS][NT] = 每线程 128 个
 * 浮点累加器, 远超寄存器预算, 全溢到 local memory —— 融合只换来 +5%。分开后每线程只留当前行的 acc[NT]=8 个,
 * 代价是码本多载一遍(64 KB/block, 走 L2, 可忽略)。
 * which=0: 算 gate 落 g32; which=1: 算 up, 就地读 g32 做 clamp+SwiGLU 写 h32(官方 Expert 的顺序与舍入点)。 */
template <int V3, int EXT>
__global__ static void vqp_fused_gu_kernel(float *dst, const float *g32, const uint8_t *blob, const vqp_item *items,
                                           const float *xs, const uint32_t *off, uint32_t IN, uint32_t MID,
                                           float clamp, uint32_t cb_bytes, int which) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const vqp_item it = items[blockIdx.y];
    const uint32_t nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
    const v41_vq_mat m = v41_vq_open<V3>(blob, it.e, which, MID, IN, NULL);
    if (!m.ok || m.nc * (V3 ? 8u : 16u) != cb_bytes) return;
    const uint32_t r0 = blockIdx.x * V41_VQP_ROWS + (threadIdx.x >> 5);
    const float *xb = xs + (uint64_t)base * IN;
    v41_vq_cb_to_shared(vqsh, m.cb, cb_bytes);
    __syncthreads();
    for (uint32_t i = 0; i < V41_VQP_ITERS; i++) {
        const uint32_t r = r0 + i * 32u;
        if (r >= MID) break;
        float acc[V41_VQP_NT];
        v41_vq_row_dotN<V3, EXT>(m, r, xb, IN, nt, vqsh, acc);
        if ((threadIdx.x & 31u) != 0) continue;
        for (uint32_t t = 0; t < nt; t++) {
            const uint64_t o = (uint64_t)(base + t) * MID + r;
            if (!which) { dst[o] = v41_bf16r(acc[t]); continue; }
            float gi = g32[o], ui = v41_bf16r(acc[t]);
            if (clamp > 0.f) { if (gi > clamp) gi = clamp; if (ui > clamp) ui = clamp; if (ui < -clamp) ui = -clamp; }
            const float sg = gi / (1.0f + expf(-gi));
            dst[o] = v41_bf16r(sg * ui);
        }
    }
}

/* down: ys[排序位][OUT] = bf16(W2·h) —— 老路写的也是这个缓冲, 下游 reduce 一字不改 */
template <int V3, int EXT>
__global__ static void vqp_fused_down_kernel(float *ys, const uint8_t *blob, const vqp_item *items, const float *h,
                                             const uint32_t *off, uint32_t MID, uint32_t OUT, uint32_t cb_bytes, const float *gr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    const vqp_item it = items[blockIdx.y];
    const uint32_t nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
    const v41_vq_mat md = v41_vq_open<V3>(blob, it.e, 2, OUT, MID, gr ? gr + (size_t)it.e * OUT : NULL);
    const uint32_t r0 = blockIdx.x * V41_VQP_ROWS + (threadIdx.x >> 5);
    if (!md.ok || md.nc * (V3 ? 8u : 16u) != cb_bytes) {   /* 载荷不对: 负责的行全写 0, 不给下游留脏值 */
        if ((threadIdx.x & 31u) == 0)
            for (uint32_t i = 0; i < V41_VQP_ITERS; i++) {
                const uint32_t r = r0 + i * 32u;
                if (r < OUT) for (uint32_t t = 0; t < nt; t++) ys[(uint64_t)(base + t) * OUT + r] = 0.f;
            }
        return;
    }
    v41_vq_cb_to_shared(vqsh, md.cb, cb_bytes);
    __syncthreads();
    const float *hb = h + (uint64_t)base * MID;
    for (uint32_t i = 0; i < V41_VQP_ITERS; i++) {
        const uint32_t r = r0 + i * 32u;
        if (r >= OUT) break;
        float acc[V41_VQP_NT];
        v41_vq_row_dotN<V3, EXT>(md, r, hb, MID, nt, vqsh, acc);
        if ((threadIdx.x & 31u) == 0)
            for (uint32_t t = 0; t < nt; t++) ys[(uint64_t)(base + t) * OUT + r] = v41_bf16r(acc[t]);
    }
}

static struct { vqp_item *d; uint64_t cap; float *h32, *g32, *xs32; uint64_t h32_cap, g32_cap, xs32_cap; uint32_t *doff; uint64_t doff_cap; } g_vqpf;
static int g_vqpf_sh = 0;   /* 0 未判定 / 1 码本进 shared / -1 放不下(那就没有本路, 硬失败) */

/* 返回 0 = 本路不可用(调用方硬失败, 不静默退回老路: 退回去就永远不知道哪条路在跑) */
static int vqp_fused_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert,
                         uint32_t nvalid, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp,
                         const float *x, const int32_t *perm, uint32_t n_expert, uint32_t layer_index, uint32_t ver) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    /* v3 的码本每词 8 个 E4M3(nc8192 = 64 KB, 与今天 nc4096 f16 同大); v2 每词 8 个 f16 */
    const uint32_t cbb = nc * (ver == 3u ? 8u : 16u);
    if (g_vqpf_sh == 0) {
        /* opt-in shared 要按【实际会发的那对实例】开: 开在别的实例上, 发射时照样报 shared 超限。
         * 四对实例一次全开(每对两个核), 哪一对将来被发都已经开好; 失败一个就判本路不可用(不静默退回老路)。 */
        bool og = true, od = true;
#define VQP_ATTR(V3, EXT) do { \
            og = og && cudaFuncSetAttribute(vqp_fused_gu_kernel<V3, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess; \
            od = od && cudaFuncSetAttribute(vqp_fused_down_kernel<V3, EXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cbb) == cudaSuccess; \
        } while (0)
        if (ver == 3u) { if (nbit == 13u) VQP_ATTR(1, 1); else VQP_ATTR(1, 0); }
        else VQP_ATTR(0, 0);
#undef VQP_ATTR
        (void)cudaGetLastError();
        g_vqpf_sh = (og && od) ? 1 : -1;
        fprintf(stderr, "ds4: [vq-prefill] 融合路码本 %u KB/本(DQVL v%u, %u 位) → %s\n", cbb >> 10, ver, nbit,
                g_vqpf_sh == 1 ? "进 shared" : "★放不下, 本路不可用★");
    }
    if (g_vqpf_sh != 1) return 0;
    /* 工作项: 每专家按 NT 切段 */
    uint32_t nit = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) nit += (cnt[e] + V41_VQP_NT - 1u) / V41_VQP_NT;
    if (!nit) return 0;
    vqp_item *ih = (vqp_item *)malloc((size_t)nit * sizeof(vqp_item));
    if (!ih) return 0;
    uint32_t k = 0;
    for (uint32_t e = 0; e < n_total_expert; e++)
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += V41_VQP_NT) {
            const uint32_t nt = cnt[e] - t0 < V41_VQP_NT ? cnt[e] - t0 : V41_VQP_NT;
            ih[k].e = (int32_t)e; ih[k].t0 = (int32_t)t0; ih[k].nt = (int32_t)nt; k++;
        }
    int ok = vqp_grow((void **)&g_vqpf.d, &g_vqpf.cap, nit, sizeof(vqp_item), "items") &&
             vqp_grow((void **)&g_vqpf.doff, &g_vqpf.doff_cap, n_total_expert + 1u, sizeof(uint32_t), "off") &&
             vqp_grow((void **)&g_vqpf.xs32, &g_vqpf.xs32_cap, ((uint64_t)nvalid + V41_VQN_PAD) * IN, sizeof(float), "xs32") &&
             vqp_grow((void **)&g_vqpf.h32, &g_vqpf.h32_cap, ((uint64_t)nvalid + V41_VQN_PAD) * MID, sizeof(float), "h32") &&
             vqp_grow((void **)&g_vqpf.g32, &g_vqpf.g32_cap, ((uint64_t)nvalid + V41_VQN_PAD) * MID, sizeof(float), "g32");
    if (ok) ok = cudaMemcpyAsync(g_vqpf.d, ih, (size_t)nit * sizeof(vqp_item), cudaMemcpyHostToDevice, g_cur_stream) == cudaSuccess &&
                 cudaMemcpyAsync(g_vqpf.doff, off_h, (size_t)(n_total_expert + 1u) * sizeof(uint32_t), cudaMemcpyHostToDevice, g_cur_stream) == cudaSuccess;
    free(ih);
    if (!ok) { (void)cudaGetLastError(); fprintf(stderr, "ds4: [vq-prefill] L%u 融合路暂存/拷贝失败\n", layer_index); return 0; }
    vqp_gather32_kernel<<<nvalid, 256, 0, g_cur_stream>>>(g_vqpf.xs32, x, perm, n_expert, IN);
    if (!cuda_ok(cudaGetLastError(), "vq prefill gather32")) return 0;
    /* ★2026-09-20: 这里原来无条件转给 vqn_prefill_run(段 5 的 NVFP4 张量核路), 现在走回本文件的融合核★
     *
     * 【为什么退】NVFP4 那条路为了用上板子的 FP4 张量核, 一层里要做**三次重量化**: VQ 解出来的 f16 权重 → E2M1、
     * 激活 x → E2M1、swiglu 出口 h → E2M1(每 16 个元素一个 e4m3 缩放, 尾数只剩 1 位)。它的文件头自认
     * "比现役融合路多一次量化, 判据是五指标/NLL" —— 而那个判据**从来没跑过**。
     * 09-19/20 用 wt2 512 引擎↔同文件 Python 学生量出来了(同一份 fp4 文件, 只换引擎):
     *   09-12 引擎(无 NVFP4)      Σmin 0.7365 / KLD 0.792 / PPL 3.314   ← 基准
     *   今日 + NVFP4 骨架 + 专家   Σmin 0.7043 / KLD 0.898 / PPL 3.849
     *   今日 骨架退 bf16, 专家仍 NVFP4  Σmin 0.7235 / KLD 0.837 / PPL 3.530
     * ⇒ 骨架那道值 0.019, **专家这道值剩下的 0.013**, 两道合计把引擎推离它自己的量化文件 0.032 Σmin,
     * 比 12 bit 降到 11 bit 的位宽代价(文件态 0.02~0.03)还大 —— 这不是"计算态精度选择", 是白掉的质量。
     * 【代价】融合路的权重解码与点积在同一个核里(码本占满 shared ⇒ 每 SM 1 个 block), 09-15 实测 398 ms/层
     * 对 NVFP4 的 64 ms/层。★预填会慢, 数在下面的战役记录里★ —— 但解码路一直走的就是这套数值,
     * 预填与解码从此是同一口径(两套数值本身就是一类 bug 的温床)。
     * 【要重开 NVFP4 怎么做】vqn_prefill_run 原样留着。接回去之前先回答: 拿什么补这 0.013 —— 例如只把
     * 权重走 FP4、激活与 h 保持 bf16(需要 cuBLASLt 支持混合, 09-15 S0 说本机没有算法), 或者 bf16 暂存
     * (字节 4×, 权重仍有 f16→bf16 的 3 位尾数损失, 未量)。**先过五指标, 再谈速度。** */
    const dim3 gm((MID + V41_VQP_ROWS - 1u) / V41_VQP_ROWS, nit);
    const dim3 gd((OUT + V41_VQP_ROWS - 1u) / V41_VQP_ROWS, nit);
    /* 版本/位宽选实例(2026-09-21): v2 与 v3 的载荷布局不同, 选错不报错只出假权重 ⇒ ver 从 blob 头读(调用方传), 位宽从码本词数推。
     * 一个宏发三发核, 三处模板参数必须同档 —— 分开写迟早漏一处。 */
#define VQP_LAUNCH(V3, EXT) do { \
        vqp_fused_gu_kernel<V3, EXT><<<gm, 32u * 32u, cbb, g_cur_stream>>>(g_vqpf.g32, NULL, blob, g_vqpf.d, g_vqpf.xs32, g_vqpf.doff, IN, MID, clamp, cbb, 0); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill fused gate")) return 0; \
        vqp_fused_gu_kernel<V3, EXT><<<gm, 32u * 32u, cbb, g_cur_stream>>>(g_vqpf.h32, g_vqpf.g32, blob, g_vqpf.d, g_vqpf.xs32, g_vqpf.doff, IN, MID, clamp, cbb, 1); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill fused up")) return 0; \
        vqp_fused_down_kernel<V3, EXT><<<gd, 32u * 32u, cbb, g_cur_stream>>>( \
            g_vqp.ys, blob, g_vqpf.d, g_vqpf.h32, g_vqpf.doff, MID, OUT, cbb, g_v41_gr[layer_index < 64u ? layer_index : 0]); \
        return cuda_ok(cudaGetLastError(), "vq prefill fused down"); \
    } while (0)
    if (ver == 3u) {
        if (nbit == 12u) VQP_LAUNCH(1, 0);
        if (nbit == 13u) VQP_LAUNCH(1, 1);
        fprintf(stderr, "ds4: [vq-prefill] L%u DQVL v3 码本 %u 词没有对应实例(只有 12/13 位)\n", layer_index, nc);
        return 0;
    }
    VQP_LAUNCH(0, 0);
#undef VQP_LAUNCH
}

/* ★第四轮判负存档(2026-09-15): "一个 warp 管 R 行, 激活读一次共用" 三个配置全比本版慢★
 * 账是对的(本版每行读 nt×20 KB 激活而权重只 960 B, 比例 170:1), 但落地撞两堵墙:
 *   ①码本 64 KB 占满 shared ⇒ 每 SM 只驻 1 个 block ⇒ **块大小就是占用率**。R 行版寄存器需求
 *     acc[R][NT]+xv[NT][8] 逼着把块降到 256 线程, 每 SM 只剩 8 个 warp: 34.3 → 38.1 s。
 *   ②硬留 1024 线程(每线程 ≤64 寄存器, NT=2/R=16 = 48 个)则编译器仍溢出到 local: 34.3 → 78.1 s。
 *   另: 三个 R 行配置的 PPL 一致落在 15.53(本版 15.17), 说明还有一处没定位的数值差 —— 重启这条路之前
 *     必须先把它查清楚, 别拿"更快"盖过去。
 * 要真正吃掉这 10 倍, 得先解决"码本不占满 shared"(例如码本分段常驻 + 行按段重排), 那是另一件事。 */
