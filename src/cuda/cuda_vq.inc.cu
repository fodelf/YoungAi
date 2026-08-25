/* cuda_vq.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * v2.2 VQ blob 专家前向(DQVL) 解码辅助。
 */
/* ================= v2.2 VQ blob 专家前向 (DQVL) =================
 * 背景: 合一 VQ GGUF 把 routed 专家字节全塞进 blk.L.ffn_exps_vq.blob, base
 * gate/up 张量不在文件里(offset=0/bytes=0)。Metal 侧(ds4_metal.m:20630-21830)
 * 早有完整实现, CUDA 侧此前一行都没有 —— routed_moe_launch 开头的类型闸
 * (gate_type!=16||down_type!=10) 直接 return 0, 表现为 "cuda prefill failed"。
 *
 * 这里按 Metal 的同一语义落地, 不发明新算法:
 *   ① CPU 多线程把活跃专家 dequant 成 f16 scratch。scratch 用 cudaMallocManaged:
 *      GB10 是 Grace+Blackwell 统一内存, CPU 写 GPU 读与 Metal 的
 *      MTLResourceStorageModeShared 同构, 无需显式 H2D 拷贝。
 *   ② selected 里是原始 expert id, 但 gather 后权重按 active 紧凑排布 ⇒ 必须
 *      remap 成 slot(Metal 侧在别处已 remap, CUDA 侧这里自己做)。
 *   ③ GPU kernel 走 f16 mm_id: gate/up 融合 SwiGLU → down 加权累加。
 *   ④ DS4_VQ_GPU=0 回 CPU 参考路(与 Metal 同名开关同语义), 供数值对齐。
 * dequant 复用 vq_fmt.h 的 ds4vq_dequant_f16 —— 三端同一份解码。 */

static void *g_vq_gate_sc = NULL, *g_vq_up_sc = NULL, *g_vq_down_sc = NULL;
static uint64_t g_vq_gu_bytes = 0, g_vq_dn_bytes = 0;
static int32_t *g_vq_sel_dev = NULL;   /* remap 后的 slot 索引(设备端) */
static uint64_t g_vq_sel_bytes = 0;

/* ---- decode 专家 dequant 缓存 ----
 * 同一个专家的 dequant 结果与 token 无关(权重不变), 而 decode 每层只用 n_expert 个
 * 专家、相邻 token 的路由高度重叠 —— 每次 forward 全量重算是纯浪费, 且 dequant 占
 * 了 92.8% 的 GPU 时间。这里给每层留 K 个常驻槽, 命中就直接复用槽里的 f16 权重。
 * 只对 decode(n_tokens==1) 启用: prefill 的 n_active 远大于 K, 会把槽冲干净, 那时
 * 走直通并把缓存置空(scratch 被覆盖, 槽内容不再有效)。 */
/* ★缓存必须是 per-layer 的存储, 不能只有 per-layer 的标记★
 * 第一版把 43 层共用一份 scratch, L1 立刻覆盖 L0 刚写的槽, 下个 token 回到 L0 时
 * 标记说"命中"、槽里装的却是 L42 的权重 ⇒ 输出退化成复读。所以缓存槽的显存按层独立
 * 分配(K×48MB×43 ≈ 16.5GB), prefill 的直通路另走一份共享临时 buffer。 */
#define DS4_VQ_CACHE_SLOTS 8u
#define DS4_VQ_CACHE_LAYERS 64u
static struct {
    int32_t  expert[DS4_VQ_CACHE_SLOTS];   /* 槽内当前专家 id, -1=空 */
    uint64_t used[DS4_VQ_CACHE_SLOTS];     /* LRU 时钟 */
    void    *gate, *up, *down;             /* 本层专属显存(K 槽), NULL=未分配 */
} g_vq_cache[DS4_VQ_CACHE_LAYERS];
static uint64_t g_vq_clock = 0;
static int g_vq_cache_ready = 0;
static uint64_t g_vq_hit = 0, g_vq_miss = 0;

static void cuda_vq_cache_reset_all(void) {
    for (uint32_t l = 0; l < DS4_VQ_CACHE_LAYERS; l++) {
        for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++) {
            g_vq_cache[l].expert[s] = -1;
            g_vq_cache[l].used[s] = 0;
        }
        g_vq_cache[l].gate = g_vq_cache[l].up = g_vq_cache[l].down = NULL;
    }
    g_vq_cache_ready = 1;
}

/* 本层缓存显存(K 槽)按需分配; 分配不到就退回直通(不缓存), 不让它变成硬失败。 */
static int cuda_vq_cache_ensure_mem(uint32_t layer, uint64_t ge, uint64_t de) {
    if (g_vq_cache[layer].gate) return 1;
    void *g = NULL, *u = NULL, *d = NULL;
    const uint64_t gu = (uint64_t)DS4_VQ_CACHE_SLOTS * ge, dn = (uint64_t)DS4_VQ_CACHE_SLOTS * de;
    if (cudaMalloc(&g, gu) != cudaSuccess || cudaMalloc(&u, gu) != cudaSuccess ||
        cudaMalloc(&d, dn) != cudaSuccess) {
        (void)cudaGetLastError();
        if (g) (void)cudaFree(g);
        if (u) (void)cudaFree(u);
        if (d) (void)cudaFree(d);
        return 0;
    }
    g_vq_cache[layer].gate = g; g_vq_cache[layer].up = u; g_vq_cache[layer].down = d;
    return 1;
}

typedef struct {
    const void *model_map; const uint8_t *blob; const uint32_t *active_ids;
    uint16_t *gbase, *ubase, *dbase;
    uint64_t down_offset, down_expert_bytes;
    uint32_t in, mid, out_dim, lo, hi;
    volatile int *err;
} cuda_vq_gather_task;

/* 逐专家 dequant, 各线程写不相交 scratch 段 ⇒ 无锁。与 Metal 的
 * ds4_vq_gather_worker 逐行同义(含冷 w2 从 base go1b 34B/256el 展开 ±d)。 */
static void *cuda_vq_gather_worker(void *arg) {
    cuda_vq_gather_task *t = (cuda_vq_gather_task *)arg;
    for (uint32_t i = t->lo; i < t->hi && !*t->err; i++) {
        const uint32_t e = t->active_ids[i];
        uint16_t *dg = t->gbase + (uint64_t)i * t->mid * t->in;
        uint16_t *du = t->ubase + (uint64_t)i * t->mid * t->in;
        uint16_t *dd = t->dbase + (uint64_t)i * t->out_dim * t->mid;
        uint64_t o1 = ds4vq_slot(t->blob, (int)e, 0);
        uint64_t o3 = ds4vq_slot(t->blob, (int)e, 1);
        uint64_t o2 = ds4vq_slot(t->blob, (int)e, 2);
        int rc1 = (!o1) ? -9 : ds4vq_dequant_f16(t->blob + o1, dg, (int)t->mid, (int)t->in);
        int rc3 = (rc1 == 0 && o3) ? ds4vq_dequant_f16(t->blob + o3, du, (int)t->mid, (int)t->in) : (!o3 ? -9 : 0);
        if (rc1 != 0 || rc3 != 0) {
            fprintf(stderr, "ds4: [cuda-vq-gather-err] e=%u o1=%llu o3=%llu rc1=%d rc3=%d mid=%u in=%u\n",
                    e, (unsigned long long)o1, (unsigned long long)o3, rc1, rc3, t->mid, t->in);
            *t->err = 1; return NULL;
        }
        if (o2) {
            int rc2 = ds4vq_dequant_f16(t->blob + o2, dd, (int)t->out_dim, (int)t->mid);
            if (rc2 != 0) {
                fprintf(stderr, "ds4: [cuda-vq-gather-err] e=%u o2=%llu rc2=%d out=%u mid=%u\n",
                        e, (unsigned long long)o2, rc2, t->out_dim, t->mid);
                *t->err = 1; return NULL;
            }
        } else {
            /* 冷 w2: blob 的 which=2 槽缺席 ⇒ 从 base go1b 字节展开。base down 是
             * 影子张量(bytes=0)时硬失败, 不读 offset 0 的垃圾当权重。 */
            if (t->down_expert_bytes == 0 || t->down_offset == 0) {
                fprintf(stderr, "ds4: [cuda-vq-gather-err] e=%u 冷 w2 槽缺失且 base down 不在文件里"
                                "(影子张量) -- aborting (no silent quality downgrade)\n", e);
                *t->err = 1; return NULL;
            }
            const uint8_t *sd = (const uint8_t *)t->model_map + t->down_offset + (uint64_t)e * t->down_expert_bytes;
            const uint64_t nblk_row = t->mid / 256u;
            for (uint32_t r = 0; r < t->out_dim; r++) {
                const uint8_t *rb = sd + (uint64_t)r * nblk_row * 34u;
                uint16_t *orow = dd + (uint64_t)r * t->mid;
                for (uint64_t b = 0; b < nblk_row; b++) {
                    uint16_t dsc; memcpy(&dsc, rb + b * 34u, 2);
                    const uint8_t *sg = rb + b * 34u + 2;
                    uint16_t *o = orow + b * 256u;
                    for (int k = 0; k < 256; k++)
                        o[k] = (sg[k >> 3] >> (k & 7)) & 1 ? dsc : (uint16_t)(dsc ^ 0x8000u);
                }
            }
        }
    }
    return NULL;
}

/* GPU 版 VQ 载荷解码。CPU 侧逐元素查表是 decode 0.6 t/s 的主因(每 token 约 6.5G
 * 元素), 而这活是纯并行查表: 值 = 码本[idx][d] * g_r[row]。blob 已随模型 mmap 被
 * cudaHostRegister, kernel 可经 UVA 直读 —— 且读的是压缩态(约 3MB/专家)而不是
 * dequant 后的 f16(16MB/专家), 字节数还少 5 倍。
 * 位流解析与 vq_fmt.h 的 ds4vq_dequant_f16 逐字同义(nbit 由 nc 推导, 9/10bit 走
 * 三字节窗口, 8bit 走字节流)。 */
__global__ static void vq_dequant_kernel(
        __half *out, const uint8_t *pay,
        uint32_t rows, uint32_t cols, uint32_t dim, uint32_t nc, uint32_t nbit) {
    /* 每 block 一行。两次优化尝试均被实测否决, 记在这里免得再走一遍:
     *   ① 码本搬 shared(grid=rows): 2048 个 block 各搬一次 4KB, 净变慢;
     *   ② 多行/block + shared 摊薄加载: 速度未验先崩 —— 输出塌成全 BOS。
     * 码本只有几 KB, 本来就常驻 L1/L2 被所有 block 共享, 全局直读是当前正确且不慢的解。 */
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / dim;
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    __half gh; memcpy(&gh, gr + (size_t)r * 2, 2);
    const float g = __half2float(gh);
    const size_t i0 = (size_t)r * nidx_row;
    __half *orow = out + (size_t)r * cols;
    for (uint32_t i = threadIdx.x; i < nidx_row; i += blockDim.x) {
        const size_t gi = i0 + i;
        uint32_t v;
        if (nbit == 8) {
            v = ix[gi];
        } else {
            /* 一次 32 位读取代三次单字节 load: nbit<=24 时目标位段必定落在这个窗口内,
             * 取值与 vq_fmt.h 的三字节拼法逐位相同。原写法每 warp 发 96 条字节 load 去
             * 覆盖仅 36 字节的地址范围, 而 GPU 最小访存事务是 32 字节 —— 几十倍放大,
             * 正是 dequant 只跑到 6.7GB/s 的原因。
             * 尾部越读的至多 3 字节仍在 blob/相邻张量内(只取低位, 不影响取值)。 */
            const size_t bit = gi * nbit;
            const size_t by = bit >> 3;
            uint32_t w; memcpy(&w, ix + by, 4);
            v = (w >> (bit & 7)) & imsk;
        }
        const uint8_t *c = cb + (size_t)v * dim * 2;
        __half *o = orow + (size_t)i * dim;
        for (uint32_t d = 0; d < dim; d++) {
            __half ch; memcpy(&ch, c + (size_t)d * 2, 2);
            o[d] = __float2half(__half2float(ch) * g);
        }
    }
}

/* ---- decode 全 GPU 路: 消除每层的 D2H 往返 ----
 * 实测 GPU 利用率只有 6%(P0/2405MHz/13W): 瓶颈不是算力也不是带宽, 而是每层都要把
 * selected 同步拷回主机做去重+remap —— 43 层就是每 token 43 次"等 GPU 全停→CPU 算
 * →再启动"。这里让 kernel 自己读 blob 的偏移表, 按 (token,pick) 对直接解码到各自的
 * scratch 段, host 侧一个字节都不用读回来。
 * 代价: 同一专家被两个 pick 选中时会解码两次。decode 只有 n_expert 个 pick, 重复概率
 * 低且每次只多一份工作; 换掉的是整整一次全设备同步, 净赚。prefill(pair 数上千)重复
 * 会爆炸, 所以那条路仍走 host 去重。 */
__device__ __forceinline__ static uint64_t vq_slot_dev(const uint8_t *blob, int e, int which) {
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    return off;
}

__global__ static void vq_dequant_pairs_kernel(
        __half *gout, __half *uout, __half *dout,
        const uint8_t *blob, const int32_t *sel,
        const uint8_t *down_base, uint64_t down_ebytes,
        uint32_t IN, uint32_t MID, uint32_t OUT) {
    const uint32_t pair = blockIdx.z, which = blockIdx.y;
    const int32_t e = sel[pair];
    if (e < 0) return;

    const uint32_t exp_rows = (which == 2u) ? OUT : MID;
    const uint32_t exp_cols = (which == 2u) ? MID : IN;
    __half *out = (which == 0u) ? (gout + (uint64_t)pair * MID * IN)
                : (which == 1u) ? (uout + (uint64_t)pair * MID * IN)
                                : (dout + (uint64_t)pair * OUT * MID);
    const uint64_t off = vq_slot_dev(blob, e, (int)which);
    const uint32_t r = blockIdx.x;
    if (r >= exp_rows) return;

    if (off == 0u) {
        /* 冷 w2: base go1b 34B/256el sign 字节 → ±d。base down 缺席时写 0 而不是读垃圾
         * (host 侧已在直通路对影子张量硬失败; 这里保守置零, 不静默用错误权重)。 */
        if (which != 2u || down_base == NULL || down_ebytes == 0u) return;
        const uint32_t nblk_row = MID / 256u;
        const uint8_t *rb = down_base + (uint64_t)e * down_ebytes + (size_t)r * nblk_row * 34u;
        uint16_t *orow = (uint16_t *)(out + (size_t)r * MID);
        for (uint32_t b = threadIdx.x; b < nblk_row; b += blockDim.x) {
            uint16_t dsc; memcpy(&dsc, rb + (size_t)b * 34u, 2);
            const uint8_t *sg = rb + (size_t)b * 34u + 2;
            uint16_t *o = orow + (size_t)b * 256u;
            for (int k = 0; k < 256; k++)
                o[k] = ((sg[k >> 3] >> (k & 7)) & 1) ? dsc : (uint16_t)(dsc ^ 0x8000u);
        }
        return;
    }

    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    if (rows != exp_rows || cols != exp_cols) return;
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;

    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)n16 * d16 * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32u) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / d16;
    __half gh; memcpy(&gh, gr + (size_t)r * 2, 2);
    const float g = __half2float(gh);
    const size_t i0 = (size_t)r * nidx_row;
    __half *orow = out + (size_t)r * cols;
    for (uint32_t i = threadIdx.x; i < nidx_row; i += blockDim.x) {
        const size_t gi = i0 + i;
        uint32_t v;
        if (nbit == 8u) {
            v = ix[gi];
        } else {
            const size_t bit = gi * nbit, by = bit >> 3;
            uint32_t w; memcpy(&w, ix + by, 4);
            v = (w >> (bit & 7)) & imsk;
        }
        const uint8_t *c = cb + (size_t)v * d16 * 2;
        __half *o = orow + (size_t)i * d16;
        for (uint32_t d = 0; d < d16; d++) {
            __half ch; memcpy(&ch, c + (size_t)d * 2, 2);
            o[d] = __float2half(__half2float(ch) * g);
        }
    }
}

/* 冷 w2(blob 无 which=2 槽): base go1b 34B/256el sign 字节 → ±d 展开。GPU 版。 */
__global__ static void vq_cold_w2_kernel(
        __half *out, const uint8_t *sd, uint32_t out_dim, uint32_t mid) {
    const uint32_t nblk_row = mid / 256u;
    const uint32_t r = blockIdx.x;
    if (r >= out_dim) return;
    const uint8_t *rb = sd + (size_t)r * nblk_row * 34u;
    __half *orow = out + (size_t)r * mid;
    for (uint32_t b = threadIdx.x; b < nblk_row; b += blockDim.x) {
        uint16_t dsc; memcpy(&dsc, rb + (size_t)b * 34u, 2);
        const uint8_t *sg = rb + (size_t)b * 34u + 2;
        uint16_t *o = (uint16_t *)(orow + (size_t)b * 256u);
        for (int k = 0; k < 256; k++)
            o[k] = ((sg[k >> 3] >> (k & 7)) & 1) ? dsc : (uint16_t)(dsc ^ 0x8000u);
    }
}

/* 载荷头解析(host 侧, 与 vq_fmt.h 同布局): 0=成功 */
static int cuda_vq_pay_hdr(const uint8_t *pay, int exp_rows, int exp_cols,
                           uint32_t *rows, uint32_t *cols, uint32_t *dim, uint32_t *nc, uint32_t *nbit) {
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return -1;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t r32, c32; memcpy(&r32, pay + 8, 4); memcpy(&c32, pay + 12, 4);
    if ((int)r32 != exp_rows || (int)c32 != exp_cols) return -2;
    int nb = 0; while ((1 << nb) < (int)n16) nb++; if (nb < 1) nb = 1;
    /* dequant kernel 把码本整块放 shared(48KB/block 上限)。当前配方 nc=512×dim=4=4KB,
     * 余量充足; 真超了就明着失败, 不静默 launch 出错。 */
    if ((size_t)n16 * d16 * 2u > 48u * 1024u) {
        fprintf(stderr, "ds4: [cuda-vq] 码本 %u×%u=%zuB 超 shared 上限 48KB\n",
                (unsigned)n16, (unsigned)d16, (size_t)n16 * d16 * 2u);
        return -3;
    }
    *rows = r32; *cols = c32; *dim = d16; *nc = n16; *nbit = (uint32_t)nb;
    return 0;
}

/* scratch 用纯设备内存(cudaMalloc)而不是托管内存。托管内存会在 CPU/GPU 间按需页迁移,
 * 而 prefill 一层就要写 4GB scratch —— 迁移开销吃掉了绝大部分时间。dequant 现在全在
 * GPU 上做, CPU 侧只有两条调试路(DS4_VQ_CPU_GATHER / DS4_VQ_GPU=0)需要碰它, 那两条
 * 各自走显式 cudaMemcpy, 不该让生产路径为它们背上托管内存的代价。 */
static int cuda_vq_ensure_scratch(uint64_t need_gu, uint64_t need_dn) {
    if (need_gu > g_vq_gu_bytes) {
        if (g_vq_gate_sc) (void)cudaFree(g_vq_gate_sc);
        if (g_vq_up_sc) (void)cudaFree(g_vq_up_sc);
        g_vq_gate_sc = g_vq_up_sc = NULL;
        if (cudaMalloc(&g_vq_gate_sc, need_gu) != cudaSuccess ||
            cudaMalloc(&g_vq_up_sc, need_gu) != cudaSuccess) {
            (void)cudaGetLastError();
            g_vq_gu_bytes = 0;
            return 0;
        }
        g_vq_gu_bytes = need_gu;
    }
    if (need_dn > g_vq_dn_bytes) {
        if (g_vq_down_sc) (void)cudaFree(g_vq_down_sc);
        g_vq_down_sc = NULL;
        if (cudaMalloc(&g_vq_down_sc, need_dn) != cudaSuccess) {
            (void)cudaGetLastError();
            g_vq_dn_bytes = 0;
            return 0;
        }
        g_vq_dn_bytes = need_dn;
    }
    return 1;
}

/* f16 mm_id MoE, 两阶段。
 * 第一版是"每 (token,pick) 一个 block、每线程串行读整行", decode 时只有 6 个 block ⇒
 * SM 几乎全空转, 且线程内跨行跳读完全不合并访存, 实测只有带宽上限的 ~2%。
 * 现在: ① 每 warp 负责一行, warp 内 32 lane 沿 IN 连续取(合并访存)后 shfl 规约;
 *       ② 把 MID/OUT 维切成 grid.z, decode 也能铺满 SM。
 * clamp 语义与 Metal CPU 参考路逐字一致: gate 只截上界, up 双向截。 */
#define DS4_VQ_WARPS_PER_BLOCK 8u
/* 码本 shared 容量(半精度个数): 当前配方 nc=512×dim=4=2048, 留一倍余量 */
#define DS4_VQ_CB_CAP_HALFS 4096u
/* 每 warp 位流暂存字数: 一行最长 4096列/4×9bit=1152B, 取 304 words=1216B 留余量 */
#define DS4_VQ_BITWORDS 304u

__device__ __forceinline__ static float vq_warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* ---- 融合路: 解码即用, 不落 f16 中间权重 ----
 * 账: decode 每 token 只需 6 专家×43 层的压缩态 blob ≈ 774MB; 而"先 dequant 成 f16
 * 再 matmul"要写 12.4GB + 读回 12.4GB = 24.8GB —— 放大 32 倍, 4.7 t/s 就是这么来的。
 * 那份 f16 解出来只被用一次就丢, 根本不该存在。这里让 matmul 直接从 blob 解码,
 * 内存流量回到压缩态本身。
 * 只用于 decode: prefill 时同一专家被上千 token 共用, 落一次 f16 再复用才划算。 */

/* 解码 (expert e, which) 矩阵第 row 行并与 vec 点乘; warp 内 lane 沿列分摊。
 * 返回的是本 lane 的部分和, 调用方做 warp 规约。 */
/* 诊断用: 0=完整 1=只解位流 2=位流+码本读(不乘 x) —— 二分定位耗时段 */
__device__ int g_vq_exp_mode = 0;
/* kernel 内自计时(零权限替代 ncu): [0]=位流预取 cycles [1]=解码+乘加 cycles
 * [2]=样本数。只在 DS4_VQ_CYC=1 时写, 用 clock64() 读 SM 时钟。 */
__device__ unsigned long long g_vq_cyc[3] = {0, 0, 0};
__device__ int g_vq_cyc_on = 0;

