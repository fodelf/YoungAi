/* cuda_vq.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * v2.2 VQ blob 专家前向(DQVL) 解码辅助。
 */
/* ================= v2.2 VQ blob 专家前向 (DQVL) =================
 * 背景: 合一 VQ GGUF 把 routed 专家字节全塞进 blk.L.ffn_exps_vq.blob, base
 * gate/up 张量不在文件里(offset=0/bytes=0)。两条生产路:
 *   decode(n_tokens ≤ fuse_max): fused2 核直接从压缩态 blob 解码即乘(cuda_vq_fused2_*);
 *   prefill(大批): 逐专家 dequant 成 f16 + cuBLAS GEMM(cuda_vq_prefill.inc.cu)。
 * 09-06 删掉的老 prefill 路(CPU 多线程 gather / 托管 scratch / 逐层专家缓存 / 全层活跃专家
 * dequant 落 12.9 GB scratch / 逐 (token,pick) warp 核)记在 fable5。本片只留公共件: GPU 载荷
 * dequant 核(位流解析与 vq_fmt.h 的 ds4vq_dequant_f16 逐字同义)、冷 w2 展开、载荷头解析。 */


/* GPU 版 VQ 载荷解码。CPU 侧逐元素查表是 decode 0.6 t/s 的主因(每 token 约 6.5G
 * 元素), 而这活是纯并行查表: 值 = 码本[idx][d] * g_r[row]。blob 已随模型 mmap 被
 * cudaHostRegister, kernel 可经 UVA 直读 —— 且读的是压缩态(约 3MB/专家)而不是
 * dequant 后的 f16(16MB/专家), 字节数还少 5 倍。
 * 位流解析与 vq_fmt.h 的 ds4vq_dequant_f16 逐字同义(nbit 由 nc 推导, 9/10bit 走
 * 三字节窗口, 8bit 走字节流)。 */
/* gov(2026-09-13, 权重侧反修): 逐行增益【缩放因子】覆盖表 [rows] f32, NULL = 按载荷原样。
 * 用乘的不是替换: 盘上的 g_r 一个字节不动, 插件只说"这一行乘多少", 不挂就是裸底座。 */
__global__ static void vq_dequant_kernel(
        __half *out, const uint8_t *pay,
        uint32_t rows, uint32_t cols, uint32_t dim, uint32_t nc, uint32_t nbit, const float *gov) {
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
    const float g = __half2float(gh) * (gov ? gov[r] : 1.0f);
    const size_t i0 = (size_t)r * nidx_row;
    __half *orow = out + (size_t)r * cols;
    /* dim=4 快路(09-06, prefill 一块 4096 token 要发 3 万次本核, 占 28%): 码本条目 8 B 按两个
     * 32 位读, 4 个 half 结果打包成一次 8 B 存 —— 原来每索引 4 次 2 B 散存, 相邻线程相距 8 B,
     * 一条 warp 存指令只盖 256 B 里的 64 B。数值逐位同(同为 rn 舍入)。码本基址不 4 对齐时走
     * 通用路(blob 槽 4 B 对齐是 fused2 探针也依赖的前提)。 */
    if (dim == 4u && (((uintptr_t)cb) & 3u) == 0u) {
        for (uint32_t i = threadIdx.x; i < nidx_row; i += blockDim.x) {
            const size_t gi = i0 + i;
            uint32_t v;
            if (nbit == 8) {
                v = ix[gi];
            } else {
                const size_t bit = gi * nbit;
                const size_t by = bit >> 3;
                uint32_t w; memcpy(&w, ix + by, 4);
                v = (w >> (bit & 7)) & imsk;
            }
            const uint32_t *c4 = (const uint32_t *)(cb + (size_t)v * 8u);
            const uint32_t w0 = c4[0], w1 = c4[1];
            __half2 h0, h1; memcpy(&h0, &w0, 4); memcpy(&h1, &w1, 4);
            const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
            const __half2 r0 = __floats2half2_rn(f0.x * g, f0.y * g);
            const __half2 r1 = __floats2half2_rn(f1.x * g, f1.y * g);
            uint2 packed; memcpy(&packed.x, &r0, 4); memcpy(&packed.y, &r1, 4);
            *(uint2 *)(orow + (size_t)i * 4u) = packed;
        }
        return;
    }
    /* dim=8 快路(2026-09-12, V4.1 配方 vq8x4096 的 prefill: 512 token 一层 768 次本核, 每次 0.7 ms = 整趟 prefill 的大头):
     * 码本条目 16 B 按两个 8 B 读(载荷只保证 8 B 对齐), 8 个 half 结果打包成一次 16 B 存。数值与通用路逐位同(同为 rn)。 */
    if (dim == 8u && (((uintptr_t)cb) & 7u) == 0u && (((uintptr_t)orow) & 15u) == 0u) {
        for (uint32_t i = threadIdx.x; i < nidx_row; i += blockDim.x) {
            const size_t gi = i0 + i;
            uint32_t v;
            if (nbit == 8) {
                v = ix[gi];
            } else {
                const size_t bit = gi * nbit;
                const size_t by = bit >> 3;
                uint32_t w; memcpy(&w, ix + by, 4);
                v = (w >> (bit & 7)) & imsk;
            }
            const uint2 *c8 = (const uint2 *)(cb + (size_t)v * 16u);
            const uint2 w0 = c8[0], w1 = c8[1];
            __half2 h0, h1, h2, h3; memcpy(&h0, &w0.x, 4); memcpy(&h1, &w0.y, 4); memcpy(&h2, &w1.x, 4); memcpy(&h3, &w1.y, 4);
            const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
            const __half2 r0 = __floats2half2_rn(f0.x * g, f0.y * g), r1 = __floats2half2_rn(f1.x * g, f1.y * g);
            const __half2 r2 = __floats2half2_rn(f2.x * g, f2.y * g), r3 = __floats2half2_rn(f3.x * g, f3.y * g);
            uint4 packed; memcpy(&packed.x, &r0, 4); memcpy(&packed.y, &r1, 4); memcpy(&packed.z, &r2, 4); memcpy(&packed.w, &r3, 4);
            *(uint4 *)(orow + (size_t)i * 8u) = packed;
        }
        return;
    }
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


/* fused 解码核的公共常量。clamp 语义与 Metal CPU 参考路逐字一致: gate 只截上界, up 双向截
 * (prefill GEMM 路的 vqp_swiglu_kernel 同式)。 */
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

