/* cuda_v41_indexer.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 indexer 三件 —— 打分 / 候选块 / topk。
 * 2026-09-18 从 cuda_v41_2.inc.cu 拆出(那片顶到 500 行); 三个核的算法一个字没动, 只加了 graph 的设备位置口径。
 * 每个核都对着官方 Indexer.forward / select_candidate_blocks 那一段写。
 *
 * ★posd(解码整步 graph, ds4_gpu_v41.h "设备位置"口径)★: 非 NULL 时位置从 posd[0] 读, 主机给的 pos0/ng/topk
 * 只是桶上限 —— 核里自算 ng = (pos+1)/ratio(= 可见组数 = 源层已完成的组数, n_tok=1 时二者相同),
 * topk = min(topk 上限, ng)。三个核的 score/mask/idx 都按 [n_tok][*] 排, n_tok=1 时行步长无所谓,
 * 所以 ng 变了缓冲布局也不变。posd == NULL 走老口径, 逐位同。 */

/* ---- indexer 打分(官方 Indexer.forward): score[i][g] = bf16(Σ_h bf16(relu(bf16(q_h·k_g))·w[i][h])) ----
 * 一 warp 一个 g; lane 分 dk/32 维, 逐头 warp 归约。
 * ★★2026-09-16 single-1.md #6: grid 从 (n_tok) 改成 (n_tok, 组块) —— 解码时它只有 1 个 block★★
 * 病(12k 提示逐核实测, 这是本轮最大的一笔): 解码 n_tok=1 ⇒ 整个核 **1 个 block**, 48 个 SM 用 1 个,
 * 8 个 warp 轮流啃全部 ng 个组。12464 token 上下文下这一个核 **80.3 ms/步**, 占整步 200 ms 的 40%
 * —— 比骨架 GEMV(22 ms)、专家核(14.5 ms)加起来还多一倍。
 * ★为什么以前没人看见★: 所有解码尺都用 3 token 的短提示(single.md 全篇), 那时 ng 只有个位数,
 * 这个核 0.2 ms, 排在表的第 12 位。它随上下文线性涨, 而用户要的 40 t/s 是在真对话里要的。
 * 修: 组维切给 blockIdx.y, 一个 block 管 nwarp 个组 ⇒ 12k 时 390 个 block 铺满 48 个 SM。
 * ★数值逐位同★: 每个 (i,g) 的算法一个字没动(同样的逐头顺序、同样的 warp 归约、同样的 bf16 舍点),
 * 只是换了谁去算它; 组与组之间本来就互不依赖。
 * q 的重复读没变多: 原来一个 warp 顺着 g 循环, 每个 g 都要把 64 个头的 q 读一遍, 总量与现在一样。 */
__global__ static void v41_indexer_score_kernel(float *score, const float *q, const uint8_t *k, const float *w, const uint8_t *cand,
                                                uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                                const int32_t *posd) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + 1u) / ratio; }   /* graph: 真位置在设备槽, 主机的 ng 是桶上限 */
    const uint32_t vis = (pos0 + i + 1u) / ratio;   /* 可见组数 compress_lens(按绝对位置) */
    const uint32_t per = dk / 32u;                 /* dk=128 ⇒ 4 维/lane */
    /* grid.y 覆盖不完(ng 超过 grid.y×nwarp)时按 grid 步长兜底, 语义与原来的单 block 循环一样 */
    const uint32_t gstride = nwarp * gridDim.y;
    for (uint32_t g = blockIdx.y * nwarp + warp; g < ng; g += gstride) {
        float out;
        if (g >= vis || (cand && !cand[(uint64_t)i * ng + g])) out = -INFINITY;
        else {
            /* ★键先解包进寄存器再进头循环★: 索引键在缓存里是打包的 MXFP4(cuda_kv_pack), 而下面 32 个头
             * 用的是同一行 —— 解一次用 32 次。原来这里是 f32 直读, 也是重读 32 次(靠 L1 接住)。 */
            const uint8_t *kg = k + (uint64_t)g * DS4_V41_IDXK_BYTES;
            float kv[4];   /* per = dk/32 = 4(dk=128, 调用方校验过 dk%32==0) */
            for (uint32_t e = 0; e < per; e++) kv[e] = v41_idxk_get(kg, lane * per + e);
            float acc = 0.f;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qh = q + ((uint64_t)i * n_head + h) * dk;
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qh[lane * per + e] * kv[e];
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                d = v41_bf16r(d);                    /* einsum 出 bf16 */
                d = fmaxf(d, 0.0f);                  /* relu_ */
                acc += v41_bf16r(d * w[(uint64_t)i * n_head + h]);   /* × weights(bf16) 再求和 */
            }
            out = v41_bf16r(acc);
        }
        if (lane == 0) score[(uint64_t)i * ng + g] = out;
    }
}
int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                                     const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_mask,
                                     uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                     const ds4_gpu_tensor *posd) {
    if (!score || !q || !k || !weights || (dk % 32u) || ratio == 0) return 0;
    if (dk / 32u > 4u) { fprintf(stderr, "ds4: [v41] indexer 打分核只实现 dk ≤ 128(每 lane ≤ 4 维)\n"); return 0; }
    if (posd && n_tok != 1u) return 0;
    if (ng == 0) return 1;
    /* 一 block 8 warp = 8 个组; 组多就多开 block(上限 65535 是 CUDA 的 grid.y 硬顶, 超了核里按 grid 步长绕) */
    uint32_t gblocks = (ng + 7u) / 8u;
    if (gblocks > 65535u) gblocks = 65535u;
    v41_indexer_score_kernel<<<dim3(n_tok, gblocks), 256, 0, g_cur_stream>>>((float *)score->ptr, (const float *)q->ptr, (const uint8_t *)k->ptr,
        (const float *)weights->ptr, cand_mask ? (const uint8_t *)cand_mask->ptr : NULL, pos0, ng, n_head, dk, ratio,
        posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 indexer score");
}

/* ---- 候选块(官方 select_candidate_blocks) ----
 * ★★2026-09-16 single-1.md: 选块从"线程 0 跑 O(kk×nb)"改成 radix select★★
 * 病(12k 提示逐核实测): 原版注释写"块数 ≤ ng/bs 小"—— 那是按预填想的。解码时候选源层的 ng 就是
 * 整段上下文(12464 ⇒ nb = 1558 块, 块大小 8), 而 topk_blocks = 2048 > nb ⇒ kk = nb,
 * **一个线程要跑 1558 × 1558 ≈ 240 万轮**, 实测 **46.4 ms/步**, 一个核占整步的 23%。
 * 32k 上下文时 nb = 4096、kk = 2048 ⇒ 840 万轮, 还要再翻三倍。
 * 这与 09-14 给 v41_topk_kernel 动的手术是同一个病(那次也是"线程 0 串行 O(topk·ng)"), 只是这个核漏了。
 * 修(与 topk 核同一套路, 语义逐项复刻):
 *   ①kk ≥ nb(现役配置在 32k 以内一直是这样): 原版等价于"凡是分数 > -inf 的块全选" —— 直接并行写,
 *     不用选。(全 -inf 的块 = 整块不可见, 下游打分核那句 `g >= vis` 本来就把它挡掉, 选不选都一样。)
 *   ②kk < nb: 并行 radix select 求第 kk 大的键, 键 > 阈值的全选; 等于阈值的按**块号升序**补到 kk 个
 *     —— 原版每轮用 `bsc[b] > bv` 扫描(严格大于), 并列时先出现的小块号胜出, 这里逐字复刻。
 * ★数值逐位同★: 选出来的块集合与原版完全一样, 只是换了怎么选。 */
/* float → 可按无符号比较的 32 位键(候选块核与 topk 核共用)。
 * −inf 与 NaN 归 0: 两个核的原版都用 `x > bv`(bv 初值 −inf)扫描, 这两类值永远选不中, 键 0 复刻它。
 * 正数: 置最高位保持序; 负数: 按位取反 —— 于是 IEEE 浮点的大小序 = 键的无符号大小序。 */
__device__ __forceinline__ static uint32_t v41_topk_key(float f) {
    if (!(f > -INFINITY)) return 0u;
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ static void v41_candidate_kernel(uint8_t *mask, const float *score, uint32_t pos0, uint32_t ng, uint32_t ratio,
                                            uint32_t topk_blocks, uint32_t bs, const int32_t *posd) {
    const uint32_t i = blockIdx.x;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + 1u) / ratio; }   /* graph: 见文件头; shared 按桶上限开, ng ≤ 上限 */
    const uint32_t nb = (ng + bs - 1u) / bs;
    extern __shared__ float bsc[];               /* [nb] 块分 + [nb] 选中标记 */
    uint8_t *sel = (uint8_t *)(bsc + nb);
    __shared__ uint32_t hist[256], sh_bucket, sh_k;
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -INFINITY;
        for (uint32_t j = b * bs; j < (b + 1u) * bs && j < ng; j++) m = fmaxf(m, score[(uint64_t)i * ng + j]);
        if (vis > 0u && b == (vis - 1u) / bs) m = INFINITY;   /* 含最新位置的块钉住 */
        bsc[b] = m; sel[b] = 0;
    }
    __syncthreads();
    const uint32_t kk = topk_blocks < nb ? topk_blocks : nb;
    if (kk >= nb) {                               /* ① 全选(除了整块不可见的) */
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) sel[b] = v41_topk_key(bsc[b]) != 0u;
    } else {                                      /* ② 并行 radix select 定阈值 */
        uint32_t prefix = 0, want = kk;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (uint32_t t = threadIdx.x; t < 256u; t += blockDim.x) hist[t] = 0;
            __syncthreads();
            for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
                const uint32_t key = v41_topk_key(bsc[b]);
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {               /* 从高桶往低累加, 找住第 want 大的那个桶 */
                uint32_t acc = 0; int t = 255;
                for (; t > 0; t--) { if (acc + hist[t] >= want) break; acc += hist[t]; }
                sh_bucket = (uint32_t)t; sh_k = want - acc;
            }
            __syncthreads();
            prefix |= sh_bucket << shift;
            want = sh_k;
            __syncthreads();
        }
        for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) sel[b] = v41_topk_key(bsc[b]) > prefix;
        __syncthreads();
        if (threadIdx.x == 0) {                   /* 与阈值相等的按块号升序补齐(并列取小块号, 同原版) */
            uint32_t eq = want;
            for (uint32_t b = 0; b < nb && eq; b++) if (v41_topk_key(bsc[b]) == prefix) { sel[b] = 1; eq--; }
        }
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < ng; j += blockDim.x) mask[(uint64_t)i * ng + j] = sel[j / bs];
}
int ds4_gpu_v41_candidate_blocks_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t block_size,
                                        const ds4_gpu_tensor *posd) {
    if (!mask || !score || block_size == 0 || ratio == 0) return 0;
    if (posd && n_tok != 1u) return 0;
    if (ng == 0) return 1;
    const uint32_t nb = (ng + block_size - 1u) / block_size;
    const size_t shm = (size_t)nb * 4u + nb;
    if (shm > 48u * 1024u) { fprintf(stderr, "ds4: [v41] candidate blocks %u 超 shared\n", nb); return 0; }
    v41_candidate_kernel<<<n_tok, 256, shm, g_cur_stream>>>((uint8_t *)mask->ptr, (const float *)score->ptr, pos0, ng, ratio,
                                                            topk_blocks, block_size, posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 candidate blocks");
}

/* ---- topk(官方: topk 后按位置升序; 不可达 → -1) ----
 * ★2026-09-14 重写: 原版是"线程 0 串行 O(topk·ng)"(当时留言"速度 P4 再说")。
 * 解码时 ng = 已有上下文长度、topk = 2048 ⇒ 3300 上下文就是 676 万次串行比较, 实测 **44 ms 一次**;
 * nsys 长上下文解码段: 这一个核吃掉 62% 的 GPU 时间, 是"上下文一长解码就塌"(5 token 上下文 8.96 t/s,
 * 3308 token 只剩 1.63 t/s)的主因。
 * 新版分两步, 语义与原版逐位等价:
 *   ①并行 radix select 求第 k 大的阈值 —— 把 float 单调映射成可按无符号比较的 32 位键, 高位到低位
 *     每轮 8 位、256 线程协作数一次直方图, 4 轮定出阈值。每线程只扫 ng/256 个元素。
 *   ②线程 0 升序扫一遍 ng 写出(O(ng), 不是 O(topk·ng))。写出必须串行才能保证"并列取小下标"与官方一致。
 * 等价性要点: 原版用 `s[g] > bv`(严格大于, bv 初值 −inf), 所以 ①并列时先出现的小下标胜出
 * ②−inf 与 NaN 永远选不中 —— 两条都在下面按原样复刻(键 0 = 不可达, 写出时跳过)。 */
__global__ static void v41_topk_kernel(int32_t *idx, const float *score, uint32_t ng, uint32_t topk, uint32_t ratio,
                                       const int32_t *posd) {
    const uint32_t i = blockIdx.x;
    if (posd) {   /* graph: ng 与 topk 都按设备位置自算(与主机直发路 min(index_topk, ng) 同式) */
        ng = ((uint32_t)posd[0] + 1u) / ratio;
        if (ng < topk) topk = ng;
    }
    const float *s = score + (uint64_t)i * ng;
    /* ★2026-09-16 判负存档: "直方图按 warp 各记各的"(hist[8][256], 消 shared 原子争用)★
     * 猜想: 第 2~4 轮活下来的元素几乎全落进同一个桶, 256 个线程的 atomicAdd 在那一个地址上排队。
     * 实测 **5.91 → 7.15 ms(慢 21%)**, 回退 —— 争用不是这个核的瓶颈, 多出来的 8 份清零(每轮 2048 个)
     * 与扫桶时每桶 8 次加(255×8×4 轮)反而更贵。真正的大头见下面写出段的注释。 */
    __shared__ uint32_t hist[256], sh_bucket, sh_k, sh_nvalid;
    if (threadIdx.x == 0) sh_nvalid = 0;
    __syncthreads();
    uint32_t cnt = 0;
    for (uint32_t g = threadIdx.x; g < ng; g += blockDim.x) cnt += (v41_topk_key(s[g]) != 0u);
    atomicAdd(&sh_nvalid, cnt);
    __syncthreads();
    const uint32_t nval = sh_nvalid;
    uint32_t prefix = 0, kk = topk < nval ? topk : nval;
    if (kk > 0) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (uint32_t b = threadIdx.x; b < 256u; b += blockDim.x) hist[b] = 0;
            __syncthreads();
            for (uint32_t g = threadIdx.x; g < ng; g += blockDim.x) {
                const uint32_t key = v41_topk_key(s[g]);
                /* shift=24 那轮没有"更高位"可比(右移 32 是未定义行为), 全部计入 */
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {               /* 从高桶往低累加, 找住第 kk 大的那个桶 */
                uint32_t acc = 0; int b = 255;
                for (; b > 0; b--) { if (acc + hist[b] >= kk) break; acc += hist[b]; }
                sh_bucket = (uint32_t)b; sh_k = kk - acc;
            }
            __syncthreads();
            prefix |= sh_bucket << shift;
            kk = sh_k;                            /* 落到这一桶内还要取几个 */
            __syncthreads();
        }
    }
    /* ★★2026-09-16 single-1.md: 写出段从"线程 0 串行扫 ng"改成 256 个线程各扫一段★★
     * 病(12k 提示实测 5.89 ms/步, 8 发 ⇒ 0.74 ms/发): radix 定阈值那几轮是并行的, 但**写出**这一段
     * 一直是线程 0 一个人从头扫到尾 —— 12480 个组、每轮一次全局读 + 分支, 一个线程 4 万多个周期。
     * 09-14 重写这个核时把 O(topk·ng) 降成 O(ng) 就收工了, 没注意 O(ng) 仍然是**单线程**的。
     * 这与本轮修掉的另外两个核(indexer 打分 grid 只有 1 个 block、候选块线程 0 串行选块)是同一个病,
     * 那两刀分别是 80.3 → 0.90 和 46.4 → 掉出表, 所以这里也照着办。
     * ★先排除过一个猜想★: "shared 原子争用"——按 warp 各记一份直方图实测 5.91 → 7.15(更慢), 已回退。
     *
     * 怎么做到**逐位同**: 原版的输出 = 按 g 升序排列的选中项, 选中 = (键 > 阈值) 或
     * (键 == 阈值 且它在等值里的升序序号 < eq)。所以第 g 项的**位置**就是"g 之前有多少个被选中的":
     *   位置 = (g 之前的 > 阈值个数) + min(g 之前的 == 阈值个数, eq)
     * 于是把 ng 切成 256 段, 每段先数出 (gt, eq) 两个计数, 块内做两次前缀和拿到每段的起点,
     * 再各段独立地按段内升序写 —— 集合、顺序、并列规则与原版逐项相同。 */
    __shared__ uint32_t cgt[256], ceq[256], sh_tgt, sh_teq;   /* ★cgt/ceq 按 block = 256 线程写死★ */
    if (threadIdx.x == 0) { sh_tgt = 0; sh_teq = 0; }
    __syncthreads();
    const uint32_t chunk = (ng + blockDim.x - 1u) / blockDim.x;
    const uint32_t g0 = threadIdx.x * chunk, g1 = (g0 + chunk) < ng ? (g0 + chunk) : ng;
    {   /* ① 各段自己数: 本段里 > 阈值 / == 阈值 各几个 */
        uint32_t ngt = 0, neq = 0;
        for (uint32_t g = g0; g < g1; g++) {
            const uint32_t key = v41_topk_key(s[g]);
            if (key == 0u) continue;
            if (key > prefix) ngt++;
            else if (key == prefix) neq++;
        }
        cgt[threadIdx.x] = ngt; ceq[threadIdx.x] = neq;
        atomicAdd(&sh_tgt, ngt); atomicAdd(&sh_teq, neq);   /* 合计: ④ 填尾巴要用 */
    }
    __syncthreads();
    /* ② 两条前缀和(256 项, Hillis-Steele 就地做成**排他**前缀: 先右移一格再逐级相加)。
     *    ★不能用 warp shuffle 版★: 要跨 8 个 warp, 而且每级都要 __syncthreads(都在分支外)。 */
    {
        uint32_t vg = threadIdx.x ? cgt[threadIdx.x - 1u] : 0u;
        uint32_t ve = threadIdx.x ? ceq[threadIdx.x - 1u] : 0u;
        __syncthreads();
        cgt[threadIdx.x] = vg; ceq[threadIdx.x] = ve;
        __syncthreads();
        for (uint32_t off = 1u; off < blockDim.x; off <<= 1) {
            const uint32_t ag = threadIdx.x >= off ? cgt[threadIdx.x - off] : 0u;
            const uint32_t ae = threadIdx.x >= off ? ceq[threadIdx.x - off] : 0u;
            __syncthreads();
            cgt[threadIdx.x] += ag; ceq[threadIdx.x] += ae;
            __syncthreads();
        }
    }
    {   /* ③ 各段按段内升序写出。eq = radix 收尾时"与阈值相等的还要取几个" */
        const uint32_t eq = kk;
        uint32_t wgt = cgt[threadIdx.x], weq = ceq[threadIdx.x];
        for (uint32_t g = g0; g < g1; g++) {
            const uint32_t key = v41_topk_key(s[g]);
            if (key == 0u) continue;
            if (key > prefix) {
                const uint32_t pos = wgt + (weq < eq ? weq : eq);
                if (pos < topk) idx[(uint64_t)i * topk + pos] = (int32_t)g;
                wgt++;
            } else if (key == prefix) {
                if (weq < eq) {
                    const uint32_t pos = wgt + weq;
                    if (pos < topk) idx[(uint64_t)i * topk + pos] = (int32_t)g;
                }
                weq++;
            }
        }
    }
    __syncthreads();
    {   /* ④ 尾巴填 -1(不可达)。总数 = 全部 > 阈值的 + min(全部 == 阈值的, eq), 与原版 w 的收尾值相同 */
        uint32_t total = sh_tgt + (sh_teq < kk ? sh_teq : kk);
        if (total > topk) total = topk;
        for (uint32_t w = total + threadIdx.x; w < topk; w += blockDim.x) idx[(uint64_t)i * topk + w] = -1;
    }
}
int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng,
                                    uint32_t topk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    if (!idx || !score || topk == 0) return 0;
    if (posd && (n_tok != 1u || ratio == 0u)) return 0;
    if (ng == 0) return 1;
    /* shared 只剩固定的 256 个桶(1 KB), 不再随 ng 走 ⇒ 原来那条 "ng > 48K 就拒" 的闸跟着作废 */
    v41_topk_kernel<<<n_tok, 256, 0, g_cur_stream>>>((int32_t *)idx->ptr, (const float *)score->ptr, ng, topk, ratio,
                                                     posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 topk");
}
