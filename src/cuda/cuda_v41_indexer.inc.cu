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
#define V41_IDX_G 4u   /* 打分核一个 warp 同时推进几个组(见核里) */
__global__ static void v41_indexer_score_kernel(float *score, const float *q, const uint8_t *k, const float *w, const uint8_t *cand,
                                                uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                                const int32_t *posd) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    /* graph: 真位置在设备槽, 主机的 ng 是桶上限。★n 行(投机验证批进图, 2026-09-22)★: 源层在本批之后的组数 = (pos0 + n)/ratio
     * (直发路 g0 + ng_new 的闭式, n = gridDim.x), 各行的可见组数仍按各自位置算(vis) —— n=1 时与原式 (pos0+1)/ratio 相同。 */
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }
    const uint32_t vis = (pos0 + i + 1u) / ratio;   /* 可见组数 compress_lens(按绝对位置) */
    const uint32_t per = dk / 32u;                 /* dk=128 ⇒ 4 维/lane */
    /* ★一个 warp 同时推进 V41_IDX_G 个组(2026-09-23, 长上下文)★: 原来一个 warp 一个组, 64 个头逐个"4 次乘加 → 5 级 shfl 规约
     * → 舍入"是一条串行依赖链, 发射槽大半空着(51k 上下文每发 ~200 µs)。现在 G 条链交错, 每个头的 q 也只读一次给 G 组共用。
     * ★逐位同★: 每个 (组, 头) 的乘加次序(e 升序)、xor 规约树、bf16 舍点、头的累加次序一个没动; 不可见的组照旧直接给 −inf。
     * grid.y 覆盖不完(ng 超过 grid.y×nwarp×G)时按 grid 步长兜底, 语义与原来的单 block 循环一样。 */
    const uint32_t gstride = nwarp * gridDim.y * V41_IDX_G;
    for (uint32_t gb = (blockIdx.y * nwarp + warp) * V41_IDX_G; gb < ng; gb += gstride) {
        float kv[V41_IDX_G][4], acc[V41_IDX_G];
        bool live[V41_IDX_G];
        #pragma unroll
        for (uint32_t j = 0; j < V41_IDX_G; j++) {
            const uint32_t g = gb + j;
            live[j] = g < ng && !(g >= vis || (cand && !cand[(uint64_t)i * ng + g]));
            acc[j] = 0.f;
            /* ★键先解包进寄存器再进头循环★: 索引键在缓存里是打包的 MXFP4(cuda_kv_pack), 解一次用 64 次 */
            const uint8_t *kg = k + (uint64_t)(live[j] ? g : 0u) * DS4_V41_IDXK_BYTES;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; e++) kv[j][e] = (live[j] && e < per) ? v41_idxk_get(kg, lane * per + e) : 0.f;
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)i * n_head + h) * dk;
            float qv[4];
            #pragma unroll
            for (uint32_t e = 0; e < 4u; e++) qv[e] = e < per ? qh[lane * per + e] : 0.f;
            const float wh = w[(uint64_t)i * n_head + h];
            float d[V41_IDX_G];
            #pragma unroll
            for (uint32_t j = 0; j < V41_IDX_G; j++) {
                d[j] = 0.f;
                #pragma unroll
                for (uint32_t e = 0; e < 4u; e++) if (e < per) d[j] += qv[e] * kv[j][e];
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                #pragma unroll
                for (uint32_t j = 0; j < V41_IDX_G; j++) d[j] += __shfl_xor_sync(0xffffffffu, d[j], o);
            }
            #pragma unroll
            for (uint32_t j = 0; j < V41_IDX_G; j++) {
                float dd = v41_bf16r(d[j]);          /* einsum 出 bf16 */
                dd = fmaxf(dd, 0.0f);                /* relu_ */
                acc[j] += v41_bf16r(dd * wh);        /* × weights(bf16) 再求和 */
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (uint32_t j = 0; j < V41_IDX_G; j++) if (gb + j < ng) score[(uint64_t)i * ng + gb + j] = live[j] ? v41_bf16r(acc[j]) : -INFINITY;
        }
    }
}
int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                                     const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_mask,
                                     uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                     const ds4_gpu_tensor *posd) {
    if (!score || !q || !k || !weights || (dk % 32u) || ratio == 0) return 0;
    if (dk / 32u > 4u) { fprintf(stderr, "ds4: [v41] indexer 打分核只实现 dk ≤ 128(每 lane ≤ 4 维)\n"); return 0; }
    if (posd && n_tok > 8u) return 0;   /* graph 路: 纯解码 1 行或投机验证批 ≤ 8 行(核里 ng 按 pos0 + n 算) */
    if (ng == 0) return 1;
    /* 一 block 8 warp × V41_IDX_G 组; 组多就多开 block(上限 65535 是 CUDA 的 grid.y 硬顶, 超了核里按 grid 步长绕) */
    uint32_t gblocks = (ng + 8u * V41_IDX_G - 1u) / (8u * V41_IDX_G);
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
/* 候选块的 [nb] 块分 + [nb] 选中标记: 住全局暂存槽(不是 shared, 原因见核里的注释)。
 * graph 捕获期不许分配 ⇒ 开捕获前由 ds4_gpu_v41_candidate_scratch_prepare 按桶上限先长够;
 * 真长了会 +1 暂存代号(v41_grow), core_decode_graph.c 发图前对代号, 变了就重捕获。 */
static v41_scratch g_v41_cand_blk;
int ds4_gpu_v41_candidate_scratch_prepare(uint32_t n_tok, uint32_t nb) {
    if (n_tok == 0u || nb == 0u) return 1;
    return v41_grow(&g_v41_cand_blk, (uint64_t)n_tok * nb * 5u, "v41 候选块") ? 1 : 0;
}
__global__ static void v41_candidate_kernel(uint8_t *mask, const float *score, uint32_t pos0, uint32_t ng, uint32_t ratio,
                                            uint32_t topk_blocks, uint32_t bs, const int32_t *posd,
                                            float *blk, uint32_t nb_cap) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t i = blockIdx.x;
    if (posd) { pos0 = (uint32_t)posd[0]; ng = (pos0 + gridDim.x) / ratio; }   /* graph: 见打分核; 暂存按桶上限开, ng ≤ 上限 */
    const uint32_t nb = (ng + bs - 1u) / bs;
    /* ★[nb] 两个数组从 shared 挪到全局暂存★(2026-09-21, 为了 500k 上下文): 静态 shared 48 KB 只装得下
     * nb ≤ 9830 块 = ratio-1 层 78k 个位置 —— 这就是 ctx 卡死在 32768 的那道墙(GB10 动态 shared 上限 99 KB 也不够,
     * 500k 要 312 KB)。行距用主机传的桶上限 nb_cap: 核里按真位置算出来的 nb 可能更小, 拿它当行距两行会重叠。
     * 块内 __syncthreads() 对全局写同样是可见性屏障, 所以下面每一步的语义与 shared 版逐字相同; 代价只是
     * radix 的 4 遍扫描从 shared 走 L2(nb 上千时几微秒, 与打分核的 O(ng) 比可忽略)。 */
    float *bsc = blk + (uint64_t)i * nb_cap;
    uint8_t *sel = (uint8_t *)(blk + (uint64_t)gridDim.x * nb_cap) + (uint64_t)i * nb_cap;
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
    if (posd && n_tok > 8u) return 0;
    if (ng == 0) return 1;
    const uint32_t nb = (ng + block_size - 1u) / block_size;
    /* 直发路(预填 / 暖身)在这里现长; graph 路进来时 prepare 已按桶上限长够, 这一发是 no-op */
    float *blk = (float *)v41_grow(&g_v41_cand_blk, (uint64_t)n_tok * nb * 5u, "v41 候选块");
    if (!blk) return 0;
    v41_candidate_kernel<<<n_tok, 256, 0, g_cur_stream>>>((uint8_t *)mask->ptr, (const float *)score->ptr, pos0, ng, ratio,
                                                            topk_blocks, block_size, posd ? (const int32_t *)posd->ptr : NULL,
                                                            blk, nb);
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
/* ★2026-09-23 长上下文重写(逐字节同)★ 51k 上下文实测每发 ~190 µs(每步 8 发 1.5 ms, 15 万上下文外推 > 4 ms, 产品请求全在 7~15 万):
 * 解码时这个核只有 1 个 block, 每遍每线程 `for (g = tid; g < ng; g += 256)` 读一次等一次 L2, 一遍 ng/256 次串行往返, 共 7 遍;
 * radix 每轮还由线程 0 串行扫 256 个桶。四处改动, 选出的集合、顺序、并列规则都不变:
 *   ①256 → 1024 线程(V41_TOPK_THREADS): 在飞的读多 4 倍; 写出段按线程分段, 位置由全局前缀计数定, 与分几段无关;
 *   ②每线程的读一批 V41_TOPK_B 个先发再用(计数/原子加的次序对整数结果无影响);
 *   ③"数有效键"那一遍并进 radix 第一轮: 第一轮对全部有效键计数, 各桶之和就是 nvalid;
 *   ④找桶: 原版 `for (b = 255; b > 0; b--) { if (acc + hist[b] >= kk) break; acc += hist[b]; }` 等价于
 *     b* = max{b ≥ 1 : suf[b] ≥ kk}(没有就 0), 落桶余量 = kk − suf[b*+1], suf 是后缀和 —— 后缀和并行求, b* 用 atomicMax 求。 */
#define V41_TOPK_THREADS 1024u
#define V41_TOPK_B 8u
__global__ static void __launch_bounds__(V41_TOPK_THREADS) v41_topk_kernel(int32_t *idx, const float *score, uint32_t ng, uint32_t topk,
                                                                          uint32_t ratio, const int32_t *posd) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    if (posd) {   /* graph: ng 与 topk 都按设备位置自算(与主机直发路 min(index_topk, ng) 同式; n 行时 ng = (pos0 + n)/ratio) */
        ng = ((uint32_t)posd[0] + gridDim.x) / ratio;
        if (ng < topk) topk = ng;
    }
    const float *s = score + (uint64_t)i * ng;
    /* ★2026-09-16 判负存档: "直方图按 warp 各记各的"(hist[8][256], 消 shared 原子争用)★ 实测 5.91 → 7.15 ms(慢 21%), 回退 ——
     * 争用不是这个核的瓶颈, 多出来的 8 份清零与扫桶时每桶 8 次加反而更贵。 */
    __shared__ uint32_t hist[256], suf[257], sh_bucket;
    uint32_t prefix = 0, kk = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = tid; b < 256u; b += nt) hist[b] = 0;
        __syncthreads();
        for (uint32_t gb = tid; gb < ng; gb += V41_TOPK_B * nt) {
            float v[V41_TOPK_B];
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) { const uint32_t g = gb + j * nt; v[j] = g < ng ? s[g] : -INFINITY; }
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) {
                const uint32_t key = v41_topk_key(v[j]);   /* 越界补的 −inf ⇒ 键 0, 与"不计入"同义 */
                /* shift=24 那轮没有"更高位"可比(右移 32 是未定义行为), 全部计入 */
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (tid < 32u) {   /* 后缀和 suf[b] = Σ_{b' ≥ b} hist[b']: lane l 管桶 [8l, 8l+8), 跨 lane 用 shfl 求"比我高的 lane 之和" */
            uint32_t loc = 0;
            for (uint32_t b = 0; b < 8u; b++) loc += hist[tid * 8u + b];
            uint32_t inc = loc;
            for (int o = 1; o < 32; o <<= 1) { const uint32_t t = __shfl_down_sync(0xffffffffu, inc, o); if (tid + (uint32_t)o < 32u) inc += t; }
            uint32_t run = inc - loc;   /* 比本 lane 高的桶之和 */
            for (int b = 7; b >= 0; b--) { run += hist[tid * 8u + (uint32_t)b]; suf[tid * 8u + (uint32_t)b] = run; }
            if (tid == 0u) { suf[256] = 0u; sh_bucket = 0u; }
        }
        __syncthreads();
        if (shift == 24) {   /* ③ 第一轮的直方图覆盖全部有效键 ⇒ suf[0] = nvalid */
            const uint32_t nval = suf[0];
            kk = topk < nval ? topk : nval;
            if (kk == 0u) break;   /* 原版: kk == 0 时不进 radix, prefix 保持 0(全块一致的分支) */
        }
        for (uint32_t b = 1u + tid; b < 256u; b += nt) if (suf[b] >= kk) atomicMax(&sh_bucket, b);
        __syncthreads();
        const uint32_t bs = sh_bucket;
        prefix |= bs << shift;
        kk -= suf[bs + 1u];   /* 落到这一桶内还要取几个 */
        __syncthreads();      /* 下一轮清 hist / 写 suf 之前, 大家都读完了 */
    }
    /* ★写出段(2026-09-16 起并行): 位置 = (g 之前的 > 阈值个数) + min(g 之前的 == 阈值个数, eq)★ ——
     * 把 ng 切成 nt 段, 每段先数 (gt, eq), 两次前缀和拿各段起点, 各段按段内升序写。集合、顺序、并列规则与原版逐项相同。 */
    __shared__ uint32_t cgt[V41_TOPK_THREADS], ceq[V41_TOPK_THREADS], sh_tgt, sh_teq;
    if (tid == 0) { sh_tgt = 0; sh_teq = 0; }
    __syncthreads();
    const uint32_t chunk = (ng + nt - 1u) / nt;
    const uint32_t g0 = tid * chunk, g1 = (g0 + chunk) < ng ? (g0 + chunk) : ng;
    {   /* ① 各段自己数: 本段里 > 阈值 / == 阈值 各几个(读一批先发) */
        uint32_t ngt = 0, neq = 0;
        for (uint32_t gb = g0; gb < g1; gb += V41_TOPK_B) {
            float v[V41_TOPK_B];
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) v[j] = gb + j < g1 ? s[gb + j] : -INFINITY;
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) {
                const uint32_t key = v41_topk_key(v[j]);
                if (key == 0u) continue;
                if (key > prefix) ngt++;
                else if (key == prefix) neq++;
            }
        }
        cgt[tid] = ngt; ceq[tid] = neq;
        atomicAdd(&sh_tgt, ngt); atomicAdd(&sh_teq, neq);   /* 合计: ④ 填尾巴要用 */
    }
    __syncthreads();
    /* ② 两条排他前缀和(nt 项, Hillis-Steele 就地: 先右移一格再逐级相加; 每级都要 __syncthreads, 都在分支外) */
    {
        uint32_t vg = tid ? cgt[tid - 1u] : 0u;
        uint32_t ve = tid ? ceq[tid - 1u] : 0u;
        __syncthreads();
        cgt[tid] = vg; ceq[tid] = ve;
        __syncthreads();
        for (uint32_t off = 1u; off < nt; off <<= 1) {
            const uint32_t ag = tid >= off ? cgt[tid - off] : 0u;
            const uint32_t ae = tid >= off ? ceq[tid - off] : 0u;
            __syncthreads();
            cgt[tid] += ag; ceq[tid] += ae;
            __syncthreads();
        }
    }
    {   /* ③ 各段按段内升序写出。eq = radix 收尾时"与阈值相等的还要取几个" */
        const uint32_t eq = kk;
        uint32_t wgt = cgt[tid], weq = ceq[tid];
        for (uint32_t gb = g0; gb < g1; gb += V41_TOPK_B) {
            float v[V41_TOPK_B];
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) v[j] = gb + j < g1 ? s[gb + j] : -INFINITY;
            #pragma unroll
            for (uint32_t j = 0; j < V41_TOPK_B; j++) {
                const uint32_t g = gb + j;
                const uint32_t key = v41_topk_key(v[j]);
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
    }
    __syncthreads();
    {   /* ④ 尾巴填 -1(不可达)。总数 = 全部 > 阈值的 + min(全部 == 阈值的, eq), 与原版 w 的收尾值相同 */
        uint32_t total = sh_tgt + (sh_teq < kk ? sh_teq : kk);
        if (total > topk) total = topk;
        for (uint32_t w = total + tid; w < topk; w += nt) idx[(uint64_t)i * topk + w] = -1;
    }
}
int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng,
                                    uint32_t topk, uint32_t ratio, const ds4_gpu_tensor *posd) {
    if (!idx || !score || topk == 0) return 0;
    if (posd && (n_tok > 8u || ratio == 0u)) return 0;
    if (ng == 0) return 1;
    /* shared 只剩固定的 256 个桶(1 KB), 不再随 ng 走 ⇒ 原来那条 "ng > 48K 就拒" 的闸跟着作废 */
    v41_topk_kernel<<<n_tok, V41_TOPK_THREADS, 0, g_cur_stream>>>((int32_t *)idx->ptr, (const float *)score->ptr, ng, topk, ratio,
                                                     posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 topk");
}
