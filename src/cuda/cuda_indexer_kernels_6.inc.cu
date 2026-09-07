/* cuda_indexer_kernels_6.inc.cu — indexer top-k radix-select 的多 block 版(2026-09-07, 1M 投机战役)。
 *
 * 为什么: kernels_5 的 radix512 每 token 只有一个 block(1024 线程)扫 26 万个分数 5 遍(4 轮直方图 + 1 遍压缩), 单 SM 的
 * 延迟堆: 1M 解码 6 ms/token, 投机 verify(4 token = 4 个 block)9.6 ms/轮。这里把每 token 的分数切成 ≤64 段, 每段一个
 * block: 4 轮直方图各一发(段内 warp 私有直方图 → 256 个全局原子), 阈值 T/k_rem 由每个 block 从已完成的全局直方图重算
 * (整数累计, 全 block 得同一结果); 压缩两发: 先各段数 key>T / key==T 的个数, 再按"段序 × 段内线程序 × 索引序"定序写入
 * ⇒ 选集与 kernels_5 逐字同(key>T 全选, key==T 索引小者优先取前 k_rem); 最后一发 512 元素双调排序与 kernels_5 同一段代码
 * ⇒ 输出顺序逐字同。8 发 ≈ 100 µs/层 vs 单 block 437(4 token) / 284(1 token)。
 * 暂存(每 token: 直方图 4×256 + 段计数 64×2 + 选中 512×u64 = 8704 B, 8 token 70 KB)在 token 图捕获前预建
 * (cuda_decode_scratch_prepare → rtk_scratch_prepare); 没建成或 token > 8(prefill 大批本就每 token 一 block 并行足够)
 * 走 kernels_5 原路。
 * 改了会怎样: 压缩若用原子计数 ⇒ 同分并列的取舍随调度漂, 温 0 不可复现; 直方图不 warp 私有 ⇒ 全局原子争用慢数倍。 */
#define DS4_RTKM_MAXB 64u
#define DS4_RTKM_MAXTOK 8u
#define DS4_RTKM_WORDS_PER_TOK (4u * 256u + DS4_RTKM_MAXB * 2u + DS4_RTK_K * 2u)   /* 2176 个 u32 */
static uint32_t *g_rtk_scratch = NULL;

static void rtk_scratch_prepare(void) {
    if (g_rtk_scratch) return;
    if (cudaMalloc(&g_rtk_scratch, (size_t)DS4_RTKM_MAXTOK * DS4_RTKM_WORDS_PER_TOK * sizeof(uint32_t)) != cudaSuccess) {
        g_rtk_scratch = NULL; (void)cudaGetLastError();
    }
}
__device__ __forceinline__ static uint32_t *rtk_hist_of(uint32_t *scratch, uint32_t t) { return scratch + (uint64_t)t * DS4_RTKM_WORDS_PER_TOK; }
__device__ __forceinline__ static uint32_t *rtk_cnt_of(uint32_t *scratch, uint32_t t) { return rtk_hist_of(scratch, t) + 4u * 256u; }
__device__ __forceinline__ static uint64_t *rtk_sel_of(uint32_t *scratch, uint32_t t) { return (uint64_t *)(rtk_cnt_of(scratch, t) + DS4_RTKM_MAXB * 2u); }

/* 从已完成的前 passes_done 轮直方图(槽 s = 3 − p, 高位轮在前)重算阈值前缀与剩余名额: 与 kernels_5 逐字同一累计规则 */
__device__ static void rtk_select(const uint32_t *hist, uint32_t passes_done, uint32_t *prefix_out, uint32_t *krem_out) {
    uint32_t prefix = 0u, k_rem = DS4_RTK_K;
    for (uint32_t s = 0; s < passes_done; s++) {
        const uint32_t shift = (3u - s) * 8u;
        const uint32_t *h = hist + s * 256u;
        uint32_t acc = 0u, bin = 0u;
        for (int b = 255; b >= 0; b--) {
            const uint32_t c = h[b];
            if (acc + c >= k_rem) { bin = (uint32_t)b; break; }
            acc += c;
        }
        prefix |= bin << shift;
        k_rem -= acc;
    }
    *prefix_out = prefix; *krem_out = k_rem;
}

/* 第 s 轮直方图: grid (段, token), 256 线程; 只统计高位已匹配前缀的分数 */
__global__ static void rtk_hist_kernel(const float *scores, uint32_t n_comp, uint32_t seg, uint32_t s, uint32_t *scratch) {
    const uint32_t t = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    __shared__ uint32_t hist[8][256];
    __shared__ uint32_t s_prefix, s_mask;
    uint32_t *H = rtk_hist_of(scratch, t);
    if (tid == 0) {
        uint32_t prefix = 0u, k_rem = 0u;
        rtk_select(H, s, &prefix, &k_rem);
        s_prefix = prefix;
        s_mask = s ? (0xffffffffu << ((4u - s) * 8u)) : 0u;
    }
    for (uint32_t b = lane; b < 256u; b += 32u) hist[warp][b] = 0u;
    __syncthreads();
    const uint32_t prefix = s_prefix, mask = s_mask, shift = (3u - s) * 8u;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t i0 = blk * seg, i1 = min(i0 + seg, n_comp);
    for (uint32_t i = i0 + tid; i < i1; i += 256u) {
        const uint32_t key = topk_float_ordered_key(row[i]);
        if ((key & mask) == prefix) atomicAdd(&hist[warp][(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    for (uint32_t b = tid; b < 256u; b += 256u) {
        uint32_t sum = 0u;
        for (uint32_t w = 0; w < 8u; w++) sum += hist[w][b];
        if (sum) atomicAdd(&H[s * 256u + b], sum);
    }
}

/* 各段数 key>T / key==T: cnt[blk] = {gt, eq} */
__global__ static void rtk_count_kernel(const float *scores, uint32_t n_comp, uint32_t seg, uint32_t *scratch) {
    const uint32_t t = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x;
    __shared__ uint32_t s_T, red_gt[256], red_eq[256];
    if (tid == 0) { uint32_t kr; rtk_select(rtk_hist_of(scratch, t), 4u, &s_T, &kr); }
    __syncthreads();
    const uint32_t T = s_T;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t i0 = blk * seg, i1 = min(i0 + seg, n_comp);
    uint32_t c_gt = 0u, c_eq = 0u;
    for (uint32_t i = i0 + tid; i < i1; i += 256u) {
        const uint32_t key = topk_float_ordered_key(row[i]);
        c_gt += key > T; c_eq += key == T;
    }
    red_gt[tid] = c_gt; red_eq[tid] = c_eq;
    __syncthreads();
    for (uint32_t off = 128u; off > 0u; off >>= 1u) {
        if (tid < off) { red_gt[tid] += red_gt[tid + off]; red_eq[tid] += red_eq[tid + off]; }
        __syncthreads();
    }
    if (tid == 0) { uint32_t *cnt = rtk_cnt_of(scratch, t); cnt[blk * 2u] = red_gt[0]; cnt[blk * 2u + 1u] = red_eq[0]; }
}

/* 定序压缩: 段序 × 段内线程子段序 × 索引序, 与 kernels_5 的"线程管索引段 + 块内前缀和"同一顺序语义 */
__global__ static void rtk_compact_kernel(const float *scores, uint32_t n_comp, uint32_t seg, uint32_t n_blk, uint32_t *scratch) {
    const uint32_t t = blockIdx.y, blk = blockIdx.x, tid = threadIdx.x;
    __shared__ uint32_t s_T, s_krem, s_gt_off, s_eq_off, s_gt_total, scan_gt[256], scan_eq[256];
    if (tid == 0) {
        rtk_select(rtk_hist_of(scratch, t), 4u, &s_T, &s_krem);
        const uint32_t *cnt = rtk_cnt_of(scratch, t);
        uint32_t go = 0u, eo = 0u, gtot = 0u;
        for (uint32_t b = 0; b < n_blk; b++) {
            if (b < blk) { go += cnt[b * 2u]; eo += cnt[b * 2u + 1u]; }
            gtot += cnt[b * 2u];
        }
        s_gt_off = go; s_eq_off = eo; s_gt_total = gtot;
    }
    __syncthreads();
    const uint32_t T = s_T, k_rem = s_krem, gt_total = s_gt_total;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t b0 = blk * seg, b1 = min(b0 + seg, n_comp);
    const uint32_t sub = (b1 > b0) ? (b1 - b0 + 255u) / 256u : 0u;
    const uint32_t i0 = min(b0 + tid * sub, b1), i1 = min(i0 + sub, b1);
    uint32_t c_gt = 0u, c_eq = 0u;
    for (uint32_t i = i0; i < i1; i++) {
        const uint32_t key = topk_float_ordered_key(row[i]);
        c_gt += key > T; c_eq += key == T;
    }
    scan_gt[tid] = c_gt; scan_eq[tid] = c_eq;
    __syncthreads();
    for (uint32_t off = 1u; off < 256u; off <<= 1u) {   /* 包含式前缀和 */
        const uint32_t vg = tid >= off ? scan_gt[tid - off] : 0u;
        const uint32_t ve = tid >= off ? scan_eq[tid - off] : 0u;
        __syncthreads();
        scan_gt[tid] += vg; scan_eq[tid] += ve;
        __syncthreads();
    }
    uint32_t pos_gt = s_gt_off + scan_gt[tid] - c_gt, pos_eq = s_eq_off + scan_eq[tid] - c_eq;
    uint64_t *sel = rtk_sel_of(scratch, t);
    for (uint32_t i = i0; i < i1; i++) {
        const float v = row[i];
        const uint32_t key = topk_float_ordered_key(v);
        if (key > T) {
            if (pos_gt < DS4_RTK_K) sel[pos_gt] = topk_pack_key(v, i);
            pos_gt++;
        } else if (key == T) {
            if (pos_eq < k_rem && gt_total + pos_eq < DS4_RTK_K) sel[gt_total + pos_eq] = topk_pack_key(v, i);
            pos_eq++;
        }
    }
}

/* 512 个选中项按 (键降序, 索引升序) 排: 与 kernels_5 同一段双调排序 */
__global__ static void rtk_sort_kernel(uint32_t *selected, uint32_t *scratch) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x;
    __shared__ uint64_t sel[DS4_RTK_K];
    const uint64_t *src = rtk_sel_of(scratch, t);
    sel[tid] = src[tid];
    __syncthreads();
    for (uint32_t k2 = 2u; k2 <= DS4_RTK_K; k2 <<= 1u) {
        for (uint32_t j = k2 >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid) {
                const uint64_t a = sel[tid], b = sel[other];
                const bool desc = (tid & k2) == 0u;
                if (desc ? (b > a) : (a > b)) { sel[tid] = b; sel[other] = a; }
            }
            __syncthreads();
        }
    }
    selected[(uint64_t)t * DS4_RTK_K + tid] = 0xffffffffu - (uint32_t)(sel[tid] & 0xffffffffu);
}

/* 发射: 8 发/层(直方图 4 + 计数 + 压缩 + 排序 + 一次清零)。返回 0 = 发射失败, -1 = 不适用(调用方走 kernels_5) */
static int rtk_multi_launch(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens) {
    if (n_tokens == 0u || n_tokens > DS4_RTKM_MAXTOK) return -1;
    if (!g_rtk_scratch) {   /* 非捕获态(首次 verify 批 / 单测)可现建; 捕获态里不能 cudaMalloc ⇒ 走单 block 原路 */
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(cudaStreamPerThread, &cs);
        if (cs != cudaStreamCaptureStatusNone) return -1;
        rtk_scratch_prepare();
        if (!g_rtk_scratch) return -1;
    }
    uint32_t n_blk = (n_comp + 4095u) / 4096u;
    if (n_blk > DS4_RTKM_MAXB) n_blk = DS4_RTKM_MAXB;
    const uint32_t seg = (n_comp + n_blk - 1u) / n_blk;
    /* 直方图区(每 token 前 4×256 字)清零: 各 token 的暂存连续, 一次 memset 覆盖 n_tokens 个 token 的全部字(计数/选中区随后重写) */
    if (cudaMemsetAsync(g_rtk_scratch, 0, (size_t)n_tokens * DS4_RTKM_WORDS_PER_TOK * sizeof(uint32_t), cudaStreamPerThread) != cudaSuccess)
        return cuda_ok(cudaGetLastError(), "rtk scratch memset");
    const dim3 grid(n_blk, n_tokens, 1);
    for (uint32_t s = 0; s < 4u; s++) {
        rtk_hist_kernel<<<grid, 256>>>(scores, n_comp, seg, s, g_rtk_scratch);
        if (!cuda_ok(cudaGetLastError(), "rtk hist launch")) return 0;
    }
    rtk_count_kernel<<<grid, 256>>>(scores, n_comp, seg, g_rtk_scratch);
    if (!cuda_ok(cudaGetLastError(), "rtk count launch")) return 0;
    rtk_compact_kernel<<<grid, 256>>>(scores, n_comp, seg, n_blk, g_rtk_scratch);
    if (!cuda_ok(cudaGetLastError(), "rtk compact launch")) return 0;
    rtk_sort_kernel<<<n_tokens, DS4_RTK_K>>>(selected, g_rtk_scratch);
    return cuda_ok(cudaGetLastError(), "rtk sort launch");
}
