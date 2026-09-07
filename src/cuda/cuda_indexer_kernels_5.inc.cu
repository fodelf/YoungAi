/* cuda_indexer_kernels_5.inc.cu — indexer top-k 的 radix-select 版(2026-09-06, 1M 上下文战役)。
 *
 * 为什么: 老路对大 n_comp 是"每 4096 个分数一个 block 双调排序取 512 → 8 路树合并再排序": 1M 上下文一层
 * 64 块 + 两级合并, decode 每 token 7.2 ms(21 层), prefill 一块 2048 token 要 2.6 s(第三大项)。
 * 这里每 token 一个 block(1024 线程): 分数映射成可排序的 32 位键(topk_float_ordered_key), 4 轮 8 位直方图
 * (高位先, 每 warp 私有直方图免原子争用)定出"第 512 大"的精确阈值键 T 与 T 之上的个数; 然后按索引序的定序
 * 压缩(块内前缀和)取出 key > T 的全部 + key == T 的前 k_rem 个(索引小者优先 = 老核 topk_score_better 的并列
 * 规则); 最后对这 512 个 (key, idx) 做一次块内双调排序, 输出顺序与老核逐字同(分数降序, 同分索引升序)。
 * 读行 5 遍(4 轮直方图 + 1 遍压缩), 1M 时每 token 5 MB, 1024 线程条带化读; 无全局暂存。
 * 改了会怎样: 直方图不按 warp 私有 ⇒ 1024 线程砸 256 个 bin 的原子, 慢数倍; 压缩若用原子计数 ⇒ 顺序随调度漂,
 * 同分并列的取舍就不再可复现。 */
#define DS4_RTK_THREADS 1024u
#define DS4_RTK_K 512u

__global__ static void __launch_bounds__(1024) indexer_topk_radix512_kernel(
        uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens) {
    const uint32_t t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    if (t >= n_tokens) return;
    const float *row = scores + (uint64_t)t * n_comp;
    __shared__ uint32_t hist[32][256];      /* 每 warp 私有直方图 */
    __shared__ uint32_t hsum[256];
    __shared__ uint32_t scan_gt[1024], scan_eq[1024];
    __shared__ uint64_t sel[DS4_RTK_K];     /* topk_pack_key(v, idx) */
    __shared__ uint32_t s_prefix, s_krem, s_gt_total;

    /* ---- 4 轮 radix select: 定出阈值键 T ---- */
    uint32_t prefix = 0u, mask = 0u, k_rem = DS4_RTK_K;
    for (int pass = 3; pass >= 0; pass--) {
        const uint32_t shift = (uint32_t)pass * 8u;
        for (uint32_t b = lane; b < 256u; b += 32u) hist[warp][b] = 0u;
        __syncthreads();
        for (uint32_t i = tid; i < n_comp; i += DS4_RTK_THREADS) {
            const uint32_t key = topk_float_ordered_key(row[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[warp][(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        for (uint32_t b = tid; b < 256u; b += DS4_RTK_THREADS) {
            uint32_t s = 0u;
            for (uint32_t w = 0; w < 32u; w++) s += hist[w][b];
            hsum[b] = s;
        }
        __syncthreads();
        if (tid == 0) {   /* 从高 bin 往下累计, 找到第 k_rem 大所在的 bin */
            uint32_t acc = 0u, bin = 0u;
            for (int b = 255; b >= 0; b--) {
                const uint32_t c = hsum[b];
                if (acc + c >= k_rem) { bin = (uint32_t)b; break; }
                acc += c;
            }
            s_prefix = prefix | (bin << shift);
            s_krem = k_rem - acc;
        }
        __syncthreads();
        prefix = s_prefix; k_rem = s_krem;
        mask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t T = prefix;   /* 精确阈值键: key > T 全选, key == T 按索引序取前 k_rem 个 */

    /* ---- 定序压缩: 线程 tid 管索引段 [tid·seg, (tid+1)·seg) ---- */
    const uint32_t seg = (n_comp + DS4_RTK_THREADS - 1u) / DS4_RTK_THREADS;
    const uint32_t i0 = tid * seg, i1 = min(i0 + seg, n_comp);
    uint32_t c_gt = 0u, c_eq = 0u;
    for (uint32_t i = i0; i < i1; i++) {
        const uint32_t key = topk_float_ordered_key(row[i]);
        c_gt += key > T; c_eq += key == T;
    }
    scan_gt[tid] = c_gt; scan_eq[tid] = c_eq;
    __syncthreads();
    for (uint32_t off = 1u; off < DS4_RTK_THREADS; off <<= 1u) {   /* 包含式前缀和 */
        const uint32_t vg = tid >= off ? scan_gt[tid - off] : 0u;
        const uint32_t ve = tid >= off ? scan_eq[tid - off] : 0u;
        __syncthreads();
        scan_gt[tid] += vg; scan_eq[tid] += ve;
        __syncthreads();
    }
    const uint32_t gt_total = scan_gt[DS4_RTK_THREADS - 1u];
    uint32_t pos_gt = scan_gt[tid] - c_gt, pos_eq = scan_eq[tid] - c_eq;
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
    __syncthreads();
    (void)s_gt_total;

    /* ---- 512 个选中项按 (键降序, 索引升序) 排: 块内双调排序 ---- */
    for (uint32_t k2 = 2u; k2 <= DS4_RTK_K; k2 <<= 1u) {
        for (uint32_t j = k2 >> 1u; j > 0u; j >>= 1u) {
            if (tid < DS4_RTK_K) {
                const uint32_t other = tid ^ j;
                if (other > tid) {
                    const uint64_t a = sel[tid], b = sel[other];
                    const bool desc = (tid & k2) == 0u;
                    if (desc ? (b > a) : (a > b)) { sel[tid] = b; sel[other] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (uint32_t i = tid; i < DS4_RTK_K; i += DS4_RTK_THREADS)
        selected[(uint64_t)t * DS4_RTK_K + i] = 0xffffffffu - (uint32_t)(sel[i] & 0xffffffffu);
}
