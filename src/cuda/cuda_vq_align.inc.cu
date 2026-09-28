/* cuda_vq_align.inc.cu — 启动缓存拷 VQ 专家 blob 时, 把每个载荷挪到"位流起点 128 B 对齐"的位置(2026-09-24)。
 *
 * 【为什么】专家核(v41_vq_stream)按 384 B 整块读位流: lane j 三条 LDG.32 拿第 j / j+32 / j+64 个字, 每条 32 个 lane 正好 128 B。
 * 可 v3 载荷在盘上只保证 8 B 对齐(首载荷偏移 9232 ≡ 16 mod 32), 位流起点 = 载荷 + 32 + 行数×2 ⇒ 大部分位流错开 8/16/24 B,
 * 每条"整线读"其实跨两条线、多碰一个扇区。微基准 gguf-tools/bench/v41_vq_persist_bench.cu(真核逐式抄本, 随机数据)在 spark 上:
 *   位流错开 0 / 64 B: 124~130 µs/层(204~214 GB/s);  错开 8 / 16 / 24 / 72 B: 146~153 µs/层(175~182 GB/s)  ⇒ 对齐快 ~16%。
 * mem_ceiling ⑧ 纯读同一形态: 对齐 231 GB/s, 错开 8 B 199。
 *
 * 【怎么做】盘上文件与主机映射一个字节不动, 只改设备副本的布局: 前缀(16 B 头 + 槽表 + v3 层码本)原样, 载荷按原偏移升序依次摆,
 * 每个载荷的新起点取"≥ 上一个的末尾、且 新起点 + 32 + 行数×2 ≡ 0 (mod 128)"的最小值, 再把新偏移写回设备副本的槽表。
 * 所有设备侧消费者(v41_vq_open / 预填 vqp_copy_hdr / 塔的解码即乘)都是"读槽表 → 找载荷", 自然跟着新布局走;
 * 码本用的 cb_off 指向前缀, 前缀不动所以不用改。每层多占 ≤ 1152 × 127 B ≈ 146 KB。
 * ★逐字节同★: 载荷字节原样搬, 只换了地址 ⇒ 输出与改前逐字节相同, 门 = 温 0 输出 cmp。
 *
 * 【出错会怎样】挪完槽表却没改 = 核按旧偏移读到填充或别人的载荷, 出一整套假权重、不报错 ⇒ 下面每一条前提不满足就**不挪**(平拷), 并出声:
 *   只认 v3 blob(v2 每载荷自带码本, 版本不同不碰); 每个载荷魔数 'DQV3'; 槽表偏移互不相同、都在 blob 内;
 *   码本(cb_off 起 nc×8 B)整个落在第一个载荷之前(否则挪载荷会把码本挪走, 而 cb_off 没人改)。
 * 主机侧读 blob 的代码(ds4_gpu_v41_routed_moe_tensor 取词数、Metal 路)读的是主机映射, 布局是原样的, 两边各自自洽。 */
#include <algorithm>

/* 返回设备指针; NULL 时看 *flat: 1 = 不适用(调用方平拷), 0 = 真失败(分配/拷贝错)。 */
static const char *cuda_vq_blob_populate_aligned(const void *model_map, uint64_t offset, uint64_t bytes, const char *what, int *flat) {
    static uint32_t s_done = 0;
    *flat = 1;
    const uint8_t *hb = (const uint8_t *)model_map + offset;
    if (!ds4vq_blob_ok(hb, (size_t)bytes)) { fprintf(stderr, "ds4: [vq-align] %s 不是合法 VQ blob, 平拷\n", what); return NULL; }
    if (ds4vq_blob_ver(hb) != 3u) return NULL;   /* v2: 载荷自带码本, 这把对齐只为 v3 量过 */
    const uint32_t ns = ds4vq_blob_nexp(hb) * 3u;
    std::vector<std::pair<uint64_t, uint32_t> > pl;   /* (原偏移, 槽号) */
    for (uint32_t k = 0; k < ns; k++) {
        const uint64_t o = ds4vq_slot(hb, (int)(k / 3u), (int)(k % 3u));
        if (!o) continue;
        if (o + 32u > bytes) { fprintf(stderr, "ds4: [vq-align] %s 槽 %u 偏移越界, 平拷\n", what, k); return NULL; }
        pl.push_back(std::make_pair(o, k));
    }
    if (pl.empty()) return NULL;
    std::sort(pl.begin(), pl.end());
    const uint64_t first = pl[0].first;
    if (first < 16u + (uint64_t)ns * 8u) { fprintf(stderr, "ds4: [vq-align] %s 首载荷压着槽表, 平拷\n", what); return NULL; }
    std::vector<uint64_t> noff(pl.size());
    uint64_t cur = first;
    for (size_t i = 0; i < pl.size(); i++) {
        const uint8_t *p = hb + pl[i].first;
        uint32_t mg, rows; uint16_t nc; uint64_t cbo;
        memcpy(&mg, p, 4); memcpy(&nc, p + 6, 2); memcpy(&rows, p + 8, 4); memcpy(&cbo, p + 24, 8);
        if (mg != DS4VQ_MAT3_MAGIC || (i && pl[i].first == pl[i - 1].first) || cbo + (uint64_t)nc * 8u > first) {
            fprintf(stderr, "ds4: [vq-align] %s 槽 %u 不满足挪动前提(魔数/共用载荷/码本不在前缀), 平拷\n", what, pl[i].second);
            return NULL;
        }
        const uint64_t lead = (32u + (uint64_t)rows * 2u) & 127u;   /* 载荷头到位流的距离 mod 128 */
        cur += ((128u - lead) - cur) & 127u;                         /* 最小的 cur' ≥ cur 使 cur' + lead ≡ 0 (mod 128) */
        noff[i] = cur;
        cur += (i + 1 < pl.size() ? pl[i + 1].first : bytes) - pl[i].first;   /* 原跨度原样搬(含载荷尾垫) */
    }
    const uint64_t total = cur;
    *flat = 0;
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [vq-align] %s 分配 %.2f MiB 失败: %s\n", what, (double)total / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }
    char *d = (char *)dev;
    std::vector<uint64_t> tab((size_t)ns);
    memcpy(tab.data(), hb + 16, (size_t)ns * 8u);
    for (size_t i = 0; i < pl.size(); i++) tab[pl[i].second] = noff[i];
    err = cudaMemset(d, 0, (size_t)total);   /* 填充字节没人读, 清零只为设备副本确定 */
    if (err == cudaSuccess) err = cudaMemcpy(d, hb, (size_t)first, cudaMemcpyHostToDevice);
    for (size_t i = 0; err == cudaSuccess && i < pl.size(); i++) {
        const uint64_t n = (i + 1 < pl.size() ? pl[i + 1].first : bytes) - pl[i].first;
        err = cudaMemcpy(d + noff[i], hb + pl[i].first, (size_t)n, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) err = cudaMemcpy(d + 16, tab.data(), (size_t)ns * 8u, cudaMemcpyHostToDevice);   /* 槽表最后写: 前缀那次拷的是旧表 */
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: [vq-align] %s 拷贝失败: %s\n", what, cudaGetErrorString(err));
        (void)cudaGetLastError(); (void)cudaFree(dev);
        return NULL;
    }
    g_model_ranges.push_back({model_map, offset, bytes, d, NULL, NULL, 0, 0, 0});   /* 登记的是主机侧跨度: 查找按原 offset/bytes 命中 blob 起点 */
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += total;
    if (s_done++ == 0)
        fprintf(stderr, "ds4: [vq-align] 专家 blob 载荷挪到位流 128 B 对齐(设备副本; 首个 %s: %zu 个载荷, 多占 %.1f KB)\n",
                what, pl.size(), (double)(total - bytes) / 1024.0);
    return d;
}
