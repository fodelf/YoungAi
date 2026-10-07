/* v41_vq_persist_ts.cuh — 常驻核的 block/warp 收尾时刻打点(2026-10-07 晚), 被 v41_vq_persist_n_bench.cu 在核定义之前包含。
 * 【为什么】引擎 nsys(10-07 n=4): gu 常驻核均时长 457 µs 里有 142 µs 与 dn 核"重叠" —— dn 的 block 只能等 gu 的 block 退出才上 SM,
 * 再在 pdl_wait 上空转到 gu 整网格结束 ⇒ 这 142 µs 就是 gu 各 block 收尾时刻的离散度(尾巴), 一轮 40 层 = 5.7 ms 白等。
 * 这里用 %globaltimer 记每个 block 的起点、每个 warp 的终点、block 的终点(末尾加一次 __syncthreads), 主机算离散度。
 * 打点只在 d_ts_on=1 的那一发生效, 计时趟不开。 */
#define TS_MAXB 64
#define TS_MAXW 32
__device__ unsigned long long d_ts_beg[TS_MAXB];
__device__ unsigned long long d_ts_end[TS_MAXB * (TS_MAXW + 1)];   /* [b*(W+1)] = block 终点, [b*(W+1)+1+w] = warp w 终点 */
__device__ int d_ts_on = 0;
__device__ __forceinline__ static unsigned long long ts_now(void) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ static void ts_begin(void) { if (d_ts_on && threadIdx.x == 0) d_ts_beg[blockIdx.x] = ts_now(); }
__device__ __forceinline__ static void ts_end(void) {   /* 全 block 都要走到(内有屏障) */
    if (!d_ts_on) return;
    if ((threadIdx.x & 31u) == 0u) d_ts_end[blockIdx.x * (TS_MAXW + 1) + 1 + (threadIdx.x >> 5)] = ts_now();
    __syncthreads();
    if (threadIdx.x == 0) d_ts_end[blockIdx.x * (TS_MAXW + 1)] = ts_now();
}
static int ts_cmp(const void *a, const void *b) { const unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b; return x < y ? -1 : x > y; }
/* 主机: 读回打点, 打一行离散度账(单位 µs): 核跨度 | block 起点差 | block 终点差 | warp 终点 p50/p90/max(相对最早的 warp) */
static void ts_report(const char *name, int nb, int nw) {
    static unsigned long long hb[TS_MAXB], he[TS_MAXB * (TS_MAXW + 1)], we[TS_MAXB * TS_MAXW];
    cudaMemcpyFromSymbol(hb, d_ts_beg, sizeof hb); cudaMemcpyFromSymbol(he, d_ts_end, sizeof he);
    unsigned long long b0 = ~0ull, b1 = 0, e0 = ~0ull, e1 = 0; int n = 0;
    for (int b = 0; b < nb; b++) {
        if (hb[b] < b0) b0 = hb[b]; if (hb[b] > b1) b1 = hb[b];
        const unsigned long long e = he[b * (TS_MAXW + 1)]; if (e < e0) e0 = e; if (e > e1) e1 = e;
        for (int w = 0; w < nw; w++) we[n++] = he[b * (TS_MAXW + 1) + 1 + w];
    }
    qsort(we, (size_t)n, sizeof we[0], ts_cmp);
    printf("   [尾巴] %-14s 核跨度 %6.1f | block 起点差 %5.1f 终点差 %6.1f | warp 终点 p50 %6.1f p90 %6.1f max %6.1f (µs, 相对最早 warp)\n",
           name, (e1 - b0) / 1e3, (b1 - b0) / 1e3, (e1 - e0) / 1e3, (we[n / 2] - we[0]) / 1e3, (we[n * 9 / 10] - we[0]) / 1e3, (we[n - 1] - we[0]) / 1e3);
}
static void ts_set(int on) { cudaMemcpyToSymbol(d_ts_on, &on, sizeof on); }
/* 按"warp 的行属于几个 token 的组(m)"分账: 复刻核里的静态划分(warp w 得行 [w·per, (w+1)·per), 组按 order 归并 ≤ M 个 token),
 * 打每档 m 的 warp 平均时长(相对本 block 起点)与个数 —— 回答"m=2 的行比 m=1 贵多少"(这是静态等行划分出尾巴的机理)。 */
static void ts_report_bym(const char *name, int nb, int nw, const int32_t *sel, const int32_t *order, uint32_t np, uint32_t rows, uint32_t M) {
    static unsigned long long hb[TS_MAXB], he[TS_MAXB * (TS_MAXW + 1)];
    cudaMemcpyFromSymbol(hb, d_ts_beg, sizeof hb); cudaMemcpyFromSymbol(he, d_ts_end, sizeof he);
    uint32_t gq[64], gm[64], ng = 0, q = 0;
    while (q < np) { const int32_t e = sel[order[q]]; uint32_t m = 1u; while (q + m < np && sel[order[q + m]] == e && m < M) m++; gq[ng] = q; gm[ng] = m; ng++; q += m; }
    const uint32_t total = ng * rows, nwarps = (uint32_t)nb * (uint32_t)nw, per = (total + nwarps - 1u) / nwarps;
    double sum[9] = { 0 }; int cnt[9] = { 0 }; double mix = 0; int nmix = 0;
    for (int b = 0; b < nb; b++) for (int w = 0; w < nw; w++) {
        const uint32_t gw = (uint32_t)b * nw + w, u0 = gw * per, u1 = u0 + per < total ? u0 + per : total;
        if (u0 >= total) continue;
        const uint32_t g0 = u0 / rows, g1 = (u1 - 1u) / rows;
        const double t = (he[b * (TS_MAXW + 1) + 1 + w] - hb[b]) / 1e3;
        if (g0 != g1 && gm[g0] != gm[g1]) { mix += t; nmix++; continue; }   /* 跨组且 m 不同的 warp 单列 */
        const uint32_t m = gm[g0]; sum[m] += t; cnt[m]++;
    }
    printf("   [按 m] %-14s", name);
    for (uint32_t m = 1; m <= M && m < 9u; m++) if (cnt[m]) printf("  m=%u: %6.1f µs ×%4d", m, sum[m] / cnt[m], cnt[m]);
    if (nmix) printf("  跨组: %6.1f µs ×%d", mix / nmix, nmix);
    printf("\n");
}
