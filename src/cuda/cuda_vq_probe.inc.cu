/* cuda_vq_probe.inc.cu — VQ 专家解码核的两个看门狗(2026-09-18 从 cuda_vq_decode.inc.cu 拆出, 那片顶到了 500 行)。
 * 只在 --v41-prof 下跑, 不改任何数值: ①激活是否真在 bf16 格点上(解码核把激活按 bf16 存的前提) ②激活/中间量有没有
 * 超出 f16 的指数范围(专家核若上 f16 张量核, 定走 f16 还是 TF32)。
 * ★必须排在 cuda_v41_4.inc.cu 之后(用 v41_grow / v41_scratch / g_cur_stream)、cuda_vq_decode.inc.cu 之前(它调用这里的探针)。 */
/* ★"激活已在 bf16 格点"这条不变量的看门狗(只在 --v41-prof 下跑)★
 * 上面那一刀的逐位同**完全建立在这条不变量上**: 值的低 16 位全是零, 打包/还原才是恒等变换。
 * 与其去重建一个改前的二进制来比输出哈希, 不如直接数一遍"低 16 位非零的元素有几个" ——
 * 是 0 就说明这个变换在数学上是恒等的, 比对哈希更硬(哈希相同只证明这一条提示没露馅)。
 * 哪天上游改了 rms_norm 的出口舍入, 这个数会立刻从 0 变成几千, 而速度和输出都"看着正常"。 */
__global__ static void v41_vq_offgrid_kernel(uint32_t *cnt, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (__float_as_uint(src[i]) & 0xffffu) != 0u) atomicAdd(cnt, 1u);
}
/* ★f16 范围看门狗(mtp-2.md §5.3, 2026-09-17)★
 *
 * 为什么要它: 专家核上张量核(cuda_vq_tc_moe.inc.cu)要把码本条目与激活当成 **f16** 片段喂给 mma。
 * 码本盘上本来就是 f16 ⇒ 权重侧无条件精确; 激活侧是 bf16 格点的 f32, **尾数 7 位 < f16 的 10 位 ⇒ 尾数无损**,
 * 唯一的风险是指数范围:
 *   |x| > 65504   ⇒ f16 直接溢出成 inf(灾难性, 而且 0×inf = NaN 会静默污染整行)
 *   0<|x|<2^-14   ⇒ 落进 f16 次正规区, 尾数被截(精度损失, 但这些数对 5120 项求和几乎没贡献)
 *   0<|x|<2^-24   ⇒ 直接变 0
 * 判据: 三把判决尺(2K/12k/finj 8191)上 **over == 0**, 且次正规占比可忽略 ⇒ 走 f16(指令少、吞吐高);
 * over 不为 0 ⇒ 走 TF32 变体(A/B 都升 f32, 无条件精确, 见 mtp-2.md §5.7)。**编译期定一个, 不留运行时回落。**
 *
 * src 是 f32 激活; b16 是 bf16 打包的中间量(gateup 的 h 出口), 两者给一个、另一个传 NULL。 */
__global__ static void v41_f16range_kernel(unsigned long long *cnt, const float *src, const uint16_t *b16, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v;
    if (src) v = src[i];
    else v = __uint_as_float((uint32_t)b16[i] << 16);
    const float a = fabsf(v);
    atomicAdd(cnt + 3, 1ull);
    if (!(a <= 65504.f)) atomicAdd(cnt + 0, 1ull);              /* 溢出或非有限 */
    else if (a > 0.f && a < 6.103515625e-05f) {                 /* 2^-14: f16 最小正规数 */
        atomicAdd(cnt + 1, 1ull);
        if (a < 5.9604645e-08f) atomicAdd(cnt + 2, 1ull);       /* 2^-24: f16 里直接变 0 */
    }
}
static v41_scratch g_v41_f16rng;
/* 累计计数(进程生命期): [0]=溢出 [1]=次正规 [2]=归零 [3]=总元素。
 * ★两个被查的量各自一套计数器★(2026-09-17 实撞): 第一版让激活(x)与中间量(h)共用一个累加器 + 一个
 * 调用计数, 于是打出来的那一行**标签是最后一次调用的、数字是两者相加的** —— 数看着挺像样, 其实分不清
 * 是谁溢出的。探针自己骗人比没有探针更糟。slot: 0 = 激活, 1 = 中间量。 */
static void v41_f16range_probe(const float *src, const uint16_t *b16, uint64_t n, int slot, const char *what) {
    unsigned long long *c = (unsigned long long *)v41_grow(&g_v41_f16rng, 32, "v41 f16 range");
    unsigned long long h[4] = { 0, 0, 0, 0 };
    static unsigned long long tot[2][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
    static uint64_t calls[2] = { 0, 0 };
    if (!c || slot < 0 || slot > 1) return;
    if (cudaMemsetAsync(c, 0, 32, g_cur_stream) != cudaSuccess) return;
    v41_f16range_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>(c, src, b16, n);
    if (cudaStreamSynchronize(g_cur_stream) != cudaSuccess) return;
    if (cudaMemcpy(h, c, 32, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    for (int i = 0; i < 4; i++) tot[slot][i] += h[i];
    if ((++calls[slot] % 200u) == 0u)   /* 一步 40 层 ⇒ 每 5 步一行 */
        fprintf(stderr, "[f16-range] 累计 %s: 溢出 %llu / 次正规 %llu / 归零 %llu / 共 %llu 个元素\n",
                what, tot[slot][0], tot[slot][1], tot[slot][2], tot[slot][3]);
}
