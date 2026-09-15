/* cuda_v41_fp4_planar.inc.cu — ds4_cuda.cu 分片: fp4x32 权重的"平面副本"(single.md §2.6 的 A 路, 2026-09-16)。
 *
 * 病(ncu 定的): 盘上 fp4x32 是**交错**排的 —— 每 32 个权重 = 16 B nibble + 1 B ue8m0 scale, 跨距 17 B。
 * 17 不是 2 的幂, 于是没有一个块是对齐的, 一个 warp 的 32 个 lane 地址散在几十个扇区上:
 * `l1tex 平均 24.92 个扇区/请求`(理想 4), 这个核只跑到 165 GB/s = 墙的 69%。
 * 对照同一块板上的 engram wkv —— 它在盘上就是**平面**的(e4m3 一片 + 缩放一片), 跑到 214 GB/s = 89%。
 *
 * 修: 装载后在设备上把每个 fp4x32 张量重排成两片 —— nibble 平面(每块 16 B, 整行连续)+ scale 平面
 * (每块 1 B)。这样一个 warp 一轮读的 8 个块 = **128 B 连续, 正好一条 cacheline**, scale 那 8 B 也连续。
 *
 * 为什么不直接改盘上格式: 那要改转换器 + 重转 118 GB 的 GGUF + 过五指标门。先用设备端重排把收益量出来,
 * 兑现了再决定要不要落盘(落盘还能省掉这份双份内存与每次启动的重排时间)。
 * ★数值逐位同★: 同一批字节换个摆法, nibble 与 scale 的配对一个没动。
 *
 * 代价: 主干骨架的 fp4 权重多一份副本(约 3.8 GB)。只对**解码热点**(n_tok ≤ 8 的 GEMV)建, 预填那条路
 * 仍读原始交错布局 —— 它走的是 NVFP4 张量核, 形状和这里不是一回事。 */

typedef struct { const uint8_t *nib, *sc; } v41_fp4_planar;
static std::unordered_map<uint64_t, v41_fp4_planar> g_fp4_planar;   /* key = 张量在文件里的偏移 */
static uint64_t g_fp4_planar_bytes = 0;
/* 建不下就不建(别 OOM 是最高约束): 到了这个上限就停, 之后的张量照旧走交错布局。
 * 3.8 GiB 盖得住主干骨架(每 token 读的 3.79 GB 全部); 三塔与预填专用的张量本来也不该占。 */
#define V41_FP4_PLANAR_BUDGET (4ull << 30)

__global__ static void v41_fp4_to_planar_kernel(uint8_t *nib, uint8_t *sc, const uint8_t *w, uint64_t nblk) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const uint8_t *p = w + b * 17u;
    uint8_t *d = nib + b * 16u;
    /* 16 B 一块, 源不对齐(17 B 跨距)目的对齐 —— 逐 4 B 搬, 编译器会发非对齐读 + 对齐写 */
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) { uint32_t v; memcpy(&v, p + i * 4u, 4); memcpy(d + i * 4u, &v, 4); }
    sc[b] = p[16];
}

/* 返回该张量的平面副本; 没有(还没建/超预算/建失败)返回 nib=NULL, 调用方回落到交错布局。 */
static v41_fp4_planar v41_fp4_planar_get(const void *model_map, uint64_t model_size, uint64_t off,
                                         uint64_t in_dim, uint64_t out_dim, const char *what) {
    v41_fp4_planar none; none.nib = NULL; none.sc = NULL;
    auto it = g_fp4_planar.find(off);
    if (it != g_fp4_planar.end()) return it->second;
    const uint64_t nblk = out_dim * (in_dim / 32u);
    const uint64_t need = nblk * 17u;   /* 平面副本也是 16+1 B/块, 只是分成两片 */
    if (g_fp4_planar_bytes + need > V41_FP4_PLANAR_BUDGET) { g_fp4_planar[off] = none; return none; }
    if (off > model_size || nblk * 17u > model_size - off) { g_fp4_planar[off] = none; return none; }
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, nblk * 17u, what);
    if (!w) { g_fp4_planar[off] = none; return none; }
    void *buf = NULL;
    if (cudaMalloc(&buf, (size_t)need) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [v41] fp4 平面副本建不下(%s, %.1f MB), 这个张量照旧走交错布局\n",
                what, (double)need / 1048576.0);
        g_fp4_planar[off] = none; return none;
    }
    v41_fp4_planar p;
    p.nib = (const uint8_t *)buf;
    p.sc = p.nib + nblk * 16u;
    v41_fp4_to_planar_kernel<<<(unsigned)((nblk + 255u) / 256u), 256, 0, g_cur_stream>>>(
        (uint8_t *)buf, (uint8_t *)buf + nblk * 16u, w, nblk);
    if (!cuda_ok(cudaGetLastError(), "v41 fp4 planar build")) { (void)cudaFree(buf); g_fp4_planar[off] = none; return none; }
    g_fp4_planar_bytes += need;
    g_fp4_planar[off] = p;
    return p;
}
