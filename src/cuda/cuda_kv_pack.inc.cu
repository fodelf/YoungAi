/* cuda_kv_pack.inc.cu — ds4_cuda.cu 分片: 全局 KV 缓存按官方格式打包(decode.md D1, 2026-09-16)。
 *
 * 说人话: 压缩 KV 和索引键原来各存一行 f32。但它们的**值早就被量化过了** ——
 * 存进缓存前走的是 act_quant(FP4 + 每 16/32 个一个缩放), 每个数只有 16 个可能取值,
 * 却占着 4 个字节。官方存的是打包后的字节: 主 KV 288 B/组、索引 K 68 B/组。
 *
 * 所以这一刀不是降精度, 是**把已经量化过的值按它本来的宽度存**:
 *   主 KV : 512 维, 每 16 个一个 E4M3 缩放 ⇒ 256 B nibble + 32 B scale = 288 B/组(官方同数)
 *   索引 K: 128 维, 每 32 个一个 E8M0 缩放 ⇒  64 B nibble +  4 B scale =  68 B, 补到 72 对齐
 * 值逐位不变: 打包存的是 (nibble, scale), 读回来算 bf16(nibble×scale) —— 与原来存的
 * f32 是同一个表达式的同一个结果(act_quant 最后一行就是 p[lane] = bf16(q*s))。
 *
 * ★为什么补那 4 个字节★: 68 不是 8 的倍数, 组与组之间的起点会错开, 一个 warp 读一组 64 B
 * 就要跨两个 32 B 扇区还对不齐。补到 72(8 对齐)多花 6% 空间, 1M 上下文也才多 10 MB。
 *
 * 出错会怎样: 缓存按 f32 的尺寸分配、按打包的尺寸写 ⇒ 越界踩别的层的缓存, 不报错,
 * 症状是长上下文答非所问; 反过来(按打包分配、按 f32 读)读到的是别的组的字节, 同样静默。
 * 所以尺寸常量只有这一份(ds4_gpu_v41.h), 分配端与核端都引它。 */

/* 打包: 一个 warp 管一个量化块(≤32 个元素), 与 v41_act_quant_kernel 的分块、求最大值、
 * 定缩放三步逐字同 —— 那三步一旦与 act_quant 漂开, 值就不再逐位相同了。
 * mode: 0 = 主 KV(块 16, E4M3 缩放) / 1 = 索引 K(块 32, E8M0 缩放)。 */
__global__ static void v41_kv_pack_kernel(uint8_t *cache, const float *rows, uint32_t g0,
                                          uint32_t dim, uint32_t blk, uint32_t row_bytes,
                                          uint32_t nib_bytes, uint32_t nb_row, int mode) {
    const uint32_t r = blockIdx.x / nb_row, b = blockIdx.x % nb_row, lane = threadIdx.x;
    const float *p = rows + (uint64_t)r * dim + (uint64_t)b * blk;
    const float v = lane < blk ? p[lane] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s;
    if (mode == 0) { a = fmaxf(a, 6.0f * ldexpf(1.0f, -9));   s = ds4_e4m3fn_round(a / 6.0f); }
    else           { a = fmaxf(a, 6.0f * ldexpf(1.0f, -126)); s = v41_pow2_ceil_log2(a / 6.0f); }
    uint8_t *row = cache + (uint64_t)(g0 + r) * row_bytes;
    if (lane == 0) row[nib_bytes + b] = mode == 0 ? ds4_e4m3fn_f32_to_byte(s) : ds4_e8m0_f32_to_byte(s);
    /* nibble: 低半字节是偶数号元素, 高半字节是奇数号 —— 与 fp4x32 GEMV 的解包约定同一套 */
    const uint8_t nib = lane < blk ? ds4_fp4_f32_to_nibble(v / s) : 0u;
    const uint8_t odd = (uint8_t)__shfl_down_sync(0xffffffffu, (uint32_t)nib, 1);
    if (lane < blk && (lane & 1u) == 0u) row[(b * blk + lane) >> 1] = (uint8_t)(nib | (odd << 4));
}

/* 解包: 缓存第 d 维的值 = bf16(nibble × scale)。与打包端合起来就是 act_quant 的那一行。 */
__device__ __forceinline__ static float v41_ckv_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
/* ★缩放先解一遍再用★(decode.md D2, 2026-09-16): 注意力把一行 512 维整行搬进 shared, 上面那个版本
 * 每个元素都要走一次 ds4_e4m3fn_to_f32(带分支 + ldexpf), 一行就是 512 次, 而这行只有 **32 个**缩放。
 * 先让一个 warp 把 32 个缩放解进 shared(每 lane 一个), 元素循环只剩一次 shared 读 —— 一行省掉 480 次解码。
 * 值一模一样(同一个 scale 同一个 nibble 同一次乘法), 所以门仍是逐字节。 */
__device__ __forceinline__ static float v41_ckv_get_s(const uint8_t *row, uint32_t d, const float *scales) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * scales[d >> 4]);
}
__device__ __forceinline__ static float v41_idxk_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1];
    const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return v41_bf16r(ds4_fp4_nibble_to_f32(nib) * ds4_e8m0_to_f32(row[DS4_V41_IDXK_NIB + (d >> 5)]));
}

/* rows[n][512](已 rope 的 latent, 还没量化) → 缓存第 g0 组起的 n 组, 量化+打包一发做完。
 * 原来是"act_quant 就地改 f32 → 整行拷进缓存"两发, 现在一发, 而且缓存小 7 倍。 */
int ds4_gpu_v41_ckv_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows) {
    if (!cache || !rows || !n_rows) return 0;
    if (cache->bytes < (uint64_t)(g0 + n_rows) * DS4_V41_CKV_BYTES) return 0;
    /* 维度从打包常量推出来(一个字节两个 nibble), 不引 DS4_N_* —— 那些宏在这个翻译单元里看不见 */
    const uint32_t dim = DS4_V41_CKV_NIB * 2u, nb = dim / DS4_V41_CKV_BLK;
    v41_kv_pack_kernel<<<(unsigned)((uint64_t)n_rows * nb), 32, 0, g_cur_stream>>>(
        (uint8_t *)cache->ptr, (const float *)rows->ptr, g0, dim, DS4_V41_CKV_BLK,
        DS4_V41_CKV_BYTES, DS4_V41_CKV_NIB, nb, 0);
    return cuda_ok(cudaGetLastError(), "v41 ckv pack");
}
int ds4_gpu_v41_idxk_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows) {
    if (!cache || !rows || !n_rows) return 0;
    if (cache->bytes < (uint64_t)(g0 + n_rows) * DS4_V41_IDXK_BYTES) return 0;
    const uint32_t dim = DS4_V41_IDXK_NIB * 2u, nb = dim / DS4_V41_IDXK_BLK;
    v41_kv_pack_kernel<<<(unsigned)((uint64_t)n_rows * nb), 32, 0, g_cur_stream>>>(
        (uint8_t *)cache->ptr, (const float *)rows->ptr, g0, dim, DS4_V41_IDXK_BLK,
        DS4_V41_IDXK_BYTES, DS4_V41_IDXK_NIB, nb, 1);
    return cuda_ok(cudaGetLastError(), "v41 idxk pack");
}

/* ---- 激活量化就地(官方 act_quant_kernel / fp4_quant_kernel, inplace 分支) ----
 * fast_round_scale(amax, inv) = 2^ceil(log2(amax·inv)); 官方的 fast_log2_ceil 是位运算版 ceil(log2)。
 * 一 warp 一个 block(≤32 元素), 一线程一元素。 */
/* v41_pow2_ceil_log2 的正本在 cuda_v41_1.inc.cu(打包核也要用它, 而那一片排在前面) */
/* ★2026-09-15 段 5: 网格从二维改成一维展平★
 * 实撞: `--v41-chunk 2048` 在 L02 报 "act quant fp4 failed: invalid argument"。原因是索引器那一路
 * 按 (n_tok × 32 个索引头) 当行数, 2048 × 32 = **65536**, 而 CUDA 的 gridDim.y 上限是 **65535** —— 差一个。
 * 报的是 invalid argument, 不是"超上限", 看着像参数写错, 实际是网格形状。展平成一维后上限变成 2^31-1,
 * 块开到 4096(段 5 要的)也够。 */
__global__ static void v41_act_quant_kernel(float *x, uint32_t dim, uint32_t block, int mode, uint32_t nb_row) {
    /* mode 0: fp8 e4m3 值 + ue8m0 scale(max 448); 1: fp4 + ue8m0(max 6); 2: fp4 + e4m3 scale(max 6) */
    const uint32_t row = blockIdx.x / nb_row, b = blockIdx.x % nb_row;
    const uint32_t lane = threadIdx.x;
    float *p = x + (uint64_t)row * dim + (uint64_t)b * block;
    float v = lane < block ? p[lane] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s;
    if (mode == 0) { a = fmaxf(a, 1e-4f); s = v41_pow2_ceil_log2(a / 448.0f); }
    else if (mode == 1) { a = fmaxf(a, 6.0f * ldexpf(1.0f, -126)); s = v41_pow2_ceil_log2(a / 6.0f); }
    else { a = fmaxf(a, 6.0f * ldexpf(1.0f, -9)); s = ds4_e4m3fn_round(a / 6.0f); }
    if (lane < block) {
        float q = v / s;
        if (mode == 0) { q = fminf(fmaxf(q, -448.0f), 448.0f); q = ds4_e4m3fn_round(q); }
        else { q = fminf(fmaxf(q, -6.0f), 6.0f); q = ds4_e2m1fn_round(q); }
        p[lane] = v41_bf16r(q * s);
    }
}
int ds4_gpu_v41_act_quant_fp8_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block) {
    if (!x || block == 0 || block > 32u || (dim % block)) return 0;
    v41_act_quant_kernel<<<(unsigned)((uint64_t)n_rows * (dim / block)), 32, 0, g_cur_stream>>>((float *)x->ptr, dim, block, 0, dim / block);
    return cuda_ok(cudaGetLastError(), "v41 act quant fp8");
}
int ds4_gpu_v41_act_quant_fp4_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, bool e4m3_scale) {
    if (!x || block == 0 || block > 32u || (dim % block)) return 0;
    v41_act_quant_kernel<<<(unsigned)((uint64_t)n_rows * (dim / block)), 32, 0, g_cur_stream>>>((float *)x->ptr, dim, block, e4m3_scale ? 2 : 1, dim / block);
    return cuda_ok(cudaGetLastError(), "v41 act quant fp4");
}
