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


/* ★2026-09-15 clear.md C0: 三个 f16 dequant 核已删★
 *   vq_dequant_kernel / vq_dequant_pairs_kernel / vq_cold_w2_kernel / cuda_vq_pay_hdr。
 * 它们把 VQ 位流解成 **f16 权重矩阵**落暂存, 再交给 cuBLAS —— 那条预填路 09-15 第三轮起
 * 已被"解码即乘 + NVFP4 张量核"取代(cuda_vq_prefill_nvfp4.inc.cu), 四个符号零调用者。
 * 留下的是 fused2 解码核真正在用的公共件: 槽表寻址、warp 规约、几个尺寸常量、剖面计数器。
 * ★注意★ 码本本身在盘上仍是 f16(每矩阵 64 KB), 所以下面 fused2 那边读码本还是 __half —— 那是
 * 盘上格式, 要改得动转换器(clear.md C1: 码本去重 + FP8), 不是这里能删掉的。 */

__device__ __forceinline__ static uint64_t vq_slot_dev(const uint8_t *blob, int e, int which) {
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    return off;
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

