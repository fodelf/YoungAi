/* cuda_sparse_attn_mma.inc.cu — 稀疏注意力的张量核版(speed.md 段 4, 2026-09-15)。
 *
 * 为什么要这一份: 标量版(cuda_v41_2.inc.cu 的 v41_sparse_attn_kernel)在预填里占 43%, 实测只有 259 GFLOPS,
 * 而这块板子的 BF16 稠密张量核实测 81~91 TFLOPS(S0, gemm_fp4_ceiling)。标量版慢的根子不是访存 ——
 * 09-15 把"扇区效率 / 占用率 / 激活重读 / 键重读"四个访存假设逐个改过, 全部判负(详见 fable5.md);
 * 唯一有效的那次改的是指令数。标量版每处理一个键、一个头, 要付一次 5 条 shuffle 的 warp 归约 + 一次 expf
 * + 两次 bf16 舍入, 才换 16 个有效 FMA。张量核把"一次点积一次 warp 归约"整个换掉。
 *
 * 形状(MLA): 64 个头**共用同一份 512 维 KV**, 所以对一个 query 来说
 *   S[64 头][nkeys] = Q[64][512] · Kᵀ[512][nkeys]        ← 是一个真 GEMM
 *   O[64][512]      = P[64][nkeys] · V[nkeys][512]       ← 同一份 KV 当 V 用(官方就是这么写的)
 * 一 block 管 1 个 query × 16 个头(= 正好一个 wmma 的 M 块), grid (n_tok, 4)。
 * 为什么不是 64 头一 block: q 进 shared 要 64×512×2 B = 64 KB, 再加键块就超 GB10 的 99 KB 上限。
 * 键被 4 个头组各读一遍无所谓 —— 整段键只有 1.3 MB, 全在 24 MB L2 里(09-15 实测: 把这个"重读"消掉反而慢 2%)。
 *
 * 两遍扫键, 不存整张 S: 第一遍算 S 只为拿每个头的 max/sum(在线 softmax 的分母), 第二遍**重算一次 S** 再出 P 与 O。
 * 代价是 S 算两遍(总算力 2 倍变 3 倍), 换来不用给 S 留 [16][640] f32 = 40 KB 的 shared。
 * 为什么不用标准的单遍 flash(边扫边把 O 按新 max 重缩放): wmma 的累加器 fragment 里"哪个线程持有哪一行"
 * 是不透明的, 按行(头)缩放拿不到那个映射; 重算一遍 S 比跟 fragment 布局较劲可靠得多。
 *
 * ★数值★: q/kv 的值本来就落在 bf16 格点上(调用方 rms_norm / act_quant 舍过), 所以转 bf16 是**无损的**;
 * 累加在 f32。与标量版只差累加序 ⇒ 判据是 NLL/五指标, 不是逐位(与 09-15 第二轮按键分块同一口径)。
 * P 在乘 V 之前舍 bf16 —— 这一步不是为了省, 是官方 acc_s_cast 就这么做, 标量版也一样。
 *
 * 出错会怎样: 无效 topk 槽(idx<0)的键行必须**写 0 并且把分数压成 -inf 两件都做**。只写 0 的话点积是 0 不是 -inf,
 * softmax 会给它一个 exp(0) 的权重, 表现是长上下文答案慢慢跑偏而不报错; 只压 -inf 不清零的话,
 * shared 里上一块的残留会被当成键参与 P·V —— 那正是 09-15 段 0 定罪的 0×NaN 那条 bug。 */

#define DS4_ATTN_MMA_HEADS 16u   /* 一 block 管几个头 = 一个 wmma M 块 */
#define DS4_ATTN_MMA_KT    16u   /* 一次进 shared 的键数 = 一个 wmma N 块 */
#define DS4_ATTN_MMA_WARPS 8u    /* 一 block 8 个 warp: 第一/二遍分 K 段, 出 O 时分输出维 */
#define DS4_ATTN_MMA_HD    512u  /* 头维, 与 v41 形状绑死(调用方已校验) */

/* 把 16 个键行搬进 shared 并转 bf16; 无效槽整行清零, valid[] 记状态(给后面压 -inf 用)。
 * 键序 = 窗口行(升序)后接 topk 压缩行, 与标量版一字不差。
 * ★压缩行的解包(2026-09-29, gguf-tools/bench/v41_attn_seg_bench.cu 定形)★: 原来每个元素都走一次 v41_ckv_get =
 * 一次 e4m3 缩放解码(带分支 + ldexpf)+ 一次 fp4 查表(static const 表 = LDC 按 lane 下标重放)+ 一次手写 RNE 舍入,
 * 一行 512 个元素解 512 次缩放而这行只有 32 个缩放。12k 真形状微基准: 这一发 n=1 2.4 → 1.4 ms/步, 验证批 n=4 7.9 → 5.0 ——
 * 慢的是解码指令, 不是字节。改: 32 个缩放 lane i 解第 i 个、元素循环 shfl 取; fp4 的 16 个值 lane l 各持一个, 查表 = 一条 shfl;
 * 乘完直接 cvt.rn(它本身就是 RNE, 与 v41_bf16r 同值)。同一个 scale 同一个 nibble 同一次乘法 ⇒ 逐位同(微基准两道逐位门 +
 * 引擎 d1 门), 预填 mma 版与解码 seg 版共用这一个 gather, 两条路一起换。 */
__device__ __forceinline__ static void ds4_attn_mma_gather_keys(
        __nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
        uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
        uint32_t window, uint32_t ng, uint32_t topk, uint32_t ring) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));   /* fp4 值表: lane l 持第 l&15 个 */
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;   /* 窗口行 f32 / 压缩行打包 FP4, 见 cuda_kv_pack */
        if (t < nt) {
            /* ring: 主路历史段是环(1); DSpark 草稿塔的窗口是线性段(0, 2026-09-29 起塔也走 mma 版), 见 v41_win_row */
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, ring) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_MMA_HD;
        if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
        else if (cpk) {
            const float sc = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);   /* 一行 32 个缩放 = 32 个 lane 各解一个 */
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ATTN_MMA_HD / 32u; j++) {
                const uint32_t d = lane + 32u * j;
                const uint8_t by = cpk[d >> 1];
                const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);   /* 与 v41_ckv_get 同一取法 */
                const float s = __shfl_sync(0xffffffffu, sc, (int)(d >> 4));
                kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int)nib) * s);
            }
        } else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}

/* 一个键块的 S 片 [16 头][16 键]: 8 个 warp 各算 K 维的 1/8(4 个 wmma k 步), partial 落 shared 后按固定序相加。
 * 固定序 = 每次都一样 ⇒ 结果确定(温 0 复现是硬要求, 见 09-15 段 0)。 */
__device__ __forceinline__ static void ds4_attn_mma_scores(
        float *stile, float *spart, const __nv_bfloat16 *qs, const __nv_bfloat16 *ks,
        const int *valid, uint32_t nt, float scale) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;   /* B[d][key] = ks[key][d] ⇒ 列主序, ldm=512 */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    const uint32_t ksteps = DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS;   /* 512/16/8 = 4 */
    for (uint32_t s = 0; s < ksteps; s++) {
        const uint32_t d0 = (warp * ksteps + s) * 16u;
        wmma::load_matrix_sync(a, qs + d0, DS4_ATTN_MMA_HD);
        wmma::load_matrix_sync(b, ks + d0, DS4_ATTN_MMA_HD);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(spart + (size_t)warp * 256u, c, 16, wmma::mem_row_major);
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) {
        float v = 0.f;
        for (uint32_t w = 0; w < DS4_ATTN_MMA_WARPS; w++) v += spart[(size_t)w * 256u + e];
        const uint32_t k = e & 15u;   /* stile 行主序 [头][键] */
        stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
    }
    __syncthreads();
}

/* 一个键块的在线 max/sum, 并行版(2026-09-29, 微基准 V7): warp w 管头 2w(lane 0..15)与 2w+1(lane 16..31), 每 lane 一个键。
 * 原来 16 个线程各自串行做 16 次 fmaxf + 16 次 expf, 其余 240 个线程干等。
 * ★逐位同★: max 是精确运算(序无关); 和 = 先 rsum·expf(m−tm), 再按 k = 0..15 顺序加 expf(s_k−tm) —— 与串行版同一序同一值。
 * 全 block 的线程都要调(warp 内 shfl 是集体操作), 不能再包在 threadIdx.x < 16 里。 */
__device__ __forceinline__ static void ds4_attn_mma_stats(const float *stile, float *rmax, float *rsum) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, k = lane & 15u;
    const uint32_t h = warp * 2u + (lane >> 4);
    const float s = stile[h * 16u + k], m = rmax[h];
    float tm = fmaxf(m, s);
    tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 8)); tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
    tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2)); tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
    const float e = expf(s - tm);
    float sm = rsum[h] * expf(m - tm);
    const int b = (int)(lane & 16u);
    #pragma unroll
    for (int kk = 0; kk < 16; kk++) sm += __shfl_sync(0xffffffffu, e, b + kk);
    if (k == 0u) { rmax[h] = tm; rsum[h] = sm; }
}

/* ★预填核换单遍 flash 形态(2026-09-29; 微基准 gguf-tools/bench/v41_attn_prefill_bench.cu 定形, 12k 形状 2048 查询: 33.6 → 12.5 ms/层块)★
 * 与两遍扫键的旧核(上面三个积木仍给解码 seg 核用)差在四处:
 *   ①q 不进 shared: 每个 warp 把自己 K 段(64 维 = 4 个 k16)的 A 片段常驻寄存器(16 个), 省 16 KB shared;
 *   ②S 用 mma.m16n8k16 直算(不走 wmma), 8 个 warp 分 K, partial 落 shared 按固定序相加(同一分工, 只换指令);
 *   ③单遍: 每个键块算完 S 就更新 max/sum, O 累加器按 alpha = exp(m_old − m_new) 缩放后加 P·V —— mma 的累加器布局是文档定死的
 *     (c0,c1 = 行 lane/4, c2,c3 = 行 +8), 按头缩放拿得到那个映射(旧核注释说 wmma 的 fragment 不透明, 这正是换 mma 的理由);
 *     键只 gather 一次、S 只算一次(旧核各两次);
 *   ④键片行距 520 个 bf16(1040 B): 行距 1024 B 时 ldmatrix/wmma 取 8 行同一列全落同一组 bank(8 路冲突)。
 * ★数值★: 与旧核不逐位同(P 的 bf16 舍入点同, 但 O 多了 alpha 缩放的 f32 乘法, 累加序也变); 微基准五个核对 f64 参考同为 3.48e-3(= bf16 出口舍入)。
 * 门 = 五指标/PPL(与 09-15 mma 版落地同一规矩)。解码 seg 核(cuda_v41_attn_mma_decode.inc.cu)一个字没动, 投机 == 纯解码逐字节门照旧。 */
#define DS4_ATTN_FA_LD 520u   /* 键片行距(bf16 元素) */
__device__ __forceinline__ static void ds4_fa_ldsm4(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void ds4_fa_ldsm4t(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ static void ds4_fa_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ static uint32_t ds4_fa_bf16x2(float lo, float hi) {
    return (__float_as_uint(v41_bf16r(lo)) >> 16) | (__float_as_uint(v41_bf16r(hi)) & 0xffff0000u);
}
/* gather: 与 ds4_attn_mma_gather_keys 同一解码式(值逐位同), 只是行距 DS4_ATTN_FA_LD */
__device__ __forceinline__ static void ds4_fa_gather(__nv_bfloat16 *ks, int *valid, const float *kvw, const uint8_t *kvc, const int32_t *idx,
                                                     uint32_t i, uint32_t base, uint32_t nt, uint32_t nwin, uint32_t lo, uint32_t pos0,
                                                     uint32_t window, uint32_t ng, uint32_t topk) {
    const uint32_t lane = threadIdx.x & 31u;
    const float tv = ds4_fp4_nibble_to_f32((uint8_t)(lane & 15u));
    for (uint32_t t = threadIdx.x / 32u; t < DS4_ATTN_MMA_KT; t += blockDim.x / 32u) {
        const uint32_t kk = base + t;
        const float *krow = NULL; const uint8_t *cpk = NULL;
        if (t < nt) {
            if (kk < nwin) krow = kvw + v41_win_row((int64_t)lo + kk, pos0, window, 1u) * DS4_ATTN_MMA_HD;
            else if (kvc && idx) { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)];
                                   if (g >= 0 && (uint32_t)g < ng) cpk = kvc + (uint64_t)g * DS4_V41_CKV_BYTES; }
        }
        if (lane == 0) valid[t] = (krow || cpk) ? 1 : 0;
        __nv_bfloat16 *kt = ks + (size_t)t * DS4_ATTN_FA_LD;
        if (krow) { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = __float2bfloat16(krow[d]); }
        else if (cpk) {
            const float sc = ds4_e4m3fn_to_f32(cpk[DS4_V41_CKV_NIB + lane]);
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ATTN_MMA_HD / 32u; j++) {
                const uint32_t d = lane + 32u * j;
                const uint8_t by = cpk[d >> 1];
                const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
                const float s = __shfl_sync(0xffffffffu, sc, (int)(d >> 4));
                kt[d] = __float2bfloat16(__shfl_sync(0xffffffffu, tv, (int)nib) * s);
            }
        } else { for (uint32_t d = lane; d < DS4_ATTN_MMA_HD; d += 32u) kt[d] = (__nv_bfloat16)0.0f; }
    }
}
/* shared: ks 16×520×2 + spart 8×16×16×4 + stile 16×16×4 + ptile 16×16×2 + rmax/rsum/alpha 3×16×4 + valid 16×4 ≈ 26 KB ⇒ 每 SM 挂 3 个 block */
__global__ __launch_bounds__(256) static void ds4_sparse_attn_mma_kernel(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                                  const int32_t *idx, const float *sink, uint32_t pos0, uint32_t window,
                                                  uint32_t ng, uint32_t topk, uint32_t n_head, float scale, uint32_t win_lo) {
    constexpr uint32_t KT = DS4_ATTN_MMA_KT, HB = DS4_ATTN_MMA_HEADS;
    extern __shared__ __align__(16) char ds4_attn_fa_smem[];   /* 另起一个名: 解码 seg 核的 ds4_attn_mma_smem 是 char[], 同名不同型编不过 */
    __nv_bfloat16 *ks = (__nv_bfloat16 *)ds4_attn_fa_smem;                        /* [16][520] */
    float *spart = (float *)(ks + (size_t)KT * DS4_ATTN_FA_LD);                     /* [8 warp][16][16] */
    float *stile = spart + 8u * HB * KT;                                            /* [16][16] */
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + HB * KT);                      /* [16][16] */
    float *rmax = (float *)(ptile + HB * KT), *rsum = rmax + HB, *alpha = rsum + HB;
    int *valid = (int *)(alpha + HB);
    const uint32_t i = blockIdx.x, h0 = blockIdx.y * HB, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t p = pos0 + i;
    uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    if (lo < win_lo) lo = win_lo;   /* 环里 win_lo 之前的槽没写过(CED), 不读(官方 -1 屏蔽同义); 见 ds4_gpu_v41.h */
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    /* q 的 A 片段常驻寄存器: warp w 管维 [64w, 64w+64) 的 4 个 k16(a0: 行 r 列 c..c+1; a1: 行 r+8; a2: 列 +8; a3: 行 +8 列 +8) */
    uint32_t qa[4][4];
    {
        const uint32_t r = lane >> 2, c = (lane & 3u) * 2u;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            const float *q0 = q + ((uint64_t)i * n_head + h0 + r) * DS4_ATTN_MMA_HD + warp * 64u + s * 16u + c;
            const float *q1 = q0 + 8u * DS4_ATTN_MMA_HD;
            qa[s][0] = ds4_fa_bf16x2(q0[0], q0[1]); qa[s][1] = ds4_fa_bf16x2(q1[0], q1[1]);
            qa[s][2] = ds4_fa_bf16x2(q0[8], q0[9]); qa[s][3] = ds4_fa_bf16x2(q1[8], q1[9]);
        }
    }
    if (threadIdx.x < HB) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    float oacc[8][4];   /* O[16 头][本 warp 64 维] = 8 个 n8 片 */
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) { oacc[j][0] = oacc[j][1] = oacc[j][2] = oacc[j][3] = 0.f; }
    __syncthreads();
    for (uint32_t base = 0; base < nkeys; base += KT) {
        const uint32_t nt = (nkeys - base) < KT ? (nkeys - base) : KT;
        __syncthreads();   /* 上一块的 ks/ptile 用完 */
        ds4_fa_gather(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        {   /* ① S 部分和: 本 warp 的 64 维 × 16 头 × 16 键(两个 n8 片) */
            float sp[2][4] = { {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f} };
            #pragma unroll
            for (uint32_t s = 0; s < 4u; s++) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt >> 1) * 8u + (lane & 7u), col = warp * 64u + s * 16u + (mt & 1u) * 8u;
                ds4_fa_ldsm4(b, ks + (size_t)key * DS4_ATTN_FA_LD + col);
                ds4_fa_mma(sp[0], qa[s], b[0], b[1]); ds4_fa_mma(sp[1], qa[s], b[2], b[3]);
            }
            #pragma unroll
            for (uint32_t nb = 0; nb < 2u; nb++) {   /* c0,c1 = 行 lane/4 列 (lane&3)*2+{0,1}; c2,c3 = 行 +8 */
                float *sw = spart + (size_t)warp * HB * KT + (lane >> 2) * KT + nb * 8u + (lane & 3u) * 2u;
                sw[0] = sp[nb][0]; sw[1] = sp[nb][1]; sw[8u * KT] = sp[nb][2]; sw[8u * KT + 1u] = sp[nb][3];
            }
        }
        __syncthreads();
        /* ② 固定序求和 + 缩放 + 屏蔽(无效槽 -1e30 ⇒ exp 为 0) */
        {
            const uint32_t e = threadIdx.x, k = e % KT;
            float v = 0.f;
            #pragma unroll
            for (uint32_t w = 0; w < 8u; w++) v += spart[(size_t)w * HB * KT + e];
            stile[e] = (k < nt && valid[k]) ? v * scale : -1e30f;
        }
        __syncthreads();
        /* ③ 在线 max/sum + P 片(bf16, 官方 acc_s_cast) + alpha: 一线程一头, 键序串行(与旧核串行版同序) */
        if (threadIdx.x < HB) {
            const uint32_t h = threadIdx.x; const float *sr = stile + h * KT; const float m = rmax[h];
            float tm = m;
            for (uint32_t k = 0; k < KT; k++) tm = fmaxf(tm, sr[k]);
            const float al = expf(m - tm);
            float sm = rsum[h] * al;
            for (uint32_t k = 0; k < KT; k++) { const float pv = expf(sr[k] - tm); sm += pv; ptile[h * KT + k] = __float2bfloat16(pv); }
            rmax[h] = tm; rsum[h] = sm; alpha[h] = al;
        }
        __syncthreads();
        /* ④ O = O·alpha + P·V: A = ptile [16 头][16 键], B = ks [键][维] 经 .trans 取; 本 warp 管维 [64w, 64w+64) 的 8 个 n8 片 */
        {
            const float a0 = alpha[lane >> 2], a1 = alpha[(lane >> 2) + 8u];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) { oacc[j][0] *= a0; oacc[j][1] *= a0; oacc[j][2] *= a1; oacc[j][3] *= a1; }
            uint32_t pa[4];
            {   const uint32_t mt = lane >> 3, row = (lane & 7u) + (mt & 1u) * 8u, col = (mt >> 1) * 8u;
                ds4_fa_ldsm4(pa, ptile + (size_t)row * KT + col); }
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j += 2u) {
                uint32_t b[4];
                const uint32_t mt = lane >> 3, key = (mt & 1u) * 8u + (lane & 7u), col = warp * 64u + j * 8u + (mt >> 1) * 8u;
                ds4_fa_ldsm4t(b, ks + (size_t)key * DS4_ATTN_FA_LD + col);
                ds4_fa_mma(oacc[j], pa, b[0], b[1]); ds4_fa_mma(oacc[j + 1], pa, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    /* 出口: 除以 (sum + exp(sink − max))(sink 只进分母, 与标量版同), 舍 bf16 写回 */
    {
        const uint32_t hA = lane >> 2, hB = hA + 8u;
        const float dA = rsum[hA] + expf(sink[h0 + hA] - rmax[hA]), dB = rsum[hB] + expf(sink[h0 + hB] - rmax[hB]);
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t d = warp * 64u + j * 8u + (lane & 3u) * 2u;
            float *oA = o + ((uint64_t)i * n_head + h0 + hA) * DS4_ATTN_MMA_HD + d, *oB = o + ((uint64_t)i * n_head + h0 + hB) * DS4_ATTN_MMA_HD + d;
            oA[0] = v41_bf16r(oacc[j][0] / dA); oA[1] = v41_bf16r(oacc[j][1] / dA);
            oB[0] = v41_bf16r(oacc[j][2] / dB); oB[1] = v41_bf16r(oacc[j][3] / dB);
        }
    }
}

/* 预填 flash 核的 shared 用量(≈ 26 KB); 解码 seg 核仍用 ds4_attn_mma_seg_smem_bytes(旧布局: qs + ks + spart + …) */
static size_t ds4_attn_mma_smem_bytes(void) {
    return (size_t)DS4_ATTN_MMA_KT * DS4_ATTN_FA_LD * sizeof(__nv_bfloat16)
         + (size_t)DS4_ATTN_MMA_WARPS * DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * sizeof(float)
         + (size_t)DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_KT * (sizeof(float) + sizeof(__nv_bfloat16))
         + 3u * DS4_ATTN_MMA_HEADS * sizeof(float) + DS4_ATTN_MMA_KT * sizeof(int);
}
static size_t ds4_attn_mma_seg_smem_bytes(void) {
    return (size_t)(DS4_ATTN_MMA_HEADS + DS4_ATTN_MMA_KT) * DS4_ATTN_MMA_HD * sizeof(__nv_bfloat16)
         + (size_t)DS4_ATTN_MMA_WARPS * 256u * sizeof(float) + 256u * sizeof(float)
         + 256u * sizeof(__nv_bfloat16) + 2u * DS4_ATTN_MMA_HEADS * sizeof(float)
         + DS4_ATTN_MMA_KT * sizeof(int);
}

/* 能不能用: 形状对得上(64 头 × 512 维, 头数能被 16 整除)+ 块够大(n_tok 小的时候 grid 只有 n_tok×4 个 block,
 * 填不满 48 个 SM, 标量版反而好) + shared 抬得上去。返回 0 = 调用方回标量版。 */
static int ds4_sparse_attn_mma_launch(float *o, const float *q, const float *kvw, const uint8_t *kvc,
                                      const int32_t *idx, const float *sink, uint32_t n_tok, uint32_t pos0,
                                      uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head,
                                      uint32_t head_dim, float scale, uint32_t win_lo) {
    if (head_dim != DS4_ATTN_MMA_HD || (n_head % DS4_ATTN_MMA_HEADS) || n_tok < 64u) return 0;
    static int s_ok = 0;   /* 0 未试 / 1 可用 / -1 抬不上去 */
    const size_t smem = ds4_attn_mma_smem_bytes();
    if (s_ok == 0) {
        s_ok = cudaFuncSetAttribute(ds4_sparse_attn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess ? 1 : -1;
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [attn] 张量核版 shared %zu KB %s\n", smem >> 10, s_ok == 1 ? "已开" : "★抬不上去, 回标量版★");
    }
    if (s_ok != 1) return 0;
    ds4_sparse_attn_mma_kernel<<<dim3(n_tok, n_head / DS4_ATTN_MMA_HEADS), DS4_ATTN_MMA_WARPS * 32u, smem, g_cur_stream>>>(
        o, q, kvw, kvc, idx, sink, pos0, window, ng, topk, n_head, scale, win_lo);
    return cuda_ok(cudaGetLastError(), "sparse attn mma");
}
