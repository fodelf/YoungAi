/* cuda_v41_q4k.inc.cu — ds4_cuda.cu 分片: 骨架 q4_K 的解码 GEMV / 预填解量化 / 嵌入取行(2026-09-19)。
 *
 * 【为什么有这一族】骨架此前只有 fp4x32(4.25 bpw) 一种。09-19 的 100 GB 配方把骨架换成 q4_K
 * (4.5 bpw, 144 B/256 元素): 每块一个 f16 主 scale + 一个 f16 主 min + 8 组 6-bit 子 scale/min,
 * 比"一个 2 的幂 × 32 列"多了 min 项 ⇒ 能表示非零均值的块, 同三张量的权重残差 1.37% → 0.51%。
 * 盘上两种格式并存, 引擎按登记类型分发(core_v41_attn.c 的 v41_tproj), 老 GGUF 一行不用改。
 *
 * 【解码值的定义只有一处】`src/common/ds4_quantfmt.c` 的 ds4_deq_q4_K 是金标(ds4_unit 有逐字节夹具),
 * 本文件的核与它逐式同源。★改这里必须回去对拍那个夹具★ —— q4_K 的 12 B 里塞了 8 组 6-bit
 * scale + 8 组 6-bit min, 打包错一位不报错, 只让整组 32 个元素偏一个常数。
 *
 * 【lane 排布 = 一次 uint32 拿 4 个字节】09-18 量到的硬约束: GB10 的 DRAM 按 64 B 取, 一条 load
 * 读不满 128 B 整线就只有 142 GB/s(整线 219)。所以 lane l 读 qs[4l..4l+3] —— 一个 warp 一轮
 * 128 B 连续。qs 的字节 p 同时给两个元素: 低 nibble → 子块 2(p/32), 高 nibble → 子块 2(p/32)+1,
 * 两者相距 32 个元素(布局见 ds4_deq_q4_K 的 j += 64 那个循环)。 */

#define V41_Q4K_BLK 256u          /* 每块元素数 */
#define V41_Q4K_BYTES 144u        /* 每块字节数: 2(d) + 2(dmin) + 12(scales) + 128(qs) */

/* 块头解出 8 组 (scale, min) 的第 j 组。与 ds4_quantfmt.c 的 q4k_scale_min 逐式同。 */
__device__ __forceinline__ static void v41_q4k_sm(const uint8_t *sc, int j, float *s, float *m) {
    uint8_t a, b;
    if (j < 4) { a = sc[j] & 63u; b = sc[j + 4] & 63u; }
    else { a = (uint8_t)((sc[j + 4] & 0xFu) | ((sc[j - 4] >> 6) << 4));
           b = (uint8_t)((sc[j + 4] >> 4)  | ((sc[j - 0] >> 6) << 4)); }
    *s = (float)a; *m = (float)b;
}

/* 块头 16 B 已在寄存器(一条 uint4 读进来)时取第 k 个 scale 字节(k ∈ [0,12))。
 * ★不许写成 ((const uint8_t *)&h)[k]★: k 随 lane 变(gidx), 对寄存器数组动态下标会掉进 local memory。
 * 这里用三选一 + 移位, 编出来是 SEL/SHF, 不碰内存。 */
__device__ __forceinline__ static uint32_t v41_q4k_scb(const uint4 &h, uint32_t k) {
    const uint32_t w = k < 4u ? h.y : (k < 8u ? h.z : h.w);
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}
/* 同 v41_q4k_sm, 只是字节来自寄存器里的块头。解出的 (s, m) 与 ds4_quantfmt.c 的 q4k_scale_min 逐式同。 */
__device__ __forceinline__ static void v41_q4k_sm_reg(const uint4 &h, uint32_t j, float *s, float *m) {
    uint32_t a, b;
    if (j < 4u) { a = v41_q4k_scb(h, j) & 63u; b = v41_q4k_scb(h, j + 4u) & 63u; }
    else { a = (v41_q4k_scb(h, j + 4u) & 0xFu) | ((v41_q4k_scb(h, j - 4u) >> 6) << 4);
           b = (v41_q4k_scb(h, j + 4u) >> 4)   | ((v41_q4k_scb(h, j) >> 6) << 4); }
    *s = (float)a; *m = (float)b;
}
/* 一个 warp 对一个 q4_K 块与 NT 条激活做点积, 结果累加进 acc[NT]; 块头与 qs 由调用方从 shared 里的整段权重取(见下面 stage/pipe)。
 * lane l: gidx = l>>3 选 64 元素组, q0 = (l&7)*4 选组内 4 个连续元素。
 * 低半 → 元素 gidx*64 + q0 + i(子块 2·gidx); 高半 → 再 +32(子块 2·gidx+1)。 */
template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_blk_acc_reg(const uint4 &h, uint32_t qw, const float *x, uint32_t x_stride,
                                                           uint32_t base, float *acc) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    const float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFFu)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm_reg(h, gidx * 2u, &s_lo, &m_lo);
    v41_q4k_sm_reg(h, gidx * 2u + 1u, &s_hi, &m_hi);
    const float dl = d * s_lo, ml = dmin * m_lo, dh = d * s_hi, mh = dmin * m_hi;
    float wl[4], wh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        wl[i] = dl * (float)(byte & 0xFu) - ml;
        wh[i] = dh * (float)(byte >> 4) - mh;
    }
    const uint32_t e_lo = base + gidx * 64u + q0, e_hi = e_lo + 32u;
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        const float *xt = x + (uint64_t)t * x_stride;
        const float4 xl = *(const float4 *)(xt + e_lo), xh = *(const float4 *)(xt + e_hi);
        acc[t] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
                + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
    }
}

/* ==== 解码 GEMV 的两个核(2026-09-23 纯解码, 09-24 扩到验证批; 微基准 gguf-tools/bench/v41_q4k_gemv_bench.cu) ====
 *
 * 为什么换结构: 原核(每 warp 预取 4 块各读各的, 09-24 删除; 微基准里仍有逐字抄本当对照)在 spark 上各形状只到
 * 175~227 GB/s, 329 发/步合计 ~19.6 ms。病在读法:
 * 144 B 的块不对齐 128 B 线, 每个 warp 各读各的几段, 同一时刻在飞的字节少。一个 CTA 负责的 rpb 行在内存里
 * 本来就是**连续一段**(rpb × nblk × 144 B), 所以让全 CTA 先按 16 B 整线把这一段搬进 shared, 再按原分工算。
 * ★逐字节同★: 每个 warp 仍管同一行同一段(kpart), 块按同样升序进 acc, 规约序同 ⇒ 输出与原核逐位相同
 *   (微基准 9 个真实形状全部逐位同; 引擎门 = d0a 成对 + 输出 cmp)。
 * 两个变体, 按"每组行的字节"选(实测, 见微基准表):
 *   stage: 一个 CTA 一组行, 读完就算。组大(wo_a/wo_b/输出头 18~23 KB)时最好: 输出头 227 → 253 GB/s。
 *   pipe : CTA 常驻、循环多组行, cp.async 在算第 k 组时搬第 k+1 组(双缓冲)。组在 4~12 KB 时最好
 *          (共享专家 gate/up 186 → 207、down 177 → 215): 组数多, 省掉的是每组一次 CTA 启停与读算串行。
 *          组大时它两份缓冲把占用率压到 2 CTA/SM, 反而慢(输出头 239) —— 所以不是一律 pipe。 */
#define V41_Q4K_PIPE_MIN 4096u     /* 每组行字节 > 这个且 ≤ MAX 时走 pipe */
#define V41_Q4K_PIPE_MAX 12288u    /* 2 份 ≤ 24 KB ⇒ 每 SM 仍挂得下 4 个 256 线程 CTA */
/* ★NT = 本发几行激活(2026-09-24)★: 纯解码 NT=1; 投机验证批 NT=1+k 行共用同一份搬进 shared 的权重, 每行各乘一遍。
 * 以前验证批落回原核(09-24 真实请求实测: 验 1 行 41.75 ms 对走图一步 35.11、每多一行 14.85 对专家账 9.61),
 * 09-23 的换结构只进了 NT=1。★逐字节同★: 每行的块序/规约序与 NT=1 完全相同(acc[t] 只是多一个下标), 所以
 * 验证批第 t 行 == 纯解码在那个位置算出来的值 —— 投机同轨门(投机 == 纯解码逐字节)就靠这一条。 */
template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_rown_smem(const uint8_t *wr, uint32_t nblk, uint32_t ksplit, uint32_t kpart,
                                                         const float *x, uint32_t x_stride, float *acc) {
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t b = kpart; b < nblk; b += ksplit) {
        const uint8_t *blk = wr + b * V41_Q4K_BYTES;
        v41_q4k_blk_acc_reg<NT>(*(const uint4 *)blk, *(const uint32_t *)(blk + 16u + 4u * lane), x, x_stride, b * V41_Q4K_BLK, acc);
    }
}
/* 规约与写出: 与原核逐式同(warp xor 规约 → red[] 按 k 升序相加 → 可选 bf16 舍入) */
template <uint32_t NT>
__device__ __forceinline__ static void v41_q4k_finishn(float *acc, float *out, uint32_t out_stride, uint32_t r, uint32_t out_dim,
                                                       uint32_t ksplit, uint32_t rloc, uint32_t kpart, int round_out, float *red) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        for (int o = 16; o; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
        if (lane == 0) red[warp * NT + t] = acc[t];
    }
    __syncthreads();
    if (r < out_dim && kpart == 0 && lane == 0) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float sum = 0.f;
            for (uint32_t k = 0; k < ksplit; k++) sum += red[(rloc * ksplit + k) * NT + t];
            out[(uint64_t)t * out_stride + r] = round_out ? v41_bf16r(sum) : sum;
        }
    }
}
template <uint32_t NT>
__global__ static void __launch_bounds__(256, 4) v41_q4k_gemv1_stage_kernel(float *out, const uint8_t *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out,
        uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 v41_q4k_st[];
    __shared__ float red[V41_GEMV_WARPS * NT];
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, rpb = V41_GEMV_WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t nblk = in_dim / V41_Q4K_BLK, r0 = blockIdx.x * rpb, r = r0 + rloc;
    const uint32_t nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb, n16 = nrows * nblk * (V41_Q4K_BYTES / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * V41_Q4K_BYTES);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) v41_q4k_st[i] = __ldcs(src + i);   /* 流式读: 权重每步只读一遍 */
    v41_pdl_wait();   /* PDL: 权重(常量)先搬, 这里才等上一个核的激活(见 cuda_internal.cuh) */
    __syncthreads();
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) v41_q4k_rown_smem<NT>((const uint8_t *)v41_q4k_st + (uint64_t)rloc * nblk * V41_Q4K_BYTES, nblk, ksplit, kpart, x, x_stride, acc);
    v41_q4k_finishn<NT>(acc, out, out_stride, r, out_dim, ksplit, rloc, kpart, round_out, red);
}
template <uint32_t NT>
__global__ static void __launch_bounds__(256, 4) v41_q4k_gemv1_pipe_kernel(float *out, const uint8_t *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t ksplit, uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out,
        uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 v41_q4k_st[];
    __shared__ float red[V41_GEMV_WARPS * NT];
    const uint32_t g = blockIdx.y;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride; w += (uint64_t)g * w_gstride;
    const uint32_t warp = threadIdx.x >> 5, rpb = V41_GEMV_WARPS / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t nblk = in_dim / V41_Q4K_BLK, ngrp = (out_dim + rpb - 1u) / rpb, grp16 = rpb * nblk * (V41_Q4K_BYTES / 16u);
    uint32_t rg = blockIdx.x, buf = 0;
    #define V41_Q4K_ISSUE(RG, BUF) do {                                                                           \
        const uint32_t r0_ = (RG) * rpb, nr_ = out_dim - r0_ < rpb ? out_dim - r0_ : rpb;                          \
        const uint32_t n16_ = nr_ * nblk * (V41_Q4K_BYTES / 16u);                                                  \
        const uint4 *src_ = (const uint4 *)(w + (uint64_t)r0_ * nblk * V41_Q4K_BYTES);                            \
        uint4 *dst_ = v41_q4k_st + (uint64_t)(BUF) * grp16;                                                        \
        for (uint32_t i_ = threadIdx.x; i_ < n16_; i_ += blockDim.x) __pipeline_memcpy_async(dst_ + i_, src_ + i_, 16); \
    } while (0)
    if (rg < ngrp) V41_Q4K_ISSUE(rg, 0u);
    __pipeline_commit();
    v41_pdl_wait();   /* PDL: 第一组权重已在飞, 这里才等上一个核的激活 */
    for (; rg < ngrp; rg += gridDim.x, buf ^= 1u) {
        if (rg + gridDim.x < ngrp) V41_Q4K_ISSUE(rg + gridDim.x, buf ^ 1u);
        __pipeline_commit();          /* 空提交也要: wait_prior(1) 按提交次数数 */
        __pipeline_wait_prior(1);     /* 本组那一半到齐(下一组那一半还在飞) */
        __syncthreads();
        const uint32_t r = rg * rpb + rloc;
        float acc[NT];
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
        if (r < out_dim) v41_q4k_rown_smem<NT>((const uint8_t *)(v41_q4k_st + (uint64_t)buf * grp16) + (uint64_t)rloc * nblk * V41_Q4K_BYTES,
                                           nblk, ksplit, kpart, x, x_stride, acc);
        v41_q4k_finishn<NT>(acc, out, out_stride, r, out_dim, ksplit, rloc, kpart, round_out, red);
        __syncthreads();              /* 这一半下一轮要被覆盖: 全 CTA 读完才许发下一次搬运 */
    }
    #undef V41_Q4K_ISSUE
}

/* q4_K → bf16(预填: 解成稠密再走 cuBLAS, 与 fp4x32 的 v41_fp4x32_to_bf16_kernel 同一套路)。
 * 一个 warp 一块, lane 排布与 GEMV 完全相同 ⇒ 两条路解出的值逐位同。 */
__global__ static void v41_q4k_to_bf16_kernel(__nv_bfloat16 *o, const uint8_t *w, uint64_t nblk) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const uint8_t *blk = w + b * V41_Q4K_BYTES;
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    __nv_bfloat16 *ob = o + b * V41_Q4K_BLK;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        ob[gidx * 64u + q0 + i]       = __float2bfloat16(d * s_lo * (float)(byte & 0xFu) - dmin * m_lo);
        ob[gidx * 64u + 32u + q0 + i] = __float2bfloat16(d * s_hi * (float)(byte >> 4)  - dmin * m_hi);
    }
}

/* 嵌入取行: token id → 该行 f32(词表 × 5120 的 q4_K 张量)。一个 warp 一块。 */
__global__ static void v41_q4k_embed_kernel(float *out, const int32_t *tok, const uint8_t *w,
                                            uint32_t n_vocab, uint32_t dim) {
    const uint32_t t = blockIdx.y, nblk = dim / V41_Q4K_BLK;
    const uint32_t b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const int32_t id = tok[t];
    if (id < 0 || (uint32_t)id >= n_vocab) return;
    const uint8_t *blk = w + ((uint64_t)id * nblk + b) * V41_Q4K_BYTES;
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    const float d = __half2float(__ushort_as_half(hd)), dmin = __half2float(__ushort_as_half(hm));
    float s_lo, m_lo, s_hi, m_hi;
    v41_q4k_sm(blk + 4, (int)(gidx * 2u), &s_lo, &m_lo);
    v41_q4k_sm(blk + 4, (int)(gidx * 2u + 1u), &s_hi, &m_hi);
    const uint32_t qw = *(const uint32_t *)(blk + 16u + 4u * lane);
    float *o = out + (uint64_t)t * dim + b * V41_Q4K_BLK;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        o[gidx * 64u + q0 + i]       = d * s_lo * (float)(byte & 0xFu) - dmin * m_lo;
        o[gidx * 64u + 32u + q0 + i] = d * s_hi * (float)(byte >> 4)  - dmin * m_hi;
    }
}

/* ---- 发射器 ---- */
/* 对齐前提: uint32 读 qs 要求每块 4 字节对齐。块长 144 是 16 的倍数, 所以只要张量起点对齐就都对齐;
 * GGUF 数据区按 32 B 对齐 ⇒ 正常成立。不成立时硬停车(返回 0), 不悄悄走一条慢路。 */
static int v41_q4k_gemv(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                        const float *x, uint32_t x_stride, float *out, uint32_t out_stride, uint32_t n_tok,
                        uint32_t n_groups, uint32_t x_gstride, uint32_t out_gstride, int round_out, const char *what) {
    if ((in_dim % V41_Q4K_BLK) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK || n_groups == 0) return 0;
    const uint64_t wg = out_dim * (in_dim / V41_Q4K_BLK) * V41_Q4K_BYTES, wbytes = wg * n_groups;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    /* 16 B: GEMV 核把块头一条 uint4 读进来(块长 144 是 16 的倍数, 张量起点对齐就处处对齐; GGUF 数据区按 32 B 对齐) */
    if (((uintptr_t)w & 15u) != 0u) { fprintf(stderr, "ds4: %s q4_K 张量起点未 16 字节对齐\n", what); return 0; }
    /* 并行度目标沿用 fp4x32 那支实测出来的 8192(见 cuda_v41_4.inc.cu 的长注释); K 的分段单位是一个
     * 256 元素块, 分不出 ksplit 段就收回来 —— 否则多出来的 warp 一轮都跑不到。 */
    uint32_t ksplit = 1;
    while (ksplit < V41_GEMV_WARPS && out_dim * ksplit * n_groups < 8192u) ksplit <<= 1;
    const uint32_t nb = (uint32_t)(in_dim / V41_Q4K_BLK);
    while (ksplit > 1u && ksplit > nb) ksplit >>= 1;
    /* 纯解码与验证批同一对核(stage / pipe, 选法见两核上方的注释), 只差激活行数 NT */
    const uint32_t rpb1 = V41_GEMV_WARPS / ksplit, ngrp = (uint32_t)((out_dim + rpb1 - 1u) / rpb1);
    const uint32_t grp = rpb1 * nb * V41_Q4K_BYTES;
    uint32_t gx = 0;
    if (grp > V41_Q4K_PIPE_MIN && grp <= V41_Q4K_PIPE_MAX) {
        static int s_sm = 0;
        if (!s_sm && cudaDeviceGetAttribute(&s_sm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return cuda_ok(cudaGetLastError(), what);
        gx = (uint32_t)s_sm * 4u / n_groups;   /* 4 = __launch_bounds__ 钉的每 SM CTA 数 */
        if (gx == 0u) gx = 1u;
        if (gx > ngrp) gx = ngrp;
    }
    /* grp 最大是输出头/wo_b 那种 8 行 × 20 块或 4 行 × 32 块 = 23 KB, 在默认 48 KB 动态 shared 之内 */
    #define V41_Q4K_LAUNCH(NT) do {                                                                                   \
        v41_pdl_register((const void *)v41_q4k_gemv1_stage_kernel<NT>);   /* 两核都在读 x / 写 out 之前 v41_pdl_wait */ \
        v41_pdl_register((const void *)v41_q4k_gemv1_pipe_kernel<NT>);                                              \
        if (gx) v41_q4k_gemv1_pipe_kernel<NT><<<dim3(gx, n_groups), 256, 2u * grp, g_cur_stream>>>(                 \
                    out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, ksplit, wg, x_gstride, out_gstride, round_out, x_stride, out_stride); \
        else v41_q4k_gemv1_stage_kernel<NT><<<dim3(ngrp, n_groups), 256, grp, g_cur_stream>>>(                      \
                    out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, ksplit, wg, x_gstride, out_gstride, round_out, x_stride, out_stride); \
    } while (0)
    switch (n_tok) {
        case 1: V41_Q4K_LAUNCH(1u); break;  case 2: V41_Q4K_LAUNCH(2u); break;
        case 3: V41_Q4K_LAUNCH(3u); break;  case 4: V41_Q4K_LAUNCH(4u); break;
        case 5: V41_Q4K_LAUNCH(5u); break;  case 6: V41_Q4K_LAUNCH(6u); break;
        case 7: V41_Q4K_LAUNCH(7u); break;  default: V41_Q4K_LAUNCH(8u); break;
    }
    #undef V41_Q4K_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

/* ---- 后训练一层内的 bf16 权重缓存(10-03) ----
 * 为什么: 训练的每层反传是"重算本层 → 本层反传", 重算里每个 q4_K 稠密矩阵解一遍 bf16 做前向 GEMM, 紧接着反传的转置乘又把同一批矩阵再解一遍
 * (10-03 逐核表: q4_K→bf16 占 GPU 时间 7.5%, 前向/重算/反传各三分之一)。开着时重算解出的整块矩阵按权重偏移留在一块环形暂存里, 反传直接拿。
 * 只有训练器在"重算 + 反传"那一层开(ds4_gpu_bwd_wcache), 推理与训练前向都关着 —— 关着时行为与原来一字不差。
 * 容量: 一层的稠密权重(wq_a/wq_b/wkv/wo_a/wo_b/共享专家/压缩器/indexer)解成 bf16 实测 8 块 309 MB(10-03 全层训练); 环满了就从头覆盖(先进先出),
 * 只收能整块解的矩阵(出口头那种分块解的不收)。值与原来同一个解码核同一份 ⇒ 反传读到的权重逐位不变。 */
#define V41_WCACHE_BYTES (384ull << 20)
#define V41_WCACHE_SLOTS 32u
static struct { int on; uint8_t *buf; uint64_t cap, head; uint32_t n; uint64_t key[V41_WCACHE_SLOTS], at[V41_WCACHE_SLOTS], bytes[V41_WCACHE_SLOTS]; } g_v41_wc;
int ds4_gpu_bwd_wcache(int mode) {   /* 契约见 ds4_gpu_bwd.h: 1 = 开新一层(清表), 0 = 关 */
    static uint64_t said;
    if (!mode && g_v41_wc.on && g_v41_wc.head > said) {   /* 一层实际占了多少(只在创新高时说一句): 定 V41_WCACHE_BYTES 的依据 */
        said = g_v41_wc.head;
        fprintf(stderr, "ds4: [bwd] 层内 bf16 权重缓存: 一层 %u 块 %.0f MB(池 %.0f MB)\n", g_v41_wc.n, (double)said / 1048576.0, (double)V41_WCACHE_BYTES / 1048576.0);
    }
    g_v41_wc.on = mode ? 1 : 0;
    if (mode) { g_v41_wc.n = 0; g_v41_wc.head = 0; }
    return 1;
}
/* 查: 开着且这块矩阵(权重偏移 off)在表里 ⇒ 它的 bf16 */
static const __nv_bfloat16 *v41_wc_find(uint64_t off) {
    if (!g_v41_wc.on) return NULL;
    for (uint32_t i = 0; i < g_v41_wc.n; i++) if (g_v41_wc.key[i] == off) return (const __nv_bfloat16 *)(g_v41_wc.buf + g_v41_wc.at[i]);
    return NULL;
}
/* 占: 开着时给 off 这块矩阵(elems 个 bf16)要一段地方, 返回 NULL = 不收(关着 / 太大 / 分不出暂存), 调用方照旧解进 g_v41_wbf */
static __nv_bfloat16 *v41_wc_alloc(uint64_t off, uint64_t elems) {
    if (!g_v41_wc.on) return NULL;
    const uint64_t bytes = (elems * 2u + 255u) & ~255ull;
    if (bytes > V41_WCACHE_BYTES / 2u) return NULL;
    if (!g_v41_wc.buf) {
        if (cudaMalloc((void **)&g_v41_wc.buf, V41_WCACHE_BYTES) != cudaSuccess) { (void)cudaGetLastError(); g_v41_wc.on = 0; return NULL; }
        g_v41_wc.cap = V41_WCACHE_BYTES;
    }
    if (g_v41_wc.head + bytes > g_v41_wc.cap) g_v41_wc.head = 0;   /* 绕回: 下面把与新段重叠的旧项清掉 */
    const uint64_t a = g_v41_wc.head, b = a + bytes;
    uint32_t k = 0;
    for (uint32_t i = 0; i < g_v41_wc.n; i++) {
        const uint64_t x = g_v41_wc.at[i], y = x + g_v41_wc.bytes[i];
        if (y <= a || x >= b) { g_v41_wc.key[k] = g_v41_wc.key[i]; g_v41_wc.at[k] = x; g_v41_wc.bytes[k] = g_v41_wc.bytes[i]; k++; }
    }
    g_v41_wc.n = k;
    if (g_v41_wc.n == V41_WCACHE_SLOTS) return NULL;
    g_v41_wc.key[g_v41_wc.n] = off; g_v41_wc.at[g_v41_wc.n] = a; g_v41_wc.bytes[g_v41_wc.n] = bytes; g_v41_wc.n++;
    g_v41_wc.head = b;
    return (__nv_bfloat16 *)(g_v41_wc.buf + a);
}

/* 预填(n_tok > V41_GEMV_MAX_TOK): 权重解成 bf16 暂存, 激活转 bf16, 每组一发 cuBLAS。
 * 与 fp4x32 的 wo_a 预填路同一形态(见 cuda_v41_1.inc.cu 里那段"为什么是 bf16 不是 f16")。
 * ★不走 NVFP4 张量核★: 那条路要把权重摆成 e2m1 nibble, q4_K 的值不在 FP4 格点上, 转过去是二次量化。 */
static int v41_q4k_gemm(const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim,
                        uint64_t out_dim, const float *x, float *out, uint32_t n_tok, uint32_t n_groups,
                        uint32_t x_stride, uint32_t out_stride, int round_out, const char *what) {
    (void)round_out;   /* 舍入在调用方(round_out 时整块一发) */
    const uint64_t bpr = in_dim / V41_Q4K_BLK, nblk_g = out_dim * bpr, nblk = nblk_g * n_groups;
    if (off > model_size || nblk * V41_Q4K_BYTES > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, nblk * V41_Q4K_BYTES, what);
    if (!w) return 0;
    if (((uintptr_t)w & 3u) != 0u) { fprintf(stderr, "ds4: %s q4_K 张量起点未 4 字节对齐\n", what); return 0; }
    /* ★权重暂存按输出维分块, 与 fp4x32 预填路 / 反传转置乘同用 g_v41_wbf、同一个上限 V41_BF16_STAGE_ELEMS★(10-02 夜):
     * 原来整块解进自己的一块暂存 —— 出口头 129280×4096 一解 1.06 GB, 是后训练暂存峰值(2.0 GB)的一半, 而 fp4x32 那条早就分块封顶了(见
     * ds4_gpu_v41_matmul_fp4x32_tensor 的 09-20 实撞)。超上限的只有出口头(分 3 块, 每块的贡献写进输出的不同行, 不累加); 其余矩阵整块一发,
     * 与原来同一发 cuBLAS。分组(wo_a)总量 67 MB, 整块解一次再逐组乘。三处共用一块暂存是安全的: 都在当前流上先后发, n ≤ 8 的侧流分叉不走 GEMM 路。 */
    const uint64_t rows_cap = V41_BF16_STAGE_ELEMS / in_dim;
    const uint64_t tile = n_groups > 1u ? out_dim : (rows_cap < 256u ? 256u : (rows_cap >= out_dim ? out_dim : (rows_cap & ~255ull)));
    const uint64_t stage = n_groups > 1u ? nblk * V41_Q4K_BLK : tile * in_dim;
    /* 层内缓存开着且整块能一次解: 已在表里就跳过解码, 不在就解进缓存(之后反传的转置乘直接拿) */
    const bool whole = n_groups > 1u || tile == out_dim;
    const __nv_bfloat16 *hit = whole ? v41_wc_find(off) : NULL;
    __nv_bfloat16 *wcs = (whole && !hit) ? v41_wc_alloc(off, nblk * V41_Q4K_BLK) : NULL;
    __nv_bfloat16 *wb = wcs ? wcs : (__nv_bfloat16 *)v41_grow(&g_v41_wbf, stage * sizeof(__nv_bfloat16), "v41 q4k w bf16");
    if (!wb) return 0;
    const uint64_t xn = (uint64_t)n_tok * x_stride;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)v41_grow(&g_v41_xbf, xn * sizeof(__nv_bfloat16), "v41 q4k x bf16");
    if (!xb) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(xb, x, xn);
    if (!cuda_ok(cudaGetLastError(), "v41 q4k x→bf16")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    const dim3 dblk(32, 8);
    (void)cublasSetStream(g_cublas, v41_cublas_stream());
    if (n_groups > 1u) {
        if (!hit) {
            v41_q4k_to_bf16_kernel<<<(unsigned)((nblk + 7) / 8), dblk, 0, g_cur_stream>>>(wb, w, nblk);
            if (!cuda_ok(cudaGetLastError(), "v41 q4k→bf16")) return 0;
        }
        const __nv_bfloat16 *wu = hit ? hit : wb;
        for (uint32_t g = 0; g < n_groups; g++) {
            cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim, &alpha,
                                             wu + (uint64_t)g * nblk_g * V41_Q4K_BLK, CUDA_R_16BF, (int)in_dim,
                                             xb + (uint64_t)g * in_dim, CUDA_R_16BF, (int)x_stride, &beta,
                                             out + (uint64_t)g * out_dim, CUDA_R_32F, (int)out_stride,
                                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(st, what)) return 0;
        }
        return 1;
    }
    for (uint64_t r0 = 0; r0 < out_dim; r0 += tile) {
        const uint64_t rows = out_dim - r0 < tile ? out_dim - r0 : tile, tb = rows * bpr;
        if (!hit) {   /* 命中缓存时 tile == out_dim, 只有这一块 */
            v41_q4k_to_bf16_kernel<<<(unsigned)((tb + 7) / 8), dblk, 0, g_cur_stream>>>(wb, w + r0 * bpr * V41_Q4K_BYTES, tb);
            if (!cuda_ok(cudaGetLastError(), "v41 q4k→bf16")) return 0;
        }
        cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)n_tok, (int)in_dim, &alpha,
                                         hit ? hit : wb, CUDA_R_16BF, (int)in_dim, xb, CUDA_R_16BF, (int)x_stride, &beta,
                                         out + r0, CUDA_R_32F, (int)out_stride, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, what)) return 0;
    }
    return 1;
}

/* ---- ds4_gpu_v41.h 契约的 q4_K 三支(命名沿用 v41_* 只为与同一伞头里的既有 API 一致;
 * 按铁律 09-15 新代码不该带版本名, 这是既有欠账, 整族改名时一起处理) ---- */
int ds4_gpu_v41_matmul_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    if (x->bytes < (uint64_t)n_tok * in_dim * 4 || out->bytes < (uint64_t)n_tok * out_dim * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK)
        return v41_q4k_gemv(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr, (uint32_t)in_dim,
                            (float *)out->ptr, (uint32_t)out_dim, n_tok, 1u, 0u, 0u, round_out, "v41 q4k gemv");
    if (!v41_q4k_gemm(model_map, model_size, weight_offset, in_dim, out_dim, (const float *)x->ptr,
                      (float *)out->ptr, n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim, round_out, "v41 q4k gemm")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;
}

int ds4_gpu_v41_grouped_matmul_q4k_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                          uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                          uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (heads->bytes < (uint64_t)n_tok * in_all * 4 || low->bytes < (uint64_t)n_tok * out_all * 4) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK) {
        if (group_dim % V41_Q4K_BLK) return 0;
        return v41_q4k_gemv(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr, (uint32_t)in_all,
                            (float *)low->ptr, (uint32_t)out_all, n_tok, n_groups, (uint32_t)group_dim, (uint32_t)rank, round_out, "v41 wo_a q4k gemv");
    }
    if (!v41_q4k_gemm(model_map, model_size, weight_offset, group_dim, rank, (const float *)heads->ptr,
                      (float *)low->ptr, n_tok, n_groups, (uint32_t)in_all, (uint32_t)out_all, round_out, "v41 wo_a q4k gemm")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;
}

int ds4_gpu_v41_embed_q4k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                 uint64_t model_size, uint64_t weight_offset, uint64_t n_vocab,
                                 uint32_t n_tok, uint64_t dim) {
    if (!out || !tokens || n_tok == 0 || (dim % V41_Q4K_BLK)) return 0;
    const uint64_t nblk = n_vocab * (dim / V41_Q4K_BLK);
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * V41_Q4K_BYTES, "v41 q4k embed");
    if (!w || ((uintptr_t)w & 3u) != 0u) return 0;
    const uint32_t nb = (uint32_t)(dim / V41_Q4K_BLK);
    const dim3 dblk(32, 8), grid((nb + 7u) / 8u, n_tok);
    v41_q4k_embed_kernel<<<grid, dblk, 0, g_cur_stream>>>((float *)out->ptr, (const int32_t *)tokens->ptr, w,
                                                          (uint32_t)n_vocab, (uint32_t)dim);
    return cuda_ok(cudaGetLastError(), "v41 q4k embed");
}

/* 出口头列平方和 Σ_v W[v][d]²(dcap 的出口度量; fp4x32 版与所以然在 cuda_v41_gemv_highprec.inc.cu)。
 * 一 block 一列, 256 线程沿 v 跨步。q4_K 里元素 (v, d) 的位置: 块 v·nblk + d/256, 块内 e = d%256,
 * 组 g = e/64, 组内 o = e%64: o<32 取低 nibble(子块 2g), 否则高 nibble(子块 2g+1), 字节 = qs[g·32 + o%32]
 * —— 与上面 GEMV 的 lane 映射是同一张表, 只是这里按列取。 */
__global__ static void v41_q4k_colnorm_kernel(float *out, const uint8_t *w, uint32_t V, uint32_t D) {
    const uint32_t d = blockIdx.x, nblk = D / V41_Q4K_BLK, bi = d / V41_Q4K_BLK, e = d % V41_Q4K_BLK;
    const uint32_t g = e >> 6, o = e & 63u, hi = o >> 5;
    __shared__ float red[256];
    float s = 0.f;
    for (uint32_t v = threadIdx.x; v < V; v += blockDim.x) {
        const uint8_t *blk = w + ((uint64_t)v * nblk + bi) * V41_Q4K_BYTES;
        const float dd = __half2float(__ushort_as_half((uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8))));
        const float dm = __half2float(__ushort_as_half((uint16_t)((uint16_t)blk[2] | ((uint16_t)blk[3] << 8))));
        float sc, mn;
        v41_q4k_sm(blk + 4, (int)(g * 2u + hi), &sc, &mn);
        const uint8_t by = blk[16u + g * 32u + (o & 31u)];
        const float wv = dd * sc * (float)(hi ? (by >> 4) : (by & 0xFu)) - dm * mn;
        s += wv * wv;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (uint32_t k = blockDim.x >> 1; k; k >>= 1) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[d] = red[0];
}
int ds4_gpu_v41_head_colnorm_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                        uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd) {
    if (!out || (n_embd % V41_Q4K_BLK) || out->bytes < (uint64_t)n_embd * 4) return 0;
    const uint64_t nblk = (uint64_t)n_vocab * (n_embd / V41_Q4K_BLK);
    if (weight_offset > model_size || nblk * V41_Q4K_BYTES > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * V41_Q4K_BYTES, "v41 q4k head colnorm");
    if (!w) return 0;
    v41_q4k_colnorm_kernel<<<n_embd, 256, 0, g_cur_stream>>>((float *)out->ptr, w, n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "v41 q4k head colnorm");
}

/* ★判负存档(2026-09-23): "在 DRAM 闲着的小核段之前把下一段 GEMV 的权重 cp.async.bulk.prefetch 进 L2"★
 * 依据: nsys 时间线上每层有 ~120 µs 的串行小核段 DRAM 闲着(hc_post→hc_mix→hc_fused、注意力核→merge→win→rope)。
 * 做法: 三处(注意力后预取路由门+共享专家 gate/up ~17 MB; 层尾预取下一层 q_a+kv+q_b 前 12 MB; 注意力核前预取 wo_a 前 12 MB),
 * TMA 批量预取, 预取核末尾 v41_pdl_wait 保依赖链。纯墙钟成对: 33.7 → 35.4 ms/步(慢 1.7 ms), 输出逐字节同。
 * 读法: 每步多 271 个节点, 而 GEMV/专家核本身是流式读(evict-first)在冲 L2, 预取进去的行多半等不到被用就被挤掉 ⇒ 同一批字节读两遍。
 * 想再走这条路, 得先让消费方的读法改成 evict_last 命中/保留, 并用 ncu 的 lts 命中率证明预取行真被用上了。 */
