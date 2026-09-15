/* cuda_v41_2.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 批前向原语 ②(2026-09-12 战役 P2)。
 * RoPE / 激活量化(fp8·fp4 就地) / 压缩器池化 / indexer 打分·候选块·topk / 稀疏注意力。
 * 每个核都对着官方 model.py / kernel.py 的那一段写, 注释里标的是对应的官方函数。 */

/* ---- RoPE(官方 precompute_freqs_cis + apply_rotary_emb): 相邻两元素 = 一个复数 ---- */
__device__ __forceinline__ static float v41_rope_freq(uint32_t i, uint32_t dim, float theta, uint32_t osl,
                                                      float factor, float beta_fast, float beta_slow) {
    float f = 1.0f / powf(theta, (float)(2u * i) / (float)dim);
    if (osl > 0u) {   /* YaRN ramp: corrected_dim(rot) = dim·ln(osl/(rot·2π)) / (2 ln θ) */
        const float lt = 2.0f * logf(theta);
        float low = floorf((float)dim * logf((float)osl / (beta_fast * 2.0f * (float)M_PI)) / lt);
        float high = ceilf((float)dim * logf((float)osl / (beta_slow * 2.0f * (float)M_PI)) / lt);
        low = fmaxf(low, 0.0f); high = fminf(high, (float)(dim - 1u));
        float ramp = ((float)i - low) / fmaxf(high - low, 1e-3f);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        f = f / factor * (1.0f - smooth) + f * smooth;
    }
    return f;
}
__global__ static void v41_rope_kernel(float *x, const int32_t *pos, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                                       float theta, uint32_t osl, float factor, float bf, float bs, int inverse) {
    const uint32_t t = blockIdx.y, h = blockIdx.x;
    const uint32_t i = threadIdx.x;                 /* 复数下标 i < n_rot/2 */
    if (i >= n_rot / 2u) return;
    float *xr = x + ((uint64_t)t * n_head + h) * head_dim + (head_dim - n_rot) + 2u * i;
    const float ang = (float)pos[t] * v41_rope_freq(i, n_rot, theta, osl, factor, bf, bs);
    float c = cosf(ang), s = sinf(ang);
    if (inverse) s = -s;
    const float a = xr[0], b = xr[1];
    xr[0] = v41_bf16r(a * c - b * s);   /* 官方: x.float() 旋转后 copy_ 回 bf16 张量 */
    xr[1] = v41_bf16r(a * s + b * c);
}
int ds4_gpu_v41_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head,
                            uint32_t head_dim, uint32_t n_rot, float theta, uint32_t original_seq_len,
                            float factor, float beta_fast, float beta_slow, bool inverse) {
    if (!x || !pos || (n_rot & 1u) || n_rot > 128u) return 0;
    v41_rope_kernel<<<dim3(n_head, n_tok), 64, 0, g_cur_stream>>>((float *)x->ptr, (const int32_t *)pos->ptr, n_head, head_dim, n_rot,
                                                                    theta, original_seq_len, factor, beta_fast, beta_slow, inverse ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "v41 rope");
}

/* ---- 激活量化就地(官方 act_quant_kernel / fp4_quant_kernel, inplace 分支) ----
 * fast_round_scale(amax, inv) = 2^ceil(log2(amax·inv)); 官方的 fast_log2_ceil 是位运算版 ceil(log2)。
 * 一 warp 一个 block(≤32 元素), 一线程一元素。 */
__device__ __forceinline__ static float v41_pow2_ceil_log2(float v) {
    int e; const float m = frexpf(v, &e);      /* v = m·2^e, m∈[0.5,1) ⇒ log2 v = e + log2 m ∈ (e-1, e] */
    return ldexpf(1.0f, (m == 0.5f) ? e - 1 : e);
}
__global__ static void v41_act_quant_kernel(float *x, uint32_t dim, uint32_t block, int mode) {
    /* mode 0: fp8 e4m3 值 + ue8m0 scale(max 448); 1: fp4 + ue8m0(max 6); 2: fp4 + e4m3 scale(max 6) */
    const uint32_t row = blockIdx.y, b = blockIdx.x;
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
    v41_act_quant_kernel<<<dim3(dim / block, n_rows), 32, 0, g_cur_stream>>>((float *)x->ptr, dim, block, 0);
    return cuda_ok(cudaGetLastError(), "v41 act quant fp8");
}
int ds4_gpu_v41_act_quant_fp4_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, bool e4m3_scale) {
    if (!x || block == 0 || block > 32u || (dim % block)) return 0;
    v41_act_quant_kernel<<<dim3(dim / block, n_rows), 32, 0, g_cur_stream>>>((float *)x->ptr, dim, block, e4m3_scale ? 2 : 1);
    return cuda_ok(cudaGetLastError(), "v41 act quant fp4");
}

/* ---- 压缩器池化(官方 Compressor.forward ratio>1, start_pos=0): 组内逐维 softmax 加权和 → bf16 ---- */
__global__ static void v41_compress_pool_kernel(float *out, const float *kv, const float *sc, uint32_t ratio, uint32_t dim) {
    const uint32_t g = blockIdx.x;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float mx = -INFINITY;
        for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, sc[((uint64_t)g * ratio + t) * dim + d]);
        float den = 0.f, acc = 0.f;
        for (uint32_t t = 0; t < ratio; t++) {
            const float e = expf(sc[((uint64_t)g * ratio + t) * dim + d] - mx);
            den += e; acc += e * kv[((uint64_t)g * ratio + t) * dim + d];
        }
        out[(uint64_t)g * dim + d] = v41_bf16r(acc / den);
    }
}
int ds4_gpu_v41_compress_pool_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score,
                                     uint32_t n_tok, uint32_t ratio, uint32_t dim) {
    if (!out || !kv || !score || ratio == 0) return 0;
    const uint32_t ng = n_tok / ratio;
    if (ng == 0) return 1;
    v41_compress_pool_kernel<<<ng, 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)kv->ptr, (const float *)score->ptr, ratio, dim);
    return cuda_ok(cudaGetLastError(), "v41 compress pool");
}

/* ---- indexer 打分(官方 Indexer.forward): score[i][g] = bf16(Σ_h bf16(relu(bf16(q_h·k_g))·w[i][h])) ----
 * 一 block 一个 query, 一 warp 一个 g; lane 分 dk/32 维, 逐头 warp 归约。 */
__global__ static void v41_indexer_score_kernel(float *score, const float *q, const float *k, const float *w, const uint8_t *cand,
                                                uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    const uint32_t vis = (pos0 + i + 1u) / ratio;   /* 可见组数 compress_lens(按绝对位置) */
    const uint32_t per = dk / 32u;                 /* dk=128 ⇒ 4 维/lane */
    for (uint32_t g = warp; g < ng; g += nwarp) {
        float out;
        if (g >= vis || (cand && !cand[(uint64_t)i * ng + g])) out = -INFINITY;
        else {
            const float *kg = k + (uint64_t)g * dk;
            float acc = 0.f;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qh = q + ((uint64_t)i * n_head + h) * dk;
                float d = 0.f;
                for (uint32_t e = 0; e < per; e++) d += qh[lane * per + e] * kg[lane * per + e];
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                d = v41_bf16r(d);                    /* einsum 出 bf16 */
                d = fmaxf(d, 0.0f);                  /* relu_ */
                acc += v41_bf16r(d * w[(uint64_t)i * n_head + h]);   /* × weights(bf16) 再求和 */
            }
            out = v41_bf16r(acc);
        }
        if (lane == 0) score[(uint64_t)i * ng + g] = out;
    }
}
int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                                     const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_mask,
                                     uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio) {
    if (!score || !q || !k || !weights || (dk % 32u) || ratio == 0) return 0;
    if (ng == 0) return 1;
    v41_indexer_score_kernel<<<n_tok, 256, 0, g_cur_stream>>>((float *)score->ptr, (const float *)q->ptr, (const float *)k->ptr,
        (const float *)weights->ptr, cand_mask ? (const uint8_t *)cand_mask->ptr : NULL, pos0, ng, n_head, dk, ratio);
    return cuda_ok(cudaGetLastError(), "v41 indexer score");
}

/* ---- 候选块(官方 select_candidate_blocks): 一 block 一 query, 线程 0 串行选块(块数 ≤ ng/bs 小) ---- */
__global__ static void v41_candidate_kernel(uint8_t *mask, const float *score, uint32_t pos0, uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t bs) {
    const uint32_t i = blockIdx.x;
    const uint32_t nb = (ng + bs - 1u) / bs;
    extern __shared__ float bsc[];               /* [nb] 块分 + [nb] 选中标记 */
    uint8_t *sel = (uint8_t *)(bsc + nb);
    const uint32_t vis = (pos0 + i + 1u) / ratio;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -INFINITY;
        for (uint32_t j = b * bs; j < (b + 1u) * bs && j < ng; j++) m = fmaxf(m, score[(uint64_t)i * ng + j]);
        if (vis > 0u && b == (vis - 1u) / bs) m = INFINITY;   /* 含最新位置的块钉住 */
        bsc[b] = m; sel[b] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t kk = topk_blocks < nb ? topk_blocks : nb;
        for (uint32_t k = 0; k < kk; k++) {
            int best = -1; float bv = -INFINITY;
            for (uint32_t b = 0; b < nb; b++) if (!sel[b] && bsc[b] > bv) { bv = bsc[b]; best = (int)b; }
            if (best < 0) break;                  /* 剩下全 -inf: 不选 */
            sel[best] = 1;
        }
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < ng; j += blockDim.x) mask[(uint64_t)i * ng + j] = sel[j / bs];
}
int ds4_gpu_v41_candidate_blocks_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t block_size) {
    if (!mask || !score || block_size == 0 || ratio == 0) return 0;
    if (ng == 0) return 1;
    const uint32_t nb = (ng + block_size - 1u) / block_size;
    const size_t shm = (size_t)nb * 4u + nb;
    if (shm > 48u * 1024u) { fprintf(stderr, "ds4: [v41] candidate blocks %u 超 shared\n", nb); return 0; }
    v41_candidate_kernel<<<n_tok, 256, shm, g_cur_stream>>>((uint8_t *)mask->ptr, (const float *)score->ptr, pos0, ng, ratio, topk_blocks, block_size);
    return cuda_ok(cudaGetLastError(), "v41 candidate blocks");
}

/* ---- topk(官方: topk 后按位置升序; 不可达 → -1) ----
 * ★2026-09-14 重写: 原版是"线程 0 串行 O(topk·ng)"(当时留言"速度 P4 再说")。
 * 解码时 ng = 已有上下文长度、topk = 2048 ⇒ 3300 上下文就是 676 万次串行比较, 实测 **44 ms 一次**;
 * nsys 长上下文解码段: 这一个核吃掉 62% 的 GPU 时间, 是"上下文一长解码就塌"(5 token 上下文 8.96 t/s,
 * 3308 token 只剩 1.63 t/s)的主因。
 * 新版分两步, 语义与原版逐位等价:
 *   ①并行 radix select 求第 k 大的阈值 —— 把 float 单调映射成可按无符号比较的 32 位键, 高位到低位
 *     每轮 8 位、256 线程协作数一次直方图, 4 轮定出阈值。每线程只扫 ng/256 个元素。
 *   ②线程 0 升序扫一遍 ng 写出(O(ng), 不是 O(topk·ng))。写出必须串行才能保证"并列取小下标"与官方一致。
 * 等价性要点: 原版用 `s[g] > bv`(严格大于, bv 初值 −inf), 所以 ①并列时先出现的小下标胜出
 * ②−inf 与 NaN 永远选不中 —— 两条都在下面按原样复刻(键 0 = 不可达, 写出时跳过)。 */
__device__ __forceinline__ static uint32_t v41_topk_key(float f) {
    if (!(f > -INFINITY)) return 0u;              /* −inf 与 NaN: 原版永不选中, 归 0 = 最小且被写出段跳过 */
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ static void v41_topk_kernel(int32_t *idx, const float *score, uint32_t ng, uint32_t topk) {
    const uint32_t i = blockIdx.x;
    const float *s = score + (uint64_t)i * ng;
    __shared__ uint32_t hist[256], sh_bucket, sh_k, sh_nvalid;
    if (threadIdx.x == 0) sh_nvalid = 0;
    __syncthreads();
    uint32_t cnt = 0;
    for (uint32_t g = threadIdx.x; g < ng; g += blockDim.x) cnt += (v41_topk_key(s[g]) != 0u);
    atomicAdd(&sh_nvalid, cnt);
    __syncthreads();
    const uint32_t nval = sh_nvalid;
    uint32_t prefix = 0, kk = topk < nval ? topk : nval;
    if (kk > 0) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (uint32_t b = threadIdx.x; b < 256u; b += blockDim.x) hist[b] = 0;
            __syncthreads();
            for (uint32_t g = threadIdx.x; g < ng; g += blockDim.x) {
                const uint32_t key = v41_topk_key(s[g]);
                /* shift=24 那轮没有"更高位"可比(右移 32 是未定义行为), 全部计入 */
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {               /* 从高桶往低累加, 找住第 kk 大的那个桶 */
                uint32_t acc = 0; int b = 255;
                for (; b > 0; b--) { if (acc + hist[b] >= kk) break; acc += hist[b]; }
                sh_bucket = (uint32_t)b; sh_k = kk - acc;
            }
            __syncthreads();
            prefix |= sh_bucket << shift;
            kk = sh_k;                            /* 落到这一桶内还要取几个 */
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {                       /* prefix = 第 k 大的键; kk = 与它相等的还要取几个 */
        uint32_t w = 0, eq = kk;
        for (uint32_t g = 0; g < ng && w < topk; g++) {
            const uint32_t key = v41_topk_key(s[g]);
            if (key == 0u) continue;
            if (key > prefix) idx[(uint64_t)i * topk + w++] = (int32_t)g;
            else if (key == prefix && eq > 0) { eq--; idx[(uint64_t)i * topk + w++] = (int32_t)g; }
        }
        for (; w < topk; w++) idx[(uint64_t)i * topk + w] = -1;
    }
}
int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng, uint32_t topk) {
    if (!idx || !score || topk == 0) return 0;
    if (ng == 0) return 1;
    /* shared 只剩固定的 256 个桶(1 KB), 不再随 ng 走 ⇒ 原来那条 "ng > 48K 就拒" 的闸跟着作废 */
    v41_topk_kernel<<<n_tok, 256, 0, g_cur_stream>>>((int32_t *)idx->ptr, (const float *)score->ptr, ng, topk);
    return cuda_ok(cudaGetLastError(), "v41 topk");
}

/* ---- 稀疏注意力(官方 sparse_attn_kernel 语义): 一 block 一 (query, 8 头)(128 线程 = 4 warp, 每 warp 2 头), grid (n, 8),
 * 键序 = 窗口行(升序) 后接 topk 压缩行; 在线 softmax 逐键; p 先舍 bf16 再乘 v(官方 acc_s_cast);
 * 分母用未舍的 exp 和; sink 只进分母。q/k 值已是 bf16 格点, 点积 f32。
 * 第一版一 block 包 64 头(1024 线程): 解码 n=1 时整层只有 1 个 block 在一个 SM 上逐键 syncthreads, 0.5 ms/层 = 20 ms/token;
 * 拆成 8 个头组 block 后键行多读 8 次(L2 命中), 换 8 个 SM 并行。 ---- */
#define V41_ATTN_HEADS_PER_BLOCK 8u
__global__ static void v41_sparse_attn_kernel(float *o, const float *q, const float *kvw, const float *kvc, const int32_t *idx,
                                              const float *sink, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                              uint32_t n_head, uint32_t hd, float scale, int use_sh) {
    const uint32_t i = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t per = hd / 32u;                 /* 512/32 = 16 维/lane */
    /* ★2026-09-14: 去掉 "整 block 把 krow 搬进 shared" 那一层★。
     * 原版每个 key 都要两次 __syncthreads(搬前搬后), 而解码时 nkeys = 窗口 + topk ≈ 6000 ⇒
     * 一层就是一万二千次 block 同步, 实测这个核占解码 GPU 时间的 46%(2.49 ms/层)。
     * 每个 lane 本来就只用 krow 里自己那 16 个连续维(warp 内 32 lane 正好盖满一整行 2 KB, 是合并访问),
     * 直接从 global 读即可: 同 block 的几个 warp 读同一行, L1 命中, 不需要手动中转, 更不需要同步。
     * 读进寄存器 kreg 后两个头复用 ⇒ 读取次数也减半。累加顺序与原版逐位相同(等价性靠这一条)。
     * ★只在解码(n_tok 小)走这条★: prefill 时 block 多、同一行被很多 query 同时要, shared 中转是划算的,
     * 去掉它实测 prefill 46.7 → 33.9 t/s。所以按 use_sh 分两路, 两路都只是把同样的 16 个数填进 kreg。 */
    __shared__ float ks[512];
    float qa[2][16], acc[2][16], mx[2], sum[2], kreg[16];
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        for (uint32_t e = 0; e < per; e++) { qa[hh][e] = q[((uint64_t)i * n_head + h) * hd + lane * per + e]; acc[hh][e] = 0.f; }
        mx[hh] = -1e30f; sum[hh] = 0.f;
    }
    const uint32_t p = pos0 + i;                   /* 绝对位置 */
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u;
    const uint32_t nkeys = nwin + topk;
    for (uint32_t kk = 0; kk < nkeys; kk++) {
        const float *krow;
        if (kk < nwin) krow = kvw + (uint64_t)((int64_t)lo + kk - ((int64_t)pos0 - (int64_t)window)) * hd;   /* 缓冲行 = 位置 - (pos0-window) */
        else { const int32_t g = idx[(uint64_t)i * topk + (kk - nwin)]; if (g < 0 || (uint32_t)g >= ng) continue; krow = kvc + (uint64_t)g * hd; }
        if (use_sh) {
            __syncthreads();
            for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) ks[d] = krow[d];
            __syncthreads();
            for (uint32_t e = 0; e < per; e++) kreg[e] = ks[lane * per + e];
        } else {
            for (uint32_t e = 0; e < per; e++) kreg[e] = krow[lane * per + e];
        }
        for (int hh = 0; hh < 2; hh++) {
            float d = 0.f;
            for (uint32_t e = 0; e < per; e++) d += qa[hh][e] * kreg[e];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float s = d * scale;
            const float nm = fmaxf(mx[hh], s);
            const float rs = expf(mx[hh] - nm);
            const float p = expf(s - nm);
            sum[hh] = sum[hh] * rs + p;
            const float pb = v41_bf16r(p);
            for (uint32_t e = 0; e < per; e++) acc[hh][e] = acc[hh][e] * rs + pb * kreg[e];
            mx[hh] = nm;
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const uint32_t h = blockIdx.y * V41_ATTN_HEADS_PER_BLOCK + warp * 2u + hh;
        const float den = sum[hh] + expf(sink[h] - mx[hh]);
        for (uint32_t e = 0; e < per; e++) o[((uint64_t)i * n_head + h) * hd + lane * per + e] = v41_bf16r(acc[hh][e] / den);
    }
}
int ds4_gpu_v41_sparse_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win,
                                   const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx,
                                   const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                   uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                   uint32_t n_head, uint32_t head_dim, float scale) {
    if (!o || !q || !kv_win || n_head != 64u || head_dim != 512u) { fprintf(stderr, "ds4: [v41] sparse attn 只实现 64 头×512\n"); return 0; }
    if (kv_win->bytes < (uint64_t)(window + n_tok) * head_dim * 4) { fprintf(stderr, "ds4: [v41] 窗口缓冲不够 %u+%u 行\n", window, n_tok); return 0; }
    const float *sink = (const float *)cuda_model_range_ptr(model_map, sink_offset, (uint64_t)n_head * 4, "v41 sink");
    if (!sink) return 0;
    v41_sparse_attn_kernel<<<dim3(n_tok, n_head / V41_ATTN_HEADS_PER_BLOCK), V41_ATTN_HEADS_PER_BLOCK * 16u, 0, g_cur_stream>>>(
        (float *)o->ptr, (const float *)q->ptr, (const float *)kv_win->ptr,
        kv_comp ? (const float *)kv_comp->ptr : NULL, idx ? (const int32_t *)idx->ptr : NULL, sink, pos0, window, ng, (kv_comp && idx) ? topk : 0u, n_head, head_dim, scale, n_tok > 1u);
    return cuda_ok(cudaGetLastError(), "v41 sparse attn");
}
