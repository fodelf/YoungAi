/* cuda_v41_sample.inc.cu — 设备采样核(2026-09-28): 温度 / top-k / top-p / min-p 全在 GPU 上做, 一块一行, 每行结果 16 B
 * 落设备槽(与 argmax 同一个槽位, 零拷贝读回), 投机的拒绝采样也在这一发里出。
 *
 * 为什么要它: 09-28 采样默认改成模型卡配方(温 1.0)之后, 请求不带 temperature 就走采样路 —— 而采样路以前是
 * "每步把 129280 个 logits 读回主机 + 主机两遍 expf + 投机关掉", 产品默认路(聊天前端大多不发 temperature)一下慢了一档
 * (fable5 09-28 尺)。这里把采样做进解码整步 graph 的末尾(替掉 argmax 那一发), 主机只读回 16 B; 投机照走。
 *
 * 抽样 = Gumbel-max: argmax_i (l_i/T + g_i), g_i = −ln(−ln u_i), u_i 独立均匀 ⇒ 抽到 i 的概率恰是 softmax(l/T)_i(精确等价, 不是近似);
 * 限制在保留集 K 上就是截断后再归一的分布。u_i 由 (seed, 位置, i, 盐) 哈希出(splitmix64 终混), 不依赖线程编排 ⇒ 同 seed 同输入必同结果,
 * 图路与直发路同一个核 ⇒ 同一条请求两种发法出同一串 token。
 * 保留集 K = {l_i ≥ L_cut}, L_cut = max(min_p 门, top_k 门, top_p 门):
 *   min_p: l ≥ M + T·ln(min_p)(相对最大概率);  top_k: 第 k 大的 logit(基数选择: 4 轮 8 位直方图, 整数计数 ⇒ 确定性);
 *   top_p: 从大到小累计概率首次 ≥ top_p 的那一位(同一套基数选择, 概率用 2^40 定点整数累加 ⇒ 确定性, 浮点原子加的次序噪声翻不了门槛)。
 *   语义与主机 ds4_sample_logits 同: top_k 先取, top_p 相对"取后集合"的总质量, min_p 相对最大概率; 边界并列一律保留
 *   (主机按 qsort 的次序砍并列, 那个次序本来就是任意的)。
 * 投机(拒绝采样, Leviathan 2023 / Chen 2023): 第 i 行的草稿 d = tok[i+1](验证批里下一行的输入)。
 *   草稿分布 q 为点质量(qlogits == NULL): 接受概率 = p(d), 拒绝从"p 去掉 d 再归一"的残差里抽;
 *   草稿按分布抽(qlogits = 草稿塔第 i 行 logits, 2026-09-29): 接受概率 = min(1, p(d)/q(d)), 拒绝从 max(0, p − q) 归一后抽 ——
 *   两种合起来吐出 token 的边缘分布都恰是 p; 后者接受率上限 Σmin(p,q), 分布平时远高于点质量的 p(argmax q)。
 *   接受硬币与 Gumbel 噪声用不同盐, 草稿抽样(stream 1)与验证(stream 0)用不同流 ⇒ 全部独立。残差为空(p == q 或 K ⊆ {d})时直接接受。
 *   末行没有草稿, 只出全分布样本。
 * 出错会怎样: 全行非有限 → 四个字全 0(与主机 sample_argmax 的兜底同: 下标 0); 温度 ≤ 0 不该进这个核(调用方走 argmax 核, 那是贪心的门)。 */

#define V41_SAMPLE_THREADS 1024u
#define V41_SAMPLE_BINS 256u
/* 2^40: 相对最大概率的定点分辨率。词表每项 ≤ 1 ⇒ 总和 < 2^58 不溢出 u64; 相对概率 < 2^-40 的项计 0 质量,
 * 它们决定不了 top_p 门槛(主机 float 累加同样吸收不了这么小的项)。 */
#define V41_SAMPLE_FIX 1099511627776.0

__device__ __forceinline__ static uint64_t v41_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31; return x;
}
/* (0,1) 开区间: 24 位 + 半格偏移 ⇒ 不取 0 也不取 1(ln 0 = −∞ 会让某一项恒胜) */
__device__ __forceinline__ static float v41_u01(uint64_t seed, int32_t pos, uint32_t i, uint32_t salt) {
    const uint64_t x = v41_mix64(seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(uint32_t)pos + 0xD1B54A32D192ED03ull * (uint64_t)i +
                                          0x8CB92BA72F3D8DD7ull * (uint64_t)(salt + 1u)));
    return ((float)(uint32_t)(x >> 40) + 0.5f) * (1.0f / 16777216.0f);
}
/* float ↔ 单调 u32 键(键升序 = float 升序), 只对有限值 */
__device__ __forceinline__ static uint32_t v41_fkey(float f) { const uint32_t u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

/* 块内 argmax(并列取小下标; 下标 -1 = 空, 永远输) */
__device__ static void v41_blk_argmax(float *sf, int32_t *si, float v, int32_t i, float *ov, int32_t *oi) {
    sf[threadIdx.x] = v; si[threadIdx.x] = i; __syncthreads();
    for (uint32_t k = blockDim.x / 2u; k > 0u; k >>= 1) {
        if (threadIdx.x < k) {
            const float bv = sf[threadIdx.x + k]; const int32_t bi = si[threadIdx.x + k];
            const float av = sf[threadIdx.x]; const int32_t ai = si[threadIdx.x];
            if (bi >= 0 && (ai < 0 || bv > av || (bv == av && bi < ai))) { sf[threadIdx.x] = bv; si[threadIdx.x] = bi; }
        }
        __syncthreads();
    }
    *ov = sf[0]; *oi = si[0]; __syncthreads();
}
/* 块内浮点求和: 固定树形次序 ⇒ 确定性(不用原子加) */
__device__ static float v41_blk_sumf(float *sf, float v) {
    sf[threadIdx.x] = v; __syncthreads();
    for (uint32_t k = blockDim.x / 2u; k > 0u; k >>= 1) { if (threadIdx.x < k) sf[threadIdx.x] += sf[threadIdx.x + k]; __syncthreads(); }
    const float s = sf[0]; __syncthreads(); return s;
}

/* 基数选择: 在 {有限 且 键 ≥ floor_key} 里, 按键从大到小累计 weight, 首次 ≥ target 的那一项的键。
 * mass=0: weight = 1(top_k, target = k);  mass=1: weight = 定点概率(top_p, target = ⌈top_p·Z⌉)。
 * 4 轮 × 256 桶(shared 整数原子加, 次序无关) → 从高桶往低桶扫到含门槛的桶 → 下一轮只看这个桶。返回门槛项的 32 位键
 * (≥ 它的全保留, 并列一起保留)。target 超过总量(只会是 top_k > 有限项数)时返回 0 = 全保留。 */
__device__ static uint32_t v41_radix_select(const float *l, uint32_t V, uint32_t floor_key, int mass, float M, float inv_T,
                                            unsigned long long target, unsigned long long *hist, uint32_t *s_prefix, unsigned long long *s_target) {
    uint32_t prefix = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const uint32_t mask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint32_t b = threadIdx.x; b < V41_SAMPLE_BINS; b += blockDim.x) hist[b] = 0ull;
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = l[i];
            if (!isfinite(v)) continue;
            const uint32_t k = v41_fkey(v);
            if (k < floor_key || (k & mask) != prefix) continue;
            unsigned long long w = 1ull;
            if (mass) w = (unsigned long long)((double)expf((v - M) * inv_T) * V41_SAMPLE_FIX + 0.5);
            atomicAdd(&hist[(k >> shift) & 0xFFu], w);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long cum = 0ull; uint32_t sel = 0u; unsigned long long rest = target;
            for (int b = (int)V41_SAMPLE_BINS - 1; b >= 0; b--) {
                cum += hist[b];
                if (cum >= target) { sel = (uint32_t)b; rest = target - (cum - hist[b]); break; }
            }
            *s_prefix = prefix | (sel << shift); *s_target = rest;
        }
        __syncthreads();
        prefix = *s_prefix; target = *s_target;
        __syncthreads();
        if (target == 0ull) return 0u;   /* 扫完所有桶都没到 target ⇒ 全保留; 到了的话余量必 ≥ 1 */
    }
    return prefix;
}

/* 一行的三道门: 返回键下界 cut(0 = 全保留), *M = 有限最大值(*Mi < 0 = 全行非有限, 调用方兜底) */
__device__ static uint32_t v41_sample_gate(const float *l, uint32_t V, float inv_T, float min_p, uint32_t top_k, float top_p,
                                           float *sf, int32_t *si, unsigned long long *hist, uint32_t *s_prefix, unsigned long long *s_u64,
                                           float *M, int32_t *Mi) {
    float best = -INFINITY; int32_t bi = -1;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) { const float v = l[i]; if (isfinite(v) && v > best) { best = v; bi = (int32_t)i; } }
    v41_blk_argmax(sf, si, best, bi, M, Mi);
    if (*Mi < 0) return 0u;
    uint32_t cut = 0u;
    if (min_p > 0.0f) cut = v41_fkey(*M + logf(min_p) / inv_T);
    uint32_t kfloor = 0u;
    if (top_k > 0u && top_k < V) {
        kfloor = v41_radix_select(l, V, 0u, 0, *M, inv_T, (unsigned long long)top_k, hist, s_prefix, s_u64);
        if (kfloor > cut) cut = kfloor;
    }
    if (top_p < 1.0f) {
        if (threadIdx.x == 0) *s_u64 = 0ull;
        __syncthreads();
        unsigned long long z = 0ull;
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = l[i];
            if (!isfinite(v) || v41_fkey(v) < kfloor) continue;
            z += (unsigned long long)((double)expf((v - *M) * inv_T) * V41_SAMPLE_FIX + 0.5);
        }
        atomicAdd(s_u64, z);
        __syncthreads();
        const unsigned long long Z = *s_u64;
        __syncthreads();
        unsigned long long target = (unsigned long long)ceil((double)top_p * (double)Z);
        if (target == 0ull) target = 1ull;
        const uint32_t pfloor = v41_radix_select(l, V, kfloor, 1, *M, inv_T, target, hist, s_prefix, s_u64);
        if (pfloor > cut) cut = pfloor;
    }
    return cut;
}

__global__ static void v41_sample_kernel(int32_t *out, const float *logits, const float *qlogits, uint32_t V, const int32_t *pos, const int32_t *tok,
                                         uint32_t row0, uint32_t n_rows, float inv_T, float min_p, uint32_t top_k, float top_p,
                                         uint64_t seed, uint32_t stream) {
    v41_pdl_wait();   /* 上游 = 出口头写 logits; 不经 PDL 发射时立即返回 */
    __shared__ float sf[V41_SAMPLE_THREADS]; __shared__ int32_t si[V41_SAMPLE_THREADS];
    __shared__ unsigned long long hist[V41_SAMPLE_BINS];
    __shared__ uint32_t s_prefix; __shared__ unsigned long long s_u64;
    const uint32_t r = row0 + blockIdx.x;
    const float *l = logits + (size_t)r * V;
    const int32_t p = pos[r];
    const int32_t d = blockIdx.x + 1u < n_rows ? tok[r + 1u] : -1;
    const float *lq = (d >= 0 && qlogits) ? qlogits + (size_t)blockIdx.x * V : NULL;
    const uint32_t salt_g = 2u * stream, salt_a = 2u * stream + 1u;   /* Gumbel 噪声 / 接受硬币, 按流分开 */
    int32_t *o = out + 4u * blockIdx.x;
    float M; int32_t Mi;
    const uint32_t cut = v41_sample_gate(l, V, inv_T, min_p, top_k, top_p, sf, si, hist, &s_prefix, &s_u64, &M, &Mi);
    if (Mi < 0) { if (threadIdx.x == 0) { o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; } return; }
    /* 草稿分布 q 的门与两个归一化常数(点质量草稿不需要: 残差 = p 去掉 d, 比值不用归一) */
    float Mq = 0.0f; int32_t Mqi = -1; uint32_t cutq = 0u; float Zp = 1.0f, Zq = 1.0f;
    if (lq) {
        cutq = v41_sample_gate(lq, V, inv_T, min_p, top_k, top_p, sf, si, hist, &s_prefix, &s_u64, &Mq, &Mqi);
        if (Mqi < 0) lq = NULL;   /* q 行全非有限: 当点质量草稿 */
    }
    if (lq) {
        float zp = 0.0f, zq = 0.0f;
        for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = l[i], vq = lq[i];
            if (isfinite(v) && v41_fkey(v) >= cut) zp += expf((v - M) * inv_T);
            if (isfinite(vq) && v41_fkey(vq) >= cutq) zq += expf((vq - Mq) * inv_T);
        }
        Zp = v41_blk_sumf(sf, zp); Zq = v41_blk_sumf(sf, zq);
    }
    /* 保留集上的 Gumbel-max: 全分布样本; 残差样本(q: max(0, p−q) 上 / 点质量: 去掉 d); 顺手算 Z_K、p(d)、q(d) 的分子 */
    float gb = -INFINITY, gxb = -INFINITY, zk = 0.0f, md = 0.0f, mqd = 0.0f; int32_t gi = -1, gxi = -1; uint32_t nk = 0u;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = l[i];
        /* q(d) 要在 p 的保留集之外也算到(09-29 自测实撞: 草稿落在 K_p 外时这里以前直接 continue, q(d) 留 0 被当成"抽不出来的草稿"而接受,
         * 吐出保留集外的 token) —— 草稿在 K_p 外 ⇒ p(d) = 0 ⇒ 必拒 */
        if (lq && (int32_t)i == d) { const float vq = lq[i]; mqd = (isfinite(vq) && v41_fkey(vq) >= cutq) ? expf((vq - Mq) * inv_T) : 0.0f; }
        if (!isfinite(v) || v41_fkey(v) < cut) continue;
        nk++;
        const float m = expf((v - M) * inv_T);
        zk += m;
        if ((int32_t)i == d) md = m;
        const float g = -logf(-logf(v41_u01(seed, p, i, salt_g)));
        const float key = v * inv_T + g;
        if (key > gb) { gb = key; gi = (int32_t)i; }
        if (lq) {
            const float vq = lq[i];
            const float qm = (isfinite(vq) && v41_fkey(vq) >= cutq) ? expf((vq - Mq) * inv_T) : 0.0f;
            const float rres = m / Zp - qm / Zq;
            if (rres > 0.0f) { const float rk = logf(rres) + g; if (rk > gxb) { gxb = rk; gxi = (int32_t)i; } }
        } else if ((int32_t)i != d && key > gxb) { gxb = key; gxi = (int32_t)i; }
    }
    float tv; int32_t full, resid;
    v41_blk_argmax(sf, si, gb, gi, &tv, &full);
    v41_blk_argmax(sf, si, gxb, gxi, &tv, &resid);
    const float ZK = v41_blk_sumf(sf, zk);
    const float MD = v41_blk_sumf(sf, md);     /* 只有一个线程持有 d 的项, 求和 = 取值 */
    const float MQD = v41_blk_sumf(sf, mqd);
    if (threadIdx.x == 0) s_u64 = 0ull;
    __syncthreads();
    atomicAdd(&s_u64, (unsigned long long)nk);
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t acc = 0;
        if (d >= 0) {
            const float pd = MD / ZK;                                  /* d ∉ K_p ⇒ MD = 0 ⇒ 必拒(残差非空; 万一为空 [2] 退到全分布样本, 仍是 p 的样本) */
            if (pd <= 0.0f) acc = 0;
            else if (resid < 0) acc = 1;                              /* 残差空(K_p ⊆ {d}, 或 p == q): 比值 = 1, 必接受 */
            else if (lq) {
                const float qd = MQD / Zq;                             /* q(d) = 0 只会是数值边界(草稿是从 q 抽的): 比值 ∞ ⇒ 接受 */
                acc = (qd <= 0.0f || v41_u01(seed, p, (uint32_t)d, salt_a) < pd / qd) ? 1 : 0;
            } else acc = v41_u01(seed, p, (uint32_t)d, salt_a) < pd ? 1 : 0;
        }
        o[0] = full; o[1] = acc; o[2] = resid >= 0 ? resid : full; o[3] = (int32_t)s_u64;
    }
}

int ds4_gpu_v41_sample_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits, uint32_t row0, uint32_t n_rows, uint32_t n_vocab,
                              const ds4_gpu_tensor *pos, const ds4_gpu_tensor *tok, const ds4_gpu_sample_params *sp, const ds4_gpu_tensor *qlogits) {
    if (!out || !logits || !pos || !tok || !sp || n_rows == 0u || n_vocab == 0u) return 0;
    if (!(sp->temperature > 0.0f)) return 0;   /* 贪心是 argmax 核的事, 这里不冒充 */
    const uint64_t last = (uint64_t)row0 + n_rows;
    if (out->bytes < (uint64_t)n_rows * 16u || logits->bytes < last * n_vocab * 4u || pos->bytes < last * 4u || tok->bytes < last * 4u) return 0;
    if (qlogits && (row0 != 0u || (n_rows > 1u && qlogits->bytes < (uint64_t)(n_rows - 1u) * n_vocab * 4u))) return 0;
    /* 参数口径与主机 ds4_sample_logits 同: top_p 出 (0,1] 当 1; min_p 负当 0; top_k ≤ 0 = 不截(设备没有主机那个 1024 栈上限) */
    float top_p = sp->top_p; if (!(top_p > 0.0f) || top_p > 1.0f) top_p = 1.0f;
    const float min_p = sp->min_p > 0.0f ? sp->min_p : 0.0f;
    const uint32_t top_k = sp->top_k > 0 ? (uint32_t)sp->top_k : 0u;
    v41_sample_kernel<<<n_rows, V41_SAMPLE_THREADS, 0, g_cur_stream>>>((int32_t *)out->ptr, (const float *)logits->ptr,
                                                                      qlogits ? (const float *)qlogits->ptr : NULL, n_vocab,
                                                                      (const int32_t *)pos->ptr, (const int32_t *)tok->ptr, row0, n_rows,
                                                                      1.0f / sp->temperature, min_p, top_k, top_p, sp->seed, sp->stream);
    return cuda_ok(cudaGetLastError(), "v41 sample launch");
}
