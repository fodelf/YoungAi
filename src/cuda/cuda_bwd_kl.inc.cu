/* cuda_bwd_kl.inc.cu — ds4_cuda.cu 分片(从 cuda_bwd_dense.inc.cu 拆出, 2026-11-05): 后训练出口的损失核 —— 教师 top-K 表的"余量桶"KL 及其对 logits 的梯度
 * (含尾权 / 边际项两个开关), 与教师 top-K 选取核。契约 ds4_gpu_bwd.h; 块内归约小工具(bwd_block_sum/max)在 cuda_bwd_dense.inc.cu, 本片必须在它之后 include。 */

#define BWD_KMAX 128u
#define BWD_VMASK_WORDS 4096u   /* 词表位图(一 token 一位): 131072 个词以内; 现役 129280 */
/* ★余量必须直接累加榜外项, 不许 1 − Σ榜上★(2026-10-01 实撞): 学生把概率几乎全押在榜上时, 1 − Σp 在 f32 里抵消成 0 甚至负数,
 * 托底成 1e-20 后 r_T/r_S 炸到 1e16, 榜外每个 token 的梯度 = p_j·(1 − 1e16) —— 批梯度范数在 0.2 与 2.7e7 之间来回跳, 一轮就把模型
 * 推成只吐"2026303030…"。现在 r_S = Σ_{j∉榜} e^{z_j−m} / Σ_j e^{z_j−m}(位图标榜上, 一遍归约), 两数都是正项和, 没有相消。 */
static struct { float tau, cap; } g_kl_tail;   /* 尾权(ds4_gpu_bwd.h ds4_gpu_bwd_kl_tail): tau 0 = 关 */
int ds4_gpu_bwd_kl_tail(float tau, float cap) { g_kl_tail.tau = tau > 0.f ? tau : 0.f; g_kl_tail.cap = cap > 1.f ? cap : 1.f; return 1; }
static struct { float lam, p1, gap; } g_kl_margin;   /* 边际项(ds4_gpu_bwd.h ds4_gpu_bwd_kl_margin): lam 0 = 关 */
int ds4_gpu_bwd_kl_margin(float lam, float p1, float gap) { g_kl_margin.lam = lam > 0.f ? lam : 0.f; g_kl_margin.p1 = p1; g_kl_margin.gap = gap > 0.f ? gap : 0.f; return 1; }
static float g_kl_jsd_beta;   /* 广义 JSD(104 轮, ds4_gpu_bwd_kl_jsd): 0 = 原前向 KL, (0,1) = 归一化 JSD_beta */
int ds4_gpu_bwd_kl_jsd(float beta) {
    if (!(beta == 0.f || (beta > 0.f && beta < 1.f))) return 0;
    g_kl_jsd_beta = beta; return 1;
}

/* 块内 argmax(同值取小下标): 每线程带着自己的 (v, j) 进来, 回去时全块都拿到最大的那对 */
__device__ static void bwd_block_argmax(float *v, int32_t *j, float *shv, int32_t *shj) {
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, *v, o); const int32_t oj = __shfl_xor_sync(0xffffffffu, *j, o);
        if (ov > *v || (ov == *v && oj < *j)) { *v = ov; *j = oj; }
    }
    const uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) { shv[w] = *v; shj[w] = *j; }
    __syncthreads();
    if (threadIdx.x < nw) { *v = shv[threadIdx.x]; *j = shj[threadIdx.x]; } else { *v = -INFINITY; *j = 0x7fffffff; }
    if (w == 0) for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, *v, o); const int32_t oj = __shfl_xor_sync(0xffffffffu, *j, o);
        if (ov > *v || (ov == *v && oj < *j)) { *v = ov; *j = oj; }
    }
    if (threadIdx.x == 0) { shv[0] = *v; shj[0] = *j; }
    __syncthreads();
    *v = shv[0]; *j = shj[0];
}

__global__ static void bwd_kl_topk_kernel(float *g, float *loss, const float *z, uint32_t row0, uint32_t V,
                                          const int32_t *tid, const float *tp, const float *trest, const float *w, uint32_t K, float scale,
                                          float tail_tau, float tail_cap, float mg_lam, float mg_p1, float mg_gap, float jsd_beta) {
    __shared__ float sh[32];
    __shared__ int32_t shj[32];
    __shared__ float ps_k[BWD_KMAX];
    __shared__ uint32_t mask[BWD_VMASK_WORDS];
    const uint32_t i = blockIdx.x;
    const float wi = w ? w[i] : 1.f;   /* 逐行权重(硬目标题: 数字位 × hard_num, 其余 × hard_txt); 损失与梯度同乘, 报出来的 KL 也是加权的 */
    const float *zr = z + (uint64_t)(row0 + i) * V;
    float *gr = g + (uint64_t)i * V;
    const int32_t *ids = tid + (uint64_t)i * K;
    const float *pt = tp + (uint64_t)i * K;
    for (uint32_t w = threadIdx.x; w < BWD_VMASK_WORDS; w += blockDim.x) mask[w] = 0u;
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) { const int32_t id = ids[k]; if (id >= 0 && (uint32_t)id < V) atomicOr(&mask[id >> 5], 1u << (id & 31)); }
    float mx = -INFINITY;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, zr[j]);
    mx = bwd_block_max(mx, sh);   /* 内含 __syncthreads: 位图此后可读 */
    float s = 0.f, sr = 0.f;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) {
        const float ev = expf(zr[j] - mx);
        s += ev;
        if (!((mask[j >> 5] >> (j & 31)) & 1u)) sr += ev;
    }
    s = bwd_block_sum(s, sh);
    sr = bwd_block_sum(sr, sh);
    const float lse = mx + logf(s), rs = sr / s;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const int32_t id = ids[k];
        ps_k[k] = (id >= 0 && (uint32_t)id < V) ? expf(zr[id] - lse) : 0.f;
    }
    __syncthreads();
    __shared__ float s_off, s_w, s_gap, s_ubar;
    __shared__ int32_t s_t1;
    if (threadIdx.x == 0) {
        if (jsd_beta > 0.f) {
            const float b = jsd_beta, om = 1.f - b, norm = 1.f / (b * om);
            float l = 0.f, ubar = 0.f;
            for (uint32_t k = 0; k < K; k++) {
                const float pk = pt[k], qk = ps_k[k], mk = fmaxf(b * pk + om * qk, 1e-30f);
                if (pk > 0.f) l += b * pk * logf(pk / mk);
                if (qk > 1e-30f) { const float uk = logf(qk / mk); l += om * qk * uk; ubar += qk * uk; }
            }
            const float rt = fmaxf(trest[i], 0.f), mr = fmaxf(b * rt + om * rs, 1e-30f);
            if (rt > 1e-12f) l += b * rt * logf(rt / mr);
            float ur = 0.f;
            if (rs > 1e-30f) { ur = logf(rs / mr); l += om * rs * ur; ubar += rs * ur; }
            l *= norm;
            s_w = wi * (tail_tau > 0.f ? fminf(tail_cap, fmaxf(1.f, l / tail_tau)) : 1.f);
            loss[i] = l * s_w;
            s_off = (ur - ubar) / b;
            s_ubar = ubar;
        } else {
            float l = 0.f;
            for (uint32_t k = 0; k < K; k++) if (pt[k] > 0.f) l += pt[k] * (logf(pt[k]) - logf(fmaxf(ps_k[k], 1e-30f)));
            const float rt = fmaxf(trest[i], 0.f);
            if (rt > 1e-12f) l += rt * (logf(rt) - logf(fmaxf(rs, 1e-30f)));
            /* 尾权: 本行自己的 KL 当常数定权重(停梯度), 损失与梯度同乘 ⇒ 一阶预测与实测仍同口径 */
            s_w = wi * (tail_tau > 0.f ? fminf(tail_cap, fmaxf(1.f, l / tail_tau)) : 1.f);
            loss[i] = l * s_w;
            /* 榜外: q_j = r_T·pS_j/r_S ⇒ g_j = pS_j·(1 − r_T/r_S); r_S 真为 0 时 pS_j 也全为 0, 这一项本来就没有梯度 */
            s_off = rs > 1e-30f ? 1.f - rt / rs : 0.f;
            s_ubar = 0.f;
        }
        /* 边际项的闸(ds4_gpu_bwd.h ds4_gpu_bwd_kl_margin): 教师 top-1 概率 ≥ p1 的位才算; 表不保证按 p 排序, 自己扫一遍找 top-1/top-2。
         * 目标领先 δ = min(gap, 教师自己的 log(p1/p2)): 只要求学生把教师的排序学过来, 不要求比教师更自信 */
        s_t1 = -1; s_gap = 0.f;
        if (mg_lam > 0.f) {
            uint32_t k1 = 0; float p2 = 0.f;
            for (uint32_t k = 1; k < K; k++) if (pt[k] > pt[k1]) k1 = k;
            for (uint32_t k = 0; k < K; k++) if (k != k1 && pt[k] > p2) p2 = pt[k];
            const int32_t id1 = ids[k1];
            if (pt[k1] >= mg_p1 && id1 >= 0 && (uint32_t)id1 < V) { s_t1 = id1; s_gap = fminf(mg_gap, logf(pt[k1] / fmaxf(p2, 1e-6f))); }
        }
    }
    __syncthreads();
    const float off = s_off, sc = scale * s_w;
    const int32_t t1 = s_t1;
    /* 学生榜上除 t1 外最高的 logit(全块归约, 要在梯度盖掉 logits 之前读) */
    float zv = -INFINITY; int32_t zj = 0x7fffffff;
    if (t1 >= 0) for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) if ((int32_t)j != t1 && zr[j] > zv) { zv = zr[j]; zj = (int32_t)j; }
    const float z1 = t1 >= 0 ? zr[t1] : 0.f;
    if (t1 >= 0) bwd_block_argmax(&zv, &zj, sh, shj);
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) gr[j] = sc * expf(zr[j] - lse) * off;   /* 同址时: 每个 j 先读后写, 只有本线程碰它 */
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const int32_t id = ids[k];
        if (id >= 0 && (uint32_t)id < V) {
            if (jsd_beta > 0.f) {
                const float pk = pt[k], qk = ps_k[k], mk = fmaxf(jsd_beta * pk + (1.f - jsd_beta) * qk, 1e-30f);
                const float uk = qk > 1e-30f ? logf(qk / mk) : 0.f;
                gr[id] = sc * (qk * (uk - s_ubar) / jsd_beta);
            } else {
                gr[id] = sc * (ps_k[k] - pt[k]);
            }
        }
    }
    __syncthreads();
    /* 边际项 λ·max(0, δ − (z_t1 − z_j*)): 差一点罚一点, 梯度只落两个 logit(t1 −λ, j* +λ), 与 KL 同缩放 scale·w_i(不吃尾权) */
    if (threadIdx.x == 0 && t1 >= 0 && z1 - zv < s_gap) {
        const float h = mg_lam * (s_gap - (z1 - zv)), scm = scale * wi * mg_lam;
        loss[i] += h * wi;
        gr[t1] -= scm; gr[zj] += scm;
    }
}
int ds4_gpu_bwd_kl_topk_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t row0,
                               uint32_t m, uint32_t n_vocab, const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp,
                               const ds4_gpu_tensor *trest, const ds4_gpu_tensor *w, uint32_t k, float scale) {
    if (!glogits || !loss || !logits || !tid || !tp || !trest || m == 0 || k == 0 || k > BWD_KMAX || n_vocab > BWD_VMASK_WORDS * 32u) return 0;
    if (logits->bytes < (uint64_t)(row0 + m) * n_vocab * 4 || glogits->bytes < (uint64_t)m * n_vocab * 4) return 0;
    if (w && w->bytes < (uint64_t)m * 4) return 0;
    /* 同址只许"glogits 就是 logits 第 row0 行起的视图"(逐行自读自写); 同一块缓冲但行错开会读到别的块写过的行 */
    if ((const char *)glogits->ptr != (const char *)logits->ptr + (uint64_t)row0 * n_vocab * 4 &&
        (const char *)glogits->ptr < (const char *)logits->ptr + logits->bytes &&
        (const char *)glogits->ptr + (uint64_t)m * n_vocab * 4 > (const char *)logits->ptr) {
        fprintf(stderr, "ds4: [bwd] KL 梯度与 logits 部分重叠(只许整行同址)\n"); return 0;
    }
    bwd_kl_topk_kernel<<<m, 1024, 0, g_cur_stream>>>((float *)glogits->ptr, (float *)loss->ptr, (const float *)logits->ptr, row0, n_vocab,
                                                     (const int32_t *)tid->ptr, (const float *)tp->ptr, (const float *)trest->ptr,
                                                     w ? (const float *)w->ptr : NULL, k, scale, g_kl_tail.tau, g_kl_tail.cap,
                                                     g_kl_margin.lam, g_kl_margin.p1, g_kl_margin.gap, g_kl_jsd_beta);
    return cuda_ok(cudaGetLastError(), "bwd kl topk");
}

/* 教师 top-K: 先整行 log-sum-exp, 再 K 轮"全块 argmax(同值取小下标) → 置 −inf" */
__global__ static void bwd_topk_kernel(int32_t *tid, float *tp, float *trest, float *z, uint32_t row0, uint32_t V, uint32_t K) {
    __shared__ float sh[32], shv[32];
    __shared__ int32_t shi[32];
    const uint32_t i = blockIdx.x;
    float *zr = z + (uint64_t)(row0 + i) * V;
    float mx = -INFINITY;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, zr[j]);
    mx = bwd_block_max(mx, sh);
    float s = 0.f;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) s += expf(zr[j] - mx);
    s = bwd_block_sum(s, sh);
    float psum = 0.f;
    for (uint32_t k = 0; k < K; k++) {
        float bv = -INFINITY; int32_t bi = 0x7fffffff;
        for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) { const float v = zr[j]; if (v > bv) { bv = v; bi = (int32_t)j; } }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u, nw = blockDim.x >> 5;
        if (l == 0) { shv[w] = bv; shi[w] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v0 = shv[0]; int32_t i0 = shi[0];
            for (uint32_t q = 1; q < nw; q++) if (shv[q] > v0 || (shv[q] == v0 && shi[q] < i0)) { v0 = shv[q]; i0 = shi[q]; }
            const float p = v0 == -INFINITY ? 0.f : expf(v0 - mx) / s;
            tid[(uint64_t)i * K + k] = v0 == -INFINITY ? -1 : i0;
            tp[(uint64_t)i * K + k] = p;
            psum += p;
            if (v0 != -INFINITY) zr[i0] = -INFINITY;
        }
        __syncthreads();
    }
    (void)psum;
    /* 余量 = 榜外直接累加(选中的已置 −inf, e^{−inf} = 0); 不用 1 − Σ榜上: 那是两个接近的数相减(见 KL 核头的实撞记录) */
    float sr = 0.f;
    for (uint32_t j = threadIdx.x; j < V; j += blockDim.x) sr += expf(zr[j] - mx);
    sr = bwd_block_sum(sr, sh);
    if (threadIdx.x == 0) trest[i] = sr / s;
}
int ds4_gpu_bwd_topk_tensor(ds4_gpu_tensor *tid, ds4_gpu_tensor *tp, ds4_gpu_tensor *trest, ds4_gpu_tensor *logits,
                            uint32_t row0, uint32_t m, uint32_t n_vocab, uint32_t k) {
    if (!tid || !tp || !trest || !logits || m == 0 || k == 0) return 0;
    if (logits->bytes < (uint64_t)(row0 + m) * n_vocab * 4 || tid->bytes < (uint64_t)m * k * 4 || tp->bytes < (uint64_t)m * k * 4) return 0;
    bwd_topk_kernel<<<m, 1024, 0, g_cur_stream>>>((int32_t *)tid->ptr, (float *)tp->ptr, (float *)trest->ptr, (float *)logits->ptr, row0, n_vocab, k);
    return cuda_ok(cudaGetLastError(), "bwd topk");
}
