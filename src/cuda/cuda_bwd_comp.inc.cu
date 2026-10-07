/* cuda_bwd_comp.inc.cu — ds4_cuda.cu 分片: 后训练反传的分界层以下三件(契约 ds4_gpu_bwd.h, 2026-10-01)。
 *
 * 为什么要这三件: 分界层(最后一个 kv 源层)及之前的层把隐状态压成压缩 KV, 之后每一层的稀疏注意力都读它; L1/L14 还有 engram。
 * 只训分界层之后时这些都是常量, 训到分界层以下就必须对它们求导, 否则梯度是错的(梯度检查必然不过)。
 *   ① 读取层对压缩行的梯度: 前向值 = 键(s = scale·<q, k_g>, o = Σ p·k_g), 所以 g_k_g = Σ_{(i,h) 选中 g} scale·g_s·q_ih + p·g_o_ih,
 *      跨 query/头/读取层原子累加进源层的 g_comp[g](倒着算时读取层都在源层之上, 轮到源层时已经收齐)。
 *      10-02 起与窗口键梯度合成一个张量核(cuda_bwd_attn.inc.cu 的 bwd_attn_kg_kernel), 这里不再有单独的核。
 *   ② 压缩器池化的反向(ratio>1): P = Σ_t w_t·kv_t, w = softmax_t(sc)(组内逐维)。
 *   ③ engram 门的反向: h' = h + gate(h)·val, key/val 只依赖 token(查表), 是常量。
 * 前向里的 bf16 舍入、压缩行的 fp4 打包、indexer 选哪些组都当直通/常量, 与解码器段反传同一口径。 */

/* ② 一 block 一组, 线程按维。算式与 v41_compress_pool_kernel 同序(mx → e^{sc−mx} → den/acc), P 用未舍 bf16 的 acc/den:
 * g_kv[t] = w_t·gP, g_sc[t] = w_t·(kv_t − P)·gP。不满一组的尾行前向没进池化, 这里不碰(调用方先清零)。 */
__global__ static void bwd_compress_pool_kernel(float *gkv, float *gsc, const float *gp, const float *kv, const float *sc, uint32_t ratio, uint32_t dim) {
    const uint32_t g = blockIdx.x;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float mx = -INFINITY;
        for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, sc[((uint64_t)g * ratio + t) * dim + d]);
        float den = 0.f, acc = 0.f;
        for (uint32_t t = 0; t < ratio; t++) {
            const float e = expf(sc[((uint64_t)g * ratio + t) * dim + d] - mx);
            den += e; acc += e * kv[((uint64_t)g * ratio + t) * dim + d];
        }
        const float P = acc / den, gP = gp[(uint64_t)g * dim + d];
        for (uint32_t t = 0; t < ratio; t++) {
            const uint64_t o = ((uint64_t)g * ratio + t) * dim + d;
            const float w = expf(sc[o] - mx) / den;
            gkv[o] = w * gP;
            gsc[o] = w * (kv[o] - P) * gP;
        }
    }
}
int ds4_gpu_bwd_compress_pool_tensor(ds4_gpu_tensor *gkv, ds4_gpu_tensor *gsc, const ds4_gpu_tensor *gpooled, const ds4_gpu_tensor *kv,
                                     const ds4_gpu_tensor *score, uint32_t n_groups, uint32_t ratio, uint32_t dim) {
    if (!gkv || !gsc || !gpooled || !kv || !score || ratio < 2u) return 0;
    if (n_groups == 0) return 1;
    bwd_compress_pool_kernel<<<n_groups, 256, 0, g_cur_stream>>>((float *)gkv->ptr, (float *)gsc->ptr, (const float *)gpooled->ptr,
                                                                 (const float *)kv->ptr, (const float *)score->ptr, ratio, dim);
    return cuda_ok(cudaGetLastError(), "bwd compress pool");
}

/* ③ 一 block 一 (token, 路), 与 v41_engram_gate_kernel 同一套归约。h 是门之前的 hc(前向存档), g 原地: 进来是对门之后 hc 的梯度,
 * 出去是对门之前 hc 的梯度。dot = S·rh·rk·E^-½(S = Σ h·w·key), z = sign(dot)·√max(|dot|, 1e-6), gate = σ(z):
 * g_h = g' + g_dot·rk·E^-½·(w·key·rh − S·rh³·h/E), g_dot = <g', val>·gate(1−gate)·dz/ddot; |dot| ≤ 1e-6 时 z 是常数, g_dot = 0。 */
__global__ static void bwd_engram_gate_kernel(float *g, const float *hc, const float *kv, const float *qw, const float *kw,
                                              uint32_t E, uint32_t n_hc, float eps) {
    const uint32_t t = blockIdx.y, c = blockIdx.x;
    const float *h = hc + ((uint64_t)t * n_hc + c) * E;
    float *gr = g + ((uint64_t)t * n_hc + c) * E;
    const float *key = kv + (uint64_t)t * (n_hc + 1u) * E + (uint64_t)c * E;
    const float *val = kv + (uint64_t)t * (n_hc + 1u) * E + (uint64_t)n_hc * E;
    float sh = 0.f, sk = 0.f, sd = 0.f, gv = 0.f;
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) {
        const float hv = h[d], kv_ = key[d], w = qw[c * E + d] * kw[c * E + d];
        sh += hv * hv; sk += kv_ * kv_; sd += hv * w * kv_; gv += gr[d] * val[d];
    }
    __shared__ float r[4][256];
    r[0][threadIdx.x] = sh; r[1][threadIdx.x] = sk; r[2][threadIdx.x] = sd; r[3][threadIdx.x] = gv; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) for (int j = 0; j < 4; j++) r[j][threadIdx.x] += r[j][threadIdx.x + k];
        __syncthreads();
    }
    const float rh = rsqrtf(r[0][0] / (float)E + eps), rk = rsqrtf(r[1][0] / (float)E + eps), S = r[2][0], ie = rsqrtf((float)E);
    const float dot = S * rh * rk * ie, ad = fabsf(dot);
    const float z = copysignf(sqrtf(fmaxf(ad, 1e-6f)), dot), gate = 1.0f / (1.0f + expf(-z));
    const float gdot = ad > 1e-6f ? r[3][0] * gate * (1.0f - gate) * 0.5f / sqrtf(ad) : 0.f;
    const float coef = gdot * rk * ie, hs = S * rh * rh * rh / (float)E;
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) gr[d] += coef * (qw[c * E + d] * kw[c * E + d] * key[d] * rh - hs * h[d]);
}
int ds4_gpu_bwd_engram_gate_tensor(ds4_gpu_tensor *g, const ds4_gpu_tensor *hc_pre, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size,
                                   uint64_t q_w_offset, uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    (void)model_size;
    if (!g || !hc_pre || !kv || n_tok == 0) return 0;
    const float *qw = (const float *)cuda_model_range_ptr(model_map, q_w_offset, (uint64_t)n_hc * n_embd * 4, "bwd engram q");
    const float *kw = (const float *)cuda_model_range_ptr(model_map, k_w_offset, (uint64_t)n_hc * n_embd * 4, "bwd engram k");
    if (!qw || !kw) return 0;
    bwd_engram_gate_kernel<<<dim3(n_hc, n_tok), 256, 0, g_cur_stream>>>((float *)g->ptr, (const float *)hc_pre->ptr, (const float *)kv->ptr,
                                                                          qw, kw, n_embd, n_hc, eps);
    return cuda_ok(cudaGetLastError(), "bwd engram gate");
}
