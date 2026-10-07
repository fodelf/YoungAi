/* cuda_bwd_hc.inc.cu — ds4_cuda.cu 分片: 后训练反传的 mHC 混合系数一族(契约 ds4_gpu_bwd.h, 2026-10-01)。
 *
 * 前向(v41_hc_split_kernel / v41_hc_fused_kernel 逐式):
 *   mix[r] = inv·Σ_i W[r][i]·h[i],  inv = 1/√(mean h² + eps),  h = 本半层入口 hc 摊平(4×E)
 *   pre[c]  = σ(mix[c]·s0 + b[c]) + eps
 *   post[c] = 2σ(mix[4+c]·s1 + b[4+c])
 *   comb    = sinkhorn(C0),  C0 = mix[8..]·s2 + b[8..]: 行 softmax + eps → 列归一 → (iters−1) × (行归一, 列归一), 归一分母都 + eps
 * 反向两步: ① 每 token 一个线程把 sinkhorn 正着重算一遍(存下每一步的 4×4 矩阵), 再倒着推回 g_C0; 连同 pre/post 的 σ 导数得 g_mix[24];
 *          ② g_h = inv·Wᵀ·g_mix − inv³/dim·<g_mix, W·h>·h  (rms 那一项; W·h = mix/inv)。 */

#define BWD_HC_ITERS_MAX 32u

__device__ static void bwd_sinkhorn_back(float *gC0, const float *C0, const float *gout, uint32_t hc, uint32_t iters, float eps) {
    /* 正向重算并存每一步: P[0] = 行 softmax + eps, P[1] = 列归一, 然后每轮两步。共 2·iters 个矩阵(≤ 64 × 16 floats, 放 local) */
    float P[2u * BWD_HC_ITERS_MAX][16];
    float sm[16];
    for (uint32_t j = 0; j < hc; j++) {
        float mx = -INFINITY; for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, C0[j * hc + k]);
        float s = 0.f; for (uint32_t k = 0; k < hc; k++) s += expf(C0[j * hc + k] - mx);
        for (uint32_t k = 0; k < hc; k++) { sm[j * hc + k] = expf(C0[j * hc + k] - mx) / s; P[0][j * hc + k] = sm[j * hc + k] + eps; }
    }
    uint32_t np = 1;
    for (uint32_t step = 1; step < 2u * iters; step++) {   /* 奇数步列归一, 偶数步行归一 */
        const float *a = P[np - 1]; float *b = P[np];
        if (step & 1u) { for (uint32_t k = 0; k < hc; k++) { float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += a[j * hc + k]; for (uint32_t j = 0; j < hc; j++) b[j * hc + k] = a[j * hc + k] / (s + eps); } }
        else { for (uint32_t j = 0; j < hc; j++) { float s = 0.f; for (uint32_t k = 0; k < hc; k++) s += a[j * hc + k]; for (uint32_t k = 0; k < hc; k++) b[j * hc + k] = a[j * hc + k] / (s + eps); } }
        np++;
    }
    /* 倒推: y = a/(s+eps), s = 沿归一方向的和 ⇒ g_a = g_y/(s+eps) − Σ(g_y·a)/(s+eps)² (同一行/列) */
    float g[16];
    for (uint32_t q = 0; q < hc * hc; q++) g[q] = gout[q];
    for (uint32_t step = 2u * iters - 1u; step >= 1u; step--) {
        const float *a = P[step - 1];
        float ga[16];
        if (step & 1u) {
            for (uint32_t k = 0; k < hc; k++) {
                float s = 0.f, t = 0.f;
                for (uint32_t j = 0; j < hc; j++) { s += a[j * hc + k]; t += g[j * hc + k] * a[j * hc + k]; }
                const float den = s + eps;
                for (uint32_t j = 0; j < hc; j++) ga[j * hc + k] = g[j * hc + k] / den - t / (den * den);
            }
        } else {
            for (uint32_t j = 0; j < hc; j++) {
                float s = 0.f, t = 0.f;
                for (uint32_t k = 0; k < hc; k++) { s += a[j * hc + k]; t += g[j * hc + k] * a[j * hc + k]; }
                const float den = s + eps;
                for (uint32_t k = 0; k < hc; k++) ga[j * hc + k] = g[j * hc + k] / den - t / (den * den);
            }
        }
        for (uint32_t q = 0; q < hc * hc; q++) g[q] = ga[q];
    }
    /* P[0] = softmax(C0 行) + eps ⇒ g_C0 = sm ⊙ (g − Σ_k g·sm) 逐行 */
    for (uint32_t j = 0; j < hc; j++) {
        float t = 0.f; for (uint32_t k = 0; k < hc; k++) t += g[j * hc + k] * sm[j * hc + k];
        for (uint32_t k = 0; k < hc; k++) gC0[j * hc + k] = sm[j * hc + k] * (g[j * hc + k] - t);
    }
}

__global__ static void bwd_hc_split_kernel(float *gmix, const float *mix, const float *gpre, const float *gpost, const float *gcomb,
                                           const float *scale, const float *base, uint32_t n_tok, uint32_t hc, uint32_t iters, float eps) {
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_tok) return;
    const uint32_t mh = 2u * hc + hc * hc;
    const float *m = mix + (uint64_t)n * mh;
    float *gm = gmix + (uint64_t)n * mh;
    for (uint32_t c = 0; c < hc; c++) {
        const float sp = 1.f / (1.f + expf(-(m[c] * scale[0] + base[c])));
        gm[c] = (gpre ? gpre[n * hc + c] : 0.f) * sp * (1.f - sp) * scale[0];
        const float so = 1.f / (1.f + expf(-(m[hc + c] * scale[1] + base[hc + c])));
        gm[hc + c] = (gpost ? gpost[n * hc + c] : 0.f) * 2.f * so * (1.f - so) * scale[1];
    }
    float C0[16], gC0[16], go[16];
    for (uint32_t q = 0; q < hc * hc; q++) { C0[q] = m[2u * hc + q] * scale[2] + base[2u * hc + q]; go[q] = gcomb ? gcomb[(uint64_t)n * hc * hc + q] : 0.f; }
    bwd_sinkhorn_back(gC0, C0, go, hc, iters, eps);
    for (uint32_t q = 0; q < hc * hc; q++) gm[2u * hc + q] = gC0[q] * scale[2];
}

/* ② g_h += inv·Wᵀ·g_mix − inv³/dim·<g_mix, mix/inv>·h。一 block 一 token。 */
__global__ static void bwd_hc_mix_kernel(float *gh, const float *gmix, const float *mix, const float *W, const float *h,
                                         uint32_t dim, uint32_t mh, float eps) {
    __shared__ float sh[32];
    __shared__ float gm[32];
    const uint64_t n = blockIdx.x;
    const float *hr = h + n * dim;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) ss += hr[i] * hr[i];
    ss = bwd_block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)dim + eps);
    if (threadIdx.x < mh) gm[threadIdx.x] = gmix[n * mh + threadIdx.x];
    __syncthreads();
    float dot = 0.f;   /* <g_mix, W·h> = Σ_r g_mix[r]·mix[r]/inv */
    for (uint32_t r = 0; r < mh; r++) dot += gm[r] * mix[n * mh + r];
    dot /= inv;
    const float c = inv * inv * inv * dot / (float)dim;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float s = 0.f;
        for (uint32_t r = 0; r < mh; r++) s += W[(uint64_t)r * dim + i] * gm[r];
        gh[n * dim + i] += inv * s - c * hr[i];
    }
}

int ds4_gpu_bwd_hc_mix_tensor(ds4_gpu_tensor *ghc, const ds4_gpu_tensor *gpre, const ds4_gpu_tensor *gpost, const ds4_gpu_tensor *gcomb,
                              const ds4_gpu_tensor *hc, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size,
                              uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                              uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps, uint32_t n_tok) {
    (void)model_size;
    if (!ghc || !hc || !mix || n_hc > 4u || iters > BWD_HC_ITERS_MAX || n_tok == 0) return 0;
    const uint32_t mh = 2u * n_hc + n_hc * n_hc, dim = n_embd * n_hc;
    const float *W = (const float *)cuda_model_range_ptr(model_map, fn_offset, (uint64_t)mh * dim * 4, "bwd hc fn");
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "bwd hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mh * 4, "bwd hc base");
    float *gm = (float *)v41_grow(&g_v41_misc, (uint64_t)n_tok * mh * 4, "bwd hc gmix");
    if (!W || !sc || !bs || !gm) return 0;
    bwd_hc_split_kernel<<<(n_tok + 63u) / 64u, 64, 0, g_cur_stream>>>(gm, (const float *)mix->ptr, gpre ? (const float *)gpre->ptr : NULL,
        gpost ? (const float *)gpost->ptr : NULL, gcomb ? (const float *)gcomb->ptr : NULL, sc, bs, n_tok, n_hc, iters, hc_eps);
    if (!cuda_ok(cudaGetLastError(), "bwd hc split")) return 0;
    bwd_hc_mix_kernel<<<n_tok, 512, 0, g_cur_stream>>>((float *)ghc->ptr, gm, (const float *)mix->ptr, W, (const float *)hc->ptr, dim, mh, norm_eps);
    return cuda_ok(cudaGetLastError(), "bwd hc mix");
}
