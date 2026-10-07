/* cuda_zchain_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * go-onebit DQZ2 zchain + 路由闭式 RTE 侧车。
 */
/* 参数表与 ds4_gpu.h 逐字一致(本文件不 include 它, 编译器不校验)。 */
int ds4_gpu_zchain_zl_set(
        const uint16_t *zlm, const uint32_t *off, const uint32_t *k,
        const uint32_t *din, const float *tr, const uint32_t *mul,
        uint32_t n_layer, uint64_t total_halves) {
    uint32_t n_zl = 0, kmax = 0;
    for (uint32_t l = 0; k && l < n_layer; l++)
        if (k[l]) { n_zl++; if (k[l] > kmax) kmax = k[l]; }
    if (!n_zl) return 1;
    if (kmax > 1024) {   /* shared pv[] 布局上限(Metal 同限) */
        fprintf(stderr, "ds4: zchain z^L rank %u > 1024 unsupported on CUDA\n", kmax);
        return 0;
    }
    if (!cuda_ok(cudaMalloc(&g_zc_zlm, total_halves * sizeof(__half)), "zchain zlm")) return 0;
    if (!cuda_ok(cudaMemcpy(g_zc_zlm, zlm, total_halves * sizeof(__half), cudaMemcpyHostToDevice), "zchain zlm up")) return 0;
    /* fp8(e4m3) U/V 影子判负存档(2026-08-20 终判, 旋钮 DS4_ZC_FP8 已删): 净 +0.35 t/s 但 e4m3 是真数值扰动
     * (KL 3.8e-2 ≈ 放大器收益 1/4), 质量门不换。g_zc_zlm8 恒 NULL, 核走 fp16 正本(zc_h2fp8_kernel 留作存档)。 */
    g_zc_zl_off = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_zl_k   = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_zl_din = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_zl_tr  = (float *)malloc(n_layer * sizeof(float));
    memcpy(g_zc_zl_off, off, n_layer * sizeof(uint32_t));
    memcpy(g_zc_zl_k, k, n_layer * sizeof(uint32_t));
    if (din) memcpy(g_zc_zl_din, din, n_layer * sizeof(uint32_t));
    else     memset(g_zc_zl_din, 0, n_layer * sizeof(uint32_t));
    memcpy(g_zc_zl_tr, tr, n_layer * sizeof(float));
    g_zc_zl_mul = (uint32_t *)calloc(n_layer, sizeof(uint32_t));
    if (mul) memcpy(g_zc_zl_mul, mul, n_layer * sizeof(uint32_t));
    uint32_t n_mul = 0;
    for (uint32_t l = 0; l < n_layer; l++) if (g_zc_zl_mul[l]) n_mul++;
    /* decode 快路 scratch(capture 内禁 cudaMalloc → 此处预分配; 上限 16tok×d4096/k1024) */
    if (!g_zc_zl_pv) {
        if (!cuda_ok(cudaMalloc(&g_zc_zl_pv, (size_t)ZC_ZL_FAST_MAXTOK * 1024u * sizeof(float)), "zl pv")) return 0;
        if (!cuda_ok(cudaMalloc(&g_zc_zl_ua, (size_t)ZC_ZL_FAST_MAXTOK * 4096u * sizeof(float)), "zl ua")) return 0;
        /* n2 = 每 token × 每 ua block(d/8 ≤ 512) × {‖ua‖², ‖r‖²} 部分和平面(09-05 去原子, 定序归约) */
        if (!cuda_ok(cudaMalloc(&g_zc_zl_n2, (size_t)ZC_ZL_FAST_MAXTOK * 512u * 2u * sizeof(float)), "zl n2")) return 0;
        if (!cuda_ok(cudaMalloc(&g_zc_zl_pvp, (size_t)ZC_ZL_FAST_MAXTOK * ZC_ZL_SEG * 1024u * sizeof(float)), "zl pvp")) return 0;
    }
    fprintf(stderr, "ds4: zchain CUDA z^L armed: %u layers (AMP=%u), kmax=%u (%.1f MB)\n",
            n_zl, n_mul, kmax, total_halves * 2.0 / 1e6);
    return 1;
}

/* ==== 路由闭式侧车(type8 zl.RTE, 2026-08-19): δlogits = U·diag(z)·tanh(Vᵀx/s)
 * 加在 router raw logits 上(select 前)。blob 布局 z[k]|U[ne*k]|V[d*k], 行主
 * [dim][k]。decode n=1 为主, 出维仅 n_expert=256 → 单 kernel 两段:
 * 先 256 线程按列并行算 pv[c]=z_c·tanh((Vᵀx)_c/s), 再按行算 δ 加进 logits。 */
static __half *g_zc_rt = NULL;
static uint32_t *g_zc_rt_off = NULL, *g_zc_rt_k = NULL;
static float *g_zc_rt_s = NULL;
static uint32_t g_zc_rt_nl = 0, g_zc_rt_ne = 0;
static float *g_zc_rt_pv = NULL;   /* [16tok × kmax1024] scratch */

static __global__ void zc_rte_pv_kernel(
        const float *x, const __half *rm, float *pv,
        uint32_t d, uint32_t k, uint32_t ne, uint32_t off, float s) {
    /* V 在 rte_set 上传时已转置为 [k][d](行 c 连续) — [d][k] 原布局按列跨步读
     * 每线程 4096 次 half 步长 k*2B, 完全不合并。warp/列: 32 lane 分段行内连续读。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t c = blockIdx.x * 8u + warp;
    const uint32_t tok = blockIdx.y;
    if (c >= k) return;
    const __half *hz = rm + off;
    const __half *hVt = hz + k + (uint64_t)ne * k + (uint64_t)c * d;
    const float *xt = x + (uint64_t)tok * d;
    float a = 0.0f;
    for (uint32_t j = lane; j < d; j += 32u) a += xt[j] * __half2float(hVt[j]);
    for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0)
        pv[(uint64_t)tok * k + c] = tanhf(a / (s > 0.0f ? s : 1.0f)) * __half2float(hz[c]);
}

static __global__ void zc_rte_add_kernel(
        float *logits, const float *pv, const __half *rm,
        uint32_t k, uint32_t ne, uint32_t off) {
    const uint32_t tok = blockIdx.y;
    const uint32_t e = blockIdx.x * 256u + threadIdx.x;
    if (e >= ne) return;
    const __half *hU = rm + off + k;
    const float *pvt = pv + (uint64_t)tok * k;
    float a = 0.0f;
    for (uint32_t c = 0; c < k; c++) a += pvt[c] * __half2float(hU[(uint64_t)e * k + c]);
    logits[(uint64_t)tok * ne + e] += a;
}

int ds4_gpu_zchain_rte_set(
        const uint16_t *rm, const uint32_t *off, const uint32_t *k,
        const float *scale, uint32_t n_layer, uint32_t n_expert, uint64_t total_halves) {
    if (!rm || !total_halves || !n_layer) return 1;
    /* 上传前把每层 V 段 [d][k] 转置成 [k][d](pv kernel 行连续合并读; U 段布局不变) */
    uint16_t *tp = (uint16_t *)malloc(total_halves * sizeof(uint16_t));
    if (!tp) return 0;
    memcpy(tp, rm, total_halves * sizeof(uint16_t));
    const uint32_t d = 4096u;
    for (uint32_t l = 0; l < n_layer; l++) {
        const uint32_t kl = k[l];
        if (!kl) continue;
        const uint64_t vo = (uint64_t)off[l] + kl + (uint64_t)n_expert * kl;
        const uint16_t *src = rm + vo;
        uint16_t *dst = tp + vo;
        for (uint32_t j = 0; j < d; j++)
            for (uint32_t c = 0; c < kl; c++)
                dst[(uint64_t)c * d + j] = src[(uint64_t)j * kl + c];
    }
    if (!cuda_ok(cudaMalloc(&g_zc_rt, total_halves * sizeof(__half)), "zchain rte")) { free(tp); return 0; }
    if (!cuda_ok(cudaMemcpy(g_zc_rt, tp, total_halves * sizeof(__half), cudaMemcpyHostToDevice), "zchain rte up")) { free(tp); return 0; }
    free(tp);
    g_zc_rt_off = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_rt_k   = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_rt_s   = (float *)malloc(n_layer * sizeof(float));
    memcpy(g_zc_rt_off, off, n_layer * sizeof(uint32_t));
    memcpy(g_zc_rt_k, k, n_layer * sizeof(uint32_t));
    memcpy(g_zc_rt_s, scale, n_layer * sizeof(float));
    g_zc_rt_nl = n_layer; g_zc_rt_ne = n_expert;
    if (!g_zc_rt_pv &&
        !cuda_ok(cudaMalloc(&g_zc_rt_pv, (size_t)16u * 1024u * sizeof(float)), "rte pv")) return 0;
    uint32_t nr = 0, kmax = 0;
    for (uint32_t l = 0; l < n_layer; l++) { if (k[l]) nr++; if (k[l] > kmax) kmax = k[l]; }
    fprintf(stderr, "ds4: zchain CUDA route 侧车 armed: %u layers, kmax=%u (%.1f MB)\n",
            nr, kmax, total_halves * 2.0 / 1e6);
    return 1;
}

static uint64_t g_zc_rt_pv_cap = 16u * 1024u;   /* floats; 预分配 16tok, prefill 按需扩 */

int ds4_gpu_zchain_route_bias(
        ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tokens) {
    if (!g_zc_rt || layer >= g_zc_rt_nl || !g_zc_rt_k || !g_zc_rt_k[layer]) return 1;
    if (!logits || !x || n_tokens == 0) return 1;
    const uint32_t k = g_zc_rt_k[layer], ne = g_zc_rt_ne, d = 4096u;
    if (x->bytes < (uint64_t)n_tokens * d * sizeof(float) ||
        logits->bytes < (uint64_t)n_tokens * ne * sizeof(float) || k > 1024u) return 1;
    const uint64_t need = (uint64_t)n_tokens * k;
    if (need > g_zc_rt_pv_cap) {
        /* prefill 大批量: 非 capture 态才可扩容; capture 态(decode n=1 恒定)不会到这 */
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(g_cur_stream, &cs);
        if (cs != cudaStreamCaptureStatusNone) return 1;
        if (g_zc_rt_pv) (void)cudaFree(g_zc_rt_pv);
        g_zc_rt_pv = NULL; g_zc_rt_pv_cap = 0;
        if (!cuda_ok(cudaMalloc(&g_zc_rt_pv, need * sizeof(float)), "rte pv grow")) return 0;
        g_zc_rt_pv_cap = need;
    }
    dim3 g1((k + 7u) / 8u, n_tokens);   /* warp/列: 8 列×32 lane per block */
    zc_rte_pv_kernel<<<g1, 256u, 0, g_cur_stream>>>(
        (const float *)x->ptr, g_zc_rt, g_zc_rt_pv, d, k, ne, g_zc_rt_off[layer], g_zc_rt_s[layer]);
    dim3 g2((ne + 255u) / 256u, n_tokens);
    zc_rte_add_kernel<<<g2, 256u, 0, g_cur_stream>>>(
        (float *)logits->ptr, g_zc_rt_pv, g_zc_rt, k, ne, g_zc_rt_off[layer]);
    return cuda_ok(cudaGetLastError(), "zchain route bias launch");
}
