/* cuda_bwd_moe.inc.cu — ds4_cuda.cu 分片: 后训练反传的 MoE 一族(契约 ds4_gpu_bwd.h, 2026-10-01)。
 * SwiGLU 反向 / 路由权重反向(√softplus + 归一) / routed 专家反向(VQ 位流直读两种核, cuda_bwd_vq.inc.cu)。
 * ★必须在 cuda_vq_row / cuda_vq_prefill / cuda_bwd_vq 之后 include★: 解码借 v41_vq_open / v41_vq_cw(盘上布局的唯一消费点), 增益覆盖借 g_v41_gr。
 * 验证(2026-10-01): 直读路 vs "解成 bf16 稠密阵 + cuBLAS"的参考路(已删, 账在 fable5): g_w 相对差 3.1e-3、g_x 3.1e-3;
 * 两种核互为精确转置(<G_O, W2·A> 与 <W2ᵀ·G_O, A> 逐配对相对差 1e-6)。参考路要写 26 GB/层的稠密阵, 一题反传 3.6 s → 直读路 1.2 s。
 * 前向算式(cuda_v41_3.inc.cu): pr = √softplus(z); 按 pr + bias 选 top-k; w_k = pr_k/(Σpr + 1e-20)·rs;
 *   专家 h = silu(min(g, L))·clamp(u, ±L), y += Σ_k w_k·W2_k·h_k。 */

__global__ static void bwd_swiglu_kernel(float *gg, float *gu, const float *gh, const float *g, const float *u, uint64_t n, float L) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    const int gpass = !(L > 0.f) || gv < L, upass = !(L > 0.f) || (uv > -L && uv < L);
    if (L > 0.f) { uv = fminf(fmaxf(uv, -L), L); gv = fminf(gv, L); }
    const float sg = 1.f / (1.f + expf(-gv)), si = gv * sg;
    gg[i] = gpass ? gh[i] * uv * sg * (1.f + gv * (1.f - sg)) : 0.f;
    gu[i] = upass ? gh[i] * si : 0.f;
}
int ds4_gpu_bwd_swiglu_tensor(ds4_gpu_tensor *ggate, ds4_gpu_tensor *gup, const ds4_gpu_tensor *gh, const ds4_gpu_tensor *gate,
                              const ds4_gpu_tensor *up, uint64_t n, float limit) {
    if (!ggate || !gup || !gh || !gate || !up || n == 0) return 0;
    bwd_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)ggate->ptr, (float *)gup->ptr, (const float *)gh->ptr,
                                                                          (const float *)gate->ptr, (const float *)up->ptr, n, limit);
    return cuda_ok(cudaGetLastError(), "bwd swiglu");
}

/* 路由: g_w[n][K] → g_z[n][NE](只选中那 K 个非零)。z 是路由 logits(前向 st->glog), 选择不求导(离散)。 */
__global__ static void bwd_router_kernel(float *gz, const float *gw, const int32_t *sel, const float *z, uint32_t NE, uint32_t K, float rs) {
    const uint32_t t = blockIdx.x;
    float *gr = gz + (uint64_t)t * NE;
    for (uint32_t e = threadIdx.x; e < NE; e += blockDim.x) gr[e] = 0.f;
    __syncthreads();
    if (threadIdx.x != 0) return;
    float pr[16], S = 0.f;
    for (uint32_t k = 0; k < K; k++) {
        const int32_t e = sel[(uint64_t)t * K + k];
        const float zz = e >= 0 ? z[(uint64_t)t * NE + e] : 0.f;
        const float sp = zz > 20.f ? zz : log1pf(expf(zz));
        pr[k] = e >= 0 ? sqrtf(sp) : 0.f;
        S += pr[k];
    }
    S += 1e-20f;
    float tw = 0.f;   /* Σ_k g_w_k·pr_k/S */
    for (uint32_t k = 0; k < K; k++) tw += gw[(uint64_t)t * K + k] * pr[k] / S;
    for (uint32_t k = 0; k < K; k++) {
        const int32_t e = sel[(uint64_t)t * K + k];
        if (e < 0 || pr[k] <= 0.f) continue;
        const float gpr = rs / S * (gw[(uint64_t)t * K + k] - tw);
        const float zz = z[(uint64_t)t * NE + e];
        const float dsp = zz > 20.f ? 1.f : 1.f / (1.f + expf(-zz));   /* softplus' = σ */
        gr[e] += gpr * dsp / (2.f * pr[k]);
    }
}
int ds4_gpu_bwd_router_tensor(ds4_gpu_tensor *gz, const ds4_gpu_tensor *gw, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *z,
                              uint32_t n_tok, uint32_t n_expert, uint32_t k, float route_scale) {
    if (!gz || !gw || !sel || !z || k > 16u || n_tok == 0) return 0;
    bwd_router_kernel<<<n_tok, 128, 0, g_cur_stream>>>((float *)gz->ptr, (const float *)gw->ptr, (const int32_t *)sel->ptr, (const float *)z->ptr,
                                                       n_expert, k, route_scale);
    return cuda_ok(cudaGetLastError(), "bwd router");
}

/* 梯度检查的"冻结选择"前向(core_ptrain.c pt_gradcheck): 选中的专家照基准前向那份(sel 给定), 只按当前 logits 重算权重 ——
 * 算式与 v41_router_kernel 逐位同: pr = √softplus(z)(阈值 20), 按选中顺序累加 wsum, w = pr/(wsum + 1e-20)·route_scale。
 * 为什么要它: 扰动穿过几十层, 总有 token 落在 top-6 的并列边上, 一翻就是 0.005~0.01 的损失跳变, 比 ε·|g| 还大, 有限差分全是噪声。 */
__global__ static void bwd_router_fixed_kernel(float *wts, const int32_t *sel, const float *logits, uint32_t n_tok, uint32_t n_expert, uint32_t topk, float rs) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) return;
    float pr[16], wsum = 0.f;
    for (uint32_t r = 0; r < topk; r++) {
        const int32_t e = sel[(uint64_t)t * topk + r];
        float p = 0.f;
        if (e >= 0 && (uint32_t)e < n_expert) { const float z = logits[(uint64_t)t * n_expert + e]; p = sqrtf(z > 20.0f ? z : log1pf(expf(z))); }
        pr[r] = p; wsum += p;
    }
    for (uint32_t r = 0; r < topk; r++) wts[(uint64_t)t * topk + r] = pr[r] / (wsum + 1e-20f) * rs;
}
int ds4_gpu_bwd_router_fixed_tensor(ds4_gpu_tensor *weights, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *logits,
                                    uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    if (!weights || !sel || !logits || topk > 16u || n_tok == 0) return 0;
    bwd_router_fixed_kernel<<<(n_tok + 127u) / 128u, 128, 0, g_cur_stream>>>((float *)weights->ptr, (const int32_t *)sel->ptr,
                                                                           (const float *)logits->ptr, n_tok, n_expert, topk, route_scale);
    return cuda_ok(cudaGetLastError(), "bwd router fixed");
}

/* 按配对表收/放行: dst[r] = src[pair[r]/K](收激活); 或 dst[pair[r]/K] += src[r](放梯度)。
 * ★放梯度必须原子加★(2026-10-01 实撞): 直读路一发核处理全层全部配对, 一个 token 有 K 个专家槽 ⇒ K 个块同时改同一行,
 * 普通 += 会丢加法 —— 表现是 g_x 偏小且每次跑结果不同(与参考路差 22%~28% 来回跳), 深层放大器的梯度因此只剩一半。 */
__global__ static void bwd_gather_bf16_kernel(__nv_bfloat16 *dst, const float *src, const int32_t *pair, uint32_t K, uint32_t D, const float *rw) {
    const uint32_t r = blockIdx.x; const int32_t p = pair[r];
    const float s = rw ? rw[p] : 1.f;
    const float *sr = src + (uint64_t)(p / (int32_t)K) * D;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) dst[(uint64_t)r * D + d] = __float2bfloat16(sr[d] * s);
}
__global__ static void bwd_scatter_add_kernel(float *dst, const float *src, const int32_t *pair, uint32_t K, uint32_t D) {
    const uint32_t r = blockIdx.x; const int32_t p = pair[r];
    float *dr = dst + (uint64_t)(p / (int32_t)K) * D;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) atomicAdd(dr + d, src[(uint64_t)r * D + d]);
}
/* g_w[pair] = <g_y[token], O[r]>(O = 专家未加权的 down 输出) */
__global__ static void bwd_rowdot_kernel(float *gw, const float *gy, const float *O, const int32_t *pair, uint32_t K, uint32_t D) {
    __shared__ float sh[32];
    const uint32_t r = blockIdx.x; const int32_t p = pair[r];
    const float *g = gy + (uint64_t)(p / (int32_t)K) * D, *o = O + (uint64_t)r * D;
    float s = 0.f;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) s += g[d] * o[d];
    s = bwd_block_sum(s, sh);
    if (threadIdx.x == 0) gw[p] = s;
}
__global__ static void bwd_swiglu_fwd_bf16_kernel(__nv_bfloat16 *a, const float *hg, const float *hu, uint64_t n, float L) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = hg[i], uv = hu[i];
    if (L > 0.f) { uv = fminf(fmaxf(uv, -L), L); gv = fminf(gv, L); }
    a[i] = __float2bfloat16((gv / (1.f + expf(-gv))) * uv);
}

/* ---- 直读位流版(cuda_bwd_vq.inc.cu 的两种核): 一层只读两遍位流, 不落稠密阵 ---- */
__global__ static void bwd_gather_f32_kernel(float *dst, const float *src, const int32_t *pair, uint32_t K, uint32_t D, const float *rw) {
    const uint32_t r = blockIdx.x; const int32_t p = pair[r];
    const float s = rw ? rw[p] : 1.f;
    const float *sr = src + (uint64_t)(p / (int32_t)K) * D;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) dst[(uint64_t)r * D + d] = sr[d] * s;
}
static v41_scratch g_bvq_pair, g_bvq_meta, g_bvq_x, g_bvq_hg, g_bvq_hu, g_bvq_a, g_bvq_og, g_bvq_ga, g_bvq_bad;

int ds4_gpu_bwd_moe_capture(int on) {   /* 契约见 ds4_gpu_bwd.h; 截留本体在 cuda_vq_prefill_mma.inc.cu(g_vqm_cap) */
    g_vqm_cap.on = on ? 1 : 0;
    if (on) g_vqm_cap.valid = 0;   /* 开新一层的截留: 旧的一律作废 */
    return 1;
}

int ds4_gpu_bwd_routed_moe_tensor(ds4_gpu_tensor *gx, ds4_gpu_tensor *gw, const ds4_gpu_tensor *gy, const ds4_gpu_tensor *x,
                                  const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw, const void *model_map, uint64_t model_size,
                                  uint64_t blob_offset, uint64_t blob_bytes, uint32_t IN, uint32_t MID, uint32_t OUT,
                                  uint32_t n_total_expert, uint32_t K, float clamp, uint32_t layer, uint32_t n_tok) {
    if (!gx || !gw || !gy || !x || !sel || !rw || n_tok == 0 || IN != OUT) return 0;
    if (blob_offset > model_size || blob_bytes > model_size - blob_offset) return 0;
    const uint8_t *bh = (const uint8_t *)model_map + blob_offset;
    if (!ds4vq_blob_ok(bh, (size_t)blob_bytes)) return 0;
    const uint32_t ver = ds4vq_blob_ver(bh);
    const uint8_t *blob = (const uint8_t *)cuda_model_range_cached_ptr(model_map, blob_offset, blob_bytes);
    if (!blob && (g_model_registered || g_model_device_owned)) blob = (const uint8_t *)cuda_model_range_ptr(model_map, blob_offset, blob_bytes, "bwd vq blob");
    if (!blob) { fprintf(stderr, "ds4: [bwd] L%u 专家 blob 不在设备上(流式装载路的反向没接)\n", layer); return 0; }
    cudaStream_t cs = g_cur_stream ? g_cur_stream : cudaStreamPerThread;
    const uint64_t npair = (uint64_t)n_tok * K;
    int32_t *sel_h = (int32_t *)malloc(npair * 4), *list = (int32_t *)malloc(npair * 4);
    uint32_t *cnt = (uint32_t *)calloc(n_total_expert + 1u, 4), *off = (uint32_t *)calloc(n_total_expert + 1u, 4);
    uint32_t *meta = (uint32_t *)malloc((size_t)n_total_expert * 3u * 4u);
    int ok = sel_h && list && cnt && off && meta && cudaMemcpyAsync(sel_h, sel->ptr, npair * 4, cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
             cudaStreamSynchronize(cs) == cudaSuccess;
    uint32_t nv = 0, nact = 0;
    if (ok) {
        for (uint64_t p = 0; p < npair; p++) if (sel_h[p] >= 0 && (uint32_t)sel_h[p] < n_total_expert) cnt[sel_h[p]]++;
        for (uint32_t e = 0; e < n_total_expert; e++) off[e + 1] = off[e] + cnt[e];
        nv = off[n_total_expert];
        uint32_t *cur = (uint32_t *)malloc((size_t)n_total_expert * 4);
        memcpy(cur, off, (size_t)n_total_expert * 4);
        for (uint64_t p = 0; p < npair; p++) if (sel_h[p] >= 0 && (uint32_t)sel_h[p] < n_total_expert) list[cur[sel_h[p]]++] = (int32_t)p;
        free(cur);
        for (uint32_t e = 0; e < n_total_expert; e++) if (cnt[e]) { meta[nact] = e; meta[n_total_expert + nact] = off[e]; meta[2u * n_total_expert + nact] = cnt[e]; nact++; }
    }
    /* 本层重算刚截留的逐对四样(g_vqm_cap, cuda_vq_prefill_mma.inc.cu): H_g/A 在 g_vqm 暂存, O 在 g_vqp.ys, H_u 在截留缓冲。
     * 同层、同配对数才认; 配对序 = 同一份 sel 的同一个计数排序(专家内保持 token 序) ⇒ 一致。认上了就不再把本层专家前向算第二遍,
     * 那几块缓冲也不另分(后面就地改写它们也无妨: 下一层的重算会整份重写)。没认上(v2 载荷 / 截留没开)= 下面照常用预填张量核重算。 */
    const int cap = ok && ver == 3u && g_vqm_cap.valid && g_vqm_cap.layer == layer && g_vqm_cap.nvalid == nv;
    if (ok && ver == 3u && !cap) {
        static int said = 0;
        if (!said) { fprintf(stderr, "ds4: [bwd] L%u 没有本层重算的截留(截留 %d: 层 %u 配对 %u, 这里配对 %u), 反传里重算专家前向\n",
                             layer, g_vqm_cap.valid, g_vqm_cap.layer, g_vqm_cap.nvalid, nv); said = 1; }
    }
    int32_t *pair = ok ? (int32_t *)v41_grow(&g_bvq_pair, (uint64_t)(nv + 1u) * 4, "bwd vq pairs") : NULL;
    uint32_t *dmeta = ok ? (uint32_t *)v41_grow(&g_bvq_meta, (uint64_t)n_total_expert * 12u, "bwd vq meta") : NULL;
    /* xb 只有 v2 的直读行点积要(v3 走 vqm_run_parts, 它自己 gather 进 g_vqm.xs16) —— 全层训练内存余量只 ~3 GB, 不分用不上的 */
    __nv_bfloat16 *xb = (ok && ver != 3u) ? (__nv_bfloat16 *)v41_grow(&g_bvq_x, (uint64_t)(nv + 1u) * IN * 2u, "bwd vq x") : NULL;
    float *hg = cap ? g_vqm.g32 : ok ? (float *)v41_grow(&g_bvq_hg, (uint64_t)(nv + 1u) * MID * 4u, "bwd vq hg") : NULL;
    float *hu = cap ? g_vqm_cap.hu : ok ? (float *)v41_grow(&g_bvq_hu, (uint64_t)(nv + 1u) * MID * 4u, "bwd vq hu") : NULL;
    __nv_bfloat16 *ab = cap ? (__nv_bfloat16 *)g_vqm.h16 : ok ? (__nv_bfloat16 *)v41_grow(&g_bvq_a, (uint64_t)(nv + 1u) * MID * 2u, "bwd vq a") : NULL;
    float *og = cap ? g_vqp.ys : ok ? (float *)v41_grow(&g_bvq_og, (uint64_t)(nv + 1u) * OUT * 4u, "bwd vq o/go/gx") : NULL;   /* ys 容量 (nvalid+2048)×OUT, IN == OUT */
    if (cap) g_vqm_cap.valid = 0;   /* 用一次即作废: 不让下一层/下一题拿到这一层的 */
    float *ga = ok ? (float *)v41_grow(&g_bvq_ga, (uint64_t)(nv + 1u) * MID * 4u, "bwd vq ga") : NULL;
    int *bad = ok ? (int *)v41_grow(&g_bvq_bad, 4, "bwd vq bad") : NULL;
    ok = ok && pair && dmeta && (xb || ver == 3u) && hg && hu && ab && og && ga && bad &&
         cudaMemcpyAsync(pair, list, (size_t)nv * 4, cudaMemcpyHostToDevice, cs) == cudaSuccess &&
         cudaMemcpyAsync(dmeta, meta, (size_t)n_total_expert * 12u, cudaMemcpyHostToDevice, cs) == cudaSuccess &&
         cudaMemsetAsync(bad, 0, 4, cs) == cudaSuccess && cudaMemsetAsync(gw->ptr, 0, npair * 4, cs) == cudaSuccess;
    if (ok && nv) {
        const uint32_t *act = dmeta, *aoff = dmeta + n_total_expert, *acnt = dmeta + 2u * n_total_expert;
        const float *gr_all = layer < 64u ? g_v41_gr[layer] : NULL;
        const uint64_t nh = (uint64_t)nv * MID;
        if (ver == 3u) {
            /* 重算 H_g / H_u / A / O(O = 未加权的 down 输出)走推理同一组张量核: 逐对排序序与这里的 pair 表一致(计数排序、专家内保持
             * token 序, 与 cuda_vq_prefill.inc.cu 同一写法), 值就是前向真算的那份(bf16 格点)⇒ 反传对着前向本身求导。
             * 以前这里是三发 VQ 直读行点积(f32、不舍 bf16): 10-02 逐核表 3.23 s/题, 占整题 GPU 时间 33.4%。 */
            ok = vqp_hdr_build(layer, blob, n_total_expert, IN, MID, OUT, ver) &&
                 (cap || vqm_run_parts(blob, cnt, off, n_total_expert, nv, IN, MID, OUT, g_vqp_hdr[layer][0].nc, clamp, (const float *)x->ptr, pair, K,
                                       layer, og, gr_all, hg, (uint16_t *)ab, hu));
        } else {   /* v2(f16 码本)转 bf16 丢尾数, 推理路也不走张量核(见 cuda_vq_prefill_fused.inc.cu): 仍用直读行点积 */
            bwd_gather_bf16_kernel<<<nv, 256, 0, g_cur_stream>>>(xb, (const float *)x->ptr, pair, K, IN, NULL);
            ok = vqb_rowdot(hg, xb, blob, ver, 0u, MID, IN, act, aoff, acnt, nact, gr_all, OUT, bad) &&
                 vqb_rowdot(hu, xb, blob, ver, 1u, MID, IN, act, aoff, acnt, nact, gr_all, OUT, bad);
            bwd_swiglu_fwd_bf16_kernel<<<(unsigned)((nh + 255) / 256), 256, 0, g_cur_stream>>>(ab, hg, hu, nh, clamp);
            if (ok) ok = vqb_rowdot(og, ab, blob, ver, 2u, OUT, MID, act, aoff, acnt, nact, gr_all, OUT, bad);
        }
        bwd_rowdot_kernel<<<nv, 256, 0, g_cur_stream>>>((float *)gw->ptr, (const float *)gy->ptr, og, pair, K, OUT);
        bwd_gather_f32_kernel<<<nv, 256, 0, g_cur_stream>>>(og, (const float *)gy->ptr, pair, K, OUT, (const float *)rw->ptr);   /* og ← G_O */
        /* 三发转置: v3 走 bf16 张量核(vqst_kernel, 工作项表三发共用, 按码本位宽切), v2 仍走直读转置累加(码本转 bf16 丢尾数, 同推理路的取舍) */
        const uint32_t nc = ver == 3u ? g_vqp_hdr[layer][0].nc : 0u;
        uint32_t nbit = 0; while (nc && (1u << nbit) < nc) nbit++;
        if (ok && ver == 3u && !vqs_blob_fits(layer, blob, n_total_expert, IN, MID, OUT)) {   /* 转置核只有寄存器直解这一版: 要对齐的设备副本(训练本来就全驻留) */
            fprintf(stderr, "ds4: [bwd] L%u 专家 blob 不是对齐的设备副本(封顶装载/映射), 转置张量核不认 —— 训练要全驻留\n", layer); ok = 0;
        }
        if (ok && ver == 3u) ok = vqt_prepare(cnt, off, n_total_expert, vqst_item_tokens(nbit));
        if (ok) ok = ver == 3u ? vqt_launch(ga, og, blob, 2u, OUT, MID, nc, gr_all, OUT, 0, bad)
                               : vqb_tdot(ga, og, blob, ver, 2u, OUT, MID, act, aoff, acnt, nact, gr_all, OUT, 0, bad);   /* G_A = G_O·W2 */
        bwd_swiglu_kernel<<<(unsigned)((nh + 255) / 256), 256, 0, g_cur_stream>>>(hg, hu, ga, hg, hu, nh, clamp);      /* hg/hu ← G_Hg/G_Hu */
        if (ok) ok = ver == 3u ? vqt_launch(og, hg, blob, 0u, MID, IN, nc, gr_all, OUT, 0, bad)
                               : vqb_tdot(og, hg, blob, ver, 0u, MID, IN, act, aoff, acnt, nact, gr_all, OUT, 0, bad);   /* og ← G_X(gate 部分) */
        if (ok) ok = ver == 3u ? vqt_launch(og, hu, blob, 1u, MID, IN, nc, gr_all, OUT, 1, bad)
                               : vqb_tdot(og, hu, blob, ver, 1u, MID, IN, act, aoff, acnt, nact, gr_all, OUT, 1, bad);   /* og += G_X(up 部分) */
        bwd_scatter_add_kernel<<<nv, 256, 0, g_cur_stream>>>((float *)gx->ptr, og, pair, K, IN);
        ok = ok && cuda_ok(cudaGetLastError(), "bwd vq moe");
    }
    int bad_h = 0;
    if (ok && bad) ok = cudaMemcpy(&bad_h, bad, 4, cudaMemcpyDeviceToHost) == cudaSuccess && !bad_h;
    if (bad_h) fprintf(stderr, "ds4: [bwd] L%u 专家载荷头不认(版本 %u) —— 反向不出假梯度, 停车\n", layer, ver);
    free(sel_h); free(list); free(cnt); free(off); free(meta);
    return ok;
}
