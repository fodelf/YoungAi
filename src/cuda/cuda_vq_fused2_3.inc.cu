/* cuda_vq_fused2_3.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * fused2 特化高速 VQ 解码路。
 * EXCEPTION: 单函数体 >500 行, 无安全切割点, 整体成片(514 行)。
 */
static int cuda_vq_moe_forward(
        ds4_gpu_tensor *out, ds4_gpu_tensor *mid_scratch, const ds4_gpu_residual_set *residual,
        const void *model_map, uint64_t down_offset, uint64_t down_expert_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens) {
    const uint8_t *blob = (const uint8_t *)residual->gate_ptr;
    if (!blob || !out || !selected || !weights || !x || n_tokens == 0 || n_expert == 0) return 0;
    const uint8_t *blob_host = blob;   /* mmap original: host-side header reads only */
    /* fused/fuse2 down 的 partial 平面(fuse_max×n_expert×OUT), 须在任何 graph capture
     * 之前分配 —— prefill(非 capture)首次路过这里时建好。 */
    static float *g_vq_partial = NULL;
    if (!g_vq_partial) {
        cudaStreamCaptureStatus vcs_ = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &vcs_);
        if (vcs_ == cudaStreamCaptureStatusNone)
            (void)cudaMalloc(&g_vq_partial, (size_t)4 * n_expert * out_dim * sizeof(float));
    }
    /* Relocate the host-mmap blob pointer onto the HBM arena copy.  The startup
     * span cache (cache_exps=1 on GB10) already copied these bytes to device
     * memory and madvise(DONTNEED)d the source pages; reading the mmap original
     * from the kernel re-faults 65 GiB from SSD per pass and double-counts
     * memory (81 GiB arena + refaulted pages > 121 GiB UMA), which under
     * reclaim pressure produced intermittent NaN/illegal-access. One lookup per
     * layer, cached. */
    if (model_map && residual->vq_bytes && layer_index < 64u) {
        static const uint8_t *reloc[64];
        static uint8_t reloc_done[64];
        if (!reloc_done[layer_index]) {
            reloc_done[layer_index] = 1;
            if ((const char *)blob >= (const char *)model_map) {
                const uint64_t off = (uint64_t)((const char *)blob - (const char *)model_map);
                const uint64_t bend = off + residual->vq_bytes;
                for (const cuda_model_range &r : g_model_ranges) {
                    if (r.host_base == model_map && off >= r.offset && bend > off &&
                        bend <= r.offset + r.bytes) {
                        reloc[layer_index] = (const uint8_t *)(r.device_ptr + (off - r.offset));
                        break;
                    }
                }
            }
            if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */) {
                fprintf(stderr, "ds4: [vq-reloc] L%u blob %s arena\n", layer_index,
                        reloc[layer_index] ? "->" : "NOT in");
                if (!reloc[layer_index] && (const char *)blob >= (const char *)model_map) {
                    const uint64_t off_ = (uint64_t)((const char *)blob - (const char *)model_map);
                    fprintf(stderr, "ds4: [vq-reloc]   off=%llu bytes=%llu nranges=%zu\n",
                            (unsigned long long)off_, (unsigned long long)residual->vq_bytes,
                            g_model_ranges.size());
                    for (const cuda_model_range &r : g_model_ranges) {
                        if (r.host_base == model_map && off_ >= r.offset && off_ < r.offset + r.bytes)
                            fprintf(stderr, "ds4: [vq-reloc]   in-range off=%llu bytes=%llu (blob end %s)\n",
                                    (unsigned long long)r.offset, (unsigned long long)r.bytes,
                                    (off_ + residual->vq_bytes <= r.offset + r.bytes) ? "inside" : "OVERFLOWS");
                    }
                }
            }
        }
        if (reloc[layer_index]) blob = reloc[layer_index];
    }
    {   /* 一次性入参快照: 维度/张量容量任一为 0 或错位, 后面 gather 就会越界写 */
        static int once = 0;
        if (!once++) {
            uint32_t mg = 0; memcpy(&mg, blob_host, 4);
            fprintf(stderr, "ds4: [cuda-vq-init] L%u ntok=%u nexp=%u ntot=%u IN=%u MID=%u OUT=%u clamp=%.3f\n"
                            "     blob=%p magic=%08x sel.bytes=%llu w.bytes=%llu x.bytes=%llu out.bytes=%llu\n"
                            "     down_off=%llu down_ebytes=%llu\n",
                    layer_index, n_tokens, n_expert, n_total_expert,
                    expert_in_dim, expert_mid_dim, out_dim, clamp,
                    (const void *)blob, mg,
                    (unsigned long long)selected->bytes, (unsigned long long)weights->bytes,
                    (unsigned long long)x->bytes, (unsigned long long)out->bytes,
                    (unsigned long long)down_offset, (unsigned long long)down_expert_bytes);
            if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
                int32_t sel[8] = {0}; float wt[8] = {0};
                uint32_t ns = n_expert < 8u ? n_expert : 8u;
                (void)ds4_gpu_tensor_read((ds4_gpu_tensor *)selected, 0, sel, ns * sizeof(int32_t));
                (void)ds4_gpu_tensor_read((ds4_gpu_tensor *)weights, 0, wt, ns * sizeof(float));
                fprintf(stderr, "ds4: [vq-route] t0 sel=%d %d %d %d %d %d w=%.3f %.3f %.3f %.3f %.3f %.3f\n",
                        sel[0], sel[1], sel[2], sel[3], sel[4], sel[5],
                        wt[0], wt[1], wt[2], wt[3], wt[4], wt[5]);
            }
            fflush(stderr);
        }
    }

    const uint64_t npair = (uint64_t)n_tokens * n_expert;

    /* ---- decode 快路: 零 host 往返 ----
     * pair 数少时不去重, 每个 (token,pick) 各占一段 scratch, kernel 自己从 blob 读偏移。
     * selected 直接当 slot 用(第 k 个 pair 的权重就在第 k 段), 省掉 D2H + 去重 + H2D。 */
    {   /* 诊断: DS4_VQ_EXP=1 只解位流(跳过码本+乘加), 用来二分定位耗时段 */
        static int exp_set = 0;
        if (!exp_set) {
            exp_set = 1;
            const char *ev = ((const char *)0) /* DS4_VQ_EXP: 路径开关已删(2026-08-22 隐形炸弹清理) */;
            const int mode = ev ? atoi(ev) : 0;
            if (mode) (void)cudaMemcpyToSymbol(g_vq_exp_mode, &mode, sizeof(int));
            const int cyc_on = 0;
            if (cyc_on) (void)cudaMemcpyToSymbol(g_vq_cyc_on, &cyc_on, sizeof(int));
        }
    }
    /* decode 直通链常驻缓冲: 非 capture 时机预分配(prefill/首调都行) */
    const char *fu_env = ((const char *)0) /* DS4_VQ_FUSE_MAX: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    const uint32_t fuse_max = fu_env ? (uint32_t)atoi(fu_env) : 4u;   /* 0 关闭融合 */
    if (n_tokens <= fuse_max && n_tokens > 0) {
        const double prof_t0 = ((const char *)0) /* DS4_VQ_PROF: 路径开关已删(2026-08-22 隐形炸弹清理) */ ? cuda_wall_sec() : 0.0;
        /* 融合路: 不需要任何 dequant scratch, 只要 h 的中间缓冲 */
        const uint64_t hneed = npair * expert_mid_dim * sizeof(float);
        if (!mid_scratch || mid_scratch->bytes < hneed) {
            fprintf(stderr, "ds4: [cuda-vq-fuse] mid scratch %llu < 需要 %llu (L%u)\n",
                    (unsigned long long)(mid_scratch ? mid_scratch->bytes : 0),
                    (unsigned long long)hneed, layer_index);
            return 0;
        }
        const uint8_t *down_base = (down_expert_bytes && down_offset)
                                 ? ((const uint8_t *)model_map + down_offset) : NULL;
        if (cudaMemsetAsync(out->ptr, 0, (size_t)n_tokens * out_dim * sizeof(float), 0) != cudaSuccess) {
            (void)cudaGetLastError(); return 0;
        }
        /* fused2 特化闸(per-layer 一次判定, 判定在 prefill 首层调用=非 capture): 纯 vq4
         * 形态(w1/w3 nc512 + w2 nc256, 全槽在位, 4B 对齐)走高速特化路。 */
        static int16_t g_vq2_w2n[64];   /* 0=未判定, -1=不可用, 256/512=w2 码本词数 */
        int use2 = 1 && layer_index < 64u;
        if (use2) {
            if (!g_vq2_w2n[layer_index]) {
                static int32_t *pd = NULL;
                cudaStreamCaptureStatus pcs2 = cudaStreamCaptureStatusNone;
                (void)cudaStreamIsCapturing(0, &pcs2);
                if (pcs2 == cudaStreamCaptureStatusNone) {
                    if (!pd) (void)cudaMalloc(&pd, sizeof(int32_t));
                    int32_t pn = 0;
                    if (pd) {
                        vq2_probe_kernel<<<1, 1>>>(blob, n_total_expert,
                                expert_in_dim, expert_mid_dim, out_dim, pd);
                        if (cudaMemcpy(&pn, pd, sizeof(pn), cudaMemcpyDeviceToHost) != cudaSuccess) {
                            (void)cudaGetLastError(); pn = 0;
                        }
                    }
                    g_vq2_w2n[layer_index] = pn ? (int16_t)pn : -1;
                }
            }
            /* 512 词 w2 的 fuse2 实测 decode 反慢于 fused(每 block 重载 20KB shared,
             * n=1 无摊销: 5.9 vs 11.3 t/s) —— 仅对验证过的 256 词形态启用。 */
            use2 = g_vq2_w2n[layer_index] == 256;
        }
        if (use2) {
            const uint32_t zg2 = (expert_mid_dim + 31u) / 32u;   /* 4 warps × 8 行 */
            const uint32_t zd2 = (out_dim + 31u) / 32u;
            vq_moe_gateup_fused2_kernel<<<dim3(n_tokens, zg2, n_expert), 128,
                (size_t)expert_in_dim * 4 + 2u * 2048u * 2u>>>(
                (float *)mid_scratch->ptr, blob, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, (const float *)x->ptr,
                n_expert, expert_in_dim, expert_mid_dim, clamp);
            const uint32_t w2n = (uint32_t)g_vq2_w2n[layer_index];
            const size_t dsm = (size_t)expert_mid_dim * 4 + (size_t)w2n * 4u * 2u;
            if (!g_vq_partial) goto vq2_skip;   /* capture 首层前必已分配 */
            if (w2n == 512u)
                vq_moe_down_fused2_kernel<512u><<<dim3(n_tokens, zd2, n_expert), 128, dsm>>>(
                    g_vq_partial, blob, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr, (const float *)mid_scratch->ptr,
                    n_expert, expert_mid_dim, out_dim);
            else
                vq_moe_down_fused2_kernel<256u><<<dim3(n_tokens, zd2, n_expert), 128, dsm>>>(
                    g_vq_partial, blob, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr, (const float *)mid_scratch->ptr,
                    n_expert, expert_mid_dim, out_dim);
            vq2_down_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tokens, 1), 256>>>(
                (float *)out->ptr, g_vq_partial, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, n_expert, out_dim);
            if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */)
                fprintf(stderr, "ds4: [cuda-vq-fuse2] L%u ntok=%u\n", layer_index, n_tokens);
            return cuda_ok(cudaGetLastError(), "vq fused2 launch");
        }
        vq2_skip:;
        const uint32_t zg = (expert_mid_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
        const uint32_t zd = (out_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
        vq_moe_gateup_fused_kernel<<<dim3(n_tokens, zg, n_expert), 32 * DS4_VQ_WARPS_PER_BLOCK,
            (size_t)DS4_VQ_WARPS_PER_BLOCK * 2u * DS4_VQ_BITWORDS * sizeof(uint32_t)
            + 2u * 2048u * sizeof(__half) + (size_t)expert_in_dim * sizeof(float)>>>(
            (float *)mid_scratch->ptr, blob, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, (const float *)x->ptr,
            n_expert, expert_in_dim, expert_mid_dim, clamp);
        if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == (uint32_t)atoi(((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */)) {
            (void)cudaDeviceSynchronize();
            float xm[8] = {0}, hm[8] = {0};
            (void)cudaMemcpy(xm, x->ptr, sizeof(xm), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(hm, mid_scratch->ptr, sizeof(hm), cudaMemcpyDeviceToHost);
            fprintf(stderr, "ds4: [vq-mid] x=%.4g %.4g %.4g %.4g h=%.4g %.4g %.4g %.4g\n",
                    xm[0], xm[1], xm[2], xm[3], hm[0], hm[1], hm[2], hm[3]);
            FILE *fx = fopen("/tmp/vq_x.bin", "wb");
            if (fx) { float *xb = (float *)malloc(4096 * 4); cudaMemcpy(xb, x->ptr, 4096 * 4, cudaMemcpyDeviceToHost); fwrite(xb, 4, 4096, fx); fclose(fx); free(xb); }
            FILE *fh = fopen("/tmp/vq_h.bin", "wb");
            if (fh) { float *hb = (float *)malloc(6 * 2048 * 4); cudaMemcpy(hb, mid_scratch->ptr, 6 * 2048 * 4, cudaMemcpyDeviceToHost); fwrite(hb, 4, 6 * 2048, fh); fclose(fh); }
            FILE *fs = fopen("/tmp/vq_sel.bin", "wb");
            if (fs) { int32_t sb[8] = {0}; float wb[8] = {0};
                cudaMemcpy(sb, selected->ptr, 6 * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(wb, weights->ptr, 6 * 4, cudaMemcpyDeviceToHost);
                fwrite(sb, 4, 6, fs); fwrite(wb, 4, 6, fs); fclose(fs); }
        }
        if (!g_vq_partial) { (void)cudaGetLastError(); return 0; }
        vq_moe_down_fused_kernel<<<dim3(n_tokens, zd, n_expert), 32 * DS4_VQ_WARPS_PER_BLOCK,
            (size_t)DS4_VQ_WARPS_PER_BLOCK * DS4_VQ_BITWORDS * sizeof(uint32_t)
            + 2048u * sizeof(__half)>>>(
            g_vq_partial, blob, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, (const float *)mid_scratch->ptr,
            down_base, down_expert_bytes, n_expert, expert_mid_dim, out_dim);
        vq2_down_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tokens, 1), 256>>>(
            (float *)out->ptr, g_vq_partial, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, n_expert, out_dim);
        if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == (uint32_t)atoi(((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */)) {
            (void)cudaDeviceSynchronize();
            FILE *fo = fopen("/tmp/vq_out.bin", "wb");
            if (fo) { float *ob = (float *)malloc(4096 * 4); cudaMemcpy(ob, out->ptr, 4096 * 4, cudaMemcpyDeviceToHost); fwrite(ob, 4, 4096, fo); fclose(fo); free(ob); }
        }
        /* DS4_VQ_PROF=1: 逐层实测两个 kernel 的墙钟耗时。nsys 显示 gateup 的
         * Max/Med = 10x、StdDev≈均值 —— 正常算力负载不会这样, 需要定位是哪些层/
         * 哪种输入触发了长尾。计时本身要同步, 只在开关打开时付这个代价。 */
        if (((const char *)0) /* DS4_VQ_PROF: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
            (void)cudaDeviceSynchronize();
            const double t1 = cuda_wall_sec();
            static double acc_ms = 0.0; static uint64_t nfwd = 0;
            acc_ms += (t1 - prof_t0) * 1000.0; nfwd++;
            fprintf(stderr, "[vqprof] L%-2u ntok=%u npair=%llu  %.3f ms  (累计 %.1f ms / %llu 次)\n",
                    layer_index, n_tokens, (unsigned long long)npair,
                    (t1 - prof_t0) * 1000.0, acc_ms, (unsigned long long)nfwd);
        }
        if (((const char *)0) /* DS4_VQ_CYC: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == 42u) {
            (void)cudaDeviceSynchronize();
            unsigned long long c[3] = {0, 0, 0};
            (void)cudaMemcpyFromSymbol(c, g_vq_cyc, sizeof(c));
            if (c[2]) {
                const double tot = (double)(c[0] + c[1]);
                fprintf(stderr, "[vqcyc] 样本 %llu | 位流预取 %.0f cy/行 (%.1f%%) | 解码+乘加 %.0f cy/行 (%.1f%%)\n",
                        c[2], (double)c[0] / c[2], 100.0 * c[0] / tot,
                        (double)c[1] / c[2], 100.0 * c[1] / tot);
            }
        }
        if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "ds4: [cuda-vq-fuse] L%u ntok=%u npair=%llu (零 dequant scratch)\n",
                    layer_index, n_tokens, (unsigned long long)npair);
        return cuda_ok(cudaGetLastError(), "vq fused launch");
    }

    int32_t *sel_h = (int32_t *)malloc(npair * sizeof(int32_t));
    if (!sel_h) return 0;
    if (cudaMemcpy(sel_h, (const char *)selected->ptr, npair * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError(); free(sel_h); return 0;
    }

    /* 活跃专家去重 + 建 id→slot 映射, 然后把 selected 原地 remap 成 slot。
     * gather 后权重按 slot 紧凑排布, kernel 只认 slot。 */
    int32_t *e2slot = (int32_t *)malloc((size_t)n_total_expert * sizeof(int32_t));
    uint32_t *active_ids = (uint32_t *)malloc((size_t)n_total_expert * sizeof(uint32_t));
    if (!e2slot || !active_ids) { free(sel_h); free(e2slot); free(active_ids); return 0; }
    for (uint32_t e = 0; e < n_total_expert; e++) e2slot[e] = -1;
    uint32_t n_active = 0;
    for (uint64_t k = 0; k < npair; k++) {
        const int32_t e = sel_h[k];
        if (e < 0 || (uint32_t)e >= n_total_expert) { sel_h[k] = -1; continue; }
        if (e2slot[e] < 0) { e2slot[e] = (int32_t)n_active; active_ids[n_active++] = (uint32_t)e; }
        sel_h[k] = e2slot[e];
    }
    if (n_active == 0) { free(sel_h); free(e2slot); free(active_ids); return 1; }   /* 无活跃专家: out 保持不变 */

    /* decode 缓存: 把本层需要的专家映射到常驻槽, 命中的免 dequant。
     * need_dq[] 收集真正要解码的 (slot, expert) 对; 未命中才进 dequant 循环。 */
    /* ★默认关闭★ — A/B 实测(2026-08-17, 同 prompt/同二进制):
     *     关: prefill 1.03 / gen 4.06 t/s      开: prefill 0.66 / gen 0.82 t/s
     * 命中率只有 43%(每层 6 个专家仍要 dequant 3.5 个), 而 per-layer 缓存把访存从
     * 集中的 1.4GB scratch 摊成分散的 16.5GB, L2 局部性塌掉 —— 省下的 dequant 远
     * 抵不过多出来的访存。留着开关是因为换配方(更大 K / 路由更集中)时值得再量一次,
     * 但默认路径不该为它买单。 */
    int cache_on = (n_tokens == 1u) && (layer_index < DS4_VQ_CACHE_LAYERS) &&
                         (n_active <= DS4_VQ_CACHE_SLOTS) &&
                         (((const char *)0) /* DS4_VQ_CACHE: 路径开关已删(2026-08-22 隐形炸弹清理) */ && ((const char *)0) /* DS4_VQ_CACHE: 路径开关已删(2026-08-22 隐形炸弹清理) */[0] == '1');
    uint32_t n_slot_total = n_active;          /* scratch 需要容纳的槽数 */
    uint32_t *dq_slot = (uint32_t *)malloc((size_t)n_active * sizeof(uint32_t));
    uint32_t *dq_expert = (uint32_t *)malloc((size_t)n_active * sizeof(uint32_t));
    uint32_t n_dq = 0;
    if (!dq_slot || !dq_expert) { free(sel_h); free(e2slot); free(active_ids); free(dq_slot); free(dq_expert); return 0; }

    if (!g_vq_cache_ready) cuda_vq_cache_reset_all();
    {   /* 显存不足时静默降级为直通, 正确性不受影响 */
        const uint64_t ge0 = (uint64_t)expert_mid_dim * expert_in_dim * 2u;
        const uint64_t de0 = (uint64_t)out_dim * expert_mid_dim * 2u;
        if (cache_on && !cuda_vq_cache_ensure_mem(layer_index, ge0, de0)) cache_on = 0;
    }
    if (cache_on) {
        n_slot_total = DS4_VQ_CACHE_SLOTS;
        int32_t *a2cache = (int32_t *)malloc((size_t)n_active * sizeof(int32_t));
        if (!a2cache) { free(sel_h); free(e2slot); free(active_ids); free(dq_slot); free(dq_expert); return 0; }
        for (uint32_t i = 0; i < n_active; i++) {
            const uint32_t e = active_ids[i];
            int32_t hit = -1;
            for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++)
                if (g_vq_cache[layer_index].expert[s] == (int32_t)e) { hit = (int32_t)s; break; }
            if (hit < 0) {
                /* 选空槽, 没有则 LRU。本轮已占用的槽不可再被淘汰。 */
                int32_t victim = -1; uint64_t oldest = ~0ull;
                for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++) {
                    int taken = 0;
                    for (uint32_t j = 0; j < i; j++) if (a2cache[j] == (int32_t)s) { taken = 1; break; }
                    if (taken) continue;
                    if (g_vq_cache[layer_index].expert[s] < 0) { victim = (int32_t)s; break; }
                    if (g_vq_cache[layer_index].used[s] < oldest) { oldest = g_vq_cache[layer_index].used[s]; victim = (int32_t)s; }
                }
                if (victim < 0) { victim = (int32_t)i; }   /* 兜底(不应发生: n_active<=SLOTS) */
                g_vq_cache[layer_index].expert[victim] = (int32_t)e;
                dq_slot[n_dq] = (uint32_t)victim; dq_expert[n_dq] = e; n_dq++;
                hit = victim;
                g_vq_miss++;
            } else {
                g_vq_hit++;
            }
            a2cache[i] = hit;
            g_vq_cache[layer_index].used[hit] = ++g_vq_clock;
        }
        /* sel_h 里现在是"紧凑 active 序号", 改写成缓存槽号 */
        for (uint64_t k = 0; k < npair; k++)
            if (sel_h[k] >= 0) sel_h[k] = a2cache[sel_h[k]];
        free(a2cache);
    } else {
        /* 直通: 顺序占槽, 并让本层缓存失效(scratch 即将被覆盖) */
        if (layer_index < DS4_VQ_CACHE_LAYERS && g_vq_cache_ready)
            for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++) g_vq_cache[layer_index].expert[s] = -1;
        for (uint32_t i = 0; i < n_active; i++) { dq_slot[i] = i; dq_expert[i] = active_ids[i]; }
        n_dq = n_active;
    }
    free(e2slot);

    const uint64_t ge = (uint64_t)expert_mid_dim * expert_in_dim * 2u;
    const uint64_t de = (uint64_t)out_dim * expert_mid_dim * 2u;
    const uint64_t need_gu = (uint64_t)n_slot_total * ge, need_dn = (uint64_t)n_slot_total * de;
    /* scratch 上限护栏(与 Metal 的 DS4_VQ_SCRATCH_GB 同名同义)。prefill 大 chunk
     * 下活跃专家可能逼近 256 个 ⇒ 不设闸会直接吃光统一内存。 */
    static uint64_t vq_cap = 0;
    if (!vq_cap) { const char *e = ((const char *)0) /* DS4_VQ_SCRATCH_GB: 路径开关已删(2026-08-22 隐形炸弹清理) */; double g = e ? atof(e) : 32.0; vq_cap = (uint64_t)(g * 1073741824.0); }
    if (2 * need_gu + need_dn > vq_cap) {
        fprintf(stderr, "ds4: VQ gather scratch %.2fGB > cap (降 prefill chunk 或调 DS4_VQ_SCRATCH_GB)\n",
                (2.0 * need_gu + need_dn) / 1073741824.0);
        free(sel_h); free(active_ids); return 0;
    }
    if (!cache_on && !cuda_vq_ensure_scratch(need_gu, need_dn)) {
        fprintf(stderr, "ds4: [cuda-vq] scratch 分配失败 L%u n_active=%u 需要 %.2fGB\n",
                layer_index, n_active, (2.0 * need_gu + need_dn) / 1073741824.0);
        free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
    }

    /* 只有 CPU gather 路才需要全设备同步(CPU 要写跨层复用的托管 scratch, 而上一层
     * kernel 可能仍在读它 —— 托管内存被 CPU 写就是段错误, 不止是数据竞争; 这正是本
     * 实现第一版 SIGSEGV 的原因)。GPU dequant 路全在同一 stream 上顺序执行, 天然有序,
     * 多同步一次就是把 43 层的流水线全打断。 */
    {
        const char *cg0 = ((const char *)0) /* DS4_VQ_CPU_GATHER: 路径开关已删(2026-08-22 隐形炸弹清理) */;
        if (cg0 && cg0[0] == '1' && cudaDeviceSynchronize() != cudaSuccess) {
            (void)cudaGetLastError(); free(sel_h); free(active_ids); return 0;
        }
    }

    uint16_t *gbase, *ubase, *dbase;
    if (cache_on) {   /* 本层专属缓存显存 */
        gbase = (uint16_t *)g_vq_cache[layer_index].gate;
        ubase = (uint16_t *)g_vq_cache[layer_index].up;
        dbase = (uint16_t *)g_vq_cache[layer_index].down;
    } else {          /* 共享临时(prefill) */
        gbase = (uint16_t *)g_vq_gate_sc; ubase = (uint16_t *)g_vq_up_sc; dbase = (uint16_t *)g_vq_down_sc;
    }
    int nth = 0;
    /* DS4_VQ_CPU_GATHER=1 回 CPU 多线程 dequant(参考实现, 慢 ~10×), 用于对拍。 */
    const char *cg = ((const char *)0) /* DS4_VQ_CPU_GATHER: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (cg && cg[0] == '1') {
        if (cache_on) {   /* CPU gather 是对拍用的参考路, 不参与缓存 */
            fprintf(stderr, "ds4: DS4_VQ_CPU_GATHER 与 decode 缓存不兼容, 请同时设 DS4_VQ_NO_CACHE=1\n");
            free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
        }
        nth = 8;
        { const char *te = ((const char *)0) /* DS4_METAL_EXPERT_GATHER_THREADS: 路径开关已删(2026-08-22 隐形炸弹清理) */; if (te && atoi(te) > 0) nth = atoi(te); }
        if ((uint32_t)nth > n_active) nth = (int)n_active;
        if (nth > 32) nth = 32;
        if (nth < 1) nth = 1;
        volatile int gerr = 0;
        pthread_t th[32]; cuda_vq_gather_task tk[32];
        for (int i = 0; i < nth; i++) {
            tk[i].model_map = model_map; tk[i].blob = blob_host; tk[i].active_ids = active_ids;
            tk[i].gbase = gbase; tk[i].ubase = ubase; tk[i].dbase = dbase;
            tk[i].down_offset = down_offset; tk[i].down_expert_bytes = down_expert_bytes;
            tk[i].in = expert_in_dim; tk[i].mid = expert_mid_dim; tk[i].out_dim = out_dim;
            tk[i].lo = (uint32_t)((uint64_t)i * n_active / nth);
            tk[i].hi = (uint32_t)((uint64_t)(i + 1) * n_active / nth);
            tk[i].err = &gerr;
            pthread_create(&th[i], NULL, cuda_vq_gather_worker, &tk[i]);
        }
        for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
        if (gerr) { free(sel_h); free(active_ids); return 0; }
    } else {
        /* GPU dequant: 只解码缓存未命中的专家(直通模式下就是全部)。每矩阵一次 launch。 */
        for (uint32_t q = 0; q < n_dq; q++) {
            const uint32_t i = dq_slot[q];
            const uint32_t e = dq_expert[q];
            const uint64_t o1 = ds4vq_slot(blob_host, (int)e, 0);
            const uint64_t o3 = ds4vq_slot(blob_host, (int)e, 1);
            const uint64_t o2 = ds4vq_slot(blob_host, (int)e, 2);
            uint32_t rr, cc, dd_, nn, nb;
            if (!o1 || !o3 ||
                cuda_vq_pay_hdr(blob_host + o1, (int)expert_mid_dim, (int)expert_in_dim, &rr, &cc, &dd_, &nn, &nb) != 0) {
                fprintf(stderr, "ds4: [cuda-vq] e=%u w1/w3 槽缺失或头错 -- aborting\n", e);
                free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
            }
            vq_dequant_kernel<<<rr, 256>>>((__half *)(gbase + (uint64_t)i * expert_mid_dim * expert_in_dim),
                                           blob + o1, rr, cc, dd_, nn, nb);
            if (cuda_vq_pay_hdr(blob_host + o3, (int)expert_mid_dim, (int)expert_in_dim, &rr, &cc, &dd_, &nn, &nb) != 0) {
                fprintf(stderr, "ds4: [cuda-vq] e=%u w3 头错 -- aborting\n", e);
                free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
            }
            vq_dequant_kernel<<<rr, 256>>>((__half *)(ubase + (uint64_t)i * expert_mid_dim * expert_in_dim),
                                           blob + o3, rr, cc, dd_, nn, nb);
            __half *dst_d = (__half *)(dbase + (uint64_t)i * out_dim * expert_mid_dim);
            if (o2) {
                if (cuda_vq_pay_hdr(blob_host + o2, (int)out_dim, (int)expert_mid_dim, &rr, &cc, &dd_, &nn, &nb) != 0) {
                    fprintf(stderr, "ds4: [cuda-vq] e=%u w2 头错 -- aborting\n", e);
                    free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
                }
                vq_dequant_kernel<<<rr, 256>>>(dst_d, blob + o2, rr, cc, dd_, nn, nb);
            } else {
                /* 冷 w2 回退: base down 是影子张量时硬失败, 不读垃圾当权重(同 CPU 路)。 */
                if (down_expert_bytes == 0 || down_offset == 0) {
                    fprintf(stderr, "ds4: [cuda-vq] e=%u 冷 w2 槽缺失且 base down 不在文件里"
                                    "(影子张量) -- aborting (no silent quality downgrade)\n", e);
                    free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
                }
                const uint8_t *sd = (const uint8_t *)model_map + down_offset + (uint64_t)e * down_expert_bytes;
                vq_cold_w2_kernel<<<out_dim, 256>>>(dst_d, sd, out_dim, expert_mid_dim);
            }
        }
        if (!cuda_ok(cudaGetLastError(), "vq_dequant launch")) { free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0; }
    }
    free(active_ids); free(dq_slot); free(dq_expert);

    const char *vg = ((const char *)0) /* DS4_VQ_GPU: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    const int use_gpu = !(vg && vg[0] == '0');
    int ok = 0;
    if (!use_gpu) {
        /* CPU 参考路读的是 GPU 刚写完的 scratch ⇒ 必须先同步 */
        if (cudaDeviceSynchronize() != cudaSuccess) { (void)cudaGetLastError(); free(sel_h); return 0; }
        /* CPU 参考路: x/out 需在主机侧 */
        float *xh = (float *)malloc((size_t)n_tokens * expert_in_dim * sizeof(float));
        float *wh = (float *)malloc((size_t)npair * sizeof(float));
        float *oh = (float *)malloc((size_t)n_tokens * out_dim * sizeof(float));
        if (xh && wh && oh &&
            cudaMemcpy(xh, x->ptr, (size_t)n_tokens * expert_in_dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(wh, weights->ptr, (size_t)npair * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) {
            ok = cuda_vq_moe_cpu_ref(oh, gbase, ubase, dbase, xh, sel_h, wh,
                                     n_tokens, n_active, n_expert,
                                     expert_in_dim, expert_mid_dim, out_dim, clamp);
            if (ok) ok = (cudaMemcpy(out->ptr, oh, (size_t)n_tokens * out_dim * sizeof(float),
                                     cudaMemcpyHostToDevice) == cudaSuccess);
        }
        free(xh); free(wh); free(oh); free(sel_h);
        if (!ok) (void)cudaGetLastError();
        return ok;
    }

    /* GPU: remap 后的 slot 索引上传设备 */
    if (npair * sizeof(int32_t) > g_vq_sel_bytes) {
        if (g_vq_sel_dev) (void)cudaFree(g_vq_sel_dev);
        g_vq_sel_dev = NULL; g_vq_sel_bytes = 0;
        if (cudaMalloc((void **)&g_vq_sel_dev, npair * sizeof(int32_t)) != cudaSuccess) {
            (void)cudaGetLastError(); free(sel_h); return 0;
        }
        g_vq_sel_bytes = npair * sizeof(int32_t);
    }
    ok = (cudaMemcpy(g_vq_sel_dev, sel_h, npair * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    free(sel_h);
    if (!ok) { (void)cudaGetLastError(); return 0; }

    /* out 累加语义 ⇒ 先清零(kernel 用 atomicAdd 汇多个 pick) */
    if (cudaMemset(out->ptr, 0, (size_t)n_tokens * out_dim * sizeof(float)) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    /* 中间 h[n_tokens][n_expert][MID]: 复用调用方给的 mid scratch(容量口径与
     * routed_moe_launch 一致) */
    const uint64_t hneed = (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
    if (!mid_scratch || mid_scratch->bytes < hneed) {
        fprintf(stderr, "ds4: [cuda-vq] mid scratch %llu < 需要 %llu\n",
                (unsigned long long)(mid_scratch ? mid_scratch->bytes : 0), (unsigned long long)hneed);
        return 0;
    }
    const uint32_t zg = (expert_mid_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
    const uint32_t zd = (out_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
    vq_moe_gateup_kernel<<<dim3(n_tokens, n_expert, zg), 32 * DS4_VQ_WARPS_PER_BLOCK>>>(
        (float *)mid_scratch->ptr, (const __half *)gbase, (const __half *)ubase,
        (const float *)x->ptr, g_vq_sel_dev, (const float *)weights->ptr, n_expert, expert_in_dim, expert_mid_dim, clamp);
    vq_moe_down_kernel<<<dim3(n_tokens, 1, zd), 32 * DS4_VQ_WARPS_PER_BLOCK>>>(
        (float *)out->ptr, (const __half *)dbase, (const float *)mid_scratch->ptr,
        g_vq_sel_dev, (const float *)weights->ptr, n_expert, expert_mid_dim, out_dim);
    if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == (uint32_t)atoi(((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */) && n_tokens > 1) {
        (void)cudaDeviceSynchronize();
        FILE *f;
        float *tb = (float *)malloc((size_t)expert_in_dim * 4);
        if ((f = fopen("/tmp/vq_px.bin", "wb"))) { cudaMemcpy(tb, (const char *)x->ptr + (size_t)(n_tokens - 1) * expert_in_dim * 4, (size_t)expert_in_dim * 4, cudaMemcpyDeviceToHost); fwrite(tb, 4, expert_in_dim, f); fclose(f); }
        if ((f = fopen("/tmp/vq_pout.bin", "wb"))) { cudaMemcpy(tb, (const char *)out->ptr + (size_t)(n_tokens - 1) * out_dim * 4, (size_t)out_dim * 4, cudaMemcpyDeviceToHost); fwrite(tb, 4, out_dim, f); fclose(f); }
        free(tb);
    }
    if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */)
        fprintf(stderr, "ds4: [cuda-vq] L%u ntok=%u n_active=%u dq=%u cache=%d hit=%llu miss=%llu scratch=%.2fGB\n",
                layer_index, n_tokens, n_active, n_dq, cache_on,
                (unsigned long long)g_vq_hit, (unsigned long long)g_vq_miss,
                (2.0 * need_gu + need_dn) / 1073741824.0);
    return cuda_ok(cudaGetLastError(), "vq_moe_pair launch");
}

