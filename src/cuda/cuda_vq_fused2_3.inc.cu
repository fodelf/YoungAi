/* cuda_vq_fused2_3.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * VQ 专家前向入口 cuda_vq_moe_forward: decode(n_tokens ≤ fuse_max)走 fused/fused2 解码即乘核,
 * 大批转 cuda_vq_prefill.inc.cu 的 GEMM 路。(09-06 删老 prefill 路后不再是 >500 行 EXCEPTION。)
 */
static int16_t g_vq2_w2n[64];   /* 每层 fused2 特化判定: 0=未判定, -1=不可用, 256/512=w2 码本词数 */
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
        if (vcs_ == cudaStreamCaptureStatusNone)   /* 小批上限 8 token(投机 verify 批) */
            (void)cudaMalloc(&g_vq_partial, (size_t)8u * n_expert * out_dim * sizeof(float));
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
            fflush(stderr);
        }
    }

    /* ★fused2 特化探针必须在非 capture 时机跑(09-05 定罪)★: 原探针只在 n_tokens<=fuse_max 的
     * 解码分支里, prefill(n=26)不经过; token graph 开着时首次解码已在 capture 内, 探针被闸
     * ⇒ 表恒 0 ⇒ 整条解码路永远回老 fused 核(慢 18 ms/token, 且与直发归约序不同=逐位分叉)。
     * 现在每层第一次路过(prefill, 非 capture)就判定, 解码分支只读表。 */
    if (layer_index < 64u && !g_vq2_w2n[layer_index]) {
        cudaStreamCaptureStatus pcs0 = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs0);
        if (pcs0 == cudaStreamCaptureStatusNone) {
            static int32_t *pd0 = NULL;
            if (!pd0) (void)cudaMalloc(&pd0, sizeof(int32_t));
            int32_t pn = 0;
            if (pd0) {
                vq2_probe_kernel<<<1, 1>>>(blob, n_total_expert, expert_in_dim, expert_mid_dim, out_dim, pd0);
                if (cudaMemcpy(&pn, pd0, sizeof(pn), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    (void)cudaGetLastError(); pn = 0;
                }
            }
            g_vq2_w2n[layer_index] = pn ? (int16_t)pn : -1;
        }
    }
    const uint64_t npair = (uint64_t)n_tokens * n_expert;

    /* ---- decode 快路: 零 host 往返 ----
     * pair 数少时不去重, 每个 (token,pick) 各占一段 scratch, kernel 自己从 blob 读偏移。
     * selected 直接当 slot 用(第 k 个 pair 的权重就在第 k 段), 省掉 D2H + 去重 + H2D。 */
    /* decode 直通链常驻缓冲: 非 capture 时机预分配(prefill/首调都行) */
    /* 融合路上限 8 token(09-07; 此前 4): 投机 verify 批 2..8 token 走逐 (token,专家) 的 fused2(grid.x=token),
     * 更大批(prefill)走 dequant+cuBLAS。并集去重两版(shared 装 n 份激活 / L2 去重)实测都不比逐 token 快 —— 逐 token 核
     * 24 对 406 µs = 278 GB/s 已超 DRAM 墙, 说明相邻 token 同专家的位流本就在 L2 命中; 并集代码已删(fable5 09-07)。 */
    const uint32_t fuse_max = 8u;
    if (n_tokens <= fuse_max && n_tokens > 0) {
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
        /* 这里曾有 cudaMemsetAsync(out, 0): 两条融合路的 down 都写各自 partial 平面, 最后由
         * vq2_down_reduce_kernel 对每个 (token, o) 整值覆写 out, 清零从来没被读到。09-07 删:
         * 图里每层少一个 memset 节点(memset 节点会把 PDL 链切成普通全依赖)。 */
        /* fused2 特化闸(per-layer 一次判定, 判定在 prefill 首层调用=非 capture): 纯 vq4
         * 形态(w1/w3 nc512 + w2 nc256, 全槽在位, 4B 对齐)走高速特化路。 */
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
            /* 09-05: 512 词 w2 放行。旧闸"fuse2-512 反慢于 fused"的病根是 down 的 dot9 在
             * cols=2048 只有 16 条 lane 有段 + 激活 shared 读 32 路 bank 冲突, 已在 fused2_1
             * 重写中修掉(NIDX 模板化 + 转置布局)。形状假设写死: IN=4096(32 idx/lane)、
             * MID=2048(16 idx/lane), 其它形状回老 fused 路。 */
            const int16_t w2n_ = g_vq2_w2n[layer_index];
            use2 = (w2n_ == 256 || w2n_ == 512) && expert_in_dim == 4096u && expert_mid_dim == 2048u;
        }
        if (use2) {
            const uint32_t zg2 = (expert_mid_dim + DS4_VQ2_ROWS_PER_BLOCK - 1u) / DS4_VQ2_ROWS_PER_BLOCK;   /* 4 warps × R 行 */
            const uint32_t zd2 = (out_dim + DS4_VQ2_ROWS_PER_BLOCK - 1u) / DS4_VQ2_ROWS_PER_BLOCK;
            const uint32_t w2n = (uint32_t)g_vq2_w2n[layer_index];
            if (!g_vq_partial) goto vq2_skip;   /* capture 首层前必已分配 */
            /* shared = 转置激活 IN f32 + 两本码本 8 KB */
            ds4_launch_pdl(vq_moe_gateup_fused2_kernel, dim3(n_tokens, zg2, n_expert), 128, (size_t)expert_in_dim * 4 + 2u * 2048u * 2u, 0,
                (float *)mid_scratch->ptr, blob, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, (const float *)x->ptr,
                n_expert, expert_in_dim, expert_mid_dim, clamp);
            const size_t dsm = (size_t)expert_mid_dim * 4 + (size_t)w2n * 4u * 2u;   /* 转置激活 + 码本 */
            if (w2n == 512u)
                ds4_launch_pdl(vq_moe_down_fused2_kernel<512u>, dim3(n_tokens, zd2, n_expert), 128, dsm, 0, 
                    g_vq_partial, blob, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr, (const float *)mid_scratch->ptr,
                    n_expert, expert_mid_dim, out_dim);
            else
                ds4_launch_pdl(vq_moe_down_fused2_kernel<256u>, dim3(n_tokens, zd2, n_expert), 128, dsm, 0, 
                    g_vq_partial, blob, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr, (const float *)mid_scratch->ptr,
                    n_expert, expert_mid_dim, out_dim);
            ds4_launch_pdl(vq2_down_reduce_kernel, dim3((out_dim + 255u) / 256u, n_tokens, 1), 256, 0, 0, 
                (float *)out->ptr, g_vq_partial, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, n_expert, out_dim);
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
        if (!g_vq_partial) { (void)cudaGetLastError(); return 0; }
        vq_moe_down_fused_kernel<<<dim3(n_tokens, zd, n_expert), 32 * DS4_VQ_WARPS_PER_BLOCK,
            (size_t)DS4_VQ_WARPS_PER_BLOCK * DS4_VQ_BITWORDS * sizeof(uint32_t)
            + 2048u * sizeof(__half)>>>(
            g_vq_partial, blob, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, (const float *)mid_scratch->ptr,
            down_base, down_expert_bytes, n_expert, expert_mid_dim, out_dim);
        ds4_launch_pdl(vq2_down_reduce_kernel, dim3((out_dim + 255u) / 256u, n_tokens, 1), 256, 0, 0, 
            (float *)out->ptr, g_vq_partial, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, n_expert, out_dim);
        return cuda_ok(cudaGetLastError(), "vq fused launch");
    }


    /* prefill / 大批(n_tokens > fuse_max): 按专家排序 + 逐专家 dequant + cuBLAS f16 GEMM
     * (cuda_vq_prefill.inc.cu)。09-06 前这里是"全层活跃专家 dequant 落 12.9 GB scratch + 逐
     * (token,pick) warp 核"的老路: 8192 token 60 万次 dequant + 44 GFLOPS 的算力, prefill 20 t/s。 */
    /* ver = 2: 这是 V4 的合一 VQ 文件路(DQVL v2), 与 V4.1 方案 v3 的 blob 不同族 —— 写死 2 是事实陈述不是默认值,
     * 真给它一份 v3 blob 的话 vqp_hdr_build 会在魔数上硬停(安全失败), 不会按 v2 布局解出假权重。 */
    return cuda_vq_moe_prefill_gemm(out, blob, model_map, down_offset, down_expert_bytes,
                                    expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                                    n_total_expert, n_expert, clamp, x, layer_index, n_tokens, 2u);
}
