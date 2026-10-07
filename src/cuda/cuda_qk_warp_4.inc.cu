/* cuda_qk_warp_4.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * q4_K/q2_K warp gemv kernel 族与 dense 入口。
 */
int ds4_gpu_matmul_q2_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    /* dense Q2_K(2026-08-19): q4_K 入口克隆, 共用 q8_K 激活量化 scratch。 */
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t w_bytes = row_bytes * out_dim;
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "dense_q2k");
    if (!w) return 0;
    const uint64_t xq_need = n_tok * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;   /* capture 中不能 cudaMalloc */
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    const int fused = (n_tok == 1);   /* 碎片税刀①: decode 融合量化免独立发射(原 DS4_Q2K_FUSEQ 旋钮, 定死开) */
    if (!fused)
        ds4_launch_pdl(q8_K_quantize_kernel, dim3(blocks, (unsigned)n_tok, 1), 256, 0, g_cur_stream, 
            (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    unsigned gx = (unsigned)((out_dim + 7u) / 8u);
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
    size_t q2sh = (blocks <= 32u && (blocks & 3u) == 0u)
                        ? (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4) : 0u;
    if (fused) q2sh += (size_t)blocks * sizeof(cuda_block_q8_K);
    const uint32_t q2lv = 2u;    /* staging 档(原 DS4_Q2K_STAGE 旋钮, 定死 2) */
    /* 刀⑤已回滚(2026-08-20): blocks≤4 rowlane 实测 33.52→30.45 且行为复读(疑数值), 撤。 */
    const uint32_t q2mb = 16u;   /* 半块/整块分界(原 DS4_Q2K_SPLIT_MAXBLK) */
    /* cp.async 双缓冲判负存档(原 DS4_Q2K_CPASYNC, 定死关): 实测 7192→7646 µs 反向(混合访问实际墙 ~190, 非延迟问题) */
    const uint32_t q2ca = 0u;
    if (q2ca && !fused && q2lv >= 2u && blocks <= 32u && (blocks & 3u) == 0u) {
        /* dot微架构刀①: cp.async 双缓冲(shared ×2), 数学与 staged 路逐位一致 */
        const size_t cash = (size_t)16u * (blocks * 84u / 16u) * sizeof(uint4);
        matmul_q2_K_warp_ca_kernel<<<dim3(gx, (unsigned)n_tok, 1), 256, cash, g_cur_stream>>>(
            (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
            row_bytes, blocks, (uint32_t)out_dim, q2mb);
        return cuda_ok(cudaGetLastError(), "dense q2_K cpasync matmul launch");
    }
    /* 大批走分片 GEMM(2026-08-21): 激活按 tile 进 shared, 块内 8 warp 共用。
     * 门限 16 token: 小批(decode/verify)继续走老路 —— 那条已与 decode
     * 逐位对齐, 不动。fused(n_tok==1 融合量化)与非 staging 档也不走。 */
    const uint32_t tmin = 16u;
    if (!fused && q2lv && n_tok >= tmin && blocks <= 32u && (blocks & 3u) == 0u) {
        const size_t wsh_bytes = (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4);
        const size_t budget = 96u * 1024u;
        uint32_t tile = (uint32_t)((budget - wsh_bytes) / ((size_t)blocks * sizeof(cuda_block_q8_K)));
        if (tile > n_tok) tile = (uint32_t)n_tok;
        if (tile > 64u) tile = 64u;
        if (tile >= 2u) {
            const size_t sh = wsh_bytes + (size_t)tile * blocks * sizeof(cuda_block_q8_K);
            if (q2k_tiled_set_smem(blocks, (int)sh)) {
                unsigned gx = (unsigned)((out_dim + 7u) / 8u);
                if (gx > 512u) gx = 512u;   /* 分片核网格封顶(原 DS4_Q2K_TILED_BLOCKS 旋钮) */
                q2k_tiled_launch(gx, sh, (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
                                 row_bytes, blocks, (uint32_t)out_dim, (uint32_t)n_tok, tile, q2lv, q2mb);
                return cuda_ok(cudaGetLastError(), "dense q2_K tiled matmul launch");
            }
        }
    }
    /* 权重驻留批: 多 token 时 grid.y=1, 块内循环 token(权重只读一遍)。 */
    /* 门限(2026-08-21): 权重驻留省带宽但把 grid.y 折成 1 ⇒ 小矩阵并行度不足
     * (out=512 只剩 64 block / 48 SM)。仅当行块数够铺满设备(SM 数 × 4 驻留块, 48 SM 上 = 192)时才驻留。 */
    const uint32_t wsmin = ds4_sm_count() * 4u;
    const uint32_t wstat = (n_tok > 1u && gx >= wsmin) ? (uint32_t)n_tok : 1u;
    /* 驻留批解除 384 封顶(2026-08-21): 该上限是 decode(grid.y=n_tok 已铺满)的在飞调优;
     * 驻留批 grid.y=1 时封顶 ⇒ 每 warp 1.33 行的尾波(x16 kernel 精确覆盖零尾波是其
     * 126 vs 52 GB/s 的差之一)。 */
    if (wstat > 1u) gx = (unsigned)((out_dim + 7u) / 8u);
    /* 分组数: 让 gx×组数 ≥ 目标块数(SM 数 × 4 块 × 5 波, 48 SM 上 = 1024), 组内仍权重驻留。 */
    const uint32_t wsblk = ds4_sm_count() * 20u;
    uint32_t ygroups = 1u, tpg = wstat;
    if (wstat > 1u && gx < wsblk) {
        ygroups = (wsblk + gx - 1u) / gx;
        if (ygroups > wstat) ygroups = wstat;
        tpg = (wstat + ygroups - 1u) / ygroups;
        ygroups = (wstat + tpg - 1u) / tpg;
    }
    /* K 切分(原 DS4_Q2K_KSPLIT 旋钮, 定死 1 = 不切): 仅在块数不足(小 out_dim)且 blocks 够分时启用; 阈值 1024 块。 */
    const uint32_t ksp_max = 1u, ksp_min_blocks = 1024u;
    uint32_t ks = 1u;
    float *part = NULL;
    /* 只对走"整块 else 分支"的矩阵切 K(blocks > split_maxblk, 即 o_b 的 32 块):
     * 半块/八分拆路径的 lane 映射覆盖全部块, 分段后每个 z 会重复算全行 → 结果被乘
     * ksplit(2026-08-21 质量闸抓到, Δlogit 均值 14.3)。 */
    if (ksp_max > 1u && blocks > q2mb && gx < ksp_min_blocks && !fused) {
        ks = (ksp_min_blocks + gx - 1u) / gx;
        if (ks > ksp_max) ks = ksp_max;
        while (ks > 1u && (blocks / ks) < 4u) ks--;
        if (ks > 1u) {
            const uint64_t need = (uint64_t)ks * n_tok * out_dim * sizeof(float);
            cudaStreamCaptureStatus pcs2 = cudaStreamCaptureStatusNone;
            (void)cudaStreamIsCapturing(0, &pcs2);
            if (pcs2 != cudaStreamCaptureStatusNone) { ks = 1u; }
            else {
                if (need > g_q2k_part_bytes) {
                    if (g_q2k_part) (void)cudaFree(g_q2k_part);
                    g_q2k_part = NULL; g_q2k_part_bytes = 0;
                    if (cudaMalloc(&g_q2k_part, need) != cudaSuccess) { (void)cudaGetLastError(); ks = 1u; }
                    else g_q2k_part_bytes = need;
                }
                part = (float *)g_q2k_part;
                if (!part) ks = 1u;
            }
        }
    }
    /* 批激活也进 shared: 权重 staging 之后再放 tpg 份 q8 激活。超 48KB 要 opt-in,
     * 失败就退回全局读(数值一致, 只是慢)。 */
    const uint32_t tpg_use = (wstat > 1u) ? tpg : 1u;
    uint32_t xsm = 0u;
    size_t sh_use = (q2lv || fused) ? q2sh : 0u;
    if (!fused && q2lv && tpg_use > 1u &&
        blocks <= 32u && (blocks & 3u) == 0u) {
        const size_t xbytes = (size_t)tpg_use * blocks * sizeof(cuda_block_q8_K);
        const size_t want = q2sh + xbytes;
        if (want <= 200u * 1024u && q2k_set_dynamic_smem(blocks, (int)want)) {
            sh_use = want; xsm = 1u;
        }
    }
    q2k_warp_launch(dim3(gx, wstat > 1u ? ygroups : (unsigned)n_tok, ks), sh_use,
        (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
        row_bytes, blocks, (uint32_t)out_dim, q2lv, q2mb,
        fused ? (const float *)x->ptr : NULL, tpg_use, (uint32_t)n_tok, part, ks, xsm);
    if (ks > 1u) {
        const uint64_t n_elem = (uint64_t)out_dim * n_tok;
        q2k_ksplit_reduce_kernel<<<(unsigned)((n_elem + 255u) / 256u), 256, 0, g_cur_stream>>>(
            (float *)out->ptr, part, (uint32_t)out_dim, (uint32_t)n_tok, ks);
    }
    return cuda_ok(cudaGetLastError(), "dense q2_K matmul launch");
}

/* q2_K 版 decode 同输入矩阵对(2026-08-20 提速: quantize 风暴去重): 一次 q8_K 量化
 * 共享 scratch + 两次矩阵发射(q_a+kv / shexp gate+up 每层省一次量化+发射)。 */
static int q2k_matmul_from_xq(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                              uint64_t weight_offset, uint32_t blocks, uint64_t out_dim,
                              const float *xraw) {
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t w_bytes = row_bytes * out_dim;
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "dense_q2k_pair");
    if (!w) return 0;
    unsigned gx = (unsigned)((out_dim + 7u) / 8u);
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
    const uint32_t q2lv = 2u, q2mb = 16u;   /* 与单矩阵入口同一档(staging 2 / 半块整块分界 16) */
    size_t q2sh = (blocks <= 32u && (blocks & 3u) == 0u)
                        ? (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4) : 0u;
    if (xraw) q2sh += (size_t)blocks * sizeof(cuda_block_q8_K);
    q2k_warp_launch(dim3(gx, 1, 1), (q2lv || xraw) ? q2sh : 0u,
        (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
        row_bytes, blocks, (uint32_t)out_dim, q2lv, q2mb, xraw, 1u, 1u, NULL, 1u, 0u);
    return cuda_ok(cudaGetLastError(), "dense q2_K pair matmul launch");
}

int ds4_gpu_matmul_q2_K_pair_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x) {
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || out0_dim == 0 || out1_dim == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    if (x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) || out1->bytes < out1_dim * sizeof(float)) return 0;
    const uint64_t xq_need = (uint64_t)blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    /* 融合量化定死开(原 DS4_Q2K_FUSEQ 旋钮): 两矩阵各自在核内量化 x, 不发独立的 q8_K_quantize_kernel */
    const float *xr = (const float *)x->ptr;
    return q2k_matmul_from_xq(out0, model_map, model_size, off0, blocks, out0_dim, xr) &&
           q2k_matmul_from_xq(out1, model_map, model_size, off1, blocks, out1_dim, xr);
}

/* decode 同输入矩阵对: 一次 q8_K 量化 + 一次发射覆盖两矩阵 */
int ds4_gpu_matmul_q4_K_pair_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x) {
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || out0_dim == 0 || out1_dim == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q4_K);
    if (off0 > model_size || row_bytes * out0_dim > model_size - off0 ||
        off1 > model_size || row_bytes * out1_dim > model_size - off1) return 0;
    if (x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) return 0;
    const char *w0 = cuda_model_range_ptr(model_map, off0, row_bytes * out0_dim, "dense_q4k_p0");
    const char *w1 = cuda_model_range_ptr(model_map, off1, row_bytes * out1_dim, "dense_q4k_p1");
    if (!w0 || !w1) return 0;
    const uint64_t xq_need = (uint64_t)blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;   /* capture 中不能 cudaMalloc */
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    ds4_launch_pdl(q8_K_quantize_kernel, dim3(blocks, 1, 1), 256, 0, g_cur_stream,
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, 1u);
    if (q4k_tile_supported(blocks, (uint32_t)out0_dim) && q4k_tile_supported(blocks, (uint32_t)out1_dim))
        return q4k_tile_pair_launch((float *)out0->ptr, (float *)out1->ptr, w0, w1,
                                    (const cuda_block_q8_K *)g_q4k_xq_sc, blocks,
                                    (uint32_t)out0_dim, (uint32_t)out1_dim);
    unsigned gx = (unsigned)((out0_dim + out1_dim + 7u) / 8u);
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
    const size_t shmem = (blocks <= 32u) ? (size_t)8u * blocks * 9u * sizeof(uint4) : 0;
    ds4_launch_pdl(matmul_q4_K_pair_warp_kernel, gx, 256, shmem, g_cur_stream, 
        (float *)out0->ptr, (float *)out1->ptr, w0, w1,
        (const cuda_block_q8_K *)g_q4k_xq_sc, row_bytes, blocks,
        (uint32_t)out0_dim, (uint32_t)out1_dim);
    return cuda_ok(cudaGetLastError(), "matmul_q4_K_pair launch");
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

