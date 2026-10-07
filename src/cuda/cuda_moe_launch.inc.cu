/* cuda_moe_launch.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * routed_moe_launch 巨型分发函数。
 * EXCEPTION: 单函数体 >500 行, 无安全切割点, 整体成片(884 行)。
 */
static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    const int q2k_path = (gate_type == 10u && down_type == 10u);   /* 全 Q2_K 专家 */
    {   static int once = 0;
        if (!once++) {
            fprintf(stderr, "ds4: [moe-init] gate_type=%u down_type=%u q2k=%d q4k=%d "
                            "gate_ebytes=%llu gate_rbytes=%llu down_ebytes=%llu down_rbytes=%llu "
                            "IN=%u MID=%u OUT=%u ntok=%u nexp=%u\n",
                    gate_type, down_type, q2k_path, q4k_path,
                    (unsigned long long)gate_expert_bytes, (unsigned long long)gate_row_bytes,
                    (unsigned long long)down_expert_bytes, (unsigned long long)down_row_bytes,
                    expert_in_dim, expert_mid_dim, out_dim, n_tokens, n_expert);
            fflush(stderr);
        }
    }
    if (!q4k_path && !q2k_path && (gate_type != 16u || down_type != 10u)) return 0;
    if (q4k_path && n_expert != 6u) return 0;   /* q4k batch(dspark drafter FFN): 朴素批链, sorted/tile(IQ2 专用)旁路 */
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_sorted_pairs = n_tokens > 1u;
        const uint32_t use_expert_tiles = use_sorted_pairs;
        const uint32_t expert_tile_m = 8u;   /* 专家瓦片 M(原 DS4_CUDA_MOE_TILE4 旋钮, 定死 8) */
        const uint32_t write_gate_up = 0;
        const uint32_t use_p2_sorted = use_sorted_pairs;
        /* ★MoE down 原子累加整族删除(2026-08-22)★ —— 连同它的两个 env 开关一起删, 不留旋钮。
         * 原判据 n_tokens>=128 让**批量 prefill(chunk=256)走 atomicAdd**、decode(n=1)走独立
         * 平面+固定序归约。多专家原子累加同一输出行 ⇒ fp32 加序随调度漂 ⇒ 同一 prompt 两次
         * 跑结果不同。实测(同二进制同 ids 连跑两次, 批量路):
         *     L8  raw_ffn_in 中位相对差  6.91%   路由 top6 相同 71.5%
         *     L16                       16.56%                  40.7%
         *     L32                       16.44%                  46.9%
         * 即中层之后一半以上的 token 选中不同的专家; 而解码路两次跑全层 100% 逐位一致 ——
         * 一条判据同时解释了"prefill 不确定 / decode 确定"。
         * 后果: (1) 同 prompt 无唯一输出, 与"默认路径=裸模型真值"直接冲突;
         *       (2) 一切 prefill 跑分不可复现; (3) 用捕获做拟合的工作全部作废
         *       (反修整轮就是这么废的: 解算器拟合的 x 与引擎部署的 x 根本不是一条轨迹)。
         * 代价: prefill 149.03 → 122.67 t/s (−17.7%)。按"删除而非默认关闭"处理, 不留开关。 */
        const uint32_t use_atomic_down = 0u;
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (0 ||
             0 ||
             0 ||
             (n_tokens >= 128u &&
              1));
        const uint32_t use_down_tile16 = use_atomic_down && expert_tile_m == 8u &&
            n_tokens >= 128u;
        /* decode LUT gate 是 IQ2_XXS 专用(用它的 256 项 grid 预表), Q2_K 没有 grid,
         * 必须绕开走 q2k 专用 kernel。 */
        const uint32_t use_decode_lut_gate =
            !q2k_path && n_tokens == 1u && xq_blocks <= 16u &&
            1;
        const uint32_t gate_row_span =
            0 ? 512u :
            0 ? 2048u : 1024u;
        const uint32_t down_row_span =
            0 ? 512u :
            0 ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_atomic_down && expert_tile_m == 8u &&
            (0 ||
             0 ||
             0 ||
             0 ||
             (use_down_tile16 &&
              1));
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u &&
            1;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        ds4_launch_pdl(q8_K_quantize_kernel, xq_grid, 256, 0, 0, xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = 256ull * sizeof(uint32_t);
            const uint64_t offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t cursors_bytes = 256ull * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + 256u;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + 256u) : 0u;
            const uint64_t tile_offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? 257ull * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                ok = cuda_ok(cudaMemsetAsync(counts, 0, counts_bytes, g_cur_stream), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile_experts, tile_starts, tile_offsets, counts, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            /* tiles 版 gate/up kernel 全部硬编码 cuda_block_iq2_xxs, Q2_K 走下面的
             * qwarp32 适配分支; tile 数据结构照常构建(down 侧还要用)。 */
            if (ok && !q2k_path && !q4k_path && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u && xq_blocks == 16u && n_tokens <= 8u &&
                           1) {
                    /* verify/draft 小批: x16 结构 + expert-tile 权重驻留 */
                    const uint32_t xt_ycap = pair_count < 16u ? (pair_count > 0u ? pair_count : 1u) : 16u;
                    dim3 xtgrid((expert_mid_dim + 7u) / 8u, xt_ycap, 1);
                    moe_gate_up_mid_iq2_x16_tile_kernel<<<xtgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, expert_mid_dim, n_expert,
                        write_gate_up, clamp, 0 ? 1u : 0u);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity < pair_count + 1u ? tile_capacity : pair_count + 1u, 1);
                    /* 小批(verify/draft)走无 smem 变体: sxq[8][16]=36.5KB 静态共享压死占用率
                     * (2026-08-21 gpu_span 归因: kernel 4.7x 偏离物理); 激活 28KB 天然驻 L2。
                     * 大批(prefill np≥8)保留 smem 变体。DS4_MOE_TILE_SMEM=1 强制旧路。 */
                    if (n_tokens > 8u) {
                        moe_gate_up_mid_expert_tile8_row32_kernel<true><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        moe_gate_up_mid_expert_tile8_row32_kernel<false><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity < pair_count + 1u ? tile_capacity : pair_count + 1u, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && !q2k_path && !q4k_path && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && !q2k_path && !q4k_path && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (q4k_path) {
                    /* 09-07: tile 结构核(cuda_moe_q4k_tile), 形状不符(-1)回老核; 必须整段留在 if(q4k_path) 里 ——
                     * 首版把它拆成独立 if 再接原 else-if 链, tile 发射后又落进 IQ2 分支把 Q4_K 权重当 IQ2 算, 草稿全烂
                     * (接受率 0.90 → 0.10)。 */
                    const int r = moe_gate_up_q4k_tile_launch((float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                                              gate_w, up_w, xq, (const int32_t *)selected->ptr,
                                                              (const float *)weights->ptr, gate_expert_bytes, xq_blocks,
                                                              expert_mid_dim, n_expert, n_tokens, write_gate_up, clamp);
                    if (r == 0) ok = false;
                    if (r < 0)   /* decode 与 batch 同 kernel(pair=blockIdx.y 天然批) */
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate && xq_blocks == 16u) {
                    dim3 ixgrid(n_tokens, (expert_mid_dim + 7u) / 8u, n_expert);
                    moe_gate_up_mid_iq2_x16_kernel<<<ixgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (q2k_path && xq_blocks == 16u) {
                    dim3 q2grid(n_tokens, (expert_mid_dim + 7u) / 8u, n_expert);
                    moe_gate_up_mid_q2k_x16_kernel<<<q2grid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else if (q2k_path) {
                    /* q2k 泛化配方(xq_blocks != 16): 原合体 kernel */
                    dim3 q2grid(n_tokens, (expert_mid_dim + 7u) / 8u, n_expert);
                    moe_gate_up_mid_q2k_qwarp32_kernel<<<q2grid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            ds4_launch_pdl(q8_K_quantize_kernel, midq_grid, 256, 0, 0, midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            {   /* partial 缓冲预分配: 必须发生在非 capture 调用(prefill)里, decode 的
                 * graph capture 期间 cudaMalloc 非法且录进 graph 的分支永久定型 */
                const uint64_t pneed0 = 6ull * out_dim * sizeof(float);
                if (pneed0 > g_down_partial_bytes) {
                    cudaStreamCaptureStatus cs0 = cudaStreamCaptureStatusNone;
                    (void)cudaStreamIsCapturing(0, &cs0);
                    if (cs0 == cudaStreamCaptureStatusNone) {
                        if (g_down_partial) (void)cudaFree(g_down_partial);
                        g_down_partial = NULL; g_down_partial_bytes = 0;
                        if (cudaMalloc(&g_down_partial, pneed0) == cudaSuccess) g_down_partial_bytes = pneed0;
                        else (void)cudaGetLastError();
                    }
                }
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    const uint64_t pneed = 6ull * out_dim * sizeof(float);
                    if (g_down_partial && pneed <= g_down_partial_bytes) {
                        dim3 pgrid((out_dim + 31u) / 32u, 6, 1);
                        moe_down_partial_qwarp32_kernel<<<pgrid, 256>>>(
                            g_down_partial,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                        moe_down_partial_reduce6_kernel<<<(out_dim + 255u) / 256u, 256>>>(
                            (float *)out->ptr, g_down_partial, out_dim);
                    } else {
                        moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    }
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (!q4k_path && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u && midq_blocks == 8u && (out_dim & 31u) == 0u &&
                           n_tokens <= 8u) {
                    dim3 dtg((out_dim + 31u) / 32u,
                             down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile_qwarp32_kernel<<<dtg, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts,
                        down_expert_bytes, down_row_bytes, midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (!q4k_path && sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (!q4k_path && sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else if (q4k_path && midq_blocks == 8u && (out_dim & 3u) == 0u) {
                /* 09-07: tile 结构核(cuda_moe_q4k_tile); 形状不符走下面老核 */
                if (moe_down_q4k_tile_launch((float *)down->ptr, down_w, midq, (const int32_t *)selected->ptr,
                                             down_expert_bytes, midq_blocks, out_dim, n_expert, n_tokens) <= 0) ok = false;
            } else if (q4k_path) {
                moe_down_q4K_pairs_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (ok && !use_atomic_down && !use_direct_down_sum6) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

