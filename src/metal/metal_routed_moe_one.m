/* metal_routed_moe_one.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const ds4_gpu_residual_set *residual,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const ds4_gpu_residual_set *go1b_res =
        (residual && residual->gate_ptr) ? residual : NULL;   /* go1b residual CPU pointers */
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        /* 合一 VQ GGUF: base gate/up 张量不在文件里(offset=0/bytes=0), do_vq 路
         * 从 blob+down 取数, gate/up view 不建也不引用 — 建 0 区间 view 会硬失败。 */
        const int vq_no_base_gate =
            (go1b_res && go1b_res->vq && gate_expert_bytes == 0);
        id<MTLBuffer> gate_buf = nil, up_buf = nil, down_buf = nil;
        if (!vq_no_base_gate) {
            gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
            up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
        }
        /* ★冷 w2 也在 blob 时 down 是 bytes=0 影子张量(2026-08-01): --no-down 合并的文件里
         * base ffn_down_exps 不存在(省 11.42 GiB), 引擎侧由 routed_down_shadow() 顶住维度,
         * 真实字节从 blob 的 which=2 槽取 ⇒ 这里不该再要 down_buf, 也不该判它 nil 为失败。 */
        /* 判据用 offset 而非 bytes: 影子张量的维度是真的(routed_down_shadow 填了 DS4 常量),
         * 所以 down_tensor_bytes 按维度算出来非零; 真正的标志是 abs_offset==0(文件里没有它)。*/
        const int vq_no_base_down = (down_offset == 0);
        if (!vq_no_base_down)
            down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
        if ((!vq_no_base_gate && (!gate_buf || !up_buf)) || (!vq_no_base_down && !down_buf)) {
            fprintf(stderr, "ds4: [moe-buf-nil] L%u gate=%d up=%d down=%d vq_no_base_gate=%d\n",
                    layer_index, gate_buf != nil, up_buf != nil, down_buf != nil, (int)vq_no_base_gate);
            return 0;
        }
        uint32_t source_n_total_expert = n_total_expert;

        /* Expert offload is meant for the full q2 target model, whose routed
         * experts are deliberately non-resident and must be gathered into
         * compact scratch/pool before the MoE kernels index them.  A fully
         * resident support model would just re-copy resident bytes through the
         * A3 scratch path, so non-q2 routed tensors stay on the direct
         * resident-buffer path. */
        const bool a3_expert_offload =
            ds4_gpu_expert_offload_enabled() &&
            gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
            down_type == DS4_METAL_TENSOR_Q2_K;
        if (a3_expert_offload) {
            g_expert_gather_nocache_call = 0;   /* decode reads stay page-cache friendly */
            const int was_batched = (g_batch_cb != nil);
            if (was_batched) {
                if (ds4_gpu_expert_drain_commands("routed MoE drain") == 0) return 0;
            }
            /* P2.1: snapshot this layer's router input and kick next-layer
             * router prediction + expert read-ahead on the background thread;
             * it overlaps both this layer's gather and the GPU compute.  The
             * batch was just drained, so x is CPU-visible and final. */
            if (ds4_gpu_expert_prefetch_enabled() &&
                xbuf.storageMode == MTLStorageModeShared) {
                const float *x_cpu = (const float *)((const uint8_t *)xbuf.contents +
                                                     (size_t)ds4_gpu_tensor_offset(x));
                ds4_gpu_expert_prefetch_enqueue(layer_index + 1u, x_cpu, expert_in_dim);
            }
            uint32_t active_ids[DS4_METAL_ACTIVE_EXPERTS_MAX];
            uint32_t n_active = 0;
            int compact_ok = ds4_gpu_compact_selected_experts(selectedbuf,
                                                              selected_off,
                                                              n_expert,
                                                              n_total_expert,
                                                              active_ids,
                                                              DS4_METAL_ACTIVE_EXPERTS_MAX,
                                                              &n_active);
            /* Compare this layer's actual active set against the latest
             * prediction made for it (stats feed the ds4-io pf= field). */
            if (compact_ok) {
                ds4_gpu_expert_prefetch_note_actual(layer_index, active_ids, n_active);
            }
            /* Real expert pool is currently limited to the full q2 routed layout.
             * MTP/q4 routed tensors can have different slot sizes and fall back to
             * the proven A3 scratch path. */
            int pool_used = 0;
            if (compact_ok && gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                pool_used = ds4_gpu_try_load_layer_experts_to_pool(model_map,
                                                                   layer_index,
                                                                   selectedbuf,
                                                                   selected_off,
                                                                   n_expert,
                                                                   n_active,
                                                                   active_ids,
                                                                   gate_offset,
                                                                   up_offset,
                                                                   down_offset,
                                                                   gate_expert_bytes,
                                                                   down_expert_bytes,
                                                                   n_total_expert,
                                                                   &gate_buf,
                                                                   &up_buf,
                                                                   &down_buf,
                                                                   &source_n_total_expert);
            }
            int load_ok = compact_ok && (pool_used ||
                          ds4_gpu_load_layer_experts_to_scratch(model_map,
                                                                 layer_index,
                                                                 n_active,
                                                                 active_ids,
                                                                 gate_offset,
                                                                 up_offset,
                                                                 down_offset,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 n_total_expert));
            if (!load_ok) {
                if (was_batched) (void)ds4_gpu_begin_commands();
                return 0;
            }
            if (!pool_used) {
                gate_buf = g_moe_scratch_gate;
                up_buf = g_moe_scratch_up;
                down_buf = g_moe_scratch_down;
                source_n_total_expert = n_active;
            }
            gate_inner = 0;
            up_inner = 0;
            down_inner = 0;
            if (was_batched && ds4_gpu_begin_commands() == 0) return 0;
        }

        const uint32_t n_tokens = 1;
        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, source_n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, source_n_total_expert,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        /* 1-bit residual is applied in the batch (mm_id) decode path; go1b decode does
         * not use this one_tensor path, so no residual is applied here. */
        (void)go1b_res;

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        int ok = 1;
        id<MTLComputePipelineState> pair_swiglu_pipeline = nil;
        if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
        } else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
        }
        /* 非融合路仍可达: --quality 或该 quant 无 pair_swiglu kernel 时走下面的
         * 分步编码, 不是死分支。 */
        const bool fuse_pair_swiglu =
            !g_quality_mode &&
            pair_swiglu_pipeline != nil;
        if (fuse_pair_swiglu) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_id_pair_swiglu(cb,
                                                        pair_swiglu_pipeline,
                                                        &gate_args,
                                                        &act_args,
                                                        gate_buf,
                                                        (NSUInteger)gate_inner,
                                                        up_buf,
                                                        (NSUInteger)up_inner,
                                                        xbuf,
                                                        ds4_gpu_tensor_offset(x),
                                                        gatebuf,
                                                        ds4_gpu_tensor_offset(gate),
                                                        upbuf,
                                                        ds4_gpu_tensor_offset(up),
                                                        midbuf,
                                                        ds4_gpu_tensor_offset(mid),
                                                        selectedbuf,
                                                        ds4_gpu_tensor_offset(selected),
                                                        weightsbuf,
                                                        ds4_gpu_tensor_offset(weights),
                                                        gate_smem,
                                                        2,
                                                        false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_iq2_xxs_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_Q4_K &&
                   g_moe_mul_mv_id_q4_k_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_q4_k_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            gate_buf,
                                            (NSUInteger)gate_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            gatebuf,
                                            ds4_gpu_tensor_offset(gate),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            up_buf,
                                            (NSUInteger)up_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            upbuf,
                                            ds4_gpu_tensor_offset(up),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false);
        }
        if (ok && !fuse_pair_swiglu) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    false);
        }

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            n_expert == 6 &&
            n_tokens == 1 &&
            down_sum6_pipeline != nil;
        if (ok && direct_down_sum) {
            ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                 down_sum6_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2);
        } else if (ok) {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                 down_mv_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 down_dst,
                                                 down_dst_off,
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2,
                                                 false);
        }
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed tensor MoE")) return 0;
    }

    return 1;
}
