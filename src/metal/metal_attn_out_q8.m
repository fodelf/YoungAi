/* metal_attn_out_q8.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !group_tmp || !low_tmp || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || (low_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output batch received invalid q8 dimensions\n");
            return 0;
        }
        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t row_b_bytes = (low_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        const uint64_t out_b_bytes = out_dim * row_b_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
            out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) {
            fprintf(stderr, "ds4: Metal attention output batch weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_tokens * n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_tokens * low_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal attention output batch received undersized buffers\n");
            return 0;
        }
        (void)group_tmp;
        (void)low_tmp;

        const bool use_direct_low =
            n_tokens < 32u && getenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == NULL;
        /* The exported TensorOps attention-output kernel is a 64-token tile.
         * Keep this on full tiles only; smaller multiples of 32 use the legacy
         * path instead of relying on cooperative tensor partial RHS bounds. */
        const bool use_mpp_low =
            n_tokens >= 32u &&
            (n_tokens % DS4_METAL_ATTN_OUT_MPP_TILE_N) == 0 &&
            ds4_gpu_use_mpp_attn_out_low_matmul();
        const NSUInteger ids_bytes = (NSUInteger)n_tokens * (NSUInteger)n_groups * sizeof(int32_t);
        id<MTLBuffer> group_ids_buffer = nil;
        if (!use_direct_low && !use_mpp_low) {
            if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                group_ids_buffer =
                    ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                if (!group_ids_buffer) {
                    return 0;
                }
            } else {
                if (!ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                     &g_attn_out_group_ids_bytes,
                                                     ids_bytes,
                                                     "ds4_attention_output_group_ids")) {
                    return 0;
                }
                group_ids_buffer = g_attn_out_group_ids_buffer;
            }
            int32_t *ids = (int32_t *)[group_ids_buffer contents];
            for (uint32_t t = 0; t < n_tokens; t++) {
                for (uint32_t group = 0; group < n_groups; group++) {
                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                }
            }
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }
        const bool attn_out_profile =
            getenv("DS4_METAL_ATTN_OUT_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        if (ok && attn_out_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                ok = false;
            } else {
                cb = ds4_gpu_command_buffer(&owned);
                if (!cb || owned) ok = false;
            }
        }
        double attn_out_t0 = attn_out_profile ? ds4_gpu_now_ms() : 0.0;
#define DS4_METAL_PROFILE_ATTN_OUT_STAGE(name) do { \
            if (ok && attn_out_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = false; \
                } else { \
                    const double now_ms = ds4_gpu_now_ms(); \
                    fprintf(stderr, \
                            "ds4: Metal attention output stage tokens=%u %s=%.3f ms\n", \
                            n_tokens, (name), now_ms - attn_out_t0); \
                    attn_out_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = false; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb || owned) ok = false; \
                    } \
                } \
            } \
        } while (0)

        if (ok) {
            /*
             * Batched attention-output projections switch from the vector
             * kernel to the SIMD matrix kernel once the batch has at least 32
             * tokens.  This preserves the single-token generation path while
             * keeping prefill accumulation stable.
             */
            if (use_mpp_low) {
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                /*
                 * Direct RHS lets MPP read the dense low-rank activation tile
                 * directly from device memory instead of staging a second
                 * threadgroup tile.  The retained attention-output path is the
                 * 64-token direct-RHS kernel; the older staged-RHS and 32-token
                 * variants were not kept as alternate runtime modes.
                 */
                const char *attn_out_pipeline_name =
                    "kernel_attn_out_low_q8_0_mpp_direct_rhs_n64";
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline(attn_out_pipeline_name, false);
                ok = ds4_gpu_encode_attn_out_low_q8_mpp(cb,
                                                          mm_pipeline,
                                                          &mm_args,
                                                          out_a_buf,
                                                          (NSUInteger)out_a_inner,
                                                          ds4_gpu_tensor_buffer(heads),
                                                          ds4_gpu_tensor_offset(heads),
                                                          ds4_gpu_tensor_buffer(low),
                                                          ds4_gpu_tensor_offset(low)) != 0;
                if (!ok) {
                    ds4_gpu_warn_mpp_fallback();
                    if (ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                        if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                            group_ids_buffer =
                                ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                        } else if (ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                                   &g_attn_out_group_ids_bytes,
                                                                   ids_bytes,
                                                                   "ds4_attention_output_group_ids")) {
                            group_ids_buffer = g_attn_out_group_ids_buffer;
                        }
                        if (group_ids_buffer) {
                            int32_t *ids = (int32_t *)[group_ids_buffer contents];
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                for (uint32_t group = 0; group < n_groups; group++) {
                                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                                }
                            }
                            ds4_gpu_mul_mm_id_map_args map_args =
                                ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                                  n_groups,
                                                                  n_groups,
                                                                  n_groups,
                                                                  n_tokens);
                            id<MTLComputePipelineState> map_pipeline =
                                ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                            id<MTLComputePipelineState> fallback_pipeline =
                                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                            ok = ds4_gpu_encode_mul_mm_id(cb,
                                                            map_pipeline,
                                                            fallback_pipeline,
                                                            &map_args,
                                                            &mm_args,
                                                            out_a_buf,
                                                            (NSUInteger)out_a_inner,
                                                            ds4_gpu_tensor_buffer(heads),
                                                            ds4_gpu_tensor_offset(heads),
                                                            ds4_gpu_tensor_buffer(low),
                                                            ds4_gpu_tensor_offset(low),
                                                            group_ids_buffer,
                                                            0) != 0;
                        }
                    }
                }
            } else if (n_tokens >= 32u && ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                ds4_gpu_mul_mm_id_map_args map_args =
                    ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                      n_groups,
                                                      n_groups,
                                                      n_groups,
                                                      n_tokens);
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                id<MTLComputePipelineState> map_pipeline =
                    ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                ok = ds4_gpu_encode_mul_mm_id(cb,
                                                map_pipeline,
                                                mm_pipeline,
                                                &map_args,
                                                &mm_args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0) != 0;
            } else if (use_direct_low) {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = 0,
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
                ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                             pipeline,
                                                             &args,
                                                             out_a_buf,
                                                             (NSUInteger)out_a_inner,
                                                             ds4_gpu_tensor_buffer(heads),
                                                             ds4_gpu_tensor_offset(heads),
                                                             ds4_gpu_tensor_buffer(low),
                                                             ds4_gpu_tensor_offset(low),
                                                             32u * 2u * sizeof(float),
                                                             4) != 0;
            } else {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = (uint64_t)n_groups * sizeof(int32_t),
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_0_f32", 4);
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                pipeline,
                                                &args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0,
                                                32u * 2u * sizeof(float),
                                                4,
                                                true) != 0;
            }
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("low_proj");

        if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                              out_b_offset,
                                              low_dim, out_dim, low, n_tokens) != 0;
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("out_proj");

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
#undef DS4_METAL_PROFILE_ATTN_OUT_STAGE
        return ok ? 1 : 0;
    }
}

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output low received invalid q8 dimensions\n");
            return 0;
        }

        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) {
            fprintf(stderr, "ds4: Metal attention output low weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = low_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes) {
            fprintf(stderr, "ds4: Metal attention output low received undersized buffers\n");
            return 0;
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }

        if (ok) {
            ds4_gpu_mul_mv_id_args args = {
                .nei0 = (int32_t)n_groups,
                .nei1 = 1,
                .nbi1 = 0,
                .ne00 = (int32_t)group_dim,
                .ne01 = (int32_t)rank,
                .ne02 = (int32_t)n_groups,
                .nb00 = 34,
                .nb01 = row_a_bytes,
                .nb02 = (uint64_t)rank * row_a_bytes,
                .ne10 = (int32_t)group_dim,
                .ne11 = (int32_t)n_groups,
                .ne12 = 1,
                .ne13 = 1,
                .nb10 = sizeof(float),
                .nb11 = (uint64_t)group_dim * sizeof(float),
                .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                .ne0 = (int32_t)rank,
                .ne1 = (int32_t)n_groups,
                .nb1 = (uint64_t)rank * sizeof(float),
                .nr0 = 2,
            };
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
            ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                         pipeline,
                                                         &args,
                                                         out_a_buf,
                                                         (NSUInteger)out_a_inner,
                                                         ds4_gpu_tensor_buffer(heads),
                                                         ds4_gpu_tensor_offset(heads),
                                                         ds4_gpu_tensor_buffer(low),
                                                         ds4_gpu_tensor_offset(low),
                                                         32u * 2u * sizeof(float),
                                                         4) != 0;
        }

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        return ok ? 1 : 0;
    }
}
