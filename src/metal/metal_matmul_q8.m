/* metal_matmul_q8.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;

    @autoreleasepool {
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        const uint64_t mask_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> maskbuf = ds4_gpu_tensor_buffer(mask);
        if (!topkbuf || !maskbuf ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(mask) < mask_bytes) {
            fprintf(stderr, "ds4: Metal dsv4 top-k mask received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)n_comp,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)n_comp * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_scatter_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)top_k * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "dsv4 top-k mask")) return 0;
    }

    return 1;
}

static int ds4_gpu_matmul_q8_0_legacy_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
            if (out_dim > 65536u) mv_dispatch.nsg = 8;
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matvec")) {
                return 0;
            }
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(1, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor mul_mv_ext")) {
                return 0;
            }
            return 1;
        }

        /*
         * Dense Q8_0 prefill is the cleanest DS4 TensorOps shape: M/N/K are
         * aligned and the RHS activation matrix is already dense.  The retained
         * kernel dequantizes each 64x32 weight tile to half in threadgroup
         * memory, then uses direct-RHS MPP for the activation tile.  This avoids
         * staging RHS into threadgroup memory and was the direct replacement for
         * the slower generic MPP prototype.
         */
        /* wave-78: round n_tok up to a multiple of 32 so non-aligned batches
         * (notably copy-spec verify batches sized 1+n_copy, e.g. 49) still hit
         * the fast NAX direct-RHS MMA path instead of the slower bounds-checked
         * generic mul_mm. The padded rows out[n_tok..n_tok_nax) compute garbage
         * the caller never reads; rows [0,n_tok) are bit-identical. Only engaged
         * when the (pc-sized) activation/output buffers actually hold the padded
         * rows -- otherwise n_tok_nax stays n_tok and the %32 gate below is the
         * original behaviour (no OOB possible). */
        uint64_t n_tok_nax = n_tok;
        if ((n_tok % 32u) != 0u) {
            const uint64_t padded = ((n_tok + 31u) / 32u) * 32u;
            if (ds4_gpu_tensor_bytes(x)   >= padded * in_dim  * sizeof(float) &&
                ds4_gpu_tensor_bytes(out) >= padded * out_dim * sizeof(float)) {
                n_tok_nax = padded;
            }
        }
        if (ds4_gpu_mpp_available() &&
            n_tok_nax >= 32u &&
            (in_dim % 64u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok_nax % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok_nax % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok_nax % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64"
                    : "kernel_mul_mm_q8_0_f32_nax_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok_nax, row_bytes);

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok_nax / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 NAX tensor matmul")) {
                    return 0;
                }
                return 1;
            }
            ds4_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_q8_0_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matmul")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    const int profile_requested =
        n_tok > 8u && ds4_gpu_env_bool("DS4_METAL_Q8_PREFILL_PROFILE") > 0;
    int profile_prefill = 0;
    int split_batch_for_profile = 0;
    const char *profile_label = NULL;
    char profile_label_buf[128];
    char profile_fallback[128];
    if (profile_requested) {
        snprintf(profile_fallback, sizeof(profile_fallback),
                 "q8 weight_off=%llu in=%llu out=%llu tok=%llu",
                 (unsigned long long)weight_offset,
                 (unsigned long long)in_dim,
                 (unsigned long long)out_dim,
                 (unsigned long long)n_tok);
        snprintf(profile_label_buf, sizeof(profile_label_buf), "%s", profile_fallback);
        profile_label = profile_label_buf;
        const char *profile_filter = getenv("DS4_METAL_Q8_PREFILL_PROFILE_FILTER");
        profile_prefill =
            profile_requested &&
            (!profile_filter || !profile_filter[0] ||
             strstr(profile_label, profile_filter) != NULL);
    }
    if (profile_prefill) {
        if (g_batch_cb) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            split_batch_for_profile = 1;
        }
    }

    const double profile_t0 = profile_prefill ? ds4_gpu_now_ms() : 0.0;
    int ok = ds4_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                                weight_offset, in_dim, out_dim,
                                                x, n_tok);
    if (profile_prefill) {
        if (split_batch_for_profile && ds4_gpu_end_commands() == 0) {
            ok = 0;
        }
        const double elapsed_ms = ds4_gpu_now_ms() - profile_t0;
        fprintf(stderr,
                "ds4: Metal Q8_0 prefill profile %s in=%llu out=%llu tok=%llu %.3f ms\n",
                profile_label ? profile_label : profile_fallback,
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (unsigned long long)n_tok,
                elapsed_ms);
        if (split_batch_for_profile && ds4_gpu_begin_commands() == 0) {
            ok = 0;
        }
    }
    return ok;
}

/* Tensor-parallel row-parallel Q8_0 matvec (decode/n_tok=1 only).
 *
 * Computes a PARTIAL out[out_dim] using only the input-dim block range
 * [0, in_dim_slice) of each weight row, where rows are strided by the FULL
 * in_dim_full row size. The caller pre-shifts `weight_offset` to the owned
 * slice's first block (base + in_block_start*34) and passes the compacted
 * x[in_dim_slice]. Each TP peer's result is a partial sum over its in-dim half;
 * an all-reduce across peers reconstructs the full matvec (associativity differs
 * from a single-machine reduction => ~1e-6 drift, not bit-identical — inherent to
 * TP row-parallel). Reuses kernel_mul_mv_q8_0_f32 (no .metal change): the trick is
 * ne00=slice (blocks to accumulate) while nb01=full row stride (row addressing). */
/* dense Q4_K matmul: CUDA-first(backbone-q4k 配方)。Metal 未实现 — 返回 0,
 * 引擎在校验/调用点会把失败向上抛, 不会静默算错。 */
int ds4_gpu_token_graph_begin(void) { return 0; }

/* CUDA-only */
void ds4_gpu_token_graph_set_pos(uint32_t pos) { (void)pos; }

int ds4_gpu_kv_rope_fp8_store_raw_tensor(
    ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache,
    uint32_t raw_cap, uint32_t raw_row, uint32_t head_dim, uint32_t n_rot,
    uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale,
    float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    /* Metal 无三合一核: 顺序回退(语义=rope→fp8→store 三发) */
    if (!ds4_gpu_rope_tail_tensor(kv, 1, 1, head_dim, n_rot, pos, n_ctx_orig, false,
                                  freq_base, freq_scale, ext_factor, attn_factor,
                                  beta_fast, beta_slow)) return 0;
    return ds4_gpu_kv_fp8_store_raw_tensor(kv, raw_cache, raw_cap, raw_row, head_dim, n_rot);
}

int ds4_gpu_head_rms_norm_rope_tail_tensor(
    ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
    uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
    float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow, float eps) {
    /* Metal 无融合核: 顺序回退(语义=两发) */
    if (!ds4_gpu_head_rms_norm_tensor(x, n_tok, n_head, head_dim, eps)) return 0;
    return ds4_gpu_rope_tail_tensor(x, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
                                    inverse, freq_base, freq_scale, ext_factor, attn_factor,
                                    beta_fast, beta_slow);
}

int ds4_gpu_token_graph_end_launch(void) { return 0; }

int ds4_gpu_token_graph_try_pending(int token, uint32_t pos, int need_logits) { (void)token; (void)pos; (void)need_logits; return 0; }

int ds4_gpu_token_graph_precapture_begin(void) { return 0; }

int ds4_gpu_side_mark(void) { return 0; }

/* CUDA-only: Metal 顺序执行 */
int ds4_gpu_side_begin(void) { return 0; }

int ds4_gpu_side_main(void) { return 1; }

int ds4_gpu_dspark_hc_mean_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *hc,
                                  uint32_t n_embd, uint32_t n_hc, uint32_t slot, uint32_t n_tokens) {
    (void)dst; (void)hc; (void)n_embd; (void)n_hc; (void)slot; (void)n_tokens;
    return 0;   /* CUDA-only for now */
}

int ds4_gpu_dspark_attn_tensor(ds4_gpu_tensor *heads,
                               const void *model_map, uint64_t model_size, uint64_t sinks_offset,
                               const ds4_gpu_tensor *q, const ds4_gpu_tensor *win_kv,
                               const ds4_gpu_tensor *blk_kv,
                               uint32_t n_win, uint32_t blk, uint32_t n_head, uint32_t head_dim,
                               uint32_t win_base, uint32_t win_cap) {
    (void)heads; (void)model_map; (void)model_size; (void)sinks_offset; (void)q;
    (void)win_kv; (void)blk_kv; (void)n_win; (void)blk; (void)n_head; (void)head_dim;
    (void)win_base; (void)win_cap;
    return 0;
}

int ds4_gpu_dspark_win_scatter_tensor(ds4_gpu_tensor *win, const ds4_gpu_tensor *rows,
                                      uint32_t n, uint32_t pos0, uint32_t win_rows, uint32_t dim) {
    (void)win; (void)rows; (void)n; (void)pos0; (void)win_rows; (void)dim;
    return 0;
}

int ds4_gpu_dspark_argmax_only_tensor(ds4_gpu_tensor *out_id,
                                      const ds4_gpu_tensor *logits_row, uint32_t vocab) {
    (void)out_id; (void)logits_row; (void)vocab; return 0;
}

int ds4_gpu_dspark_markov_step_tensor(ds4_gpu_tensor *out_id, ds4_gpu_tensor *logits_row,
                                      const void *model_map, uint64_t model_size,
                                      uint64_t w1_offset, uint64_t w2_offset,
                                      const ds4_gpu_tensor *prev_id, uint32_t vocab, uint32_t rank) {
    (void)out_id; (void)logits_row; (void)model_map; (void)model_size;
    (void)w1_offset; (void)w2_offset; (void)prev_id; (void)vocab; (void)rank;
    return 0;
}

int ds4_gpu_dspark_confidence_tensor(ds4_gpu_tensor *out_conf, const ds4_gpu_tensor *x,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t conf_w_offset, uint64_t markov_w1_offset,
                                     const ds4_gpu_tensor *prev_ids,
                                     uint32_t dim, uint32_t rank, uint32_t vocab, uint32_t n_pos) {
    (void)out_conf; (void)x; (void)model_map; (void)model_size;
    (void)conf_w_offset; (void)markov_w1_offset; (void)prev_ids;
    (void)dim; (void)rank; (void)vocab; (void)n_pos;
    return 0;   /* CUDA-only(spark 置信头); Metal 无实现, 与本组其余 dspark stub 同约定 */
}

int ds4_gpu_side_join(void) { return 1; }

int ds4_gpu_matmul_q4_K_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t off0, uint64_t off1,
                                    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                    const ds4_gpu_tensor *x) {
    (void)out0; (void)out1; (void)model_map; (void)model_size; (void)off0; (void)off1;
    (void)in_dim; (void)out0_dim; (void)out1_dim; (void)x;
    return 0;   /* CUDA-only; 调用方回退两次单矩阵 */
}

int ds4_gpu_matmul_q2_K_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t off0, uint64_t off1,
                                    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                    const ds4_gpu_tensor *x) {
    (void)out0; (void)out1; (void)model_map; (void)model_size; (void)off0; (void)off1;
    (void)in_dim; (void)out0_dim; (void)out1_dim; (void)x;
    return 0;   /* CUDA-only; 调用方回退两次单矩阵 */
}

int ds4_gpu_token_graph_precapture_end(uint32_t pos, int need_logits, int encode_ok) { (void)pos; (void)need_logits; (void)encode_ok; return 0; }

int ds4_gpu_attention_output_q4k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out; (void)low; (void)model_map; (void)model_size; (void)out_a_offset;
    (void)out_b_offset; (void)group_dim; (void)rank; (void)n_groups; (void)out_dim;
    (void)heads; (void)n_tokens;
    return 0;
}

int ds4_gpu_attention_output_low_q4k_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, const ds4_gpu_tensor *heads) {
    (void)low; (void)model_map; (void)model_size; (void)out_a_offset;
    (void)group_dim; (void)rank; (void)n_groups; (void)heads;
    return 0;   /* CUDA-first; Metal 未实现 */
}

int ds4_gpu_matmul_q4_K_hc_expand_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)residual_hc; (void)split; (void)n_embd; (void)n_hc;
    return 0;
}

int ds4_gpu_matmul_q4_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}
