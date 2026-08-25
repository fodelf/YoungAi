struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};

struct ds4_metal_args_dsv4_softmax_pool {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexed_attention {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
kernel void kernel_dsv4_directional_steering_project_f32(
        constant ds4_metal_args_dsv4_directional_steering_project & args,
        device float *x,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir = directions + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        sum += xr[i] * dir[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff * dir[i];
    }
}

// Decode-only DS4 ratio-4 indexer score builder.  One threadgroup owns one
// compressed row for the current token, stages that 128-wide row once, then
// walks the 64 indexer heads in four-head groups.  This avoids materializing the
// intermediate [compressed rows x heads] score matrix used by the generic
// matvec + weighted-sum path.
kernel void kernel_dsv4_indexer_score_one_direct(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;        // [128]
    threadgroup float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        device const float *krow = (device const float *)(index_comp +
            (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * args.q_head_stride);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            device const float *w = (device const float *)weights;
            psum[sg] = max(s, 0.0f) * (w[head] * args.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float *dst = (device float *)scores;
        dst[row] = acc;
    }
}

// Decode router post-processing for one token. The selected expert ids are
// already known; this gathers their probabilities, normalizes by the selected
// sum, clamps the denominator like the reference path, and applies DS4's 1.5
// expert-weight scale in one tiny dispatch.
kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

// Reduced-expert routing id translation for a shrunken (keep-map) model.
//
// The router selects experts using the full 256-wide original id space, but a
// shrunken model's expert tensors only carry the kept rows. This rewrites each
// selected id in-place from its original id to the compact slot in the kept
// tensors, using a per-layer 256-entry lookup table (orig id -> slot, -1 if the
// expert was dropped). Dropped experts fall back to slot 0 so the matvec never
// indexes out of bounds (the routed weight on a dropped expert is small; smoke
// quality is not the goal here). One thread per (token, used-slot); a full model
// never dispatches this kernel (no keep-map -> no-op at the host).
struct ds4_metal_args_dsv4_route_translate {
    uint32_t layer;            // which 256-entry LUT row to use
    uint32_t n_expert_used;    // entries per token in `selected` (== 6)
    uint32_t n_tokens;
    uint32_t n_total_expert;   // kept expert count for this layer (clamp upper bound)
    uint32_t verify;           // B2b: 1 => atomic-tally every clamp into clamp_count
};

// B2b fail-loud clamp counter (react-go-execution-plan M-1.2). A dropped or
// out-of-range expert id is silently clamped to slot 0 below; under Mode P (cold
// experts masked out of routing entirely) that must never fire, so verify mode
// tallies each clamp keyed by (layer, original id) into clamp_count[layer*256 +
// orig] and the host dumps a fail-loud report at exit. A non-zero tally means the
// router-side -inf mask (non-hash layers) or the hash-layer keep-all guard (B2a)
// leaked a cold expert into the top-k. Zero runtime cost when verify == 0.
kernel void kernel_dsv4_route_translate(
        constant ds4_metal_args_dsv4_route_translate & args,
        device const int16_t *lut,      // [n_layer * 256]
        device       int32_t *selected, // [n_tokens * n_expert_used], rewritten in place
        device       atomic_uint *clamp_count, // [n_layer * 256], verify-mode tally
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.n_expert_used;
    if (gid >= total) return;

    device const int16_t *row = lut + (uint64_t)args.layer * 256u;
    const int32_t orig = selected[gid];
    int32_t slot = (orig >= 0 && orig < 256) ? (int32_t)row[orig] : -1;
    if (slot < 0 || slot >= (int32_t)args.n_total_expert) {
        if (args.verify) {
            const uint key = args.layer * 256u +
                             (uint)((orig >= 0 && orig < 256) ? orig : 0);
            atomic_fetch_add_explicit(&clamp_count[key], 1u, memory_order_relaxed);
        }
        slot = 0;               // dropped/out-of-range expert -> safe slot 0
    }
    selected[gid] = slot;
}

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Fills the dense compressed-attention mask with -inf. The selected top-k rows
// are enabled by kernel_dsv4_topk_mask_scatter in a second ordered dispatch.
kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

// Sorts each token's selected compressed rows by row id. The indexer selects by
// score, but attention scans compressed K/V in cache order in the dense graph.
// Sorting preserves that order while still letting the indexed attention kernel
// touch only the selected rows.
kernel void kernel_dsv4_sort_i32_rows_asc(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * src,
        device       char * dst,
        threadgroup int32_t * row_tmp [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= top_k) {
        return;
    }

    row_tmp[tid] = *((device const int32_t *) (src + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid && other < top_k) {
                const int32_t a = row_tmp[tid];
                const int32_t b = row_tmp[other];
                const bool up = (tid & k) == 0;
                if ((up && a > b) || (!up && a < b)) {
                    row_tmp[tid] = b;
                    row_tmp[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    *((device int32_t *) (dst + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[tid];
}

