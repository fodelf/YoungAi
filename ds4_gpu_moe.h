#ifndef DS4_GPU_MOE_H
#define DS4_GPU_MOE_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4_gpu_core.h"

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

/* go1b "hidden variable z^L" four-loss correction (resident sidecar tensors).
 *
 * ds4_gpu_corr_router_bias adds the per-expert router-logit bias delta[e] to the
 * raw router logits (pre softplus/sqrt) BEFORE top-k selection, broadcast across
 * all n_tokens rows: logits[t][e] += delta[e]. Apply to score-routed layers only.
 *
 * ds4_gpu_corr_apply adds the per-selected-expert correction to the already-summed
 * routed-MoE output, in place:  out[t][d] += sum over selected e of
 *   ( U @ ( C[e] (.*) (V @ x[t]) ) )[d] + b[d] + beta[e].
 * x is the per-token FFN input fed to the experts (post-RMSNorm activation).
 * U is [d_model][d_l] row-major, V is [d_l][d_model] row-major, C is [n_expert][d_l]
 * row-major, b is [d_model], beta is [n_expert]; selected is [n_tokens][n_expert_used]
 * (original 0..n_expert-1 ids). One threadgroup per token; vx is recomputed per token. */
int ds4_gpu_corr_router_bias(
        ds4_gpu_tensor       *logits,
        const ds4_gpu_tensor *delta,
        uint32_t                n_expert,
        uint32_t                n_tokens);

int ds4_gpu_corr_apply(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t                d_model,
        uint32_t                d_l,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        uint32_t                n_tokens);

/* Store variant of corr_apply: writes the raw correction term into delta_out
 * (delta_out[t][d] = corr term) instead of accumulating into routed_out. The
 * decode shared-down HC fusion then adds routed[d]+delta[d] — the same fadd
 * the in-place kernel performed, so results are bit-identical, while the tiny
 * corr dispatch stops write-hazarding the hot routed_out buffer (a measured
 * ~23ms/layer full-pipeline bubble on Metal). Only used when
 * ds4_gpu_corr_delta_supported() returns nonzero AND the fused shared-down
 * consumer runs (single-host decode default); every other path keeps the
 * in-place kernel. */
int ds4_gpu_corr_apply_delta(
        ds4_gpu_tensor       *delta_out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t                d_model,
        uint32_t                d_l,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        uint32_t                n_tokens);

int ds4_gpu_corr_delta_supported(void);

/* go1b corr needs the ORIGINAL top-k expert ids, but the offload MoE remaps the
 * selected buffer to compact slots IN PLACE (ds4_gpu_remap_selected_to_slots)
 * before the corr reads it.  ds4_gpu_routed_moe_batch_tensor snapshots the
 * pre-remap ids into a persistent Shared buffer whenever it runs the go1b path;
 * the corr indexes C[e]/beta[e] from this copy instead of the corrupted buffer.
 * Returns NULL until the first go1b MoE call snapshots (callers fall back to the
 * live selected tensor). Layout matches selected: [n_tokens][n_expert_used]. */
ds4_gpu_tensor *ds4_gpu_corr_saved_selected(void);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

/* Reduced-expert (keep-map) routing support. A shrunken model's expert tensors
 * only carry the kept rows, but the router still selects in the full 256-wide
 * original id space. ds4_gpu_set_expert_keep_lut() uploads the per-layer
 * original-id -> compact-slot table (n_layer * 256 int16, -1 == dropped) into a
 * small resident GPU buffer once at load. ds4_gpu_translate_expert_ids() rewrites
 * a `selected` tensor from original ids to compact slots in place, between router
 * selection and the routed-MoE matvec. Both are no-ops for a full model (no LUT
 * set). Returns 1 on success, 0 on failure. */
int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer);
int ds4_gpu_translate_expert_ids(
        ds4_gpu_tensor       *selected,
        uint32_t                layer,
        uint32_t                n_expert_used,
        uint32_t                n_tokens,
        uint32_t                n_total_expert);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        uint32_t                layer);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens,
        uint32_t                layer);

/* Optional 1-bit residual expert weights (go1b) for the routed MoE: a second 1-bit
 * layer Q1(W-Q1(W)) summed into each expert matmul. NULL/gate=NULL => no residual.
 * The buffers are RESIDENT GPU copies of the _res go1b tensors (a wrapped mmap view
 * of an offload model reads as ZEROS on the GPU, so the residual MUST be resident,
 * exactly like the corr sidecar). Strides/dims reuse the base expert args. */
typedef struct {
    /* CPU pointers to the go1b residual expert weights (sidecar mmap). The decode
     * path CPU-gathers the active experts into a compacted scratch (like the base
     * offload gather), then runs the go1b mm_id mapped-tile matmul over them. */
    const void *gate_ptr;   /* blk.L.ffn_gate_exps_res weights (K experts if sparse) */
    const void *up_ptr;     /* blk.L.ffn_up_exps_res weights */
    const void *down_ptr;   /* blk.L.ffn_down_exps_res weights */
    /* Sparse residual: lut[expert_id] = slot in [0,K) or -1 (no residual for that
     * expert). NULL => dense (gate_ptr indexed directly by expert id, K=256). */
    const float *lut;
    /* Nonzero when the sidecar tensors are go2b (type 41: offline-merged
     * base+residual, exact 4-level sum). The metal batch path then routes hot
     * picks through a single go2b matmul pass and masks them out of the base
     * pass, instead of the legacy base+residual add (3 extra matmuls/layer). */
    int merged2b;
    /* Nonzero when gate_ptr points at a v2.2 VQ layer blob (DQVL: 表+DQVQ 载荷).
     * The metal path dequants every active expert to an f16 scratch at gather
     * (cold w2 expanded from the base go1b bytes) and runs the F16W mm_id. */
    int vq;
    /* Total DQVL blob bytes when vq!=0. The CUDA path uses it to relocate the
     * host-mmap blob pointer onto the HBM arena copy (the startup span cache
     * already holds these bytes; reading the mmap original would re-fault pages
     * that were madvise(DONTNEED)d after the copy). */
    uint64_t vq_bytes;
} ds4_gpu_residual_set;

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
        uint32_t                layer_index);

int ds4_gpu_routed_moe_batch_tensor(
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
        uint32_t                layer_index,
        uint32_t                n_tokens,
        uint32_t                slot_start,   /* TP Phase-3 batch split: owned slot range */
        uint32_t                slot_count,   /* 0 or n_expert => no split (full) */
        bool                   *mid_is_f16);


#endif
