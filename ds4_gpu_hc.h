#ifndef DS4_GPU_HC_H
#define DS4_GPU_HC_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4_gpu_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* corr_delta (nullable): when non-NULL the kernel consumes routed[d]+delta[d]
 * — the go1b corr store-variant output — with the exact fadd the in-place corr
 * kernel used, keeping results bit-identical without a routed_out write hazard. */
int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        const ds4_gpu_tensor *corr_delta,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* go-onebit DQZ2 zchain (multiplicative correction chain; math contract in
 * ds4_zchain.h). ds4_gpu_zchain_set() uploads the whole packed table once at
 * load into small resident buffers:
 *   ops       [n_ops_total][16] f32, layer-major. Slot layout per op:
 *             [0]=type (1 GL | 2 dyn2 | 3 dyn8 | 4 TREF), [1]=g or t,
 *             [2..5]=w2p, [6..14]=w8, [15]=dyn8 V8 block index (-1 = none).
 *   layer_off [n_layer+1] op range per layer (ops[layer_off[l]..layer_off[l+1]))
 *   v8        concatenated fp16 [n_v8_blocks][8][d_model] dyn8 projections
 *   ge        [n_layer][n_expert] f32 router-weight gains (1.0-filled rows for
 *             layers without GE)
 *   ge_present[n_layer] flags so no-GE layers skip the dispatch entirely.
 *
 * ds4_gpu_zchain_ge_apply(): weights[t][k] *= ge[layer][selected[t][k]] --
 * BEFORE the routed matvec and before any compact-slot remap (original ids).
 * ds4_gpu_zchain_scale_routed(): routed[t][:] *= λ_layer(x[t]) with λ folded
 * in-kernel over the layer's ops (feature reductions on x per token) -- AFTER
 * the routed accumulate (and any TP all-reduce), BEFORE the additive corr.
 * Both return 1 on success (including the layer-has-nothing fast path). */
int ds4_gpu_zchain_set(
        const float        *ops,
        const uint32_t     *layer_off,
        const uint16_t     *v8,
        const float        *ge,
        const uint8_t      *ge_present,
        uint32_t             n_layer,
        uint32_t             n_expert,
        uint32_t             d_model,
        uint32_t             n_ops_total,
        uint32_t             n_v8_blocks);

int ds4_gpu_zchain_ge_apply(
        ds4_gpu_tensor       *weights,
        const ds4_gpu_tensor *selected,
        uint32_t                layer,
        uint32_t                n_expert_used,
        uint32_t                n_tokens);

int ds4_gpu_zchain_scale_routed(
        ds4_gpu_tensor       *routed,
        const ds4_gpu_tensor *x,
        uint32_t                layer,
        uint32_t                n_tokens);

/* frozen z^L (type 6, 2026-07-14): upload packed fp16 factors (concat of
 * z[k]|U[d*k]|V[d*k] per zl layer) + per-layer {offset-in-halves, rank, trust
 * factor}. The scale_routed dispatch applies it after the λ scale. k[l]==0 for
 * every layer (or n_layer==0) = nothing to do, returns 1. */
int ds4_gpu_zchain_zl_set(
        const uint16_t *zlm,
        const uint32_t *off,
        const uint32_t *k,
        const uint32_t *din,    /* md86: per-layer V input dim (NULL = all d_model; 3d = ftA) */
        const float    *tr,     /* type6: trust cap | type7 AMP: tanh 定标 scale */
        const uint32_t *mul,    /* per-layer 1=乘性AMP(type7, 2026-08-19), NULL=全加性 */
        uint32_t         n_layer,
        uint64_t         total_halves);

/* 路由闭式侧车(type8 zl.RTE, 2026-08-19): blob = z[k]|U[n_expert*k]|V[d_model*k]
 * per rte layer; 应用 δlogits = U·diag(z)·tanh(Vᵀx/s) 加在 router raw logits 上
 * (select 前)。k[l]==0 = 该层无侧车。 */
int ds4_gpu_zchain_rte_set(
        const uint16_t *rm,
        const uint32_t *off,
        const uint32_t *k,
        const float    *scale,
        uint32_t         n_layer,
        uint32_t         n_expert,
        uint64_t         total_halves);

/* logits[n_tokens][n_expert] += route 侧车 δ(x); 无侧车层零成本返回 1。 */
int ds4_gpu_zchain_route_bias(
        ds4_gpu_tensor       *logits,
        const ds4_gpu_tensor *x,
        uint32_t               layer,
        uint32_t               n_tokens);

#ifdef __cplusplus
}
#endif

#endif
