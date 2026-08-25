/* core_moe.c — routed MoE one/batch (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
void layer_routed_moe_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        bool                trace) {
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    float *gate = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(gate[0])) : NULL;
    float *up = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(up[0])) : NULL;
    float *mid = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(mid[0])) : NULL;
    float *mid_all = trace ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(mid_all[0]));
    float *down = trace ? xmalloc((size_t)DS4_N_EMBD * sizeof(down[0])) : NULL;
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");
    block_q8_K *xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(xq[0]));
    block_q8_K *midq = trace ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(midq[0]));

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x, corr_layer_delta(model, il));
    }
    /* zchain GE: fold per-expert gains into the router weights before the expert
     * matmuls (quantizer bytes_moe parity: the gain scales each selected expert's
     * contribution linearly, exactly gate-weight scaling). */
    {
        const float *zge = ds4_zchain_layer_ge(model->zchain, il);
        if (zge) for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            const int ze = selected[i];
            if (ze >= 0 && (uint32_t)ze < DS4_N_EXPERT) expert_weight[i] *= zge[ze];
        }
    }

    if (!trace) {
        matvec_iq2_xxs_experts_mid_prequant(mid_all, model,
                                            layer->ffn_gate_exps,
                                            layer->ffn_up_exps,
                                            xq,
                                            selected,
                                            expert_weight,
                                            DS4_N_EXPERT_USED,
                                            clamp);
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                                  midq + (uint64_t)i * (down_in_dim / QK_K),
                                  (int64_t)down_in_dim);
        }
        matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);
    } else {
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            const uint32_t expert = (uint32_t)selected[i];

            matvec_iq2_xxs_expert_pair_prequant(gate, up, model,
                                                 layer->ffn_gate_exps,
                                                 layer->ffn_up_exps,
                                                 xq,
                                                 expert);
            char name[64];
            snprintf(name, sizeof(name), "blk.%u expert %u gate", il, expert);
            print_vec_stats(name, gate, DS4_N_FF_EXP);
            snprintf(name, sizeof(name), "blk.%u expert %u up", il, expert);
            print_vec_stats(name, up, DS4_N_FF_EXP);

            /*
             * DeepSeek V4 clamps routed expert gate/up values before SwiGLU and
             * applies the router weight before the down projection.
             */
            const float limit = clamp;
            for (uint32_t j = 0; j < DS4_N_FF_EXP; j++) {
                if (limit > 1.0e-6f) {
                    if (gate[j] > limit) gate[j] = limit;
                    if (up[j] > limit) up[j] = limit;
                    if (up[j] < -limit) up[j] = -limit;
                }
                mid[j] = silu(gate[j]) * up[j] * expert_weight[i];
            }

            snprintf(name, sizeof(name), "blk.%u expert %u mid", il, expert);
            print_vec_stats(name, mid, DS4_N_FF_EXP);

            matvec_q2_k_expert(down, model, layer->ffn_down_exps, mid, expert);
            snprintf(name, sizeof(name), "blk.%u expert %u down", il, expert);
            print_vec_stats(name, down, DS4_N_EMBD);
            for (uint32_t j = 0; j < DS4_N_EMBD; j++) out[j] += down[j];
        }
    }

    /* zchain λ(x): per-token scale on the routed sum (the whole GL/dyn/TREF chain
     * collapses to this scalar; the additive corr sidecar lands after, matching
     * the quantizer's op order). */
    if (ds4_zchain_layer_has_lambda(model->zchain, il)) {
        const float zlam = ds4_zchain_lambda(model->zchain, il, x);
        for (uint32_t j = 0; j < DS4_N_EMBD; j++) out[j] *= zlam;
    }
    /* frozen z^L (type 6): rank-k additive direction fix on the routed sum,
     * after the λ scale (record-order parity with the quantizer's bytes_moe). */
    { const ds4_zchain_zl *zzl = ds4_zchain_layer_zl(model->zchain, il);
      if (zzl) ds4_zchain_zl_apply(zzl, DS4_N_EMBD, x, out); }

    /* go1b correction: add the low-rank per-expert residual on top of the 1-bit
     * expert sum (no-op when no corr sidecar is loaded). */
    corr_apply_moe_host(out, model, il, x, selected, DS4_N_EXPERT_USED);

    free(midq);
    free(xq);
    free(down);
    free(mid_all);
    free(mid);
    free(up);
    free(gate);
}

/* Decode version of routed MoE: same math as layer_routed_moe_one(), but all
 * large temporaries come from the persistent scratch arena. */
void layer_routed_moe_one_prealloc(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        float              * mid_all,
        block_q8_K         * xq,
        block_q8_K         * midq) {
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x, corr_layer_delta(model, il));
    }
    /* zchain GE: fold per-expert gains into the router weights (see the
     * layer_routed_moe_one() copy of this hook for the contract). */
    {
        const float *zge = ds4_zchain_layer_ge(model->zchain, il);
        if (zge) for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            const int ze = selected[i];
            if (ze >= 0 && (uint32_t)ze < DS4_N_EXPERT) expert_weight[i] *= zge[ze];
        }
    }

    matvec_iq2_xxs_experts_mid_prequant(mid_all, model,
                                        layer->ffn_gate_exps,
                                        layer->ffn_up_exps,
                                        xq,
                                        selected,
                                        expert_weight,
                                        DS4_N_EXPERT_USED,
                                        clamp);

    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);

    /* zchain λ(x): scale the routed sum before the additive corr (op-order parity). */
    if (ds4_zchain_layer_has_lambda(model->zchain, il)) {
        const float zlam = ds4_zchain_lambda(model->zchain, il, x);
        for (uint32_t j = 0; j < DS4_N_EMBD; j++) out[j] *= zlam;
    }
    /* frozen z^L (type 6): rank-k additive direction fix on the routed sum,
     * after the λ scale (record-order parity with the quantizer's bytes_moe). */
    { const ds4_zchain_zl *zzl = ds4_zchain_layer_zl(model->zchain, il);
      if (zzl) ds4_zchain_zl_apply(zzl, DS4_N_EMBD, x, out); }

    /* go1b correction (no-op without a corr sidecar). */
    corr_apply_moe_host(out, model, il, x, selected, DS4_N_EXPERT_USED);
}

/* Prefill MoE groups token/expert pairs by expert so each active expert's
 * rows are scanned once for the whole token batch. */
void layer_routed_moe_batch(
        float             * moe,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        float               clamp) {
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t expert_out_dim = routed_expert_mid_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
    if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
    if (down_in_dim % QK_K != 0) ds4_die("Q2_K expert input is not QK_K aligned");
    if (expert_out_dim != down_in_dim || down_out_dim != DS4_N_EMBD) {
        ds4_die("routed expert tensor layout is unexpected");
    }

    const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;
    uint32_t counts[DS4_MAX_EXPERT + 1] = {0};
    uint32_t cursor[DS4_MAX_EXPERT] = {0};
    uint32_t active_expert[DS4_MAX_EXPERT];
    uint32_t n_active = 0;

    int *selected = xmalloc((size_t)total_pairs * sizeof(selected[0]));
    float *pair_weight = xmalloc((size_t)total_pairs * sizeof(pair_weight[0]));
    ds4_expert_pair *pairs = xmalloc((size_t)total_pairs * sizeof(pairs[0]));

    const uint64_t xq_blocks = expert_in_dim / QK_K;
    block_q8_K *xq = xmalloc((size_t)n_tok * xq_blocks * sizeof(xq[0]));
    for (uint32_t t = 0; t < n_tok; t++) {
        ds4_quantize_row_q8_K(norm + (uint64_t)t * expert_in_dim,
                              xq + (uint64_t)t * xq_blocks,
                              (int64_t)expert_in_dim);

        int sel[DS4_MAX_EXPERT_USED];
        float weights[DS4_MAX_EXPERT_USED];
        if (layer->ffn_gate_tid2eid) {
            layer_hash_selected_experts(sel, model, layer, token_ids[t]);
            layer_hash_router_weights_one(weights, model, layer, norm + (uint64_t)t * expert_in_dim, sel);
        } else {
            layer_topk_selected_experts(sel, weights, model, layer, norm + (uint64_t)t * expert_in_dim,
                                        corr_layer_delta(model, il));
        }
        /* zchain GE: fold per-expert gains into the pair weights (see
         * layer_routed_moe_one() for the contract). */
        {
            const float *zge = ds4_zchain_layer_ge(model->zchain, il);
            if (zge) for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
                const int ze = sel[slot];
                if (ze >= 0 && (uint32_t)ze < DS4_N_EXPERT) weights[slot] *= zge[ze];
            }
        }

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint32_t pair_id = t * DS4_N_EXPERT_USED + slot;
            selected[pair_id] = sel[slot];
            pair_weight[pair_id] = weights[slot];
            pairs[pair_id] = (ds4_expert_pair){ .token = t, .slot = slot };
            if (sel[slot] < 0 || (uint32_t)sel[slot] >= DS4_N_EXPERT) ds4_die("selected expert is outside range");
            counts[(uint32_t)sel[slot] + 1]++;
        }
    }

    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        counts[e + 1] += counts[e];
        cursor[e] = counts[e];
        if (counts[e + 1] != counts[e]) active_expert[n_active++] = e;
    }

    uint32_t *pair_ids = xmalloc((size_t)total_pairs * sizeof(pair_ids[0]));
    for (uint32_t p = 0; p < total_pairs; p++) {
        const uint32_t e = (uint32_t)selected[p];
        pair_ids[cursor[e]++] = p;
    }

    float *mid = xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));

    matvec_iq2_xxs_batch_mid_ctx mid_ctx = {
        .mid = mid,
        .xq = xq,
        .pairs = pairs,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .pair_weight = pair_weight,
        .clamp = clamp,
        .in_dim = expert_in_dim,
        .out_dim = expert_out_dim,
        .xq_blocks = xq_blocks,
    };

    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t e = active_expert[ai];
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                   &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
        mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                 &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
        if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
            gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
            ds4_die("IQ2_XXS batch expert tensor layout mismatch");
        }
    }

    ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_iq2_xxs_batch_mid_worker, &mid_ctx);

    const uint64_t midq_blocks = down_in_dim / QK_K;
    block_q8_K *midq = xmalloc((size_t)total_pairs * midq_blocks * sizeof(midq[0]));
    quantize_mid_pairs_ctx quant_ctx = {
        .mid = mid,
        .midq = midq,
        .down_in_dim = down_in_dim,
        .down_blocks = midq_blocks,
    };
    ds4_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
    free(mid);

    matvec_q2_k_batch_accum_rows_ctx down_ctx = {
        .moe = moe,
        .midq = midq,
        .pairs = pairs,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .n_active = n_active,
        .n_tok = n_tok,
        .in_dim = down_in_dim,
        .out_dim = down_out_dim,
        .midq_blocks = midq_blocks,
    };

    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t e = active_expert[ai];
        uint64_t in_dim, out_dim;
        down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                               &in_dim, &out_dim, &down_ctx.row_bytes[e]);
        if (in_dim != down_in_dim || out_dim != down_out_dim) {
            ds4_die("Q2_K batch expert tensor layout mismatch");
        }
    }

    ds4_parallel_for(down_out_dim, matvec_q2_k_batch_accum_rows_worker, &down_ctx);

    /* zchain λ(x) per token: scale the routed sums before the additive corr
     * (op-order parity with the quantizer replay). */
    if (ds4_zchain_layer_has_lambda(model->zchain, il)) {
        for (uint32_t t = 0; t < n_tok; t++) {
            const float zlam = ds4_zchain_lambda(model->zchain, il,
                                                 norm + (uint64_t)t * expert_in_dim);
            float *zmt = moe + (uint64_t)t * DS4_N_EMBD;
            for (uint32_t j = 0; j < DS4_N_EMBD; j++) zmt[j] *= zlam;
        }
    }
    /* frozen z^L (type 6) per token, after λ (record-order parity). */
    { const ds4_zchain_zl *zzl = ds4_zchain_layer_zl(model->zchain, il);
      if (zzl) for (uint32_t t = 0; t < n_tok; t++)
          ds4_zchain_zl_apply(zzl, DS4_N_EMBD,
                              norm + (uint64_t)t * expert_in_dim,
                              moe + (uint64_t)t * DS4_N_EMBD); }

    /* go1b correction per token (no-op without a corr sidecar). The pairs array is
     * grouped-by-expert, but `selected` keeps the per-(token,slot) expert ids. */
    if (model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present) {
        for (uint32_t t = 0; t < n_tok; t++) {
            corr_apply_moe_host(moe + (uint64_t)t * DS4_N_EMBD, model, il,
                                norm + (uint64_t)t * expert_in_dim,
                                &selected[(uint64_t)t * DS4_N_EXPERT_USED], DS4_N_EXPERT_USED);
        }
    }

    free(midq);
    free(pair_ids);
    free(xq);
    free(pairs);
    free(pair_weight);
    free(selected);
}


/* Full FFN sublayer for one token: HC pre, RMSNorm, routed MoE, shared expert,
 * sum, and HC post. */
