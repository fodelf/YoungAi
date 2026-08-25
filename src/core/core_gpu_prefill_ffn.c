/* core_gpu_prefill_ffn.c — prefill FFN 批编码 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    return metal_graph_encode_layer_ffn_batch_ex(g, model, layer, il, pos0, n_tokens, false);
}
bool metal_graph_encode_layer_ffn_batch_ex(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens,
        bool                    no_zchain) {
    /* no_zchain(2026-08-21): dspark drafter 复用 il=0 批段, 旧注释"L0 无 z 语义中性"
     * 在 amp42 链(L0 有 AMP)下过时 → mtp 层误吃 L0 放大器。drafter 调用旁路侧车。 */
    const struct ds4_zchain *zch = no_zchain ? NULL : model->zchain;
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t expert_mid_dim = routed_expert_mid_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = (layer->ffn_gate_exps ? routed_expert_row_bytes(layer->ffn_gate_exps) : 0);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_ffn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                       g->batch_ffn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->ffn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("norm");
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_router_logits,
                                             model->map,
                                             model->size,
                                             layer->ffn_gate_inp->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    /* go1b correction: bias raw router logits by delta[e] before top-k (broadcast
     * over the n_tokens rows). Score-routed layers only; δ≡0 skips the dispatch. */
    if (ok && model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present &&
        model->corr->layer[il].has_delta && layer->ffn_gate_tid2eid == NULL) {
        ok = ds4_gpu_corr_router_bias(g->batch_router_logits, model->corr->layer[il].gdelta,
                                      DS4_N_EXPERT, n_tokens) != 0;
    }

    /* 路由闭式侧车(type8): 批路同挂, select 前(2026-08-19) */
    if (ok && ds4_zchain_layer_rte(zch, il))
        ok = ds4_gpu_zchain_route_bias(g->batch_router_logits, g->batch_ffn_norm, il, n_tokens) != 0;
    if (ok) ok = ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens, il) != 0;
    /* 路由空槽消毒(必须在 sorted-pairs 记账之前): -1 槽会造成 counts[-1] 越界原子写 +
     * 该 pair 无 tile ⇒ mid 槽未初始化 ⇒ down 读垃圾(drafter 同输入两跑草稿全不同的真因)。 */
    if (ok) ok = ds4_gpu_sanitize_router_tensor(g->batch_router_selected, g->batch_router_weights,
                                                (uint32_t)n_tokens * DS4_N_EXPERT_USED,
                                                model_expert_kept_count(model, il)) != 0;
    if (ok) router_freq_collect(g->batch_router_logits, il, n_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("router");

    /* zchain GE: fold per-expert gains into the router weights (original ids;
     * before the compact-slot translate and before the go1b MoE's in-place remap). */
    if (ok && model->zchain && ds4_zchain_layer_ge(model->zchain, il)) {
        ok = ds4_gpu_zchain_ge_apply(g->batch_router_weights, g->batch_router_selected,
                                     il, DS4_N_EXPERT_USED, n_tokens) != 0;
    }
    /* Translate full-256 router ids to compact slots for a shrunken model (no-op
     * for a full model). All n_tokens rows share the same per-layer LUT. */
    if (ok && model->expert_shrunken) {
        ok = ds4_gpu_translate_expert_ids(g->batch_router_selected, il, DS4_N_EXPERT_USED,
                                          n_tokens, model_expert_kept_count(model, il)) != 0;
    }

    /* TP Phase-3 batch split (DS4_TP_EXPERT_SPLIT): each peer gathers/computes its
     * half of every token's routed experts; the partial batch_routed_out is
     * all-reduce SUMMED below. Ends the prefill redundancy (both peers had run the
     * FULL gather) -> halves the cold batch (prefill) expert IO. Covers the batched
     * paths (prefill chunks + verify batches); decode bare rounds use the single-
     * token split. Default OFF => full gather. */
    /* Separate gate from the single-token decode split: the batched path AR's once
     * per (chunk, layer) -> prefill issues thousands of ARs whose drain overhead
     * outweighs the IO halving (wave-72: 281s redundant < 688s split). Default OFF;
     * opt-in for cold-batch experiments. Single-token decode keeps DS4_TP_EXPERT_SPLIT. */
    const char *tp_es_env_b = getenv("DS4_TP_EXPERT_SPLIT_BATCH");
    const bool tp_batch_split =
        g->tp && il < g->tp_layers && tp_es_env_b && tp_es_env_b[0] && tp_es_env_b[0] != '0' &&
        DS4_N_EXPERT_USED >= 2u;
    uint32_t tpb_slot_start = 0u, tpb_slot_count = 0u;
    if (tp_batch_split) {
        const uint32_t k = DS4_N_EXPERT_USED / 2u;
        tpb_slot_start = g->tp_owns_low ? 0u : k;
        tpb_slot_count = g->tp_owns_low ? k : (DS4_N_EXPERT_USED - k);
    }
    /* 小批 MoE 快路(2026-08-21 投机战役): ≤8 tok 逐 token 走 decode 单 token 路
     * (lut_gate/direct_down_sum6, 实测 0.2ms/层/token vs tile8 批路 2.35ms/层)。
     * verify(6)/draft(5)/replay(1-3) 全吃到; prefill 大批不受影响。DS4_SPEC_MOE_BATCH=1 回退。 */
    if (ok && n_tokens <= 8u && !tp_batch_split && !model->expert_shrunken &&
        getenv("DS4_SPEC_MOE_PER_TOKEN") != NULL) {   /* 实验开关: 默认批路(逐token路 -21ms 但 acc -0.4 净亏) */
        for (uint32_t t = 0; ok && t < n_tokens; t++) {
            ds4_gpu_tensor *xv = metal_graph_tensor_row_view(g->batch_ffn_norm, t, DS4_N_EMBD);
            ds4_gpu_tensor *ov = metal_graph_tensor_row_view(g->batch_routed_out, t, DS4_N_EMBD);
            ds4_gpu_tensor *sv = metal_graph_tensor_row_view(g->batch_router_selected, t, DS4_N_EXPERT_USED);
            ds4_gpu_tensor *wv = metal_graph_tensor_row_view(g->batch_router_weights, t, DS4_N_EXPERT_USED);
            ok = xv && ov && sv && wv &&
                 ds4_gpu_routed_moe_one_tensor(ov,
                                               g->batch_routed_gate, g->batch_routed_up,
                                               g->batch_routed_mid, g->batch_routed_down,
                                               residual_set_for(model, il),
                                               model->map, model->size,
                                               routed_expert_gate_off(layer),
                                               routed_expert_up_off(layer),
                                               layer->ffn_down_exps->abs_offset,
                                               routed_expert_quant_type(layer),
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes, gate_row_bytes,
                                               down_expert_bytes, down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               sv, wv,
                                               model_expert_kept_count(model, il),
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               xv, il) != 0;
            ds4_gpu_tensor_free(wv); ds4_gpu_tensor_free(sv);
            ds4_gpu_tensor_free(ov); ds4_gpu_tensor_free(xv);
        }
        g->batch_routed_mid_is_f16 = false;   /* 单 token 路 mid 中间态与批口径无关 */
    } else if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
                                               residual_set_for(model, il),
                                               model->map,
                                               model->size,
                                               routed_expert_gate_off(layer),
                                               routed_expert_up_off(layer),
                                               layer->ffn_down_exps->abs_offset,
                                               routed_expert_quant_type(layer),
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               model_expert_kept_count(model, il),
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens,
                                               tpb_slot_start,
                                               tpb_slot_count,
                                               &g->batch_routed_mid_is_f16) != 0;
    }
    /* batch all-reduce: sum each peer's partial routed_out [n_tokens x n_embd] into
     * the full routed output (same MTLSharedEvent fast-wait path as the decode AR;
     * routed_moe_batch leaves the batch CB open on the was_batched path). */
    if (ok && tp_batch_split) {
        const uint64_t ar_n = (uint64_t)n_tokens * (uint64_t)DS4_N_EMBD;
        const uint64_t ev = ds4_gpu_tp_signal_after_batch();
        ok = ev != 0;
        if (ok) ok = ds4_gpu_flush_commands() != 0;
        if (ok) ok = ds4_gpu_tp_host_wait(ev) != 0;
        float *arbuf = ok ? malloc((size_t)ar_n * sizeof(float)) : NULL;
        if (ok && !arbuf) ok = false;
        if (ok) ok = ds4_gpu_tensor_read(g->batch_routed_out, 0, arbuf,
                                         ar_n * sizeof(float)) != 0;
        if (ok) ok = ds4_dist_tp_allreduce_f32(g->tp, arbuf, (uint32_t)ar_n) == 0;
        if (ok) ok = ds4_gpu_tensor_write(g->batch_routed_out, 0, arbuf,
                                          ar_n * sizeof(float)) != 0;
        free(arbuf);
    }
    /* zchain λ(x): scale the (fully summed) routed output before the additive
     * corr — quantizer op-order parity. */
    if (ok && (ds4_zchain_layer_has_lambda(model->zchain, il) ||
               ds4_zchain_layer_zl(model->zchain, il))) {   /* λ 和/或 冻结 z^L 同一派发 */
        ok = ds4_gpu_zchain_scale_routed(g->batch_routed_out, g->batch_ffn_norm,
                                         il, n_tokens) != 0;
    }
    /* Engine-trajectory batch capture (DS4_CAP_DIR [+DS4_CAP_LAYERS lo-hi]):
     * append this chunk's x̂ / routing / raw router logits / gate weights as
     * raw f16/i16 shards — THE ground-truth student trajectory for error-
     * feedback calibration (the Python fp32-backbone simulation drifts from
     * the engine in deep layers; R2 verdict). ffn_norm and router_logits are
     * final at this point; selected uses the same pre-remap snapshot the corr
     * dispatch uses. Off unless DS4_CAP_DIR is set. */
    if (ok) cap_batch_layer(g, il, n_tokens);
    /* go1b correction: per-token low-rank residual onto the full routed MoE output
     * (after any TP all-reduce). x=g->batch_ffn_norm, selected=g->batch_router_selected. */
    if (ok && model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present) {
        const ds4_corr_layer *cl = &model->corr->layer[il];
        /* Same remap hazard as decode: the go1b batch MoE rewrites batch_router_selected
         * to compact slots in place, so read the pre-remap snapshot (per-token original
         * ids, [n_tokens][n_expert_used]); fall back to the live tensor if not go1b. */
        const ds4_gpu_tensor *corr_sel = ds4_gpu_corr_saved_selected();
        if (!corr_sel) corr_sel = g->batch_router_selected;
        ok = ds4_gpu_corr_apply(g->batch_routed_out,
                                model->corr->phi_yhat ? g->batch_routed_out : g->batch_ffn_norm,
                                cl->gU, cl->gV, cl->gC, cl->gb, cl->gbeta,
                                corr_sel,
                                DS4_N_EMBD, cl->d_l, DS4_N_EXPERT, DS4_N_EXPERT_USED, n_tokens) != 0;
        if (getenv("DS4_RESIDUAL_DEBUG"))
            fprintf(stderr, "ds4: [corr-batch] L%u ok=%d d_l=%u ntok=%u gU=%p gC=%p\n",
                    il, ok, cl->d_l, n_tokens, (void*)cl->gU, (void*)cl->gC);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->batch_routed_gate,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            metal_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                              routed_mid_elems, il, pos0);
        } else {
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("routed_moe");
    /* shexp gate+up 批 pair 融合(2026-08-21): 同输入两矩阵一发 —— 小矩阵单发在 6-token
     * 批下只有 ~35GB/s(gx 小并行度不足), 合并后行数翻倍且权重只读一遍。不适用则回退。 */
    bool shexp_pair_done = false;
    if (ok && n_tokens > 1u &&
        layer->ffn_gate_shexp->type == DS4_TENSOR_Q2_K &&
        layer->ffn_up_shexp->type == DS4_TENSOR_Q2_K &&
        getenv("DS4_NO_SHEXP_PAIR") == NULL) {
        shexp_pair_done = ds4_gpu_matmul_q2_K_pair_batch_tensor(
                g->batch_shared_gate, g->batch_shared_up,
                model->map, model->size,
                layer->ffn_gate_shexp->abs_offset, layer->ffn_up_shexp->abs_offset,
                DS4_N_EMBD, shared_dim, shared_dim,
                g->batch_ffn_norm, n_tokens) != 0;
    }
    if (ok && !shexp_pair_done) ok = metal_graph_matmul_q8_0_named_tensor("shared_gate",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_gate,
                                                      model,
                                                      layer->ffn_gate_shexp,
                                                      DS4_N_EMBD,
                                                      shared_dim,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
    if (ok && !shexp_pair_done) ok = metal_graph_matmul_q8_0_named_tensor("shared_up",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_up,
                                                      model,
                                                      layer->ffn_up_shexp,
                                                      DS4_N_EMBD,
                                                      shared_dim,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
    DS4_METAL_PROFILE_FFN_STAGE("shared_gate_up");
    if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                         g->batch_shared_gate,
                                         g->batch_shared_up,
                                         (uint32_t)((uint64_t)n_tokens * shared_dim),
                                         DS4_SWIGLU_CLAMP_EXP,
                                         1.0f) != 0;
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_down",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_out,
                                                      model,
                                                      layer->ffn_down_shexp,
                                                      shared_dim,
                                                      DS4_N_EMBD,
                                                      g->batch_shared_mid,
                                                      n_tokens);
    DS4_METAL_PROFILE_FFN_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }

    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos0);
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_METAL_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
/* DS4_EVAL_HDUMP=<dir>: 每层出口 HC 隐状态整批追加写出 h_L%02d.bin
 * (f32, [S][DS4_N_HC][DS4_N_EMBD], chunk 顺序即 token 顺序)。
 * 挂在 HC 交换之后 —— 那时 batch_cur_hc 才是本层出口。GPU 定格用 cap_batch_layer
 * 同款 signal→flush→host_wait: 本层的核还在队列里, 不 drain 就 read 会拿到上一层
 * 残值(2026-07-22 那次 off-by-one 就是这么来的)。流式写盘, 不驻留。 */
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_prefill_ffn_nonempty_tu; /* 空TU防御(CPU构建) */
