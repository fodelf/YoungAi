/* core_gpu_head.c — 输出头编码/plain matmul (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->flat_hc,
                                             1) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = dense_matmul_typed(g->logits, model, weights->output,
                                      DS4_N_EMBD, vocab_dim, g->output_norm, 1) != 0;
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
bool metal_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;
    ds4_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = dense_matmul_typed(logits, model, weights->output,
                                      DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: Metal plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

bool metal_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = dense_matmul_typed(out, model, w, in_dim, out_dim, x, n_tok) != 0;
    return ok;
}

#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_head_nonempty_tu; /* 空TU防御(CPU构建) */
