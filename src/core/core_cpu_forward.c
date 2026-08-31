/* core_cpu_forward.c — CPU 逐层前向 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static void layer_forward_raw_swa_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token,
        const float             * steering_dirs,
        float                     steering_attn_scale,
        float                     steering_ffn_scale,
        ds4_cpu_decode_scratch  * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    memcpy(scratch->attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one_scratch(model,
                                  layer->hc_attn_fn,
                                  layer->hc_attn_scale,
                                  layer->hc_attn_base,
                                  scratch->attn_residual, scratch->attn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);
    layer_attn_norm_one(scratch->attn_norm, model, layer, scratch->attn_cur);
    const uint32_t ratio = cache->compress_ratio;
    layer_q_projection_with_lora_one_decode_scratch(model, layer,
                                                    scratch->attn_norm,
                                                    scratch->q,
                                                    scratch->qr_norm,
                                                    scratch);
    layer_kv_projection_normed_one_decode_scratch(model, layer,
                                                  scratch->attn_norm,
                                                  scratch->kv,
                                                  scratch);
    rope_tail_layer_inplace(scratch->q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(scratch->kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(scratch->kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, scratch->kv);

    if (ratio != 0) {
        if (compressor_decode_one_decode_scratch(scratch->comp, model,
                                                 layer->attn_compressor_kv,
                                                 layer->attn_compressor_gate,
                                                 layer->attn_compressor_ape,
                                                 layer->attn_compressor_norm,
                                                 scratch->attn_norm,
                                                 cache->attn_state_kv,
                                                 cache->attn_state_score,
                                                 DS4_N_HEAD_DIM,
                                                 ratio,
                                                 il,
                                                 pos,
                                                 scratch)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, scratch->comp);
        }

        if (ratio == 4) {
            if (compressor_decode_one_decode_scratch(scratch->index_comp, model,
                                                     layer->indexer_compressor_kv,
                                                     layer->indexer_compressor_gate,
                                                     layer->indexer_compressor_ape,
                                                     layer->indexer_compressor_norm,
                                                     scratch->attn_norm,
                                                     cache->index_state_kv,
                                                     cache->index_state_score,
                                                     DS4_N_INDEXER_HEAD_DIM,
                                                     ratio,
                                                     il,
                                                     pos,
                                                     scratch)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap,
                                   DS4_N_INDEXER_HEAD_DIM, scratch->index_comp);
            }
        }
    }
    if (ratio == 4) {
        comp_allowed = indexer_allowed_decode_one_decode_scratch(model, layer,
                                                                 scratch->attn_norm,
                                                                 scratch->qr_norm,
                                                                 cache->index_comp_kv,
                                                                 cache->n_index_comp,
                                                                 il, pos,
                                                                 scratch);
    }

    if (ratio != 0) {
        layer_attention_mixed_one_decode_scratch(scratch->heads, model, layer, scratch->q,
                                                 cache->raw_kv, cache->n_raw,
                                                 cache->attn_comp_kv, cache->n_comp,
                                                 comp_allowed,
                                                 scratch);
    } else {
        layer_attention_rows_one(scratch->heads, model, layer, scratch->q, cache->raw_kv, cache->n_raw);
    }

    rope_tail_layer_inplace(scratch->heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one_decode_scratch(scratch->attn_out, model, layer, scratch->heads, scratch);
    cpu_directional_steering_project_rows(scratch->attn_out, steering_dirs, il, 1, steering_attn_scale);
    hc_post_one(scratch->after_attn_hc, scratch->attn_out, scratch->attn_residual, post, comb, DS4_N_EMBD, n_hc);

    layer_ffn_one_decode_scratch(out_hc, model, layer, scratch->after_attn_hc, il, token,
                                 steering_dirs, steering_ffn_scale, scratch);
}

/* CPU decode for one token through all 43 layers.  The caller owns scratch and
 * cache lifetimes so no per-token allocations are needed. */
void forward_token_raw_swa_cpu_decode_scratch(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos,
        const float       * steering_dirs,
        float               steering_attn_scale,
        float               steering_ffn_scale,
        ds4_cpu_decode_scratch * scratch) {
    float *cur = scratch->cur;
    float *next = scratch->next;

    embed_token_f16(model, weights, token, scratch->plain);
    hc_from_plain_embedding(cur, scratch->plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_raw_swa_one(next, model, &weights->layer[il], &cache->layer[il],
                                  cur, il, pos, token,
                                  steering_dirs,
                                  steering_attn_scale,
                                  steering_ffn_scale,
                                  scratch);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    if (logits) {
        output_logits_one_decode_scratch(logits, model, weights, cur, scratch);
    }
}

#ifndef DS4_NO_GPU
void forward_token_raw_swa_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos) {
    ds4_cpu_decode_scratch scratch;
    uint32_t ctx_guess = pos + 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = cache->layer[il].compress_ratio;
        if (ratio != 0 && cache->layer[il].comp_cap > 2) {
            const uint32_t ctx_from_comp = (cache->layer[il].comp_cap - 2u) * ratio;
            if (ctx_guess < ctx_from_comp) ctx_guess = ctx_from_comp;
        }
    }
    cpu_decode_scratch_init(&scratch, ctx_guess);
    forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, cache, token, pos,
                                             NULL, 0.0f, 0.0f, &scratch);
    cpu_decode_scratch_free(&scratch);
}
#endif

/* CPU prefill in layer-major order.  All prompt tokens pass through layer 0,
 * then layer 1, etc., which exposes batch matmul opportunities. */
void prefill_layer_major_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        const token_vec   * prompt,
        const float       * steering_dirs,
        float               steering_attn_scale,
        float               steering_ffn_scale) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t n_tok = (uint64_t)prompt->len;
    float *cur = xmalloc((size_t)n_tok * hc_dim * sizeof(cur[0]));
    float *next = xmalloc((size_t)n_tok * hc_dim * sizeof(next[0]));
    float *attn = xmalloc((size_t)n_tok * hc_dim * sizeof(attn[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint64_t t = 0; t < n_tok; t++) {
        embed_token_f16(model, weights, prompt->v[t], plain);
        hc_from_plain_embedding(cur + t * hc_dim, plain, DS4_N_EMBD, DS4_N_HC);
    }

    free(plain);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        fprintf(stderr, "ds4: prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
        fflush(stderr);

        layer_attention_raw_swa_batch(attn,
                                      model,
                                      &weights->layer[il],
                                      &cache->layer[il],
                                      cur,
                                      (uint32_t)n_tok,
                                      il,
                                      0,
                                      steering_dirs,
                                      steering_attn_scale);
        layer_ffn_shared_batch(next,
                               model,
                               &weights->layer[il],
                               attn,
                               prompt->v,
                               (uint32_t)n_tok,
                               il,
                               steering_dirs,
                               steering_ffn_scale);

        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    kv_cache_finish_prefill_states(cache, (uint32_t)n_tok);

    if (logits) {
        output_logits_one(logits, model, weights, cur + (n_tok - 1) * hc_dim);
    }

    free(next);
    free(cur);
    free(attn);
}

/* Diagnostic first-token layer without cache history: the token attends only
 * to itself, useful for checking a minimal end-to-end slice. */
