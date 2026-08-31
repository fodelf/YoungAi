/* core_cpu_swa.c — CPU raw SWA 注意力 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
/* Batched prefill attention.  It projects Q/KV for all tokens, streams them
 * through the same raw/compressed cache updates, then runs prefix attention. */
void layer_attention_raw_swa_batch(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  n_tok,
        uint32_t                  il,
        uint32_t                  pos0,
        const float             * steering_dirs,
        float                     steering_scale) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_tok * hc_dim * sizeof(attn_residual[0]));
    const uint32_t q_rank = DS4_N_LORA_Q;
    float *qr = xmalloc((size_t)n_tok * q_rank * sizeof(qr[0]));
    float *qr_norm = xmalloc((size_t)n_tok * q_rank * sizeof(qr_norm[0]));
    float *q = xmalloc((size_t)n_tok * q_dim * sizeof(q[0]));
    float *kv_raw = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv_raw[0]));
    float *kv = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_out[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);
    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    hc_pre_norm_batch(model,
                      layer->hc_attn_fn,
                      layer->hc_attn_scale,
                      layer->hc_attn_base,
                      layer->attn_norm,
                      inp_hc,
                      attn_residual,
                      attn_cur,
                      attn_norm,
                      post,
                      comb,
                      n_tok);

    matmul_q8_0_batch(qr, model, layer->attn_q_a, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(qr_norm + (uint64_t)t * q_rank,
                        qr + (uint64_t)t * q_rank,
                        q_a_norm,
                        q_rank,
                        DS4_RMS_EPS);
    }
    matmul_q8_0_batch(q, model, layer->attn_q_b, qr_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        head_rms_norm_inplace(q + (uint64_t)t * q_dim,
                              DS4_N_HEAD,
                              DS4_N_HEAD_DIM,
                              DS4_RMS_EPS);
    }

    matmul_q8_0_batch(kv_raw, model, layer->attn_kv, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(kv + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_raw + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_norm,
                        DS4_N_HEAD_DIM,
                        DS4_RMS_EPS);
    }

    const uint32_t ratio = cache->compress_ratio;
    float *comp_scratch = NULL;
    float *index_comp_scratch = NULL;

    if (ratio != 0) {
        comp_scratch = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp_scratch[0]));

        if (ratio == 4) {
            index_comp_scratch = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp_scratch[0]));
        }
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        float *q_t = q + (uint64_t)t * q_dim;
        float *kv_t = kv + (uint64_t)t * DS4_N_HEAD_DIM;
        bool *comp_allowed = NULL;

        rope_tail_layer_inplace(q_t, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
        rope_tail_layer_inplace(kv_t, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
        dsv4_fp8_kv_quantize_row_inplace_cpu(kv_t, DS4_N_HEAD_DIM, DS4_N_ROT);

        kv_cache_push_raw(cache, kv_t);

        if (ratio != 0) {
            float *comp = comp_scratch;
            const bool have_comp = compressor_decode_one(comp, model,
                                                         layer->attn_compressor_kv,
                                                         layer->attn_compressor_gate,
                                                         layer->attn_compressor_ape,
                                                         layer->attn_compressor_norm,
                                                         attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                         cache->attn_state_kv,
                                                         cache->attn_state_score,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         il,
                                                         pos);
            if (have_comp) {
                kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
            }

            if (ratio == 4) {
                float *index_comp = index_comp_scratch;
                const bool have_index_comp = compressor_decode_one(index_comp, model,
                                                                   layer->indexer_compressor_kv,
                                                                   layer->indexer_compressor_gate,
                                                                   layer->indexer_compressor_ape,
                                                                   layer->indexer_compressor_norm,
                                                                   attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                                   cache->index_state_kv,
                                                                   cache->index_state_score,
                                                                   DS4_N_INDEXER_HEAD_DIM,
                                                                   ratio,
                                                                   il,
                                                                   pos);
                if (have_index_comp) {
                    kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
                }

                comp_allowed = indexer_allowed_decode_one(model, layer,
                                                          attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                          qr_norm + (uint64_t)t * q_rank,
                                                          cache->index_comp_kv,
                                                          cache->n_index_comp,
                                                          il, pos);
            }

            layer_attention_mixed_one(heads + (uint64_t)t * q_dim, model, layer, q_t,
                                      cache->raw_kv, cache->n_raw,
                                      cache->attn_comp_kv, cache->n_comp,
                                      comp_allowed);
        } else {
            layer_attention_rows_one(heads + (uint64_t)t * q_dim, model, layer, q_t, cache->raw_kv, cache->n_raw);
        }

        rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                DS4_N_HEAD,
                                DS4_N_HEAD_DIM,
                                DS4_N_ROT,
                                pos,
                                il,
                                true);

        free(comp_allowed);
    }

    layer_grouped_out_batch(attn_out, model, layer, heads, n_tok);
    cpu_directional_steering_project_rows(attn_out, steering_dirs, il, n_tok, steering_scale);

    hc_post_batch(after_attn_hc,
                  attn_out,
                  attn_residual,
                  post,
                  comb,
                  n_tok,
                  DS4_N_EMBD,
                  n_hc);

    free(index_comp_scratch);
    free(comp_scratch);
    free(comb);
    free(post);
    free(attn_out);
    free(heads);
    free(kv);
    free(kv_raw);
    free(q);
    free(qr_norm);
    free(qr);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

/* Full transformer layer for one decode token: attention sublayer followed by
 * FFN sublayer, both operating on the HC state. */
