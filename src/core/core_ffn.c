/* core_ffn.c — layer FFN one/batch/tokens (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
void layer_ffn_one(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        uint32_t            il,
        int                 token,
        const float       * steering_dirs,
        float               steering_scale,
        bool                trace) {
    const uint32_t n_hc = DS4_N_HC;
    float *ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)DS4_N_EMBD * sizeof(shared[0]));
    float *ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_out[0]));
    float post[4];
    float comb[16];

    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          inp_hc, ffn_cur, post, comb);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_cur", il);
        print_vec_stats(name, ffn_cur, DS4_N_EMBD);
    }

    const float *ffn_norm = tensor_data(model, layer->ffn_norm);
    rms_norm_weight(norm, ffn_cur, ffn_norm, DS4_N_EMBD, DS4_RMS_EPS);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_norm", il);
        print_vec_stats(name, norm, DS4_N_EMBD);
    }

    layer_routed_moe_one(moe, model, layer, norm, il, token, DS4_SWIGLU_CLAMP_EXP, trace);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u routed_moe", il);
        print_vec_stats(name, moe, DS4_N_EMBD);
    }
    layer_shared_ffn_one(shared, model, layer, norm);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u shared_ffn", il);
        print_vec_stats(name, shared, DS4_N_EMBD);
    }

    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
        ffn_out[i] = moe[i] + shared[i];
    }
    cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, 1, steering_scale);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_out", il);
        print_vec_stats(name, ffn_out, DS4_N_EMBD);
    }

    hc_post_one(out_hc, ffn_out, inp_hc, post, comb, DS4_N_EMBD, n_hc);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_post_hc", il);
        print_vec_stats(name, out_hc, (uint64_t)n_hc * DS4_N_EMBD);
    }

    free(ffn_out);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

/* Allocation-free decode FFN using the persistent CPU scratch buffers. */
void layer_ffn_one_decode_scratch(
        float                  * out_hc,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * inp_hc,
        uint32_t                 il,
        int                      token,
        const float            * steering_dirs,
        float                    steering_scale,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    float post[4];
    float comb[16];

    hc_pre_from_state_one_scratch(model,
                                  layer->hc_ffn_fn,
                                  layer->hc_ffn_scale,
                                  layer->hc_ffn_base,
                                  inp_hc, scratch->ffn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);

    const float *ffn_norm = tensor_data(model, layer->ffn_norm);
    rms_norm_weight(scratch->ffn_norm, scratch->ffn_cur, ffn_norm, DS4_N_EMBD, DS4_RMS_EPS);

    layer_routed_moe_one_prealloc(scratch->ffn_moe,
                                  model,
                                  layer,
                                  scratch->ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  scratch->routed_mid_all,
                                  scratch->routed_xq,
                                  scratch->routed_midq);

    layer_shared_ffn_one_decode_scratch(scratch->ffn_shared, model, layer, scratch->ffn_norm, scratch);

    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
        scratch->ffn_out[i] = scratch->ffn_moe[i] + scratch->ffn_shared[i];
    }
    cpu_directional_steering_project_rows(scratch->ffn_out, steering_dirs, il, 1, steering_scale);
    hc_post_one(out_hc, scratch->ffn_out, inp_hc, post, comb, DS4_N_EMBD, n_hc);
}

typedef struct {
    float *moe;
    const ds4_model *model;
    const ds4_layer_weights *layer;
    const float *norm;
    const int *token_ids;
    uint64_t expert_in_dim;
    uint64_t down_in_dim;
    uint32_t il;
} routed_moe_tokens_ctx;

static void routed_moe_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    routed_moe_tokens_ctx *ctx = vctx;
    float *routed_mid = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = xmalloc((size_t)(ctx->expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (ctx->down_in_dim / QK_K) * sizeof(routed_midq[0]));

    for (uint64_t t = t0; t < t1; t++) {
        layer_routed_moe_one_prealloc(ctx->moe + t * DS4_N_EMBD,
                                      ctx->model,
                                      ctx->layer,
                                      ctx->norm + t * DS4_N_EMBD,
                                      ctx->il,
                                      ctx->token_ids[t],
                                      DS4_SWIGLU_CLAMP_EXP,
                                      routed_mid,
                                      routed_xq,
                                      routed_midq);
    }

    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
}

static void layer_routed_moe_tokens_parallel(
        float             * moe,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il) {
    routed_moe_tokens_ctx ctx = {
        .moe = moe,
        .model = model,
        .layer = layer,
        .norm = norm,
        .token_ids = token_ids,
        .expert_in_dim = routed_expert_in_dim(layer),
        .down_in_dim = layer->ffn_down_exps->dim[0],
        .il = il,
    };
    ds4_parallel_for_min_rows(n_tok, routed_moe_tokens_worker, &ctx, 1);
}

/* Default prefill FFN path.  HC and shared expert are batched, while routed
 * experts can run either token-parallel or expert-grouped depending on size. */
void layer_ffn_shared_batch(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    const uint32_t n_hc = DS4_N_HC;
    float *ffn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(shared[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_token_parallel = n_tok >= 64;
    float *routed_mid = routed_token_parallel ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = routed_token_parallel ? NULL : xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = routed_token_parallel ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(routed_midq[0]));

    hc_pre_norm_batch(model,
                      layer->hc_ffn_fn,
                      layer->hc_ffn_scale,
                      layer->hc_ffn_base,
                      layer->ffn_norm,
                      inp_hc,
                      NULL,
                      ffn_cur,
                      norm,
                      post,
                      comb,
                      n_tok);

    if (routed_token_parallel) {
        layer_routed_moe_tokens_parallel(moe, model, layer, norm, token_ids, n_tok, il);
    } else {
        for (uint32_t t = 0; t < n_tok; t++) {
            layer_routed_moe_one_prealloc(moe + (uint64_t)t * DS4_N_EMBD,
                                          model,
                                          layer,
                                          norm + (uint64_t)t * DS4_N_EMBD,
                                          il,
                                          token_ids[t],
                                          DS4_SWIGLU_CLAMP_EXP,
                                          routed_mid,
                                          routed_xq,
                                          routed_midq);
        }
    }

    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);

    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * DS4_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      DS4_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          DS4_N_EMBD,
                          n_hc);
    }

    free(comb);
    free(post);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}


