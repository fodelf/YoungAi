/* core_ffn_shared.c — 共享专家 FFN + 路由选择 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
/* =========================================================================
 * Mixture-of-Experts FFN.
 * =========================================================================
 *
 * This is the FFN half of each layer.  It includes the shared expert, routed
 * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
 * and the HC post step that returns the result to four-stream state.
 */

/* The shared expert is a normal Q8_0 SwiGLU MLP that runs for every token. */
void layer_shared_ffn_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    float *gate = xmalloc((size_t)DS4_N_FF_EXP * sizeof(gate[0]));
    float *up = xmalloc((size_t)DS4_N_FF_EXP * sizeof(up[0]));
    float *mid = xmalloc((size_t)DS4_N_FF_EXP * sizeof(mid[0]));
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        ds4_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_pair_prequant(gate, up, model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              xq, xscale);
    swiglu(mid, gate, up, DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0(out, model, layer->ffn_down_shexp, mid);

    free(xscale);
    free(xq);
    free(mid);
    free(up);
    free(gate);
}

void layer_shared_ffn_one_decode_scratch(
        float                  * out,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        ds4_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    matvec_q8_0_pair_decode_scratch(scratch->shared_gate,
                                    scratch->shared_up,
                                    model,
                                    layer->ffn_gate_shexp,
                                    layer->ffn_up_shexp,
                                    x,
                                    scratch);
    swiglu(scratch->shared_mid, scratch->shared_gate, scratch->shared_up, DS4_N_FF_EXP,
           DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0_decode_scratch(out, model, layer->ffn_down_shexp, scratch->shared_mid, scratch);
}

typedef struct {
    float *mid;
    const float *gate;
    const float *up;
    uint64_t n;
    float clamp;
} swiglu_batch_ctx;

static void swiglu_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    swiglu_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        swiglu(ctx->mid + t * ctx->n,
               ctx->gate + t * ctx->n,
               ctx->up + t * ctx->n,
               ctx->n,
               ctx->clamp);
    }
}

void layer_shared_ffn_batch(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            n_tok) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t hidden = layer->ffn_gate_shexp->dim[1];

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_down_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim ||
        layer->ffn_up_shexp->dim[1] != hidden ||
        layer->ffn_down_shexp->dim[0] != hidden) {
        ds4_die("shared expert tensors do not share the expected Q8_0 layout");
    }

    float *gate = xmalloc((size_t)n_tok * hidden * sizeof(gate[0]));
    float *up = xmalloc((size_t)n_tok * hidden * sizeof(up[0]));
    float *mid = xmalloc((size_t)n_tok * hidden * sizeof(mid[0]));

    matmul_q8_0_pair_batch(gate, up, model,
                           layer->ffn_gate_shexp,
                           layer->ffn_up_shexp,
                           x,
                           n_tok);

    swiglu_batch_ctx swiglu_ctx = {
        .mid = mid,
        .gate = gate,
        .up = up,
        .n = hidden,
        .clamp = DS4_SWIGLU_CLAMP_EXP,
    };
    ds4_parallel_for(n_tok, swiglu_batch_worker, &swiglu_ctx);

    matmul_q8_0_batch(out, model, layer->ffn_down_shexp, mid, n_tok);

    free(mid);
    free(up);
    free(gate);
}

/* Early DS4 layers use token-id hash routing instead of top-k routing. */
void layer_hash_selected_experts(
        int                    selected[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        int                    token) {
    ds4_tensor *t = layer->ffn_gate_tid2eid;
    if (!t) ds4_die("hash routing table is missing for this layer");
    if (t->type != 26 || t->ndim != 2 || t->dim[0] != DS4_N_EXPERT_USED) {
        ds4_die("ffn_gate_tid2eid.weight has an unexpected layout");
    }
    if (token < 0 || (uint64_t)token >= t->dim[1]) {
        ds4_die("token id is outside the hash routing table");
    }

    const int32_t *table = tensor_data(model, t);
    const int32_t *row = table + (uint64_t)token * DS4_N_EXPERT_USED;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) selected[i] = row[i];
}

/* Router scores use sqrt(softplus(logit)); normalization happens only after
 * the six selected experts are known. logit_bias (go1b corr delta, or NULL) is
 * added to the raw logits before softplus/sqrt, matching the GPU corr path. */
static void layer_router_probs_one(
        float             probs[DS4_MAX_EXPERT],
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        const float       * logit_bias) {
    float logits[DS4_MAX_EXPERT];

    matvec_f16(logits, model, layer->ffn_gate_inp, x);
    if (logit_bias) {
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) logits[i] += logit_bias[i];
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
}

static void layer_hash_router_weights_from_probs(
        float             weights_out[DS4_MAX_EXPERT_USED],
        const float       probs[DS4_MAX_EXPERT],
        const int          selected[DS4_MAX_EXPERT_USED]) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= DS4_N_EXPERT) ds4_die("hash-selected expert is outside router range");
        weights_out[i] = probs[selected[i]];
        sum += weights_out[i];
    }

    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        weights_out[i] = weights_out[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}

void layer_hash_router_weights_one(
        float             weights_out[DS4_MAX_EXPERT_USED],
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        const int          selected[DS4_MAX_EXPERT_USED]) {
    float probs[DS4_MAX_EXPERT];

    layer_router_probs_one(probs, model, layer, x, NULL);
    layer_hash_router_weights_from_probs(weights_out, probs, selected);
}

static void topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

/* Later layers choose the six experts by biased top-k, but weight them using
 * the unbiased router probabilities. */
static void layer_topk_selected_experts_from_probs(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           probs[DS4_MAX_EXPERT]);

void layer_topk_selected_experts(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           *x,
        const float           *logit_bias) {
    float probs[DS4_MAX_EXPERT];

    layer_router_probs_one(probs, model, layer, x, logit_bias);
    layer_topk_selected_experts_from_probs(selected, expert_weight, model, layer, probs);
}

static void layer_topk_selected_experts_from_probs(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           probs[DS4_MAX_EXPERT]) {
    float selection[DS4_MAX_EXPERT];

    memcpy(selection, probs, sizeof(selection));

    if (layer->ffn_exp_probs_b) {
        const float *bias = tensor_data(model, layer->ffn_exp_probs_b);
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) selection[i] += bias[i];
    }

    topk_desc(selection, (int)DS4_N_EXPERT, (int)DS4_N_EXPERT_USED, selected);

    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = probs[selected[i]];
        sum += expert_weight[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = expert_weight[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}


/* Single-token routed MoE.  It selects six experts, runs IQ2_XXS gate/up,
 * applies SwiGLU and router weights, then accumulates Q2_K down projections. */
