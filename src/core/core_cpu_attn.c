/* core_cpu_attn.c — CPU 压缩注意力/indexer (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
bool compressor_decode_one_decode_scratch(
        float                  * out_comp,
        const ds4_model        * model,
        const ds4_tensor       * wkv,
        const ds4_tensor       * wgate,
        const ds4_tensor       * ape,
        const ds4_tensor       * norm,
        const float            * x,
        float                  * state_kv,
        float                  * state_score,
        uint32_t                 head_dim,
        uint32_t                 compress_ratio,
        uint32_t                 il,
        uint32_t                 pos,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % compress_ratio;
    const uint32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = ((pos + 1) % compress_ratio) == 0;

    if (width > 2u * DS4_N_HEAD_DIM) ds4_die("compressor scratch width is outside the fixed model layout");
    float *kv_cur = scratch->comp_kv_cur;
    float *sc_cur = scratch->comp_sc_cur;

    if (wkv->type == 8 &&
        wgate->type == 8 &&
        wkv->ndim == 2 &&
        wgate->ndim == 2 &&
        wkv->dim[0] == wgate->dim[0]) {
        matvec_q8_0_pair_decode_scratch(kv_cur, sc_cur, model, wkv, wgate, x, scratch);
    } else {
        matvec_any_decode_scratch(kv_cur, model, wkv, x, scratch);
        matvec_any_decode_scratch(sc_cur, model, wgate, x, scratch);
    }

    for (uint32_t j = 0; j < width; j++) {
        sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod);
    }

    memcpy(state_kv + (uint64_t)row * width, kv_cur, (size_t)width * sizeof(kv_cur[0]));
    memcpy(state_score + (uint64_t)row * width, sc_cur, (size_t)width * sizeof(sc_cur[0]));

    if (!should_compress) {
        return false;
    }

    float *pooled = scratch->comp_pooled;
    compressor_pool_decode_state(pooled, state_kv, state_score, head_dim, compress_ratio);

    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    const float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_RMS_EPS);
    for (uint32_t i = 0; i < head_dim; i++) {
        out_comp[i] = pooled[i] * rms * tensor_1d_value(model, norm, i);
    }

    const uint32_t comp_pos = pos + 1 - compress_ratio;
    rope_tail_layer_inplace(out_comp, 1, head_dim, DS4_N_ROT, comp_pos, il, false);
    if (head_dim == DS4_N_HEAD_DIM) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT);
    } else if (head_dim == DS4_N_INDEXER_HEAD_DIM) {
        dsv4_indexer_qat_row_inplace_cpu(out_comp, head_dim);
    }

    if (compress_ratio == 4) {
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)r * width,
                   state_kv + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)r * width,
                   state_score + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_score[0]));
        }
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)(compress_ratio + r) * width,
                   state_kv + (uint64_t)r * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)(compress_ratio + r) * width,
                   state_score + (uint64_t)r * width,
                   (size_t)width * sizeof(state_score[0]));
        }
    }

    return true;
}

/* Attention over raw SWA rows plus optional compressed rows.  Ratio-4 layers
 * pass an indexer mask to hide compressed rows not selected for this token. */
void layer_attention_mixed_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * raw_kv,
        uint32_t            n_raw,
        const float       * comp_kv,
        uint32_t            n_comp,
        const bool        * comp_allowed) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t n_total = n_raw + n_comp;
    float score_stack[512];
    float *score = n_total <= 512 ? score_stack : xmalloc((size_t)n_total * sizeof(score[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;
        float max_score = sinks[h];
        uint32_t idx = 0;

        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) {
                score[idx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float weight = expf(score[idx] - max_score);
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[idx] - max_score);
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

void layer_attention_mixed_one_decode_scratch(
        float                  * out_heads,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * q,
        const float            * raw_kv,
        uint32_t                 n_raw,
        const float            * comp_kv,
        uint32_t                 n_comp,
        const bool             * comp_allowed,
        ds4_cpu_decode_scratch * scratch) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t n_total = n_raw + n_comp;
    if (n_total > scratch->attn_score_cap) ds4_die("CPU decode attention score scratch buffer is too small");
    float *score = scratch->attn_score;

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;
        float max_score = sinks[h];
        uint32_t idx = 0;

        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) {
                score[idx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float weight = expf(score[idx] - max_score);
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[idx] - max_score);
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }
}

/* Ratio-4 layers use an auxiliary indexer to select which compressed rows are
 * visible to attention.  This is the CPU allocation-owning helper. */
bool *indexer_allowed_decode_one(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos) {
    if (n_comp == 0) return NULL;

    bool *allowed = xcalloc(n_comp, sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = xmalloc((size_t)head_dim * n_head * sizeof(q[0]));
    float *weights = xmalloc((size_t)n_head * sizeof(weights[0]));
    float *scores = xmalloc((size_t)n_comp * sizeof(scores[0]));

    matvec_any(q, model, layer->indexer_attn_q_b, qr_norm);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);
    dsv4_indexer_qat_rows_inplace_cpu(q, n_head, head_dim);

    matvec_any(weights, model, layer->indexer_proj, cur);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    free(scores);
    free(weights);
    free(q);
    return allowed;
}

/* Scratch-backed indexer selection for decode. */
bool *indexer_allowed_decode_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos,
        ds4_cpu_decode_scratch  * scratch) {
    if (n_comp == 0) return NULL;
    if (n_comp > scratch->comp_cap) ds4_die("CPU decode indexer scratch buffer is too small");

    bool *allowed = scratch->index_allowed;
    memset(allowed, 0, (size_t)n_comp * sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = scratch->index_q;
    float *weights = scratch->index_weights;
    float *scores = scratch->index_scores;

    matvec_any_decode_scratch(q, model, layer->indexer_attn_q_b, qr_norm, scratch);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);
    dsv4_indexer_qat_rows_inplace_cpu(q, n_head, head_dim);

    matvec_any_decode_scratch(weights, model, layer->indexer_proj, cur, scratch);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    return allowed;
}

/* Single-token attention sublayer with raw SWA cache and DS4 compression. */
