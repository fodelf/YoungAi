/* core_gpu_prefill_util.c — prefill 杂项/embedding 上载/warmup (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
/* =========================================================================
 * Metal Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing Metal backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

static bool metal_graph_env_bool_enabled(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && !(v[0] == '0' && v[1] == '\0');
}

bool metal_graph_direct_expert_read_enabled(void) {
    return metal_graph_env_bool_enabled("DS4_METAL_EXPERT_OFFLOAD_DIRECT");
}

uint32_t metal_graph_token_split_after_layers(void) {
    uint32_t split_after_layers = metal_graph_direct_expert_read_enabled() ? 1u : 4u;
    const char *split_env = getenv("DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }
    return split_after_layers;
}

/* Encode a full single-token decode step on Metal.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
bool metal_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);

    bool ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    const uint32_t split_after_layers = metal_graph_token_split_after_layers();

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        /* DSpark: target 层输出 HC 均值 → main_hidden[slot](官方 h.mean(dim=2) 语义;
         * 0731 固定 dspark_target_layer_ids=[40,41,42]) */
        if (ok && g->dspark_capture && g->dspark_main_hidden &&
            il >= 40u && il <= 42u) {
            ok = ds4_gpu_dspark_hc_mean_tensor(g->dspark_main_hidden, g->cur_hc,
                                               DS4_N_EMBD, DS4_N_HC, il - 40u, 1) != 0;
        }
        if (ok && allow_split_flush && split_after_layers != 0 &&
            ((il + 1u) % split_after_layers) == 0 && il + 1u < (uint32_t)DS4_N_LAYER) {
            if (metal_graph_direct_expert_read_enabled()) {
                ok = ds4_gpu_end_commands() != 0 && ds4_gpu_begin_commands() != 0;
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
        }
    }

    if (ok && need_logits) {
        ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}

ds4_gpu_tensor *metal_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
bool metal_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
bool metal_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                         model->map,
                                         model->size,
                                         kv_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                         model->map,
                                         model->size,
                                         score_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool metal_graph_upload_prompt_embeddings_hc_cpu(
        ds4_gpu_tensor   *out_hc,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    float *hc = xmalloc((size_t)total * sizeof(hc[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        float *dst = hc + (uint64_t)t * hc_dim;
        for (uint32_t h = 0; h < DS4_N_HC; h++) {
            memcpy(dst + (uint64_t)h * DS4_N_EMBD,
                   plain,
                   (size_t)DS4_N_EMBD * sizeof(plain[0]));
        }
    }

    const bool ok = ds4_gpu_tensor_write(out_hc, 0, hc, total * sizeof(hc[0])) != 0;
    free(plain);
    free(hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the Metal get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
bool metal_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    /* 默认 1(2026-08-21): CPU 侧 embed_token_f16 与 decode 路的 GPU embed kernel 是两套
     * 解量化实现, 同一 token 差 ~2e-4 —— 这是批 verify 与 decode 分叉的 L0 种子, 实测把
     * 批/解码 argmax 一致率从 91.7% 抬到 97.9%。批走 GPU 路即与 decode 同源。 */
    uint32_t gpu_min = 1;
    const char *gpu_min_env = getenv("DS4_METAL_GPU_BATCH_EMBED_MIN");
    if (gpu_min_env && gpu_min_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(gpu_min_env, &end, 10);
        if (end != gpu_min_env && v <= UINT32_MAX) gpu_min = (uint32_t)v;
    }

    if (tokens && n_tokens >= gpu_min) {
        return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                                tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                n_tokens,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
    }

    return metal_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
}

bool metal_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed || getenv("DS4_METAL_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay Metal's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_hc_mix,
                                         model->map,
                                         model->size,
                                         weights->layer[0].hc_attn_fn->abs_offset,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: Metal prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: metal indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current Metal
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

bool metal_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_prefill_util_nonempty_tu; /* 空TU防御(CPU构建) */
