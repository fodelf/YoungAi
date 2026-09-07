/* core_gpu_prefill_attn.c — prefill 注意力批编码 (机械拆分自 ds4.c, 重构阶段4)。 */
/* EXCEPTION(>500行): 单函数 metal_graph_encode_layer_attention_batch_stages, 函数内拆分是后续工序(需真模型逐位闸) */
#include "core_internal.h"
#ifndef DS4_NO_GPU
/* 小批 kv 入环(09-07): 解码同款 rope+fp8+环存三合一核逐 token 发(数值与单 token 解码同序), 大批/无 rope 段走原批存。
 * 必须排在 spec 原始行快照之后(快照要的是入环前的旧行)。 */
static bool batch_store_kv(ds4_gpu_graph *g, uint32_t il, uint32_t pos0, uint32_t n_tokens, bool fused, bool compressed,
                           float freq_base, float freq_scale, float ext_factor, float attn_factor) {
    if (!fused)
        return ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il], g->batch_kv, g->raw_cap, pos0, n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
    return ds4_gpu_kv_rope_fp8_store_raw_batch_tensor(g->batch_kv, g->layer_raw_cache[il], g->raw_cap, pos0, n_tokens,
                                                      DS4_N_HEAD_DIM, DS4_N_ROT, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                      freq_base, freq_scale, ext_factor, attn_factor,
                                                      DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
}

bool metal_graph_encode_layer_attention_batch_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                stages) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    const bool small_batch = n_tokens <= 8u;   /* 投机 verify 批: 核选择全按解码同款(同轨) */
    /* 压缩器走解码攒行机制只给续接小批(verify); 从 0 起的短提示仍走批版 prefill(整块回放 + 余行入 state) */
    const bool small_comp = small_batch && !zero_prefix;
    if (ok && (stages & DS4_ATTN_STAGE_PRE)) {
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_attn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    /* 小批(投机 verify ≤8 token, 09-07): hc 混合 + attn_norm 走解码同款融合核(wsn_fast), 与纯解码同序 ⇒ 同轨;
     * 大批仍是分开的批核。 */
    if (ok && small_batch) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view, g->batch_attn_norm, hc_split_view, hc_mix_view,
                                                         g->batch_cur_hc, model->map, model->size,
                                                         layer->hc_attn_scale->abs_offset, layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset, DS4_N_EMBD, DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS) != 0;
    } else {
    if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                        hc_split_view,
                                                        hc_mix_view,
                                                        g->batch_cur_hc,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;

    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                       g->batch_attn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->attn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    }
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      g->batch_qr,
                                                      model,
                                                      layer->attn_q_a,
                                                      DS4_N_EMBD,
                                                      q_rank,
                                                      g->batch_attn_norm,
                                                      n_tokens);
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_kv",
                                                      il,
                                                      pos0,
                                                      g->batch_kv_raw,
                                                      model,
                                                      layer->attn_kv,
                                                      DS4_N_EMBD,
                                                      DS4_N_HEAD_DIM,
                                                      g->batch_attn_norm,
                                                      n_tokens);
    if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                         g->batch_qr,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_q_a_norm->abs_offset,
                                                         (uint32_t)q_rank,
                                                         g->batch_kv,
                                                         g->batch_kv_raw,
                                                         layer->attn_kv_a_norm->abs_offset,
                                                         DS4_N_HEAD_DIM,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_b",
                                                      il,
                                                      pos0,
                                                      g->batch_q,
                                                      model,
                                                      layer->attn_q_b,
                                                      q_rank,
                                                      q_dim,
                                                      g->batch_qr_norm,
                                                      n_tokens);
    /* 小批: q 的 head_rms+rope 走解码的融合核(scale 折进旋转的序与单 token 同) */
    const bool fused_q = small_batch && !(stages & DS4_ATTN_STAGE_NOROPE);
    if (ok && fused_q) ok = ds4_gpu_head_rms_norm_rope_tail_tensor(g->batch_q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
    if (ok && !fused_q) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                n_tokens,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_RMS_EPS) != 0;
    if (ok && !fused_q && !(stages & DS4_ATTN_STAGE_NOROPE)) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    /* 小批 kv: rope+fp8+环存三合一走解码同款核, 在 KV 段逐 token 发(须在 spec 原始行快照之后); 这里跳过分开的三发 */
    const bool fused_kv = small_batch && !(stages & DS4_ATTN_STAGE_NOROPE) && DS4_N_HEAD_KV == 1;
    if (ok && !fused_kv && !(stages & DS4_ATTN_STAGE_NOROPE)) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok && !fused_kv && !(stages & DS4_ATTN_STAGE_NOROPE))
        ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM,
                                                 DS4_N_ROT) != 0;
    }   /* stage PRE */
    if (ok && (stages & DS4_ATTN_STAGE_KV)) {
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          DS4_N_HEAD,
                                                          DS4_N_HEAD_DIM) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        if (ok && g->spec_comp_capture)
            ok = metal_graph_spec_raw_snapshot(g, il, pos0, n_tokens);
        if (ok && il == 0) eval_hdump_raw_rows(g->batch_kv, 97u, 0, n_tokens, n_tokens);   /* 入环前 f32 kv 行(L97) */
        if (ok) ok = batch_store_kv(g, il, pos0, n_tokens, small_batch && !(stages & DS4_ATTN_STAGE_NOROPE) && DS4_N_HEAD_KV == 1,
                                    compressed, freq_base, freq_scale, ext_factor, attn_factor);
        if (ok && il == 0) eval_hdump_raw_rows(g->layer_raw_cache[il], 98u, pos0 % g->raw_cap, n_tokens, g->raw_cap);
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                   g->raw_window,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ds4_comp_row_slots(ratio);
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: Metal layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok && small_comp) {
            /* 小批(投机 verify, 09-07): 压缩器(attn + ratio-4 的 indexer)走解码同一套攒行机制 —— 每 token 把 attn_norm 行
             * 推进 comp_x_ring, emit 位一次算攒下的行(含解码之前攒的). 此前批路每 token 自己投影自己 update, 解码攒在环里
             * 的行永远进不了批的 state ⇒ 压缩行错(对拍钉在第 2 层注意力出口)。spec 捕获改存 attn_norm 行(16 KB/行)供快进重放。 */
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                ds4_gpu_tensor *xrow = metal_graph_tensor_row_view(g->batch_attn_norm, t, DS4_N_EMBD);
                ok = xrow != NULL;
                if (ok && g->spec_comp_capture && t == 0 && g->spec_emit_t[il] >= (int8_t)1) {
                    /* 回滚重放只需 emit 位之前的行(t < acc ≤ t_e): 一次拷 t_e 行(batch_attn_norm 行连续) */
                    if (!g->spec_comp_rows_kv[il])
                        g->spec_comp_rows_kv[il] = ds4_gpu_tensor_alloc(8ull * DS4_N_EMBD * sizeof(float));
                    ok = g->spec_comp_rows_kv[il] &&
                         ds4_gpu_tensor_copy(g->spec_comp_rows_kv[il], 0, g->batch_attn_norm, 0,
                                             (uint64_t)g->spec_emit_t[il] * DS4_N_EMBD * sizeof(float)) != 0;
                }
                if (ok) ok = metal_graph_comp_push_row(g, il, ratio, pos, xrow);
                if (xrow) ds4_gpu_tensor_free(xrow);
                if (ok && ((pos + 1u) % ratio) == 0u) {
                    ok = metal_graph_comp_emit_step(g, model, layer, il, ratio, compressed, freq_base, freq_scale,
                                                    ext_factor, attn_factor);
                    if (!ok) fprintf(stderr, "ds4: 小批压缩器 emit 失败 L%u pos %u\n", il, pos);
                }
                if (!ok) fprintf(stderr, "ds4: 小批压缩器 token %u(pos %u) L%u 失败\n", t, pos, il);
                if (comp_counts) comp_counts[t] = (pos + 1u) / ratio;
                if (index_counts) index_counts[t] = (pos + 1u) / ratio;
            }
        } else if (ok) {
            ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                             model->map,
                                             model->size,
                                             layer->attn_compressor_kv->abs_offset,
                                             DS4_N_EMBD,
                                             comp_width,
                                             g->batch_attn_norm,
                                             n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD,
                                                     comp_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
        }
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            /* 缓存 f16: 压缩器写 f32 暂存(行 0 起), 量化后提交转 f16 */
            ds4_gpu_tensor *attn_comp_target = g->attn_comp_stage;
            if (ok) {
                ok = ds4_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS) != 0;
                if (ok && n_comp != 0) {
                    ok = metal_graph_commit_attn_comp_stage(g, il, 0, n_comp);
                }
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                }
            }
        } else if (!small_comp) {   /* 小批的压缩器已在上面按解码机制做完 */
            /* spec 捕获(update 前, 行未被就地处理): 本批压缩器输入行 */
            if (ok && g->spec_comp_capture && n_tokens <= 8u) {
                const uint64_t rb = (uint64_t)n_tokens * comp_width * sizeof(float);
                if (!g->spec_comp_rows_kv[il])
                    g->spec_comp_rows_kv[il] = ds4_gpu_tensor_alloc(8ull * comp_width * sizeof(float));
                if (!g->spec_comp_rows_sc[il])
                    g->spec_comp_rows_sc[il] = ds4_gpu_tensor_alloc(8ull * comp_width * sizeof(float));
                ok = g->spec_comp_rows_kv[il] && g->spec_comp_rows_sc[il] &&
                     ds4_gpu_tensor_copy(g->spec_comp_rows_kv[il], 0, g->batch_comp_kv, 0, rb) != 0 &&
                     ds4_gpu_tensor_copy(g->spec_comp_rows_sc[il], 0, g->batch_comp_sc, 0, rb) != 0;
            }
            const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u && n_tokens > 8u;   /* 小批(投机 verify ≤8)走逐 token 增量更新 = 解码同轨(09-07); 整块回放的累加序不同 */
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target = g->attn_comp_stage;   /* 缓存 f16: 回放写 f32 暂存行 0 起 */
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = metal_graph_commit_attn_comp_stage(g, il, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                }
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            g->attn_comp_stage,   /* 缓存 f16: 先写 f32 暂存行 0 */
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            0u,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(g->attn_comp_stage, 0,
                                                                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
                        ok = comp_row_view &&
                             ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                   1,
                                                                   DS4_N_HEAD_DIM,
                                                                   DS4_N_ROT) != 0;
                        ds4_gpu_tensor_free(comp_row_view);
                        if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t == 0) ok = metal_graph_capture_prefix1_attn_state(g, il);
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }

        /* ★小批与解码同规(09-07, 链 31 定罪)★: 解码的稀疏选择门槛 = n_comp > 1024(decode_util) 且 n_index_comp > TOP_K, 逐 token
         * 用该 token emit 之后的计数判; 批路原来是 n_comp > 512 ⇒ 2.5K 上下文 (512, 1024] 区间一路选 top-512 一路全扫, spec/plain
         * 分叉。批内跨门槛(每会话各一次)的那轮两种都算, 按 token 拼行。 */
        uint8_t small_ix_tok[8] = {0};
        bool small_ix_any = false, small_ix_all = false;
        if (small_batch && ratio == 4 && comp_counts && index_counts) {
            const uint32_t thr = metal_graph_decode_indexer_sparse_threshold(g);
            small_ix_all = true;
            for (uint32_t t = 0; t < n_tokens; t++) {
                small_ix_tok[t] = (comp_counts[t] > thr && index_counts[t] > DS4_N_INDEXER_TOP_K) ? 1u : 0u;
                small_ix_any = small_ix_any || small_ix_tok[t];
                small_ix_all = small_ix_all && small_ix_tok[t];
            }
        }
        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: Metal layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok && !small_comp) {   /* 小批的 indexer 压缩器已随 attn 压缩器在 emit 步里算完 */
                ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                                 model->map,
                                                 model->size,
                                                 layer->indexer_compressor_kv->abs_offset,
                                                 DS4_N_EMBD,
                                                 index_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD,
                                                         index_width,
                                                         g->batch_attn_norm,
                                                         n_tokens) != 0;
            }
            /* 小批(09-07): indexer q/权重只有消费端会用, 短上下文无条件算是每轮 2.3 ms 白扔; 消费端规则 = 解码逐 token 同规
             * (small_ix_tok, 见下文注意力处), 任一 token 要选就算。 */
            const bool indexer_used = !small_batch || small_ix_any;
            if (ok && indexer_used) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_q,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_attn_q_b->abs_offset,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            if (ok && indexer_used) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok && indexer_used) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok && indexer_used) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_weights,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                /* indexer 缓存 f16: 先写 f32 暂存(QAT 也在暂存上做), 再提交转 f16 */
                ds4_gpu_tensor *index_target = g->attn_comp_stage;
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(index_target,
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_target,
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok && n_comp != 0) ok = metal_graph_commit_index_comp_stage(g, il, 0, n_comp);
                if (ok) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                    }
                }
            } else if (!small_comp) {
                const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u && n_tokens > 8u;   /* 小批(投机 verify ≤8)走逐 token 增量更新 = 解码同轨(09-07); 整块回放的累加序不同 */
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    /* indexer 缓存 f16: 回放写进 f32 暂存(行 0 起), QAT 后提交转 f16 */
                    ds4_gpu_tensor *index_view = g->attn_comp_stage;
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    if (ok && index_chunk != 0) ok = metal_graph_commit_index_comp_stage(g, il, index_before, index_chunk);
                    if (ok) {
                        ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                    }
                } else {
                    /* spec 捕获(update 前): indexer 压缩器输入行 */
                    if (ok && g->spec_comp_capture && n_tokens <= 8u) {
                        const uint64_t rb = (uint64_t)n_tokens * index_width * sizeof(float);
                        if (!g->spec_idx_rows_kv[il])
                            g->spec_idx_rows_kv[il] = ds4_gpu_tensor_alloc(8ull * index_width * sizeof(float));
                        if (!g->spec_idx_rows_sc[il])
                            g->spec_idx_rows_sc[il] = ds4_gpu_tensor_alloc(8ull * index_width * sizeof(float));
                        ok = g->spec_idx_rows_kv[il] && g->spec_idx_rows_sc[il] &&
                             ds4_gpu_tensor_copy(g->spec_idx_rows_kv[il], 0, g->batch_comp_kv, 0, rb) != 0 &&
                             ds4_gpu_tensor_copy(g->spec_idx_rows_sc[il], 0, g->batch_comp_sc, 0, rb) != 0;
                    }
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                g->attn_comp_stage,   /* f16 缓存: 先写 f32 暂存行 0 */
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                0u,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS) != 0;
                        if (ok && emit) {
                            ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                                    g->attn_comp_stage, 0,
                                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else {
                                ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD_DIM) != 0;
                                ds4_gpu_tensor_free(index_row_view);
                            }
                            if (ok) ok = metal_graph_commit_index_comp_stage(g, il, index_row, 1);
                        }
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t == 0) ok = metal_graph_capture_prefix1_index_state(g, il);
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;

            if (ok && g->spec_comp_capture)
                ok = metal_graph_spec_raw_snapshot(g, il, pos0, n_tokens);
            if (ok) ok = batch_store_kv(g, il, pos0, n_tokens, small_batch && !(stages & DS4_ATTN_STAGE_NOROPE) && DS4_N_HEAD_KV == 1,
                                    compressed, freq_base, freq_scale, ext_factor, attn_factor);
            const bool want_indexed = small_batch ? small_ix_any : (ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K);
            if (ok && ratio == 4 && want_indexed) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                ok = ds4_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                  g->batch_indexer_q,
                                                                  g->batch_indexer_weights,
                                                                  g->layer_index_comp_cache[il],
                                                                  n_comp,
                                                                  n_tokens,
                                                                  pos0,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  ratio,
                                                                  index_scale) != 0;
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = small_batch ? 0u : 1u;   /* 解码稠密路无 mask; 小批同规 */
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              g->batch_q,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              g->comp_selected,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                    if (ok && small_batch && !small_ix_all) {
                        /* 跨门槛轮: 稠密再算一遍到暂存, 把不选 top-k 的 token 行拷回(每会话两次, 各 ~50 µs) */
                        const uint64_t rowb = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
                        if (!g->spec_heads_alt) g->spec_heads_alt = ds4_gpu_tensor_alloc(8ull * rowb);
                        ok = g->spec_heads_alt &&
                             ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->spec_heads_alt, model->map, model->size,
                                     layer->attn_sinks->abs_offset, g->batch_q, g->layer_raw_cache[il], g->layer_attn_comp_cache[il],
                                     NULL, 0u, n_tokens, pos0, n_raw, g->raw_cap, raw_start, n_comp, g->raw_window, ratio,
                                     DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
                        for (uint32_t t = 0; ok && t < n_tokens; t++)
                            if (!small_ix_tok[t])
                                ok = ds4_gpu_tensor_copy(g->batch_heads, (uint64_t)t * rowb, g->spec_heads_alt, (uint64_t)t * rowb, rowb) != 0;
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             g->batch_q,
                                                                             g->layer_raw_cache[il],
                                                                             g->layer_attn_comp_cache[il],
                                                                             use_comp_mask ? g->comp_mask : NULL,
                                                                             use_comp_mask,
                                                                             n_tokens,
                                                                             pos0,
                                                                             n_raw,
                                                                             g->raw_cap,
                                                                             raw_start,
                                                                             n_comp,
                                                                             g->raw_window,
                                                                             ratio,
                                                                             DS4_N_HEAD,
                                                                             DS4_N_HEAD_DIM) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            ok = ds4_gpu_indexer_scores_prefill_tensor(g->indexer_scores,
                                                         g->batch_indexer_q,
                                                         g->batch_indexer_weights,
                                                         g->layer_index_comp_cache[il],
                                                         n_comp,
                                                         n_tokens,
                                                         DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM,
                                                         ratio,
                                                         index_scale) != 0;
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->indexer_scores,
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          g->batch_q,
                                                                          g->layer_raw_cache[il],
                                                                          g->layer_attn_comp_cache[il],
                                                                          g->comp_selected,
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv,
                                                                       g->layer_attn_comp_cache[il],
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       DS4_N_HEAD,
                                                                       DS4_N_HEAD_DIM) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = metal_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = metal_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            DS4_N_INDEXER_HEAD,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       DS4_N_INDEXER_TOP_K) != 0 &&
                         ds4_gpu_dsv4_topk_mask_tensor(g->comp_mask,
                                                         g->comp_selected,
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = g->comp_mask;
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = metal_graph_tensor_row_view(g->batch_q, t, q_dim);
                ds4_gpu_tensor *kv_cache_view = metal_graph_tensor_row_view(g->batch_kv, t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = metal_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              g->comp_selected,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                    ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->attn_sinks->abs_offset,
                                                                 q_view,
                                                                 g->layer_raw_cache[il],
                                                                 n_raw,
                                                                 g->raw_cap,
                                                                 raw_start,
                                                                 cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                                                                 cur_comp,
                                                                 comp_mask,
                                                                 n_selected,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    }   /* stage KV */
    if (ok && (stages & DS4_ATTN_STAGE_POST)) {
    if (ok && !(stages & DS4_ATTN_STAGE_NOROPE)) ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        ok = ((layer->attn_output_a->type == DS4_TENSOR_Q4_K || layer->attn_output_a->type == DS4_TENSOR_Q2_K)
                  ? attn_output_kq_batch(layer->attn_output_a, g->batch_attn_out,
                                                              g->batch_attn_low,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_output_b->abs_offset,
                                                              group_dim,
                                                              rank,
                                                              n_groups,
                                                              DS4_N_EMBD,
                                                              g->batch_heads,
                                                              n_tokens)
                  : ds4_gpu_attention_output_q8_batch_tensor(g->batch_attn_out,
                                                             g->batch_attn_low,
                                                             g->batch_group_tmp,
                                                             g->batch_low_tmp,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_output_a->abs_offset,
                                                             layer->attn_output_b->abs_offset,
                                                             group_dim,
                                                             rank,
                                                             n_groups,
                                                             DS4_N_EMBD,
                                                             g->batch_heads,
                                                             n_tokens)) != 0;
    }
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }   /* stage POST */
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
    return ok;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_prefill_attn_nonempty_tu; /* 空TU防御(CPU构建) */
