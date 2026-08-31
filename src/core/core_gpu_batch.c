/* core_gpu_batch.c — 层批编码/spec 存档/dspark 状态 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
void eval_hdump_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens) {
    const char *dir = ds4_tool_eval_hdump();
    if (!dir || !dir[0] || n_tokens == 0) return;
    {
        const uint64_t ev = ds4_gpu_tp_signal_after_batch();
        if (ev) { (void)ds4_gpu_flush_commands(); (void)ds4_gpu_tp_host_wait(ev); }
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const size_t nf = (size_t)n_tokens * hc_dim;
    float *buf = malloc(nf * sizeof(float));
    if (!buf) return;
    if (ds4_gpu_tensor_read(g->batch_cur_hc, 0, buf, nf * sizeof(float))) {
        char p[1024];
        snprintf(p, sizeof p, "%s/h_L%02u.bin", dir, il);
        FILE *f = fopen(p, "ab");
        if (!f) {
            fprintf(stderr, "ds4: [EVAL_HDUMP] 打不开 %s -- aborting\n", p);
            exit(1);
        }
        if (fwrite(buf, sizeof(float), nf, f) != nf) {
            fprintf(stderr, "ds4: [EVAL_HDUMP] 写 %s 短写 -- aborting\n", p);
            exit(1);
        }
        fclose(f);
    }
    free(buf);
}

bool metal_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    return metal_graph_encode_layer_attention_batch_stages(g, model, layer, il, pos0,
                                                           n_tokens, DS4_ATTN_STAGE_ALL);
}

bool metal_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    /* DSpark prefill 抓取(层出口 HC 均值)与建窗: prompt 每 token 的 main_kv 进环形窗,
     * 与官方 prefill(start_pos==0 只建 KV)语义一致 */
    if (ok && g->dspark_capture && g->dspark_pf_hidden && il >= 40u && il <= 42u) {
        ok = ds4_gpu_dspark_hc_mean_tensor(g->dspark_pf_hidden, g->batch_cur_hc,
                                           DS4_N_EMBD, DS4_N_HC, il - 40u, n_tokens) != 0;
        if (ok && il == 42u && g_dspark_bound_for_prefill) {
            const ds4_dspark_weights *dw = g_dspark_bound_for_prefill;
            const ds4_model *dmodel = dw->src ? dw->src : model;
            const float fb = DS4_ROPE_FREQ_BASE, fs = 1.0f;
            ok = dense_matmul_typed(g->dspark_pf_x, dmodel, dw->main_proj,
                                    3ull * DS4_N_EMBD, DS4_N_EMBD, g->dspark_pf_hidden, n_tokens) != 0;
            if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->dspark_pf_x, g->dspark_pf_x,
                                                             dmodel->map, dmodel->size,
                                                             dw->main_norm->abs_offset,
                                                             DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0;
            for (uint32_t b = 0; ok && b < (uint32_t)dw->n_blocks; b++) {
                ok = dense_matmul_typed(g->dspark_pf_kv, dmodel, dw->block[b].attn_kv,
                                        DS4_N_EMBD, DS4_N_HEAD_DIM, g->dspark_pf_x, n_tokens) != 0;
                if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->dspark_pf_kv, g->dspark_pf_kv,
                                                                 dmodel->map, dmodel->size,
                                                                 dw->block[b].attn_kv_a_norm->abs_offset,
                                                                 DS4_N_HEAD_DIM, n_tokens, DS4_RMS_EPS) != 0;
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->dspark_pf_kv, n_tokens, 1, DS4_N_HEAD_DIM,
                                                      DS4_N_ROT, pos0, 0, false, fb, fs, 0.0f, 1.0f,
                                                      DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
                /* verify 批(spec_comp_capture=1)先暂存不落窗(2026-08-21 修): drafter 的 128 环形窗
                 * 是"位置 p → 槽 p%128"。verify 会写 k 个候选, 其中被拒的那几个把 128 位之前
                 * 仍在窗内的有效行盖掉, 且没有任何东西会把它们改回来 —— 窗口逐轮累积污染,
                 * drafter 越跑越瞎(实测首位接受率 0.82 → 0.59)。改为按接受数提交。 */
                if (ok && g->spec_comp_capture && b < 3u && g->dspark_spec_kv[b] &&
                    n_tokens <= (uint32_t)DS4_DSPARK_BLK + 1u) {
                    ok = ds4_gpu_tensor_copy(g->dspark_spec_kv[b], 0, g->dspark_pf_kv, 0,
                                             (uint64_t)n_tokens * DS4_N_HEAD_DIM * sizeof(float)) != 0;
                } else if (ok) {
                    ok = ds4_gpu_dspark_win_scatter_tensor(g->dspark_win_kv[b], g->dspark_pf_kv,
                                                           n_tokens, pos0, DS4_DSPARK_WIN,
                                                           DS4_N_HEAD_DIM) != 0;
                }
            }
        }
    }
    if (ok) eval_hdump_batch_layer(g, il, n_tokens);
    return ok;
}

/* Execute one Metal decode token and read back logits. */
/* =========================================================================
 * DSpark 块并行 drafter (2026-08-18, 语义=hf/inference/model.py)。
 * 每 decode 步: ①main_x=main_norm(main_proj(main_hidden[3×4096]))
 * ②每块层 main_kv=rope(kv_norm(wkv(main_x))) 写环形窗 pos%128
 * ③draft: [anchor,noise×4] embed→HC→3 层(手写 attn: 窗+块内因果 / FFN 复用批段)
 * ④hc_head→norm→lm_head→markov 链 5 步 → draft ids。
 * drafter KV 全 f32(草稿路径, verify 兜底正确性)。 */

/* spec replay 消除(2026-08-20): restore(轮前态)后, 用 verify 批捕获的压缩器/indexer
 * 输入行快进 acc 位。与 replay 全前向等价 —— KV raw 行 verify 已写好且 restore 不动,
 * 唯一需要推进的就是压缩器滚动态; 输入行两次前向逐位相同(同 token 同前缀)。 */
bool metal_graph_spec_comp_fastforward(ds4_gpu_graph *g, const ds4_model *model,
                                              const ds4_weights *weights,
                                              uint32_t pos0, uint32_t acc) {
    bool ok = true;
    for (uint32_t il = 0; ok && il < (uint32_t)DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const ds4_layer_weights *layer = &weights->layer[il];
        const uint32_t coff = ds4_comp_row_slots(ratio);
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const float freq_base = layer_rope_freq_base(il);
        const float freq_scale = layer_rope_freq_scale(il);
        const float ext_factor = DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
        float attn_factor = 1.0f;
        if (ext_factor != 0.0f && freq_scale > 0.0f)
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        if (!g->spec_comp_rows_kv[il] || !g->spec_comp_rows_sc[il]) return false;
        for (uint32_t t = 0; ok && t < acc; t++) {
            const uint32_t pos = pos0 + t;
            const bool emit = ((pos + 1u) % ratio) == 0u;
            if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) { ok = false; break; }
            ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(g->spec_comp_rows_kv[il], t, comp_width);
            ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(g->spec_comp_rows_sc[il], t, comp_width);
            const uint32_t comp_row = g->layer_n_comp[il];
            ok = kv_view && sc_view &&
                 ds4_gpu_compressor_update_tensor(kv_view, sc_view,
                        g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                        metal_graph_attn_comp_update_target(g, il),
                        model->map, model->size,
                        layer->attn_compressor_ape->abs_offset, layer->attn_compressor_ape->type,
                        layer->attn_compressor_norm->abs_offset, layer->attn_compressor_norm->type,
                        DS4_N_HEAD_DIM, ratio, pos,
                        metal_graph_attn_comp_update_row(comp_row),
                        DS4_N_ROT, (uint32_t)DS4_ROPE_ORIG_CTX,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
            if (ok && emit) {
                ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
                ok = comp_row_view &&
                     ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                ds4_gpu_tensor_free(comp_row_view);
                if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
            }
            if (ok && emit) g->layer_n_comp[il]++;
            ds4_gpu_tensor_free(sc_view);
            ds4_gpu_tensor_free(kv_view);
        }
        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!g->spec_idx_rows_kv[il] || !g->spec_idx_rows_sc[il]) return false;
            for (uint32_t t = 0; ok && t < acc; t++) {
                const uint32_t pos = pos0 + t;
                const bool emit = ((pos + 1u) % ratio) == 0u;
                if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) { ok = false; break; }
                ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(g->spec_idx_rows_kv[il], t, index_width);
                ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(g->spec_idx_rows_sc[il], t, index_width);
                const uint32_t index_row = g->layer_n_index_comp[il];
                ok = kv_view && sc_view &&
                     ds4_gpu_compressor_update_tensor(kv_view, sc_view,
                            g->layer_index_state_kv[il], g->layer_index_state_score[il],
                            g->layer_index_comp_cache[il],
                            model->map, model->size,
                            layer->indexer_compressor_ape->abs_offset, layer->indexer_compressor_ape->type,
                            layer->indexer_compressor_norm->abs_offset, layer->indexer_compressor_norm->type,
                            DS4_N_INDEXER_HEAD_DIM, ratio, pos, index_row,
                            DS4_N_ROT, (uint32_t)DS4_ROPE_ORIG_CTX,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
                if (ok && emit) {
                    ds4_gpu_tensor *iv = ds4_gpu_tensor_view(g->layer_index_comp_cache[il],
                            (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                            (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                    ok = iv && ds4_gpu_dsv4_indexer_qat_tensor(iv, 1, DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(iv);
                }
                if (ok && emit) g->layer_n_index_comp[il]++;
                ds4_gpu_tensor_free(sc_view);
                ds4_gpu_tensor_free(kv_view);
            }
        }
    }
    return ok;
}

/* verify 前后的压缩器状态快照/恢复(partial-accept 用 restore+重放, 官方
 * checkpoint-restore 同口径)。快照 ~数十 MB 拷贝, 0.2ms 级。 */
/* verify 批把 k 个候选的 raw KV 写进 SWA 环, 而环容量恰等于窗口(raw_cap==raw_window),
 * 于是被拒候选的行会盖掉"仍在窗内"的旧位置, 且此后没有任何东西把它们改回来 ——
 * 主模型后续 token 的注意力就会读到被拒草稿的 KV。写前存旧行, 定了 acc 再把被拒的还原。
 * 行是 (pos0+t)%cap 连续段, 最多两段拷贝/层。 */
bool metal_graph_spec_raw_snapshot(ds4_gpu_graph *g, uint32_t il, uint32_t pos0, uint32_t n) {
    if (il >= (uint32_t)DS4_N_LAYER || !g->spec_raw_save[il] || !g->layer_raw_cache[il] ||
        n == 0 || n > (uint32_t)DS4_DSPARK_BLK + 1u || g->raw_cap == 0) return true;
    const uint64_t rb = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    const uint32_t start = pos0 % g->raw_cap;
    const uint32_t first = (start + n <= g->raw_cap) ? n : (g->raw_cap - start);
    if (!ds4_gpu_tensor_copy(g->spec_raw_save[il], 0, g->layer_raw_cache[il],
                             (uint64_t)start * rb, (uint64_t)first * rb)) return false;
    if (first < n &&
        !ds4_gpu_tensor_copy(g->spec_raw_save[il], (uint64_t)first * rb,
                             g->layer_raw_cache[il], 0, (uint64_t)(n - first) * rb)) return false;
    return true;
}

bool metal_graph_spec_raw_restore(ds4_gpu_graph *g, uint32_t pos0, uint32_t from, uint32_t to) {
    if (from >= to || g->raw_cap == 0) return true;
    /* 被拒的是 [from,to) 这一段连续位置 ⇒ 环上最多两段, 每层 1-2 次拷贝(逐行拷会发
     * 215 次小拷贝, 实测吃掉 ~3ms/轮)。 */
    const uint64_t rb = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    const uint32_t n = to - from;
    const uint32_t start = (pos0 + from) % g->raw_cap;
    const uint32_t first = (start + n <= g->raw_cap) ? n : (g->raw_cap - start);
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        if (!g->spec_raw_save[il] || !g->layer_raw_cache[il]) continue;
        if (!ds4_gpu_tensor_copy(g->layer_raw_cache[il], (uint64_t)start * rb,
                                 g->spec_raw_save[il], (uint64_t)from * rb,
                                 (uint64_t)first * rb)) return false;
        if (first < n &&
            !ds4_gpu_tensor_copy(g->layer_raw_cache[il], 0,
                                 g->spec_raw_save[il], (uint64_t)(from + first) * rb,
                                 (uint64_t)(n - first) * rb)) return false;
    }
    return true;
}

/* 把 verify 批暂存的 drafter KV 按接受数提交进环形窗(只提交真正被接受的位置)。 */
bool metal_graph_dspark_win_commit(ds4_gpu_graph *g, uint32_t pos0, uint32_t n_acc) {
    if (n_acc == 0) return true;
    for (uint32_t b = 0; b < 3u; b++) {
        if (!g->dspark_spec_kv[b] || !g->dspark_win_kv[b]) continue;
        if (!ds4_gpu_dspark_win_scatter_tensor(g->dspark_win_kv[b], g->dspark_spec_kv[b],
                                               n_acc, pos0, DS4_DSPARK_WIN,
                                               DS4_N_HEAD_DIM)) return false;
    }
    return true;
}

bool metal_graph_dspark_state_snapshot(ds4_gpu_graph *g) {
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        if (!g->spec_prefix1_attn_state_kv[il] || !g->layer_attn_state_kv[il]) continue;
        const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
        if (!ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) ||
            !ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes)) return false;
        if (g->spec_prefix1_index_state_kv[il] && g->layer_index_state_kv[il]) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
            if (!ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                     g->layer_index_state_kv[il], 0, ib) ||
                !ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                     g->layer_index_state_score[il], 0, ib)) return false;
        }
    }
    return true;
}

bool metal_graph_dspark_state_restore(ds4_gpu_graph *g) {
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        if (!g->spec_prefix1_attn_state_kv[il] || !g->layer_attn_state_kv[il]) continue;
        const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        g->layer_n_comp[il] = g->spec_prefix1_n_comp[il];
        if (!ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                 g->spec_prefix1_attn_state_kv[il], 0, bytes) ||
            !ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                 g->spec_prefix1_attn_state_score[il], 0, bytes)) return false;
        if (g->spec_prefix1_index_state_kv[il] && g->layer_index_state_kv[il]) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            g->layer_n_index_comp[il] = g->spec_prefix1_n_index_comp[il];
            if (!ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                     g->spec_prefix1_index_state_kv[il], 0, ib) ||
                !ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                     g->spec_prefix1_index_state_score[il], 0, ib)) return false;
        }
    }
    return true;
}

/* out_conf(可空): 逐位置置信 c_k, 调度器用 ∏c 选验证长度(论文 Alg.1)。 */
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_batch_nonempty_tu; /* 空TU防御(CPU构建) */
