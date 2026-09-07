/* core_gpu_decode_comp.c — 解码期压缩器投影攒到 emit 一次算(2026-09-05)。
 *
 * 怎么用: 解码每层先 metal_graph_comp_push(把本 token 的 attn_norm 行推进层环, 16 KB),
 * 只有 emit 位(pos+1 整除 ratio)才 metal_graph_comp_project_pending 一次算出攒下的 n 行
 * 投影(g->comp_kv_batch/comp_sc_batch), 再交给 ds4_gpu_compressor_update_batch_tensor 入
 * state + 池化。任何绕过解码步改 pos/读 state 的入口(prefill、快照保存、回卷)必须先
 * metal_graph_comp_flush_pending: 把攒着的行投影入 state(不池化), 否则 state 少了这些行,
 * prefill 的 ratio-4 replay / 快照里的 state 就是缺行的 —— 不崩, 数值漂。
 *
 * 为什么: 压缩器 kv/gate 投影是每 token 609 MB 的 f16 权重读(2.6 ms/token, 09-05 nsys),
 * 而 state 只是逐 token 投影的暂存区, emit 才池化 —— 投影按 token 逐行独立, 攒 n 行一次算,
 * 权重读量 ratio-4 层降 4×、ratio-128 层降 16×。每行数值与单 token 路逐字相同(核内同累加序)。
 * 环上连续性: 攒着的 n 行是 [last_pos+1-n, last_pos], 都在同一压缩块内(emit/冲刷都清零),
 * 所以 pos0%ratio..last_pos%ratio 连续不回绕, 取一个 view 就是 n 行矩阵。
 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
bool metal_graph_comp_push(ds4_gpu_graph *g, uint32_t il, uint32_t ratio, uint32_t pos) {
    if (!g->comp_x_ring[il] || ratio == 0) return false;
    if (!ds4_gpu_compressor_ring_push_tensor(g->comp_x_ring[il], pos % ratio, g->attn_norm, DS4_N_EMBD))
        return false;
    g->comp_x_pending[il]++;
    g->comp_x_last_pos[il] = pos;
    return g->comp_x_pending[il] <= ratio;
}

bool metal_graph_comp_project_pending(
        ds4_gpu_graph    *g,
        const ds4_model  *model,
        const ds4_tensor *kv_weight,
        const ds4_tensor *gate_weight,
        uint32_t          il,
        uint32_t          ratio,
        uint32_t          width,
        uint32_t         *n_out,
        uint32_t         *pos0_out) {
    const uint32_t n = g->comp_x_pending[il];
    if (n == 0 || n > ratio || !g->comp_x_ring[il] || !kv_weight || !gate_weight) {
        fprintf(stderr, "ds4: comp project_pending L%u: 攒行 %u(ratio %u) 非法或权重缺\n", il, n, ratio);
        return false;
    }
    const uint32_t pos0 = g->comp_x_last_pos[il] + 1u - n;
    ds4_gpu_tensor *xv = ds4_gpu_tensor_view(g->comp_x_ring[il],
                                             (uint64_t)(pos0 % ratio) * DS4_N_EMBD * sizeof(float),
                                             (uint64_t)n * DS4_N_EMBD * sizeof(float));
    if (!xv) return false;
    const bool ok = ds4_gpu_matmul_f16_pair_rows_tensor(g->comp_kv_batch, g->comp_sc_batch,
                                                        model->map, model->size,
                                                        kv_weight->abs_offset, gate_weight->abs_offset,
                                                        DS4_N_EMBD, width, xv, n) != 0;
    ds4_gpu_tensor_free(xv);
    *n_out = n;
    *pos0_out = pos0;
    return ok;
}

/* 同 comp_push 但行由调用方给(verify 批的 batch_attn_norm 第 t 行): 小批压缩器走与解码同一套攒行机制(09-07)。 */
bool metal_graph_comp_push_row(ds4_gpu_graph *g, uint32_t il, uint32_t ratio, uint32_t pos, const ds4_gpu_tensor *row) {
    if (!g->comp_x_ring[il] || ratio == 0 || !row) return false;
    if (!ds4_gpu_compressor_ring_push_tensor(g->comp_x_ring[il], pos % ratio, row, DS4_N_EMBD)) {
        fprintf(stderr, "ds4: comp push_row L%u pos %u: 环 push 失败\n", il, pos);
        return false;
    }
    g->comp_x_pending[il]++;
    g->comp_x_last_pos[il] = pos;
    if (g->comp_x_pending[il] > ratio) {
        fprintf(stderr, "ds4: comp push_row L%u pos %u: 攒行 %u 超 ratio %u(emit 位没清?)\n", il, pos, g->comp_x_pending[il], ratio);
        return false;
    }
    return true;
}

/* emit 位一步 = 解码 core_gpu_decode_layer 的同一序列(逐字): attn 压缩器 project_pending → update_batch → fp8 → commit
 * → n_comp++; ratio-4 再 indexer 压缩器 project_pending → update_batch → QAT → commit → n_index_comp++; 最后清攒行。
 * verify 批(小批)与快进重放都调它 ⇒ 与纯解码逐核同序同输入。 */
bool metal_graph_comp_emit_step(ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer,
                                uint32_t il, uint32_t ratio, bool compressed, float freq_base, float freq_scale,
                                float ext_factor, float attn_factor) {
    const uint32_t coff = ds4_comp_row_slots(ratio);
    const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
    if (g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
        fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
        return false;
    }
    bool ok = true;
    const uint32_t comp_row = g->layer_n_comp[il];
    uint32_t n_pend = 0, pos0 = 0;
    ok = metal_graph_comp_project_pending(g, model, layer->attn_compressor_kv, layer->attn_compressor_gate,
                                          il, ratio, comp_width, &n_pend, &pos0);
    if (ok) ok = ds4_gpu_compressor_update_batch_tensor(g->comp_kv_batch, g->comp_sc_batch,
                                                        g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                                                        g->attn_comp_stage, model->map, model->size,
                                                        layer->attn_compressor_ape->abs_offset, layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset, layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM, ratio, pos0, n_pend, 0u, DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base, freq_scale, ext_factor, attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
    if (ok) {
        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(g->attn_comp_stage, 0, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        if (!comp_row_view) ok = false;
        else {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
            ds4_gpu_tensor_free(comp_row_view);
        }
        if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
    }
    if (ok) g->layer_n_comp[il]++;
    if (ok && ratio != 4u) g->comp_x_pending[il] = 0;
    if (ok && ratio == 4u) {
        const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
        if (g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
            return false;
        }
        const uint32_t index_row = g->layer_n_index_comp[il];
        ok = metal_graph_comp_project_pending(g, model, layer->indexer_compressor_kv, layer->indexer_compressor_gate,
                                              il, ratio, index_width, &n_pend, &pos0);
        if (ok) ok = ds4_gpu_compressor_update_batch_tensor(g->comp_kv_batch, g->comp_sc_batch,
                                                            g->layer_index_state_kv[il], g->layer_index_state_score[il],
                                                            g->attn_comp_stage, model->map, model->size,
                                                            layer->indexer_compressor_ape->abs_offset, layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset, layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM, ratio, pos0, n_pend, 0u, DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base, freq_scale, ext_factor, attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
        if (ok) {
            ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(g->attn_comp_stage, 0, (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
            if (!index_row_view) ok = false;
            else {
                ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view, 1, DS4_N_INDEXER_HEAD_DIM) != 0;
                ds4_gpu_tensor_free(index_row_view);
            }
            if (ok) ok = metal_graph_commit_index_comp_stage(g, il, index_row, 1);
        }
        if (ok) g->layer_n_index_comp[il]++;
        if (ok) g->comp_x_pending[il] = 0;
    }
    return ok;
}

void metal_graph_comp_pending_clear(ds4_gpu_graph *g) {
    memset(g->comp_x_pending, 0, sizeof(g->comp_x_pending));
}

bool metal_graph_comp_flush_pending(ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g->comp_x_pending[il] || !metal_graph_layer_is_active(g, il)) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) { g->comp_x_pending[il] = 0; continue; }
        const ds4_layer_weights *layer = &weights->layer[il];
        const uint32_t coff = ds4_comp_row_slots(ratio);
        uint32_t n = 0, pos0 = 0;
        /* 攒着的行末位永远不在 emit 位(emit 当场清零), 所以 update_batch 只入 state 不池化;
         * rope/norm 参数只过校验不参与计算。走 update_batch 而非 store_batch 是因为 Metal
         * 契约只有前者。 */
        if (!metal_graph_comp_project_pending(g, model, layer->attn_compressor_kv, layer->attn_compressor_gate,
                                              il, ratio, coff * DS4_N_HEAD_DIM, &n, &pos0)) return false;
        if (!ds4_gpu_compressor_update_batch_tensor(g->comp_kv_batch, g->comp_sc_batch,
                                                    g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                                                    g->attn_comp_stage,   /* 不 emit, 只过校验; 缓存是 f16 不能当 f32 目标 */
                                                    model->map, model->size,
                                                    layer->attn_compressor_ape->abs_offset,
                                                    layer->attn_compressor_ape->type,
                                                    layer->attn_compressor_norm->abs_offset,
                                                    layer->attn_compressor_norm->type,
                                                    DS4_N_HEAD_DIM, ratio, pos0, n,
                                                    0u,
                                                    DS4_N_ROT, 0, 1.0f, 1.0f, 0.0f, 1.0f,
                                                    DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                                                    DS4_RMS_EPS)) return false;
        if (ratio == 4u) {
            if (!metal_graph_comp_project_pending(g, model, layer->indexer_compressor_kv,
                                                  layer->indexer_compressor_gate,
                                                  il, ratio, coff * DS4_N_INDEXER_HEAD_DIM, &n, &pos0)) return false;
            if (!ds4_gpu_compressor_update_batch_tensor(g->comp_kv_batch, g->comp_sc_batch,
                                                        g->layer_index_state_kv[il], g->layer_index_state_score[il],
                                                        g->attn_comp_stage,   /* 不 emit, 只过校验; 缓存是 f16 不能当 f32 目标 */
                                                        model->map, model->size,
                                                        layer->indexer_compressor_ape->abs_offset,
                                                        layer->indexer_compressor_ape->type,
                                                        layer->indexer_compressor_norm->abs_offset,
                                                        layer->indexer_compressor_norm->type,
                                                        DS4_N_INDEXER_HEAD_DIM, ratio, pos0, n,
                                                        0u,
                                                        DS4_N_ROT, 0, 1.0f, 1.0f, 0.0f, 1.0f,
                                                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS)) return false;
        }
        g->comp_x_pending[il] = 0;
    }
    return true;
}
#endif
