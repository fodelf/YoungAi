/* core_gpu_decode_layer.c — 单token decode 层编码 (机械拆分自 ds4.c, 重构阶段4)。 */
/* EXCEPTION(>500行): 单函数 metal_graph_encode_decode_layer, 函数内拆分是后续工序(需真模型逐位闸) */
#include "core_internal.h"
#ifndef DS4_NO_GPU
bool metal_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    /* decode 路径 zchain 句柄(batch 路径经 _ex(no_zchain) 旁路, 此处恒主模型侧车) */
    const struct ds4_zchain *zch = model->zchain;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t expert_mid_dim = routed_expert_mid_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL;
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    const bool fuse_hc_norm =
        DS4_MODEL_VARIANT == DS4_VARIANT_FLASH &&
        !metal_graph_use_reference_hc_decode() &&
        !metal_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", g->attn_norm, DS4_N_EMBD, il, pos);
    }
    const bool qa_kv_pair = qkv_rms_fused &&
        layer->attn_q_a->type == DS4_TENSOR_Q4_K && layer->attn_kv->type == DS4_TENSOR_Q4_K;
    if (ok && qa_kv_pair) {
        ok = dense_matmul_pair_typed(g->qr, g->kv_raw, model,
                                     layer->attn_q_a, layer->attn_kv,
                                     DS4_N_EMBD, q_rank, DS4_N_HEAD_DIM, g->attn_norm) != 0;
    } else if (ok) {
        ok = dense_matmul_typed(g->qr, model, layer->attn_q_a,
                                DS4_N_EMBD, q_rank, g->attn_norm, 1) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    if (qkv_rms_fused) {
        if (ok && !qa_kv_pair) ok = dense_matmul_typed(g->kv_raw, model, layer->attn_kv,
                                          DS4_N_EMBD, DS4_N_HEAD_DIM, g->attn_norm, 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = dense_matmul_typed(g->q, model, layer->attn_q_b,
                                      q_rank, q_dim, g->qr_norm, 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    {   /* G1a 融合(2026-08-20 megakernel 施工): q 的 head_rms+rope 一发(CUDA 融合核
         * 早已在库但零接线; scale 折进旋转=容差级序差)。DS4_FUSE_QROPE=0 回两发。 */
        static int fq = -1;
        if (fq < 0) { const char *e = getenv("DS4_FUSE_QROPE"); fq = e ? atoi(e) : 1; }
        if (fq) {
            if (ok) ok = ds4_gpu_head_rms_norm_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                                            DS4_RMS_EPS) != 0;
        } else {
            if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = dense_matmul_typed(g->kv_raw, model, layer->attn_kv,
                                          DS4_N_EMBD, DS4_N_HEAD_DIM, g->attn_norm, 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
        }
    }
    {   /* G1b 三合一(2026-08-20 megakernel 施工): rope(kv)+fp8+store 一发(CUDA 真核;
         * rope 作用 rot 尾段/fp8 作用 nope 前段不相交, fp8 64线程树逐位照抄)。
         * DS4_FUSE_KVTAIL=0 或 n_head_kv≠1 走原三发。 */
        static int fkv = -1;
        if (fkv < 0) { const char *e = getenv("DS4_FUSE_KVTAIL"); fkv = e ? atoi(e) : 1; }
        if (fkv && DS4_N_HEAD_KV == 1 && !metal_graph_use_reference_kv_decode()) {
            if (ok) ok = ds4_gpu_kv_rope_fp8_store_raw_tensor(g->kv, raw_cache, raw_cap, raw_row,
                                            DS4_N_HEAD_DIM, DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        } else {
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) {
                metal_graph_debug_dump_tensor("KVrope", g->kv, DS4_N_HEAD_DIM, il, pos);
            }
            if (ok) ok = metal_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", g->kv, DS4_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    int comp_side = 0;   /* 非 emit token: comp 链发侧流与 indexer/attention 并发 */
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: Metal graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        /* comp 链回归主流串行(2026-08-19): 曾试双流并发(2026-08-17 第九夜), 实测
         * decode 吞吐零收益, 且与主流 indexer 链存在未根除的数据竞争 —— 温 0 长生成
         * (n=256)每 run 输出漂移, 关闭后逐字节可复现。侧流机制保留给 shared expert
         * 段(已验证确定且无竞争)。 */
        comp_side = 0; (void)emit;
        if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
            ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_side,
                                                  g->comp_sc_side,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_compressor_kv->abs_offset,
                                                  layer->attn_compressor_gate->abs_offset,
                                                  DS4_N_EMBD,
                                                  comp_width,
                                                  g->attn_norm,
                                                  1) != 0;
        } else {
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_side, model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_side, model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_side,
                                                        g->comp_sc_side,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        metal_graph_attn_comp_update_target(g, il),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        metal_graph_attn_comp_update_row(comp_row),
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS) != 0;
        if (comp_side) (void)ds4_gpu_side_main();   /* comp 链留侧流, 主流继续 indexer */
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    metal_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
                ds4_gpu_tensor_free(comp_row_view);
            }
            if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: Metal graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
                ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->indexer_compressor_kv->abs_offset,
                                                      layer->indexer_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      index_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            g->layer_index_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
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
                        g->layer_index_comp_cache[il],
                        (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                }
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                metal_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: Metal graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: Metal graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_q, model->map, model->size,
                                                         layer->indexer_attn_q_b->abs_offset,
                                                         q_rank, indexer_q_dim,
                                                         g->qr_norm, 1) != 0;
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_weights, model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         g->attn_norm, 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           DS4_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * DS4_N_INDEXER_TOP_K.
                     */
                    n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? DS4_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_METAL_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    metal_graph_attn_comp_cache_is_f16(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         metal_graph_attn_comp_cache_is_f16(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    if (comp_side) { (void)ds4_gpu_side_join(); comp_side = 0; }
    DS4_METAL_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc =
        !metal_graph_directional_steering_attn_enabled(g) &&
        !metal_graph_use_reference_attn_out_hc() &&
        layer->attn_output_a->type != DS4_TENSOR_Q2_K &&   /* 全q2: 融合家族无 q2 实现, 落批量路 */
        layer->attn_output_b->type != DS4_TENSOR_Q2_K;
    if (ok && fuse_attn_out_hc) {
        const bool attn_out_q4k = layer->attn_output_a->type == DS4_TENSOR_Q4_K;
        ok = (attn_out_q4k
                  ? ds4_gpu_attention_output_low_q4k_tensor(g->attn_low,
                                                            model->map,
                                                            model->size,
                                                            layer->attn_output_a->abs_offset,
                                                            group_dim,
                                                            rank,
                                                            n_groups,
                                                            g->heads)
                  : ds4_gpu_attention_output_low_q8_tensor(g->attn_low,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_output_a->abs_offset,
                                                           group_dim,
                                                           rank,
                                                           n_groups,
                                                           g->heads)) != 0;
        if (ok) {
            ok = (layer->attn_output_b->type == DS4_TENSOR_Q4_K
                      ? ds4_gpu_matmul_q4_K_hc_expand_tensor(g->after_attn_hc,
                                                             g->attn_out,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_output_b->abs_offset,
                                                             (uint64_t)n_groups * rank,
                                                             DS4_N_EMBD,
                                                             g->attn_low,
                                                             g->cur_hc,
                                                             g->hc_split,
                                                             DS4_N_EMBD,
                                                             DS4_N_HC)
                      : ds4_gpu_matmul_q8_0_hc_expand_tensor(g->after_attn_hc,
                                                             g->attn_out,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_output_b->abs_offset,
                                                             (uint64_t)n_groups * rank,
                                                             DS4_N_EMBD,
                                                             g->attn_low,
                                                             g->cur_hc,
                                                             g->hc_split,
                                                             DS4_N_EMBD,
                                                             DS4_N_HC)) != 0;
        }
    } else if (ok) {
        ok = ((layer->attn_output_a->type == DS4_TENSOR_Q4_K || layer->attn_output_a->type == DS4_TENSOR_Q2_K)
                  ? attn_output_kq_batch(layer->attn_output_a, g->attn_out,
                                                              g->attn_low,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_output_b->abs_offset,
                                                              group_dim, rank,
                                                              n_groups, DS4_N_EMBD,
                                                              g->heads, 1)
                  : ds4_gpu_attention_output_q8_batch_tensor(g->attn_out,
                                                             g->attn_low,
                                                             g->batch_group_tmp,
                                                             g->batch_low_tmp,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_output_a->abs_offset,
                                                             layer->attn_output_b->abs_offset,
                                                             group_dim, rank,
                                                             n_groups, DS4_N_EMBD,
                                                             g->heads, 1)) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", g->attn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, g->attn_out, il, 1);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
    }
    const uint64_t gate_row_bytes = (layer->ffn_gate_exps ? routed_expert_row_bytes(layer->ffn_gate_exps) : 0);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
    /* go1b correction: bias the raw router logits by delta[e] before top-k. Only
     * score-routed layers select by logits (hash layers select by token id).
     * δ≡0 sidecars skip the dispatch entirely — it costs an owned-CB sync per
     * layer on the offload decode path. */
    if (ok && model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present &&
        model->corr->layer[il].has_delta && layer->ffn_gate_tid2eid == NULL) {
        ok = ds4_gpu_corr_router_bias(g->router_logits, model->corr->layer[il].gdelta,
                                      DS4_N_EXPERT, 1) != 0;
    }
    /* 路由闭式侧车(type8): δlogits 加在 raw logits 上, select 前(2026-08-19) */
    if (ok && ds4_zchain_layer_rte(model->zchain, il))
        ok = ds4_gpu_zchain_route_bias(g->router_logits, g->ffn_norm, il, 1) != 0;
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                DS4_N_EXPERT,
                                                DS4_N_EXPERT_USED,
                                                DS4_EXPERT_WEIGHT_SCALE,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits, il) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
    }
    /* 判决钩: 路由+专家输入钉锚(解算 yq 口径完整还原)。router kernel 仍在队列 →
     * 必须先定格(signal→flush→host_wait, cap_batch_layer 同款)再覆写, 否则 host 写
     * 会被之后才执行的 kernel 冲掉。专家前向的 x 同时钉锚: 解算的 yq=专家(锚X0),
     * 只钉路由不钉专家输入, routed 本身仍与解算对不上。 */
    ds4_gpu_tensor *anc_moe_x = g->ffn_norm;
    if (ok && ampanc_on() && g_ampanc.route_on && pos < g_ampanc.S && il < g_ampanc.nl) {
        const uint64_t anc_ev = ds4_gpu_tp_signal_after_batch();
        if (anc_ev) { (void)ds4_gpu_flush_commands(); (void)ds4_gpu_tp_host_wait(anc_ev); }
        const uint64_t anc_row = (uint64_t)il * g_ampanc.S + pos;
        ok = ds4_gpu_tensor_write(g->router_selected, 0,
                                  g_ampanc.ridx + anc_row * g_ampanc.nact,
                                  DS4_N_EXPERT_USED * sizeof(int32_t)) != 0;
        if (ok) ok = ds4_gpu_tensor_write(g->router_weights, 0,
                                          g_ampanc.rw + anc_row * g_ampanc.nact,
                                          DS4_N_EXPERT_USED * sizeof(float)) != 0;
        if (ok && il < DS4_MAX_LAYER) {
            if (!g_ampanc.xbuf[il])
                g_ampanc.xbuf[il] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
            if (g_ampanc.xbuf[il] &&
                ds4_gpu_tensor_write(g_ampanc.xbuf[il], 0,
                                     g_ampanc.fin + anc_row * g_ampanc.dim,
                                     (uint64_t)DS4_N_EMBD * sizeof(float)) != 0)
                anc_moe_x = g_ampanc.xbuf[il];
        }
    }
    /* zchain GE: fold per-expert gains into the router weights. Must run on
     * ORIGINAL expert ids, i.e. before the compact-slot translate below. */
    if (ok && zch && ds4_zchain_layer_ge(zch, il)) {
        ok = ds4_gpu_zchain_ge_apply(g->router_weights, g->router_selected,
                                     il, DS4_N_EXPERT_USED, 1) != 0;
    }
    /* Translate full-256 router ids to compact slots for a shrunken model. No-op
     * for a full model (no LUT set). Runs after the router (weights were gathered
     * with original ids) and before the routed matvec indexes the kept tensors. */
    if (ok && model->expert_shrunken) {
        ok = ds4_gpu_translate_expert_ids(g->router_selected, il, DS4_N_EXPERT_USED, 1,
                                          model_expert_kept_count(model, il)) != 0;
    }
    /* TP Phase 3 (DS4_TP_EXPERT_SPLIT): each peer gathers + computes only its half
     * of the routed experts (owns_low: slots [0,k); owns_high: [k,n_used)), then the
     * partial routed_out is all-reduce SUMMED below (no zero-half). The gather
     * (compact_selected_experts reads n_expert slots from selected_off) thus fetches
     * only k experts per peer => halves the per-token routed-expert SSD IO, the
     * dominant cold-decode cost. Views pick the owned slots (contiguous for the
     * single decode token). Default OFF => full gather + skeleton zero-half. */
    const char *tp_es_env = getenv("DS4_TP_EXPERT_SPLIT");
    bool tp_expert_split = false;
    ds4_gpu_tensor *es_sel = NULL, *es_wt = NULL;
    uint32_t es_n = DS4_N_EXPERT_USED;
    if (g->tp && il < g->tp_layers && tp_es_env && tp_es_env[0] && tp_es_env[0] != '0' &&
        DS4_N_EXPERT_USED >= 2u) {
        const uint32_t n_used = DS4_N_EXPERT_USED;
        /* Asymmetric split (DS4_TP_SPLIT_LOW): the coordinator (owns_low, faster M4)
         * takes k experts, the worker (owns_high, slower M1) takes n_used-k. Tuning k
         * up balances the per-layer time so the AR's lockstep wait (the dominant cold
         * TP cost) shrinks. Default n_used/2 (50/50). */
        uint32_t k = n_used / 2u;
        const char *sl = getenv("DS4_TP_SPLIT_LOW");
        if (sl && sl[0]) { unsigned long v = strtoul(sl, NULL, 10); if (v >= 1u && v < n_used) k = (uint32_t)v; }
        const uint32_t slot_start = g->tp_owns_low ? 0u : k;
        const uint32_t cnt = g->tp_owns_low ? k : (n_used - k);
        es_sel = ds4_gpu_tensor_view(g->router_selected,
                                     (uint64_t)slot_start * sizeof(int32_t),
                                     (uint64_t)cnt * sizeof(int32_t));
        es_wt = ds4_gpu_tensor_view(g->router_weights,
                                    (uint64_t)slot_start * sizeof(float),
                                    (uint64_t)cnt * sizeof(float));
        if (es_sel && es_wt) { tp_expert_split = true; es_n = cnt; }
        else { ds4_gpu_tensor_free(es_sel); ds4_gpu_tensor_free(es_wt); es_sel = es_wt = NULL; }
    }
    const int moe_side_mark = ok ? ds4_gpu_side_mark() : 0;
    (void)moe_side_mark;
    if (ok && (routed_expert_quant_type(layer) == DS4_TENSOR_GO1B ||
               routed_expert_quant_type(layer) == DS4_TENSOR_GO2B)) {
        /* go1b (strict 1-bit) and go2b (2-bit ±d1±d2, monolithic mixed base) routed
         * experts have no hand-written mul_mv_id decode kernel; they run exclusively
         * through the grouped mm_id matmul, which is a correct general GEMM even at
         * n_tokens=1.  Route single-token decode through the batch-tensor path (it
         * forces the mm_id kernel for mv-less quant types via force_mm=nil-mv-pipeline).
         * go1b_mid_f16 is an unused output here (only the batch caller threads it). */
        bool go1b_mid_f16 = false;
        ok = ds4_gpu_routed_moe_batch_tensor(g->routed_out,
                                             g->routed_gate,
                                             g->routed_up,
                                             g->routed_mid,
                                             g->routed_down,
                                             residual_set_for(model, il),
                                             model->map, model->size,
                                             routed_expert_gate_off(layer),
                                             routed_expert_up_off(layer),
                                             layer->ffn_down_exps->abs_offset,
                                             routed_expert_quant_type(layer),
                                             layer->ffn_down_exps->type,
                                             gate_expert_bytes, gate_row_bytes,
                                             down_expert_bytes, down_row_bytes,
                                             (uint32_t)expert_in_dim,
                                             (uint32_t)down_in_dim,
                                             (uint32_t)routed_out_dim,
                                             tp_expert_split ? es_sel : g->router_selected,
                                             tp_expert_split ? es_wt  : g->router_weights,
                                             model_expert_kept_count(model, il),
                                             es_n, DS4_SWIGLU_CLAMP_EXP, anc_moe_x,
                                             il, 1u, 0u, 0u, &go1b_mid_f16) != 0;
    } else if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 residual_set_for(model, il),
                                                 model->map, model->size,
                                                 routed_expert_gate_off(layer),
                                                 routed_expert_up_off(layer),
                                                 layer->ffn_down_exps->abs_offset,
                                                 routed_expert_quant_type(layer),
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 tp_expert_split ? es_sel : g->router_selected,
                                                 tp_expert_split ? es_wt  : g->router_weights,
                                                 model_expert_kept_count(model, il),
                                                 es_n, DS4_SWIGLU_CLAMP_EXP, anc_moe_x,
                                                 il) != 0;
    ds4_gpu_tensor_free(es_sel);   /* NULL-safe; views are cheap wrappers */
    ds4_gpu_tensor_free(es_wt);
    DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
    }
    /* TP Stage 2: recombine routed_out across the two peers. Rather than draining
     * the whole pipeline (waitUntilCompleted), signal a MTLSharedEvent at the end
     * of the batch, flush (commit without a full wait), and host-wait that value
     * on the fast event path so routed_out becomes host-visible cheaply. Each peer
     * then zeros its non-owned half and sum-all-reduces, reproducing the single-
     * machine routed output bit-for-bit. The host writeback lands in unified
     * memory before the (now reopened) batch's combine is committed, so no GPU
     * wait-back is needed. Guarded by g->tp ⇒ non-TP path never touches this. */
    if (ok && g->tp && il < g->tp_layers) {
        /* wave-72 diag (DS4_TP_AR_LOG): the decode TP all-reduce had never run
         * dual-host before Phase 1 (prefill uses prefill_layer_major, not this
         * encode_decode_layer path). Per-step logging pins which of the 6 calls
         * flips ok on first real use. */
        const bool ar_log = getenv("DS4_TP_AR_LOG") != NULL;
        const double ar_t0 = ar_log ? now_sec() : 0.0;
        const uint64_t ev = ds4_gpu_tp_signal_after_batch();
        ok = ev != 0;
        if (ok) ok = ds4_gpu_flush_commands() != 0;   /* commit (no full drain), reopen batch */
        if (ok) ok = ds4_gpu_tp_host_wait(ev) != 0;    /* fast wait until routed_moe done */
        const double ar_t_drain = ar_log ? now_sec() : 0.0;   /* signal+flush+host_wait = GPU drain */
        if (ok) ok = ds4_gpu_tensor_read(g->routed_out, 0, g->tp_vec,
                                         (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        if (ok) {
            /* Phase 3: with the expert split, routed_out is already this peer's
             * partial (weighted sum over its owned slots) -> AR-sum reconstructs
             * the full routed output. Without it (skeleton), each peer computed the
             * FULL routed_out, so zero the non-owned n_embd half before the sum to
             * avoid double-counting (the original element-range placeholder). */
            if (!tp_expert_split) {
                const uint32_t half = DS4_N_EMBD / 2;
                if (g->tp_owns_low) {
                    for (uint32_t i = half; i < DS4_N_EMBD; i++) g->tp_vec[i] = 0.0f;
                } else {
                    for (uint32_t i = 0; i < half; i++) g->tp_vec[i] = 0.0f;
                }
            }
            ok = ds4_dist_tp_allreduce_f32(g->tp, g->tp_vec, DS4_N_EMBD) == 0;
        }
        if (ok) ok = ds4_gpu_tensor_write(g->routed_out, 0, g->tp_vec,
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        if (ar_log) {
            const double ar_t1 = now_sec();
            fprintf(stderr, "ds4: tp-ar il=%u pos=%u ok=%d drain=%.1fms net=%.1fms total=%.1fms\n",
                    il, pos, ok, (ar_t_drain - ar_t0) * 1e3,
                    (ar_t1 - ar_t_drain) * 1e3, (ar_t1 - ar_t0) * 1e3);
        }
    }
    /* zchain λ(x): scale the (now complete) routed output. After the TP
     * all-reduce so every path sees the full routed sum; before the additive
     * corr, matching the quantizer's op order (λ never scales corr terms). */
    if (ok && (ds4_zchain_layer_has_lambda(zch, il) ||
               ds4_zchain_layer_zl(zch, il))) {   /* λ 和/或 冻结 z^L 同一派发 */
        /* 判决钩: 放大器的 x 钉锚(解算口径还原)。xbuf 无队列写者, host 写先于后续
         * kernel 提交即序正确; per-layer buf 防跨层复用被未执行的前层 kernel 误读。 */
        ds4_gpu_tensor *anc_zx = g->ffn_norm;
        if (ampanc_on() && pos < g_ampanc.S && il < g_ampanc.nl && il < DS4_MAX_LAYER) {
            if (!g_ampanc.xbuf[il])
                g_ampanc.xbuf[il] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
            if (g_ampanc.xbuf[il] &&
                ds4_gpu_tensor_write(g_ampanc.xbuf[il], 0,
                                     g_ampanc.fin + ((uint64_t)il * g_ampanc.S + pos) * g_ampanc.dim,
                                     (uint64_t)DS4_N_EMBD * sizeof(float)) != 0)
                anc_zx = g_ampanc.xbuf[il];
        }
        ok = ds4_gpu_zchain_scale_routed(g->routed_out, anc_zx, il, 1) != 0;
    }
    /* go1b correction: add the low-rank per-expert residual onto the (now full)
     * routed MoE output. Placed after the TP all-reduce so routed_out is complete
     * for every path (single-host, layer-sliced, and TP expert-split); g->ffn_norm
     * (the expert input x) and g->router_selected (full 6 ids) are still intact. */
    /* Engine-trajectory decode capture (DS4_CAP_DIR): the perplexity scorer
     * (teacher-forced trajectory runs) flows token-by-token through HERE, not
     * the batch path — same shards, one token per call. Reads happen before
     * the corr mutates routed_out semantics for downstream x̂ definitions
     * (x̂ = ffn_norm is already final at this point). */
    if (ok) cap_decode_layer(g, il);
    /* Consumer-path predicates, hoisted above the corr dispatch: the corr may
     * only take the store-to-delta form when the fused shared-down consumer
     * (the one kernel that performs the routed+delta add) is what will run
     * below — every other consumer keeps the legacy in-place corr. */
    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos);
    const char *tp_split_env = getenv("DS4_TP_SHARED_SPLIT");
    const bool tp_shared_split =
        g->tp && il < g->tp_layers &&
        tp_split_env && tp_split_env[0] && tp_split_env[0] != '0' &&
        (shared_dim % 64u) == 0;   /* half must stay 32-block aligned */
    const bool fuse_shared_down_hc =
        !tp_shared_split &&
        layer->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&   /* 融合 down+hc kernel 是 q8 专用;
                                                             * q4_K 落到下方非融合 dense_matmul_typed */
        !keep_ffn_out && !metal_graph_use_reference_shared_down_hc();
    bool corr_delta_live = false;
    /* DS4_CORR_SKIP=1: load the sidecar but skip the decode dispatch — perf
     * splitter isolating "corr resident" from "corr kernel dispatched". */
    if (ok && model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present &&
        !getenv("DS4_CORR_SKIP")) {
        const ds4_corr_layer *cl = &model->corr->layer[il];
        /* For go1b the routed MoE (offload batch-tensor path) REMAPS g->router_selected
         * to compact slots IN PLACE before this point, so the corr must index per-expert
         * C[e]/beta[e] from the pre-remap snapshot the MoE saved, not the live (corrupted)
         * selected tensor. Fall back to router_selected for any non-go1b/resident path
         * that never remaps (snapshot NULL). */
        const ds4_gpu_tensor *corr_sel = ds4_gpu_corr_saved_selected();
        if (!corr_sel) corr_sel = g->router_selected;
        /* φ selector: legacy x = ffn_norm; --feat yhat sidecars read routed_out
         * itself (kernel phase1 consumes φ fully before phase2 writes out). */
        const ds4_gpu_tensor *phi = model->corr->phi_yhat ? g->routed_out : g->ffn_norm;
        if (getenv("DS4_CORR_MODE_TRACE")) {
            static int traced;
            if (traced++ < 4)
                fprintf(stderr, "ds4: corr-mode il=%u fuse=%d delta_buf=%d supported=%d\n",
                        il, (int)fuse_shared_down_hc, g->corr_delta != NULL,
                        ds4_gpu_corr_delta_supported());
        }
        if (fuse_shared_down_hc && g->corr_delta && ds4_gpu_corr_delta_supported() &&
            !getenv("DS4_CORR_INPLACE")) {
            /* Store variant: the tiny corr dispatch writing the hot routed_out
             * costs a full pipeline drain per layer (measured ~23ms; φ=ŷ makes
             * it a R/W self-alias). Write the correction to corr_delta and let
             * the fused shared-down consumer add it — bit-identical fadd. */
            ok = ds4_gpu_corr_apply_delta(g->corr_delta, phi,
                                          cl->gU, cl->gV, cl->gC, cl->gb, cl->gbeta,
                                          corr_sel,
                                          DS4_N_EMBD, cl->d_l, DS4_N_EXPERT, DS4_N_EXPERT_USED, 1) != 0;
            corr_delta_live = ok;
        } else {
            ok = ds4_gpu_corr_apply(g->routed_out, phi,
                                    cl->gU, cl->gV, cl->gC, cl->gb, cl->gbeta,
                                    corr_sel,
                                    DS4_N_EMBD, cl->d_l, DS4_N_EXPERT, DS4_N_EXPERT_USED, 1) != 0;
        }
    }
    /* TP Phase 1 (DS4_TP_SHARED_SPLIT): split the shared-expert dense FFN across
     * the two peers. gate/up are column-parallel (each peer computes its half of
     * the shared_dim rows, compacted into mid[0,half)); down is row-parallel
     * (partial out[n_embd] over the owned input-dim half) then all-reduced. Decode
     * only (this is the n_tok=1 layer encode; prefill stays replicated/full). Halves
     * the shared-FFN weight bandwidth per peer. Off => byte-identical to the legacy
     * path. tp_owns_low owns the low half, matching the routed-MoE skeleton above.
     * (tp_shared_split itself is hoisted above the corr dispatch: the corr
     * delta-vs-in-place choice must know which shared-down consumer runs.) */
    const uint32_t tp_half = shared_dim / 2u;
    const uint32_t tp_out_start = g->tp_owns_low ? 0u : tp_half;
    const bool fuse_shared_gate_up =
        !tp_shared_split &&
        !g->quality &&
        layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&   /* 融合 kernel 是 q8 专用 */
        getenv("DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    int shared_side = 0;   /* 本层 shared 三件套是否发在侧流(与 MoE 并发) */
    if (ok && tp_shared_split) {
        /* column-parallel gate/up: owned half rows -> shared_mid[0, tp_half). The
         * kernel's output row index starts at 0, so shifting the weight offset by
         * the owned rows writes the owned slice compacted into [0, tp_half). */
        const uint64_t gate_row_bytes = ((uint64_t)DS4_N_EMBD / 32u) * 34u;
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset + (uint64_t)tp_out_start * gate_row_bytes,
                                                         layer->ffn_up_shexp->abs_offset   + (uint64_t)tp_out_start * gate_row_bytes,
                                                         DS4_N_EMBD,
                                                         tp_half,
                                                         g->ffn_norm,
                                                         DS4_SWIGLU_CLAMP_EXP) != 0;
    } else if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm,
                                                         DS4_SWIGLU_CLAMP_EXP) != 0;
    } else {
        /* 侧流并发: 与 routed MoE(主流)同读 ffn_norm, 输出到 hc_expand_add_split 前汇合 */
        shared_side = (ok && moe_side_mark) ? ds4_gpu_side_begin() : 0;
        if (ok) ok = dense_matmul_pair_typed(g->shared_gate, g->shared_up, model,
                                             layer->ffn_gate_shexp, layer->ffn_up_shexp,
                                             DS4_N_EMBD, shared_dim, shared_dim, g->ffn_norm) != 0;
        if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
                                           shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        if (ok && shared_side) {
            ok = dense_matmul_typed(g->shared_out, model, layer->ffn_down_shexp,
                                      shared_dim, DS4_N_EMBD, g->shared_mid, 1) != 0;
            if (ok) ok = ds4_gpu_side_join() != 0;
        } else {
            (void)ds4_gpu_side_join();
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
    /* keep_ffn_out / fuse_shared_down_hc are declared above the corr dispatch */
    if (ok && tp_shared_split) {
        /* row-parallel down: partial out[n_embd] over the owned in-dim half (x is
         * the compacted shared_mid[0, tp_half); the weight starts tp_out_start/32
         * blocks * 34 bytes into each row), then all-reduce to the full down output.
         * Same MTLSharedEvent fast-wait + host all-reduce as the routed-MoE skeleton:
         * signal end of batch, flush (no full drain), fast-wait, read partial, sum
         * across peers, write back. Combine into the residual via the unfused
         * hc_expand_add_split path below (fuse_shared_down_hc forced false). */
        ok = ds4_gpu_matmul_q8_0_rowslice_tensor(g->shared_out, model->map, model->size,
                                                  layer->ffn_down_shexp->abs_offset + ((uint64_t)tp_out_start / 32u) * 34u,
                                                  shared_dim, tp_half, DS4_N_EMBD,
                                                  g->shared_mid) != 0;
        if (ok) {
            const uint64_t ev = ds4_gpu_tp_signal_after_batch();
            ok = ev != 0;
            if (ok) ok = ds4_gpu_flush_commands() != 0;
            if (ok) ok = ds4_gpu_tp_host_wait(ev) != 0;
            if (ok) ok = ds4_gpu_tensor_read(g->shared_out, 0, g->tp_vec,
                                             (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            if (ok) ok = ds4_dist_tp_allreduce_f32(g->tp, g->tp_vec, DS4_N_EMBD) == 0;
            if (ok) ok = ds4_gpu_tensor_write(g->shared_out, 0, g->tp_vec,
                                              (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        }
    } else if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         corr_delta_live ? g->corr_delta : NULL,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok && !shared_side) {
        ok = dense_matmul_typed(g->shared_out, model, layer->ffn_down_shexp,
                                  shared_dim, DS4_N_EMBD, g->shared_mid, 1) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->ffn_out,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_METAL_PROFILE_DECODE_STAGE
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
    }
    if (ok) trace_hnorm_layer(g, il, pos, hc_dim);
    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on Metal. */
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_decode_layer_nonempty_tu; /* 空TU防御(CPU构建) */
