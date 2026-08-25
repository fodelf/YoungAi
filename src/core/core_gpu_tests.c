/* core_gpu_tests.c — metal decode/first-token 数值测试 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
int metal_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        /* this diagnostic threads layer index 0 into the prealloc above; keep the
         * router delta on the same layer so the selection stays consistent. */
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm, corr_layer_delta(model, 0));
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, layer);
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = metal_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        ds4_gpu_tensor *embedded_hc = g.cur_hc;
        g.cur_hc = g.after_ffn_hc;
        g.after_ffn_hc = embedded_hc;
    }
    if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_gpu_tensor_read(g.after_ffn_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(g.router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(g.routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: Metal graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("metal graph q", gpu_q, q_dim);
        print_vec_stats("metal graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("metal graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: Metal graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph test failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}

int metal_graph_first_token_full_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: full Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = weights->output->dim[1];
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));

    forward_first_token_cpu(cpu_hc, model, weights, token);
    output_logits_one(cpu_logits, model, weights, cpu_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, &weights->layer[0]);
    const bool trace_layers = getenv("DS4_METAL_GRAPH_TRACE_LAYERS") != NULL;
    if (trace_layers && ok) {
        g.materialize_ffn_out = true;
        const bool teacher_force = getenv("DS4_METAL_GRAPH_TEACHER_FORCE") != NULL;
        const char *stage_layer_env = getenv("DS4_METAL_GRAPH_TRACE_STAGE_LAYER");
        const long stage_layer = stage_layer_env ? strtol(stage_layer_env, NULL, 10) : -1;
        float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        float *cpu_cur = xmalloc((size_t)hc_dim * sizeof(float));
        float *cpu_next = xmalloc((size_t)hc_dim * sizeof(float));

        embed_token_f16(model, weights, token, plain);
        hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC);
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            if (teacher_force) {
                ok = ds4_gpu_tensor_write(g.cur_hc, 0, cpu_cur, hc_dim * sizeof(float)) != 0;
            }
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                       il, 0, g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
            if (ok) ok = ds4_gpu_end_commands() != 0;

            layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
            if (ok) ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0;
            if (ok) {
                fprintf(stderr,
                        "ds4: Metal full graph layer %u%s hc_max=%g hc_rms=%g\n",
                        il,
                        teacher_force ? " teacher" : "",
                        max_abs_diff(cpu_next, gpu_hc, hc_dim),
                        rms_abs_diff(cpu_next, gpu_hc, hc_dim));
                if (stage_layer == (long)il) {
                    metal_graph_trace_layer_stages(&g, model, &weights->layer[il], cpu_cur, il, token);
                }
            }
            float *ctmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = ctmp;
        }

        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;

        free(cpu_next);
        free(cpu_cur);
        free(plain);
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                 il, 0, g.layer_raw_cache[il],
                                                 g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
        }

        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        const uint64_t cpu_top = argmax_f32(cpu_logits, vocab_dim);
        const uint64_t gpu_top = argmax_f32(gpu_logits, vocab_dim);
        fprintf(stderr,
                "ds4: Metal full first-token graph diffs: final_hc_max=%g final_hc_rms=%g logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                rms_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                rms_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
    } else {
        fprintf(stderr, "ds4: Metal full first-token graph test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after full graph failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_hc);
    free(cpu_hc);
    return ok ? 0 : 1;
}

#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_tests_nonempty_tu; /* 空TU防御(CPU构建) */
