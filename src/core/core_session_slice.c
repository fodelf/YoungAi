/* core_session_slice.c — 层切片 eval/verify (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen) {
    if (!s || !s->engine) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    if (layer_start > layer_end || layer_end >= (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice layer range %u:%u",
                             layer_start, layer_end);
        return 1;
    }
    if (layer_start != 0 && !input_hc) {
        if (errlen) snprintf(err, errlen, "layer-slice layer %u requires input hidden-state",
                             layer_start);
        return 1;
    }
    if (output_logits && layer_end + 1u != (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "layer-slice logits require final transformer layer");
        return 1;
    }
    if (output_logits && !logits) {
        if (errlen) snprintf(err, errlen, "layer-slice logits output is missing");
        return 1;
    }
    /* A distributed prefill pipeline may need only the KV side effect for
     * non-final chunks. In that case both output_hc and logits are NULL. */
    if (ds4_session_slice_check_timeline(s, tokens, n_tokens, pos0, err, errlen) != 0) {
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        if (errlen) snprintf(err, errlen, "layer slices require the graph backend");
        s->checkpoint_valid = false;
        return 1;
    }
#ifdef DS4_NO_GPU
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    s->checkpoint_valid = false;
    return 1;
#else
    if (n_tokens > s->prefill_cap) {
        if (errlen) snprintf(err, errlen, "layer-slice chunk %u exceeds prefill cap %u",
                             n_tokens, s->prefill_cap);
        return 1;
    }

    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    if (!input_hc && !output_hc && output_logits &&
        layer_start == 0 && layer_end + 1u == (uint32_t)DS4_N_LAYER) {
        bool ok = false;
        ds4_tokens span = {0};
        if (pos0 == 0) {
            span.v = (int *)tokens;
            span.len = (int)n_tokens;
            span.cap = (int)n_tokens;
            ok = metal_graph_prefill_layer_major(g,
                                                 &e->model,
                                                 &e->weights,
                                                 &span,
                                                 0,
                                                 n_tokens,
                                                 logits,
                                                 false,
                                                 NULL,
                                                 NULL,
                                                 NULL);
        } else if (n_tokens == 1) {
            ok = metal_graph_eval_token_raw_swa(g,
                                                &e->model,
                                                &e->weights,
                                                tokens[0],
                                                pos0,
                                                logits);
        } else {
            if (pos0 > (uint32_t)INT_MAX - n_tokens) {
                if (errlen) snprintf(err, errlen, "layer-slice full span is too large");
                s->checkpoint_valid = false;
                return 1;
            }
            span.len = (int)(pos0 + n_tokens);
            span.cap = span.len;
            span.v = calloc((size_t)span.len, sizeof(span.v[0]));
            if (span.v) {
                for (uint32_t i = 0; i < n_tokens; i++) span.v[pos0 + i] = tokens[i];
                ok = metal_graph_prefill_layer_major(g,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     pos0,
                                                     n_tokens,
                                                     logits,
                                                     false,
                                                     NULL,
                                                     NULL,
                                                     NULL);
            }
            free(span.v);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice full failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice full evaluation failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }
        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = (uint64_t)n_tokens * hc_dim * sizeof(float);
    if (n_tokens == 1 && pos0 > 0) {
        if (g->raw_cap == 0) {
            if (errlen) snprintf(err, errlen, "%s layer-slice decode has no raw KV cache",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        bool ok = true;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(g->cur_hc, 0, input_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok && !input_hc) {
            ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                               e->model.map,
                                               e->model.size,
                                               e->weights.token_embd->abs_offset,
                                               (uint32_t)e->weights.token_embd->dim[1],
                                               (uint32_t)tokens[0],
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;
        }
        const uint32_t raw_row = pos0 % g->raw_cap;
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, 1);
        const uint32_t split_after_layers = metal_graph_token_split_after_layers();
        uint32_t encoded_layers = 0;
        for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
            ok = metal_graph_encode_decode_layer(g,
                                                 &e->model,
                                                 &e->weights.layer[il],
                                                 il,
                                                 pos0,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 tokens[0]);
            ds4_gpu_tensor *tmp = g->cur_hc;
            g->cur_hc = g->after_ffn_hc;
            g->after_ffn_hc = tmp;
            encoded_layers++;
            if (ok &&
                split_after_layers != 0 &&
                (encoded_layers % split_after_layers) == 0 &&
                il < layer_end)
            {
                if (metal_graph_direct_expert_read_enabled()) {
                    ok = ds4_gpu_end_commands() != 0 && ds4_gpu_begin_commands() != 0;
                } else {
                    ok = ds4_gpu_flush_commands() != 0;
                }
            }
        }
        if (ok && output_logits) {
            ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
        if (ok && output_hc) {
            ok = ds4_gpu_tensor_read(g->cur_hc, 0, output_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok && output_logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice decode failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice decode failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    ds4_tokens span = {
        .v = (int *)tokens,
        .len = (int)n_tokens,
        .cap = (int)n_tokens,
    };

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, &span, 0, n_tokens);
    if (ok && input_hc) {
        ok = ds4_gpu_tensor_write(g->batch_cur_hc, 0, input_hc, hc_bytes) != 0;
    } else if (ok) {
        ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     0,
                                                     n_tokens);
    }

    ds4_gpu_tensor *last_hc = NULL;
    ds4_gpu_tensor *saved_cur = NULL;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
        ok = metal_graph_encode_layer_batch(g,
                                            &e->model,
                                            &e->weights.layer[il],
                                            il,
                                            pos0,
                                            n_tokens);
    }
    if (ok && output_logits) {
        saved_cur = g->cur_hc;
        last_hc = metal_graph_tensor_row_view(g->batch_cur_hc, n_tokens - 1u, hc_dim);
        ok = last_hc != NULL;
        if (ok) {
            g->cur_hc = last_hc;
            ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
            g->cur_hc = saved_cur;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (saved_cur) g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);

    if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(g->batch_cur_hc, 0, output_hc, hc_bytes) != 0;
    }
    if (ok && output_logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after layer-slice failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s layer-slice failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }

    ds4_session_slice_commit_timeline(s, tokens, n_tokens);
    return 0;
#endif
}

/* docs/archive/mtp.md Phase 1 (Scheme A) cross-machine verifier: run a K-token candidate
 * batch through this worker's layer slice (layer_start..layer_end, which must be
 * the final transformer layer) and emit the per-row logits into
 * row_logits[i*vocab .. ]. This is the batch verification pass (docs/archive/mtp.md §3.2.2,
 * "末端出 K 组 logits"): row i predicts batch position i+1, so the coordinator
 * argmaxes each row to find the accepted speculative prefix and reuses the
 * boundary row to seed the next sampling step. The batch writes layer KV for
 * positions pos0..pos0+n_tokens-1 and commits all n_tokens to the timeline; the
 * rejected tail is rolled back afterward by truncating the timeline
 * (ds4_session_layer_slice_rollback) so the stale ring rows are overwritten on
 * the next eval. */
int ds4_session_verify_batch_argmax(ds4_session *s,
                                    const int *tokens,
                                    uint32_t n_tokens,
                                    uint32_t pos0,
                                    uint32_t layer_start,
                                    uint32_t layer_end,
                                    const float *input_hc,
                                    float *row_logits,
                                    char *err,
                                    size_t errlen) {
    if (!s || !s->engine || !tokens || !row_logits || n_tokens == 0) {
        if (errlen) snprintf(err, errlen, "invalid verify batch request");
        return 1;
    }
    if (layer_end + 1u != (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "verify batch requires the final transformer layer");
        return 1;
    }
    if (layer_start != 0 && !input_hc) {
        if (errlen) snprintf(err, errlen, "verify batch on a nonzero layer needs input hidden-state");
        return 1;
    }
    if (ds4_session_slice_check_timeline(s, tokens, n_tokens, pos0, err, errlen) != 0) {
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)pos0; (void)layer_start;
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    if (!g->spec_logits) {
        if (errlen) snprintf(err, errlen, "verify batch needs the MTP spec-logits buffer");
        return 1;
    }
    if (n_tokens > s->prefill_cap) {
        if (errlen) snprintf(err, errlen, "verify batch %u exceeds prefill cap %u",
                             n_tokens, s->prefill_cap);
        return 1;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = (uint64_t)n_tokens * hc_dim * sizeof(float);
    ds4_tokens span = { .v = (int *)tokens, .len = (int)n_tokens, .cap = (int)n_tokens };

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, &span, 0, n_tokens);
    if (ok && input_hc) {
        ok = ds4_gpu_tensor_write(g->batch_cur_hc, 0, input_hc, hc_bytes) != 0;
    } else if (ok) {
        ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     0,
                                                     n_tokens);
    }
    const bool vprof = getenv("DS4_SPEC_PROF") != NULL;
    const double vt0 = vprof ? now_sec() : 0.0;
    if (vprof) ds4_gpu_span_begin();
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    /* 批 CUDA 图(2026-08-21): 43 层 ~3.5k kernel 的相邻间隙吃掉 31% GPU 时间。
     * 捕获成图后单次发射, 间隙归零。编码会推进 host 侧压缩器计数 ⇒ 捕获失败必须先
     * 还原再重编码, 否则 KV 记账错位。 */
    uint32_t saved_n_comp[DS4_N_LAYER], saved_n_index[DS4_N_LAYER];
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        saved_n_comp[il] = g->layer_n_comp[il];
        saved_n_index[il] = g->layer_n_index_comp[il];
    }
    const int bgraph = (ok && layer_start == 0) ? ds4_gpu_batch_graph_begin((int)(pos0 & 3u)) : 0;
    for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
        ok = metal_graph_encode_layer_batch(g, &e->model, &e->weights.layer[il],
                                            il, pos0, n_tokens);
    }
    if (bgraph) {
        const int r = ds4_gpu_batch_graph_end_launch(ok ? 1 : 0);
        if (r < 0) {   /* 捕获失败: 图内 kernel 未执行 ⇒ 还原计数后重编码直发 */
            for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
                g->layer_n_comp[il] = saved_n_comp[il];
                g->layer_n_index_comp[il] = saved_n_index[il];
            }
            ok = true;
            for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
                ok = metal_graph_encode_layer_batch(g, &e->model, &e->weights.layer[il],
                                                    il, pos0, n_tokens);
            }
        }
    }
    const double vt1 = vprof ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (vprof) {
        static double enc_acc, wait_acc, gpu_acc; static uint32_t vn;
        const float gpu_ms = ds4_gpu_span_end();
        enc_acc += vt1 - vt0; wait_acc += now_sec() - vt1; vn++;
        if (gpu_ms > 0.0f) gpu_acc += gpu_ms;
        if ((vn & 15u) == 0)
            fprintf(stderr, "ds4: [vfy-prof] n=%u avg_ms: encode_cpu=%.1f end_wait=%.1f gpu_span=%.1f\n",
                    vn, enc_acc * 1e3 / vn, wait_acc * 1e3 / vn, gpu_acc / vn);
    }
    else (void)ds4_gpu_synchronize();
    if (!ok) {
        if (errlen) snprintf(err, errlen, "%s verify batch layers failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }

    /* Output head on all n_tokens rows -> spec_logits, read back K logit rows. */
    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head_batch(g, &e->model, &e->weights,
                                                      n_tokens, e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) {
        ok = ds4_gpu_tensor_read(g->spec_logits, 0, row_logits,
                                 (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    if (!ok) {
        if (errlen) snprintf(err, errlen, "%s verify batch output head failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    /* Commit all K candidate tokens to the timeline so checkpoint.len advances by
     * n_tokens (KV rows pos0..pos0+n_tokens-1 are live). The rejected tail is
     * trimmed afterward via ds4_session_layer_slice_rollback. */
    ds4_session_slice_commit_timeline(s, tokens, n_tokens);
    return 0;
#endif
}

/* docs/archive/mtp.md Phase 1: truncate the layer-slice timeline back to new_len positions
 * after a speculative batch so the rejected tail is dropped. The position-indexed
 * KV ring rows are not cleared; the next eval at new_len overwrites them, exactly
 * like the single-machine MTP rollback. */
int ds4_session_layer_slice_rollback(ds4_session *s, uint32_t new_len,
                                     char *err, size_t errlen) {
    if (!s) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    if (!s->checkpoint_valid || (uint32_t)s->checkpoint.len < new_len) {
        if (errlen) snprintf(err, errlen, "layer-slice rollback target %u exceeds timeline %d",
                             new_len, s->checkpoint.len);
        return 1;
    }
    s->checkpoint.len = (int)new_len;
    s->mtp_draft_valid = false;
    return 0;
}

uint32_t ds4_session_layer_slice_len(const ds4_session *s) {
    if (!s || !s->checkpoint_valid || s->checkpoint.len < 0) return 0;
    return (uint32_t)s->checkpoint.len;
}

