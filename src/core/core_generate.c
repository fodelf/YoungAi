/* core_generate.c — generate 循环(CPU/Metal) (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
int generate_raw_swa_cpu(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        const float       * directional_steering_dirs,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    (void)progress;
    (void)progress_ud;
    fprintf(stderr, "ds4: using CPU generation with layer-major prefill\n");

    ds4_kv_cache cache;
    kv_cache_init(&cache, (uint32_t)ctx_size, 0);
    ds4_cpu_decode_scratch decode_scratch;
    cpu_decode_scratch_init(&decode_scratch, (uint32_t)ctx_size);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    int pos = prompt->len;
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const double t_prefill0 = now_sec();

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        free(logits);
        cpu_decode_scratch_free(&decode_scratch);
        kv_cache_free(&cache);
        return 1;
    }

    prefill_layer_major_cpu(logits, model, weights, &cache, prompt,
                            directional_steering_dirs,
                            directional_steering_attn,
                            directional_steering_ffn);

    const double t_prefill1 = now_sec();
    fprintf(stderr, "ds4: prefill %d/%d done\n", prompt->len, prompt->len);
    const char *dump_prefill_logits = getenv("DS4_CPU_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            cpu_decode_scratch_free(&decode_scratch);
            kv_cache_free(&cache);
            return 1;
        }
        fprintf(stderr, "ds4: wrote CPU prefill logits to %s\n", dump_prefill_logits);
    }

    int n_generated = 0;
    int n_decode_eval = 0;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        /* The CPU decode step is expected to reuse buffers from
         * cpu_decode_scratch.  Keep the allocation guard tightly scoped to the
         * decode math itself; sampling, token emission, tracing, and callbacks
         * may allocate small temporary strings without invalidating that
         * guarantee. */
        ds4_alloc_guard_begin("CPU token decode");
        forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, &cache, token, (uint32_t)pos,
                                                 directional_steering_dirs,
                                                 directional_steering_attn,
                                                 directional_steering_ffn,
                                                 &decode_scratch);
        ds4_alloc_guard_end();
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    free(logits);
    cpu_decode_scratch_free(&decode_scratch);
    kv_cache_free(&cache);
    return 0;
}

#ifndef DS4_NO_GPU
/* Metal generation entry point.  The model runs as one local whole-graph
 * pipeline: chunked/layer-major prefill followed by graph decode steps. */
int generate_metal_graph_raw_swa(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        int                 power_percent,
        const char        * directional_steering_file,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using GPU graph generation with layer-major graph prefill\n");

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }

    const uint32_t prefill_cap = metal_graph_prefill_cap_for_prompt(prompt->len);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);
    if (prefill_cap < (uint32_t)prompt->len) {
        fprintf(stderr,
                "ds4: using chunked GPU prefill (%u-token chunks for %d prompt tokens)\n",
                prefill_cap,
                prompt->len);
    }
    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false,
                                        0, (uint32_t)DS4_N_LAYER - 1u, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate GPU graph runtime\n");
        return 1;
    }
    g.quality = quality;
    g.power_percent = power_percent > 0 ? (uint32_t)power_percent : 100u;
    if (!metal_graph_load_directional_steering(&g,
                                               directional_steering_file,
                                               directional_steering_attn,
                                               directional_steering_ffn)) {
        metal_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;

    const double t_prefill0 = now_sec();
    if (prefill_cap < (uint32_t)prompt->len) {
        ok = metal_graph_prefill_chunked(&g, model, weights, prompt,
                                         prompt->len, logits, false,
                                         progress, progress_ud,
                                         progress, progress_ud);
    } else {
        ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt,
                                         prompt->len, logits, true,
                                         progress, progress_ud);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after prefill");

    if (!ok) {
        free(logits);
        metal_graph_free(&g);
        return 1;
    }
    const char *dump_prefill_logits = getenv("DS4_METAL_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            metal_graph_free(&g);
            return 1;
        }
        fprintf(stderr, "ds4: wrote GPU prefill logits to %s\n", dump_prefill_logits);
    }

        /* DSpark 探针(DS4_DSPARK_PROBE=1): 每 decode token 后跑 drafter 块, 打印草稿与
     * 后续真实 token 的命中; capture buffer 补分配(graph alloc 时 flag 未置)。 */
    static ds4_dspark_weights g_dsw;
    const bool dspark_probe = getenv("DS4_DSPARK_PROBE") != NULL;
    if (getenv("DS4_DSPARK_PROBE"))
        fprintf(stderr, "ds4: [dspark-probe] enter generate loop, probe=%d\n", (int)dspark_probe);
    int dspark_pending[DS4_DSPARK_BLK]; int dspark_pending_n = 0; uint32_t dspark_hit = 0, dspark_tot = 0;
    if (dspark_probe) {
        dspark_bind_with_draft(&g_dsw, model, true);
        if (g_dsw.ready) {
            g.dspark_capture = 1;
            if (!g.dspark_main_hidden) {
                const uint32_t B = 5u;
                g.dspark_main_hidden = ds4_gpu_tensor_alloc(3ull * DS4_N_EMBD * sizeof(float));
                g.dspark_main_x_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
                g.dspark_main_x = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
                g.dspark_kv_tmp = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
                for (int b = 0; b < 3; b++)
                    g.dspark_win_kv[b] = ds4_gpu_tensor_alloc(128ull * DS4_N_HEAD_DIM * sizeof(float));
                g.dspark_ids = ds4_gpu_tensor_alloc(B * sizeof(int32_t));
                g.dspark_hc_pre = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_HC * sizeof(float));
                g.dspark_hc_w = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_HC * sizeof(float));
                g.dspark_flat = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_EMBD * sizeof(float));
                g.dspark_flat_norm = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_EMBD * sizeof(float));
                g.dspark_logits = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_VOCAB * sizeof(float));
                g.dspark_prev_id = ds4_gpu_tensor_alloc(sizeof(int32_t));
                g.dspark_out_id = ds4_gpu_tensor_alloc(sizeof(int32_t));
            }
        } else {
            fprintf(stderr, "ds4: [dspark-probe] drafter not armed, probe off\n");
        }
    }
    int pos = prompt->len;
    int n_generated = 0;
    int n_decode_eval = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;
        if (dspark_probe && dspark_pending_n > 0) {
            dspark_tot++;
            if (dspark_pending[0] == token) dspark_hit++;
            dspark_pending_n = 0;
        }

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = metal_graph_eval_token_raw_swa(&g,
                                            model,
                                            weights,
                                            (uint32_t)token,
                                            (uint32_t)pos,
                                            logits);
        if (!ok) break;
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: gpu decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
        if (dspark_probe && g_dsw.ready) {
            int ids[DS4_DSPARK_BLK] = {0};
            if (metal_graph_dspark_step(&g, model, weights, &g_dsw, token, (uint32_t)(pos - 1), ids)) {
                memcpy(dspark_pending, ids, sizeof(int) * DS4_DSPARK_BLK);
                dspark_pending_n = DS4_DSPARK_BLK;
                if (n_decode_eval <= 8)
                    fprintf(stderr, "ds4: [dspark] pos=%d anchor=%d draft= %d %d %d %d %d\n",
                            pos - 1, token, ids[0], ids[1], ids[2], ids[3], ids[4]);
            } else if (n_decode_eval <= 3) {
                fprintf(stderr, "ds4: [dspark] step failed at pos=%d\n", pos - 1);
            }
        }
    }
    if (dspark_probe && dspark_tot)
        fprintf(stderr, "ds4: [dspark] first-token hit %u/%u = %.1f%%\n",
                dspark_hit, dspark_tot, 100.0 * dspark_hit / dspark_tot);
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before graph free");
    free(logits);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}
#endif

#ifdef DS4_NO_GPU
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    (void)backend;
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    m.raw_cap = ds4_default_raw_cap(ctx);
    m.raw_bytes = (uint64_t)DS4_N_LAYER *
                  m.raw_cap *
                  DS4_N_HEAD_DIM *
                  sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t comp_cap = ctx / ratio + 2u;
        if (ratio == 4) m.comp_cap = comp_cap;
        m.compressed_bytes += (uint64_t)comp_cap *
                              DS4_N_HEAD_DIM *
                              sizeof(float);
        if (ratio == 4) {
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_INDEXER_HEAD_DIM *
                                  sizeof(float);
        }
    }
    if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
    m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(bool));
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}
#endif
