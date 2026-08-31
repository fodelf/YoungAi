/* core_gpu_prefill.c — prefill layer-major 主循环 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
bool metal_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->mtp_n_raw = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!metal_graph_layer_is_active(g, il)) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ds4_comp_row_slots(ratio);
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}

/* Execute Metal prefill in layer-major order so intermediate activations stay
 * on the GPU and cache state is built exactly once. */
static void metal_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                start,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total) {
    if (!display_progress) return;
    if (layer_done > (uint32_t)DS4_N_LAYER) layer_done = (uint32_t)DS4_N_LAYER;
    uint64_t done = (uint64_t)n_tokens * layer_done / (uint32_t)DS4_N_LAYER;
    if (layer_done == (uint32_t)DS4_N_LAYER) done = n_tokens;
    display_progress(display_progress_ud, "prefill_display",
                     (int)(start + (uint32_t)done), total);
}

bool metal_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (!ok) return false;

    if (!metal_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) return false;

    /*
     * A full long-prompt prefill can keep the GPU busy long enough for macOS
     * to watchdog WindowServer. Also split non-tiny prefills when a frontend
     * asked for display progress: completed layer command buffers are real
     * scheduling/keepalive points, while callbacks emitted while encoding one
     * huge command buffer would only be cosmetic.
     */
    const bool throttle = graph_power_throttle_enabled(g);
    const bool callback_split = display_progress != NULL && n_tokens >= 32;
    const bool split_commands = throttle || callback_split ||
                                n_tokens > 2048 || imatrix != NULL;

    if (!split_commands) {
        ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     model,
                                                     weights,
                                                     prompt,
                                                     start,
                                                     n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                start,
                                                n_tokens);
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);
        if (display_progress)
            display_progress(display_progress_ud, "prefill_display",
                             (int)(start + n_tokens), prompt->len);

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        const uint32_t output_row = (uint32_t)n_tokens - 1u;
        ds4_gpu_tensor *saved_cur = g->cur_hc;
        ds4_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = metal_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc = last_hc;
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc = saved_cur;
        }

        if (ok) ok = ds4_gpu_end_commands() != 0;
        g->cur_hc = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        if (logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        return ok;
    }

    ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const double t_chunk0 = throttle ? now_sec() : 0.0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_layer_batch(g,
                                                    model,
                                                    &weights->layer[il],
                                                    il,
                                                    start,
                                                    n_tokens);
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = throttle ? now_sec() : 0.0;
        if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
        const double layer_elapsed = t_done - t_chunk0;
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        graph_power_note_prefill_layer(g, il, layer_elapsed);
        metal_graph_report_prefill_display_progress(display_progress,
                                                    display_progress_ud,
                                                    start,
                                                    n_tokens,
                                                    il + 1,
                                                    prompt->len);
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t output_row = (uint32_t)n_tokens - 1u;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *last_hc = NULL;

    if (logits) {
        last_hc = metal_graph_tensor_row_view(g->batch_cur_hc,
                                              output_row,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    return ok;
}

bool metal_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    return metal_graph_prefill_layer_major(g,
                                           model,
                                           weights,
                                           prompt,
                                           0,
                                           (uint32_t)n_tokens,
                                           logits,
                                           show_progress,
                                           NULL,
                                           display_progress,
                                           display_progress_ud);
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_prefill_nonempty_tu; /* 空TU防御(CPU构建) */
