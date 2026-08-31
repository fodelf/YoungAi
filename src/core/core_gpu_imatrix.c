/* core_gpu_imatrix.c — eval_token + imatrix 采集 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = throttle ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    /* decode 单 token CUDA graph: capture 包住 encode(纯 kernel 段), 失败则重编码直跑。
     * 流水线(2026-08-17): 上一 token 的 GPU 窗口里已预编码本 pos 的图 ⇒ 直接发射,
     * encode 移出临界路径; 发射后趁 GPU 忙再预编码 pos+1(token id 走参数槽间接)。 */
    int launched = 0;
    if (ok && ds4_gpu_token_graph_try_pending(token, pos, logits != NULL) > 0)
        launched = 1;
    ds4_gpu_token_graph_set_pos(pos);
    const int tok_graph = (ok && !launched) ? ds4_gpu_token_graph_begin() : 0;
    if (ok && !launched) ok = metal_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    if (ok && !launched && tok_graph) {
        if (ds4_gpu_token_graph_end_launch() < 0)
            ok = metal_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    } else if (!ok && tok_graph) {
        /* encode 在 capture 里失败(如 comp cache 溢出): 必须收掉悬挂 capture,
         * 否则后续所有 launch 报 "previous error during capture" 永久污染(2026-08-18
         * 328 题基准 500 连锁的第二层根因)。 */
        (void)ds4_gpu_token_graph_end_launch();
    }
    if (ok && (launched || tok_graph)) {
        if (ds4_gpu_token_graph_precapture_begin() > 0) {
            const bool pok = metal_graph_encode_token_raw_swa(g, model, weights, 0, pos + 1u, true, true);
            (void)ds4_gpu_token_graph_precapture_end(pos + 1u, 1, pok ? 1 : 0);
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = throttle ? now_sec() : 0.0;
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

/* =========================================================================
 * Imatrix Collection.
 * =========================================================================
 *
 * The 2-bit DS4 quants care most about routed MoE experts.  For expert gate
 * and up matrices the matmul input is the FFN-normalized activation row.  For
 * expert down matrices the matmul input is the routed SwiGLU row after route
 * weighting.  During Metal prefill those tensors are already materialized as
 * `batch_ffn_norm`, `batch_router_selected`, and `batch_routed_mid`, so the
 * collector observes the exact release graph without changing inference math.
 *
 * The output is llama.cpp's legacy imatrix `.dat` format.  Entries are packed
 * by expert: one tensor entry contains `n_expert * n_columns` floats and the
 * quantizer slices the vector for each expert.
 */

bool imatrix_collector_init(ds4_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->cap_tokens = cap_tokens ? cap_tokens : 1u;
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_EMBD;
    const size_t down_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_FF_EXP;
    c->gate_up_sum2 = xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->ffn_norm_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_buf[0]));
    c->routed_mid_f16_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_f16_buf[0]));
    c->selected_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * sizeof(c->selected_buf[0]));
    c->sq_tmp = xmalloc((size_t)DS4_N_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->ffn_norm_buf &&
           c->routed_mid_buf && c->routed_mid_f16_buf && c->selected_buf && c->sq_tmp;
}

void imatrix_collector_free(ds4_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->routed_mid_f16_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}

static float *imatrix_gate_up_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_EMBD;
}

static float *imatrix_down_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->down_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_FF_EXP;
}

bool imatrix_collect_layer_batch(
        ds4_imatrix_collector *c,
        ds4_gpu_graph         *g,
        uint32_t               il,
        uint32_t               n_tokens) {
    if (!c || n_tokens == 0) return true;
    if (n_tokens > c->cap_tokens) return false;

    const uint64_t norm_bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * (g->batch_routed_mid_is_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t sel_bytes = (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int);
    void *mid_dst = g->batch_routed_mid_is_f16
        ? (void *)c->routed_mid_f16_buf
        : (void *)c->routed_mid_buf;
    if (ds4_gpu_tensor_read(g->batch_ffn_norm, 0, c->ffn_norm_buf, norm_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_routed_mid, 0, mid_dst, mid_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_router_selected, 0, c->selected_buf, sel_bytes) == 0)
    {
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * DS4_N_EMBD;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const int expert = c->selected_buf[(size_t)t * DS4_N_EXPERT_USED + slot];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) continue;

            float *gate_up = imatrix_gate_up_ptr(c, il, (uint32_t)expert);
            for (uint32_t i = 0; i < DS4_N_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            c->gate_up_count[il][expert]++;

            float *down = imatrix_down_ptr(c, il, (uint32_t)expert);
            const size_t mid_off = ((size_t)t * DS4_N_EXPERT_USED + slot) * DS4_N_FF_EXP;
            if (g->batch_routed_mid_is_f16) {
                const uint16_t *mid = c->routed_mid_f16_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) {
                    const float v = f16_to_f32(mid[i]);
                    down[i] += v * v;
                }
            } else {
                const float *mid = c->routed_mid_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) down[i] += mid[i] * mid[i];
            }
            c->down_count[il][expert]++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tokens;
    c->chunks++;
    return true;
}

static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) ds4_die("failed to write imatrix");
}

static void imatrix_write_entry(
        FILE       *fp,
        const char *name,
        const float *sum2,
        const uint32_t *counts,
        uint32_t n_expert,
        uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) ds4_die("failed to write imatrix name");
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) ds4_die("failed to write imatrix values");
    }
    free(tmp);
}

bool imatrix_collector_save(
        const ds4_imatrix_collector *c,
        const ds4_weights           *weights,
        const char                  *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }

    const int32_t entries = (int32_t)(DS4_N_LAYER * 3);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->ffn_gate_exps || !layer->ffn_up_exps) continue;   /* 内嵌 VQ: 无 gate/up 名可写 */
        char name[256];
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_gate_exps->name.len, layer->ffn_gate_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_up_exps->name.len, layer->ffn_up_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_down_exps->name.len, layer->ffn_down_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->down_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_FF_EXP,
                            c->down_count[il],
                            DS4_N_EXPERT,
                            DS4_N_FF_EXP);
    }

    const int32_t chunks = (int32_t)c->chunks;
    imatrix_write_i32(fp, chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len && fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        ds4_die("failed to write imatrix dataset name");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_imatrix_nonempty_tu; /* 空TU防御(CPU构建) */
