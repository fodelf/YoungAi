/* core_engine_open.c — 拆分中间态: 批次尚未切出的切片仍并在本文件(自 ds4.c). */
#include "core_internal.h"
#ifndef DS4_NO_GPU
/* ---- engine-trajectory batch capture (DS4_CAP_DIR) ------------------------
 * Appends, per routed layer and per prefill chunk, the tensors the offline
 * error-feedback calibration needs, as raw little-endian shards:
 *   raw_ffn_in_L{L}.f16        x̂ = post-RMSNorm expert input   [n×4096]
 *   raw_route_L{L}.i16         selected expert ids (pre-remap) [n×6]
 *   raw_route_logits_L{L}.f16  RAW router logits (pre-δ)       [n×256]
 *   raw_route_w_L{L}.f16       applied gate weights            [n×6]
 * Token count = file bytes / (width × elem size); cap_raw2npy.py converts to
 * the cap npy schema. DS4_CAP_LAYERS="lo-hi" filters layers (default all). */
static int cap_layer_enabled(uint32_t il) {
    const char *r = getenv("DS4_CAP_LAYERS");
    if (!r || !r[0]) return 1;
    unsigned lo = 0, hi = DS4_MAX_LAYER;
    if (sscanf(r, "%u-%u", &lo, &hi) != 2) return 1;
    return il >= lo && il <= hi;
}

/* handle cache: the decode path appends per TOKEN per layer — reopening per
 * append would be ~1M syscalls per full-corpus capture. One append handle per
 * (layer, kind), opened lazily, flushed by exit / the libc atexit machinery
 * (single ds4 instance; capture is a calibration-run-only mode). */
static FILE *cap_handle(const char *dir, const char *name, uint32_t il, int kind) {
    static FILE *cache[DS4_MAX_LAYER][5];
    if (il >= DS4_MAX_LAYER || kind < 0 || kind > 4) return NULL;
    if (!cache[il][kind]) {
        char p[1024];
        snprintf(p, sizeof p, "%s/%s_L%u", dir, name, il);
        cache[il][kind] = fopen(p, "ab");
    }
    return cache[il][kind];
}

static void cap_append_k(const char *dir, const char *name, uint32_t il, int kind,
                         const void *buf, size_t bytes) {
    FILE *f = cap_handle(dir, name, il, kind);
    if (!f) return;
    fwrite(buf, 1, bytes, f);
    /* flush per append: capture workers get SIGTERM-harvested (dist capture
     * pipeline), and a signal death skips atexit — an unflushed stdio tail
     * desyncs the five per-layer shards' row counts (12B/row route hurts most).
     * Cost is noise next to the GPU readbacks that precede every append. */
    fflush(f);
}
#define cap_append(dir, name, il, buf, bytes) cap_append_k(dir, name, il, \
    (strcmp(name, "raw_ffn_in") == 0 ? 0 : strcmp(name, "raw_route_logits") == 0 ? 1 : \
     strcmp(name, "raw_route_w") == 0 ? 2 : strcmp(name, "raw_ffn_out") == 0 ? 4 : 3), buf, bytes)

void cap_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens) {
    const char *dir = getenv("DS4_CAP_DIR");
    if (!dir || !dir[0] || n_tokens == 0 || !cap_layer_enabled(il)) return;

    /* ★捕获前 GPU 定格(2026-07-22 根因修复)★: batch_routed_out 由仍在队列里的
     * 本层 MoE 核写入 — 不 drain 就 tensor_read 会拿到上一层残值(首捕获层=全零,
     * 之后逐层错位一格; off-by-one 实锤: 错位对齐后 cos(O_BASE,O_REF)=0.9995)。
     * ffn_norm 恰因更早同步点已定格, 掩盖了此病 — dsml 时代 P2 侧车 NO-GO 同根因。
     * 用 TP 块同款 signal→flush→host_wait 快路径(MTLSharedEvent), 只在捕获时付。 */
    {
        const uint64_t cap_ev = ds4_gpu_tp_signal_after_batch();
        if (cap_ev) { (void)ds4_gpu_flush_commands(); (void)ds4_gpu_tp_host_wait(cap_ev); }
    }

    const uint32_t d = DS4_N_EMBD, ne = DS4_N_EXPERT, ku = DS4_N_EXPERT_USED;
    size_t nf = (size_t)n_tokens * d;
    float *fb = malloc(nf * sizeof(float));
    uint16_t *hb = malloc(nf * sizeof(uint16_t));
    if (!fb || !hb) { free(fb); free(hb); return; }

    if (ds4_gpu_tensor_read(g->batch_ffn_norm, 0, fb, nf * sizeof(float))) {
        for (size_t i = 0; i < nf; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_ffn_in", il, hb, nf * sizeof(uint16_t));
    }
    /* P2 侧车管线需要 O_BASE: routed-MoE 输出, 捕获点在 corr 应用之前(调用序
     * 见 cap_batch_layer 调用处注释), 与 raw_ffn_in 同 token 对齐。f16 存储。 */
    if (ds4_gpu_tensor_read(g->batch_routed_out, 0, fb, nf * sizeof(float))) {
        for (size_t i = 0; i < nf; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_ffn_out", il, hb, nf * sizeof(uint16_t));
    }
    size_t nl = (size_t)n_tokens * ne;
    if (ds4_gpu_tensor_read(g->batch_router_logits, 0, fb, nl * sizeof(float))) {
        for (size_t i = 0; i < nl; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_route_logits", il, hb, nl * sizeof(uint16_t));
    }
    size_t nw = (size_t)n_tokens * ku;
    if (ds4_gpu_tensor_read(g->batch_router_weights, 0, fb, nw * sizeof(float))) {
        for (size_t i = 0; i < nw; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_route_w", il, hb, nw * sizeof(uint16_t));
    }
    /* pre-remap ids: same snapshot the corr dispatch consumes (go1b batch MoE
     * rewrites the live tensor to compact slots in place) */
    const ds4_gpu_tensor *sel = ds4_gpu_corr_saved_selected();
    if (!sel) sel = g->batch_router_selected;
    int32_t *ib = (int32_t *)fb;   /* reuse: n×6 i32 fits in the f32 buffer */
    if (ds4_gpu_tensor_read(sel, 0, ib, nw * sizeof(int32_t))) {
        int16_t *sb = (int16_t *)hb;
        for (size_t i = 0; i < nw; i++) sb[i] = (int16_t)ib[i];
        cap_append(dir, "raw_route", il, sb, nw * sizeof(int16_t));
    }
    free(fb); free(hb);
    /* observability iron rule: unbuffered progress on stderr, rate-limited so
     * the decode path (1 token per call) doesn't flood — never a black box. */
    static uint64_t cap_tok_total;
    cap_tok_total += n_tokens;
    if (n_tokens > 1 || (cap_tok_total % 256) == 0)
        fprintf(stderr, "ds4: [cap] L%u +%u (total %llu tok-layers)\n",
                il, n_tokens, (unsigned long long)cap_tok_total);
}

/* ---- 反修百分百还原判决钩(DS4_AMP_ANCHOR, 2026-08-19) ----------------------
 * 判决实验专用, 不进产线: 把反修解算时的输入条件在线还原 — zchain(放大器)的 x 用
 * 解算锚(DQA2)的 FP fin; DS4_AMP_ANCHOR_ROUTE=1 时路由(selected/weights)也钉锚
 * ridx/rw。用于把"解算产物/引擎应用有病"与"解算口径 vs 在线口径漂移"分开归因。
 * 只挂 decode 路(score 链除首 token 外全走这里; 首 token 批路不钉, 偏差 1/S 在案)。 */
ds4_ampanc_state g_ampanc;

int ampanc_on(void) {
    if (g_ampanc.state) return g_ampanc.state > 0;
    g_ampanc.state = -1;
    const char *p = getenv("DS4_AMP_ANCHOR");
    if (!p || !p[0]) return 0;
    int fd = open(p, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "ds4: AMP_ANCHOR %s: open failed\n", p); return 0; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { fprintf(stderr, "ds4: AMP_ANCHOR mmap failed\n"); return 0; }
    const uint32_t *hd = (const uint32_t *)m;
    if (hd[0] != 0x32415144u) { fprintf(stderr, "ds4: AMP_ANCHOR bad magic\n"); return 0; }
    g_ampanc.S = hd[1]; g_ampanc.dim = hd[3]; g_ampanc.nl = hd[4]; g_ampanc.nact = hd[6];
    if (g_ampanc.dim != DS4_N_EMBD || g_ampanc.nact != DS4_N_EXPERT_USED ||
        (uint64_t)st.st_size < 40u + (uint64_t)g_ampanc.nl * g_ampanc.S *
            (4ull * g_ampanc.dim + 8ull * g_ampanc.nact)) {
        fprintf(stderr, "ds4: AMP_ANCHOR 头/长度不合, 拒绝\n");
        return 0;
    }
    const uint8_t *q = (const uint8_t *)m + 40;
    g_ampanc.fin = (const float *)q;
    q += (size_t)g_ampanc.nl * g_ampanc.S * g_ampanc.dim * 4u;
    g_ampanc.ridx = (const int32_t *)q;
    q += (size_t)g_ampanc.nl * g_ampanc.S * g_ampanc.nact * 4u;
    g_ampanc.rw = (const float *)q;
    g_ampanc.route_on = getenv("DS4_AMP_ANCHOR_ROUTE") != NULL;
    g_ampanc.state = 1;
    fprintf(stderr, "ds4: AMP_ANCHOR armed: S=%u NL=%u x=钉锚 route=%s (判决钩)\n",
            g_ampanc.S, g_ampanc.nl, g_ampanc.route_on ? "钉锚" : "在线");
    return 1;
}

/* decode-path capture: same shards, one token per call. The perplexity scorer
 * (and any decode) runs token-by-token through here — this is where the bulk
 * of a teacher-forced trajectory capture actually flows (the batch hook only
 * sees the 32-token seed prefix). */
void cap_decode_layer(ds4_gpu_graph *g, uint32_t il) {
    const char *dir = getenv("DS4_CAP_DIR");
    if (!dir || !dir[0] || !cap_layer_enabled(il)) return;

    const uint32_t d = DS4_N_EMBD, ne = DS4_N_EXPERT, ku = DS4_N_EXPERT_USED;
    float fb[DS4_N_EXPERT > 4096 ? DS4_N_EXPERT : 4096];
    uint16_t hb[DS4_N_EXPERT > 4096 ? DS4_N_EXPERT : 4096];

    if (ds4_gpu_tensor_read(g->ffn_norm, 0, fb, (size_t)d * sizeof(float))) {
        for (uint32_t i = 0; i < d; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_ffn_in", il, hb, (size_t)d * sizeof(uint16_t));
    }
    /* P2: O_BASE (decode 路径逐 token), 对齐 raw_ffn_in。 */
    if (ds4_gpu_tensor_read(g->routed_out, 0, fb, (size_t)d * sizeof(float))) {
        for (uint32_t i2 = 0; i2 < d; i2++) hb[i2] = f32_to_f16(fb[i2]);
        cap_append(dir, "raw_ffn_out", il, hb, (size_t)d * sizeof(uint16_t));
    }
    if (ds4_gpu_tensor_read(g->router_logits, 0, fb, (size_t)ne * sizeof(float))) {
        for (uint32_t i = 0; i < ne; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_route_logits", il, hb, (size_t)ne * sizeof(uint16_t));
    }
    if (ds4_gpu_tensor_read(g->router_weights, 0, fb, (size_t)ku * sizeof(float))) {
        for (uint32_t i = 0; i < ku; i++) hb[i] = f32_to_f16(fb[i]);
        cap_append(dir, "raw_route_w", il, hb, (size_t)ku * sizeof(uint16_t));
    }
    const ds4_gpu_tensor *sel = ds4_gpu_corr_saved_selected();
    if (!sel) sel = g->router_selected;
    int32_t ib[DS4_N_EXPERT_USED];
    if (ds4_gpu_tensor_read(sel, 0, ib, (size_t)ku * sizeof(int32_t))) {
        int16_t sb[DS4_N_EXPERT_USED];
        for (uint32_t i = 0; i < ku; i++) sb[i] = (int16_t)ib[i];
        cap_append(dir, "raw_route", il, sb, (size_t)ku * sizeof(int16_t));
    }
    static uint64_t cap_tok_total2;
    if ((++cap_tok_total2 % (256 * 23)) == 0)
        fprintf(stderr, "ds4: [cap] decode total %llu tok-layers\n",
                (unsigned long long)cap_tok_total2);
}

bool graph_power_throttle_enabled(const ds4_gpu_graph *g) {
    return g && g->power_percent > 0 && g->power_percent < 100;
}

static double graph_power_update_avg(double avg, double sample) {
    if (sample <= 0.0 || !isfinite(sample)) return avg;
    if (avg <= 0.0 || !isfinite(avg)) return sample;
    return avg * 0.875 + sample * 0.125;
}

static void graph_power_sleep(double work_sec, uint32_t power_percent) {
    if (power_percent == 0 || power_percent >= 100) return;
    /* Target duty cycle: work / (work + sleep) = power / 100.
     * At --power 50 this sleeps for one measured work interval; at 25 it
     * sleeps for three. */
    const double sleep = work_sec * (100.0 - (double)power_percent) /
                         (double)power_percent;
    sleep_sec(sleep);
}

void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    if (il >= DS4_N_LAYER) return;
    g->prefill_layer_avg_sec[il] =
        graph_power_update_avg(g->prefill_layer_avg_sec[il], elapsed_sec);
    graph_power_sleep(g->prefill_layer_avg_sec[il], g->power_percent);
}

void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    g->decode_token_avg_sec =
        graph_power_update_avg(g->decode_token_avg_sec, elapsed_sec);
    graph_power_sleep(g->decode_token_avg_sec, g->power_percent);
}

/* Release every Metal tensor owned by the whole-model graph runtime. */
#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU
void metal_graph_free(ds4_gpu_graph *g) {
    free(g->tp_vec); /* TP host staging buffer (the tp socket is owned by the engine) */
    ds4_gpu_tensor_free(g->directional_steering_dirs);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_routed_out);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_mid);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_shared_out);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_shared_up);
    ds4_gpu_tensor_free(g->batch_shared_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_ffn_cur);
    ds4_gpu_tensor_free(g->batch_after_attn_hc);
    ds4_gpu_tensor_free(g->batch_low_tmp);
    ds4_gpu_tensor_free(g->batch_group_tmp);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_attn_low);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_comp_sc);
    ds4_gpu_tensor_free(g->batch_comp_kv);
    ds4_gpu_tensor_free(g->batch_kv);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_qr_norm);
    ds4_gpu_tensor_free(g->batch_qr);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_attn_cur);
    ds4_gpu_tensor_free(g->batch_hc_split);
    ds4_gpu_tensor_free(g->batch_hc_mix);
    ds4_gpu_tensor_free(g->batch_flat_hc);
    ds4_gpu_tensor_free(g->batch_next_hc);
    ds4_gpu_tensor_free(g->batch_cur_hc);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->mtp_raw_cache);
    ds4_gpu_tensor_free(g->mtp_next_hc);
    ds4_gpu_tensor_free(g->mtp_state_hc);
    ds4_gpu_tensor_free(g->mtp_input_hc);
    ds4_gpu_tensor_free(g->mtp_hproj_hc);
    ds4_gpu_tensor_free(g->mtp_hnorm_hc);
    ds4_gpu_tensor_free(g->mtp_eproj_hc);
    ds4_gpu_tensor_free(g->mtp_eproj);
    ds4_gpu_tensor_free(g->mtp_enorm);
    ds4_gpu_tensor_free(g->dspark_main_hidden);
    ds4_gpu_tensor_free(g->dspark_main_x_raw);
    ds4_gpu_tensor_free(g->dspark_main_x);
    ds4_gpu_tensor_free(g->dspark_kv_tmp);
    for (int b = 0; b < 3; b++) ds4_gpu_tensor_free(g->dspark_win_kv[b]);
    ds4_gpu_tensor_free(g->dspark_ids);
    ds4_gpu_tensor_free(g->dspark_hc_pre);
    ds4_gpu_tensor_free(g->dspark_hc_w);
    ds4_gpu_tensor_free(g->dspark_flat);
    ds4_gpu_tensor_free(g->dspark_flat_norm);
    ds4_gpu_tensor_free(g->dspark_logits);
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) ds4_gpu_tensor_free(g->spec_raw_save[il]);
    for (uint32_t b = 0; b < 3u; b++) ds4_gpu_tensor_free(g->dspark_spec_kv[b]);
    ds4_gpu_tensor_free(g->dspark_conf);
    ds4_gpu_tensor_free(g->dspark_prev_ids);
    ds4_gpu_tensor_free(g->dspark_prev_id);
    ds4_gpu_tensor_free(g->dspark_out_id);
    ds4_gpu_tensor_free(g->dspark_pf_hidden);
    ds4_gpu_tensor_free(g->dspark_pf_x);
    ds4_gpu_tensor_free(g->dspark_pf_kv);
    ds4_gpu_tensor_free(g->mtp_embed);
    ds4_gpu_tensor_free(g->spec_logits);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->output_embd);
    ds4_gpu_tensor_free(g->output_weights);
    ds4_gpu_tensor_free(g->output_pre);
    ds4_gpu_tensor_free(g->after_ffn_hc);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_out);
    ds4_gpu_tensor_free(g->corr_delta);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_mid);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->shared_out);
    ds4_gpu_tensor_free(g->shared_mid);
    ds4_gpu_tensor_free(g->shared_up);
    ds4_gpu_tensor_free(g->shared_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->ffn_cur);
    ds4_gpu_tensor_free(g->after_attn_hc);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->attn_low);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->comp_sc_cur);
    ds4_gpu_tensor_free(g->comp_kv_cur);
    ds4_gpu_tensor_free(g->comp_sc_side);
    ds4_gpu_tensor_free(g->comp_kv_side);
    ds4_gpu_tensor_free(g->attn_comp_stage);
    ds4_gpu_tensor_free(g->comp_mask);
    ds4_gpu_tensor_free(g->comp_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_comp_rows_kv[il]);
        ds4_gpu_tensor_free(g->spec_comp_rows_sc[il]);
        ds4_gpu_tensor_free(g->spec_idx_rows_kv[il]);
        ds4_gpu_tensor_free(g->spec_idx_rows_sc[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_score[il]);
    }
    ds4_gpu_tensor_free(g->kv);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->qr_norm);
    ds4_gpu_tensor_free(g->qr);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->attn_cur);
    ds4_gpu_tensor_free(g->hc_comb);
    ds4_gpu_tensor_free(g->hc_post);
    ds4_gpu_tensor_free(g->hc_pre);
    ds4_gpu_tensor_free(g->hc_split);
    ds4_gpu_tensor_free(g->hc_mix);
    ds4_gpu_tensor_free(g->flat_hc);
    ds4_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}

bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the Metal graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

bool metal_graph_load_directional_steering(
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    float *dirs = xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        g->directional_steering_dirs = ds4_gpu_tensor_alloc(n * sizeof(dirs[0]));
        ok = g->directional_steering_dirs != NULL &&
             ds4_gpu_tensor_write(g->directional_steering_dirs, 0, dirs, n * sizeof(dirs[0])) != 0;
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "ds4: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_attn_scale != 0.0f;
}

bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_ffn_scale != 0.0f;
}

static bool metal_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !g->directional_steering_dirs || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            g->directional_steering_dirs,
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}

bool metal_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}

bool metal_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}

static uint64_t metal_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    uint64_t bytes = (uint64_t)DS4_N_LAYER *
                     raw_cap *
                     DS4_N_HEAD_DIM *
                     sizeof(float);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * DS4_N_HEAD_DIM *
                 (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
        if (ratio == 4) {
            bytes += comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    }
    return bytes;
}

uint64_t metal_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = metal_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    uint64_t bytes = kv_cache_bytes +
                     2ull * comp_cap * prefill_cap * sizeof(float);
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        uint64_t attn_stage_cap = (uint64_t)(prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        bytes += attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    }
    return bytes;
}

ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? ds4_gpu_tensor_alloc_managed(bytes) : ds4_gpu_tensor_alloc(bytes);
}
#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU

/* =========================================================================
 * Metal Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_METAL_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */


static bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!prefix || !prefix[0]) return false;

    const char *name_env = getenv("DS4_METAL_GRAPH_DUMP_NAME");
    if (name_env && name_env[0] && strstr(name_env, name) == NULL) return false;

    const char *layer_env = getenv("DS4_METAL_GRAPH_DUMP_LAYER");
    if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0 &&
        (uint32_t)strtoul(layer_env, NULL, 10) != il) return false;

    const char *pos_env = getenv("DS4_METAL_GRAPH_DUMP_POS");
    if (pos_env && pos_env[0] && (uint32_t)strtoul(pos_env, NULL, 10) != pos) return false;

    return true;
}

/* 同跑双路对账标签(2026-08-21): 投机 XCHECK 下同一位置会被批路和解码路各算一遍,
 * 文件名带标签才不会互相覆盖。空标签 = 老行为。 */
const char *g_dump_tag = "";

void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s%s_%s-%u_pos%u.bin", prefix, g_dump_tag, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

/* DS4_TRACE_HNORM: 逐层打 ‖h‖ / ‖routed‖ / ‖shexp‖(decode 单 token)。
 * 这是 "每层数学都对" 与 "模型输出" 之间那段的探针 —— 各层数值正确并不能保证整合
 * 正确: routed 没被加进残差流, 每层的 ‖routed‖ 照样漂亮, 但 ‖h‖ 会一路不变。轨迹形态
 * 直接给出病灶层: 突然爆炸 / 塌缩 / 不变(不变 = routed 没进 h)。
 * 代价是每层一次 GPU 同步, 只在开了 env 时才走。 */
static int trace_hnorm_enabled(void) {
    static int v = -1;
    if (v < 0) {
        /* 与模型无关: 这条追踪挂在 decode 层前向的唯一出口上, 对任何路由量化类型
         * 一视同仁。启用时打一条横幅 —— 没看到横幅就说明跑的是旧二进制, 而不是
         * "这条路没走到", 双机对照时这两种情况必须能一眼分开。 */
        v = getenv("DS4_TRACE_HNORM") ? 1 : 0;
        if (v) fprintf(stderr, "ds4: [HNORM] 逐层追踪已启用(与模型无关, decode 路每层一行)\n");
    }
    return v;
}
static double trace_l2_of(ds4_gpu_tensor *t, uint64_t n_f32) {
    if (!t || !n_f32) return -1.0;
    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    double acc = -1.0;
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        acc = 0.0;
        for (uint64_t i = 0; i < n_f32; i++) acc += (double)buf[i] * (double)buf[i];
        acc = sqrt(acc);
    }
    free(buf);
    return acc;
}
void trace_hnorm_layer(ds4_gpu_graph *g, uint32_t il, uint32_t pos, uint64_t hc_dim) {
    if (!trace_hnorm_enabled()) return;
    if (ds4_gpu_synchronize() == 0) return;
    /* pos 必须打: 这条 trace 只在 decode 路触发, 而 decode 只处理"生成的" token —
     * prompt 那几个位置即使在 ./ds4 -p 下也是走批路的。拿它和 HDUMP 的批文件比对时,
     * 必须挑 HDUMP 里同一个 pos 的那一行, 否则比的是不同 token 的不同位置。
     * ||attn|| 是本层 FFN 之前的 HC 态(embed+attn 之后), 用来把分歧夹在 attn / FFN 之间。 */
    const double an = trace_l2_of(g->after_attn_hc, hc_dim);
    const double hn = trace_l2_of(g->after_ffn_hc, hc_dim);
    const double rn = trace_l2_of(g->routed_out, DS4_N_EMBD);
    const double sn = trace_l2_of(g->shared_out, DS4_N_EMBD);
    fprintf(stderr,
            "[HNORM] L%02u pos=%u ||attn||=%.6g ||routed||=%.6g ||shexp||=%.6g ||h||=%.6g\n",
            il, pos, an, rn, sn, hn);
    (void)ds4_gpu_begin_commands();
}
/* head 之前的最终 ‖h‖ + top-5 logits: 看崩塌形态(单点尖峰 = 塌到某个 token;
 * 全平 = 信息没传上来)。 */
void trace_hnorm_head(ds4_gpu_graph *g, uint32_t vocab_dim) {
    if (!trace_hnorm_enabled()) return;
    if (ds4_gpu_synchronize() == 0) return;
    const double en = trace_l2_of(g->output_embd, DS4_N_EMBD);
    const double nn = trace_l2_of(g->output_norm, DS4_N_EMBD);
    float *lg = xmalloc((size_t)vocab_dim * sizeof(lg[0]));
    if (ds4_gpu_tensor_read(g->logits, 0, lg, (uint64_t)vocab_dim * sizeof(lg[0])) != 0) {
        int top[5] = {0, 0, 0, 0, 0};
        for (int k = 0; k < 5; k++) {
            int best = -1;
            for (uint32_t i = 0; i < vocab_dim; i++) {
                int dup = 0;
                for (int j = 0; j < k; j++) if (top[j] == (int)i) { dup = 1; break; }
                if (dup) continue;
                if (best < 0 || lg[i] > lg[best]) best = (int)i;
            }
            top[k] = best < 0 ? 0 : best;
        }
        fprintf(stderr, "[HNORM] HEAD ||embd||=%.6g ||norm||=%.6g top5:", en, nn);
        for (int k = 0; k < 5; k++) fprintf(stderr, " %d=%.4f", top[k], lg[top[k]]);
        fprintf(stderr, "\n");
    }
    free(lg);
    (void)ds4_gpu_begin_commands();
}

void metal_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_f16 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    uint16_t *hbuf = xmalloc((size_t)n_f16 * sizeof(hbuf[0]));
    float *fbuf = xmalloc((size_t)n_f16 * sizeof(fbuf[0]));
    if (ds4_gpu_tensor_read(t, 0, hbuf, n_f16 * sizeof(hbuf[0])) != 0) {
        for (uint64_t i = 0; i < n_f16; i++) fbuf[i] = f16_to_f32(hbuf[i]);
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, fbuf, n_f16)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(fbuf);
    free(hbuf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

void metal_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_i32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s%s_%s-%u_pos%u.i32", prefix, g_dump_tag, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

/* Router-frequency collector (DS4_ROUTER_FREQ_FILE). Tally, per layer, which
 * experts the model's *intact* full-256 router would pick (raw top-k over the
 * raw logits, BEFORE the keep-map mask). Lets us rebuild the keep-map mask from
 * what THIS base model actually routes to for the calibration corpus, instead of
 * a chat-derived specialty ranking. Env-gated -> zero cost when off; only active
 * during an explicit collection run. Syncs/resumes like the debug dump. */
#define DS4_ROUTER_FREQ_MAXL 64
#define DS4_ROUTER_FREQ_MAXE 512
static uint64_t g_router_freq[DS4_ROUTER_FREQ_MAXL][DS4_ROUTER_FREQ_MAXE];
static int g_router_freq_on = -1;
static const char *g_router_freq_file = NULL;

static void router_freq_flush(void) {
    if (g_router_freq_on != 1 || !g_router_freq_file) return;
    FILE *fp = fopen(g_router_freq_file, "w");
    if (!fp) { fprintf(stderr, "ds4: router-freq: cannot write %s\n", g_router_freq_file); return; }
    const uint32_t ne = DS4_N_EXPERT;
    fprintf(fp, "# layer expert count  (raw top-%u routing freq over %u experts)\n",
            DS4_N_EXPERT_USED, ne);
    for (uint32_t il = 0; il < DS4_ROUTER_FREQ_MAXL; il++)
        for (uint32_t e = 0; e < ne && e < DS4_ROUTER_FREQ_MAXE; e++)
            if (g_router_freq[il][e])
                fprintf(fp, "%u %u %llu\n", il, e, (unsigned long long)g_router_freq[il][e]);
    fclose(fp);
    fprintf(stderr, "ds4: router-freq written to %s\n", g_router_freq_file);
}

/* The distributed worker is shut down with SIGTERM (default = terminate, no
 * atexit). During a collection run install a handler so the worker flushes its
 * freq file before dying. Gated by the freq env -> not installed on normal runs. */
static void router_freq_sigterm(int sig) { (void)sig; exit(0); }

void router_freq_collect(ds4_gpu_tensor *logits_t, uint32_t il, uint32_t n_tokens) {
    if (g_router_freq_on < 0) {
        g_router_freq_file = getenv("DS4_ROUTER_FREQ_FILE");
        g_router_freq_on = (g_router_freq_file && g_router_freq_file[0]) ? 1 : 0;
        if (g_router_freq_on) { atexit(router_freq_flush); signal(SIGTERM, router_freq_sigterm); }
    }
    if (g_router_freq_on != 1 || !logits_t || il >= DS4_ROUTER_FREQ_MAXL) return;
    const uint32_t ne = DS4_N_EXPERT, nu = DS4_N_EXPERT_USED;
    if (ne > DS4_ROUTER_FREQ_MAXE || nu > 16 || n_tokens == 0) return;
    float *buf = xmalloc((size_t)n_tokens * ne * sizeof(float));
    if (ds4_gpu_synchronize() != 0 &&
        ds4_gpu_tensor_read(logits_t, 0, buf, (uint64_t)n_tokens * ne * sizeof(float)) != 0) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float *lg = buf + (size_t)t * ne;
            int used[16];
            for (uint32_t k = 0; k < nu; k++) {
                int best = -1; float bv = -3.0e38f;
                for (uint32_t e = 0; e < ne; e++) {
                    bool taken = false;
                    for (uint32_t j = 0; j < k; j++) if (used[j] == (int)e) { taken = true; break; }
                    if (!taken && lg[e] > bv) { bv = lg[e]; best = (int)e; }
                }
                used[k] = best;
                if (best >= 0) g_router_freq[il][best]++;
            }
        }
    }
    free(buf);
    ds4_gpu_begin_commands();
}

bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return metal_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           metal_graph_debug_wants("ffn_out", il, pos);
}

bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}

bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}
#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU

/* =========================================================================
 * Metal Release Graph Allocation.
 * ========================================================================= */

/* Allocate the Metal graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
bool metal_graph_layer_is_active(const ds4_gpu_graph *g, uint32_t il) {
    if (!g) return il < (uint32_t)DS4_N_LAYER;
    /* The single-block MTP drafter reuses graph compressor/indexer state at
     * synthetic layer index 1.  Keep that small state allocated on a layer-slice
     * worker even when its target-model slice is the tail layers. */
    if (g->mtp_enabled && il == 1u) return true;
    if (!g->active_layer_slice) return il < (uint32_t)DS4_N_LAYER;
    return il >= g->active_layer_start && il <= g->active_layer_end && il < (uint32_t)DS4_N_LAYER;
}

bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp,
        uint32_t                active_layer_start,
        uint32_t                active_layer_end,
        bool                    active_layer_slice) {
    memset(g, 0, sizeof(*g));
    g->dspark_capture = g_dspark_ready_global;
    g->active_layer_slice = active_layer_slice;
    if (active_layer_start >= (uint32_t)DS4_N_LAYER) active_layer_start = 0;
    if (active_layer_end >= (uint32_t)DS4_N_LAYER) active_layer_end = (uint32_t)DS4_N_LAYER - 1u;
    if (active_layer_end < active_layer_start) {
        active_layer_start = 0;
        active_layer_end = (uint32_t)DS4_N_LAYER - 1u;
        g->active_layer_slice = false;
    }
    g->active_layer_start = active_layer_start;
    g->active_layer_end = active_layer_end;
    /* Sharded per-machine slice: callers pass &weights->layer[0] as the
     * representative layer for per-layer dims (compression ratio, attn shape),
     * but a worker slice may not hold layer 0. Fall back to the first layer this
     * slice actually holds. Whole-model GGUFs hold layer 0 -> unchanged. */
    if (weights && (!layer || !layer->attn_norm)) {
        layer = &weights->layer[active_layer_start];
    }
    g->mtp_enabled = enable_mtp;
    /* Single-machine copy-spec reuses the spec frontier snapshot buffers
     * (spec_attn/index_state_{kv,score}), so allocate those when copy-spec is on,
     * not only for MTP. The MTP-only prefix1 buffers stay gated on enable_mtp. */
    const bool enable_spec = enable_mtp || !active_layer_slice;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        g->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
        if (g->attn_comp_stage_cap < 2u) g->attn_comp_stage_cap = 2u;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = routed_expert_mid_dim(layer);
    /* Sharded worker slice (本机-MTP loopback) holds no output head: it returns
     * hidden state and the coordinator runs the output head. weights->output is
     * then NULL, so the per-row logits buffer is unneeded -- guard the deref. */
    const uint64_t vocab_dim = weights->output ? weights->output->dim[1] : 0u;
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        metal_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!metal_graph_layer_is_active(g, il)) continue;
        g->layer_raw_cache[il] = metal_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = metal_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM *
                    (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float)));
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (enable_spec) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (enable_mtp) {
                g->spec_prefix1_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = metal_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (enable_spec) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (enable_mtp) {
                    g->spec_prefix1_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_kv_side = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_side = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        g->attn_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->attn_comp_stage_cap *
                                                  DS4_N_HEAD_DIM * sizeof(float));
    }
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->corr_delta = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc((vocab_dim ? vocab_dim : 1u) * sizeof(float));
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = metal_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->spec_logits = ds4_gpu_tensor_alloc((uint64_t)64 * DS4_N_VOCAB * sizeof(float));
        g->mtp_n_raw = 0;
    }
    /* PC.1 copy speculation (project.md §3.5): the VERIFY batch reads its K
     * logit rows out of spec_logits but needs none of the MTP drafter tensors.
     * The copy drafter is always armed (single-machine and distributed), so the
     * final-layer owner must always be able to serve VERIFY: allocate the
     * buffer unconditionally (~8 MiB). */
    if (!g->spec_logits) {
        g->spec_logits = ds4_gpu_tensor_alloc((uint64_t)64 * DS4_N_VOCAB * sizeof(float));
    }

    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
    g->batch_low_tmp = ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        if (!metal_graph_layer_is_active(g, il)) continue;
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL &&
                               g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL &&
                               g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    (!DS4_GPU_ATTN_COMP_CACHE_F16 || g->attn_comp_stage) &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out && g->corr_delta &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache && g->spec_logits)) &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) metal_graph_free(g);
    if (g->dspark_capture && !g->dspark_main_hidden) {
        /* DSpark drafter 工作 buffers(独立于 MTP 条件; ~2.6MB) */
        const uint32_t B = 5u;
        g->dspark_main_hidden = ds4_gpu_tensor_alloc(3ull * DS4_N_EMBD * sizeof(float));
        g->dspark_main_x_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->dspark_main_x = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->dspark_kv_tmp = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        for (int b = 0; b < 3; b++)
            g->dspark_win_kv[b] = ds4_gpu_tensor_alloc(128ull * DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_ids = ds4_gpu_tensor_alloc(5u * sizeof(int32_t));
        g->dspark_hc_pre = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_HC * sizeof(float));
        g->dspark_hc_w = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_HC * sizeof(float));
        g->dspark_flat = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_EMBD * sizeof(float));
        g->dspark_flat_norm = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_EMBD * sizeof(float));
        g->dspark_logits = ds4_gpu_tensor_alloc((uint64_t)B * DS4_N_VOCAB * sizeof(float));
        for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++)
            g->spec_raw_save[il] = ds4_gpu_tensor_alloc((uint64_t)(DS4_DSPARK_BLK + 1u) * DS4_N_HEAD_DIM * sizeof(float));
        for (uint32_t b = 0; b < 3u; b++)
            g->dspark_spec_kv[b] = ds4_gpu_tensor_alloc((uint64_t)(DS4_DSPARK_BLK + 1u) * DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_conf = ds4_gpu_tensor_alloc((uint64_t)DS4_DSPARK_BLK * sizeof(float));
        g->dspark_prev_ids = ds4_gpu_tensor_alloc((uint64_t)(DS4_DSPARK_BLK + 1u) * sizeof(int32_t));
        g->dspark_prev_id = ds4_gpu_tensor_alloc(sizeof(int32_t));
        g->dspark_out_id = ds4_gpu_tensor_alloc(sizeof(int32_t));
        g->dspark_pf_hidden = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * 3u * DS4_N_EMBD * sizeof(float));
        g->dspark_pf_x = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
        g->dspark_pf_kv = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_HEAD_DIM * sizeof(float));
    }

    return ok;
}

bool metal_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    return metal_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA, 1, false,
                                     0, (uint32_t)DS4_N_LAYER - 1u, false);
}

uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU
bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

uint32_t metal_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g) {
    (void)g;
    static int parsed = -1;
    static uint32_t cached = 0;
    if (parsed < 0) {
        parsed = 0;
        const char *env = getenv("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            while (end && isspace((unsigned char)*end)) end++;
            if (end != env && end && *end == '\0' &&
                (v == 64ul || v == 128ul || v == 256ul || v == 512ul ||
                 v == 1024ul || v == 2048ul || v == 4096ul)) {
                cached = (uint32_t)v;
                parsed = 1;
            } else {
                fprintf(stderr,
                        "ds4: invalid DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=%s; "
                        "expected 64, 128, 256, 512, 1024, 2048, or 4096\n",
                        env);
            }
        }
    }
    if (parsed > 0) return cached;

    /* Keep dense attention longer than the legacy 512-row window by default.
     * Around the 2K frontier the sparse path's score/top-k setup dominates
     * the smaller attention scan, while larger contexts benefit from sparse
     * indexed attention.  This threshold changes only the implementation used
     * to consume the compressed rows; it must not lower the 512-row indexer
     * selection defined by DS4_N_INDEXER_TOP_K. */
    return 1024u;
}

/* =========================================================================
 * Metal Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_METAL_DISABLE_*_FUSION environment switches.
 */

static bool metal_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

bool metal_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_FUSION", &cache);
}

bool metal_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_KV_FUSION", &cache);
}

bool metal_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_NORM_FUSION", &cache);
}

bool metal_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

bool metal_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_NORM_FUSION", &cache);
}

bool metal_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

bool metal_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

bool metal_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (metal_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

bool metal_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (metal_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

static uint64_t metal_graph_attn_comp_cache_row_bytes(void) {
    return (uint64_t)DS4_N_HEAD_DIM *
           (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
}

uint32_t metal_graph_attn_comp_cache_is_f16(void) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 1u : 0u;
}

static bool metal_graph_store_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g || il >= DS4_N_LAYER) return false;
    if (rows == 0) return true;
    if (!g->layer_attn_comp_cache[il] || !g->attn_comp_stage) return false;
    if (rows > g->attn_comp_stage_cap || first_row > g->layer_comp_cap[il] ||
        rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }

    const uint64_t count = (uint64_t)rows * DS4_N_HEAD_DIM;
    const uint64_t dst_offset = (uint64_t)first_row *
                                metal_graph_attn_comp_cache_row_bytes();
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_copy_f32_to_f16(g->layer_attn_comp_cache[il],
                                               dst_offset,
                                               g->attn_comp_stage,
                                               0,
                                               count) != 0;
    }

    return ds4_gpu_tensor_copy(g->layer_attn_comp_cache[il],
                               dst_offset,
                               g->attn_comp_stage,
                               0,
                               count * sizeof(float)) != 0;
}

ds4_gpu_tensor *metal_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return DS4_GPU_ATTN_COMP_CACHE_F16
        ? g->attn_comp_stage
        : g->layer_attn_comp_cache[il];
}

uint32_t metal_graph_attn_comp_update_row(uint32_t row) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 0u : row;
}

bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) return true;
    return metal_graph_store_attn_comp_stage(g, il, first_row, rows);
}

ds4_gpu_tensor *metal_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_view(g->attn_comp_stage,
                                   0,
                                   (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

ds4_gpu_tensor *metal_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16) return g->attn_comp_stage;
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}

void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) ds4_gpu_tensor_free(t);
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */

#endif /* !DS4_NO_GPU */
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
#ifndef DS4_NO_GPU
bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->flat_hc,
                                             1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_pre", g->output_pre, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_weights", g->output_weights, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc", g->output_embd, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_norm", g->output_norm, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = dense_matmul_typed(g->logits, model, weights->output,
                                      DS4_N_EMBD, vocab_dim, g->output_norm, 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, DS4_N_LAYER, 0);
    }
    if (ok) trace_hnorm_head(g, vocab_dim);
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
bool metal_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;
    ds4_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = dense_matmul_typed(logits, model, weights->output,
                                      DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: Metal plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

bool metal_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = dense_matmul_typed(out, model, w, in_dim, out_dim, x, n_tok) != 0;
    return ok;
}

/* =========================================================================
 * Metal Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * Metal tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

void metal_graph_trace_layer_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim, DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm, corr_layer_delta(model, il));
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];

    bool ok = ds4_gpu_tensor_read(g->attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_gate, 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_up, 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_mid, 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_gpu_tensor_read(g->router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_gpu_tensor_read(g->routed_mid, 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: Metal stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: Metal shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: Metal routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}

#endif /* !DS4_NO_GPU */
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
#ifndef DS4_NO_GPU
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
    const bool index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    const bool layer_stage_profile = getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL;
    const bool q_stage_profile = getenv("DS4_METAL_Q_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define DS4_METAL_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = metal_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    if (ok && (stages & DS4_ATTN_STAGE_PRE)) {
    if (ok) {
        metal_graph_debug_dump_tensor("hc_in", g->batch_cur_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
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
    if (ok) {
        metal_graph_debug_dump_tensor("hc_mix", hc_mix_view,
                                      (uint64_t)n_tokens * mix_hc, il, pos0);
    }
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            g->batch_cur_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
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
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", g->batch_attn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                       g->batch_attn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->attn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", g->batch_attn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("norm");
    DS4_METAL_PROFILE_Q_STAGE("pre_q");
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
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", g->batch_qr,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
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
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
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
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_qr_norm,
                                                           g->batch_qr,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", g->batch_qr_norm,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a_norm");
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
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_b");
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                n_tokens,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qnorm", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("head_norm");
    if (ok && !(stages & DS4_ATTN_STAGE_NOROPE)) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
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
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("rope");
    DS4_METAL_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
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
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_kv,
                                                           g->batch_kv_raw,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok && !(stages & DS4_ATTN_STAGE_NOROPE)) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
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
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    if (ok && !(stages & DS4_ATTN_STAGE_NOROPE))
        ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM,
                                                 DS4_N_ROT) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("kv_path");
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
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * DS4_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
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
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: Metal layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
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
        if (ok) metal_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              g->batch_comp_kv,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) metal_graph_debug_dump_tensor("attn_comp_score_raw",
                                              g->batch_comp_sc,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && DS4_GPU_ATTN_COMP_CACHE_F16 && n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            ds4_gpu_tensor *attn_comp_target = NULL;
            if (ok) {
                attn_comp_target = metal_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                ok = attn_comp_target != NULL &&
                     ds4_gpu_compressor_prefill_tensor(attn_comp_target,
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
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)n_comp * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                metal_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                metal_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
            metal_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
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
            const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && DS4_GPU_ATTN_COMP_CACHE_F16 && comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target =
                    ok ? metal_graph_attn_comp_prefill_target(g, il, comp_before, comp_chunk) : NULL;
                if (ok && !attn_comp_target) ok = false;
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
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)comp_chunk * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                metal_graph_attn_comp_prefill_target_free(attn_comp_target);
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
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
                        ok = comp_row_view &&
                             ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                   1,
                                                                   DS4_N_HEAD_DIM,
                                                                   DS4_N_ROT) != 0;
                        if (ok) {
                            metal_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
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
        DS4_METAL_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: Metal layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
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
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  g->batch_comp_kv,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  g->batch_comp_sc,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_q,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_attn_q_b->abs_offset,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
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
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_weights,
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
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(g->layer_index_comp_cache[il],
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
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->layer_index_comp_cache[il],
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
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
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      g->layer_index_comp_cache[il],
                                                      (uint64_t)n_comp * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    metal_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                g->layer_index_comp_cache[il],
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
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
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    ds4_gpu_tensor_free(index_view);
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
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t == 0) ok = metal_graph_capture_prefix1_index_state(g, il);
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) DS4_METAL_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            if (ok && g->spec_comp_capture)
                ok = metal_graph_spec_raw_snapshot(g, il, pos0, n_tokens);
            if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
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
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_tensor("indexer_scores",
                                                  g->indexer_scores,
                                                  (uint64_t)n_comp * n_tokens,
                                                  il,
                                                  pos0);
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                          g->comp_selected,
                                                          (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                          il,
                                                          pos0);
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
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
                                                                              metal_graph_attn_comp_cache_is_f16(),
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
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             g->batch_q,
                                                                             g->layer_raw_cache[il],
                                                                             g->layer_attn_comp_cache[il],
                                                                             metal_graph_attn_comp_cache_is_f16(),
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
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
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
            if (ok && index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
                metal_graph_debug_dump_tensor("indexer_scores",
                                              g->indexer_scores,
                                              (uint64_t)n_comp * n_tokens,
                                              il,
                                              pos0);
            }
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->indexer_scores,
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                      g->comp_selected,
                                                      (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                      il,
                                                      pos0);
                }
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          g->batch_q,
                                                                          g->layer_raw_cache[il],
                                                                          g->layer_attn_comp_cache[il],
                                                                          metal_graph_attn_comp_cache_is_f16(),
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
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
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
                                                                       metal_graph_attn_comp_cache_is_f16(),
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
                                                                              metal_graph_attn_comp_cache_is_f16(),
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
                                                                 metal_graph_attn_comp_cache_is_f16(),
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
    DS4_METAL_PROFILE_ATTN_STAGE("attention");
    }   /* stage KV */
    if (ok && (stages & DS4_ATTN_STAGE_POST)) {
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
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
        metal_graph_debug_dump_tensor("kqv_back", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("inv_rope");
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
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups * rank,
                                      il,
                                      pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("output_proj");
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", g->batch_after_attn_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_post");
    }   /* stage POST */
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef DS4_METAL_PROFILE_ATTN_STAGE
#undef DS4_METAL_PROFILE_Q_STAGE
    return ok;
}
#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    return metal_graph_encode_layer_ffn_batch_ex(g, model, layer, il, pos0, n_tokens, false);
}
bool metal_graph_encode_layer_ffn_batch_ex(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens,
        bool                    no_zchain) {
    /* no_zchain(2026-08-21): dspark drafter 复用 il=0 批段, 旧注释"L0 无 z 语义中性"
     * 在 amp42 链(L0 有 AMP)下过时 → mtp 层误吃 L0 放大器。drafter 调用旁路侧车。 */
    const struct ds4_zchain *zch = no_zchain ? NULL : model->zchain;
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = routed_expert_in_dim(layer);
    const uint64_t expert_mid_dim = routed_expert_mid_dim(layer);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = (layer->ffn_gate_exps ? routed_expert_row_bytes(layer->ffn_gate_exps) : 0);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_ffn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                       g->batch_ffn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->ffn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("norm");
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_router_logits,
                                             model->map,
                                             model->size,
                                             layer->ffn_gate_inp->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    /* go1b correction: bias raw router logits by delta[e] before top-k (broadcast
     * over the n_tokens rows). Score-routed layers only; δ≡0 skips the dispatch. */
    if (ok && model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present &&
        model->corr->layer[il].has_delta && layer->ffn_gate_tid2eid == NULL) {
        ok = ds4_gpu_corr_router_bias(g->batch_router_logits, model->corr->layer[il].gdelta,
                                      DS4_N_EXPERT, n_tokens) != 0;
    }

    /* 路由闭式侧车(type8): 批路同挂, select 前(2026-08-19) */
    if (ok && ds4_zchain_layer_rte(zch, il))
        ok = ds4_gpu_zchain_route_bias(g->batch_router_logits, g->batch_ffn_norm, il, n_tokens) != 0;
    if (ok) ok = ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens, il) != 0;
    /* 路由空槽消毒(必须在 sorted-pairs 记账之前): -1 槽会造成 counts[-1] 越界原子写 +
     * 该 pair 无 tile ⇒ mid 槽未初始化 ⇒ down 读垃圾(drafter 同输入两跑草稿全不同的真因)。 */
    if (ok) ok = ds4_gpu_sanitize_router_tensor(g->batch_router_selected, g->batch_router_weights,
                                                (uint32_t)n_tokens * DS4_N_EXPERT_USED,
                                                model_expert_kept_count(model, il)) != 0;
    if (ok) router_freq_collect(g->batch_router_logits, il, n_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("router");

    /* zchain GE: fold per-expert gains into the router weights (original ids;
     * before the compact-slot translate and before the go1b MoE's in-place remap). */
    if (ok && model->zchain && ds4_zchain_layer_ge(model->zchain, il)) {
        ok = ds4_gpu_zchain_ge_apply(g->batch_router_weights, g->batch_router_selected,
                                     il, DS4_N_EXPERT_USED, n_tokens) != 0;
    }
    /* Translate full-256 router ids to compact slots for a shrunken model (no-op
     * for a full model). All n_tokens rows share the same per-layer LUT. */
    if (ok && model->expert_shrunken) {
        ok = ds4_gpu_translate_expert_ids(g->batch_router_selected, il, DS4_N_EXPERT_USED,
                                          n_tokens, model_expert_kept_count(model, il)) != 0;
    }

    /* TP Phase-3 batch split (DS4_TP_EXPERT_SPLIT): each peer gathers/computes its
     * half of every token's routed experts; the partial batch_routed_out is
     * all-reduce SUMMED below. Ends the prefill redundancy (both peers had run the
     * FULL gather) -> halves the cold batch (prefill) expert IO. Covers the batched
     * paths (prefill chunks + verify batches); decode bare rounds use the single-
     * token split. Default OFF => full gather. */
    /* Separate gate from the single-token decode split: the batched path AR's once
     * per (chunk, layer) -> prefill issues thousands of ARs whose drain overhead
     * outweighs the IO halving (wave-72: 281s redundant < 688s split). Default OFF;
     * opt-in for cold-batch experiments. Single-token decode keeps DS4_TP_EXPERT_SPLIT. */
    const char *tp_es_env_b = getenv("DS4_TP_EXPERT_SPLIT_BATCH");
    const bool tp_batch_split =
        g->tp && il < g->tp_layers && tp_es_env_b && tp_es_env_b[0] && tp_es_env_b[0] != '0' &&
        DS4_N_EXPERT_USED >= 2u;
    uint32_t tpb_slot_start = 0u, tpb_slot_count = 0u;
    if (tp_batch_split) {
        const uint32_t k = DS4_N_EXPERT_USED / 2u;
        tpb_slot_start = g->tp_owns_low ? 0u : k;
        tpb_slot_count = g->tp_owns_low ? k : (DS4_N_EXPERT_USED - k);
    }
    /* 小批 MoE 快路(2026-08-21 投机战役): ≤8 tok 逐 token 走 decode 单 token 路
     * (lut_gate/direct_down_sum6, 实测 0.2ms/层/token vs tile8 批路 2.35ms/层)。
     * verify(6)/draft(5)/replay(1-3) 全吃到; prefill 大批不受影响。DS4_SPEC_MOE_BATCH=1 回退。 */
    if (ok && n_tokens <= 8u && !tp_batch_split && !model->expert_shrunken &&
        getenv("DS4_SPEC_MOE_PER_TOKEN") != NULL) {   /* 实验开关: 默认批路(逐token路 -21ms 但 acc -0.4 净亏) */
        for (uint32_t t = 0; ok && t < n_tokens; t++) {
            ds4_gpu_tensor *xv = metal_graph_tensor_row_view(g->batch_ffn_norm, t, DS4_N_EMBD);
            ds4_gpu_tensor *ov = metal_graph_tensor_row_view(g->batch_routed_out, t, DS4_N_EMBD);
            ds4_gpu_tensor *sv = metal_graph_tensor_row_view(g->batch_router_selected, t, DS4_N_EXPERT_USED);
            ds4_gpu_tensor *wv = metal_graph_tensor_row_view(g->batch_router_weights, t, DS4_N_EXPERT_USED);
            ok = xv && ov && sv && wv &&
                 ds4_gpu_routed_moe_one_tensor(ov,
                                               g->batch_routed_gate, g->batch_routed_up,
                                               g->batch_routed_mid, g->batch_routed_down,
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
                                               sv, wv,
                                               model_expert_kept_count(model, il),
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               xv, il) != 0;
            ds4_gpu_tensor_free(wv); ds4_gpu_tensor_free(sv);
            ds4_gpu_tensor_free(ov); ds4_gpu_tensor_free(xv);
        }
        g->batch_routed_mid_is_f16 = false;   /* 单 token 路 mid 中间态与批口径无关 */
    } else if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
                                               residual_set_for(model, il),
                                               model->map,
                                               model->size,
                                               routed_expert_gate_off(layer),
                                               routed_expert_up_off(layer),
                                               layer->ffn_down_exps->abs_offset,
                                               routed_expert_quant_type(layer),
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               model_expert_kept_count(model, il),
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens,
                                               tpb_slot_start,
                                               tpb_slot_count,
                                               &g->batch_routed_mid_is_f16) != 0;
    }
    /* batch all-reduce: sum each peer's partial routed_out [n_tokens x n_embd] into
     * the full routed output (same MTLSharedEvent fast-wait path as the decode AR;
     * routed_moe_batch leaves the batch CB open on the was_batched path). */
    if (ok && tp_batch_split) {
        const uint64_t ar_n = (uint64_t)n_tokens * (uint64_t)DS4_N_EMBD;
        const uint64_t ev = ds4_gpu_tp_signal_after_batch();
        ok = ev != 0;
        if (ok) ok = ds4_gpu_flush_commands() != 0;
        if (ok) ok = ds4_gpu_tp_host_wait(ev) != 0;
        float *arbuf = ok ? malloc((size_t)ar_n * sizeof(float)) : NULL;
        if (ok && !arbuf) ok = false;
        if (ok) ok = ds4_gpu_tensor_read(g->batch_routed_out, 0, arbuf,
                                         ar_n * sizeof(float)) != 0;
        if (ok) ok = ds4_dist_tp_allreduce_f32(g->tp, arbuf, (uint32_t)ar_n) == 0;
        if (ok) ok = ds4_gpu_tensor_write(g->batch_routed_out, 0, arbuf,
                                          ar_n * sizeof(float)) != 0;
        free(arbuf);
    }
    /* zchain λ(x): scale the (fully summed) routed output before the additive
     * corr — quantizer op-order parity. */
    if (ok && (ds4_zchain_layer_has_lambda(model->zchain, il) ||
               ds4_zchain_layer_zl(model->zchain, il))) {   /* λ 和/或 冻结 z^L 同一派发 */
        ok = ds4_gpu_zchain_scale_routed(g->batch_routed_out, g->batch_ffn_norm,
                                         il, n_tokens) != 0;
    }
    /* Engine-trajectory batch capture (DS4_CAP_DIR [+DS4_CAP_LAYERS lo-hi]):
     * append this chunk's x̂ / routing / raw router logits / gate weights as
     * raw f16/i16 shards — THE ground-truth student trajectory for error-
     * feedback calibration (the Python fp32-backbone simulation drifts from
     * the engine in deep layers; R2 verdict). ffn_norm and router_logits are
     * final at this point; selected uses the same pre-remap snapshot the corr
     * dispatch uses. Off unless DS4_CAP_DIR is set. */
    if (ok) cap_batch_layer(g, il, n_tokens);
    /* go1b correction: per-token low-rank residual onto the full routed MoE output
     * (after any TP all-reduce). x=g->batch_ffn_norm, selected=g->batch_router_selected. */
    if (ok && model->corr && il < DS4_MAX_LAYER && model->corr->layer[il].present) {
        const ds4_corr_layer *cl = &model->corr->layer[il];
        /* Same remap hazard as decode: the go1b batch MoE rewrites batch_router_selected
         * to compact slots in place, so read the pre-remap snapshot (per-token original
         * ids, [n_tokens][n_expert_used]); fall back to the live tensor if not go1b. */
        const ds4_gpu_tensor *corr_sel = ds4_gpu_corr_saved_selected();
        if (!corr_sel) corr_sel = g->batch_router_selected;
        ok = ds4_gpu_corr_apply(g->batch_routed_out,
                                model->corr->phi_yhat ? g->batch_routed_out : g->batch_ffn_norm,
                                cl->gU, cl->gV, cl->gC, cl->gb, cl->gbeta,
                                corr_sel,
                                DS4_N_EMBD, cl->d_l, DS4_N_EXPERT, DS4_N_EXPERT_USED, n_tokens) != 0;
        if (getenv("DS4_RESIDUAL_DEBUG"))
            fprintf(stderr, "ds4: [corr-batch] L%u ok=%d d_l=%u ntok=%u gU=%p gC=%p\n",
                    il, ok, cl->d_l, n_tokens, (void*)cl->gU, (void*)cl->gC);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->batch_routed_gate,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            metal_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                              routed_mid_elems, il, pos0);
        } else {
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("routed_moe");
    /* shexp gate+up 批 pair 融合(2026-08-21): 同输入两矩阵一发 —— 小矩阵单发在 6-token
     * 批下只有 ~35GB/s(gx 小并行度不足), 合并后行数翻倍且权重只读一遍。不适用则回退。 */
    bool shexp_pair_done = false;
    if (ok && n_tokens > 1u &&
        layer->ffn_gate_shexp->type == DS4_TENSOR_Q2_K &&
        layer->ffn_up_shexp->type == DS4_TENSOR_Q2_K &&
        getenv("DS4_NO_SHEXP_PAIR") == NULL) {
        shexp_pair_done = ds4_gpu_matmul_q2_K_pair_batch_tensor(
                g->batch_shared_gate, g->batch_shared_up,
                model->map, model->size,
                layer->ffn_gate_shexp->abs_offset, layer->ffn_up_shexp->abs_offset,
                DS4_N_EMBD, shared_dim, shared_dim,
                g->batch_ffn_norm, n_tokens) != 0;
    }
    if (ok && !shexp_pair_done) ok = metal_graph_matmul_q8_0_named_tensor("shared_gate",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_gate,
                                                      model,
                                                      layer->ffn_gate_shexp,
                                                      DS4_N_EMBD,
                                                      shared_dim,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
    if (ok && !shexp_pair_done) ok = metal_graph_matmul_q8_0_named_tensor("shared_up",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_up,
                                                      model,
                                                      layer->ffn_up_shexp,
                                                      DS4_N_EMBD,
                                                      shared_dim,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
    DS4_METAL_PROFILE_FFN_STAGE("shared_gate_up");
    if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                         g->batch_shared_gate,
                                         g->batch_shared_up,
                                         (uint32_t)((uint64_t)n_tokens * shared_dim),
                                         DS4_SWIGLU_CLAMP_EXP,
                                         1.0f) != 0;
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_down",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_out,
                                                      model,
                                                      layer->ffn_down_shexp,
                                                      shared_dim,
                                                      DS4_N_EMBD,
                                                      g->batch_shared_mid,
                                                      n_tokens);
    DS4_METAL_PROFILE_FFN_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }

    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos0);
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_METAL_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
/* DS4_EVAL_HDUMP=<dir>: 每层出口 HC 隐状态整批追加写出 h_L%02d.bin
 * (f32, [S][DS4_N_HC][DS4_N_EMBD], chunk 顺序即 token 顺序)。
 * 挂在 HC 交换之后 —— 那时 batch_cur_hc 才是本层出口。GPU 定格用 cap_batch_layer
 * 同款 signal→flush→host_wait: 本层的核还在队列里, 不 drain 就 read 会拿到上一层
 * 残值(2026-07-22 那次 off-by-one 就是这么来的)。流式写盘, 不驻留。 */
#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU
void eval_hdump_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens) {
    const char *dir = getenv("DS4_EVAL_HDUMP");
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
    const bool attn_ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    bool ok = attn_ok;
    if (ok) ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    if (!ok && getenv("DS4_GRAPH_FAIL_TRACE"))
        fprintf(stderr, "ds4: [fail-trace] L%u %s_batch failed\n", il, attn_ok ? "ffn" : "attention");
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    /* DSpark prefill 抓取(层出口 HC 均值)与建窗: prompt 每 token 的 main_kv 进环形窗,
     * 与官方 prefill(start_pos==0 只建 KV)语义一致 */
    if (getenv("DS4_DSPARK_DIAG") && il == 42u)
        fprintf(stderr, "ds4: [dspark-diag] prefill L42 cap=%d pf=%p bound=%p n=%u\n",
                g->dspark_capture, (void *)g->dspark_pf_hidden,
                (const void *)g_dspark_bound_for_prefill, n_tokens);
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
        const uint32_t coff = ratio == 4 ? 2u : 1u;
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
#ifndef DS4_NO_GPU
bool metal_graph_dspark_step_n(
        ds4_gpu_graph *g,
        const ds4_model *model,
        const ds4_weights *weights,
        const ds4_dspark_weights *dw,
        int anchor_token,
        uint32_t pos,
        int out_ids[DS4_DSPARK_BLK],
        uint32_t n_need,
        float *out_conf) {
    /* drafter 权重可能在 DS4_DRAFT_GGUF 副文件; 主模型张量(token_embd/output)仍走 model */
    const ds4_model *dmodel = dw->src ? dw->src : model;
    const bool sprof = getenv("DS4_DSPARK_STEP_PROF") != NULL;
    /* GPU 跨度(2026-08-21): 与墙钟对照分离"GPU 真忙" vs "CPU 编码/同步阻塞"。
     * SP_MARK 自带同步会污染分段归因, 这条跨度不受影响。 */
    const bool sspan = getenv("DS4_DSPARK_STEP_SPAN") != NULL;
    const double span_t0 = sspan ? now_sec() : 0.0;
    if (sspan) ds4_gpu_span_begin();
    double sp_t0 = sprof ? now_sec() : 0.0;
    static double sp_acc[6]; static uint32_t sp_n;
    float sp_sink;
#define SP_MARK(idx) do { if (sprof) { \
        (void)ds4_gpu_tensor_read(g->dspark_main_x, 0, &sp_sink, 4); /* 流同步 */ \
        sp_acc[idx] += now_sec() - sp_t0; sp_t0 = now_sec(); } } while (0)
    const uint32_t B = DS4_DSPARK_BLK;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q_rank = dw->block[0].attn_q_a->dim[1];
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    /* drafter compress_ratio==0: 非压缩 rope 口径(theta=10000, scale=1) —— L0 可能是
     * 压缩层, 不能借 layer_rope_freq_base(0) */
    const float freq_base = DS4_ROPE_FREQ_BASE;
    const float freq_scale = 1.0f;
    bool ok = true;

    /* ---- ① main_x ---- */
    if (ok) ok = dense_matmul_typed(g->dspark_main_x_raw, dmodel, dw->main_proj,
                                    3ull * DS4_N_EMBD, DS4_N_EMBD,
                                    g->dspark_main_hidden, 1) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x, g->dspark_main_x_raw,
                                                dmodel->map, dmodel->size,
                                                dw->main_norm->abs_offset,
                                                DS4_N_EMBD, DS4_RMS_EPS) != 0;

    /* ---- ② 每块层 main_kv → 环形窗 ---- */
    const uint32_t win_row = pos % DS4_DSPARK_WIN;
    if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-step] stage A_mainproj ok=%d\n", (int)ok);
    for (uint32_t b = 0; ok && b < (uint32_t)dw->n_blocks; b++) {
        ds4_gpu_tensor *rowv = ds4_gpu_tensor_view(g->dspark_win_kv[b],
                (uint64_t)win_row * DS4_N_HEAD_DIM * sizeof(float),
                (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        ok = rowv != NULL;
        if (ok) ok = dense_matmul_typed(g->dspark_kv_tmp, dmodel, dw->block[b].attn_kv,
                                        DS4_N_EMBD, DS4_N_HEAD_DIM,
                                        g->dspark_main_x, 1) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(rowv, g->dspark_kv_tmp,
                                                    dmodel->map, dmodel->size,
                                                    dw->block[b].attn_kv_a_norm->abs_offset,
                                                    DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(rowv, 1, 1, DS4_N_HEAD_DIM, DS4_N_ROT,
                                              pos, 0, false, freq_base, freq_scale,
                                              0.0f, 1.0f,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        ds4_gpu_tensor_free(rowv);
    }
    uint32_t n_win = (pos + 1u < DS4_DSPARK_WIN) ? (pos + 1u) : DS4_DSPARK_WIN;
    if (getenv("DS4_DSPARK_NOWIN")) n_win = 0;   /* 判别实验: 窗贡献归零对照 */
    /* 对拍 dump(DS4_DSPARK_DUMP=/tmp/dir): main_hidden/main_x/窗行/后续 q/kv 落盘 */
    const char *ddump = getenv("DS4_DSPARK_DUMP");
    if (ddump) {
        char p[512]; FILE *f;
        float *tmpbuf = xmalloc(3ull * DS4_N_EMBD * sizeof(float));
        snprintf(p, sizeof(p), "%s/mh_pos%u.bin", ddump, pos);
        if (ds4_gpu_tensor_read(g->dspark_main_hidden, 0, tmpbuf, 3ull * DS4_N_EMBD * sizeof(float)) &&
            (f = fopen(p, "wb"))) { fwrite(tmpbuf, 4, 3ull * DS4_N_EMBD, f); fclose(f); }
        snprintf(p, sizeof(p), "%s/mx_pos%u.bin", ddump, pos);
        if (ds4_gpu_tensor_read(g->dspark_main_x, 0, tmpbuf, (uint64_t)DS4_N_EMBD * sizeof(float)) &&
            (f = fopen(p, "wb"))) { fwrite(tmpbuf, 4, DS4_N_EMBD, f); fclose(f); }
        {
            float *wf = xmalloc(128ull * DS4_N_HEAD_DIM * sizeof(float));
            for (uint32_t wb = 0; wb < (uint32_t)dw->n_blocks; wb++) {
                snprintf(p, sizeof(p), "%s/winfull_b%u_pos%u.bin", ddump, wb, pos);
                if (ds4_gpu_tensor_read(g->dspark_win_kv[wb], 0, wf, 128ull * DS4_N_HEAD_DIM * sizeof(float)) &&
                    (f = fopen(p, "wb"))) { fwrite(wf, 4, 128ull * DS4_N_HEAD_DIM, f); fclose(f); }
            }
            free(wf);
        }
        snprintf(p, sizeof(p), "%s/kvtmp_b2_pos%u.bin", ddump, pos);
        if (ds4_gpu_tensor_read(g->dspark_kv_tmp, 0, tmpbuf, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) &&
            (f = fopen(p, "wb"))) { fwrite(tmpbuf, 4, DS4_N_HEAD_DIM, f); fclose(f); }
        for (uint32_t b = 0; b < (uint32_t)dw->n_blocks; b++) {
            snprintf(p, sizeof(p), "%s/winrow_b%u_pos%u.bin", ddump, b, pos);
            if (ds4_gpu_tensor_read(g->dspark_win_kv[b],
                    (uint64_t)(pos % DS4_DSPARK_WIN) * DS4_N_HEAD_DIM * sizeof(float),
                    tmpbuf, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) &&
                (f = fopen(p, "wb"))) { fwrite(tmpbuf, 4, DS4_N_HEAD_DIM, f); fclose(f); }
        }
        free(tmpbuf);
    }

    /* ---- ③ 块 embed: [anchor, noise×4] ---- */
    int32_t ids[DS4_DSPARK_BLK];
    ids[0] = anchor_token;
    for (uint32_t i = 1; i < B; i++) ids[i] = DS4_DSPARK_NOISE;
    if (ok) ok = ds4_gpu_tensor_write(g->dspark_ids, 0, ids, sizeof(ids)) != 0;
    if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-step] stage B_kvwin ok=%d\n", (int)ok);
    if (ok) ok = ds4_gpu_embed_tokens_hc_tensor(g->batch_cur_hc, g->dspark_ids,
                                                model->map, model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                B, DS4_N_EMBD, DS4_N_HC) != 0;

    SP_MARK(0);   /* main_x + 窗 + embed */
    /* ---- 3 层 drafter ---- */
    if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-step] stage C_embed ok=%d\n", (int)ok);
    for (uint32_t b = 0; ok && b < (uint32_t)dw->n_blocks; b++) {
        const ds4_layer_weights *layer = &dw->block[b];
        const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
        ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(g->batch_hc_mix, 0, (uint64_t)B * mix_hc * sizeof(float));
        ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(g->batch_hc_split, 0, (uint64_t)B * mix_hc * sizeof(float));
        ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(g->batch_attn_cur, 0, (uint64_t)B * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(g->batch_after_attn_hc, 0, (uint64_t)B * hc_dim * sizeof(float));
        ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
        /* hc_pre(attn) */
        if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                                        (uint32_t)hc_dim, B, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view, dmodel->map, dmodel->size,
                                               layer->hc_attn_fn->abs_offset,
                                               hc_dim, mix_hc, g->batch_flat_hc, B) != 0;
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view, hc_split_view, hc_mix_view,
                                                          g->batch_cur_hc, dmodel->map, dmodel->size,
                                                          layer->hc_attn_scale->abs_offset,
                                                          layer->hc_attn_base->abs_offset,
                                                          DS4_N_EMBD, DS4_N_HC,
                                                          DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm, g->batch_attn_cur,
                                                         dmodel->map, dmodel->size,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD, B, DS4_RMS_EPS) != 0;
        /* q 链 + 块内 kv */
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_a", 43 + (int)b, pos,
                                                          g->batch_qr, dmodel, layer->attn_q_a,
                                                          DS4_N_EMBD, q_rank, g->batch_attn_norm, B);
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_kv", 43 + (int)b, pos,
                                                          g->batch_kv_raw, dmodel, layer->attn_kv,
                                                          DS4_N_EMBD, DS4_N_HEAD_DIM, g->batch_attn_norm, B);
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm, g->batch_qr,
                                                           dmodel->map, dmodel->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           g->batch_kv, g->batch_kv_raw,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM, B, DS4_RMS_EPS) != 0;
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_b", 43 + (int)b, pos,
                                                          g->batch_q, dmodel, layer->attn_q_b,
                                                          q_rank, q_dim, g->batch_qr_norm, B);
        if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q, B, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                                  DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q, B, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                              DS4_N_ROT, pos + 1u, 0, false,
                                              freq_base, freq_scale, 0.0f, 1.0f,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_kv, B, 1, DS4_N_HEAD_DIM,
                                              DS4_N_ROT, pos + 1u, 0, false,
                                              freq_base, freq_scale, 0.0f, 1.0f,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok && ddump && b == 0) {
            char p[512]; FILE *f;
            float *qb = xmalloc((uint64_t)B * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float));
            snprintf(p, sizeof(p), "%s/q_b0_pos%u.bin", ddump, pos);
            if (ds4_gpu_tensor_read(g->batch_q, 0, qb, (uint64_t)B * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float)) &&
                (f = fopen(p, "wb"))) { fwrite(qb, 4, (uint64_t)B * DS4_N_HEAD * DS4_N_HEAD_DIM, f); fclose(f); }
            snprintf(p, sizeof(p), "%s/blkkv_b0_pos%u.bin", ddump, pos);
            if (ds4_gpu_tensor_read(g->batch_kv, 0, qb, (uint64_t)B * DS4_N_HEAD_DIM * sizeof(float)) &&
                (f = fopen(p, "wb"))) { fwrite(qb, 4, (uint64_t)B * DS4_N_HEAD_DIM, f); fclose(f); }
            free(qb);
        }
        /* drafter attention: 窗 + 块内 */
        /* 窗首位置(2026-08-21 修被拒候选污染): 环行按位置映射, 只读 [pos+1-n_win, pos]。
         * 原来读 ring[0..n_win-1], 上下文过 128 后会把 verify 批写入的被拒(未来)行读进来。 */
        const uint32_t win_base = (uint32_t)(((uint64_t)pos + 1u + DS4_DSPARK_WIN - n_win) % DS4_DSPARK_WIN);
        if (ok) ok = ds4_gpu_dspark_attn_tensor(g->batch_heads, dmodel->map, dmodel->size,
                                                layer->attn_sinks->abs_offset,
                                                g->batch_q, g->dspark_win_kv[b], g->batch_kv,
                                                n_win, B, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                                win_base, DS4_DSPARK_WIN) != 0;
        if (ok && ddump && b == 0) {
            char p2[512]; FILE *f2;
            float *hb = xmalloc((uint64_t)B * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float));
            snprintf(p2, sizeof(p2), "%s/heads_b0_pos%u.bin", ddump, pos);
            if (ds4_gpu_tensor_read(g->batch_heads, 0, hb, (uint64_t)B * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float)) &&
                (f2 = fopen(p2, "wb"))) { fwrite(hb, 4, (uint64_t)B * DS4_N_HEAD * DS4_N_HEAD_DIM, f2); fclose(f2); }
            free(hb);
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_heads, B, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                              DS4_N_ROT, pos + 1u, 0, true,
                                              freq_base, freq_scale, 0.0f, 1.0f,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ((layer->attn_output_a->type == DS4_TENSOR_Q4_K || layer->attn_output_a->type == DS4_TENSOR_Q2_K)
                  ? attn_output_kq_batch(layer->attn_output_a, g->batch_attn_out, g->batch_attn_low,
                        dmodel->map, dmodel->size,
                        layer->attn_output_b->abs_offset,
                        group_dim, rank, n_groups, DS4_N_EMBD, g->batch_heads, B)
                  : ds4_gpu_attention_output_q8_batch_tensor(g->batch_attn_out, g->batch_attn_low,
                        g->batch_group_tmp, g->batch_low_tmp,
                        dmodel->map, dmodel->size,
                        layer->attn_output_a->abs_offset, layer->attn_output_b->abs_offset,
                        group_dim, rank, n_groups, DS4_N_EMBD, g->batch_heads, B)) != 0;
        if (ok && ddump && b == 0) {
            char p2[512]; FILE *f2;
            float *ab = xmalloc((uint64_t)B * DS4_N_EMBD * sizeof(float));
            snprintf(p2, sizeof(p2), "%s/attnout_b0_pos%u.bin", ddump, pos);
            if (ds4_gpu_tensor_read(g->batch_attn_out, 0, ab, (uint64_t)B * DS4_N_EMBD * sizeof(float)) &&
                (f2 = fopen(p2, "wb"))) { fwrite(ab, 4, (uint64_t)B * DS4_N_EMBD, f2); fclose(f2); }
            snprintf(p2, sizeof(p2), "%s/attncur_b0_pos%u.bin", ddump, pos);
            if (ds4_gpu_tensor_read(g->batch_attn_cur, 0, ab, (uint64_t)B * DS4_N_EMBD * sizeof(float)) &&
                (f2 = fopen(p2, "wb"))) { fwrite(ab, 4, (uint64_t)B * DS4_N_EMBD, f2); fclose(f2); }
            free(ab);
        }
        if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view, g->batch_attn_out,
                                                    g->batch_cur_hc, hc_split_view,
                                                    DS4_N_EMBD, DS4_N_HC) != 0;
        ds4_gpu_tensor_free(after_attn_hc_view);
        ds4_gpu_tensor_free(attn_cur_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(hc_mix_view);
        if (ok && getenv("DS4_DSPARK_DIAG")) {
            float ao[2] = {0}, hh[2] = {0}, qq[2] = {0}, kk[2] = {0}, wk[2] = {0};
            (void)ds4_gpu_tensor_read(g->batch_q, 0, qq, sizeof(qq));
            (void)ds4_gpu_tensor_read(g->batch_kv, 0, kk, sizeof(kk));
            (void)ds4_gpu_tensor_read(g->dspark_win_kv[b], 0, wk, sizeof(wk));
            const float *snk = (const float *)tensor_data(dmodel, layer->attn_sinks);
            (void)ds4_gpu_tensor_read(g->dspark_win_kv[b],
                    (uint64_t)((pos % 128u)) * DS4_N_HEAD_DIM * sizeof(float), wk, sizeof(wk));
            fprintf(stderr, "ds4: [dspark-diag] blk%u q=%.3g %.3g kv=%.3g %.3g winrow=%.3g %.3g sink=%.3g %.3g\n",
                    b, qq[0], qq[1], kk[0], kk[1], wk[0], wk[1], snk ? snk[0] : -999.0f, snk ? snk[1] : -999.0f);
            (void)ds4_gpu_tensor_read(g->batch_attn_out, 0, ao, sizeof(ao));
            (void)ds4_gpu_tensor_read(g->batch_heads, 0, hh, sizeof(hh));
            fprintf(stderr, "ds4: [dspark-diag] blk%u heads=%.3g %.3g attn_out=%.3g %.3g\n",
                    b, hh[0], hh[1], ao[0], ao[1]);
        }
        SP_MARK(4);   /* 本块 hc+attn 手写段 */
        /* FFN 批段整段复用(corr/zchain/hash 对 drafter 自然旁路)。
         * ★缓冲协议(2026-08-20 命中率根因修复): 主干管线 ffn_batch 读
         * batch_after_attn_hc(hc_post(attn) 出口已在那), 此处此前多做了一次
         * cur<->after swap ⇒ FFN 读到 embed 时代的原始 hc 流(attention 信号
         * 全丢, Fin noise 行全同) ⇒ 草稿烂。不 swap 才是主干同款协议。 */
        if (ok) {
            if (ddump) {   /* 对拍: hc_post(attn) 出口 [B, hc_dim] */
                char pd[512]; FILE *fd;
                const uint64_t hbytes = (uint64_t)B * DS4_N_HC * DS4_N_EMBD * sizeof(float);
                float *hb = xmalloc(hbytes);
                snprintf(pd, sizeof(pd), "%s/hcpost_b%u_pos%u.bin", ddump, b, pos);
                if (ds4_gpu_tensor_read(g->batch_after_attn_hc, 0, hb, hbytes) &&
                    (fd = fopen(pd, "wb"))) { fwrite(hb, 1, hbytes, fd); fclose(fd); }
                free(hb);
            }
            /* il 传 0: 43+b 会越界 per-layer 表(compress_ratio/collect); L0 无 hash
             * (tid2eid 按 layer 指针判)无 corr 无 z, 对 drafter 语义中性 */
            /* il=43+b: 合并 zchain 的 drafter 槽; 无 draft 侧车时 dmodel->zchain=NULL
             * 自然旁路。per-layer 表访问对 43..45 全有界(kept_count/residual/ampanc)。 */
            ok = metal_graph_encode_layer_ffn_batch_ex(g, dmodel, layer,
                                                       getenv("DS4_DSPARK_FFN_IL_LOW") ? b : 43u + b, pos + 1u, B,
                                                       dmodel->zchain == NULL);
            ds4_gpu_tensor *tmp = g->batch_cur_hc;   /* ffn 出口在 batch_next_hc → 下一块读 cur */
            g->batch_cur_hc = g->batch_next_hc;
            g->batch_next_hc = tmp;
        }
        /* 放大器锚捕获(DS4_DSPARK_ANCHOR=path): drafter FFN 的 (Fin, route, o_ref) 落盘。
         * 教师文件(exps 高精)跑一遍即锚; 学生侧由 zlayer 用 q2 gguf 对同 Fin 重放。
         * batch_routed_out 是 zchain 缩放前的 routed 和(bypass 下天然教师口径);
         * record: u32[5]{blk,pos,B,K,D} + fin[B*D] + sel[B*K]i32 + rw[B*K] + oref[B*D]。 */
        if (ok && getenv("DS4_DSPARK_ANCHOR")) {
            static FILE *anchf = NULL;
            if (!anchf) anchf = fopen(getenv("DS4_DSPARK_ANCHOR"), "wb");
            if (anchf) {
                const uint32_t K = DS4_N_EXPERT_USED, D = DS4_N_EMBD;
                float *fin = xmalloc((size_t)B * D * 4), *oref = xmalloc((size_t)B * D * 4);
                float *rw = xmalloc((size_t)B * K * 4);
                int32_t *sel = xmalloc((size_t)B * K * 4);
                if (ds4_gpu_tensor_read(g->batch_ffn_norm, 0, fin, (uint64_t)B * D * 4) &&
                    ds4_gpu_tensor_read(g->batch_routed_out, 0, oref, (uint64_t)B * D * 4) &&
                    ds4_gpu_tensor_read(g->batch_router_selected, 0, sel, (uint64_t)B * K * 4) &&
                    ds4_gpu_tensor_read(g->batch_router_weights, 0, rw, (uint64_t)B * K * 4)) {
                    const uint32_t hdr[5] = { b, pos, B, K, D };
                    fwrite(hdr, 4, 5, anchf);
                    fwrite(fin, 4, (size_t)B * D, anchf);
                    fwrite(sel, 4, (size_t)B * K, anchf);
                    fwrite(rw, 4, (size_t)B * K, anchf);
                    fwrite(oref, 4, (size_t)B * D, anchf);
                    fflush(anchf);
                }
                free(fin); free(oref); free(rw); free(sel);
            }
        }
        if (ok && getenv("DS4_DSPARK_DIAG")) {
            float ff[2] = {0};
            (void)ds4_gpu_tensor_read(g->batch_cur_hc, 0, ff, sizeof(ff));
            fprintf(stderr, "ds4: [dspark-diag] blk%u ffn_out=%.3g %.3g\n", b, ff[0], ff[1]);
        }
        /* 数值护栏: drafter 约 1-4% 行会产 NaN(既有问题, 新旧 MoE kernel 同现), 一旦落到
         * 锚位就整轮草稿作废。每块出口清洗一次, 代价可忽略。DS4_DSPARK_NO_SANITIZE=1 关。 */
        if (ok && getenv("DS4_DSPARK_NO_SANITIZE") == NULL)
            ok = ds4_gpu_sanitize_finite_tensor(g->batch_cur_hc,
                                                (uint64_t)B * DS4_N_HC * DS4_N_EMBD) != 0;
        SP_MARK(5);   /* 本块 ffn_batch 段 */
    }
    if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-step] stage D_blocks ok=%d\n", (int)ok);
    SP_MARK(1);   /* 3 块层(hc+attn+ffn) */

    /* ---- ④ hc_head → norm → lm_head → markov ---- */
    const ds4_model *hmodel = dw->head_src ? dw->head_src : dmodel;   /* 出口侧张量归属 */
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                                    (uint32_t)hc_dim, B, DS4_RMS_EPS) != 0;
    if (ok) ok = (dw->hc_head_fn->type == DS4_TENSOR_F32
                  ? ds4_gpu_matmul_f32_tensor(g->dspark_hc_pre, hmodel->map, hmodel->size,
                                              dw->hc_head_fn->abs_offset,
                                              hc_dim, DS4_N_HC, g->batch_flat_hc, B)
                  : ds4_gpu_matmul_f16_tensor(g->dspark_hc_pre, hmodel->map, hmodel->size,
                                              dw->hc_head_fn->abs_offset,
                                              hc_dim, DS4_N_HC, g->batch_flat_hc, B)) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->dspark_hc_w, g->dspark_hc_pre,
                                                  hmodel->map, hmodel->size,
                                                  dw->hc_head_scale->abs_offset,
                                                  dw->hc_head_base->abs_offset,
                                                  DS4_N_HC, DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->dspark_flat, g->batch_cur_hc,
                                                g->dspark_hc_w, DS4_N_EMBD, DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->dspark_flat_norm, g->dspark_flat,
                                                     hmodel->map, hmodel->size,
                                                     dw->norm->abs_offset,
                                                     DS4_N_EMBD, B, DS4_RMS_EPS) != 0;
    if (ok) ok = dense_matmul_typed(g->dspark_logits, model, weights->output,
                                    DS4_N_EMBD, DS4_N_VOCAB, g->dspark_flat_norm, B) != 0;
    if (ddump) {
        char p3[512]; FILE *f3;
        float *lgb = xmalloc(5ull * DS4_N_VOCAB * sizeof(float));
        snprintf(p3, sizeof(p3), "%s/logits_premarkov_pos%u.bin", ddump, pos);
        if (ds4_gpu_tensor_read(g->dspark_logits, 0, lgb, 5ull * DS4_N_VOCAB * sizeof(float)) &&
            (f3 = fopen(p3, "wb"))) { fwrite(lgb, 4, 5ull * DS4_N_VOCAB, f3); fclose(f3); }
        free(lgb);
    }
    SP_MARK(2);   /* hc_head + norm + lm_head(logits) */
    /* markov 链: 位置 i 的 prev=位置 i-1 的 argmax(串行 5 步) */
    /* 逐步 host read 已去掉(2026-08-21): 每步 4B 设备→主机读 = 一次流同步, 5 步就是 5 次
     * 管线气泡。链内只需要设备侧的 prev_id, 主机侧到末尾一次读回整条链即可。
     * prev_ids[0]=输入 token, prev_ids[i+1]=第 i 位草稿 ⇒ 同一数组既喂链又喂置信头。 */
    if (ok) ok = ds4_gpu_tensor_write(g->dspark_prev_id, 0, &ids[0], sizeof(int32_t)) != 0;
    if (ok) ok = ds4_gpu_tensor_write(g->dspark_prev_ids, 0, &ids[0], sizeof(int32_t)) != 0;
    /* 候选数 k<6 时后几位草稿用不上: markov 链是串行 5 步(每步一次设备往返), 按需截断。 */
    const uint32_t n_chain = (n_need && n_need < B) ? n_need : B;
    for (uint32_t i = 0; ok && i < n_chain; i++) {
        ds4_gpu_tensor *lrow = ds4_gpu_tensor_view(g->dspark_logits,
                (uint64_t)i * DS4_N_VOCAB * sizeof(float),
                (uint64_t)DS4_N_VOCAB * sizeof(float));
        ok = lrow != NULL;
        if (getenv("DS4_DSPARK_NOMARKOV") || !dw->markov_w1 || !dw->markov_w2) {
            /* markov 头可缺(checkpoint 的 mtp.* 不含时): 纯 argmax 草稿 */
            if (ok) ok = ds4_gpu_dspark_argmax_only_tensor(g->dspark_out_id, lrow,
                                                           (uint32_t)DS4_N_VOCAB) != 0;
        } else if (ok) {
            ok = ds4_gpu_dspark_markov_step_tensor(g->dspark_out_id, lrow,
                                                   dmodel->map, dmodel->size,
                                                   dw->markov_w1->abs_offset,
                                                   dw->markov_w2->abs_offset,
                                                   g->dspark_prev_id,
                                                   (uint32_t)DS4_N_VOCAB, 256u) != 0;
        }
        /* out_id 变 prev_id: 设备内 4B 拷贝(下一步读) + 落到链数组第 i+1 位 */
        if (ok) ok = ds4_gpu_tensor_copy(g->dspark_prev_id, 0, g->dspark_out_id, 0, sizeof(int32_t)) != 0;
        if (ok) ok = ds4_gpu_tensor_copy(g->dspark_prev_ids, (uint64_t)(i + 1u) * sizeof(int32_t),
                                         g->dspark_out_id, 0, sizeof(int32_t)) != 0;
        ds4_gpu_tensor_free(lrow);
    }
    /* 置信头(论文 Alg.1 输入): 位置 i 的输入 token 是 ids[0](i=0) 或 out_ids[i-1]。
     * markov 链跑完才知道这些 id, 所以一次性算完整块, 一次设备往返。 */
    if (ok && out_conf && dw->confidence_proj && dw->markov_w1 && g->dspark_conf && g->dspark_prev_ids) {
        /* prev_ids[0..n_chain-1] 就是每位置的输入 token, 链跑完已在设备上, 无需回写 */
        ok = ds4_gpu_dspark_confidence_tensor(g->dspark_conf, g->dspark_flat,
                                              dmodel->map, dmodel->size,
                                              dw->confidence_proj->abs_offset,
                                              dw->markov_w1->abs_offset,
                                              g->dspark_prev_ids,
                                              (uint32_t)DS4_N_EMBD, 256u,
                                              (uint32_t)DS4_N_VOCAB, n_chain) != 0 != 0;
    }
    {   /* 一次读回整条链 */
        int32_t chain[DS4_DSPARK_BLK + 1];
        if (ok) ok = ds4_gpu_tensor_read(g->dspark_prev_ids, 0, chain,
                                         (uint64_t)(n_chain + 1u) * sizeof(int32_t)) != 0;
        if (ok) for (uint32_t i = 0; i < n_chain; i++) out_ids[i] = chain[i + 1];
    }
    if (ok && out_conf && dw->confidence_proj && dw->markov_w1 && g->dspark_conf)
        ok = ds4_gpu_tensor_read(g->dspark_conf, 0, out_conf, (uint64_t)n_chain * sizeof(float)) != 0;
    if (sspan) {
        static double sp_wall = 0.0, sp_gpu = 0.0; static uint32_t sp_sn = 0;
        const float gms = ds4_gpu_span_end();
        sp_wall += (now_sec() - span_t0) * 1e3; if (gms > 0.0f) sp_gpu += gms; sp_sn++;
        if ((sp_sn & 15u) == 0)
            fprintf(stderr, "ds4: [dstep-span] n=%u avg_ms: wall=%.2f gpu=%.2f (空转=%.2f)\n",
                    sp_sn, sp_wall / sp_sn, sp_gpu / sp_sn, (sp_wall - sp_gpu) / sp_sn);
    }
    if (sprof) {
        sp_acc[3] += now_sec() - sp_t0; sp_n++;   /* markov 5 步(含逐步 host read) */
        if ((sp_n & 15u) == 0)
            fprintf(stderr, "ds4: [dstep-prof] n=%u avg_ms: pre=%.1f blocks=%.1f (attn=%.1f ffn=%.1f) head=%.1f markov=%.1f\n",
                    sp_n, sp_acc[0] * 1e3 / sp_n, sp_acc[1] * 1e3 / sp_n,
                    sp_acc[4] * 1e3 / sp_n, sp_acc[5] * 1e3 / sp_n,
                    sp_acc[2] * 1e3 / sp_n, sp_acc[3] * 1e3 / sp_n);
    }
    if (ok && getenv("DS4_DSPARK_DUMP")) {   /* e2e 对拍: anchor+pos+draft ids 落盘 */
        char p[512]; FILE *f;
        snprintf(p, sizeof(p), "%s/draft_pos%u.txt", getenv("DS4_DSPARK_DUMP"), pos);
        if ((f = fopen(p, "w"))) {
            fprintf(f, "%d", anchor_token);
            for (uint32_t i2 = 0; i2 < B; i2++) fprintf(f, " %d", out_ids[i2]);
            fclose(f);
        }
    }
    return ok;
#undef SP_MARK
}

bool metal_graph_dspark_step(
        ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights,
        const ds4_dspark_weights *dw, int anchor_token, uint32_t pos,
        int out_ids[DS4_DSPARK_BLK]) {
    return metal_graph_dspark_step_n(g, model, weights, dw, anchor_token, pos, out_ids,
                                    DS4_DSPARK_BLK, NULL);
}

#endif /* !DS4_NO_GPU */
#ifndef DS4_NO_GPU
bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool profile = getenv("DS4_METAL_GRAPH_TOKEN_PROFILE") != NULL;
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;

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
        const bool tdbg = getenv("DS4_TOK_GRAPH_DEBUG") != NULL;
        const double tp0 = tdbg ? now_sec() : 0.0;
        if (ds4_gpu_token_graph_precapture_begin() > 0) {
            const double tp1 = tdbg ? now_sec() : 0.0;
            const bool pok = metal_graph_encode_token_raw_swa(g, model, weights, 0, pos + 1u, true, true);
            const double tp2 = tdbg ? now_sec() : 0.0;
            (void)ds4_gpu_token_graph_precapture_end(pos + 1u, 1, pok ? 1 : 0);
            if (tdbg) fprintf(stderr, "[tokdbg] precap pos=%u begin=%.1f encode=%.1f end=%.1f (ms)\n",
                              pos, (tp1 - tp0) * 1e3, (tp2 - tp1) * 1e3, (now_sec() - tp2) * 1e3);
        }
    }
    const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
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
#ifndef DS4_NO_GPU
bool metal_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->mtp_n_raw = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!metal_graph_layer_is_active(g, il)) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
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

    const bool split_profile = getenv("DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    /*
     * A full long-prompt prefill can keep the GPU busy long enough for macOS
     * to watchdog WindowServer. Also split non-tiny prefills when a frontend
     * asked for display progress: completed layer command buffers are real
     * scheduling/keepalive points, while callbacks emitted while encoding one
     * huge command buffer would only be cosmetic.
     */
    const bool throttle = graph_power_throttle_enabled(g);
    const bool callback_split = display_progress != NULL && n_tokens >= 32;
    const bool direct_expert_split = metal_graph_direct_expert_read_enabled();
    const bool split_commands = direct_expert_split || split_profile || throttle || callback_split ||
                                n_tokens > 2048 || imatrix != NULL;
    const bool profile = getenv("DS4_METAL_GRAPH_PREFILL_PROFILE") != NULL || split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

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
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = getenv("DS4_METAL_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
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

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    double t_layer0 = (profile || throttle) ? now_sec() : 0.0;
    ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    const double t_embed_encoded = (profile || throttle) ? now_sec() : 0.0;
    const double t_embed_done = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: metal layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        double layer_elapsed = 0.0;
        if (split_profile) {
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            start,
                                                            n_tokens);
            if (ok) {
                ds4_gpu_tensor *tmp = g->batch_cur_hc;
                g->batch_cur_hc = g->batch_next_hc;
                g->batch_next_hc = tmp;
            }
            /* split_profile 这条内联展开路自己做 HC 交换, 绕开了 metal_graph_encode_layer_batch
             * —— 钩子必须在这里也挂一份, 否则开着剖析跑 HDUMP 会静默少写整层。 */
            if (ok) eval_hdump_batch_layer(g, il, (uint32_t)n_tokens);
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = (t_attn_done - t_attn0) + (t_ffn_done - t_ffn0);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: metal layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = (profile || throttle) ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        start,
                                                        n_tokens);
            const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = (profile || throttle) ? now_sec() : 0.0;
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = t_done - t_chunk0;
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
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
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = getenv("DS4_METAL_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *last_hc = NULL;

    const double t_head0 = profile ? now_sec() : 0.0;
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
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
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
#ifndef DS4_NO_GPU
bool metal_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_imatrix_collector *imatrix) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile = getenv("DS4_METAL_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    /* 尾批仪器(DS4_METAL_PREFILL_TAIL_BATCH=k, 2026-08-21): 前缀逐 token 灌, 最后 k 个
     * 一次成批 —— 这正是 verify 批的场景(前缀由单 token 路建, 然后一个 pos0>0 的 k 行批)。
     * 与全程 chunk=1 对跑, 同一 pos0 的 dump 差异就纯是"批 vs 单", 不含前缀污染。 */
    uint32_t tail_batch = 0;
    { const char *tb = getenv("DS4_METAL_PREFILL_TAIL_BATCH");
      if (tb && atoi(tb) > 0) tail_batch = (uint32_t)atoi(tb); }
    for (uint32_t pos0 = start; pos0 < end; ) {
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (tail_batch) local_cap = (remaining > tail_batch) ? 1u : tail_batch;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        const uint32_t chunk_end = pos0 + chunk;
        float *chunk_logits = (progress || chunk_end == end) ? logits : NULL;
        bool ok = metal_graph_prefill_layer_major(g,
                                                  model,
                                                  weights,
                                                  prompt,
                                                  pos0,
                                                  chunk,
                                                  chunk_logits,
                                                  show_progress,
                                                  imatrix,
                                                  display_progress,
                                                  display_progress_ud);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)chunk_end, prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)chunk_end, prompt->len);
        }
        pos0 = chunk_end;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}

/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
bool metal_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens <= 0) return false;
    return metal_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             display_progress,
                                             display_progress_ud,
                                             NULL);
}

/* Pick a raw SWA cache size for Metal.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
uint32_t metal_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    const char *env = getenv("DS4_METAL_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }

    return raw_cap;
}

/* Choose the prefill ubatch size.  Whole-batch is fastest for normal prompts;
 * long prompts default to 4096-token chunks. */
uint32_t metal_graph_prefill_cap_for_prompt(int prompt_len) {
    return ds4_default_prefill_cap_for_prompt(prompt_len);
}

/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  On an M3
 * Max, prefill is faster from 2-token suffixes upward; keep the default at 4
 * as a conservative crossover.  The env knob remains useful for retuning. */
uint32_t metal_graph_resume_prefill_min_tokens(void) {
    const char *env = getenv("DS4_METAL_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return 4u;
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (ds4_backend_uses_graph(backend)) {
        m.prefill_cap = metal_graph_prefill_cap_for_prompt((int)ctx);
        m.raw_cap = metal_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            m.compressed_bytes += (uint64_t)layer_comp_cap *
                                  DS4_N_HEAD_DIM *
                                  (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        uint64_t attn_stage_cap = (uint64_t)(m.prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          m.prefill_cap *
                          sizeof(float) +
                          attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    } else {
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
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

int metal_graph_prompt_logits_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                ctx_size) {
    int n_test = prompt->len;
    const char *n_test_env = getenv("DS4_METAL_GRAPH_PROMPT_TOKENS");
    if (n_test_env && n_test_env[0]) {
        char *endp = NULL;
        const long v = strtol(n_test_env, &endp, 10);
        if (endp != n_test_env && v > 0 && v <= prompt->len) n_test = (int)v;
    }

    if (n_test <= 0 || n_test > ctx_size) {
        fprintf(stderr, "ds4: Metal graph prompt test needs 1..%d prompt tokens\n", ctx_size);
        return 1;
    }

    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, (uint32_t)n_test);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, (uint32_t)n_test, false,
                                        0, (uint32_t)DS4_N_LAYER - 1u, false);
    if (!ok) {
        metal_graph_free(&g);
        fprintf(stderr, "ds4: failed to initialize Metal graph prompt test runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    ds4_kv_cache cpu_cache;
    kv_cache_init(&cpu_cache, (uint32_t)ctx_size, raw_cap);
    float *cpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *gpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *oracle_logits = NULL;

    const char *oracle_path = getenv("DS4_ORACLE_LOGITS");
    if (oracle_path && oracle_path[0]) {
        oracle_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!read_f32_binary_file(oracle_path, oracle_logits, DS4_N_VOCAB)) {
            free(oracle_logits);
            oracle_logits = NULL;
        }
    }

    for (int t = 0; t < n_test; t++) {
        const bool last = t == n_test - 1;
        forward_token_raw_swa_cpu(last ? cpu_logits : NULL,
                                  model,
                                  weights,
                                  &cpu_cache,
                                  prompt->v[t],
                                  (uint32_t)t);
    }
    ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt, n_test,
                                     gpu_logits, true, NULL, NULL);
    if (memory_report) ds4_gpu_print_memory_report("after prompt graph");

    if (ok) {
        const char *dump_gpu = getenv("DS4_METAL_GRAPH_DUMP_LOGITS");
        if (dump_gpu && dump_gpu[0]) {
            if (write_f32_binary_file(dump_gpu, gpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote Metal graph logits to %s\n", dump_gpu);
            }
        }
        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
        if (dump_cpu && dump_cpu[0]) {
            if (write_f32_binary_file(dump_cpu, cpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote CPU logits to %s\n", dump_cpu);
            }
        }
        if (getenv("DS4_METAL_GRAPH_TRACE_CACHE") != NULL ||
            getenv("DS4_METAL_GRAPH_TRACE_COMP") != NULL) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                const uint32_t n_raw = cpu_cache.layer[il].n_raw;
                if (n_raw != 0) {
                    const uint64_t raw_phys_n = (uint64_t)raw_cap * DS4_N_HEAD_DIM;
                    const uint64_t raw_logical_n = (uint64_t)n_raw * DS4_N_HEAD_DIM;
                    const uint32_t raw_start = n_raw < raw_cap ? 0u : ((uint32_t)n_test % raw_cap);
                    float *gpu_raw_phys = xmalloc((size_t)raw_phys_n * sizeof(float));
                    float *gpu_raw_logical = xmalloc((size_t)raw_logical_n * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, gpu_raw_phys, raw_phys_n * sizeof(float)) != 0) {
                        for (uint32_t r = 0; r < n_raw; r++) {
                            const uint32_t phys = (raw_start + r) % raw_cap;
                            memcpy(gpu_raw_logical + (uint64_t)r * DS4_N_HEAD_DIM,
                                   gpu_raw_phys + (uint64_t)phys * DS4_N_HEAD_DIM,
                                   (size_t)DS4_N_HEAD_DIM * sizeof(float));
                        }
                        fprintf(stderr,
                                "ds4: cache trace layer %u raw_n=%u raw_start=%u raw_max=%g raw_rms=%g\n",
                                il, n_raw, raw_start,
                                max_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n),
                                rms_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n));
                    }
                    free(gpu_raw_logical);
                    free(gpu_raw_phys);
                }

                const uint32_t n_comp = cpu_cache.layer[il].n_comp;
                if (n_comp == 0) continue;
                const uint64_t n = (uint64_t)n_comp * DS4_N_HEAD_DIM;
                float *gpu_comp = xmalloc((size_t)n * sizeof(float));
                bool comp_read = false;
                if (DS4_GPU_ATTN_COMP_CACHE_F16) {
                    uint16_t *gpu_comp_h = xmalloc((size_t)n * sizeof(uint16_t));
                    if (ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                            gpu_comp_h, n * sizeof(uint16_t)) != 0) {
                        for (uint64_t i = 0; i < n; i++) gpu_comp[i] = f16_to_f32(gpu_comp_h[i]);
                        comp_read = true;
                    }
                    free(gpu_comp_h);
                } else {
                    comp_read = ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                                    gpu_comp, n * sizeof(float)) != 0;
                }
                if (comp_read) {
                    fprintf(stderr,
                            "ds4: comp trace layer %u n=%u attn_max=%g attn_rms=%g\n",
                            il, n_comp,
                            max_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n),
                            rms_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n));
                }
                free(gpu_comp);

                const uint32_t n_index = cpu_cache.layer[il].n_index_comp;
                if (n_index != 0 && g.layer_index_comp_cache[il]) {
                    const uint64_t ni = (uint64_t)n_index * DS4_N_INDEXER_HEAD_DIM;
                    float *gpu_index = xmalloc((size_t)ni * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_index_comp_cache[il], 0, gpu_index, ni * sizeof(float)) != 0) {
                        fprintf(stderr,
                                "ds4: comp trace layer %u n=%u index_max=%g index_rms=%g\n",
                                il, n_index,
                                max_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni),
                                rms_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni));
                    }
                    free(gpu_index);
                }
            }
        }
        const uint64_t cpu_top = argmax_f32(cpu_logits, DS4_N_VOCAB);
        const uint64_t gpu_top = argmax_f32(gpu_logits, DS4_N_VOCAB);
        fprintf(stderr,
                "ds4: Metal prompt graph logits: tokens=%d logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                n_test,
                max_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                rms_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
        if (oracle_logits) {
            const uint64_t oracle_top = argmax_f32(oracle_logits, DS4_N_VOCAB);
            fprintf(stderr,
                    "ds4: oracle logits: tokens=%d oracle_top=%llu oracle_top_logit=%g cpu_max=%g cpu_rms=%g metal_max=%g metal_rms=%g\n",
                    n_test,
                    (unsigned long long)oracle_top,
                    oracle_logits[oracle_top],
                    max_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    max_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB));
        }
    } else {
        fprintf(stderr, "ds4: Metal prompt graph logits test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after prompt graph failure also failed\n");
        }
    }

    free(gpu_logits);
    free(cpu_logits);
    free(oracle_logits);
    kv_cache_free(&cpu_cache);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}

#endif /* !DS4_NO_GPU */
void embed_prompt(
        const ds4_model   * model,
        const ds4_weights * weights,
        const token_vec   * tokens,
        uint32_t            n_embd,
        float             * out) {
    for (int i = 0; i < tokens->len; i++) {
        embed_token_f16(model, weights, tokens->v[i], out + (uint64_t)i * n_embd);
    }
}

/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    token_vec_push(tv, token);
}

void ds4_tokens_free(ds4_tokens *tv) {
    token_vec_free(tv);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}


/* knowledge-MTP bridge (ds4_mtp.c): the module is tokenizer-agnostic, so the
 * engine adapts its own tokenizer to the module callback shape. Token arrays
 * are plain malloc'd (token_vec uses xrealloc), so ownership can move to the
 * module and be freed there. */
void engine_tokenize_cb(void *ctx, const char *text, int **toks, int *n) {
    ds4_engine *e = ctx;
    ds4_tokens t = {0};
    ds4_tokenize_text(e, text, &t);
    *toks = t.v;
    *n = t.len;
}

/* 前端域 enricher adapters (ds4_spatial.c / ds4_css.c): both are pure
 * closed-form derivations over the UI-sketch text, registered on the image
 * family at open. Non-sketch encoder output makes them return NULL, which
 * the registry treats as "no section". */
char *engine_mm_spatial_enrich(void *ctx, const char *modality,
                                      const char *text) {
    (void)ctx;
    (void)modality;
    return ds4_spatial_annotate(text);
}

char *engine_mm_css_enrich(void *ctx, const char *modality,
                                  const char *text) {
    (void)ctx;
    (void)modality;
    return ds4_css_annotate(text);
}

bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale) {
    return dirs && scale != 0.0f;
}

void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale) {
    if (!cpu_directional_steering_enabled(dirs, scale) || !x || rows == 0) return;

    const float *dir = dirs + (uint64_t)il * DS4_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * DS4_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            dot += xr[i] * dir[i];
        }
        const float coeff = scale * dot;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            xr[i] -= coeff * dir[i];
        }
    }
}

bool cpu_load_directional_steering(ds4_engine *e) {
    if (!e ||
        (e->directional_steering_attn_scale == 0.0f &&
         e->directional_steering_ffn_scale == 0.0f)) {
        return true;
    }

    const char *path = e->directional_steering_file;
    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    e->directional_steering_dirs = xmalloc((size_t)n * sizeof(e->directional_steering_dirs[0]));
    if (!read_f32_binary_file(path, e->directional_steering_dirs, n)) {
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    fprintf(stderr, "ds4: CPU directional steering enabled: %s attn=%g ffn=%g\n",
            path,
            (double)e->directional_steering_attn_scale,
            (double)e->directional_steering_ffn_scale);
    return true;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
/* JoyAI/DeepSeek pre-tokenization.  The split shape matters: different pieces
 * lead to different BPE merges even when the final text bytes are identical. */
static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                /*
                 * JoyAI lets a single leading space join the following word or
                 * punctuation run.  For "    int", the pre-tokenizer therefore emits
                 * "   " then " int", not "    " then "int".
                 */
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */
void vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));

    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    vocab->bos_id       = vocab_lookup(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab_lookup(vocab, "<｜end▁of▁sentence｜>");
    vocab->user_id      = vocab_lookup(vocab, "<｜User｜>");
    vocab->assistant_id = vocab_lookup(vocab, "<｜Assistant｜>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->dsml_id = vocab_lookup(vocab, "｜DSML｜");
}

void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
static void encode_chat_prompt(
        const ds4_vocab *vocab,
        const char      *system,
        const char      *prompt,
        ds4_think_mode   think_mode,
        token_vec       *out) {
    token_vec_push(out, vocab->bos_id);
    if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
    if (system && system[0]) {
        bpe_tokenize_text(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (ds4_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, token_vec *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text,
                                         token_vec *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    token_vec_push(tokens, e->vocab.bos_id);
}

void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out) {
    encode_chat_prompt(&e->vocab, system, prompt ? prompt : "", think_mode, out);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, DS4_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content) {
    ds4_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        bpe_tokenize_text(vocab, content, tokens);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(tokens, vocab->assistant_id);
        if (strncmp(content, "<think>", 7) != 0 && strncmp(content, "</think>", 8) != 0) {
            token_vec_push(tokens, vocab->think_end_id);
        }
        bpe_tokenize_text(vocab, content, tokens);
    } else {
        token_vec_push(tokens, vocab->user_id);
        if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            bpe_tokenize_text(vocab, "Tool: ", tokens);
        }
        bpe_tokenize_text(vocab, content, tokens);
    }
}

void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    token_vec_push(tokens, ds4_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}

void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

void dump_tokens(const ds4_vocab *vocab, const token_vec *tokens) {
    dump_tokens_fp(stdout, vocab, tokens);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    ds4_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int ds4_token_eos(ds4_engine *e) {
    return e->vocab.eos_id;
}

int ds4_token_user(ds4_engine *e) {
    return e->vocab.user_id;
}

int ds4_token_assistant(ds4_engine *e) {
    return e->vocab.assistant_id;
}

int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    sample_candidate *cand = xmalloc((size_t)finite * sizeof(cand[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}

/* CPU generation entry point.  It runs layer-major prefill once, then decodes
 * one token at a time using the persistent KV cache and scratch arena. */
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


static int g_ds4_lock_fd = -1;


/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

/* 取料入口 setter 家族(ds4.h 同名注释): CLI 参数是对外入口, 进程内的唯一消费点
 * 目前仍是 DS4_CAP_DIR/DS4_EVAL_* 的 getenv 读点(散在 capture/终审仪器几处),
 * 所以 setter 落到 setenv——单一事实源不变, 旗标即时生效, 不造第二条配置路径。
 * (2026-08-22 env→CLI 迁移只 land 了 CLI 半边, setter 无实现曾链接失败;
 * 消费点集中化到进程内全局属 ds4.c 拆分工序, 见重构阶段4。) */
void ds4_tool_set_cap_dir(const char *p)     { if (p) setenv("DS4_CAP_DIR", p, 1); }
const char *ds4_tool_cap_dir(void)           { return getenv("DS4_CAP_DIR"); }
void ds4_tool_set_eval_ids(const char *p)    { if (p) setenv("DS4_EVAL_IDS", p, 1); }
const char *ds4_tool_eval_ids(void)          { return getenv("DS4_EVAL_IDS"); }
void ds4_tool_set_eval_hdump(const char *p)  { if (p) setenv("DS4_EVAL_HDUMP", p, 1); }
const char *ds4_tool_eval_hdump(void)        { return getenv("DS4_EVAL_HDUMP"); }
void ds4_tool_set_eval_logits(const char *p) { if (p) setenv("DS4_EVAL_LOGITS", p, 1); }
const char *ds4_tool_eval_logits(void)       { return getenv("DS4_EVAL_LOGITS"); }
void ds4_tool_set_eval_no_bos(int v)         { if (v) setenv("DS4_EVAL_NO_BOS", "1", 1); else unsetenv("DS4_EVAL_NO_BOS"); }

const char *ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_METAL: return "metal";
    case DS4_BACKEND_CUDA:  return "cuda";
    case DS4_BACKEND_CPU:   return "cpu";
    }
    return "unknown";
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE: return "none";
    case DS4_THINK_HIGH: return "high";
    case DS4_THINK_MAX:  return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return DS4_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}


/* Per-request sampling-policy defaults. Called at session creation AND from
 * ds4_session_invalidate: the server reuses one session across requests, so a
 * stale lane / request penalty / spec-greedy flag from the previous request
 * must never leak into the next (the ds4.h contract makes frontends
 * re-declare all of them per request). repeat_gen_start goes back to -1 =
 * unmarked, same as a checkpoint rebuild. */
void session_reset_request_policy(ds4_session *s) {
    s->lane = DS4_LANE_FREE;
    s->req_freq = 0.0f;
    s->req_presence = 0.0f;
    s->spec_greedy = 1;
    s->repeat_gen_start = -1;
}

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated Metal tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > DS4_SESSION_IO_CHUNK ? DS4_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

#ifndef DS4_NO_GPU
/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical Metal tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (ds4_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read Metal F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (ds4_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}
#endif

bool ds4_session_is_cpu(const ds4_session *s) {
    return s && s->engine && s->engine->backend == DS4_BACKEND_CPU;
}

uint32_t session_cpu_raw_live_rows(const ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    uint32_t rows = ds4_default_raw_cap((uint32_t)s->ctx_size);
    if (rows > (uint32_t)s->checkpoint.len) rows = (uint32_t)s->checkpoint.len;
    return rows;
}

uint32_t session_cpu_comp_cap(const ds4_session *s) {
    if (!s) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio == 4) return layer->comp_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio != 0) return layer->comp_cap;
    }
    return (uint32_t)s->ctx_size;
}

uint64_t session_cpu_payload_live_tensor_bytes(const ds4_session *s) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_cpu_raw_live_rows(s);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;
        bytes += (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

void session_cpu_reset_cache(ds4_session *s) {
    kv_cache_free(&s->cpu_cache);
    kv_cache_init(&s->cpu_cache, (uint32_t)s->ctx_size, 0);
}

static bool ds4_layer_payload_range_valid(uint32_t layer_start, uint32_t layer_end) {
    return layer_start <= layer_end && layer_end < (uint32_t)DS4_N_LAYER;
}

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end) {
    if (!s || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end))
        return 0;
    if (ds4_session_is_cpu(s)) return 0;
#ifdef DS4_NO_GPU
    (void)layer_start;
    (void)layer_end;
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint64_t bytes = (uint64_t)DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint32_t n_layers = layer_end - layer_start + 1u;
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
#endif
}

int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload save");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before layer snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint32_t header[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        layer_start,
        layer_end,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = layer_start; rc == 0 && il <= layer_end; il++) {
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_write_tensor_span_f16_as_f32(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !tokens ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload load");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)payload_bytes;
    (void)n_tokens;
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session layer payload version");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_layer_start = h[11];
    const uint32_t saved_layer_end = h[12];
    const uint32_t saved_raw_live = h[13];
    (void)saved_prefill_cap;
    if (saved_layer_start != layer_start || saved_layer_end != layer_end) {
        payload_set_err(err, errlen, "KV shard layer range does not match requested worker");
        return 1;
    }
    if (saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens != n_tokens ||
        saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV shard does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM) {
        payload_set_err(err, errlen, "KV shard was written for a different DS4 layout");
        return 1;
    }
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV shard graph chunk layout does not match current runtime");
        return 1;
    }
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap) {
        payload_set_err(err, errlen, "KV shard raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV shard compressed cache is larger than current context");
        return 1;
    }

    const uint32_t n_layers = layer_end - layer_start + 1u;
    uint32_t *n_comp = xcalloc(n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = xcalloc(n_layers, sizeof(n_index_comp[0]));
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_comp[i] > saved_comp_cap || n_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_index_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_index_comp[i] > saved_comp_cap || n_index_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        free(n_comp);
        free(n_index_comp);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV shard restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                     g->layer_attn_comp_cache[il],
                                                     0,
                                                     (uint64_t)n_comp[i] * DS4_N_HEAD_DIM,
                                                     buf,
                                                     DS4_SESSION_IO_CHUNK,
                                                     &remaining,
                                                     err,
                                                     errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[i] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[i] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc == 0 && remaining != 0) {
        payload_set_err(err, errlen, "KV shard has trailing payload bytes");
        rc = 1;
    }
    if (rc == 0 && ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV shard restore");
        rc = 1;
    }
    if (rc == 0) {
        token_vec_free(&s->checkpoint);
        memset(&s->checkpoint, 0, sizeof(s->checkpoint));
        for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
        for (uint32_t i = 0; i < n_layers; i++) {
            const uint32_t il = layer_start + i;
            g->layer_n_comp[il] = n_comp[i];
            g->layer_n_index_comp[il] = n_index_comp[i];
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        g->mtp_n_raw = 0;
    }
    free(n_comp);
    free(n_index_comp);
    return rc;
#endif
}

int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    /* Sharded per-machine slice: layer 0 may belong to the other machine, so
     * probe the first routed-expert layer this slice actually holds. The routed
     * quant is uniform across layers, so any present layer reports the profile. */
    const ds4_tensor *gate = NULL;
    for (uint32_t il = 0; il < DS4_N_LAYER && !gate; il++) {
        gate = e->weights.layer[il].ffn_gate_exps;
    }
    /* 合一 VQ GGUF: gate 死重不入文件, routed 源=内嵌 blob(等效 go1b 档) → 报 2。 */
    if (!gate) return (e->model.residual && e->model.residual->present) ? 2 : 0;
    return gate->type == DS4_TENSOR_Q4_K ? 4 : 2;
}

/* Mode P / Mode G dynamic routing brain: a cheap, model-free heuristic that
 * classifies a user prompt as a programming task (-> resident programming model,
 * Mode P) vs everyday chat (-> full cached model, Mode G). Pure text signals so
 * the router decides BEFORE any model is loaded. Returns true for programming. */
bool ds4_prompt_is_programming(const char *prompt) {
    if (!prompt) return false;
    const size_t len = strlen(prompt);
    if (len == 0) return false;
    int score = 0;
    if (strstr(prompt, "```")) score += 6;                 /* fenced code block */
    /* Strong code signals (a single one already routes to Mode P). */
    static const char *const kw[] = {
        "def ", "function ", "class ", "import ", "return ", "public ", "private ",
        "void ", "const ", "let ", "var ", "func ", "struct ", "#include", "println",
        "console.log", "printf", "std::", "() {", ");", "=>", "->", "elif ",
        "async ", "await ", "lambda", "useState", "useEffect", "self.", "this.",
        "</", "/>", "@app", "SELECT ", "npm ", "git ", NULL};
    for (int i = 0; kw[i]; i++) if (strstr(prompt, kw[i])) score += 3;
    static const char *const ext[] = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cpp",
        ".sh", ".sql", ".html", ".css", ".json", ".yaml", NULL};
    for (int i = 0; ext[i]; i++) if (strstr(prompt, ext[i])) score += 3;
    /* Programming-intent words (EN + 中文 + frameworks). +2 each. */
    static const char *const verb[] = {
        "implement", "debug", "refactor", "compile", "stack trace", "exception",
        "syntax", "runtime", "API", "React", "Vue", "Python", "JavaScript",
        "TypeScript", "Golang", "Rust", "函数", "代码", "编译", "报错", "算法",
        "重构", "变量", "数组", "循环", "接口", "调试", "返回值", "递归", "指针",
        "编程", "脚本", "排序", "组件", "登录", "数据库", "框架", "前端", "后端",
        "正则", "并发", "异步", "类型", "继承", "封装", "bug", "方法", "对象",
        "装饰器", "闭包", "泛型", "多态", "线程", "进程", "队列", "哈希", "迭代器",
        "生成器", "协程", "序列化", "指令", "编译器", "解释器", "字节码", NULL};
    for (int i = 0; verb[i]; i++) if (strstr(prompt, verb[i])) score += 2;
    /* Symbol density: code is punctuation-heavy relative to prose. */
    size_t sym = 0;
    for (size_t i = 0; i < len; i++) {
        switch (prompt[i]) {
            case '{': case '}': case ';': case '(': case ')': case ':':
            case '[': case ']': case '<': case '>': case '=': sym++; break;
            default: break;
        }
    }
    if (sym * 100u / len >= 8u) score += 3;                 /* >= 8% symbols */
    /* Everyday prompts carry zero of these signals, so a low bar is safe. */
    return score >= 3;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}

#ifndef DS4_NO_GPU
typedef struct {
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    uint32_t mtp_n_raw;
} ds4_spec_frontier;

#endif

uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    if (s->distributed) return 0;
    if (ds4_session_is_cpu(s)) {
        uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += session_cpu_payload_live_tensor_bytes(s);
        return bytes;
    }
#ifdef DS4_NO_GPU
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
#endif
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = ds4_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_save_payload(s->distributed, s, fp, err, errlen);
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t raw_live = session_cpu_raw_live_rows(s);
        const uint32_t raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t comp_cap = session_cpu_comp_cap(s);
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            raw_cap,
            raw_cap,
            comp_cap,
            (uint32_t)s->checkpoint.len,
            DS4_N_LAYER,
            DS4_N_HEAD_DIM,
            DS4_N_INDEXER_HEAD_DIM,
            DS4_N_VOCAB,
            raw_live,
        };
        for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (int i = 0; i < s->checkpoint.len; i++) {
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_index_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (raw_live > layer->n_raw) {
                payload_set_err(err, errlen, "CPU session raw cache has fewer live rows than checkpoint");
                return 1;
            }
            const uint32_t raw_start = layer->n_raw - raw_live;
            if (payload_write_bytes(fp,
                                    layer->raw_kv + (uint64_t)raw_start * DS4_N_HEAD_DIM,
                                    (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            if (payload_write_bytes(fp,
                                    layer->attn_comp_kv,
                                    (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (ratio == 4) {
                if (payload_write_bytes(fp,
                                        layer->index_comp_kv,
                                        (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                        err,
                                        errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
            }
        }
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_write_tensor_span_f16_as_f32(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_load_payload(s->distributed, s, fp, payload_bytes, err, errlen);
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_raw_cap = h[4];
        const uint32_t saved_raw_window = h[5];
        const uint32_t saved_comp_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_raw_live = h[12];
        const uint32_t cpu_raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t cpu_comp_cap = session_cpu_comp_cap(s);
        if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint does not fit current context");
            return 1;
        }
        if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
            h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
        {
            payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
            return 1;
        }
        /* prefill_cap is scratch scheduling capacity, not durable KV layout.
         * Old checkpoints remain valid as long as the raw KV window matches. */
        (void)saved_prefill_cap;
        if (saved_raw_window != cpu_raw_cap) {
            payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
            return 1;
        }
        const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
        if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
            saved_raw_live > saved_raw_cap || saved_raw_live > cpu_raw_cap)
        {
            payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
            return 1;
        }
        if (saved_comp_cap > cpu_comp_cap) {
            payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0)
        {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        uint32_t n_comp[DS4_MAX_LAYER];
        uint32_t n_index_comp[DS4_MAX_LAYER];
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_comp[il] > saved_comp_cap || n_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
                return 1;
            }
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
                return 1;
            }
        }

        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        session_cpu_reset_cache(s);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (payload_read_bytes(fp,
                                   layer->raw_kv,
                                   (uint64_t)saved_raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            layer->n_raw = saved_raw_live;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            layer->n_comp = n_comp[il];
            layer->n_index_comp = n_index_comp[il];
            if (payload_read_bytes(fp,
                                   layer->attn_comp_kv,
                                   (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (ratio == 4) {
                if (payload_read_bytes(fp,
                                       layer->index_comp_kv,
                                       (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                       &remaining,
                                       err,
                                       errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0)
                {
                    token_vec_free(&new_checkpoint);
                    return 1;
                }
            }
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
            return 1;
        }
        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                     g->layer_attn_comp_cache[il],
                                                     0,
                                                     (uint64_t)n_comp[il] * DS4_N_HEAD_DIM,
                                                     buf,
                                                     DS4_SESSION_IO_CHUNK,
                                                     &remaining,
                                                     err,
                                                     errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

#ifndef DS4_NO_GPU
static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "ds4: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "ds4: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}

static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}
#endif

int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)dataset_path;
    (void)output_path;
    (void)ctx_size;
    (void)max_prompts;
    (void)max_tokens;
    fprintf(stderr, "ds4: imatrix collection requires a graph backend build\n");
    return 1;
#else
    if (!e || !dataset_path || !output_path) return 1;
    /* 收集器本身是后端无关的: 采集循环全在宿主侧, 回读只用 ds4_gpu_tensor_read
     * (Metal/CUDA 都实现), 三个采样张量 batch_ffn_norm / batch_router_selected /
     * batch_routed_mid 也都由共享宿主图物化。旧的 "requires --metal" 是 Mac 独占
     * 时代的遗留闸, 拆掉 (2026-08-21: 全 q2 基座第一次要在 CUDA 上吃语料量化)。 */
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) {
        fprintf(stderr, "ds4: imatrix collection requires a GPU graph backend (--metal / --cuda)\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false,
                                        0, (uint32_t)DS4_N_LAYER - 1u, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate imatrix graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;
    g.power_percent = (uint32_t)e->power_percent;

    ds4_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "ds4: failed to allocate imatrix collector\n");
        metal_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "ds4: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, DS4_MODEL_SHAPE_NAME, DS4_N_LAYER, DS4_N_EXPERT, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            ds4_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!metal_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "ds4: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = metal_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector);
                } else {
                    ok = metal_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
                if (!ok) {
                    fprintf(stderr, "ds4: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "ds4: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "ds4: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    metal_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (ds4_backend_uses_graph(e->backend)) {
#ifndef DS4_NO_GPU
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s generation requested but the graph backend is unavailable\n",
                    ds4_backend_name(e->backend));
            return 1;
        }
        return generate_metal_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->power_percent,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
#else
        fprintf(stderr, "ds4: %s generation requested but this build has no graph backend support\n",
                ds4_backend_name(e->backend));
        return 1;
#endif
    }

    return generate_raw_swa_cpu(model, vocab, weights, prompt, n_predict,
                                ctx_size,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale,
                                emit, done, emit_ud, progress, progress_ud);
}

int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_decode_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal full graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_first_token_full_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal full graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal prompt graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_prompt_logits_test(&e->model, &e->weights, prompt, ctx_size);
#else
    (void)e;
    (void)prompt;
    (void)ctx_size;
    fprintf(stderr, "ds4: Metal prompt graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: head test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = xmalloc((size_t)prompt->len * DS4_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, DS4_N_EMBD, prompt_embd);

    const uint32_t n_hc = DS4_N_HC;
    float *hc0 = xmalloc((size_t)DS4_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * DS4_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, DS4_N_EMBD);

    float *attn_norm0 = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    float *q0 = xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, DS4_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv0, DS4_N_HEAD_DIM);

    float *attn_heads = xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, DS4_N_EMBD);

    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, DS4_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *after_ffn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}

int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: first-token test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    float *hc = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(hc[0]));
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    forward_first_token_cpu(hc, model, weights, prompt->v[0]);
    print_vec_stats("first-token final_hc", hc, (uint64_t)DS4_N_HC * DS4_N_EMBD);
    output_logits_one(logits, model, weights, hc);
    print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after first-token whole-model CPU pass:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(hc);
    return 0;
}

#ifndef DS4_NO_GPU
/* project.md P2.1: register the locally-loaded routed layers' router metadata
 * (F16 gate_inp + optional exp_probs_b + expert tensor offsets) so the GPU
 * backend can re-evaluate the next layer's router on the CPU during decode and
 * read predicted experts ahead.  Only the local slice is registered: issuing
 * read-ahead for layers another machine executes would waste SSD bandwidth.
 * Purely advisory metadata; never affects inference results. */
void engine_register_layer_routers(ds4_engine *e, uint32_t start, uint32_t end) {
    if (e->model.expert_shrunken) return;   /* compact-slot models: ids differ */
    if (end >= DS4_MAX_LAYER) end = DS4_MAX_LAYER - 1;
    uint32_t registered = 0;
    for (uint32_t il = start; il <= end && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (!l->ffn_gate_exps) continue;   /* no routed MoE on this layer */
        const char *why = NULL;
        uint64_t hash_off = UINT64_MAX;
        uint32_t hash_k = 0, hash_rows = 0;
        if (!l->ffn_up_exps || !l->ffn_down_exps) why = "missing expert tensors";
        else if (!l->ffn_gate_inp) why = "no gate_inp";
        else if (l->ffn_gate_inp->type != DS4_TENSOR_F16 &&
                 l->ffn_gate_inp->type != DS4_TENSOR_F32) why = "gate_inp quantized";
        else if (l->ffn_gate_tid2eid) {
            /* Early layers hash-route by token id (tid2eid I32 [k][n_vocab]):
             * exact prediction, register the table. */
            const ds4_tensor *t = l->ffn_gate_tid2eid;
            if (t->type == DS4_TENSOR_I32 && t->ndim == 2 &&
                t->dim[0] == DS4_N_EXPERT_USED && t->dim[1] > 0) {
                hash_off = t->abs_offset;
                hash_k = (uint32_t)t->dim[0];
                hash_rows = (uint32_t)t->dim[1];
            } else {
                why = "unrecognized tid2eid layout";
            }
        }
        if (!l->ffn_gate_exps || !l->ffn_up_exps) continue;   /* 内嵌 VQ: blob 为源, 不注册 base 专家预取 */
        const uint64_t n_exp = l->ffn_gate_exps->dim[2];
        if (!why && (n_exp == 0 || n_exp != DS4_N_EXPERT)) why = "expert count";
        if (why) {
            fprintf(stderr,
                    "ds4: layer %u router not registered for prefetch (%s; gate_inp type=%u, "
                    "n_exp=%llu)\n",
                    il, why,
                    l->ffn_gate_inp ? l->ffn_gate_inp->type : 9999u,
                    (unsigned long long)n_exp);
            continue;
        }
        if (ds4_gpu_register_layer_router(e->model.map,
                                          il,
                                          l->ffn_gate_inp->abs_offset,
                                          l->ffn_gate_inp->type == DS4_TENSOR_F32,
                                          l->ffn_exp_probs_b ? l->ffn_exp_probs_b->abs_offset
                                                             : UINT64_MAX,
                                          l->ffn_gate_exps->abs_offset,
                                          l->ffn_up_exps->abs_offset,
                                          l->ffn_down_exps->abs_offset,
                                          l->ffn_gate_exps->bytes / n_exp,
                                          l->ffn_down_exps->bytes / n_exp,
                                          (uint32_t)l->ffn_gate_inp->dim[0],
                                          (uint32_t)n_exp,
                                          hash_off,
                                          hash_k,
                                          hash_rows)) {
            registered++;
        }
    }
    if (registered) {
        fprintf(stderr,
                "ds4: registered %u local routed layers for cross-layer expert prefetch\n",
                registered);
    }
}
#endif

/* ---- DS4_EVAL_IDS 终审仪器 -------------------------------------------------
 * 用途: 与量化器锚文件里同序列的 FP logits 直接对账(held 区 Σmin), 判"引擎口径
 * 还原率 vs 量化器口径"。差距大时再用 HDUMP 的逐层 hidden 找第一分歧层。
 *
 *   DS4_EVAL_IDS=<ids文件>     原始 token id 流, 空白分隔的十进制整数(每行一个亦可)
 *   DS4_EVAL_LOGITS=<out.bin>  每位置最终 logits, f32 [S][DS4_N_VOCAB], 顺序流式写盘
 *   DS4_EVAL_HDUMP=<dir>       每层出口 hidden → h_L%02d.bin(见 eval_hdump_batch_layer)
 *   DS4_EVAL_NO_BOS=1          不在流首插 BOS(默认插, 即"裸 BOS 起")
 *
 * 走的是分布式推理那条已验证的裸 token 通道(ds4_session_eval_layer_slice), 天然不过
 * chat 模板/DSML, 与量化器 embed(ids) 直喂同口径。跑完 exit(0), 不采样。
 * 内存: logits 每次只留一个位置(VOCAB f32 ≈ 0.5 MB), 写一行冲一行, 不驻留 S×VOCAB。 */
static int *eval_ids_load(const char *path, uint32_t *out_n, uint32_t vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ds4: [EVAL_IDS] 打不开 %s -- aborting\n", path);
        exit(1);
    }
    uint32_t cap = 4096, n = 0;
    int *ids = xmalloc((size_t)cap * sizeof(int));
    long v;
    while (fscanf(f, "%ld", &v) == 1) {
        if (v < 0 || (uint64_t)v >= (uint64_t)vocab) {
            fprintf(stderr, "ds4: [EVAL_IDS] %s 第 %u 个 id=%ld 越界 [0,%u) -- aborting\n",
                    path, n, v, vocab);
            exit(1);
        }
        if (n == cap) {
            cap *= 2;
            int *nb = realloc(ids, (size_t)cap * sizeof(int));
            if (!nb) { fprintf(stderr, "ds4: [EVAL_IDS] ids 扩容失败\n"); exit(1); }
            ids = nb;
        }
        ids[n++] = (int)v;
    }
    fclose(f);
    if (!n) {
        fprintf(stderr, "ds4: [EVAL_IDS] %s 里没有解析出任何 id -- aborting\n", path);
        exit(1);
    }
    *out_n = n;
    return ids;
}

/* 并发批实测(DS4_MULTI_BENCH=N, 2026-08-21): 建 N 个会话各喂不同 prompt, 先各自
 * 单独解码若干步做基准, 再用 ds4_session_eval_multi 批量推进同样的步数, 对比
 *   ①聚合 token/s(红利有多大)  ②每个会话选出的 token 序列是否与单独解码逐字相同(无损)。
 * 用 DS4_MULTI_BENCH_STEPS 调步数(默认 48)。 */
void ds4_multi_bench_run(ds4_engine *e) {
    const char *env = getenv("DS4_MULTI_BENCH");
    if (!env || !env[0]) return;
    uint32_t n = (uint32_t)atoi(env);
    if (n < 2u && !getenv("DS4_MULTI_FORCE")) n = 2u;
    if (n < 1u) n = 1u;
    if (n > 8u) n = 8u;
    uint32_t steps = 48u;
    { const char *sv = getenv("DS4_MULTI_BENCH_STEPS"); if (sv && atoi(sv) > 0) steps = (uint32_t)atoi(sv); }
    if (steps > 500u) steps = 500u;
    /* 只跑批模式(跳过单路基准): 演示 8 路并发时不必等基准 */
    const int batch_only = getenv("DS4_MULTI_BENCH_BATCH_ONLY") != NULL;

    /* 8 道真代码题(并发批处理演示用): 各自独立、长度相近, 便于横向比较 */
    static const char *prompts[8] = {
        "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Return only the code.",
        "Write a Go function `LRUCache` with Get and Put in O(1). Return only the code.",
        "Write a C function that reverses a singly linked list in place. Return only the code.",
        "Write a SQL query that returns each department's second-highest salary. Return only the query.",
        "Write a Python function `binary_search(arr, target)` returning the index or -1. Return only the code.",
        "Write a bash script that finds the 10 largest files under a directory. Return only the script.",
        "Write a JavaScript function `debounce(fn, wait)` and explain nothing. Return only the code.",
        "Write a Rust function that counts word frequencies in a string. Return only the code.",
    };
    char err[256];
    ds4_session *ss[8] = {0};
    ds4_tokens toks[8];
    memset(toks, 0, sizeof(toks));
    for (uint32_t i = 0; i < n; i++) {
        if (ds4_session_create(&ss[i], e, 4096) != 0 || !ss[i]) {
            fprintf(stderr, "ds4: [multi-bench] 会话 %u 创建失败\n", i); exit(1);
        }
        { uint32_t pi = i;
          const char *sv = getenv("DS4_MULTI_BENCH_START");
          if (sv) pi = ((uint32_t)atoi(sv) + i) & 7u;
          const char *ptext = getenv("DS4_MULTI_BENCH_SAMEPROMPT") ? prompts[0] : prompts[pi];
          /* 与 CLI 同一条渲染路: 走 chat 模板才是正经问答, 裸文本只会续写题面 */
          if (getenv("DS4_MULTI_BENCH_RAW")) ds4_tokenize_text(e, ptext, &toks[i]);
          else ds4_encode_chat_prompt(e, NULL, ptext, DS4_THINK_NONE, &toks[i]); }
        if (ds4_session_sync(ss[i], &toks[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 会话 %u prefill 失败: %s\n", i, err); exit(1);
        }
    }
    /* 基准: 各会话单独逐 token 解码 steps 步, 记下选出的 token */
    int (*ref)[512] = xmalloc((size_t)n * sizeof(*ref));
    float *ref_l1 = xmalloc((size_t)n * DS4_N_VOCAB * sizeof(float));   /* 第 1 步后的 logits */
    int (*bat)[512] = xmalloc((size_t)n * sizeof(*bat));
    double t0 = now_sec();
    for (uint32_t i = 0; !batch_only && i < n; i++) {
        for (uint32_t k = 0; k < steps; k++) {
            int best = 0; float bv = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (ss[i]->logits[v] > bv) { bv = ss[i]->logits[v]; best = (int)v; }
            ref[i][k] = best;
            if (ds4_session_eval(ss[i], best, err, sizeof err) != 0) {
                fprintf(stderr, "ds4: [multi-bench] 单路解码失败: %s\n", err); exit(1);
            }
            if (k == 0) memcpy(ref_l1 + (uint64_t)i * DS4_N_VOCAB, ss[i]->logits,
                               (size_t)DS4_N_VOCAB * sizeof(float));
        }
    }
    const double seq_s = now_sec() - t0;
    if (!batch_only)
        fprintf(stderr, "ds4: [multi-bench] 单路逐个跑: %u 会话 × %u token = %u, 用时 %.2fs ⇒ 聚合 %.2f t/s\n",
                n, steps, n * steps, seq_s, (double)(n * steps) / seq_s);

    /* 批: 重建会话到同一起点, 用 eval_multi 同步推进 */
    for (uint32_t i = 0; i < n; i++) {
        ds4_session_free(ss[i]);
        ss[i] = NULL;
        if (ds4_session_create(&ss[i], e, 4096) != 0 ||
            ds4_session_sync(ss[i], &toks[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 会话重建失败\n"); exit(1);
        }
    }
    uint32_t mismatch = 0, first_bad = 0;
    int cur[8];
    t0 = now_sec();
    for (uint32_t k = 0; k < steps; k++) {
        for (uint32_t i = 0; i < n; i++) {
            int best = 0; float bv = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (ss[i]->logits[v] > bv) { bv = ss[i]->logits[v]; best = (int)v; }
            cur[i] = best;
            bat[i][k] = best;
            if (best != ref[i][k]) { if (!mismatch) first_bad = k; mismatch++; }
        }
        if (ds4_session_eval_multi(ss, cur, n, err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 批解码失败: %s\n", err); exit(1);
        }
        if (k == 0 && !batch_only) {   /* 第 1 步 logits 与单路对账 */
            for (uint32_t i = 0; i < n; i++) {
                const float *r = ref_l1 + (uint64_t)i * DS4_N_VOCAB;
                double dmax = 0.0; int ar = 0, ab = 0; float br = -1e30f, bb = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++) {
                    const double d = fabs((double)r[v] - (double)ss[i]->logits[v]);
                    if (d > dmax) dmax = d;
                    if (r[v] > br) { br = r[v]; ar = (int)v; }
                    if (ss[i]->logits[v] > bb) { bb = ss[i]->logits[v]; ab = (int)v; }
                }
                fprintf(stderr, "ds4: [multi-bench] 会话%u(位置%d) 第1步: max|Δlogit|=%.4g 单路top1=%d 批top1=%d %s\n",
                        i, ss[i]->checkpoint.len, dmax, ar, ab, ar == ab ? "(一致)" : "(不同)");
            }
        }
    }
    const double bat_s = now_sec() - t0;
    fprintf(stderr, "ds4: [multi-bench] %u 路批处理: %u token, 用时 %.2fs ⇒ 聚合 %.2f t/s%s\n",
            n, n * steps, bat_s, (double)(n * steps) / bat_s,
            batch_only ? "" : "");
    if (!batch_only)
        fprintf(stderr, "ds4: [multi-bench] 相对单路加速 %.2fx\n", seq_s / bat_s);
    /* 质量目视: 两种模式各自的文本(逐位不同是数值等价的正常结果, 关键看是否连贯) */
    if (getenv("DS4_MULTI_BENCH_TEXT")) {
        for (uint32_t i = 0; i < n; i++) {
            static char buf[16384]; size_t off = 0;
            for (uint32_t k = 0; k < steps && off < sizeof(buf) - 64; k++) {
                size_t l = 0;
                const char *p = ds4_token_text(e, ref[i][k], &l);
                if (p && off + l < sizeof(buf) - 1) { memcpy(buf + off, p, l); off += l; }
            }
            buf[off] = 0;
            if (!batch_only) fprintf(stderr, "ds4: [multi-bench] 会话%u 单路: %s\n", i, buf);
            off = 0;
            for (uint32_t k = 0; k < steps && off < sizeof(buf) - 64; k++) {
                size_t l = 0;
                const char *p = ds4_token_text(e, bat[i][k], &l);
                if (p && off + l < sizeof(buf) - 1) { memcpy(buf + off, p, l); off += l; }
            }
            buf[off] = 0;
            fprintf(stderr, "\n===== 会话 %u =====\n%s\n", i, buf);
        }
    }
    if (!batch_only) fprintf(stderr, "ds4: [multi-bench] 与单路逐位对照: %s (不同 %u/%u%s)\n",
            mismatch ? "有差异" : "完全一致", mismatch, n * steps,
            mismatch ? "" : ", 无损");
    if (mismatch && !batch_only) fprintf(stderr, "ds4: [multi-bench] 首个不同在第 %u 步\n", first_bad);
    for (uint32_t i = 0; i < n; i++) ds4_session_free(ss[i]);
    free(ref);
    exit(0);
}

void ds4_eval_ids_run(ds4_engine *e) {
    const char *idp = getenv("DS4_EVAL_IDS");
    if (!idp || !idp[0]) return;
#ifdef DS4_NO_GPU
    (void)e;
    fprintf(stderr, "ds4: [EVAL_IDS] 需要 graph 后端 -- aborting\n");
    exit(1);
#else
    const uint32_t vocab = (uint32_t)DS4_N_VOCAB;
    uint32_t nfile = 0;
    int *file_ids = eval_ids_load(idp, &nfile, vocab);

    /* 默认在流首插 BOS(裸 BOS 起); DS4_EVAL_NO_BOS=1 则原样喂。 */
    const int no_bos = getenv("DS4_EVAL_NO_BOS") != NULL;
    const uint32_t n = no_bos ? nfile : nfile + 1u;
    int *ids = xmalloc((size_t)n * sizeof(int));
    if (no_bos) {
        memcpy(ids, file_ids, (size_t)nfile * sizeof(int));
    } else {
        ids[0] = e->vocab.bos_id;
        memcpy(ids + 1, file_ids, (size_t)nfile * sizeof(int));
    }
    free(file_ids);
    /* 首 8 个生效 id 打出来: 与锚文件对齐与否一眼可验(错位一格 Σmin 就全废)。 */
    fprintf(stderr, "ds4: [EVAL_IDS] S=%u (文件 %u%s) vocab=%u 首8: ",
            n, nfile, no_bos ? ", 无 BOS" : ", 首插 BOS", vocab);
    for (uint32_t i = 0; i < n && i < 8u; i++) fprintf(stderr, "%d ", ids[i]);
    fprintf(stderr, "\n");

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, (int)n + 64) != 0 || !s) {
        fprintf(stderr, "ds4: [EVAL_IDS] 会话创建失败 -- aborting\n");
        exit(1);
    }
    const uint32_t pcap = (uint32_t)ds4_session_prefill_cap(s);
    if (!pcap) { fprintf(stderr, "ds4: [EVAL_IDS] prefill_cap=0 -- aborting\n"); exit(1); }
    /* 不设 DS4_METAL_PREFILL_CHUNK 时 prefill_cap 等于整个 ctx(一次灌完), 那会让 hc
     * 暂存和单批显存都按 S 放大。仪器自己按 512 分块(DS4_EVAL_CHUNK 可调), 与 A3
     * 的 PREFILL_CHUNK=512 口径一致; 分块只影响批大小, 不影响数值。 */
    uint32_t cap = 512u;
    { const char *cv = getenv("DS4_EVAL_CHUNK"); if (cv && atoi(cv) > 0) cap = (uint32_t)atoi(cv); }
    if (cap > pcap) cap = pcap;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    float *hc = xmalloc((size_t)cap * hc_dim * sizeof(float));

    FILE *lf = NULL; float *lg = NULL;
    const char *lp = getenv("DS4_EVAL_LOGITS");
    if (lp && lp[0]) {
        lf = fopen(lp, "wb");
        if (!lf) {
            fprintf(stderr, "ds4: [EVAL_IDS] 打不开 %s -- aborting\n", lp);
            exit(1);
        }
        lg = xmalloc((size_t)vocab * sizeof(float));
    }

    char err[256];
    for (uint32_t p0 = 0; p0 < n; p0 += cap) {
        const uint32_t nt = (n - p0 < cap) ? (n - p0) : cap;
        /* layer_start=0 ⇒ 由 token 直接 embed(input_hc=NULL); 走到最后一层拿整批出口 HC。
         * 逐层 hidden 由 eval_hdump_batch_layer 在同一趟前向里顺带写出, 不重复前向。 */
        if (ds4_session_eval_layer_slice(s, ids + p0, nt, p0, 0, (uint32_t)DS4_N_LAYER - 1u,
                                         NULL, hc, false, NULL, err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [EVAL_IDS] prefill 失败 @pos %u: %s -- aborting\n", p0, err);
            exit(1);
        }
        if (lf) {
            /* 逐位置过输出头: eval_output_head_from_hc 只算传入批的最后一行, 所以按
             * n_tokens=1 逐位置喂该位置的 HC 隐状态。 */
            for (uint32_t t = 0; t < nt; t++) {
                if (ds4_session_eval_output_head_from_hc(s, hc + (uint64_t)t * hc_dim, 1u,
                                                         lg, err, sizeof err) != 0) {
                    fprintf(stderr, "ds4: [EVAL_IDS] 输出头失败 @pos %u: %s -- aborting\n",
                            p0 + t, err);
                    exit(1);
                }
                if (fwrite(lg, sizeof(float), vocab, lf) != vocab) {
                    fprintf(stderr, "ds4: [EVAL_IDS] logits 短写 @pos %u -- aborting\n", p0 + t);
                    exit(1);
                }
            }
            fflush(lf);
        }
        fprintf(stderr, "ds4: [EVAL_IDS] %u/%u\n", p0 + nt, n);
    }
    if (lf && fclose(lf) != 0) {
        fprintf(stderr, "ds4: [EVAL_IDS] 关闭 %s 失败 -- aborting\n", lp);
        exit(1);
    }
    fprintf(stderr, "ds4: [EVAL_IDS] 完成 S=%u vocab=%u%s%s\n", n, vocab,
            lp && lp[0] ? " logits已写" : "",
            getenv("DS4_EVAL_HDUMP") ? " hidden已写" : "");
    ds4_session_free(s);
    free(hc); free(lg); free(ids);
    exit(0);
#endif
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->distributed = opt->distributed;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    if (e->power_percent > 100) e->power_percent = 100;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = ds4_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

    bool load_slice = opt->load_slice;
    uint32_t load_layer_start = opt->load_layer_start;
    uint32_t load_layer_end = opt->load_layer_end;
    bool load_output = opt->load_output;
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE &&
        opt->distributed.layers.set)
    {
        load_slice = true;
        load_layer_start = opt->distributed.layers.start;
        load_layer_end = opt->distributed.layers.has_output ?
                         UINT32_MAX : opt->distributed.layers.end;
        load_output = opt->distributed.layers.has_output;
    }
    /* MTP drafter 拓扑整族已删除(2026-08-05)。 */
    const bool include_output_head = load_output;
    const bool mtp_keep_token_embd = false;
    const bool graph_backend = ds4_backend_uses_graph(opt->backend);
    ds4_profile_load_begin();
    /* BASE model: the only open allowed to arm go1b/go2b env defaults (the
     * MTP draft open below and all sidecar opens leave the flag false). */
    g_model_open_arm_env_defaults = true;
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    g_model_open_arm_env_defaults = false;
    if (opt->warm_weights) model_warm_weights(&e->model);
    if (!opt->inspect_only) vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    weights_bind(&e->weights, &e->model);
    dspark_bind_with_draft(&e->dspark, &e->model, graph_backend);
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }

    /* go1b "hidden variable z^L" four-loss correction sidecar. An explicit --corr
     * PATH is always honoured; otherwise, when this model uses strict-1-bit (go1b)
     * routed experts, auto-detect ds4-go1b-corr.gguf next to the -m model. Absent
     * or unreadable => exactly the pure 1-bit path (fully backward compatible). */
    {
        const char *corr_path = opt->corr_path;
        char corr_auto[1024];
        if (!corr_path || !corr_path[0]) {
            bool is_go1b = false;
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                if (e->weights.layer[il].ffn_gate_exps &&
                    e->weights.layer[il].ffn_gate_exps->type == DS4_TENSOR_GO1B) {
                    is_go1b = true;
                    break;
                }
            }
            if (is_go1b && opt->model_path) {
                const char *slash = strrchr(opt->model_path, '/');
                if (slash) {
                    size_t dlen = (size_t)(slash - opt->model_path) + 1;   /* keep '/' */
                    if (dlen < sizeof(corr_auto) - sizeof("ds4-go1b-corr.gguf")) {
                        memcpy(corr_auto, opt->model_path, dlen);
                        memcpy(corr_auto + dlen, "ds4-go1b-corr.gguf", sizeof("ds4-go1b-corr.gguf"));
                        if (access(corr_auto, R_OK) == 0) corr_path = corr_auto;
                    }
                } else if (access("ds4-go1b-corr.gguf", R_OK) == 0) {
                    snprintf(corr_auto, sizeof(corr_auto), "ds4-go1b-corr.gguf");
                    corr_path = corr_auto;
                }
            }
        }
        if (corr_path && corr_path[0]) {
            e->model.corr = corr_load(corr_path, graph_backend);
        }
        /* 1-bit residual sidecar (--residual): a second go1b layer per hot expert,
         * summed into the base expert output. Absent => single 1-bit (today). */
        {   /* --residual / DS4_RESIDUAL(2026-07-14): server 等无 CLI 旋钮的宿主经 env 挂热残差侧车 */
            const char *res_path = (opt->residual_path && opt->residual_path[0])
                                 ? opt->residual_path : getenv("DS4_RESIDUAL");
            const char *vq_dir = getenv("DS4_VQ_DIR");
            if (vq_dir && !e->model.residual) {
                e->model.residual = vq_dir_load(vq_dir);
                if (e->model.residual) res_path = NULL;
            }
            if (res_path && res_path[0])
                e->model.residual = residual_load(res_path, graph_backend);
            /* 合一 VQ GGUF: env 均未指定时, 文件自带 blob 张量即自动装载(文件即权威)。 */
            if (!e->model.residual)
                e->model.residual = vq_model_load(&e->model);
        }
        /* go-onebit 优化链: 外部 --zchain/DS4_ZCHAIN 文件优先(实验覆盖); 否则合一
         * GGUF 内嵌 blk.L.opt_* 张量(ds4.zchain.present)自动装载。GE 增益乘进
         * 路由权重 + 逐 token routed 缩放 λ(x)。都缺 => 素颜 1bit+signref 基座。 */
        {
            const char *zchain_path = opt->zchain_path && opt->zchain_path[0]
                                    ? opt->zchain_path : getenv("DS4_ZCHAIN");
            if (zchain_path && zchain_path[0]) {
                e->model.zchain = ds4_zchain_load(zchain_path, DS4_N_LAYER,
                                                  DS4_N_EXPERT, DS4_N_EMBD);
                /* 显式请求的侧车打不开/空链 => 硬失败。静默裸跑过一次假对照
                 * (+z==裸, 2026-08-20), 判决容不得兜底。 */
                if (!e->model.zchain) {
                    fprintf(stderr, "ds4: zchain %s requested but unusable -- aborting (no silent bare-model fallback)\n",
                            zchain_path);
                    exit(1);
                }
            } else {
                e->model.zchain = zchain_from_model(&e->model);
            }
        /* 第4文件(2026-08-20 用户四文件设计): drafter 反修放大器侧车 DS4_DRAFT_ZCHAIN。
         * 3 层链(mtp.0/1/2)合并进主链尾部槽 43..45 ⇒ 单 GPU 表一次上传;
         * 合并链同挂主/draft 两个 model, ffn_batch 的 zch=model->zchain 两侧都取到,
         * drafter FFN 按 il=43+b 索引。显式请求打不开 => 硬失败(无静默兜底)。 */
        {
            const char *draft_zc = getenv("DS4_DRAFT_ZCHAIN");
            if (draft_zc && draft_zc[0]) {
                if (!g_draft_model || !e->dspark.ready) {
                    fprintf(stderr, "ds4: DS4_DRAFT_ZCHAIN requires a mounted drafter (DS4_DRAFT_GGUF)\n");
                    exit(1);
                }
                struct ds4_zchain *dz = ds4_zchain_load(draft_zc, 3, DS4_N_EXPERT, DS4_N_EMBD);
                if (!dz) {
                    fprintf(stderr, "ds4: draft zchain %s unusable -- aborting\n", draft_zc);
                    exit(1);
                }
                struct ds4_zchain *base = e->model.zchain;
                struct ds4_zchain *mg = xmalloc(sizeof(*mg));
                memset(mg, 0, sizeof(*mg));
                mg->n_layer = (uint32_t)DS4_N_LAYER + 3u;
                mg->n_expert = DS4_N_EXPERT;
                mg->d_model = DS4_N_EMBD;
                mg->layer = xcalloc(mg->n_layer, sizeof(mg->layer[0]));
                if (base) {
                    memcpy(mg->layer, base->layer, (size_t)DS4_N_LAYER * sizeof(mg->layer[0]));
                    mg->n_ops_total = base->n_ops_total;
                    mg->n_ge_layers = base->n_ge_layers;
                    mg->map = base->map; mg->map_size = base->map_size;
                }
                memcpy(mg->layer + DS4_N_LAYER, dz->layer, 3u * sizeof(mg->layer[0]));
                mg->n_ops_total += dz->n_ops_total;
                mg->n_ge_layers += dz->n_ge_layers;
                /* base/dz 壳被 merged alias(单例, 进程生命周期), 不 free */
                e->model.zchain = mg;
                g_draft_model->zchain = mg;
                fprintf(stderr, "ds4: draft zchain merged: %s (3 layers @ slots 43..45)\n", draft_zc);
            } else if (g_draft_model && e->model.zchain) {
                /* 无 draft 侧车但有主侧车: drafter FFN 的 zch 取 dmodel->zchain,
                 * 保持 NULL 即旁路(主链槽 0..42 与 drafter il 43+b 互不相扰) */
                g_draft_model->zchain = NULL;
            }
        }
#ifndef DS4_NO_GPU
            if (e->model.zchain && graph_backend &&
                !zchain_gpu_upload(e->model.zchain)) {
                fprintf(stderr, "ds4: zchain GPU upload failed -- aborting (no silent quality downgrade)\n");
                exit(1);
            }
#endif
        }
        /* go-trie/ref-corpus drafter 设施已随 copy-spec/MTP 整族删除(2026-08-05 用户裁决)。 */
        /* 多模态 registry, same tokenizer bridge. The image family binds an
         * external encoder command when one is present -- resolution:
         * DS4_MM_IMAGE_CMD env, else ./mm-ui (the frontend-domain UI-sketch
         * tool, `make mm-ui`; cwd-relative like the metal shader dir, so
         * --chdir applies). Absent => image content is honestly rejected
         * upstream (server 400s image blocks instead of dropping them). */
        e->mm = ds4_mm_create(engine_tokenize_cb, e);
        if (e->mm) {
            const char *mm_cmd = getenv("DS4_MM_IMAGE_CMD");
            if ((!mm_cmd || !mm_cmd[0]) && access("mm-ui", X_OK) == 0)
                mm_cmd = "./mm-ui";
            if (mm_cmd && mm_cmd[0] &&
                ds4_mm_register_command(e->mm, "image", mm_cmd) == 0)
                fprintf(stderr, "ds4: multimodal image encoder: %s\n", mm_cmd);
            /* 前端域两插件, 顺序=节顺序: 物理方位先(关系), CSS 后(换算)。
             * 与编码器解耦: 换编码器(DS4_MM_IMAGE_CMD)后输出若非草图格式,
             * 两节自然缺席, 不伪造。 */
            ds4_mm_register_enricher(e->mm, "image", engine_mm_spatial_enrich, NULL);
            ds4_mm_register_enricher(e->mm, "image", engine_mm_css_enrich, NULL);
        }
    }
    if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    /* MTP 支持模型加载已整族删除(2026-08-05 用户裁决: Go 定型优化)。 */
#ifndef DS4_NO_GPU
    if (e->backend == DS4_BACKEND_CUDA) {
#ifdef __APPLE__
        fprintf(stderr, "ds4: CUDA backend requested but this build is linked with Metal, not CUDA\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (e->backend == DS4_BACKEND_METAL) {
#ifndef __APPLE__
        fprintf(stderr, "ds4: Metal backend requested but this build is linked with CUDA, not Metal\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (graph_backend) {
        e->metal_ready = ds4_gpu_init() != 0;
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_quality(e->quality);
        (void)ds4_gpu_set_model_fd(e->model.fd);
        /* project.md P2.2 low-cost variant: when DS4_DIST_EXPERT_FETCH_SERVE=1
         * (worker side), serve raw model-file range reads so the peer's expert
         * gather can draw from this machine's faster idle SSD over Thunderbolt. */
        (void)ds4_dist_expert_fetch_maybe_serve(e->model.fd, e->model.size);
        /* Wave 30 reverse-established variant: when the worker cannot dial out
         * (asymmetric bridge), the coordinator dials the worker's accept-mode
         * listener instead and serves preads on the dialed sockets. */
        (void)ds4_dist_expert_fetch_serve_dial(e->model.fd, e->model.size);
        int model_map_ok = 0;
        uint64_t base_l1_resident_bytes = 0;
        /* Under DS4_MTP_NO_RESIDENCY the draft model is left evictable (not
         * wired), so it must not count against the L1 resident budget gate. */
        const uint64_t mtp_l1_resident_bytes =
            (e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") == NULL) ?
            e->mtp_model.size - e->mtp_model.tensor_data_pos : 0;
        /* Dynamic resident/offload route (replaces the old hardcoded env flag):
         * keep routed experts resident -- direct GPU read, no per-layer CPU gather,
         * full decode speed -- whenever the fully-resident model fits the memory
         * budget; only stream when it would bust it. DS4_METAL_EXPERT_OFFLOAD still
         * works as an explicit override (1 = force stream, 0 = force resident). */
        const char *expert_offload_env = getenv("DS4_METAL_EXPERT_OFFLOAD");
        bool expert_offload_requested;
        if (expert_offload_env && expert_offload_env[0]) {
            expert_offload_requested =
                !(expert_offload_env[0] == '0' && expert_offload_env[1] == '\0');
        } else {
            uint64_t full_resident_bytes = 0;
            if (load_slice) {
                ds4_model_map_span_vec bb_probe, exp_probe;
                if (weights_model_map_spans_split_slice(&e->weights, load_layer_start,
                        load_layer_end, include_output_head, mtp_keep_token_embd,
                        &bb_probe, &exp_probe)) {
                    for (uint32_t i = 0; i < bb_probe.len; i++)
                        full_resident_bytes += bb_probe.v[i].end - bb_probe.v[i].off;
                    for (uint32_t i = 0; i < exp_probe.len; i++)
                        full_resident_bytes += exp_probe.v[i].end - exp_probe.v[i].off;
                    free(bb_probe.v);
                    free(exp_probe.v);
                }
            } else {
                full_resident_bytes = e->model.size - e->model.tensor_data_pos;
            }
            uint64_t auto_budget = ds4_runtime_mem_budget_bytes();
            if (auto_budget == 0) auto_budget = ds4_gpu_recommended_max_working_set_bytes();
            const uint64_t planned = full_resident_bytes + mtp_l1_resident_bytes;
            expert_offload_requested = (auto_budget > 0) && (full_resident_bytes > 0) &&
                (planned > (uint64_t)((double)auto_budget * 0.85));
            fprintf(stderr,
                    "ds4: expert-offload AUTO: full-resident %.2f GiB vs %.2f GiB budget -> %s\n",
                    (double)planned / DS4_GIB, (double)auto_budget / DS4_GIB,
                    expert_offload_requested ? "stream (offload)" : "resident (fast)");
        }
        /* VQ blob 专家不进 Metal span(ds4.c:2074): 前向唯一路径=CPU gather→f16 scratch
         * (ds4.c:2277 从 mmap vq_raw 读)。resident(offload=0)会让 MoE kernel 去 residency
         * set 直读不存在的专家 span → 崩。双机切层后 planned<budget 时 AUTO 会误选 resident,
         * 故 VQ blob 恒强制 offload。 */
        if (g_vq_experts_blob && !expert_offload_requested) {
            fprintf(stderr, "ds4: VQ blob 专家: 强制 offload(CPU gather 是唯一前向路径, 覆盖 AUTO resident)\n");
            expert_offload_requested = true;
        }
        ds4_gpu_set_expert_offload(expert_offload_requested ? 1 : 0);
        if (load_slice) {
            char load_end[32];
            if (load_output && load_layer_end == UINT32_MAX) {
                snprintf(load_end, sizeof(load_end), "output");
            } else if (load_output) {
                snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
            } else {
                snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
            }

            if (expert_offload_requested) {
                ds4_model_map_span_vec bb, exp;
                if (!weights_model_map_spans_split_slice(&e->weights,
                                                         load_layer_start,
                                                         load_layer_end,
                                                         include_output_head,
                                                         mtp_keep_token_embd,
                                                         &bb,
                                                         &exp))
                {
                    fprintf(stderr, "ds4: invalid expert-offload model load layer slice %u:%s\n",
                            load_layer_start,
                            load_end);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
                const uint32_t total = bb.len + exp.len;
                uint64_t *offsets = xmalloc((size_t)total * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)total * sizeof(sizes[0]));
                bool *resident = xmalloc((size_t)total * sizeof(resident[0]));
                uint64_t resident_bytes = 0, reclaimable_bytes = 0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < bb.len; i++) {
                    offsets[n] = bb.v[i].off;
                    sizes[n] = bb.v[i].end - bb.v[i].off;
                    resident[n] = true;
                    resident_bytes += sizes[n];
                    n++;
                }
                for (uint32_t i = 0; i < exp.len; i++) {
                    offsets[n] = exp.v[i].off;
                    sizes[n] = exp.v[i].end - exp.v[i].off;
                    resident[n] = false;
                    reclaimable_bytes += sizes[n];
                    n++;
                }
                uint64_t split_max_tensor = bb.max_tensor_bytes;
                if (exp.max_tensor_bytes > split_max_tensor) split_max_tensor = exp.max_tensor_bytes;
                base_l1_resident_bytes = resident_bytes;
                fprintf(stderr,
                        "ds4: restricting %s model map to layers %u:%s with expert offload "
                        "(%.2f GiB backbone resident, %.2f GiB routed experts reclaimable; "
                        "%u backbone + %u expert spans)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        (double)resident_bytes / 1073741824.0,
                        (double)reclaimable_bytes / 1073741824.0,
                        bb.len,
                        exp.len);
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans_split(e->model.map,
                                                                 e->model.size,
                                                                 offsets,
                                                                 sizes,
                                                                 resident,
                                                                 total,
                                                                 split_max_tensor);
                if (model_map_ok) {
                    engine_register_layer_routers(e, load_layer_start, load_layer_end);
                }
                free(offsets);
                free(sizes);
                free(resident);
                free(bb.v);
                free(exp.v);
            } else {
                ds4_model_map_span_vec spans;
                if (!weights_model_map_spans(&e->weights,
                                             load_layer_start,
                                             load_layer_end,
                                             include_output_head,
                                             mtp_keep_token_embd,
                                             &spans))
                {
                    fprintf(stderr, "ds4: invalid model load layer slice %u:%s\n",
                            load_layer_start,
                            load_end);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
                uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
                uint64_t span_bytes = 0;
                for (uint32_t i = 0; i < spans.len; i++) {
                    offsets[i] = spans.v[i].off;
                    sizes[i] = spans.v[i].end - spans.v[i].off;
                    span_bytes += sizes[i];
                }
                base_l1_resident_bytes = span_bytes;
                fprintf(stderr,
                        "ds4: restricting %s model map to layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        spans.len,
                        (double)span_bytes / 1073741824.0);
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                            e->model.size,
                                                            offsets,
                                                            sizes,
                                                            spans.len,
                                                            spans.max_tensor_bytes);
                free(offsets);
                free(sizes);
                free(spans.v);
            }
        } else if (expert_offload_requested) {
            /* Reduced-memory load: wire only the backbone (attn / shared FFN /
             * embedding / output) into the GPU residency set and keep the routed
             * experts reclaimable. The hot path still resolves every tensor's
             * buffer; cold experts just are not pinned resident. */
            ds4_model_map_span_vec bb, exp;
            if (!weights_model_map_spans_split(&e->weights, &bb, &exp)) {
                fprintf(stderr,
                        "ds4: DS4_METAL_EXPERT_OFFLOAD requested but span split failed; "
                        "falling back to the full-residency loader\n");
                base_l1_resident_bytes = e->model.size - e->model.tensor_data_pos;
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                           e->model.size,
                                                           e->model.tensor_data_pos,
                                                           e->model.size - e->model.tensor_data_pos,
                                                           e->model.max_tensor_bytes);
            } else {
                const uint32_t total = bb.len + exp.len;
                uint64_t *offsets = xmalloc((size_t)total * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)total * sizeof(sizes[0]));
                bool *resident = xmalloc((size_t)total * sizeof(resident[0]));
                uint64_t resident_bytes = 0, reclaimable_bytes = 0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < bb.len; i++) {
                    offsets[n] = bb.v[i].off;
                    sizes[n] = bb.v[i].end - bb.v[i].off;
                    resident[n] = true;
                    resident_bytes += sizes[n];
                    n++;
                }
                for (uint32_t i = 0; i < exp.len; i++) {
                    offsets[n] = exp.v[i].off;
                    sizes[n] = exp.v[i].end - exp.v[i].off;
                    resident[n] = false;
                    reclaimable_bytes += sizes[n];
                    n++;
                }
                uint64_t split_max_tensor = bb.max_tensor_bytes;
                if (exp.max_tensor_bytes > split_max_tensor) split_max_tensor = exp.max_tensor_bytes;
                fprintf(stderr,
                        "ds4: expert-offload model map: %.2f GiB backbone resident, "
                        "%.2f GiB routed experts reclaimable (%u backbone + %u expert spans)\n",
                        (double)resident_bytes / 1073741824.0,
                        (double)reclaimable_bytes / 1073741824.0,
                        bb.len, exp.len);
                base_l1_resident_bytes = resident_bytes;
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                if (getenv("DS4_METAL_EXPERT_OFFLOAD_DEBUG") != NULL) {
                    for (uint32_t i = 0; i < total; i++) {
                        fprintf(stderr, "ds4:   span[%u] %s %.4f..%.4f GiB (%.1f MiB)\n",
                                i, resident[i] ? "RES" : "exp",
                                (double)offsets[i] / 1073741824.0,
                                (double)(offsets[i] + sizes[i]) / 1073741824.0,
                                (double)sizes[i] / 1048576.0);
                    }
                }
                model_map_ok = ds4_gpu_set_model_map_spans_split(e->model.map,
                                                                 e->model.size,
                                                                 offsets,
                                                                 sizes,
                                                                 resident,
                                                                 total,
                                                                 split_max_tensor);
                if (model_map_ok) {
                    engine_register_layer_routers(e, 0, DS4_MAX_LAYER - 1);
                }
                free(offsets);
                free(sizes);
                free(resident);
                free(bb.v);
                free(exp.v);
            }
        } else {
            base_l1_resident_bytes = e->model.size - e->model.tensor_data_pos;
            ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
            model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                       e->model.size,
                                                       e->model.tensor_data_pos,
                                                       e->model.size - e->model.tensor_data_pos,
                                                       e->model.max_tensor_bytes);
        }
        if (!model_map_ok) {
            fprintf(stderr,
                    "ds4: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        /* DS4_MTP_NO_RESIDENCY: wrap the draft model's views evictable (not
         * wired) so they do not pin the worker's GPU residency budget. */
        const bool mtp_nonresident = e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") != NULL;
        if (mtp_nonresident) ds4_gpu_set_model_map_nonresident_hint(1);
        if (e->mtp_ready &&
            !ds4_gpu_set_model_map_range(e->mtp_model.map,
                                           e->mtp_model.size,
                                           e->mtp_model.tensor_data_pos,
                                           e->mtp_model.size - e->mtp_model.tensor_data_pos,
                                           e->mtp_model.max_tensor_bytes))
        {
            if (mtp_nonresident) ds4_gpu_set_model_map_nonresident_hint(0);
            fprintf(stderr,
                    "ds4: %s failed to map MTP model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (mtp_nonresident) ds4_gpu_set_model_map_nonresident_hint(0);
        /* 副 drafter map 注册必须在主模型 map 之后: ds4_gpu_set_model_map 换 base 时
         * release_all 会连带释放先注册的 range(2026-08-21 实测顺序坑)。 */
        /* 副 map 注册失败 => drafter 读到的是不可用的裸指针(实测 logits NaN, 草稿全废、
         * acc 塌到 1.00 而速度看似"正常")。明确停用 drafter, 不接受静默劣化。 */
        /* ★默认关(2026-08-21 实测): 副 map 整体 cudaHostRegister + HostGetDevicePointer
         * 在 5.59GB 文件映射上不可靠 —— drafter 同输入两跑 MoE 输出不同(Fin/路由/权重
         * 逐字节相同), 关掉即完全确定; 注册前把页全部触实也无效。改走 cuda_model_range_ptr
         * 的按 range 懒注册(页对齐分段, 已验证确定)。DS4_AUX_REG=1 可强开做对照。 */
        if (g_draft_model && getenv("DS4_NO_AUX_REG") == NULL &&
            !ds4_gpu_register_aux_model_map(g_draft_model->map, g_draft_model->size)) {
            fprintf(stderr, "ds4: draft gguf aux map registration failed -- disabling drafter "
                            "(speculation off; plain decode continues)\n");
            e->dspark.ready = false;
            g_dspark_ready_global = 0;
            g_dspark_bound_for_prefill = NULL;
        }
        if (!accelerator_cache_model_tensors(e->backend, &e->model)) {
            fprintf(stderr, "ds4: %s failed to prepare startup model cache\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        /* Also populate the HBM cache for the MTP support model when loaded.
         * Without this, MTP-block tensor reads at decode time hit the UVA-
         * mapped pointer (slow) instead of cudaMalloc'd HBM copies (fast).
         * The MoE expert filter in accelerator_cache_model_tensor_spans
         * skips `mtp.0.ffn_*_exps.weight` automatically.
         *
         * DS4_MTP_NO_RESIDENCY=1: skip wiring the MTP draft model into the GPU
         * residency set.  On Metal the draft tensors stay no-copy mmap shared
         * buffers (read directly, not slow -- same as the main model), just
         * evictable.  This frees the ~3.8 GiB the draft would otherwise pin,
         * letting the memory-tight worker (M1, ~10.67 GiB GPU budget) hold its
         * layer-slice backbone + the draft on the default fast split instead of
         * OOMing ("residency wired 51 views" / kIOGPUCommandBufferCallbackError
         * OutOfMemory).  The draft forward pages the (hot, every-step) tensors
         * back via the page cache. */
        if (e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") == NULL &&
            !accelerator_cache_model_tensors(e->backend, &e->mtp_model)) {
            fprintf(stderr, "ds4: %s failed to prepare MTP startup model cache\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") != NULL) {
            fprintf(stderr,
                    "ds4: MTP draft model left non-resident (DS4_MTP_NO_RESIDENCY=1); "
                    "draft tensors stay evictable mmap views to save GPU residency\n");
        }
        fprintf(stderr, "ds4: %s backend initialized for graph diagnostics\n",
                ds4_backend_name(e->backend));
    }
#else
    if (graph_backend) {
        fprintf(stderr, "ds4: %s backend requested but this build has no graph backend support; aborting startup\n",
                ds4_backend_name(e->backend));
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
#endif

    ds4_profile_load_end();
    ds4_multi_bench_run(e);   /* DS4_MULTI_BENCH=N: 并发批实测(跑完退出) */
    ds4_eval_ids_run(e);   /* DS4_EVAL_IDS 在场则跑完仪器直接退出, 不返回 */
    *out = e;
    return 0;
}

void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
}

int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

int ds4_engine_power(ds4_engine *e) {
    return e ? e->power_percent : 100;
}

/* R3-h multi-domain sidecar plugin: swap the corr between generations. Load
 * the NEW sidecar first so a bad path keeps the current domain intact. */
int ds4_engine_corr_switch(ds4_engine *e, const char *path) {
    if (!e) return -1;
    struct ds4_corr *next = NULL;
    if (path && path[0]) {
        next = corr_load(path, e->backend == DS4_BACKEND_METAL);
        if (!next) return -1;
    }
    if (e->model.corr) corr_free(e->model.corr);
    e->model.corr = next;
    return 0;
}

int ds4_engine_set_power(ds4_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}

int ds4_engine_layer_count(ds4_engine *e) {
    (void)e;
    return (int)DS4_N_LAYER;
}

uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer) {
    (void)e;
    if (layer >= DS4_N_LAYER) return 0;
    return ds4_layer_compress_ratio(layer);
}

uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) {
    (void)e;
    return (uint64_t)DS4_N_HC * DS4_N_EMBD;
}

int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return (int)DS4_MODEL_VARIANT;
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    ds4_mm_free(e->mm);
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    if (e->mtp_ready) model_close(&e->mtp_model);
    model_close(&e->model);
#ifndef DS4_NO_GPU
    ds4_gpu_cleanup();
#endif
    ds4_release_instance_lock();
    ds4_dist_tp_free(e->tp);
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}

/* TP run loops (ds4_distributed.c) reach the engine's peer link through this. */
ds4_dist_tp *ds4_engine_tp(ds4_engine *e) { return e ? e->tp : NULL; }

/* Server /v1/messages image blocks reach the modality registry through this. */
ds4_mm *ds4_engine_mm(ds4_engine *e) { return e ? e->mm : NULL; }

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (e->backend == DS4_BACKEND_CPU) {
        if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
            fprintf(stderr, "ds4: distributed coordinator sessions require the graph backend\n");
            return 1;
        }
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = e;
        session_reset_request_policy(s);
        s->ctx_size = ctx_size;
        s->prefill_cap = ds4_default_prefill_cap_for_prompt(ctx_size);
        kv_cache_init(&s->cpu_cache, (uint32_t)ctx_size, 0);
        cpu_decode_scratch_init(&s->cpu_scratch, (uint32_t)ctx_size);
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        *out = s;
        return 0;
    }
#ifdef DS4_NO_GPU
    return 1;
#else
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    session_reset_request_policy(s);
    s->ctx_size = ctx_size;
    s->prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    const char *dist_prefill_env = getenv("DS4_DIST_PREFILL_CAP");
    if (dist_prefill_env && dist_prefill_env[0] &&
        e->distributed.role != DS4_DISTRIBUTED_NONE && e->distributed.layers.set) {
        char *endp = NULL;
        unsigned long v = strtoul(dist_prefill_env, &endp, 10);
        if (endp != dist_prefill_env && v > 0 && v <= (unsigned long)ctx_size) {
            s->prefill_cap = (uint32_t)v;
        }
    }
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    bool active_slice = false;
    uint32_t active_start = 0;
    uint32_t active_end = (uint32_t)DS4_N_LAYER - 1u;
    if (e->distributed.role != DS4_DISTRIBUTED_NONE && e->distributed.layers.set) {
        active_slice = true;
        active_start = e->distributed.layers.start;
        active_end = e->distributed.layers.has_output ?
                     ((uint32_t)DS4_N_LAYER - 1u) : e->distributed.layers.end;
    }
    s->graph.dspark_capture = e->dspark.ready ? 1 : 0;
    if (!metal_graph_alloc_raw_cap(&s->graph, &e->weights, &e->weights.layer[0],
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap, e->mtp_ready,
                                   active_start, active_end, active_slice))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    s->graph.power_percent = (uint32_t)e->power_percent;
    if (e->distributed.tp_enabled) {
        /* Establish the TP peer link once per engine. By default the coordinator
         * listens and the worker connects. DS4_TP_REVERSE_CONNECT=1 flips the
         * network roles (coordinator connects, worker listens) to work around a
         * host where one direction's connect() fails (observed: an M1 where ds4's
         * outbound connect returns EHOSTUNREACH while nc/plain connect succeed).
         * The TP all-reduce is a symmetric sum, so connect direction does not
         * affect results; tp_owns_low stays tied to role, not to who listens.
         * Blocks until both peers are up; KB-level buffers only. */
        if (!e->tp) {
            char terr[256] = {0};
            const char *rev = getenv("DS4_TP_REVERSE_CONNECT");
            bool reverse = (rev && *rev && rev[0] != '0');
            bool coordinator = (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR);
            bool i_listen = reverse ? !coordinator : coordinator;
            if (i_listen) {
                e->tp = ds4_dist_tp_listen(e->distributed.listen_host,
                                           e->distributed.listen_port, terr, sizeof(terr));
            } else {
                e->tp = ds4_dist_tp_connect(e->distributed.coordinator_host,
                                            e->distributed.coordinator_port, terr, sizeof(terr));
            }
            e->tp_owns_low = coordinator;
            if (!e->tp) {
                fprintf(stderr, "ds4: TP peer connection failed: %s\n", terr);
                metal_graph_free(&s->graph);
                free(s);
                return 1;
            }
        }
        s->graph.tp = e->tp;
        s->graph.tp_layers = e->distributed.tp_layers ? e->distributed.tp_layers : UINT32_MAX;
        s->graph.tp_owns_low = e->tp_owns_low;
        s->graph.tp_vec = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    }
    if (!metal_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        metal_graph_free(&s->graph);
        free(s);
        return 1;
    }
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
        s->mtp_draft_token = -1;
    }
    if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR && !e->distributed.tp_enabled) {
        char err[256];
        if (ds4_dist_session_create(&s->distributed,
                                    e,
                                    &e->distributed,
                                    s,
                                    ctx_size,
                                    err,
                                    sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: failed to create distributed coordinator session: %s\n",
                    err[0] ? err : "unknown error");
            metal_graph_free(&s->graph);
            free(s->logits);
            free(s->mtp_logits);
            free(s);
            return 1;
        }
    }
    *out = s;
    return 0;
#endif
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    ds4_dist_session_free(s->distributed);
    if (ds4_session_is_cpu(s)) {
        kv_cache_free(&s->cpu_cache);
        cpu_decode_scratch_free(&s->cpu_scratch);
    }
#ifndef DS4_NO_GPU
    else {
        metal_graph_free(&s->graph);
    }
#endif
    token_vec_free(&s->checkpoint);
    free(s->logits);
    free(s->mtp_logits);
    free(s);
}

int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen) {
    if (!s || !s->distributed) {
        if (errlen) snprintf(err, errlen, "session is not a distributed coordinator");
        return -1;
    }
    return ds4_dist_session_route_ready(s->distributed, err, errlen);
}

int ds4_session_power(ds4_session *s) {
    if (!s || !s->engine) return 100;
    return s->engine->power_percent;
}

bool ds4_session_is_distributed(ds4_session *s) {
    return s && s->distributed != NULL;
}

int ds4_session_set_power(ds4_session *s, int power_percent) {
    if (!s || !s->engine || power_percent < 1 || power_percent > 100) return 1;
    s->engine->power_percent = power_percent;
#ifndef DS4_NO_GPU
    if (!ds4_session_is_cpu(s)) s->graph.power_percent = (uint32_t)power_percent;
#endif
    return 0;
}

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total) {
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}

int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen) {
    if (!s) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    ds4_session_invalidate(s);
    if (ds4_session_is_cpu(s)) {
        session_cpu_reset_cache(s);
        return 0;
    }
#ifdef DS4_NO_GPU
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        if (errlen) snprintf(err, errlen, "%s layer-slice state reset failed",
                             ds4_backend_name(s->engine->backend));
        return 1;
    }
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen) {
    if (!s || !s->engine || !hidden_hc || n_tokens == 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid output-head hidden-state input");
        return 1;
    }

    ds4_engine *e = s->engine;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const float *last_hc = hidden_hc + (uint64_t)(n_tokens - 1u) * hc_dim;

    if (ds4_session_is_cpu(s)) {
        output_logits_one(logits, &e->model, &e->weights, last_hc);
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)e;
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_tensor_write(g->cur_hc,
                                   0,
                                   last_hc,
                                   hc_dim * sizeof(float)) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head(g,
                                                &e->model,
                                                &e->weights,
                                                e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(g->logits,
                                     0,
                                     logits,
                                     (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after output-head hidden-state failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s output-head hidden-state evaluation failed",
                             ds4_backend_name(e->backend));
        return 1;
    }
    return 0;
#endif
}

int ds4_session_slice_check_timeline(
        ds4_session *s,
        const int   *tokens,
        uint32_t     n_tokens,
        uint32_t     pos0,
        char        *err,
        size_t       errlen) {
    if (!s || !tokens || n_tokens == 0) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice token span");
        return 1;
    }
    const uint32_t ctx_size = (uint32_t)s->ctx_size;
    if (pos0 > (uint32_t)INT_MAX || n_tokens > (uint32_t)INT_MAX ||
        pos0 > ctx_size || n_tokens > ctx_size - pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice token span exceeds context");
        return 1;
    }
    if (!s->checkpoint_valid) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "layer-slice session needs reset before pos %u", pos0);
            return 1;
        }
        return 0;
    }
    if ((uint32_t)s->checkpoint.len != pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice KV position mismatch: have %d want %u",
                             s->checkpoint.len, pos0);
        return 1;
    }
    return 0;
}

DS4_MAYBE_UNUSED void ds4_session_slice_commit_timeline(ds4_session *s, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
}

void ds4_session_invalidate(ds4_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    /* Also drops lane/request-penalty/spec-greedy back to defaults (and
     * repeat_gen_start to -1): an invalidated checkpoint means the next
     * request re-renders and re-declares its policy from scratch. */
    session_reset_request_policy(s);
    s->mtp_draft_valid = false;
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
    /* 冷回卷(pos==0)= 从头重放: 必须连 comp/indexer 计数与压缩器累积 state 一起清。
     * 此前只截 token 时间线 ⇒ 连跑多题时 layer_n_comp 只涨不回, 2050 行容量在
     * ~28 题后溢出("compressed KV cache capacity exceeded", 2026-08-18 328 题
     * 基准 500 连锁的第一层根因)。部分回卷(pos>0)的 comp 精确回滚仍是已知债
     * (state 含非边界脏贡献), CC 增量场景语义不变。 */
#ifndef DS4_NO_GPU
    if (pos == 0) (void)metal_graph_reset_prefill_state(&s->graph);
#endif
}

int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}

int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}

int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}

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

#ifndef DS4_NO_GPU
typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        p->session->repeat_gen_start = -1;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->mtp_draft_valid = false;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}
#endif

/* Bring the live backend state to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (s->distributed) {
        const ds4_tokens *checkpoint = s->checkpoint_valid ? &s->checkpoint : NULL;
        return ds4_dist_session_sync(s->distributed,
                                     s,
                                     checkpoint,
                                     prompt,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint))
        {
            s->mtp_draft_valid = false;
            for (int i = s->checkpoint.len; i < prompt->len; i++) {
                forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                         &e->model,
                                                         &e->weights,
                                                         &s->cpu_cache,
                                                         prompt->v[i],
                                                         (uint32_t)s->checkpoint.len,
                                                         e->directional_steering_dirs,
                                                         e->directional_steering_attn_scale,
                                                         e->directional_steering_ffn_scale,
                                                         &s->cpu_scratch);
                token_vec_push(&s->checkpoint, prompt->v[i]);
                if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i + 1, prompt->len);
            }
            s->checkpoint_valid = true;
            return 0;
        }

        session_cpu_reset_cache(s);
        prefill_layer_major_cpu(s->logits,
                                &e->model,
                                &e->weights,
                                &s->cpu_cache,
                                prompt,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale);
        ds4_tokens_copy(&s->checkpoint, prompt);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", prompt->len, prompt->len);
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)prompt;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    const char *backend_name = ds4_backend_name(e->backend);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint))
    {
        s->mtp_draft_valid = false;
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = metal_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            ds4_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            ds4_session_progress_fn progress_fn =
                s->progress ? ds4_session_note_prefill_progress : NULL;
            bool ok = metal_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        progress_fn,
                                                        progress_fn ? &progress : NULL,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL);
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    if (s->prefill_cap < (uint32_t)prompt->len) {
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ds4_session_progress_fn progress_fn =
            s->progress ? ds4_session_note_prefill_progress : NULL;
        ok = metal_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         progress_fn, progress_fn ? &progress : NULL,
                                         s->display_progress,
                                         s->display_progress_ud);
    } else {
        ok = metal_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud);
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!g_prof.enabled) return ds4_session_sync_internal(s, prompt, err, errlen);
    /* Tokens actually prefilled = prompt length minus the matching live prefix.
     * Captured before the call because the internal sync mutates the checkpoint. */
    const int start_len = (s && s->checkpoint_valid &&
                           prompt && prompt->len >= s->checkpoint.len &&
                           ds4_tokens_starts_with(prompt, &s->checkpoint))
                          ? s->checkpoint.len : 0;
    const double t0 = now_sec();
    int rc = ds4_session_sync_internal(s, prompt, err, errlen);
    if (rc == 0 && prompt && prompt->len > start_len) {
        ds4_profile_add_prefill((uint64_t)(prompt->len - start_len), now_sec() - t0);
    }
    return rc;
}

/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}


int ds4_session_argmax(ds4_session *s) {
    /* Non-FREE lanes see raw logits by contract (ds4.h): skip the scratch
     * copy entirely -- tool-syntax/copy emission sit in the hot decode loop
     * and repeat_penalize_buf would be a no-op for them anyway. */
    if (s && s->lane != DS4_LANE_FREE)
        return sample_argmax(s->logits, DS4_N_VOCAB);
    if (session_penalties_active(s) && s && s->checkpoint_valid && s->checkpoint.len > 0) {
        /* penalized greedy must match ds4_session_sample(temp 0); work on a
         * scratch copy so diagnostic readers of s->logits stay unpolluted */
        static float *scratch = NULL;
        if (!scratch) scratch = xmalloc((size_t)DS4_MAX_VOCAB * sizeof(scratch[0]));
        memcpy(scratch, s->logits, (size_t)DS4_N_VOCAB * sizeof(scratch[0]));
        repeat_penalize_buf(s, scratch, (uint32_t)s->checkpoint.len);
        return sample_argmax(scratch, DS4_N_VOCAB);
    }
    return sample_argmax(s->logits, DS4_N_VOCAB);
}

/* Raw-logits argmax excluding one id -- deliberately penalty-free, unlike
 * ds4_session_argmax: its only caller today is ds4-bench forced continuation
 * (exclude EOS to keep generating), which wants the model's unmodified
 * second choice for stable speed measurement, not a sampling path. */
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature, top_k, top_p, min_p, rng);
}

/* Session-aware penalty-activity gate: env-armed penalties (loop break /
 * DS4_REPEAT_FREQ) plus this session's per-request penalties -- a session
 * carrying only req_freq/req_presence must still take the penalized
 * argmax/anticycle-prep paths. Non-FREE lanes report inactive: the lane
 * contract (ds4.h) bypasses every penalty, so callers skip the scratch work
 * outright. */
/* 惩罚只剩客户端显式传的 per-request(OpenAI frequency/presence)。引擎自造的
 * env 惩罚(DS4_REPEAT_FREQ)、断环器、熵门整族已删(2026-08-21 用户铁律:
 * 引擎不得擅自修改模型输出, 默认路径=裸模型真值)。 */
int session_penalties_active(const ds4_session *s) {
    if (s && s->lane != DS4_LANE_FREE) return 0;
    return s && (s->req_freq != 0.0f || s->req_presence != 0.0f);
}

void repeat_penalize_buf(ds4_session *s, float *logits, uint32_t end) {
    if (!s || !logits || end == 0) return;
    /* Lane gate (ds4.h contract): non-FREE lanes get raw logits, no penalty
     * of any kind — a penalty-diverted token corrupts forced tool-call
     * syntax. */
    if (s->lane != DS4_LANE_FREE) return;
    /* Unmarked (-1) → 0 = whole context, NOT the prompt boundary (end).
     * Pinning to `end` excluded the prompt from the repeat window, which left
     * only the few generated tokens penalized early in generation → far too
     * weak for the fragile 2-bit model, so greedy decode drifted/looped. The
     * distributed coordinator's session starts invalidated
     * (repeat_gen_start=-1) and hit this path → dual-host drifted where
     * single-host (which stays at 0) wrote clean code (2026-07-06 root-cause:
     * dual "bug" was this single-vs-dist gen_start mismatch, not cross-GPU
     * fp). 0 keeps both paths identical for unmarked sessions. NEW (lane
     * era): frontends now mark the real boundary via
     * ds4_session_mark_generation_start / _set_generation_start, which scopes
     * the anticycle bans + request penalties below to the generated region so
     * quoting the prompt is never banned; the env freq window keeps ignoring
     * the boundary either way (see repeat_penalize_core). */
    if (s->repeat_gen_start < 0 || s->repeat_gen_start > (int)end)
        s->repeat_gen_start = 0;
    const uint32_t gstart = (uint32_t)s->repeat_gen_start;
    /* Per-request OpenAI penalties, stacked on top of the env freq penalty:
     * logits[t] -= req_freq*count(t) + (count(t)>0 ? req_presence : 0), with
     * count over the generated region only. Negative values are legal and
     * ADD probability (OpenAI [-2,2]). The `seen` scratch marks first
     * occurrences and is wiped by re-walking the same range, so cost stays
     * O(region), not O(vocab); single graph worker => static is safe (same
     * discipline as the ds4_session_argmax scratch). */
    if (s->req_freq != 0.0f || s->req_presence != 0.0f) {
        static uint8_t *seen = NULL;
        if (!seen) seen = xcalloc((size_t)DS4_MAX_VOCAB, sizeof(seen[0]));
        for (uint32_t i = gstart; i < end; i++) {
            const int t = s->checkpoint.v[i];
            if (t < 0 || t >= (int)DS4_N_VOCAB) continue;
            logits[t] -= s->req_freq;
            if (!seen[t]) { seen[t] = 1; logits[t] -= s->req_presence; }
        }
        for (uint32_t i = gstart; i < end; i++) {
            const int t = s->checkpoint.v[i];
            if (t >= 0 && t < (int)DS4_N_VOCAB) seen[t] = 0;
        }
    }
}

/* ---- Sampling-lane / per-request policy (contract in ds4.h) ----
 * Pure session-state setters/getters; the semantics live in
 * repeat_penalize_buf / session_penalties_active above. All NULL-tolerant:
 * frontends call them unconditionally on paths where the session may not
 * exist yet. */
void ds4_session_set_lane(ds4_session *s, int lane) {
    if (!s) return;
    /* Unknown lane ids fall back to FREE (penalties active): the safe default
     * is current behavior, not an accidental penalty bypass. */
    if (lane != DS4_LANE_TOOL_SYNTAX && lane != DS4_LANE_COPY_EMISSION)
        lane = DS4_LANE_FREE;
    s->lane = lane;
}

int ds4_session_lane(const ds4_session *s) {
    return s ? s->lane : DS4_LANE_FREE;
}

/* Pin the generation boundary at the live checkpoint length: call after the
 * prompt is fully prefilled, before the first sampled token of a response. */
void ds4_session_mark_generation_start(ds4_session *s) {
    if (!s) return;
    s->repeat_gen_start = s->checkpoint.len;
}

/* Explicit boundary for rebuilds whose checkpoint already contains generated
 * tokens (server re-sync of a transcript with a known assistant tail). */
void ds4_session_set_generation_start(ds4_session *s, int pos) {
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->repeat_gen_start = pos;
}

void ds4_session_set_request_penalties(ds4_session *s, float freq, float presence) {
    if (!s) return;
    s->req_freq = freq;
    s->req_presence = presence;
}

void ds4_session_set_spec_greedy(ds4_session *s, int greedy_ok) {
    if (!s) return;
    s->spec_greedy = greedy_ok ? 1 : 0;
}

int ds4_session_spec_greedy_ok(const ds4_session *s) {
    return s ? s->spec_greedy : 1;
}

static void session_apply_repeat_penalty(ds4_session *s) {
    if (!s || !s->logits || !s->checkpoint_valid || s->checkpoint.len <= 0) return;
    repeat_penalize_buf(s, s->logits, (uint32_t)s->checkpoint.len);
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (getenv("DS4_DECODE_DIAG")) {
        /* [decode-diag] inspect the logits the sampler is about to draw from:
         * argmax + its value + how many entries are non-finite (NaN/inf => the
         * distributed worker returned garbage logits rather than a real head). */
        int argmax = 0;
        float amv = s->logits[0];
        uint32_t nonfinite = 0;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
            const float v = s->logits[i];
            if (!isfinite(v)) { nonfinite++; continue; }
            if (v > amv) { amv = v; argmax = (int)i; }
        }
        fprintf(stderr,
                "ds4: [decode-diag] logits argmax=%d val=%.4f nonfinite=%u/%u\n",
                argmax, amv, nonfinite, (unsigned)DS4_N_VOCAB);
    }
    session_apply_repeat_penalty(s);
    return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)DS4_N_VOCAB) return 0;

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : DS4_NEG_INF;
    return 1;
}

int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    if (!s || !out || cap < (int)DS4_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)DS4_N_VOCAB * sizeof(out[0]));
    return (int)DS4_N_VOCAB;
}

int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    if (!s || !logits || n != (int)DS4_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}

int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
    if (!s) return 1;
    if (s->distributed) {
        if (!s->checkpoint_valid) {
            if (errlen) snprintf(err, errlen, "distributed decode requires a valid checkpoint");
            return 1;
        }
        (void)probe_mtp;
        return ds4_dist_session_eval(s->distributed,
                                     s,
                                     &s->checkpoint,
                                     token,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                 &e->model,
                                                 &e->weights,
                                                 &s->cpu_cache,
                                                 token,
                                                 (uint32_t)s->checkpoint.len,
                                                 e->directional_steering_dirs,
                                                 e->directional_steering_attn_scale,
                                                 e->directional_steering_ffn_scale,
                                                 &s->cpu_scratch);
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        (void)probe_mtp;
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)token;
    (void)probe_mtp;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    /* MTP probe 已整族删除(2026-08-05)。 */
    (void)probe_mtp;
    if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    /* MTP draft 已整族删除(2026-08-05)。 */
    return 0;
#endif
}

/* ===== 请求批处理: 多会话共享一次前向 (2026-08-21) =====================
 * 为什么值得做: 本机是带宽墙 —— 每 token 要把 4.64GB 权重从 LPDDR5x 搬进计算单元
 * (骨干 2.19GB + 该 token 路由到的 6 个专家/层 2.10GB + 输出头 0.35GB), 29ms/token
 * 就是这么来的。多路请求各跑各的, 这 4.64GB 要各读一遍; 拼进同一次前向, 行无关的
 * 部分只读一遍。实测批路径每行成本: 1 行 35.9ms / 2 行 26.4 / 4 行 18.9 / 8 行 14.9
 * ⇒ 8 路聚合约 1.96×。
 *
 * 实现上不动 KV 结构(那是 6 个数组 180 处引用): 层内本来就分两半 ——
 *   ①注意力半层(KV/压缩器/索引器) 各会话回自己的图算, n_tokens=1;
 *   ②FFN/MoE 半层完全行无关, 借 ss[0] 的图当执行器一次算 N 行。
 * 两半的接口是 batch_after_attn_hc(注意力写) → batch_next_hc(FFN 写), 中间按行拷贝
 * (每层每行 64KB, N=4 时 22MB/步 ≈ 0.1ms, 可忽略)。
 * 注意力半层用 n_tokens=1 与单 token 解码路已验证逐位一致(尾批对账 43 层全同)。 */
int ds4_session_eval_multi(ds4_session **ss, const int *tokens, uint32_t n,
                           char *err, size_t errlen) {
#ifdef DS4_NO_GPU
    (void)ss; (void)tokens; (void)n;
    if (errlen) snprintf(err, errlen, "multi-session batch needs a GPU backend");
    return 1;
#else
    if (!ss || !tokens || n == 0) {
        if (errlen) snprintf(err, errlen, "invalid multi-session batch request");
        return 1;
    }
    if (n == 1 && !getenv("DS4_MULTI_FORCE")) return ds4_session_eval(ss[0], tokens[0], err, errlen);
    ds4_engine *e = ss[0]->engine;
    ds4_gpu_graph *gb = &ss[0]->graph;          /* 共享执行器(它的 batch_* 缓冲够 N 行) */
    if (n > gb->prefill_cap) {
        if (errlen) snprintf(err, errlen, "multi-session batch %u exceeds prefill cap %u",
                             n, gb->prefill_cap);
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (!ss[i] || ss[i]->engine != e || ss[i]->distributed || ds4_session_is_cpu(ss[i])) {
            if (errlen) snprintf(err, errlen, "multi-session batch needs same-engine GPU sessions");
            return 1;
        }
        if (ss[i]->checkpoint.len + 1 >= ss[i]->ctx_size) {
            if (errlen) snprintf(err, errlen, "multi-session batch: session %u out of context", i);
            return 1;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    const uint64_t q_bytes  = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    const uint64_t nm_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);

    bool ok = ds4_gpu_begin_commands() != 0;
    /* 前 3 层是哈希路由层: 专家由 token id 查表决定 ⇒ 共享图的 token 缓冲必须是本批的
     * N 个 id, 否则读到上一次 prefill 的残留 id, 每行选错专家(实测: 同 prompt 的 4 路
     * 输出互不相同, ffn_moe_probs 全同但 ffn_moe_topk 6/6 不同)。 */
    {
        ds4_tokens span = { .v = (int *)tokens, .len = (int)n, .cap = (int)n };
        if (ok) ok = metal_graph_upload_prompt_tokens(gb->prefill_tokens, &span, 0, n);
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_tokens one = { .v = (int *)&tokens[i], .len = 1, .cap = 1 };
            ok = metal_graph_upload_prompt_tokens(ss[i]->graph.prefill_tokens, &one, 0, 1u);
        }
    }
    /* N 个 token 的嵌入一次落共享图的 N 行。必须用批版(token 从张量读): 单 token 版
     * ds4_gpu_embed_token_hc_tensor 把 id 放在一个**全局单槽**里异步拷到设备, 连续调 N 次
     * 时 GPU 真正执行前那个槽已被最后一个 token 覆盖 —— 实测 4 行全拿到同一个嵌入。 */
    if (ok) ok = ds4_gpu_embed_tokens_hc_tensor(gb->batch_cur_hc,
                                                gb->prefill_tokens,
                                                e->model.map, e->model.size,
                                                e->weights.token_embd->abs_offset,
                                                (uint32_t)e->weights.token_embd->dim[1],
                                                n, DS4_N_EMBD, DS4_N_HC) != 0;
    /* 对照开关(DS4_MULTI_NO_SHARE=1): 每个会话整层都在自己的图上跑, 只共享输出头。
     * 用来把"驱动接线"和"分段共享"分开定位 —— 这条路应当与单路解码同残差档。 */
    static int no_share = -1;
    if (no_share < 0) no_share = getenv("DS4_MULTI_NO_SHARE") ? 1 : 0;
    if (ok && no_share) {
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_gpu_graph *gi = &ss[i]->graph;
            /* 直接嵌入各自图: 不能从 gb 取行 —— 会话 0 跑完 43 层后 gb->batch_cur_hc
             * 指针已被交换 43 次, 那里放的是它自己的层输出, 不是嵌入。 */
            if (i != 0) ok = ds4_gpu_embed_tokens_hc_tensor(gi->batch_cur_hc, gi->prefill_tokens,
                                                            e->model.map, e->model.size,
                                                            e->weights.token_embd->abs_offset,
                                                            (uint32_t)e->weights.token_embd->dim[1],
                                                            1u, DS4_N_EMBD, DS4_N_HC) != 0;
            for (uint32_t il = 0; ok && il < (uint32_t)DS4_N_LAYER; il++) {
                ok = metal_graph_encode_layer_attention_batch(gi, &e->model, &e->weights.layer[il],
                                                              il, (uint32_t)ss[i]->checkpoint.len, 1u) &&
                     metal_graph_encode_layer_ffn_batch(gi, &e->model, &e->weights.layer[il],
                                                        il, (uint32_t)ss[i]->checkpoint.len, 1u);
                if (ok) { ds4_gpu_tensor *t = gi->batch_cur_hc; gi->batch_cur_hc = gi->batch_next_hc; gi->batch_next_hc = t; }
            }
            if (ok && i != 0) ok = ds4_gpu_tensor_copy(gb->batch_cur_hc, (uint64_t)i * hc_bytes,
                                                       gi->batch_cur_hc, 0, hc_bytes) != 0;
        }
        goto multi_head;
    }
    for (uint32_t il = 0; ok && il < (uint32_t)DS4_N_LAYER; il++) {
        const ds4_layer_weights *lw = &e->weights.layer[il];
        const uint64_t qr_bytes = lw->attn_q_a->dim[1] * sizeof(float);
        /* ① 投影段: 共享图一次算 N 行(q_a/q_b/kv 权重只读一遍)。跳过 rope —— N 行来自
         *    不同会话, 位置各不相同, 下面按行用真实位置补。 */
        ok = metal_graph_encode_layer_attention_batch_stages(gb, &e->model, lw, il, 0u, n,
                                                             DS4_ATTN_STAGE_PRE | DS4_ATTN_STAGE_NOROPE);
        /* ①b 逐行 rope + kv fp8 量化(各用自己的位置) */
        {
            const uint32_t ratio_l = ds4_layer_compress_ratio(il);
            const bool comp_l = ratio_l != 0;
            const float fb = layer_rope_freq_base(il), fs = layer_rope_freq_scale(il);
            const float ef = (comp_l && DS4_ROPE_SCALE_FACTOR > 1.0f) ? 1.0f : 0.0f;
            float af = 1.0f;
            if (ef != 0.0f && fs > 0.0f) af /= 1.0f + 0.1f * logf(1.0f / fs);
            for (uint32_t i = 0; ok && i < n; i++) {
                const uint32_t p = (uint32_t)ss[i]->checkpoint.len;
                ds4_gpu_tensor *qv = ds4_gpu_tensor_view(gb->batch_q, (uint64_t)i * q_bytes, q_bytes);
                ds4_gpu_tensor *kvv = ds4_gpu_tensor_view(gb->batch_kv, (uint64_t)i * kv_bytes, kv_bytes);
                ok = qv && kvv &&
                     ds4_gpu_rope_tail_tensor(qv, 1u, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, p,
                                              comp_l ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                                              fb, fs, ef, af,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0 &&
                     ds4_gpu_rope_tail_tensor(kvv, 1u, 1u, DS4_N_HEAD_DIM, DS4_N_ROT, p,
                                              comp_l ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                                              fb, fs, ef, af,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0 &&
                     ds4_gpu_dsv4_fp8_kv_quantize_tensor(kvv, 1u, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                ds4_gpu_tensor_free(kvv);
                ds4_gpu_tensor_free(qv);
            }
        }
        /* ② KV 段: 每个会话回自己的图算(压缩器/索引器/注意力状态都在那儿) */
        static int pre_per_sess = -1;
        if (pre_per_sess < 0) pre_per_sess = getenv("DS4_MULTI_PRE_PER_SESSION") ? 1 : 0;
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_gpu_graph *gi = &ss[i]->graph;
            if (pre_per_sess && i != 0) {
                /* 二分用: 会话自己跑投影段(要先把自己的 hc 行搬回来), 不走拷贝集 */
                ok = ds4_gpu_tensor_copy(gi->batch_cur_hc, 0, gb->batch_cur_hc,
                                         (uint64_t)i * hc_bytes, hc_bytes) != 0 &&
                     metal_graph_encode_layer_attention_batch_stages(gi, &e->model, lw, il,
                                                                    (uint32_t)ss[i]->checkpoint.len,
                                                                    1u, DS4_ATTN_STAGE_PRE);
            } else if (i != 0) {
                ok = ds4_gpu_tensor_copy(gi->batch_q, 0, gb->batch_q, (uint64_t)i * q_bytes, q_bytes) != 0 &&
                     ds4_gpu_tensor_copy(gi->batch_kv, 0, gb->batch_kv, (uint64_t)i * kv_bytes, kv_bytes) != 0 &&
                     ds4_gpu_tensor_copy(gi->batch_attn_norm, 0, gb->batch_attn_norm,
                                         (uint64_t)i * nm_bytes, nm_bytes) != 0 &&
                     ds4_gpu_tensor_copy(gi->batch_qr_norm, 0, gb->batch_qr_norm,
                                         (uint64_t)i * qr_bytes, qr_bytes) != 0;
            }
            if (ok) ok = metal_graph_encode_layer_attention_batch_stages(
                    gi, &e->model, lw, il, (uint32_t)ss[i]->checkpoint.len, 1u,
                    DS4_ATTN_STAGE_KV);
            if (ok && i != 0)
                ok = ds4_gpu_tensor_copy(gb->batch_heads, (uint64_t)i * q_bytes,
                                         gi->batch_heads, 0, q_bytes) != 0;
        }
        /* ③ 出口段 + FFN/MoE: 共享图一次算 N 行。出口段开头还有一次"逆 rope"(把注意力
         *    输出转回), 同样依赖每行的真实位置 ⇒ 先逐行补, 再让出口段跳过。 */
        {
            const uint32_t ratio_l = ds4_layer_compress_ratio(il);
            const bool comp_l = ratio_l != 0;
            const float fb = layer_rope_freq_base(il), fs = layer_rope_freq_scale(il);
            const float ef = (comp_l && DS4_ROPE_SCALE_FACTOR > 1.0f) ? 1.0f : 0.0f;
            float af = 1.0f;
            if (ef != 0.0f && fs > 0.0f) af /= 1.0f + 0.1f * logf(1.0f / fs);
            for (uint32_t i = 0; ok && i < n; i++) {
                ds4_gpu_tensor *hv = ds4_gpu_tensor_view(gb->batch_heads, (uint64_t)i * q_bytes, q_bytes);
                ok = hv && ds4_gpu_rope_tail_tensor(hv, 1u, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
                                                    (uint32_t)ss[i]->checkpoint.len,
                                                    comp_l ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
                                                    fb, fs, ef, af,
                                                    DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
                ds4_gpu_tensor_free(hv);
            }
        }
        static int post_per_sess = -1;
        if (post_per_sess < 0) post_per_sess = getenv("DS4_MULTI_POST_PER_SESSION") ? 1 : 0;
        if (ok && post_per_sess) {
            /* 二分用: 出口段也各自跑(需要把该行的 hc 与 hc_split 搬回去) */
            const uint64_t mix_bytes = (2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC) * sizeof(float);
            for (uint32_t i = 0; ok && i < n; i++) {
                ds4_gpu_graph *gi = &ss[i]->graph;
                if (i != 0) {
                    ok = ds4_gpu_tensor_copy(gi->batch_cur_hc, 0, gb->batch_cur_hc,
                                             (uint64_t)i * hc_bytes, hc_bytes) != 0 &&
                         ds4_gpu_tensor_copy(gi->batch_hc_split, 0, gb->batch_hc_split,
                                             (uint64_t)i * mix_bytes, mix_bytes) != 0;
                }
                if (ok) ok = metal_graph_encode_layer_attention_batch_stages(
                        gi, &e->model, lw, il, (uint32_t)ss[i]->checkpoint.len, 1u,
                        DS4_ATTN_STAGE_POST | DS4_ATTN_STAGE_NOROPE);
                if (ok && i != 0)
                    ok = ds4_gpu_tensor_copy(gb->batch_after_attn_hc, (uint64_t)i * hc_bytes,
                                             gi->batch_after_attn_hc, 0, hc_bytes) != 0;
            }
        } else if (ok) {
            ok = metal_graph_encode_layer_attention_batch_stages(gb, &e->model, lw, il, 0u, n,
                                                                 DS4_ATTN_STAGE_POST | DS4_ATTN_STAGE_NOROPE);
        }
        if (ok) ok = metal_graph_encode_layer_ffn_batch(gb, &e->model, lw, il, 0u, n);
        if (ok) {   /* 与 encode_layer_batch 同款 cur/next 交换 */
            ds4_gpu_tensor *tmp = gb->batch_cur_hc;
            gb->batch_cur_hc = gb->batch_next_hc;
            gb->batch_next_hc = tmp;
        }
    }
multi_head:
    if (ok) ok = metal_graph_encode_output_head_batch(gb, &e->model, &e->weights,
                                                      n, e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && !gb->spec_logits) {
        if (errlen) snprintf(err, errlen, "multi-session batch needs the batch-logits buffer");
        ok = false;
    }
    static float *mb_logits = NULL; static uint32_t mb_cap = 0;
    if (ok && mb_cap < n) {
        free(mb_logits);
        mb_logits = xmalloc((size_t)n * DS4_N_VOCAB * sizeof(float));
        mb_cap = n;
    }
    if (ok) ok = ds4_gpu_tensor_read(gb->spec_logits, 0, mb_logits,
                                     (uint64_t)n * DS4_N_VOCAB * sizeof(float)) != 0;
    if (!ok) {
        if (errlen) snprintf(err, errlen, "%s multi-session batch failed",
                             ds4_backend_name(e->backend));
        for (uint32_t i = 0; i < n; i++) ss[i]->checkpoint_valid = false;
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        memcpy(ss[i]->logits, mb_logits + (uint64_t)i * DS4_N_VOCAB,
               (size_t)DS4_N_VOCAB * sizeof(float));
        token_vec_push(&ss[i]->checkpoint, tokens[i]);
        ss[i]->checkpoint_valid = true;
    }
    return 0;
#endif
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    if (!g_prof.enabled) return ds4_session_eval_internal(s, token, true, err, errlen);
    const double t0 = now_sec();
    int rc = ds4_session_eval_internal(s, token, true, err, errlen);
    if (rc == 0) ds4_profile_add_decode(1, now_sec() - t0);
    return rc;
}

/* Append N KNOWN tokens to the live KV in ONE layer-major batch (2026-07-14).
 *
 * Motivation: the guided tool-call primer injects dozens of tokens whose values
 * the SERVER already knows (the DSML structure) -- no sampling involved. Feeding
 * them one at a time pays the full per-token decode-bandwidth wall each
 * (measured dual-host: inject=28.9 s for 39 structure tokens vs a 352-token
 * batched prefill at 9.9 t/s -- the same expert bytes, 7x the throughput,
 * because a K-token batch reads the backbone ONCE; see the copy-spec note above
 * for the same physics).
 *
 * This is the batch path WITHOUT the speculative accept/rollback machinery: the
 * tokens are given, not drafted, so every one commits. Distributed sessions send
 * a single multi-token WORK span; Metal/CPU sessions use the same batched entry
 * the resume-prefill already uses. Falls back to a per-token loop on any error
 * so the caller never needs a second code path. Returns 0 on success. */
int ds4_session_eval_span(ds4_session *s, const int *tokens, int n,
                          char *err, size_t errlen) {
    if (!s || !tokens || n <= 0) {
        if (errlen) snprintf(err, errlen, "invalid span eval request");
        return 1;
    }
    if (ds4_session_pos(s) + n >= s->ctx_size) {
        if (errlen) snprintf(err, errlen, "span exceeds context");
        return 1;
    }
    if (n == 1) return ds4_session_eval(s, tokens[0], err, errlen);

    /* Batched path: extend the checkpoint timeline with the known span. The
     * session's own sync entry drives the backend-specific batch (dist: one
     * WORK span; metal: resume-prefill), and it is exactly the prompt-prefill
     * path, so KV rows finalize in the same order as a cold prompt. */
    ds4_tokens want = {0};
    if (s->checkpoint_valid)
        for (int i = 0; i < s->checkpoint.len; i++) ds4_tokens_push(&want, s->checkpoint.v[i]);
    for (int i = 0; i < n; i++) ds4_tokens_push(&want, tokens[i]);
    const double t0 = now_sec();
    int rc = ds4_session_sync_internal(s, &want, err, errlen);
    ds4_tokens_free(&want);
    if (rc == 0) {
        if (g_prof.enabled) ds4_profile_add_decode((uint64_t)n, now_sec() - t0);
        return 0;
    }
    /* Any batch failure: fall back to the exact per-token semantics. */
    for (int i = 0; i < n; i++)
        if (ds4_session_eval(s, tokens[i], err, errlen) != 0) return 1;
    return 0;
}

/* copy-spec 单机整族已删除(2026-08-05 用户裁决)。 */

/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let MTP recursively draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    if (s->distributed) {
        if (!accepted) return 0;
        if (!s->checkpoint_valid) {
            if (errlen) snprintf(err, errlen, "distributed decode requires a valid checkpoint");
            return -1;
        }
        /* docs/archive/mtp.md Phase 1: cross-machine MTP speculation. The driver commits
         * first_token + verified drafts into the session checkpoint and returns
         * the committed count; s->logits is left predicting the next token. */
        int cap = accepted_cap < max_tokens ? accepted_cap : max_tokens;
        int n = ds4_dist_session_eval_speculative(s->distributed, s, &s->checkpoint,
                                                  first_token, eos_token,
                                                  accepted, cap, s->logits,
                                                  err, errlen);
        if (n < 0) { s->checkpoint_valid = false; return -1; }
        return n;
    }
    if (ds4_session_is_cpu(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "GPU support is not compiled in");
    return -1;
#else
    ds4_engine *e = s->engine;

    /* copy-spec 整族已删除(2026-08-05 用户裁决: 环境变量硬编码型行为机制, 与
     * auto-arm 同判)。无显式 MTP draft 模型 => 纯单 token 平解码; 投机只剩
     * 显式配置的 MTP drafter 一条路。 */
    /* DSpark 投机主循环(DS4_DSPARK_SPEC=1, 2026-08-18): draft 块(5)+bonus 走 6 位
     * verify 批; 接受链 argmax 对照; 部分接受= state 恢复+接受位重放(官方
     * checkpoint-restore 口径)。verify 批复用 batch 层包装 ⇒ mh 抓取/drafter 建窗自动。 */
    if (getenv("DS4_DSPARK_PROBE")) {
        static int gate_diag = 0;
        if (!gate_diag) { gate_diag = 1;
            fprintf(stderr, "ds4: [dspark-gate] ready=%d capture=%d greedy=%d env=%d\n",
                    (int)e->dspark.ready, s->graph.dspark_capture, s->spec_greedy,
                    getenv("DS4_DSPARK_SPEC") != NULL);
        }
    }
    if (e->dspark.ready && s->graph.dspark_capture && s->spec_greedy &&
        getenv("DS4_DSPARK_SPEC") != NULL) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        int n_acc = 0;
        accepted[n_acc++] = first_token;
        static float *row_logits = NULL;
        if (!row_logits) row_logits = xmalloc(6ull * DS4_N_VOCAB * sizeof(float));
        /* 候选数可调(2026-08-21 调度杠杆): verify 是字节受限, 而专家并集随候选数增长
         * (实测 6 候选=22.1/36 唯一专家)。每字节产出 acc/bytes 在 k=3-4 处更优。 */
        static uint32_t spec_k = 0u;
        if (spec_k == 0u) {
            const char *e = getenv("DS4_SPEC_CAND");
            /* 默认 3(2026-08-21 扫参): 每 token 毫秒 k=6 48 / k=4 44 / k=3 38.7 / k=2 39.8。
             * verify 是字节受限且专家并集随候选数增长(实测 6 候选=22.1 唯一专家), 多草稿
             * 的边际接受收益跑不过边际字节成本。verify 语义与 k 无关 ⇒ 质量不受影响。 */
            spec_k = e ? (uint32_t)atoi(e) : 3u;
            if (spec_k < 2u) spec_k = 2u;
            if (spec_k > 6u) spec_k = 6u;
        }
        const bool spec_prof = getenv("DS4_SPEC_PROF") != NULL;
        double pf_draft = 0, pf_snap = 0, pf_verify = 0, pf_replay = 0, pf_misc = 0, pf_round_wall = 0;
        uint32_t pf_rounds = 0; double pf_t0 = 0;
        while (n_acc < max_tokens && n_acc + (int)DS4_DSPARK_BLK + 1 <= accepted_cap) {
            if (spec_prof) pf_t0 = now_sec();
            int next = 0; float best = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (s->logits[v] > best) { best = s->logits[v]; next = (int)v; }
            if (next == eos_token) break;
            int cand[6];
            cand[0] = next;
            int ids[DS4_DSPARK_BLK] = {0};
            const uint32_t pos_now = (uint32_t)s->checkpoint.len;
            if (spec_prof) { pf_misc += now_sec() - pf_t0; pf_t0 = now_sec(); }
            /* 置信调度(DS4_SPEC_SCHED=1, 论文 2607.05147 Alg.1 单请求版):
             * 草稿一次出满块并拿到逐位置信 c_i; 前缀存活率 a_j = ∏_{i<=j} c_i 就是"验第 j 个
             * 候选能多拿到的期望 token 数"。多验一个候选的边际成本是 m 毫秒(在线最小二乘
             * 从 (k-1, verify_ms) 拟合), 当前吞吐 T = 已接受 token / 已用毫秒。只有
             * a_j > T*m 时这个候选才划算 —— 这正是把"能不能白送"这个物理事实写进调度。 */
            static int sched = -1;
            if (sched < 0) sched = getenv("DS4_SPEC_SCHED") ? 1 : 0;
            static double sch_tok = 0.0, sch_ms = 0.0;          /* 在线吞吐 */
            static double rg_n = 0, rg_x = 0, rg_y = 0, rg_xx = 0, rg_xy = 0;  /* 边际回归 */
            static double sch_m = 12.0;                          /* 每候选边际 ms */
            /* 在线校准(论文的 STS 在线版): 原始置信头没针对"q2 drafter + 贪心验证"标定,
             * 用自己的接受结果做逐位置乘性校正 g_j = 实测接受 / 预测和。低估就放大, 高估
             * 就收缩 —— 调度阈值才对得上真实的边际收益。 */
            static double cal_sum[DS4_DSPARK_BLK] = {0};
            static double cal_hit[DS4_DSPARK_BLK] = {0};
            static uint32_t cal_n[DS4_DSPARK_BLK] = {0};
            /* 投机开关闸(2026-08-21): drafter 不是白送的 —— 一轮草稿 ~9ms, 每个候选边际
             * ~12ms。文本难预测时(bash/概念解释)接受率掉到 1.8, 投机反而比纯解码慢 20%。
             * 在线对比两种模式的实测 token/ms, 谁快用谁, 并按固定比例回探另一种以便文本
             * 变好预测时能切回来。纯解码轮直接走单 token 解码路 = 无损且零草稿开销。 */
            static double md_tok[2] = {0, 0}, md_ms[2] = {0, 0};   /* 0=纯解码 1=投机 */
            static uint32_t md_n[2] = {0, 0}, md_round = 0;
            int mode = 1;
            if (sched) {
                /* 起步先给投机 16 轮把 T/m/校准跑热, 之后按实测吞吐择优, 每 32 轮回探 1 轮。 */
                const int explore = (md_round % 32u) == 31u;
                if (md_round >= 16u && md_n[0] >= 3u && md_n[1] >= 8u) {
                    const double t0 = md_tok[0] / md_ms[0], t1 = md_tok[1] / md_ms[1];
                    mode = (t1 >= t0 * 0.98) ? 1 : 0;   /* 平手偏投机(它还带 acc 上升空间) */
                } else if (md_round >= 16u && md_n[0] < 3u) {
                    mode = 0;                            /* 先取几个纯解码样本 */
                }
                if (explore) mode = 1 - mode;
                md_round++;
            }
            const double mode_t0 = now_sec();
            if (sched && mode == 0) {
                /* 纯解码轮: 提交 next, 不草稿不批验证 */
                if (ds4_session_eval(s, next, err, errlen) != 0) { s->checkpoint_valid = false; return -1; }
                accepted[n_acc++] = next;
                const double dt = (now_sec() - mode_t0) * 1e3;
                md_tok[0] += 1.0; md_ms[0] += dt; md_n[0]++;
                /* 不喂 sch_tok/sch_ms: k 调度器的 T 是"投机模式下的吞吐", 掺进纯解码轮会
                 * 抬高阈值 → k 变小 → 投机更差 → 更偏纯解码, 形成死亡螺旋。 */
                if (getenv("DS4_SPEC_SCHED_LOG") && (md_round % 32u) == 0)
                    fprintf(stderr, "ds4: [sched-mode] 纯解码 %.4f tok/ms(n=%u) vs 投机 %.4f(n=%u)\n",
                            md_n[0] ? md_tok[0] / md_ms[0] : 0.0, md_n[0],
                            md_n[1] ? md_tok[1] / md_ms[1] : 0.0, md_n[1]);
                continue;
            }

            float conf[DS4_DSPARK_BLK] = {0};
            uint32_t draft_n = sched ? (uint32_t)DS4_DSPARK_BLK : spec_k - 1u;
            if (!metal_graph_dspark_step_n(&s->graph, &e->model, &e->weights, &e->dspark,
                                           next, pos_now - 1u, ids, draft_n,
                                           sched ? conf : NULL)) {
                if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-gate] step FAILED\n");
                break;
            }
            uint32_t round_k = spec_k;
            if (sched) {
                const double T = (sch_ms > 1.0) ? (sch_tok / sch_ms) : 0.030;   /* token/ms */
                const double thr = T * sch_m;
                double a = 1.0;
                uint32_t adm = 0;
                for (uint32_t j = 0; j < draft_n; j++) {
                    double c = (double)conf[j];
                    if (cal_n[j] >= 8u && cal_sum[j] > 1e-6) {
                        double g = (cal_hit[j] + 1.0) / (cal_sum[j] + 1.0);
                        c *= g;
                        if (c > 0.999) c = 0.999;
                        if (c < 0.001) c = 0.001;
                    }
                    a *= c;
                    if (a <= thr) break;
                    adm++;
                }
                round_k = adm + 1u;
                if (round_k < 2u) round_k = 2u;
                if (round_k > (uint32_t)DS4_DSPARK_BLK + 1u) round_k = (uint32_t)DS4_DSPARK_BLK + 1u;
                if (getenv("DS4_SPEC_SCHED_LOG"))
                    fprintf(stderr, "ds4: [sched] c=[%.2f %.2f %.2f %.2f %.2f] T=%.4f m=%.1f 阈=%.3f k=%u\n",
                            conf[0], conf[1], conf[2], conf[3], conf[4], T, sch_m, thr, round_k);
            }
            for (uint32_t i = 0; i + 1u < round_k; i++) cand[1 + i] = ids[i];
            if (spec_prof) { pf_draft += now_sec() - pf_t0; pf_t0 = now_sec(); }
            if (!metal_graph_dspark_state_snapshot(&s->graph)) {
                if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-gate] snapshot FAILED\n");
                break;
            }
            if (spec_prof) { pf_snap += now_sec() - pf_t0; pf_t0 = now_sec(); }
            /* 诊断路径(DS4_SPEC_SEQ_VERIFY=1): verify 不走批, 而是逐候选走单 token
             * 解码路 —— 与纯解码逐字节同一条 kernel 链。没有批的加速, 只用来量
             * drafter 对"真解码"的接受率上限, 以及判定批 verify 是否引入了偏差。
             * 状态天然随接受推进, 不需要 snapshot/restore/快进。 */
            const bool xcheck = getenv("DS4_SPEC_XCHECK") != NULL;
            const bool seq_verify = xcheck || getenv("DS4_SPEC_SEQ_VERIFY") != NULL;
            int seq_acc = 1;
            static float *xc_batch = NULL;
            if (xcheck) {
                /* 同一位置先走批 verify(标签 b), 再走单 token 解码路(标签 d), 逐行比 logits。
                 * 批的结果只用于对账, 提交仍走解码路 ⇒ 轨迹与纯解码相同。 */
                if (!xc_batch) xc_batch = xmalloc(6ull * DS4_N_VOCAB * sizeof(float));
                g_dump_tag = "b";
                s->graph.spec_comp_capture = 1;
                const int rc = ds4_session_verify_batch_argmax(s, cand, round_k, pos_now,
                                                              0u, (uint32_t)DS4_N_LAYER - 1u,
                                                              NULL, row_logits, err, errlen);
                s->graph.spec_comp_capture = 0;
                g_dump_tag = "";
                if (rc != 0) { s->checkpoint_valid = false; return -1; }
                memcpy(xc_batch, row_logits, (size_t)round_k * DS4_N_VOCAB * sizeof(float));
                if (!metal_graph_dspark_state_restore(&s->graph)) break;
                s->checkpoint.len = (int)pos_now;
            }
            if (seq_verify) {
                if (xcheck) g_dump_tag = "d";
                for (uint32_t i = 0; i < round_k; i++) {
                    if (ds4_session_eval(s, cand[i], err, errlen) != 0) {
                        s->checkpoint_valid = false;
                        return -1;
                    }
                    memcpy(row_logits + (uint64_t)i * DS4_N_VOCAB, s->logits,
                           (size_t)DS4_N_VOCAB * sizeof(float));
                    if (i + 1u >= round_k) break;
                    int am = 0; float bb = -1e30f;
                    for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                        if (s->logits[v] > bb) { bb = s->logits[v]; am = (int)v; }
                    if (am != cand[i + 1]) break;
                    seq_acc++;
                }
                if (xcheck) {
                    g_dump_tag = "";
                    static uint32_t xc_n = 0, xc_same = 0; static double xc_dmax = 0.0;
                    for (uint32_t i = 0; i < round_k; i++) {
                        const float *rb = xc_batch + (uint64_t)i * DS4_N_VOCAB;
                        const float *rd = row_logits + (uint64_t)i * DS4_N_VOCAB;
                        int ab = 0, ad = 0; float bb = -1e30f, bd = -1e30f; double dmax = 0.0;
                        for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++) {
                            if (rb[v] > bb) { bb = rb[v]; ab = (int)v; }
                            if (rd[v] > bd) { bd = rd[v]; ad = (int)v; }
                            const double d = fabs((double)rb[v] - (double)rd[v]);
                            if (d > dmax) dmax = d;
                        }
                        xc_n++; if (ab == ad) xc_same++;
                        if (dmax > xc_dmax) xc_dmax = dmax;
                        if (i == 0 && ab != ad)
                            fprintf(stderr, "ds4: [xcheck] pos=%u row0 批argmax=%d 解码argmax=%d max|Δlogit|=%.4g\n",
                                    pos_now, ab, ad, dmax);
                    }
                    if ((xc_n % 16u) == 0)
                        fprintf(stderr, "ds4: [xcheck] 行数=%u argmax一致=%.1f%% 全程max|Δlogit|=%.4g\n",
                                xc_n, 100.0 * xc_same / xc_n, xc_dmax);
                }
            } else {
            s->graph.spec_comp_capture = 1;   /* verify 批捕获压缩器输入行(快进用) */
            if (ds4_session_verify_batch_argmax(s, cand, round_k, pos_now,
                                                0u, (uint32_t)DS4_N_LAYER - 1u,
                                                NULL, row_logits, err, errlen) != 0) {
                s->graph.spec_comp_capture = 0;
                s->checkpoint_valid = false;
                return -1;
            }
            s->graph.spec_comp_capture = 0;
            }
            if (spec_prof) { pf_verify += now_sec() - pf_t0; pf_t0 = now_sec(); }
            if (getenv("DS4_DSPARK_DIAG")) {
                /* drafter 主干 top1(markov 前口径不可得, 打 markov 后 d0) vs 主模型 row0 top1 */
                int am0 = 0; float b0 = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (row_logits[v] > b0) { b0 = row_logits[v]; am0 = (int)v; }
                static int dn = 0;
                if (dn < 8) {
                    float mh2[2] = {0}, mx2[2] = {0}, lg2[2] = {0};
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_hidden, 0, mh2, sizeof(mh2));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_x, 0, mx2, sizeof(mx2));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_logits, 0, lg2, sizeof(lg2));
                    fprintf(stderr, "ds4: [dspark-diag] verify_row0_top1=%d draft0=%d draft=[%d %d %d %d %d] mh=%.3g %.3g mx=%.3g %.3g lg=%.3g %.3g\n",
                            am0, cand[1], cand[1], cand[2], cand[3], cand[4], cand[5],
                            mh2[0], mh2[1], mx2[0], mx2[1], lg2[0], lg2[1]);
                    dn++;
                }
            }
            int acc = 1;
            for (int i = 0; !seq_verify && i + 1 < (int)round_k; i++) {
                int am = 0; float bb = -1e30f;
                const float *row = row_logits + (uint64_t)i * DS4_N_VOCAB;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (row[v] > bb) { bb = row[v]; am = (int)v; }
                if (am != cand[i + 1]) break;
                acc++;
            }
            if (seq_verify) acc = seq_acc;
            static int raw_restore_on = -1;
            if (raw_restore_on < 0) raw_restore_on = getenv("DS4_SPEC_NO_RAW_RESTORE") ? 0 : 1;
            if (raw_restore_on && !seq_verify && acc < (int)round_k &&
                !metal_graph_spec_raw_restore(&s->graph, pos_now, (uint32_t)acc, round_k)) {
                if (errlen) snprintf(err, errlen, "spec raw KV restore failed");
                s->checkpoint_valid = false;
                return -1;
            }
            if (!seq_verify && !metal_graph_dspark_win_commit(&s->graph, pos_now, (uint32_t)acc)) {
                if (errlen) snprintf(err, errlen, "dspark window commit failed");
                s->checkpoint_valid = false;
                return -1;
            }
            /* timeline 已 commit 6 位 → 截到接受数 */
            if (!seq_verify) s->checkpoint.len = (int)(pos_now + (uint32_t)acc);
            if (!seq_verify && acc < (int)round_k) {
                if (!metal_graph_dspark_state_restore(&s->graph)) break;
                if (getenv("DS4_SPEC_REPLAY_OLD")) {
                    /* A/B 对照: 旧 replay 全前向路径 */
                    s->checkpoint.len = (int)pos_now;
                    if (ds4_session_verify_batch_argmax(s, cand, (uint32_t)acc, pos_now,
                                                        0u, (uint32_t)DS4_N_LAYER - 1u,
                                                        NULL, row_logits, err, errlen) != 0) {
                        s->checkpoint_valid = false;
                        return -1;
                    }
                } else if (!metal_graph_spec_comp_fastforward(&s->graph, &e->model, &e->weights,
                                                              pos_now, (uint32_t)acc)) {
                    /* replay 消除(2026-08-20): KV raw 行 verify 已写好且 restore 不动;
                     * 压缩器/indexer 态用 verify 捕获的输入行快进 acc 位。 */
                    if (errlen) snprintf(err, errlen, "spec compressor fast-forward failed");
                    s->checkpoint_valid = false;
                    return -1;
                }
                s->checkpoint.len = (int)(pos_now + (uint32_t)acc);
            }
            memcpy(s->logits, row_logits + (uint64_t)(acc - 1) * DS4_N_VOCAB,
                   (size_t)DS4_N_VOCAB * sizeof(float));
            if (spec_prof) {
                pf_replay += now_sec() - pf_t0; pf_t0 = now_sec();
                pf_rounds++;
                { static double last_top = 0.0; const double nowv = now_sec();
                  if (last_top > 0.0) pf_round_wall += nowv - last_top;
                  last_top = nowv; }
                if ((pf_rounds & 7u) == 0) {
                    fprintf(stderr, "ds4: [spec-prof] rounds=%u win8_ms: draft=%.1f snap=%.1f verify=%.1f replay=%.1f misc=%.1f | round_wall=%.1f\n",
                            pf_rounds, pf_draft * 1e3 / 8.0, pf_snap * 1e3 / 8.0,
                            pf_verify * 1e3 / 8.0, pf_replay * 1e3 / 8.0, pf_misc * 1e3 / 8.0,
                            pf_round_wall * 1e3 / 8.0);
                    pf_draft = pf_snap = pf_verify = pf_replay = pf_misc = pf_round_wall = 0;   /* 滑窗 */
                }
            }
            if (sched) {
                const double dt_mode = (now_sec() - mode_t0) * 1e3;
                md_tok[1] += (double)acc; md_ms[1] += dt_mode; md_n[1]++;
                /* 校准喂数: 草稿位 j 被真正验证过(前缀全接受)才计数; j = acc-1 是被拒的那位。 */
                for (uint32_t j = 0; j + 1u < round_k && j < (uint32_t)acc; j++) {
                    cal_n[j]++; cal_sum[j] += (double)conf[j];
                    if ((int)j < acc - 1) cal_hit[j] += 1.0;
                }
                if (getenv("DS4_SPEC_SCHED_LOG")) {
                    static uint32_t cl = 0;
                    if (((++cl) % 32u) == 0) {
                        fprintf(stderr, "ds4: [sched-cal]");
                        for (uint32_t j = 0; j < (uint32_t)DS4_DSPARK_BLK; j++)
                            fprintf(stderr, " p%u: 预测%.2f 实测%.2f(n=%u)", j + 1,
                                    cal_n[j] ? cal_sum[j] / cal_n[j] : 0.0,
                                    cal_n[j] ? cal_hit[j] / cal_n[j] : 0.0, cal_n[j]);
                        fprintf(stderr, "\n");
                    }
                }
                /* 在线标定: 本轮墙钟与接受数喂吞吐; (k-1, verify_ms) 喂边际最小二乘。
                 * x 有方差后才用拟合值, 否则保持上一次的 m(初值 12ms)。 */
                static double last_top2 = 0.0;
                const double nowv2 = now_sec();
                if (last_top2 > 0.0) {
                    const double dt_ms = (nowv2 - last_top2) * 1e3;
                    sch_tok += (double)acc; sch_ms += dt_ms;
                    const double x = (double)(round_k - 1u);
                    rg_n += 1; rg_x += x; rg_y += dt_ms; rg_xx += x * x; rg_xy += x * dt_ms;
                    const double den = rg_n * rg_xx - rg_x * rg_x;
                    if (rg_n >= 8 && den > 1e-6) {
                        const double slope = (rg_n * rg_xy - rg_x * rg_y) / den;
                        if (slope > 1.0 && slope < 60.0) sch_m = slope;
                    }
                }
                last_top2 = nowv2;
            }
            if (getenv("DS4_DSPARK_STAT")) {
                static uint32_t st_rounds = 0, st_acc = 0;
                st_rounds++; st_acc += (uint32_t)acc;
                if ((st_rounds & 7u) == 0)
                    fprintf(stderr, "ds4: [dspark-stat] rounds=%u avg_acc=%.2f\n",
                            st_rounds, (double)st_acc / st_rounds);
            }
            /* mh: verify/重放批的末接受位 → dspark_main_hidden(3 slot 连续拷贝)。
             * seq 诊断路径下单 token 解码自己写 mh, 不需要也不能从批里拷。 */
            for (uint32_t sl = 0; !seq_verify && sl < 3u; sl++)
                (void)ds4_gpu_tensor_copy(s->graph.dspark_main_hidden,
                                          (uint64_t)sl * DS4_N_EMBD * sizeof(float),
                                          s->graph.dspark_pf_hidden,
                                          ((uint64_t)(acc - 1) * 3u + sl) * DS4_N_EMBD * sizeof(float),
                                          (uint64_t)DS4_N_EMBD * sizeof(float));
            for (int i = 0; i < acc && n_acc < accepted_cap; i++) accepted[n_acc++] = cand[i];
            if (acc >= 1 && cand[acc - 1] == eos_token) break;
        }
        return n_acc;
    }
    if (!e->mtp_ready) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        /* DSpark 探针(DS4_DSPARK_PROBE=1): eval 后 main_hidden 已被 graph 抓取,
         * 跑一次 drafter 块打印草稿(活性/质量人工判读, 主循环接线前的最小验证)。 */
        if (getenv("DS4_DSPARK_PROBE")) {
            static int diag1 = 0;
            if (!diag1) { diag1 = 1;
                fprintf(stderr, "ds4: [dspark-diag] ready=%d g=%p capture=%d buf=%p\n",
                        (int)e->dspark.ready, (void *)&s->graph, s->graph.dspark_capture, (void *)s->graph.dspark_main_hidden);
            }
        }
        if (e->dspark.ready && s->graph.dspark_capture && getenv("DS4_DSPARK_PROBE")) {
            /* 零轨迹噪声命中统计(2026-08-21): PROBE 下草稿不被接受 ⇒ 生成序列与纯解码
             * 完全一致, 三种 drafter 配置走同一条轨迹 ⇒ 命中率可直接比, 不含 acc 那种
             * ±0.09 的轨迹漂移噪声。上一轮的草稿首位 vs 本轮真实 token。 */
            /* 逐位命中(2026-08-21): 首位 91% 但 SPEC acc 仅 2.0/3 ⇒ 推出第二位接受率
             * ~9%, 与 probe 原始输出"常连中 5 位"矛盾 ⇒ 需要逐位真值来定位是能力还是
             * SPEC 路径的问题。ring 存最近一次草稿的 5 位, 与后续 5 个真实 token 比。 */
            static int pend = -1;
            static uint32_t hit = 0, tot = 0;
            static int dr[DS4_DSPARK_BLK] = {-1,-1,-1,-1,-1};
            static int dr_age = 99;
            static uint32_t phit[DS4_DSPARK_BLK] = {0}, ptot[DS4_DSPARK_BLK] = {0};
            static int probe_n = 0;
            if (probe_n < 100000) {
                int next = 0; float best = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (s->logits[v] > best) { best = s->logits[v]; next = (int)v; }
                /* 上一轮草稿首位预测的正是本轮的 next(不是本轮的输入 token) */
                if (pend >= 0) { tot++; if (pend == next) hit++; }
                if (dr_age < (int)DS4_DSPARK_BLK) {
                    ptot[dr_age]++;
                    if (dr[dr_age] == next) phit[dr_age]++;
                    else dr_age = 99;   /* 链式: 一旦某位错, 后续位不再计(与 SPEC 接受语义一致) */
                    if (dr_age != 99) dr_age++;
                }
                if (getenv("DS4_DSPARK_PROBE_STAT") && tot && (tot % 64u) == 0) {
                    fprintf(stderr, "ds4: [probe-hit] %u/%u = %.1f%% | 逐位链式:", hit, tot, 100.0 * hit / tot);
                    for (uint32_t q = 0; q < DS4_DSPARK_BLK; q++)
                        fprintf(stderr, " p%u=%.0f%%(%u/%u)", q + 1,
                                ptot[q] ? 100.0 * phit[q] / ptot[q] : 0.0, phit[q], ptot[q]);
                    fprintf(stderr, "\n");
                }
                int ids[DS4_DSPARK_BLK] = {0};
                if (metal_graph_dspark_step(&s->graph, &e->model, &e->weights, &e->dspark,
                                            next, (uint32_t)(s->checkpoint.len - 1), ids)) {
                    pend = ids[0];
                    if (dr_age >= (int)DS4_DSPARK_BLK) {   /* 上一条草稿已用完/断链, 换新的 */
                        for (uint32_t q = 0; q < DS4_DSPARK_BLK; q++) dr[q] = ids[q];
                        dr_age = 0;
                    }
                    float mh[4] = {0}, lg[4] = {0}, mx[4] = {0};
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_hidden, 0, mh, sizeof(mh));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_x, 0, mx, sizeof(mx));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_logits, 0, lg, sizeof(lg));
                    if (probe_n < 8)
                        fprintf(stderr, "ds4: [dspark] pos=%d next=%d draft= %d %d %d %d %d | mh=%.3g %.3g mx=%.3g %.3g lg=%.3g %.3g\n",
                                s->checkpoint.len - 1, next, ids[0], ids[1], ids[2], ids[3], ids[4],
                                mh[0], mh[1], mx[0], mx[1], lg[0], lg[1]);
                } else if (probe_n < 8) {
                    fprintf(stderr, "ds4: [dspark] step failed at pos=%d\n", s->checkpoint.len - 1);
                }
                probe_n++;
            }
        }
        return 1;
    }
    /* MTP 投机整族已删除(2026-08-05 用户裁决: Go 定型优化)。mtp_ready 恒 false,
     * 上面的 plain 路径即全部行为; 此处永不可达。 */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

