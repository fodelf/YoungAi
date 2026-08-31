/* core_gpu_capture.c — 引擎轨迹捕获(DS4_CAP_DIR)+ampanc (机械拆分自 ds4.c, 重构阶段4)。 */
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
    const char *r = ds4_tool_cap_layers();
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
    const char *dir = ds4_tool_cap_dir();
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
    const char *p = ds4_tool_amp_anchor();
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
    g_ampanc.route_on = ds4_tool_amp_anchor_route();
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
    const char *dir = ds4_tool_cap_dir();
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
typedef int ds4_core_gpu_capture_nonempty_tu; /* 空TU防御(CPU构建) */
