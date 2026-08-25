/* core_gpu_debug.c — metal 诊断 dump/trace/router freq (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
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
typedef int ds4_core_gpu_debug_nonempty_tu; /* 空TU防御(CPU构建) */
