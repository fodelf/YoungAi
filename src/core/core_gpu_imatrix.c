/* core_gpu_imatrix.c — eval_token + imatrix 采集 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
#ifdef DS4_STATE_DUMP
ds4_gpu_tensor *g_sd_hc[64];   /* 诊断: 逐层输出 hc 快照(prefill_util 逐层 D2D 拷入) */
ds4_gpu_tensor *g_sd_pt[8];    /* 诊断: 层 0 内部四点快照(decode_layer 拷入) */
#endif
bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = throttle ? now_sec() : 0.0;
#ifdef DS4_STATE_DUMP
    if (!g_sd_hc[0]) {   /* capture 开始前分配(capture 内禁 cudaMalloc) */
        for (uint32_t il = 0; il < DS4_N_LAYER && il < 64u; il++)
            g_sd_hc[il] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));
        for (int k = 0; k < 8; k++)
            g_sd_pt[k] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));
    }
#endif

    bool ok = ds4_gpu_begin_commands() != 0;
    /* decode 单 token CUDA graph: capture 包住 encode(纯 kernel 段), 失败则重编码直跑。
     * 流水线(2026-08-17): 上一 token 的 GPU 窗口里已预编码本 pos 的图 ⇒ 直接发射,
     * encode 移出临界路径; 发射后趁 GPU 忙再预编码 pos+1(token id 走参数槽间接)。
     * 预发射(2026-09-07): 贪心解码时 pos+1 的图也在本图跑完前排进流(token 由图末尾的设备
     * argmax 供给), 本图 logits 异步回传 ⇒ token 边界 GPU 不再空转(此前 D2H+采样+cudaGraphLaunch
     * 串行 1.26 ms/token)。字段说明见 core_gpu_graph.h 预发射段。 */
    int launched = 0;
    /* --cap-dir 取料(引擎真值捕获)在解码层里逐 token 做 D2H 落盘, 捕获态里是非法操作 ⇒
     * "operation failed due to a previous error during capture", 回退直发也已被污染, 位置 1
     * 就死(2026-09-06 q2k 反修取料实撞, 只捕到 1 行)。取料路一律不进图: 取料要的是确定性
     * 与部署同路数值, 不要速度。 */
    const bool capdir = ds4_tool_cap_dir() != NULL || ds4_tool_eval_hdump() != NULL;   /* 取料/逐层落盘都做 D2H, 不进图 */
    if (g->prelaunch_capable == 0) g->prelaunch_capable = ds4_gpu_decode_prelaunch_capable() ? 1 : -1;
    /* 预捕获对账: 上一轮为 pend_pos 预捕获时主机计数器已推进; 本次 eval 不在那个位置 ⇒ 作废+回滚 */
    if (g->pend_pos >= 0 && g->pend_pos != (int64_t)pos) metal_graph_token_pending_discard(g);
    const bool can_async = g->prelaunch_capable > 0 && !capdir && logits != NULL;
    if (can_async && !g->logits_pinned) {
        g->logits_pinned = ds4_gpu_host_alloc((uint64_t)DS4_N_VOCAB * sizeof(float) + 64u);
        g->tok_next_pinned = g->logits_pinned ? (int32_t *)(g->logits_pinned + DS4_N_VOCAB) : NULL;
    }
    /* 预发射对账: 上一轮已把本 pos 的图按设备 argmax 排进流 */
    if (ok && g->prelaunched_pos == (int64_t)pos) {
        const int claimed = ds4_gpu_token_graph_prelaunch_claim(pos);
        g->prelaunched_pos = -1;
        if (claimed && (int32_t)token == g->prelaunched_token) {
            launched = 1;
        } else {
            /* 调用方喂的不是那个 argmax(采样/惩罚/注入): 预发射只在非 emit 位做, 设备侧写的全是
             * 可覆写行; 主机计数器回滚到预捕获前, 走正常路重编码本 pos, 流序在预发射图之后覆写其行。 */
            memcpy(g->layer_n_comp, g->snap_n_comp, sizeof(g->snap_n_comp));
            memcpy(g->layer_n_index_comp, g->snap_n_index_comp, sizeof(g->snap_n_index_comp));
            memcpy(g->comp_x_pending, g->snap_x_pending, sizeof(g->snap_x_pending));
            memcpy(g->comp_x_last_pos, g->snap_x_last, sizeof(g->snap_x_last));
        }
    }
    if (ok && !launched && !capdir && ds4_gpu_token_graph_try_pending(token, pos, logits != NULL) > 0)
        launched = 1;
    /* 预捕获的就是本 pos: 发射了 ⇒ 计数器已是 pos 之后的正确态; 没发射(需 logits 不同/图关) ⇒ 下面重编码
     * 会再推一次, 先回滚到预捕获前。 */
    if (g->pend_pos == (int64_t)pos) {
        if (!launched) metal_graph_token_pending_discard(g);
        g->pend_pos = -1;
    }
    ds4_gpu_token_graph_set_pos(pos);
    const int tok_graph = (ok && !launched && !capdir) ? ds4_gpu_token_graph_begin() : 0;
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
    /* 本图 logits + 设备 argmax 异步回传到 pinned: 排在本图之后、下一图之前(logits 缓冲会被下一图覆写) */
    bool async_rb = false;
    if (ok && can_async && g->logits_pinned)
        async_rb = ds4_gpu_decode_readback_async(g->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                                                 g->logits_pinned, g->tok_next_pinned) != 0;
    if (ok && (launched || tok_graph)) {
        /* 预发射条件: 会话声明贪心 + 回传已排 + pos+1 非 emit 位。emit = (位置+1) 整除 ratio, ratio-4 层
         * 每 4 位一次(ratio-128 是其子集) ⇒ 预发射的位置 pos+1 是 emit 当且仅当 (pos+2)%4==0; emit 会
         * 改压缩器 state/追加压缩行, 不可覆写, 那一位不预发射(付一次原来的边界税)。 */
        const bool can_pre = async_rb && g->prelaunch_want && ((pos + 2u) % 4u) != 0u;
        /* 计数器快照恒做(不只预发射时): 预捕获的 encode 会把 layer_n_comp/comp_x_pending 推到 pos+1 之后,
         * 下一步不是 eval(pos+1) 时(投机 verify 批/回退/捕获失败)靠它回滚。 */
        memcpy(g->snap_n_comp, g->layer_n_comp, sizeof(g->snap_n_comp));
        memcpy(g->snap_n_index_comp, g->layer_n_index_comp, sizeof(g->snap_n_index_comp));
        memcpy(g->snap_x_pending, g->comp_x_pending, sizeof(g->snap_x_pending));
        memcpy(g->snap_x_last, g->comp_x_last_pos, sizeof(g->snap_x_last));   /* comp_push 也推 last_pos, 一并快照(见 graph.h 注) */
        ds4_gpu_token_graph_set_pos(pos + 1u);   /* 预捕获图的取 token 核按 pos+1 相位选槽 */
        if (ds4_gpu_token_graph_precapture_begin() > 0) {
            const bool pok = metal_graph_encode_token_raw_swa(g, model, weights, 0, pos + 1u, true, true);
            const int pend = ds4_gpu_token_graph_precapture_end(pos + 1u, 1, pok ? 1 : 0);
            g->pend_pos = (int64_t)pos + 1;
            if (pend <= 0) metal_graph_token_pending_discard(g);   /* 没成图: encode 副作用照样回滚 */
            else if (can_pre && ds4_gpu_token_graph_prelaunch(pos + 1u, 1) > 0)
                g->prelaunched_pos = (int64_t)pos + 1;
        }
    }
    /* 异步回传时不做全设备同步(那会把刚预发射的下一图也等完); 回传事件即本图完成点 */
    if (ok && !async_rb) ok = ds4_gpu_end_commands() != 0;
#ifdef DS4_STATE_DUMP
    /* 诊断(09-05 token graph 分叉定位): 首个解码 token 跑完后把逐层 KV/压缩器状态整块落盘,
     * 开图/直发两次运行逐文件 cmp, 找第一个不同的层与张量。默认不编译。 */
    {
        static int dumped = 0;
        if (ok && !dumped++) {
            if (system("mkdir -p /tmp/statedump") != 0) fprintf(stderr, "ds4: [statedump] mkdir failed\n");
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                struct { const char *nm; ds4_gpu_tensor *t; } ts[7] = {
                    {"raw", g->layer_raw_cache[il]}, {"acomp", g->layer_attn_comp_cache[il]},
                    {"askv", g->layer_attn_state_kv[il]}, {"assc", g->layer_attn_state_score[il]},
                    {"icomp", g->layer_index_comp_cache[il]}, {"iskv", g->layer_index_state_kv[il]},
                    {"issc", g->layer_index_state_score[il]} };
                for (int k = 0; k < 7; k++) {
                    if (!ts[k].t) continue;
                    const uint64_t nb = ds4_gpu_tensor_bytes(ts[k].t);
                    void *buf = malloc(nb);
                    char p[128]; snprintf(p, sizeof p, "/tmp/statedump/L%02u_%s.bin", il, ts[k].nm);
                    FILE *f = fopen(p, "wb");
                    if (buf && f && ds4_gpu_tensor_read(ts[k].t, 0, buf, nb)) fwrite(buf, 1, nb, f);
                    if (f) fclose(f);
                    free(buf);
                }
            }
            for (uint32_t il = 0; il < DS4_N_LAYER && il < 64u; il++) {   /* 逐层输出 hc 快照 */
                if (!g_sd_hc[il]) continue;
                const uint64_t nb = ds4_gpu_tensor_bytes(g_sd_hc[il]);
                void *buf = malloc(nb);
                char p[128]; snprintf(p, sizeof p, "/tmp/statedump/L%02u_hc.bin", il);
                FILE *f = fopen(p, "wb");
                if (buf && f && ds4_gpu_tensor_read(g_sd_hc[il], 0, buf, nb)) fwrite(buf, 1, nb, f);
                if (f) fclose(f);
                free(buf);
            }
            for (int k = 0; k < 8; k++) {   /* 层 0 内部四点 */
                if (!g_sd_pt[k]) continue;
                const uint64_t nb = ds4_gpu_tensor_bytes(g_sd_pt[k]);
                void *buf = malloc(nb);
                char p[128]; snprintf(p, sizeof p, "/tmp/statedump/L00_pt%d.bin", k);
                FILE *f = fopen(p, "wb");
                if (buf && f && ds4_gpu_tensor_read(g_sd_pt[k], 0, buf, nb)) fwrite(buf, 1, nb, f);
                if (f) fclose(f);
                free(buf);
            }
            fprintf(stderr, "ds4: [statedump] pos=%u 43 层状态已落 /tmp/statedump\n", pos);
        }
    }
#endif

    if (ok && logits) {
        if (async_rb) {
            ok = ds4_gpu_decode_readback_wait() != 0;
            if (ok) {
                memcpy(logits, g->logits_pinned, (size_t)DS4_N_VOCAB * sizeof(float));
                /* 预发射的 pos+1 图消费的就是这个值(同一设备槽, 流序在回传之后) */
                g->prelaunched_token = g->tok_next_pinned ? *g->tok_next_pinned : -1;
                /* 对账: 本图实际消费的 token 必须等于喂进来的 token, 否则取槽链路有病, 立即报 */
                if (g->tok_next_pinned && g->tok_next_pinned[1] != (int32_t)token)
                    fprintf(stderr, "ds4: ★取槽错位★ pos=%u 喂 %d 图消费 %d(设备 argmax %d, launched=%d)\n",
                            pos, token, g->tok_next_pinned[1], g->tok_next_pinned[0], launched);
            }
        } else {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
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

/* 预捕获图作废: 主机计数器回到预捕获前 + 设备侧"待发射"标记清掉。预捕获本身不执行任何 kernel,
 * 设备状态无需回滚。已预发射(prelaunched_pos)的由 eval 开头的认领路另行处理。 */
void metal_graph_token_pending_discard(ds4_gpu_graph *g) {
    if (g->pend_pos < 0) return;
    memcpy(g->layer_n_comp, g->snap_n_comp, sizeof(g->snap_n_comp));
    memcpy(g->layer_n_index_comp, g->snap_n_index_comp, sizeof(g->snap_n_index_comp));
    memcpy(g->comp_x_pending, g->snap_x_pending, sizeof(g->snap_x_pending));
    memcpy(g->comp_x_last_pos, g->snap_x_last, sizeof(g->snap_x_last));
    (void)ds4_gpu_token_graph_pending_discard();
    g->pend_pos = -1;
}
/* 只作废不回滚: 计数器被快照载入/合成填充/整体重置等"外部来源"整个改写时, 预捕获前的快照已过时,
 * 回滚它反而把新值盖掉(09-07 bench 8192 点 "compressed KV cache capacity exceeded" 的根因)。 */
void metal_graph_token_pending_forget(ds4_gpu_graph *g) {
    if (g->pend_pos < 0) return;
    (void)ds4_gpu_token_graph_pending_discard();
    g->pend_pos = -1;
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

#else
void metal_graph_token_pending_discard(ds4_gpu_graph *g) { (void)g; }   /* CPU 构建无 token 图 */
void metal_graph_token_pending_forget(ds4_gpu_graph *g) { (void)g; }
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_imatrix_nonempty_tu; /* 空TU防御(CPU构建) */
