/* core_gpu_batch.c — 层批编码/spec 存档/dspark 状态 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
/* 09-07: 抽成"任一 hc 张量前 n 行"版, 解码路也能写同一格式(h_L%02u.bin 逐行追加) —— 投机 verify 批 vs 纯解码
 * 逐层对拍就靠它(同 prompt 两条路各跑一次, 比 prefill 之后的行)。 */
void eval_hdump_tensor_rows(const ds4_gpu_tensor *hc, uint32_t il, uint32_t n_tokens) {
    const char *dir = ds4_tool_eval_hdump();
    if (!dir || !dir[0] || n_tokens == 0 || !hc) return;
    {
        const uint64_t ev = ds4_gpu_tp_signal_after_batch();
        if (ev) { (void)ds4_gpu_flush_commands(); (void)ds4_gpu_tp_host_wait(ev); }
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const size_t nf = (size_t)n_tokens * hc_dim;
    float *buf = malloc(nf * sizeof(float));
    if (!buf) return;
    if (ds4_gpu_tensor_read(hc, 0, buf, nf * sizeof(float))) {
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
/* --eval-hdump 原始行(09-07 长上下文对拍): 层 0 每 token 入环后的原始 KV 行(fp8 前段 + rope 尾段, DS4_N_HEAD_DIM f32), 记为 L98,
 * 槽位按环绕。解码路与批路各在 store 之后调一次, 逐字节 cmp 定"环里的行"是否同。 */
void eval_hdump_raw_rows(const ds4_gpu_tensor *raw, uint32_t tag, uint32_t slot0, uint32_t n, uint32_t raw_cap) {
    const char *dir = ds4_tool_eval_hdump();
    if (!dir || !dir[0] || !raw || n == 0 || raw_cap == 0) return;
    {
        const uint64_t ev = ds4_gpu_tp_signal_after_batch();
        if (ev) { (void)ds4_gpu_flush_commands(); (void)ds4_gpu_tp_host_wait(ev); }
    }
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    float *buf = malloc((size_t)rowb);
    if (!buf) return;
    char pth[1024];
    snprintf(pth, sizeof pth, "%s/h_L%02u.bin", dir, tag);
    FILE *f = fopen(pth, "ab");
    if (!f) { fprintf(stderr, "ds4: [EVAL_HDUMP] 打不开 %s -- aborting\n", pth); exit(1); }
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t slot = (slot0 + i) % raw_cap;
        if (ds4_gpu_tensor_read(raw, (uint64_t)slot * rowb, buf, rowb) &&
            fwrite(buf, 1, (size_t)rowb, f) != (size_t)rowb) { fprintf(stderr, "ds4: [EVAL_HDUMP] 写 %s 短写\n", pth); exit(1); }
    }
    fclose(f);
    free(buf);
}
/* --eval-hdump 位置边车(09-07 链 39 教训): 每写一批 L99 行就追加这批行的 position(u32) 到 h_pos.bin。投机 verify 批写出的行
 * 含被拒草稿(token 本身就不是 plain 的), 按行号对 plain 会把它误判成"第 0 层注意力分叉"(链 34~39 实撞); 对拍脚本按位置对齐,
 * 同一位置最后一次写出 = 提交行(被拒位置总被下一轮从 pos0 重写)。 */
static FILE *eval_hdump_open(const char *name) {
    const char *dir = ds4_tool_eval_hdump();
    if (!dir || !dir[0]) return NULL;
    char p[1024];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "ab");
    if (!f) { fprintf(stderr, "ds4: [EVAL_HDUMP] 打不开 %s -- aborting\n", p); exit(1); }
    return f;
}
void eval_hdump_pos(uint32_t pos0, uint32_t n) {
    FILE *f = n ? eval_hdump_open("h_pos.bin") : NULL;
    if (!f) return;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t v = pos0 + i;
        if (fwrite(&v, sizeof v, 1, f) != 1) { fprintf(stderr, "ds4: [EVAL_HDUMP] 写 h_pos.bin 短写\n"); exit(1); }
    }
    fclose(f);
}
/* 主机侧 logits 行(L96, 每行 DS4_N_VOCAB f32): 解码路 1 行 / verify 批 n 行, 与 L99/h_pos 同序。层出口全同而 logits 不同 =
 * 输出头两条路不同轨(批头 rows 核 vs 单 token 核), 温 0 近平局时 argmax 翻转 —— 这是逐层 hc 对拍看不见的最后一段。 */
void eval_hdump_logits_rows(const float *logits, uint32_t n) {
    FILE *f = (n && logits) ? eval_hdump_open("h_L96.bin") : NULL;
    if (!f) return;
    const size_t nf = (size_t)n * DS4_N_VOCAB;
    if (fwrite(logits, sizeof(float), nf, f) != nf) { fprintf(stderr, "ds4: [EVAL_HDUMP] 写 h_L96.bin 短写\n"); exit(1); }
    fclose(f);
}

void eval_hdump_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens) {
    eval_hdump_tensor_rows(g->batch_cur_hc, il, n_tokens);
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
    bool ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) eval_hdump_tensor_rows(g->batch_after_attn_hc, 50u + il, n_tokens);   /* --eval-hdump: 注意力块出口 L50+il */
    if (ok) ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    /* DSpark prefill 抓取(层出口 HC 均值)与建窗: prompt 每 token 的 main_kv 进环形窗,
     * 与官方 prefill(start_pos==0 只建 KV)语义一致 */
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
    /* 09-07: 只有"emit 用了被拒行"(t_e ≥ acc)的层被回滚到轮前, 这里把接受位 t < acc 的行重推进环(输入行 = verify 时
     * 捕获的 attn_norm 行, 与解码同核同序); 因 acc ≤ t_e, 重放段内不会再到 emit 位。其余层 restore 已只改计数。 */
    (void)model; (void)weights;
    bool ok = true;
    for (uint32_t il = 0; ok && il < (uint32_t)DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0 || g->spec_emit_t[il] < (int8_t)1 || (uint32_t)g->spec_emit_t[il] < acc) continue;
        if (!g->spec_comp_rows_kv[il]) { fprintf(stderr, "ds4: spec 快进 L%u 没有捕获行\n", il); return false; }
        for (uint32_t t = 0; ok && t < acc; t++) {
            ds4_gpu_tensor *xrow = metal_graph_tensor_row_view(g->spec_comp_rows_kv[il], t, DS4_N_EMBD);
            ok = xrow && metal_graph_comp_push_row(g, il, ratio, pos0 + t, xrow);
            if (xrow) ds4_gpu_tensor_free(xrow);
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
    if (pos0 + n <= g->raw_cap) return true;   /* 还没绕过环: 这些槽位上没有窗内旧行, 无需保存(短上下文省 43~86 次拷贝/轮) */
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
    if (from >= to || g->raw_cap == 0 || pos0 + to <= g->raw_cap) return true;   /* 与 snapshot 同条件 */
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

/* 批内 emit 位: (pos0+t+1) % ratio == 0 的 t; 无则 -1 */
static int spec_emit_index(uint32_t ratio, uint32_t pos0, uint32_t k) {
    for (uint32_t t = 0; t < k; t++) if (((pos0 + t + 1u) % ratio) == 0u) return (int)t;
    return -1;
}
bool metal_graph_dspark_state_snapshot(ds4_gpu_graph *g, uint32_t pos0, uint32_t k) {
    g->spec_pos0 = pos0; g->spec_k = k;
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        g->spec_emit_t[il] = -1;
        if (ratio == 0 || !g->layer_attn_state_kv[il]) continue;
        g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
        g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
        g->spec_prefix1_x_pending[il] = g->comp_x_pending[il];
        g->spec_prefix1_x_last[il] = g->comp_x_last_pos[il];
        const int te = spec_emit_index(ratio, pos0, k);
        g->spec_emit_t[il] = (int8_t)te;
        /* 无 emit: state 不动, 只需计数。emit 在 t=0: 首候选必接受 ⇒ 永不回滚, 也不用拷。 */
        if (te < 1) continue;
        if (!g->spec_prefix1_attn_state_kv[il]) { fprintf(stderr, "ds4: spec 快照 L%u 缺缓冲(enable_mtp?)\n", il); return false; }
        const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        /* 攒行环: emit 之后的行会从环头覆盖旧块的 pending 行, 回滚要用 ⇒ 存 pending 段 [last+1-n, last](环上连续) */
        {
            const uint32_t n = g->comp_x_pending[il];
            if (n && g->comp_x_ring[il]) {
                const uint64_t rowb = (uint64_t)DS4_N_EMBD * sizeof(float);
                if (!g->spec_ring_save[il]) g->spec_ring_save[il] = ds4_gpu_tensor_alloc((uint64_t)ratio * rowb);
                const uint32_t r0 = (g->comp_x_last_pos[il] + 1u - n) % ratio;
                if (!g->spec_ring_save[il] ||
                    !ds4_gpu_tensor_copy(g->spec_ring_save[il], 0, g->comp_x_ring[il], (uint64_t)r0 * rowb, (uint64_t)n * rowb))
                    return false;
            }
        }
        if (!ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) ||
            !ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes)) return false;
        if (g->spec_prefix1_index_state_kv[il] && g->layer_index_state_kv[il]) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            if (!ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                     g->layer_index_state_kv[il], 0, ib) ||
                !ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                     g->layer_index_state_score[il], 0, ib)) return false;
        }
    }
    return true;
}

bool metal_graph_dspark_state_restore(ds4_gpu_graph *g, uint32_t acc) {
    metal_graph_token_pending_forget(g);   /* verify 前已作废; 这里再保一次, 计数器回到轮前快照 */
    const uint32_t pos0 = g->spec_pos0;
    if (acc == 0 || acc > g->spec_k) { fprintf(stderr, "ds4: spec restore: acc %u 非法(k %u)\n", acc, g->spec_k); return false; }
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0 || !g->layer_attn_state_kv[il]) continue;
        const int te = g->spec_emit_t[il];
        if (te < 0) {   /* 批内无 emit: 环里 t<acc 的行已在正确槽位, 被拒行下轮原槽覆盖 ⇒ 只截计数 */
            g->comp_x_pending[il] = g->spec_prefix1_x_pending[il] + acc;
            g->comp_x_last_pos[il] = pos0 + acc - 1u;
            continue;
        }
        if ((uint32_t)te < acc) {   /* emit 位被接受: emit 正确; 之后的行 t_e+1..acc-1 仍攒着 */
            g->comp_x_pending[il] = acc - 1u - (uint32_t)te;
            g->comp_x_last_pos[il] = pos0 + acc - 1u;
            continue;
        }
        /* emit 用了被拒行(t_e ≥ acc ≥ 1): state/计数/环回到轮前, 快进再重推 t<acc */
        const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        g->layer_n_comp[il] = g->spec_prefix1_n_comp[il];
        {
            const uint32_t n = g->spec_prefix1_x_pending[il];
            g->comp_x_pending[il] = n;
            g->comp_x_last_pos[il] = g->spec_prefix1_x_last[il];
            if (n && g->comp_x_ring[il] && g->spec_ring_save[il]) {
                const uint64_t rowb = (uint64_t)DS4_N_EMBD * sizeof(float);
                const uint32_t r0 = (g->comp_x_last_pos[il] + 1u - n) % ratio;
                if (!ds4_gpu_tensor_copy(g->comp_x_ring[il], (uint64_t)r0 * rowb, g->spec_ring_save[il], 0, (uint64_t)n * rowb))
                    return false;
            }
        }
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
typedef int ds4_core_gpu_batch_nonempty_tu; /* 空TU防御(CPU构建) */
