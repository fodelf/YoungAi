/* core_session_fill.c — 合成上下文(2026-09-06): 把会话摆到"已经处理了 N 个 token"的状态而不算它们。
 *
 * 怎么用: ds4-bench --fill-ctx N(见 bench_main.c)。把各层 KV 行数按 N 摆好、缓存与压缩器 state 全部置零、
 * checkpoint 记下这 N 个 token id, 之后从 N 起正常 prefill/decode: 引擎走的核、读的字节、查的行数与真灌了
 * N 个 token 一样, 只是内容是零。用来几分钟内量任意上下文位置(比如 1M)的 prefill/decode 速度 —— 真灌 1M
 * 在这台 Spark 上要 3 小时(09-06 实测)。
 * 为什么能代表速度: 解码/prefill 的开销只看行数(原始窗口固定, top-k 固定 512, indexer 打分扫全部压缩 key),
 * 与内容无关; 唯一内容相关的是 top-k 选到哪些行(KV 散读的局部性): 全零分数下 top-k 按并列规则选最前面的
 * 512 行, 局部性偏好 ⇒ 数字落在真实值的乐观一侧, 报数时要说明。不代表质量: 输出是垃圾, 只能测速。
 * 字段与 ds4_session_load_snapshot 同一套(那里从文件填, 这里填零)。
 * 改了会怎样: 少清一个 state 张量 = 上一段残留(可能是 NaN)污染后续压缩行; 行数不按 floor(N/ratio) 摆 =
 * 下一次 emit 位与压缩块边界错位。 */
#include "core_internal.h"

int ds4_session_fill_synthetic(ds4_session *s, const ds4_tokens *tokens, char *err, size_t errlen) {
    if (!s || !tokens || !tokens->v || tokens->len <= 0) {
        payload_set_err(err, errlen, "invalid synthetic context fill");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "synthetic context fill is not supported on distributed sessions");
        return 1;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "synthetic context fill needs the graph backend");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    const uint32_t n = (uint32_t)tokens->len;
    if ((int)n >= s->ctx_size) {
        payload_set_err(err, errlen, "synthetic context does not fit the session context");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before synthetic fill");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;
    metal_graph_token_pending_forget(g);   /* 合成填充整体改写计数器, 预捕获快照作废 */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (g->layer_raw_cache[il] &&
            !ds4_gpu_tensor_fill_f32(g->layer_raw_cache[il], 0.0f, (uint64_t)g->raw_cap * DS4_N_HEAD_DIM)) {
            payload_set_err(err, errlen, "synthetic fill: raw cache clear failed");
            return 1;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) { g->layer_n_comp[il] = 0; g->layer_n_index_comp[il] = 0; continue; }
        const uint32_t n_comp = n / ratio;   /* emit 位 = (pos+1) 整除 ratio ⇒ N 个 token 出 floor(N/ratio) 行 */
        if (n_comp > g->layer_comp_cap[il]) {
            payload_set_err(err, errlen, "synthetic fill: compressed rows exceed layer capacity");
            return 1;
        }
        const uint64_t comp_f32 = (uint64_t)g->layer_comp_cap[il] * DS4_GPU_COMP_ROW_BYTES / sizeof(float);   /* 按 f32 计数清零 */
        int ok = 1;
        if (g->layer_attn_comp_cache[il]) ok = ok && ds4_gpu_tensor_fill_f32(g->layer_attn_comp_cache[il], 0.0f, comp_f32);
        if (g->layer_attn_state_kv[il]) ok = ok && ds4_gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, layer_attn_state_bytes(ratio) / 4u);
        if (g->layer_attn_state_score[il]) ok = ok && ds4_gpu_tensor_fill_f32(g->layer_attn_state_score[il], 0.0f, layer_attn_state_bytes(ratio) / 4u);
        if (ratio == 4u) {
            if (g->layer_index_comp_cache[il])
                ok = ok && ds4_gpu_tensor_fill_f32(g->layer_index_comp_cache[il], 0.0f,
                                                   (uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM *
                                                   sizeof(uint16_t) / sizeof(float));   /* f16 缓存, 按 f32 计数清零 */
            if (g->layer_index_state_kv[il]) ok = ok && ds4_gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, layer_index_state_bytes(ratio) / 4u);
            if (g->layer_index_state_score[il]) ok = ok && ds4_gpu_tensor_fill_f32(g->layer_index_state_score[il], 0.0f, layer_index_state_bytes(ratio) / 4u);
        }
        if (!ok) {
            payload_set_err(err, errlen, "synthetic fill: compressed cache clear failed");
            return 1;
        }
        g->layer_n_comp[il] = n_comp;
        g->layer_n_index_comp[il] = (ratio == 4u) ? n_comp : 0u;
    }
    metal_graph_comp_pending_clear(g);   /* 压缩块边界上(N 整除 ratio 时)无攒行; 否则攒行为零内容, 与 state 同义 */
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator after synthetic fill");
        return 1;
    }
    token_vec_free(&s->checkpoint);
    memset(&s->checkpoint, 0, sizeof(s->checkpoint));
    for (int i = 0; i < tokens->len; i++) token_vec_push(&s->checkpoint, tokens->v[i]);
    s->checkpoint_valid = true;
    return 0;
#endif
}
