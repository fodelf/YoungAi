/* core_session_eval.c — 多会话批 eval/span (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
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
