/* core_gpu_dspark.c — dspark 步进 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
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
typedef int ds4_core_gpu_dspark_nonempty_tu; /* 空TU防御(CPU构建) */
