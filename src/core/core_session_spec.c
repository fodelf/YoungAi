/* core_session_spec.c — 投机 eval(纯平解码收束) (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
/* --spec 投机账(09-07): 进程级累计, 生成结束由 CLI 打印(ds4_spec_stats_print)。速度判决要的是轮成本
 * 分解(draft/verify/恢复)和逐位接受率 p_i(候选 i 被验证时命中的比例, 只在其前全接受时才算被验证),
 * 单看 t/s 分不清是 drafter 不准还是 verify 太贵。 */
static struct {
    uint64_t rounds_spec, rounds_plain, tok_spec, tok_plain, crash_rounds;
    double ms_spec, ms_plain, ms_draft, ms_verify, ms_restore;
    uint64_t pos_test[8], pos_hit[8], k_hist[8];
} g_spec_stats;
void ds4_spec_stats_print(void) {
    const uint64_t R = g_spec_stats.rounds_spec, R0 = g_spec_stats.rounds_plain;
    if (R + R0 == 0) return;
    fprintf(stderr, "ds4: spec 账: 投机轮 %llu (token %llu, 均接受 %.2f, %.1f ms/轮 = %.1f ms/token)",
            (unsigned long long)R, (unsigned long long)g_spec_stats.tok_spec,
            R ? (double)g_spec_stats.tok_spec / (double)R : 0.0,
            R ? g_spec_stats.ms_spec / (double)R : 0.0,
            g_spec_stats.tok_spec ? g_spec_stats.ms_spec / (double)g_spec_stats.tok_spec : 0.0);
    fprintf(stderr, " | 纯解码轮 %llu (%.1f ms/token)", (unsigned long long)R0,
            R0 ? g_spec_stats.ms_plain / (double)R0 : 0.0);
    fprintf(stderr, " | 每轮 draft %.1f verify %.1f 恢复 %.1f ms | 崩轮 %llu\n",
            R ? g_spec_stats.ms_draft / (double)R : 0.0, R ? g_spec_stats.ms_verify / (double)R : 0.0,
            R ? g_spec_stats.ms_restore / (double)R : 0.0, (unsigned long long)g_spec_stats.crash_rounds);
    fprintf(stderr, "ds4: spec 逐位接受率:");
    for (int i = 0; i < 8; i++)
        if (g_spec_stats.pos_test[i])
            fprintf(stderr, " p%d=%.3f(n=%llu)", i + 1,
                    (double)g_spec_stats.pos_hit[i] / (double)g_spec_stats.pos_test[i],
                    (unsigned long long)g_spec_stats.pos_test[i]);
    fprintf(stderr, " | 候选数分布:");
    for (int k = 0; k < 8; k++)
        if (g_spec_stats.k_hist[k]) fprintf(stderr, " k%d×%llu", k, (unsigned long long)g_spec_stats.k_hist[k]);
    fprintf(stderr, "\n");
}
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
    /* DSpark 投机主循环(--spec, 2026-08-18): draft 块(5)+bonus 走 6 位
     * verify 批; 接受链 argmax 对照; 部分接受= state 恢复+接受位重放(官方
     * checkpoint-restore 口径)。verify 批复用 batch 层包装 ⇒ mh 抓取/drafter 建窗自动。 */
    if (e->dspark.ready && s->graph.dspark_capture && s->spec_greedy &&
        g_ds4_spec_enabled) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        int n_acc = 0;
        accepted[n_acc++] = first_token;
        static float *row_logits = NULL;
        if (!row_logits) row_logits = xmalloc(6ull * DS4_N_VOCAB * sizeof(float));
        /* 候选数可调(2026-08-21 调度杠杆): verify 是字节受限, 而专家并集随候选数增长
         * (实测 6 候选=22.1/36 唯一专家)。每字节产出 acc/bytes 在 k=3-4 处更优。 */
        /* 3(2026-08-21 扫参): 每 token 毫秒 k=6 48 / k=4 44 / k=3 38.7 / k=2 39.8。
         * verify 是字节受限且专家并集随候选数增长(实测 6 候选=22.1 唯一专家), 多草稿
         * 的边际接受收益跑不过边际字节成本。verify 语义与 k 无关 ⇒ 质量不受影响。 */
        const uint32_t spec_k = 3u;
        while (n_acc < max_tokens && n_acc + (int)DS4_DSPARK_BLK + 1 <= accepted_cap) {
            int next = 0; float best = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (s->logits[v] > best) { best = s->logits[v]; next = (int)v; }
            if (next == eos_token) break;
            int cand[6];
            cand[0] = next;
            int ids[DS4_DSPARK_BLK] = {0};
            const uint32_t pos_now = (uint32_t)s->checkpoint.len;
            /* 置信调度(论文 2607.05147 Alg.1 单请求版, 随 --spec 恒开):
             * 草稿一次出满块并拿到逐位置信 c_i; 前缀存活率 a_j = ∏_{i<=j} c_i 就是"验第 j 个
             * 候选能多拿到的期望 token 数"。多验一个候选的边际成本是 m 毫秒(在线最小二乘
             * 从 (k-1, verify_ms) 拟合), 当前吞吐 T = 已接受 token / 已用毫秒。只有
             * a_j > T*m 时这个候选才划算 —— 这正是把"能不能白送"这个物理事实写进调度。 */
            static const int sched = 1;   /* 调度仲裁是 --spec 的一部分: 文本难预测时自动回纯解码 */
            static double sch_tok = 0.0, sch_ms = 0.0;          /* 在线吞吐 */
            static double rg_n = 0, rg_x = 0, rg_y = 0, rg_xx = 0, rg_xy = 0;  /* 边际回归 */
            static double sch_m = 12.0;                          /* 每候选边际 ms */
            /* 在线校准(论文的 STS 在线版): 原始置信头没针对"q2 drafter + 贪心验证"标定,
             * 用自己的接受结果做逐位置乘性校正 g_j = 实测接受 / 预测和。低估就放大, 高估
             * 就收缩 —— 调度阈值才对得上真实的边际收益。 */
            static double cal_sum[DS4_DSPARK_BLK] = {0};
            static double cal_hit[DS4_DSPARK_BLK] = {0};
            static uint32_t cal_n[DS4_DSPARK_BLK] = {0};
            /* 投机/纯解码的墙钟仲裁("谁快用谁", 08-21)已删(09-07): 它让温 0 的输出随时钟变(两次同参跑, 纯解码轮
             * 5 vs 32, 文本从第 421 字节分叉), 温 0 必须可复现; 投机在同底座 drafter 下稳定快于纯解码, 慢就是核的
             * 问题, 不靠回退遮。 */
            const double mode_t0 = now_sec();

            float conf[DS4_DSPARK_BLK] = {0};
            /* 草稿位数 = 候选上限 − 1 = 3(09-07): verify 批 ≤ 4 token 才走 VQ fused2 解码即乘核(fuse_max=4),
             * 5~6 token 掉进 prefill 的 dequant+cuBLAS 路(实测 verify 393 ms/轮)。并集去重批核落地后再放开。 */
            uint32_t draft_n = sched ? 3u : spec_k - 1u;
            const double t_draft0 = now_sec();
            if (!metal_graph_dspark_step_n(&s->graph, &e->model, &e->weights, &e->dspark,
                                           next, pos_now - 1u, ids, draft_n,
                                           sched ? conf : NULL)) {
                break;
            }
            g_spec_stats.ms_draft += (now_sec() - t_draft0) * 1e3;
            {   /* 崩轮指纹(08-21): 草稿全 0 = drafter 输入态坏了, 整轮白验 */
                int allz = 1;
                for (uint32_t i = 0; i < draft_n; i++) if (ids[i] != 0) { allz = 0; break; }
                if (allz) g_spec_stats.crash_rounds++;
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
                /* 09-07: 置信门控停用, 候选数钉死 = 草稿位 + 1。同底座 drafter 每位接受率 ~0.9, 而门控的阈值
                 * T·m 随 verify 变快自动抬高, 实测把 k 压到 2(均接受 3.1 → 2.4, 36 → 31 t/s)。置信头留着当账。 */
                round_k = draft_n + 1u;
            }
            for (uint32_t i = 0; i + 1u < round_k; i++) cand[1 + i] = ids[i];
            g_spec_stats.k_hist[round_k < 8u ? round_k : 7u]++;
            /* ★先作废上一步解码预编码的 pos+1 图, 再快照★(09-07 定罪, 2.8K 意语分叉 + 1M "攒行 5 超 ratio 4" 同根):
             * 预编码把 comp_x_pending/last_pos/n_comp 推到 pos+1 之后, 回滚原来发生在 verify 批里 —— 晚于这里的快照 ⇒
             * 快照存的是污染值, 部分接受回滚后再快进一行, 攒行多 1: 凑巧 =ratio 时压缩块多混一行陈旧行(温 0 分叉),
             * 超 ratio 时环拷贝越界/push 报错。 */
            metal_graph_token_pending_discard(&s->graph);
            /* 上下文末尾(09-07 1M 尺实撞 "token span exceeds context"): verify 批 round_k 个位置放不下就不再投机,
             * 交回调用方按单 token 走完最后几位。 */
            if (pos_now + round_k > (uint32_t)s->ctx_size) break;
            const double t_snap0 = now_sec();
            if (!metal_graph_dspark_state_snapshot(&s->graph, pos_now, round_k)) {
                break;
            }
            const double t_vfy0 = now_sec();
            g_spec_stats.ms_restore += (t_vfy0 - t_snap0) * 1e3;
            s->graph.spec_comp_capture = 1;   /* verify 批捕获压缩器输入行(快进用) */
            if (ds4_session_verify_batch_argmax(s, cand, round_k, pos_now,
                                                0u, (uint32_t)DS4_N_LAYER - 1u,
                                                NULL, row_logits, err, errlen) != 0) {
                s->graph.spec_comp_capture = 0;
                s->checkpoint_valid = false;
                return -1;
            }
            s->graph.spec_comp_capture = 0;
            int acc = 1;
            for (int i = 0; i + 1 < (int)round_k; i++) {
                int am = 0; float bb = -1e30f;
                const float *row = row_logits + (uint64_t)i * DS4_N_VOCAB;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (row[v] > bb) { bb = row[v]; am = (int)v; }
                g_spec_stats.pos_test[i < 8 ? i : 7]++;
                if (am != cand[i + 1]) break;
                g_spec_stats.pos_hit[i < 8 ? i : 7]++;
                acc++;
            }
            const double t_rst0 = now_sec();
            g_spec_stats.ms_verify += (t_rst0 - t_vfy0) * 1e3;
            if (acc < (int)round_k &&
                !metal_graph_spec_raw_restore(&s->graph, pos_now, (uint32_t)acc, round_k)) {
                if (errlen) snprintf(err, errlen, "spec raw KV restore failed");
                s->checkpoint_valid = false;
                return -1;
            }
            if (!metal_graph_dspark_win_commit(&s->graph, pos_now, (uint32_t)acc)) {
                if (errlen) snprintf(err, errlen, "dspark window commit failed");
                s->checkpoint_valid = false;
                return -1;
            }
            /* timeline 已 commit 6 位 → 截到接受数 */
            s->checkpoint.len = (int)(pos_now + (uint32_t)acc);
            if (acc < (int)round_k) {
                if (!metal_graph_dspark_state_restore(&s->graph, (uint32_t)acc)) break;
                if (!metal_graph_spec_comp_fastforward(&s->graph, &e->model, &e->weights,
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
            g_spec_stats.ms_restore += (now_sec() - t_rst0) * 1e3;
            {
                const double dt_round = (now_sec() - mode_t0) * 1e3;
                g_spec_stats.rounds_spec++; g_spec_stats.tok_spec += (uint64_t)acc; g_spec_stats.ms_spec += dt_round;
            }
            if (sched) {
                /* 校准喂数: 草稿位 j 被真正验证过(前缀全接受)才计数; j = acc-1 是被拒的那位。 */
                for (uint32_t j = 0; j + 1u < round_k && j < (uint32_t)acc; j++) {
                    cal_n[j]++; cal_sum[j] += (double)conf[j];
                    if ((int)j < acc - 1) cal_hit[j] += 1.0;
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
            /* mh: verify/重放批的末接受位 → dspark_main_hidden(3 slot 连续拷贝)。 */
            for (uint32_t sl = 0; sl < 3u; sl++)
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
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

