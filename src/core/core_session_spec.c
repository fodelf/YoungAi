/* core_session_spec.c — 投机 eval(纯平解码收束) (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
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

