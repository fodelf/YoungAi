/* train_gate.c — wt2 门 + 选轮(2026-10-10, 原 z_nightly_spark.sh kd_pick/stage_docgate + v41_judge.sh engine 臂的 C 版)。总述见 train_internal.h。
 * 门("不忘老本事"): 同一把判决料(WikiText-2 前 TR_GATE_NTOK token), 学生 = 引擎 --score-ids(部署同路), 两臂 ②态 与 ②+③态, 教师锚 = FP 教师的
 *   top-K 落盘(gguf/v41judge/teacher_g7_wt2_n512.bin, 发布包随带; 没有 HF 出厂权重的机器算不出别的), 判决器 = anchor_metrics 五指标。
 *   过门 = Σmin 相对 ② 态退不过 0.5pp 且 Mean KLD 涨不过 3%(Same top / 中位 KLD 照打不判: 10-02 实撞 top-1 一致率"过"而 Σmin 退 1.4pp)。
 * 选轮(10-02 用户: "不要从结果看, 从技术指标看"): 过门的轮里留出损失最低者; 不把两类加权成一个分(换算比例没有依据), 门是底线。
 * 与脚本的两处不同: ① ② 臂每趟只跑一次(它不依赖 ckpt, 脚本每轮重跑一次白装一次模型); ② 学生文件落在趟目录 eval/ 下(脚本写 gguf/v41judge/, 多趟互相盖)。
 * 判决全文落 eval/gate_ckpt_eNN.txt(先 ② 后 ②+③), 记录页(train_runs.c gate_json)按同一口径读。 */
#include "train_internal.h"
#include <dirent.h>
#include <sys/stat.h>

#define TR_SCORE_BUDGET_MB 40000   /* 打分路的引擎预算 + 权重缓存(v41_judge.sh 同值: 打分只做预填, 不需要服务那么多常驻) */
#define TR_SCORE_WCACHE_MB 88000
#define TR_WT2_IDS "gguf/go-onebit/g7/wt2.ids"
#define TR_WT2_TEACHER "gguf/v41judge/teacher_g7_wt2_n512.bin"
#define TR_ANCHOR_METRICS "gguf-tools/bench/anchor_metrics"

typedef struct { double smin, kld; bool ok; } arm_read;

/* anchor_metrics 全文里取 "分布还原率 Σmin = X" 与 "Mean KLD = Y" */
static arm_read read_arm(const char *text) {
    arm_read r = { 0, 0, false };
    /* 行是 "分布还原率 Σmin = X" / "Mean KLD        = Y"(对齐用的多个空格): 只按标签找, 等号前的空白交给 sscanf 的格式空白吞 */
    const char *s = text ? strstr(text, "Σmin") : NULL, *k = text ? strstr(text, "Mean KLD") : NULL;
    r.ok = s && k && sscanf(s, "Σmin = %lf", &r.smin) == 1 && sscanf(k, "Mean KLD = %lf", &r.kld) == 1 && r.kld > 0;
    return r;
}

/* 一臂: ./ds4 --score-ids → stu.bin, 再 anchor_metrics → 文本(返回 malloc; NULL = 失败) */
static char *score_arm(const char *run_dir, const char *tag, const char *mdl, const char *zch, char *const extra[], int nextra, const char *pt,
                       const char *ids_n, FILE *log, tr_cancel_fn cancel, void *ud) {
    char stu[TR_PATH + 320], lp[TR_PATH + 320], am[TR_PATH + 64], ref[TR_PATH + 64], ids[TR_PATH + 64], bud[16], wc[16];
    snprintf(stu, sizeof stu, "%s/eval/stu_%s.bin", run_dir, tag);
    snprintf(lp, sizeof lp, "%s/eval/score_%s.log", run_dir, tag);
    snprintf(am, sizeof am, "%s/" TR_ANCHOR_METRICS, tr_root); snprintf(ref, sizeof ref, "%s/" TR_WT2_TEACHER, tr_root); snprintf(ids, sizeof ids, "%s/" TR_WT2_IDS, tr_root);
    snprintf(bud, sizeof bud, "%d", TR_SCORE_BUDGET_MB); snprintf(wc, sizeof wc, "%d", TR_SCORE_WCACHE_MB);
    const char *argv[64]; int na = tr_ds4_base_argv(argv, 40, mdl, zch, extra, nextra);
    argv[na++] = "--mem-budget-mb"; argv[na++] = bud; argv[na++] = "--weight-cache-mb"; argv[na++] = wc;
    argv[na++] = "--score-ids"; argv[na++] = ids_n; argv[na++] = "--score-out"; argv[na++] = stu;
    if (pt) { argv[na++] = "--posttrain"; argv[na++] = pt; }
    argv[na] = NULL;
    tr_logf(log, "门: 学生臂 %s(%s) 打分 %d token", tag, pt ? "②+③" : "②", TR_GATE_NTOK);
    const pid_t pid = tr_child_spawn(argv, lp, false);
    const int rc = pid > 0 ? tr_child_wait(pid, cancel, ud, log) : -1;
    struct stat st;
    if (rc != 0 || stat(stu, &st) || !st.st_size) { tr_logf(log, "★学生臂 %s 失败(rc=%d, 见 %s)★", tag, rc, lp); return NULL; }
    snprintf(lp, sizeof lp, "%s/eval/metrics_%s.txt", run_dir, tag);
    const char *av[] = { am, "--ref-raw", ref, "--ids", ids, "--student", stu, NULL };
    const pid_t p2 = tr_child_spawn(av, lp, false);
    const int rc2 = p2 > 0 ? tr_child_wait(p2, cancel, ud, log) : -1;
    char *text = tr_slurp(lp, NULL);
    if (rc2 != 0 || !text || !read_arm(text).ok) { tr_logf(log, "★判决器失败 %s(rc=%d, 见 %s)★", tag, rc2, lp); free(text); return NULL; }
    remove(stu);   /* 学生 top-K 表 265 MB 一臂, 指标已落 metrics_*.txt, 不留(一趟四臂 1 GB) */
    return text;
}

/* train.log 最后一趟(最后一个 "step 0 " 之后)的 epoch N eval_kl X; 返回轮数, eval[N] 填上(N ≤ 99) */
static int last_run_epochs(const char *run_dir, double eval[100]) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%s/train.log", run_dir);
    char *t = tr_slurp(p, NULL); if (!t) return 0;
    char *start = t;
    for (char *q = t; (q = strstr(q, "step 0 ")); q += 7) if (q == t || q[-1] == '\n') start = q;
    for (int i = 0; i < 100; i++) eval[i] = -1;
    int n = 0; char *sv = NULL;
    for (char *ln = strtok_r(start, "\n", &sv); ln; ln = strtok_r(NULL, "\n", &sv)) {
        unsigned ep; double ev;
        if (sscanf(ln, "epoch %u eval_kl %lf", &ep, &ev) == 2 && ep < 100) { eval[ep] = ev; n++; }
    }
    free(t);
    return n;
}

bool tr_gate_pick(const char *run_dir, const char *mdl, const char *zch, char *const extra[], int nextra, FILE *log, tr_cancel_fn cancel, void *ud, char *pick, size_t pn) {
    pick[0] = 0;
    char ev[TR_PATH + 48], ids_n[TR_PATH + 96], src[TR_PATH + 64], p[TR_PATH + 400];
    snprintf(ev, sizeof ev, "%s/eval", run_dir); mkdir(ev, 0755);
    snprintf(src, sizeof src, "%s/" TR_WT2_IDS, tr_root);
    struct stat st;
    snprintf(p, sizeof p, "%s/" TR_WT2_TEACHER, tr_root);
    if (stat(p, &st)) { tr_logf(log, "★没有教师锚 %s(发布包随带; 要在有 HF 出厂权重的机器上用 v41_judge.sh 出)★", p); return false; }
    snprintf(p, sizeof p, "%s/" TR_ANCHOR_METRICS, tr_root);
    if (stat(p, &st)) { tr_logf(log, "★没有判决器 %s(make -C gguf-tools anchor_metrics)★", p); return false; }
    /* 判决料 = wt2.ids 前 TR_GATE_NTOK 行 */
    snprintf(ids_n, sizeof ids_n, "%s/wt2_n%d.ids", ev, TR_GATE_NTOK);
    { FILE *in = fopen(src, "r"), *out = in ? fopen(ids_n, "w") : NULL; char ln[256]; int k = 0;
      if (!in || !out) { tr_logf(log, "★没有判决料 %s★", src); if (in) fclose(in); return false; }
      while (k < TR_GATE_NTOK && fgets(ln, sizeof ln, in)) { fputs(ln, out); k++; }
      fclose(in); fclose(out);
      if (k < TR_GATE_NTOK) { tr_logf(log, "★判决料只有 %d 行 < %d★", k, TR_GATE_NTOK); return false; } }
    double eval[100]; const int ne = last_run_epochs(run_dir, eval);
    if (!ne) { tr_logf(log, "★train.log 里没有这趟的 epoch 行★"); return false; }
    char *base = score_arm(run_dir, "base", mdl, zch, extra, nextra, NULL, ids_n, log, cancel, ud);
    if (!base) return false;
    const arm_read b = read_arm(base);
    double best = 0; bool any = false;
    snprintf(p, sizeof p, "%s/pick.txt", run_dir); remove(p);   /* 重选先清: 旧的选中不许留给记录页 */
    for (unsigned e = 1; e < 100; e++) {
        char ck[TR_PATH + 64], amp[TR_PATH + 96], tag[32];
        snprintf(ck, sizeof ck, "%s/ckpt_e%02u", run_dir, e); snprintf(amp, sizeof amp, "%s/amp_L39.bin", ck);
        if (stat(amp, &st)) continue;
        if (eval[e] < 0) { tr_logf(log, "选轮 ckpt_e%02u: train.log 最后一趟没有第 %u 轮(上一趟的残留), 不参选", e, e); continue; }
        snprintf(tag, sizeof tag, "ckpt_e%02u", e);
        char *arm = score_arm(run_dir, tag, mdl, zch, extra, nextra, ck, ids_n, log, cancel, ud);
        if (!arm) { free(base); return false; }
        snprintf(p, sizeof p, "%s/eval/gate_ckpt_e%02u.txt", run_dir, e);
        FILE *gf = fopen(p, "w"); if (gf) { fputs(base, gf); fputs("\n", gf); fputs(arm, gf); fclose(gf); }
        const arm_read a = read_arm(arm); free(arm);
        const double dpp = 100.0 * (a.smin - b.smin), rel = a.kld / b.kld - 1.0;
        const bool pass = dpp >= TR_GATE_SMIN_PP && rel <= TR_GATE_KLD_REL;
        tr_logf(log, "选轮 ckpt_e%02u: 留出 %.4f | Σmin %.4f → %.4f(%+.2fpp) KLD %.5f → %.5f(%+.1f%%) | %s", e, eval[e], b.smin, a.smin, dpp, b.kld, a.kld, 100 * rel, pass ? "过门" : "★没过门★");
        if (pass && (!any || eval[e] < best)) { any = true; best = eval[e]; snprintf(pick, pn, "ckpt_e%02u", e); }
    }
    free(base);
    if (!any) { tr_logf(log, "★选轮: 没有一轮过 wt2 门, 这趟不进开关(读数见 eval/gate_*.txt)★"); return true; }
    snprintf(p, sizeof p, "%s/pick.txt", run_dir);
    FILE *pf = fopen(p, "w"); if (pf) { fprintf(pf, "%s\n", pick); fclose(pf); }
    tr_logf(log, "选轮结果: %s(过门的轮里留出损失最低, %.4f)", pick, best);
    return true;
}
