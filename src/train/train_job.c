/* train_job.c — 训练/出题作业线程(2026-10-10, 原 train_cycle.sh + z_nightly_spark.sh stage_train/stage_gen 的 C 版)。总述见 train_internal.h。
 * 一次一条作业: 停模型子进程 → fork ./ds4 --ptrain(或 --gen-jobs) → (训练)wt2 门 + 选轮 → 模型子进程按 gguf/serve_pick.txt 装回来。
 * ★训完只产出, 不挂★(10-10 用户: "训练完就是训练完不要直接挂"): 不碰 serve_pick.txt, 装回来的还是训前那套; ③ 关联训练时装着的侧车
 *   (③ 的 base.fnv = 它的指纹), 聊天页切到那份侧车、打开"后训练"开关才挂, 切走就卸。
 * 训练用的 ①② = serve_pick.txt 里的那对(工作台选了哪套就训哪套; ③ 是对着这对解的, 对不上引擎拒挂); 第 3 行起只取 --engram-dir 给 ./ds4
 *   (发布包里模型在 gguf/hub, n-gram 表不在转换机的路径上, 训练器/打分器同样要它); ③ 不继承(起点恒为 ①+②)。
 * 出错会怎样: 训练失败/被停/没过门, 模型照样装回来, 页面回到能聊; 原因在 ui_logs/cycle_*.log 与那趟的 train.out。 */
#include "train_internal.h"
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define TR_TRAIN_BUDGET_MB 110000   /* 训练/出题的引擎预算(与起服同: 机器空着时整机给它) */

static struct {
    pthread_mutex_t mu; bool active, gen, stop; tr_job_phase phase; pid_t child; int port;
    char run[256], run_abs[TR_PATH + 320], data[TR_PATH], epochs[64], layers[64], lr[64], extra[512]; int rounds;
} g = { PTHREAD_MUTEX_INITIALIZER, false, false, false, TR_JOB_NONE, 0, 0, "", "", "", "", "", "", "", 1 };

static void set_phase(tr_job_phase p, pid_t child) { pthread_mutex_lock(&g.mu); g.phase = p; g.child = child; pthread_mutex_unlock(&g.mu); }
static int want_stop(void *ud) { (void)ud; pthread_mutex_lock(&g.mu); const bool s = g.stop; pthread_mutex_unlock(&g.mu); return s; }

void tr_job_status(tr_job_info *o) {
    pthread_mutex_lock(&g.mu);
    o->active = g.active; o->gen = g.gen; o->phase = g.phase; o->child = g.child; snprintf(o->run, sizeof o->run, "%s", g.run);
    pthread_mutex_unlock(&g.mu);
}
bool tr_job_owns(const char *run_dir_abs) {
    pthread_mutex_lock(&g.mu); const bool r = g.active && g.run_abs[0] && !strcmp(g.run_abs, run_dir_abs); pthread_mutex_unlock(&g.mu); return r;
}

int tr_ds4_base_argv(const char *argv[], int max, const char *mdl, const char *zch, char *const extra[], int nextra) {
    int na = 0;
    argv[na++] = "./ds4";
#ifdef __APPLE__
    argv[na++] = "--metal";
#else
    argv[na++] = "--cuda";
#endif
    argv[na++] = "-m"; argv[na++] = mdl;
    if (zch && zch[0]) { argv[na++] = "--zchain"; argv[na++] = zch; }
    for (int i = 0; i < nextra && na < max - 1; i++) argv[na++] = extra[i];
    return na;
}

/* 训练配置(与 z_nightly pt_cfg_write 同一份键; 秩 64 / batch 4 / top-K 64 / maxlen 1024 / 探针 6 题 96 token 是 10-01~10-03 定下的训练器缺省) */
static bool write_cfg(const char *out, const char *data_abs, const char *hold, FILE *log) {
    char p[TR_PATH + 340]; snprintf(p, sizeof p, "%s/ptrain.cfg", out);
    FILE *f = fopen(p, "w"); if (!f) { tr_logf(log, "★写不了 %s★", p); return false; }
    (void)data_abs;   /* data= 写相对仓库根的路径(./ds4 在仓库根跑): 记录页与 /api/train/runs 按它匹配趟, z_nightly 写的也是相对路径 */
    fprintf(f, "data=%s\nout=%s\nhold=%s\nlayers=%s\nrank=64\nlr=%s\nepochs=%s\nbatch=4\ntopk=64\nmaxlen=1024\nprobe_n=6\nprobe_tok=96\nepoch_tok=0\n",
            g.data, out, hold, g.layers, g.lr, g.epochs);
    for (const char *s = g.extra; *s; ) { const size_t L = strcspn(s, ","); if (L) fprintf(f, "%.*s\n", (int)L, s); s += L + (s[L] == ','); }   /* k=v,k=v → 一行一个 */
    fclose(f);
    return true;
}

static int run_ds4(const char *argv[], const char *out_log, FILE *log) {
    const pid_t pid = tr_child_spawn(argv, out_log, false);
    if (pid < 0) { tr_logf(log, "★fork 失败★"); return -1; }
    pthread_mutex_lock(&g.mu); g.child = pid; pthread_mutex_unlock(&g.mu);
    const int rc = tr_child_wait(pid, want_stop, NULL, log);
    pthread_mutex_lock(&g.mu); g.child = 0; pthread_mutex_unlock(&g.mu);
    return rc;
}

static void do_train(const char *mdl, const char *zch, char *const eng[], int neng, FILE *log) {
    char data_abs[TR_PATH + 340], hold[TR_PATH + 64], out[TR_PATH + 320], cfg[TR_PATH + 340], tout[TR_PATH + 340], bud[16];
    snprintf(data_abs, sizeof data_abs, "%s/%s", tr_root, g.data);
    snprintf(hold, sizeof hold, "%s/gguf-tools/data/posttrain/hold.jsonl", tr_root);
    struct stat st;
    if (stat(hold, &st)) { tr_logf(log, "★没有保持料 %s(通用题 + 部署态回答, 发布包随带)★", hold); return; }
    /* 目录名带发车时间: 同料同参数再训一次是新目录, 不盖掉上一次的 ckpt / 门读数 */
    const char *nm = strrchr(g.data, '/'); nm = nm ? nm + 1 : g.data;
    char name[256]; snprintf(name, sizeof name, "%s", nm); { char *dot = strstr(name, ".jsonl"); if (dot) *dot = 0; }
    time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm);
    snprintf(out, sizeof out, "%s/kd-%s-L%s-lr%s-e%s-%02d%02d%02d%02d", tr_ftd, name, g.layers, g.lr, g.epochs, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
    mkdir(tr_ftd, 0755); mkdir(out, 0755);
    pthread_mutex_lock(&g.mu); snprintf(g.run, sizeof g.run, "%s", strrchr(out, '/') + 1); snprintf(g.run_abs, sizeof g.run_abs, "%s", out); pthread_mutex_unlock(&g.mu);
    if (!write_cfg(out, data_abs, hold, log)) return;
    tr_logf(log, "训练: %s → 层 %s lr %s 轮 %s → %s; MemAvailable %ld MB", g.data, g.layers, g.lr, g.epochs, out, tr_mem_avail_mb());
    snprintf(cfg, sizeof cfg, "%s/ptrain.cfg", out); snprintf(tout, sizeof tout, "%s/train.out", out); snprintf(bud, sizeof bud, "%d", TR_TRAIN_BUDGET_MB);
    const char *argv[64]; int na = tr_ds4_base_argv(argv, 40, mdl, zch, eng, neng);
    argv[na++] = "--mem-budget-mb"; argv[na++] = bud; argv[na++] = "--no-dspark"; argv[na++] = "--ptrain"; argv[na++] = cfg; argv[na] = NULL;
    set_phase(TR_JOB_TRAIN, 0);
    const int rc = run_ds4(argv, tout, log);
    if (rc != 0) { tr_logf(log, "★训练失败(rc=%d, 见 %s/train.out)★", rc, out); return; }
    set_phase(TR_JOB_GATE, 0);
    char pick[64];
    if (!tr_gate_pick(out, mdl, zch, eng, neng, log, want_stop, NULL, pick, sizeof pick)) { tr_logf(log, "★门没跑完(见上)★"); return; }
    tr_logf(log, "TRAIN_DONE %s 选中 %s (关联侧车 %s; 没挂, 聊天页开关挂)", out, pick[0] ? pick : "无(没过门)", zch[0] ? zch : "无");
}

static void do_gen(const char *mdl, const char *zch, char *const eng[], int neng, FILE *log) {
    char dir[TR_PATH + 340], jobs[TR_PATH + 400], gout[TR_PATH + 400], bud[16]; unsigned nj = 0;
    snprintf(dir, sizeof dir, "%s/%s", tr_root, g.data);
    pthread_mutex_lock(&g.mu); snprintf(g.run, sizeof g.run, "%s", g.data); g.run_abs[0] = 0; pthread_mutex_unlock(&g.mu);
    if (!tr_gen_jobs(dir, g.rounds, jobs, sizeof jobs, &nj, log)) return;
    if (nj) {
        snprintf(gout, sizeof gout, "%s/gen/gen.out", dir); snprintf(bud, sizeof bud, "%d", TR_TRAIN_BUDGET_MB);
        const char *argv[64]; int na = tr_ds4_base_argv(argv, 40, mdl, zch, eng, neng);
        argv[na++] = "--mem-budget-mb"; argv[na++] = bud; argv[na++] = "--no-dspark"; argv[na++] = "--gen-jobs"; argv[na++] = jobs; argv[na] = NULL;
        set_phase(TR_JOB_GEN, 0);
        const int rc = run_ds4(argv, gout, log);
        if (rc != 0) { tr_logf(log, "★合批出题失败(rc=%d, 见 %s)★", rc, gout); return; }
    }
    tr_logf(log, "GEN_DONE %u 题", tr_gen_split(dir, log));
}

static void *job_thread(void *arg) {
    (void)arg;
    char lp[TR_PATH + 96]; tr_ui_log_path("cycle", lp, sizeof lp);
    FILE *log = fopen(lp, "w");
    char mdl[TR_PATH * 2], zch[TR_PATH * 2], store[4096], *extra[32], *eng[2]; int neng = 0;
    if (!tr_pick_read(mdl, sizeof mdl, zch, sizeof zch, extra, 31, store, sizeof store)) tr_logf(log, "★没有 gguf/serve_pick.txt: 先在模型页选一套并加载(训练用的 ①② 就是它)★");
    else {
        for (int i = 0; extra[i]; i++) if (!strcmp(extra[i], "--engram-dir") && extra[i + 1]) { eng[0] = extra[i]; eng[1] = extra[i + 1]; neng = 2; break; }
        set_phase(TR_JOB_STOP_MODEL, 0);
        tr_model_stop_sync(log);
        static const char *const busy[] = { "ds4", "ds4-bench", "ds4-server", "ds4quant_run", "zlayer", "v41_amp_run", NULL };
        bool idle = true;
        for (int i = 0; busy[i]; i++) if (tr_proc_kill(busy[i], 0)) { tr_logf(log, "★机器非空(实例锁): %s 在跑★", busy[i]); idle = false; }
        if (idle && !want_stop(NULL)) { if (g.gen) do_gen(mdl, zch, eng, neng, log); else do_train(mdl, zch, eng, neng, log); }
    }
    set_phase(TR_JOB_RESTART, 0);
    tr_logf(log, "模型子进程装回来(按 serve_pick.txt)");
    tr_model_start_sync(g.port, log);
    pthread_mutex_lock(&g.mu); g.active = false; g.stop = false; g.phase = TR_JOB_NONE; g.child = 0; g.run[0] = 0; g.run_abs[0] = 0; pthread_mutex_unlock(&g.mu);
    if (log) fclose(log);
    return NULL;
}

static bool launch(bool gen, int model_port, char *err, size_t errn) {
    const tr_model_phase mp = tr_model_state();
    if (mp == TR_MODEL_STARTING || mp == TR_MODEL_STOPPING) { snprintf(err, errn, "正在加载模型, 等它装完"); return false; }
    pthread_mutex_lock(&g.mu);
    if (g.active) { pthread_mutex_unlock(&g.mu); snprintf(err, errn, "有训练/出题在跑, 先停它"); return false; }
    g.active = true; g.gen = gen; g.stop = false; g.phase = TR_JOB_STOP_MODEL; g.child = 0; g.port = model_port; g.run[0] = 0; g.run_abs[0] = 0;
    pthread_mutex_unlock(&g.mu);
    pthread_t th; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    const int rc = pthread_create(&th, &at, job_thread, NULL); pthread_attr_destroy(&at);
    if (rc != 0) { pthread_mutex_lock(&g.mu); g.active = false; g.phase = TR_JOB_NONE; pthread_mutex_unlock(&g.mu); snprintf(err, errn, "线程开不出来"); return false; }
    return true;
}

bool tr_job_train(const char *data_rel, const char *epochs, const char *layers, const char *lr, const char *extra, int model_port, char *err, size_t errn) {
    pthread_mutex_lock(&g.mu);
    if (g.active) { pthread_mutex_unlock(&g.mu); snprintf(err, errn, "有训练/出题在跑, 先停它"); return false; }
    snprintf(g.data, sizeof g.data, "%s", data_rel); snprintf(g.epochs, sizeof g.epochs, "%s", epochs); snprintf(g.layers, sizeof g.layers, "%s", layers);
    snprintf(g.lr, sizeof g.lr, "%s", lr); snprintf(g.extra, sizeof g.extra, "%s", extra ? extra : "");
    pthread_mutex_unlock(&g.mu);
    return launch(false, model_port, err, errn);
}

bool tr_job_gen(const char *dir_rel, const char *rounds, int model_port, char *err, size_t errn) {
    pthread_mutex_lock(&g.mu);
    if (g.active) { pthread_mutex_unlock(&g.mu); snprintf(err, errn, "有训练/出题在跑, 先停它"); return false; }
    snprintf(g.data, sizeof g.data, "%s", dir_rel); g.rounds = atoi(rounds) > 0 ? atoi(rounds) : 1;
    pthread_mutex_unlock(&g.mu);
    return launch(true, model_port, err, errn);
}

bool tr_job_stop(char *err, size_t errn) {
    pthread_mutex_lock(&g.mu);
    if (!g.active) { pthread_mutex_unlock(&g.mu); snprintf(err, errn, "没有在跑的训练"); return false; }
    g.stop = true;   /* 作业线程的 tr_child_wait 下一秒看到就杀 ./ds4, 然后照样把模型装回来 */
    pthread_mutex_unlock(&g.mu);
    return true;
}
