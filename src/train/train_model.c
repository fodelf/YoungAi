/* train_model.c — 模型子进程 ds4-server 的生命周期(2026-10-10, 原 serve_1m_spark.sh start/stop 的 C 版; 脚本留作命令行工具)。总述见 train_internal.h。
 * 起 = 起跑清单(实例锁: 大模型进程只许一个; MemAvailable ≥ TR_SERVE_MIN_AVAIL_MB) → 按 gguf/serve_pick.txt 拼参数 exec → 等 /v1/models 可达
 *      → 余量 ≥ TR_SERVE_FLOOR_MB(低于它会 swap/读盘假死) → 冒烟一条中文金融问答(09-19 实撞: /v1/models 正常, 每条 chat 却回 cuda prefill failed)
 *      → 看门狗线程(TR_WD_KILL_MB 连续两次就杀)。没过任何一关就杀掉, 原因在 ui_logs/serve_*.log(模型页日志栏)。
 * 停 = 按程序名 SIGTERM(主进程重起后接上的老服务也认), 等它从 /proc 消失(卸 100+ GB 映射几十秒; 没等干净就起训练会撞实例锁)。
 * 内存账(121 GB spark, 09-19 实测): 起来后 MemAvailable 11 GB, 首条请求的前向缓冲分完后稳定 6.4~6.7 GB; ★别调大 --mem-budget-mb 硬塞★。 */
#include "train_internal.h"
#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define TR_SERVE_BUDGET_MB 110000     /* 引擎常驻预算(spark 121 GB: 主干 106.5 GiB 进缓存, 三塔挤在外面, 服务不用它们) */
#define TR_SERVE_MIN_AVAIL_MB 100000  /* 起服前机器要空到这个数, 否则装不下 */
#define TR_SERVE_FLOOR_MB 10000       /* 装完后的余量地板(在分请求缓冲之前量) */
#define TR_UP_TRIES 120               /* 等 /v1/models 可达: 120 × 5 s = 10 分钟(冷启拷 106 GB 到页缓存约 1.5 分钟; 盘慢的机器给足) */
#define TR_SMOKE_PROMPT "{\"model\":\"deepseek-chat\",\"temperature\":0,\"max_tokens\":48,\"messages\":[{\"role\":\"user\",\"content\":\"用一句话解释什么是市盈率(PE), 并说明它偏高通常意味着什么。\"}]}"

static struct { pthread_mutex_t mu; tr_model_phase phase; pid_t pid; int port; } g = { PTHREAD_MUTEX_INITIALIZER, TR_MODEL_IDLE, 0, 0 };
static void set_phase(tr_model_phase p) { pthread_mutex_lock(&g.mu); g.phase = p; pthread_mutex_unlock(&g.mu); }
tr_model_phase tr_model_state(void) { pthread_mutex_lock(&g.mu); const tr_model_phase p = g.phase; pthread_mutex_unlock(&g.mu); return p; }

bool tr_model_live(void) {
    const tr_model_phase p = tr_model_state();
    if (p == TR_MODEL_UP) return true;
    if (p != TR_MODEL_IDLE) return false;
    tr_procs pr; tr_proc_scan(&pr);   /* 主进程重起后接上的老服务(不是我们的子进程): 在 /proc 里就算活 */
    return pr.server;
}

bool tr_pick_read(char *g_, size_t gn, char *z, size_t zn, char *extra[], int max_extra, char *storage, size_t sn) {
    char p[TR_PATH + 32]; snprintf(p, sizeof p, "%s/gguf/serve_pick.txt", tr_root);
    g_[0] = z[0] = 0; size_t used = 0; int ne = 0;
    FILE *f = fopen(p, "r"); if (!f) return false;
    if (fgets(g_, (int)gn, f)) g_[strcspn(g_, "\n")] = 0;
    if (fgets(z, (int)zn, f)) z[strcspn(z, "\n")] = 0;
    if (!strcmp(z, "none")) z[0] = 0;
    char ln[TR_PATH + 64];
    while (ne < max_extra && fgets(ln, sizeof ln, f)) {
        ln[strcspn(ln, "\n")] = 0;
        const size_t L = strlen(ln);
        if (!L || used + L + 1 > sn) continue;
        extra[ne++] = storage + used; memcpy(storage + used, ln, L + 1); used += L + 1;
    }
    extra[ne] = NULL;
    fclose(f);
    return g_[0] != 0;
}

static bool has_gr(const char *dir) {
    DIR *d = opendir(dir); struct dirent *de; bool hit = false;
    while (d && !hit && (de = readdir(d))) hit = !strncmp(de->d_name, "gr_L", 4) && strstr(de->d_name, ".bin");
    if (d) closedir(d);
    return hit;
}

/* 看门狗线程: 与子进程同生共死; 它退了就把状态放回 IDLE */
static void *watchdog(void *arg) {
    const pid_t pid = (pid_t)(intptr_t)arg; int bad = 0, st = 0;
    for (;;) {
        sleep(5);
        if (waitpid(pid, &st, WNOHANG) == pid) break;
        const long a = tr_mem_avail_mb();
        if (a >= 0 && a < TR_WD_KILL_MB) bad++; else bad = 0;
        if (bad >= 2) {
            char lp[TR_PATH + 96]; tr_ui_log_path("serve", lp, sizeof lp);
            FILE *lf = fopen(lp, "a"); tr_logf(lf, "★看门狗: MemAvailable %ld MB < %d 连续两次, 杀 ds4-server★", a, TR_WD_KILL_MB); if (lf) fclose(lf);
            tr_child_kill(pid); break;
        }
    }
    pthread_mutex_lock(&g.mu); if (g.pid == pid) { g.pid = 0; g.phase = TR_MODEL_IDLE; } pthread_mutex_unlock(&g.mu);
    return NULL;
}

bool tr_model_start_sync(int model_port, FILE *log) {
    set_phase(TR_MODEL_STARTING);
    bool ok = false; pid_t pid = -1;
    char mdl[TR_PATH * 2], zch[TR_PATH * 2], store[4096], *extra[32], trace[TR_PATH + 96], ps[16], bud[16];
    static const char *const busy[] = { "ds4", "ds4-bench", "ds4-server", "ds4quant_run", "zlayer", "v41_amp_run", NULL };
    for (int i = 0; busy[i]; i++) if (tr_proc_kill(busy[i], 0)) { tr_logf(log, "★机器非空(实例锁: 大模型进程只许一个): %s 在跑★", busy[i]); goto out; }
    const long avail = tr_mem_avail_mb();
    if (avail >= 0 && avail < TR_SERVE_MIN_AVAIL_MB) { tr_logf(log, "★MemAvailable %ld MB < %d, 不起★", avail, TR_SERVE_MIN_AVAIL_MB); goto out; }
    if (!tr_pick_read(mdl, sizeof mdl, zch, sizeof zch, extra, 31, store, sizeof store)) { tr_logf(log, "★没有 gguf/serve_pick.txt: 先在模型页选一套并加载★"); goto out; }
    struct stat st;
    if (stat(mdl, &st) || !S_ISREG(st.st_mode)) { tr_logf(log, "★模型缺 %s★", mdl); goto out; }
    if (zch[0] && !has_gr(zch)) { tr_logf(log, "★侧车目录缺或没有 gr_L*.bin: %s★", zch); goto out; }
    /* ★每次起服都开 --trace★(09-29): 后训练的料是真实请求的原字节 + 引擎真吐的 token id, 只在 --trace 开着时落盘; 路径写进 trace/current 给采样段读 */
    char td[TR_PATH + 32]; snprintf(td, sizeof td, "%s/gguf/v41/night/trace", tr_root);
    for (char *s = td + strlen(tr_root) + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(td, 0755); *s = '/'; }
    mkdir(td, 0755);
    time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm);
    snprintf(trace, sizeof trace, "%s/serve_%04d%02d%02d_%02d%02d%02d.txt", td, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    { char cp[TR_PATH + 48]; snprintf(cp, sizeof cp, "%s/current", td); FILE *cf = fopen(cp, "w"); if (cf) { fprintf(cf, "%s\n", trace); fclose(cf); } }
    const char *argv[64]; int na = tr_ds4_base_argv(argv, 40, mdl, zch, NULL, 0);
    argv[0] = "./ds4-server";
    snprintf(ps, sizeof ps, "%d", model_port); snprintf(bud, sizeof bud, "%d", TR_SERVE_BUDGET_MB);
    argv[na++] = "--mem-budget-mb"; argv[na++] = bud; argv[na++] = "--host"; argv[na++] = "127.0.0.1"; argv[na++] = "--port"; argv[na++] = ps;
    argv[na++] = "--trace"; argv[na++] = trace;
    for (int i = 0; extra[i] && na < 62; i++) argv[na++] = extra[i];   /* 第 3 行起原样透传(--engram-dir / 用户选的别的引擎参数) */
    argv[na] = NULL;
    char slog[TR_PATH + 96]; tr_ui_log_path("server", slog, sizeof slog);
    tr_logf(log, "起模型子进程: %s (%.1f GB) 侧车 %s 预算 %d MB available %ld MB → 127.0.0.1:%d, 引擎日志 %s", mdl, (double)st.st_size / 1e9, zch[0] ? zch : "无", TR_SERVE_BUDGET_MB, avail, model_port, slog);
    pid = tr_child_spawn(argv, slog, false);
    if (pid < 0) { tr_logf(log, "★fork 失败★"); goto out; }
    pthread_mutex_lock(&g.mu); g.pid = pid; g.port = model_port; pthread_mutex_unlock(&g.mu);
    int st2 = 0; bool up = false;
    for (int i = 0; i < TR_UP_TRIES; i++) {
        if (tr_http_local(model_port, "GET", "/v1/models", NULL, NULL, 3) == 200) { up = true; break; }
        if (waitpid(pid, &st2, WNOHANG) == pid) { pid = -1; break; }
        sleep(5);
    }
    if (!up) { tr_logf(log, "★没起来(%s), 看 %s★", pid < 0 ? "进程已退" : "10 分钟 /v1/models 还不可达", slog); goto out; }
    const long a2 = tr_mem_avail_mb();
    tr_logf(log, "起来后 MemAvailable %ld MB(地板 %d)", a2, TR_SERVE_FLOOR_MB);
    if (a2 >= 0 && a2 < TR_SERVE_FLOOR_MB) { tr_logf(log, "★余量 %ld MB < %d, 会 swap/读盘假死, 停★", a2, TR_SERVE_FLOOR_MB); goto out; }
    ds4_buf ans = {0};
    const int code = tr_http_local(model_port, "POST", "/v1/chat/completions", TR_SMOKE_PROMPT, &ans, 600);
    const bool smoke = code == 200 && ans.ptr && strstr(ans.ptr, "\"choices\"");
    tr_logf(log, "冒烟 %s: %.*s", smoke ? "过" : "★失败★", ans.ptr ? (int)(ans.len > 600 ? 600 : ans.len) : 0, ans.ptr ? ans.ptr : "");
    ds4_buf_free(&ans);
    if (!smoke) goto out;
    pthread_t th; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_create(&th, &at, watchdog, (void *)(intptr_t)pid); pthread_attr_destroy(&at);
    set_phase(TR_MODEL_UP);
    tr_logf(log, "SERVE1M_UP pid %d, 看门狗红线 %d MB", (int)pid, TR_WD_KILL_MB);
    ok = true;
out:
    if (!ok) {
        if (pid > 0) tr_child_kill(pid);
        pthread_mutex_lock(&g.mu); g.pid = 0; g.phase = TR_MODEL_IDLE; pthread_mutex_unlock(&g.mu);
    }
    return ok;
}

void tr_model_stop_sync(FILE *log) {
    set_phase(TR_MODEL_STOPPING);
    pthread_mutex_lock(&g.mu); const pid_t mine = g.pid; g.pid = 0; pthread_mutex_unlock(&g.mu);
    const int n = tr_proc_kill("ds4-server", SIGTERM);
    if (mine > 0) tr_child_kill(mine);
    if (n || mine > 0) {
        tr_logf(log, "停模型子进程(%d 个), 等它卸完映射", n);
        if (!tr_prog_wait_gone("ds4-server", 10)) { tr_proc_kill("ds4-server", SIGKILL); tr_prog_wait_gone("ds4-server", 120); }
        tr_logf(log, "模型子进程已退, MemAvailable %ld MB", tr_mem_avail_mb());
    }
    set_phase(TR_MODEL_IDLE);
}

typedef struct { int port; bool restart; } start_arg;
static void *start_thread(void *arg) {
    start_arg *a = arg;
    char lp[TR_PATH + 96]; tr_ui_log_path("serve", lp, sizeof lp);
    FILE *log = fopen(lp, "w");
    if (a->restart) tr_model_stop_sync(log);
    tr_model_start_sync(a->port, log);
    if (log) fclose(log);
    free(a);
    return NULL;
}

bool tr_model_start(int model_port, bool restart, char *err, size_t errn) {
    tr_job_info j; tr_job_status(&j);
    if (j.active) { snprintf(err, errn, "有训练在跑, 训完会按选择装回来"); return false; }
    const tr_model_phase p = tr_model_state();
    if (p == TR_MODEL_STARTING || p == TR_MODEL_STOPPING) { snprintf(err, errn, "正在加载模型, 等它装完"); return false; }
    if (!restart && tr_model_live()) { snprintf(err, errn, "服务已经在跑"); return false; }
    set_phase(restart ? TR_MODEL_STOPPING : TR_MODEL_STARTING);   /* 先占住, 线程起来前别放第二个进来(路由是单线程的, 这里不会并发) */
    start_arg *a = malloc(sizeof *a); *a = (start_arg){ model_port, restart };
    pthread_t th; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    const int rc = pthread_create(&th, &at, start_thread, a); pthread_attr_destroy(&at);
    if (rc != 0) { free(a); set_phase(p); snprintf(err, errn, "线程开不出来"); return false; }
    return true;
}
