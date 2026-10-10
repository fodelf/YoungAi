/* train_ctl.c — /proc 扫进程 + 仅剩的脚本调用(模型页下载 = hf_install.sh)。总述见 train_internal.h。
 * 进程一律按程序名认(程序本身叫 name, 或 bash/sh 正在跑名叫 name 的脚本)。★不许在整条命令行里找子串★(10-10 实撞): 别的进程命令行里带着这个词 ——
 * tail -f ds4-server-1m.log、ssh 远程命令、编辑器开着脚本 —— 就被当成"服务在跑", 模型页点加载一直报"服务已经在跑", 而其实什么都没跑。
 * 训练/起服的流程编排已不在脚本里(train_job.c / train_model.c), 这里不再认 z_nightly / v41_judge / train_cycle / serve_1m_spark。 */
#include "train_internal.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

char tr_root[TR_PATH], tr_ftd[TR_PATH], tr_datad[TR_PATH];

void tr_init(const char *root) {
    snprintf(tr_root, sizeof tr_root, "%s", root);
    snprintf(tr_ftd, sizeof tr_ftd, "%.900s/gguf/v41/posttrain", root);
    snprintf(tr_datad, sizeof tr_datad, "%.900s/gguf-tools/data/posttrain", root);
}

/* 逐个 /proc/<pid>/cmdline 回调(NUL 换成空格); 非 Linux 返回 false */
static bool proc_each(void (*fn)(pid_t, const char *, void *), void *ud) {
#ifdef __linux__
    DIR *d = opendir("/proc"); struct dirent *de;
    if (!d) return false;
    const pid_t me = getpid();
    while ((de = readdir(d))) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;
        const pid_t pid = (pid_t)atoi(de->d_name);
        if (pid == me) continue;
        char p[64]; snprintf(p, sizeof p, "/proc/%s/cmdline", de->d_name);
        FILE *f = fopen(p, "rb");
        if (!f) continue;
        char cmd[4096]; size_t n = fread(cmd, 1, sizeof cmd - 1, f); fclose(f);
        if (!n) continue;
        for (size_t i = 0; i < n; i++) if (!cmd[i]) cmd[i] = ' ';
        cmd[n] = 0;
        fn(pid, cmd, ud);
    }
    closedir(d);
    return true;
#else
    (void)fn; (void)ud;
    return false;
#endif
}
/* 命令行(NUL 已换空格)第 i 个词的文件名部分 */
static void word_base(const char *cmd, int i, char *out, size_t n) {
    const char *p = cmd;
    for (int k = 0; k < i && p; k++) { p = strchr(p, ' '); if (p) p++; }
    out[0] = 0;
    if (!p) return;
    char w[512]; const size_t L = strcspn(p, " ");
    snprintf(w, sizeof w, "%.*s", (int)(L < sizeof w - 1 ? L : sizeof w - 1), p);
    const char *b = strrchr(w, '/');
    snprintf(out, n, "%s", b ? b + 1 : w);
}
static bool prog_is(const char *cmd, const char *name) {
    char w0[256], w1[256];
    word_base(cmd, 0, w0, sizeof w0);
    if (!strcmp(w0, name)) return true;
    if (strcmp(w0, "bash") && strcmp(w0, "sh")) return false;
    word_base(cmd, 1, w1, sizeof w1);
    return !strcmp(w1, name);
}
static void scan_fn(pid_t pid, const char *cmd, void *ud) {
    tr_procs *p = ud; (void)pid;
    if (prog_is(cmd, "ds4") && (strstr(cmd, " --cuda") || strstr(cmd, " --metal"))) p->ds4 = true;
    if (prog_is(cmd, "ds4-server")) p->server = true;
    if (prog_is(cmd, "hf_install.sh")) p->dl = true;
}
void tr_proc_scan(tr_procs *p) {
    memset(p, 0, sizeof *p);
    p->scanned = proc_each(scan_fn, p);
}

typedef struct { const char *prog; int sig, n; } kill_arg;
static void kill_fn(pid_t pid, const char *cmd, void *ud) { kill_arg *k = ud; if (prog_is(cmd, k->prog)) { if (k->sig) kill(pid, k->sig); k->n++; } }
int tr_proc_kill(const char *prog, int sig) { kill_arg k = { prog, sig, 0 }; proc_each(kill_fn, &k); return k.n; }

/* 正在跑的 ds4-server 用的模型与侧车(cmdline 里 -m / --zchain / --posttrain 后的那一项; 起服传的都是不含空格的路径) */
typedef struct { char *gguf, *zch, *pt; size_t gn, zn, pn; bool hit; } srv_arg;
static void copy_arg(const char *cmd, const char *flag, char *out, size_t n) {
    const char *p = strstr(cmd, flag);
    if (!p || !n) return;
    p += strlen(flag);
    size_t k = 0; while (p[k] && p[k] != ' ' && k + 1 < n) { out[k] = p[k]; k++; }
    out[k] = 0;
}
static void srv_fn(pid_t pid, const char *cmd, void *ud) {
    srv_arg *a = ud; (void)pid;
    if (!prog_is(cmd, "ds4-server")) return;
    a->hit = true;
    copy_arg(cmd, " -m ", a->gguf, a->gn); copy_arg(cmd, " --zchain ", a->zch, a->zn); copy_arg(cmd, " --posttrain ", a->pt, a->pn);
}
bool tr_server_model(char *gguf, size_t gn, char *zch, size_t zn, char *pt, size_t pn) {
    gguf[0] = zch[0] = pt[0] = 0;
    srv_arg a = { gguf, zch, pt, gn, zn, pn, false };
    proc_each(srv_fn, &a);
    return a.hit;
}

/* ui_logs/<tag>_<时间>.log: 逐层建目录(mkdir -p) —— 刚解压的发布包里没有 gguf/v41/posttrain, 只建最后一层会失败, 下载/加载的日志就没处写,
 * 模型页的日志栏一直空着, 加载失败也看不到原因(10-10 发布包实测撞到) */
bool tr_ui_log_path(const char *tag, char *out, size_t n) {
    char ld[TR_PATH + 16]; snprintf(ld, sizeof ld, "%s/ui_logs", tr_ftd);
    for (char *s = ld + strlen(tr_root) + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(ld, 0755); *s = '/'; }
    mkdir(ld, 0755);
    time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm);
    snprintf(out, n, "%s/%s_%04d%02d%02d_%02d%02d%02d.log", ld, tag, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return true;
}

/* 模型页下载: bash hf_install.sh download --dir <hub> …(断点续传、逐片 sha256、拼装都在脚本里 —— 下载本身就是 python 的 hf CLI, 脚本只是它的包装) */
bool tr_spawn_download(const char *const args[], char *err, size_t errn) {
    tr_procs p; tr_proc_scan(&p);
    if (p.dl) { snprintf(err, errn, "已经在下载"); return false; }
    char script[TR_PATH + 64], logp[TR_PATH + 96];
    snprintf(script, sizeof script, "%.900s/gguf-tools/scripts/hf_install.sh", tr_root);
    const char *argv[24] = { "bash", script, "download" };
    int na = 3;
    for (int i = 0; args[i] && na < 23; i++) argv[na++] = args[i];
    argv[na] = NULL;
    tr_ui_log_path("hub", logp, sizeof logp);
    if (!tr_child_spawn_detached(argv, logp)) { snprintf(err, errn, "fork 失败"); return false; }
    return true;   /* 归 init, 不等它: 在不在由 /proc 扫判断 */
}
/* 取消下载: 杀 hf_install.sh 和它拉起的 hf download(只认 --local-dir 在 hub 目录下的, 别人手敲的 hf download 不碰)。半截文件留着, 下次从断点接着下。 */
typedef struct { const char *hub; int sig, n; } dl_kill;
static void dl_kill_fn(pid_t pid, const char *cmd, void *ud) {
    dl_kill *k = ud;
    char w1[64]; word_base(cmd, 1, w1, sizeof w1);   /* hf 命令行是 python3 …/bin/hf download …: 第 2 个词是 hf */
    if (prog_is(cmd, "hf_install.sh") || (!strcmp(w1, "hf") && strstr(cmd, " download ") && strstr(cmd, k->hub))) { kill(pid, k->sig); k->n++; }
}
bool tr_cancel_download(const char *hub, char *err, size_t errn) {
    dl_kill k = { hub, SIGTERM, 0 };
    proc_each(dl_kill_fn, &k);
    if (!k.n) { snprintf(err, errn, "没有在下载"); return false; }
    sleep(2);
    k.sig = SIGKILL; proc_each(dl_kill_fn, &k);
    return true;
}
