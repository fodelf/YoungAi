/* train_child.c — 主进程 ds4-train 管子进程的底座(2026-10-10): fork/setsid/exec + 日志重定向 + 内存看门狗 + 本机 HTTP 一条请求。
 * 总述见 train_internal.h。模型子进程(train_model.c)、训练/打分/出题的 ./ds4(train_job.c / train_gate.c)都走这里, 看门狗只写一处。
 * 子进程自成会话: ds4-train 被 Ctrl-C / SIGTERM 时模型子进程留着(下次起主进程直接接上, 与此前 nohup 起服同一语义)。
 * fd≥3 不继承: ★服务的监听 socket 带进子进程树★是 10-10 实撞(训练树攥着 8000, 主进程与回来的服务都绑不上)。
 * 出错会怎样: 看门狗红线 TR_WD_KILL_MB 连续两次 → SIGTERM 子进程(ds4 没有 SIGTERM 处理器, 默认退, 卸 100 GB 映射几秒), 3 秒后 SIGKILL; tr_child_wait 返回 -2。 */
#include "train_internal.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

void tr_logf(FILE *f, const char *fmt, ...) {
    if (!f) return;
    time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm);
    fprintf(f, "[%02d:%02d:%02d] ", tm.tm_hour, tm.tm_min, tm.tm_sec);
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap);
    fputc('\n', f); fflush(f);
}

/* detach = 两次 fork 让孙子归 init(父不等它, 也不留僵尸): 只给下载这种"起了就不管"的用。模型/训练子进程是直接子进程, 各自的线程 waitpid 盯着;
 * ★全进程禁止 waitpid(-1)★ —— 会把别的线程正盯着的子进程退出状态抢走, 那边就永远等不到 */
static pid_t spawn(const char *const argv[], const char *log_path, bool append, bool detach) {
    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        setsid();
        if (detach && fork() != 0) _exit(0);
        const int nul = open("/dev/null", O_RDWR);
        const int fd = log_path ? open(log_path, O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC), 0644) : nul;
        if (nul >= 0) dup2(nul, 0);
        if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); }
        for (int k = 3; k < 4096; k++) close(k);
        if (chdir(tr_root) != 0) _exit(126);
        execvp(argv[0], (char *const *)argv);
        _exit(127);
    }
    if (detach) { int st; while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {} }
    return pid;
}
pid_t tr_child_spawn(const char *const argv[], const char *log_path, bool append) { return spawn(argv, log_path, append, false); }
bool tr_child_spawn_detached(const char *const argv[], const char *log_path) { return spawn(argv, log_path, false, true) > 0; }

static bool pid_alive(pid_t pid) { return kill(pid, 0) == 0 || errno == EPERM; }

void tr_child_kill(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    int st = 0;
    for (int i = 0; i < 30 && waitpid(pid, &st, WNOHANG) == 0; i++) usleep(100000);   /* 3 秒 */
    if (pid_alive(pid) && waitpid(pid, &st, WNOHANG) == 0) {
        kill(pid, SIGKILL);
        while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    }
}

/* 每秒 waitpid 一次; 每 5 秒看一次 MemAvailable(没有 /proc 的机器跳过看门狗) */
int tr_child_wait(pid_t pid, tr_cancel_fn cancel, void *ud, FILE *log) {
    int st = 0, bad = 0, tick = 0;
    for (;;) {
        const pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) {
            if (WIFEXITED(st)) return WEXITSTATUS(st);
            return WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1;
        }
        if (r < 0 && errno != EINTR) return -1;
        if (cancel && cancel(ud)) { tr_logf(log, "收到停止, 杀子进程 %d", (int)pid); tr_child_kill(pid); return 128 + SIGTERM; }
        if (++tick % 5 == 0) {
            const long a = tr_mem_avail_mb();
            if (a >= 0 && a < TR_WD_KILL_MB) bad++; else bad = 0;
            if (bad >= 2) { tr_logf(log, "★看门狗: MemAvailable %ld MB < %d 连续两次, 杀子进程 %d★", a, TR_WD_KILL_MB, (int)pid); tr_child_kill(pid); return -2; }
        }
        sleep(1);
    }
}

bool tr_prog_wait_gone(const char *prog, int timeout_s) {
    for (int i = 0; i < timeout_s; i++) {
        if (tr_proc_kill(prog, 0) == 0) return true;
        sleep(1);
    }
    return tr_proc_kill(prog, 0) == 0;
}

static bool send_all(int fd, const char *p, size_t n) {
    while (n) {
        const ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += (size_t)w; n -= (size_t)w;
    }
    return true;
}

/* 返回状态码; 响应体(头之后)进 out。超时按整条请求算(冒烟一条回答十几秒, 装模型期间 /v1/models 连不上是常态) */
int tr_http_local(int port, const char *method, const char *path, const char *body, ds4_buf *out, int timeout_s) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof a) < 0) { close(fd); return -1; }
    ds4_buf req = {0};
    const size_t bl = body ? strlen(body) : 0;
    ds4_buf_printf(&req, "%s %s HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", method, path, bl);
    if (bl) ds4_buf_append(&req, body, bl);
    const bool sent = send_all(fd, req.ptr, req.len);
    ds4_buf_free(&req);
    if (!sent) { close(fd); return -1; }
    ds4_buf in = {0}; char tmp[8192];
    const time_t t0 = time(NULL);
    for (;;) {
        struct pollfd pf = { fd, POLLIN, 0 };
        const int left = timeout_s - (int)(time(NULL) - t0);
        if (left <= 0) break;
        const int pr = poll(&pf, 1, left * 1000);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) break;
        const ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) break;
        ds4_buf_append(&in, tmp, (size_t)n);
    }
    close(fd);
    int code = -1;
    if (in.ptr && sscanf(in.ptr, "HTTP/%*s %d", &code) == 1) {
        const char *e = strstr(in.ptr, "\r\n\r\n");
        if (e && out) ds4_buf_append(out, e + 4, in.len - (size_t)(e + 4 - in.ptr));
    }
    ds4_buf_free(&in);
    return code;
}
