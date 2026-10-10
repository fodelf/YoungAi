/* train_main.c — ds4-train 入口: 工作台的主进程(永远不退), 一层薄 HTTP 套在 tr_api 上; 模型(ds4-server)与训练链都是它拉起的子进程。总述见 train_internal.h。 */
#include "train_internal.h"
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

static int g_port = 8000;

static int handle(const tr_req *rq, ds4_buf *out, const char **ctype) { return tr_api(rq, out, ctype, 0, g_port); }

int main(int argc, char **argv) {
    const char *host = "0.0.0.0", *root = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) g_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--root") && i + 1 < argc) root = argv[++i];
        else { fprintf(stderr, "用法: ds4-train [--host 0.0.0.0] [--port 8000] [--root <仓库根, 缺省当前目录>]   (模型子进程起在 127.0.0.1:<端口+1>)\n"); return 1; }
    }
    char cwd[TR_PATH], exe[TR_PATH], p[TR_PATH + 64];
    struct stat st;
    /* 没给 --root: 先认可执行文件所在目录(发布包解压后用全路径执行也能起, 不要求先 cd 进去), 那里没有页面再认当前目录(源码树里 ./ds4-train) */
    if (!root) {
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);   /* Linux */
#ifdef __APPLE__
        { uint32_t sz = sizeof exe; char raw[TR_PATH]; uint32_t rs = sizeof raw;   /* Mac: 发布包解压到哪都能全路径执行, 不要求先 cd 进去 */
          if (_NSGetExecutablePath(raw, &rs) == 0 && realpath(raw, exe)) n = (ssize_t)strlen(exe); (void)sz; }
#endif
        if (n > 0) {
            exe[n] = 0;
            char *sl = strrchr(exe, '/');
            if (sl) { *sl = 0; snprintf(p, sizeof p, "%.900s/web/studio.html", exe); if (!stat(p, &st)) root = exe; }
        }
    }
    if (!root) { if (!getcwd(cwd, sizeof cwd)) { perror("getcwd"); return 1; } root = cwd; }
    tr_init(root);
    snprintf(p, sizeof p, "%.900s/web/studio.html", tr_root);
    if (stat(p, &st)) { fprintf(stderr, "ds4-train: %s 不存在 —— 要在仓库根起, 或给 --root\n", p); return 1; }
    const int rc = tr_http_serve(host, g_port, handle);
    fprintf(stderr, "ds4-train: 退出(监听失败)\n");
    return rc;
}
