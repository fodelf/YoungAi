/* train_proxy.c — 主进程把归模型子进程的请求原字节转给它(2026-10-10; 用户: "主服务是永远不停的, 模型和后训练都是子实现")。
 * 总述见 train_internal.h。ds4-server 绑 127.0.0.1:<页面端口+1>; 页面与 API 客户端只认主进程的端口。
 * 每条转发开一个线程: 主循环是单线程阻塞的(train_http.c), 一条回答几分钟, 不能把页面的轮询堵死。线程只碰自己的两个 socket, 不碰任何共享状态。
 * 响应边收边转(流式回答不攒); ds4-server 每条响应都 Connection: close(server_http.c), 所以"上游关了 = 这条响应完了", 不用解析 chunked。
 * 出错会怎样: 连不上子进程 → 502 一句话(页面当"模型没装着"); 浏览器中途关页 → 这边关上游, ds4-server 自己探到断开就停生成(client_disconnected)。 */
#include "train_internal.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

bool tr_proxy_path(const char *path) {
    return !strncmp(path, "/v1/", 4) || !strcmp(path, "/monitor") || !strcmp(path, "/monitor.html") || !strcmp(path, "/metrics")
        || !strcmp(path, "/api/models/plugins") || !strcmp(path, "/api/models/list");
}

typedef struct { int cfd, port; char *raw; size_t n; } proxy_job;

static bool send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += (size_t)w; n -= (size_t)w;
    }
    return true;
}

static int connect_model(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof a) < 0) { close(fd); return -1; }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

static void *proxy_run(void *arg) {
    proxy_job *j = arg;
    const int up = connect_model(j->port);
    if (up < 0) {
        const char *body = "{\"error\":\"模型子进程连不上(没装着, 或正在装)\"}";
        char h[256]; const int hn = snprintf(h, sizeof h, "HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", strlen(body));
        if (send_all(j->cfd, h, (size_t)hn)) send_all(j->cfd, body, strlen(body));
    } else if (send_all(up, j->raw, j->n)) {
        /* 两头都盯: 上游有字节就转给客户端; 客户端先断(关页/超时)就撤, 别让子进程对着没人收的 socket 接着算 */
        struct pollfd pf[2] = { { up, POLLIN, 0 }, { j->cfd, POLLIN, 0 } };
        char buf[65536];
        for (;;) {
            if (poll(pf, 2, -1) < 0) { if (errno == EINTR) continue; break; }
            if (pf[0].revents) {
                const ssize_t r = recv(up, buf, sizeof buf, 0);
                if (r <= 0 || !send_all(j->cfd, buf, (size_t)r)) break;
            }
            if (pf[1].revents) {   /* 请求已整条转走, 客户端这头再来字节只会是断开(0)或没用的尾巴 */
                const ssize_t r = recv(j->cfd, buf, sizeof buf, 0);
                if (r <= 0) break;
            }
        }
    }
    if (up >= 0) close(up);
    close(j->cfd); free(j->raw); free(j);
    return NULL;
}

void tr_proxy_start(int cfd, char *raw, size_t raw_len, int model_port) {
    proxy_job *j = malloc(sizeof *j);
    *j = (proxy_job){ cfd, model_port, raw, raw_len };
    pthread_t th; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&th, &at, proxy_run, j) != 0) proxy_run(j);   /* 线程开不出来就在主循环里转完这条(慢, 但不丢请求) */
    pthread_attr_destroy(&at);
}
