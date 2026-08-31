/* dist_rfetch.c — 机械拆自 ds4_distributed.c: 远端专家 pread 服务(remote expert pread service)。行为零变化。 */
#include "dist_internal.h"

/* ============================================================================
 * project.md P2.2 (low-cost variant): remote expert pread service.
 *
 * P0.3 measured the two SSDs: coordinator (M4 mini) tops out at ~2.4-2.6GB/s
 * for every pattern while the worker (M1 MacBook) reads 5.5-6.7GB/s -- and
 * the worker's disk sits idle during the coordinator's half of every decoded
 * token (serial layer pipeline).  Both machines load the byte-identical GGUF,
 * so the coordinator can stream cold expert bytes from the worker's faster,
 * idle disk over Thunderbolt *in parallel with* its own SSD: aggregate supply
 * ~2.5 + min(TB, worker SSD) GB/s.
 *
 * Protocol (one TCP connection per client thread, strictly sequential):
 *   handshake: client sends {u64 magic}, server replies {u64 magic, u64 size}
 *   request:   {u64 off, u32 len}            (len <= 8MiB, off+len <= size)
 *   response:  {u32 status} + len bytes when status == 0
 * Same-arch little-endian Macs on a direct link: fixed-width fields are sent
 * raw, consistent with the existing payload framing in this file.
 * ========================================================================== */

#define DS4_EFETCH_MAGIC 0x4453344546563101ULL   /* "DS4EFV1" + 0x01 */
#define DS4_EFETCH_MAX_LEN (8u << 20)
#define DS4_EFETCH_MAX_CONNS 8
#define DS4_EFETCH_DEFAULT_PORT 5606        /* 服务端监听 */
#define DS4_EFETCH_DEFAULT_DIAL_PORT 5607   /* serve-dial 拨对端 accept 口 */

typedef struct {
    uint64_t off;
    uint32_t len;
} ds4_efetch_req;

/* dist_set_socket_low_latency asks for 128MB buffers, which exceeds macOS
 * kern.ipc.maxsockbuf and silently leaves the (much smaller, autotuned)
 * defaults.  Throughput over the Thunderbolt bridge wants a real, valid
 * window: 4MiB succeeds and measured 4.5GB/s single-stream. */
static void ds4_efetch_set_buffers(int fd) {
    int b = 4 << 20;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &b, sizeof(b));
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &b, sizeof(b));
}

typedef struct {
    int listen_fd;
    int model_fd;
    uint64_t model_size;
} ds4_efetch_server;

static void *ds4_efetch_conn_thread(void *arg) {
    void **pack = (void **)arg;
    int fd = (int)(intptr_t)pack[0];
    ds4_efetch_server *srv = (ds4_efetch_server *)pack[1];
    free(pack);

    uint8_t *buf = malloc(DS4_EFETCH_MAX_LEN);
    if (!buf) { close(fd); return NULL; }

    uint64_t magic = 0;
    if (dist_read_full(fd, &magic, sizeof(magic)) != 1 || magic != DS4_EFETCH_MAGIC) {
        free(buf); close(fd); return NULL;
    }
    uint64_t hello[2] = { DS4_EFETCH_MAGIC, srv->model_size };
    if (dist_write_full(fd, hello, sizeof(hello)) != 0) {
        free(buf); close(fd); return NULL;
    }

    for (;;) {
        ds4_efetch_req req;
        int rc = dist_read_full(fd, &req, sizeof(req));
        if (rc <= 0) break;
        uint32_t status = 0;
        if (req.len == 0 || req.len > DS4_EFETCH_MAX_LEN ||
            req.off > srv->model_size || (uint64_t)req.len > srv->model_size - req.off) {
            status = 1;
        } else {
            uint8_t *p = buf;
            uint64_t off = req.off;
            size_t left = req.len;
            while (left > 0) {
                ssize_t n = pread(srv->model_fd, p, left, (off_t)off);
                if (n < 0) { if (errno == EINTR) continue; status = 2; break; }
                if (n == 0) { status = 2; break; }
                p += (size_t)n; off += (uint64_t)n; left -= (size_t)n;
            }
        }
        if (dist_write_full(fd, &status, sizeof(status)) != 0) break;
        if (status == 0 && dist_write_full(fd, buf, req.len) != 0) break;
    }
    free(buf);
    close(fd);
    return NULL;
}

static void *ds4_efetch_accept_thread(void *arg) {
    ds4_efetch_server *srv = (ds4_efetch_server *)arg;
    for (;;) {
        int fd = accept(srv->listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        dist_set_socket_low_latency(fd);
        ds4_efetch_set_buffers(fd);
        void **pack = malloc(2 * sizeof(void *));
        if (!pack) { close(fd); continue; }
        pack[0] = (void *)(intptr_t)fd;
        pack[1] = srv;
        pthread_t th;
        if (pthread_create(&th, NULL, ds4_efetch_conn_thread, pack) == 0) {
            pthread_detach(th);
        } else {
            free(pack);
            close(fd);
        }
    }
    return NULL;
}

/* --expert-fetch-serve / --expert-fetch-dial 的进程级配置 (dist_cli.c 置位)。 */
int g_efetch_serve = 0;
int g_efetch_port = DS4_EFETCH_DEFAULT_PORT;
const char *g_efetch_dial_host = NULL;
int g_efetch_dial_port = DS4_EFETCH_DEFAULT_DIAL_PORT;

int ds4_dist_expert_fetch_maybe_serve(int model_fd, uint64_t model_size) {
    if (!g_efetch_serve || model_fd < 0 || model_size == 0) return 0;

    const int port = g_efetch_port;

    static ds4_efetch_server srv;   /* one server per process */
    if (srv.listen_fd > 0) return 1;

    int dup_fd = dup(model_fd);
    if (dup_fd < 0) return 0;
    char err[256] = {0};
    int lfd = dist_open_listener(NULL, port, err, sizeof(err));
    if (lfd < 0) {
        fprintf(stderr, "ds4: expert-fetch server failed to listen on :%d (%s)\n", port, err);
        close(dup_fd);
        return 0;
    }
    srv.listen_fd = lfd;
    srv.model_fd = dup_fd;
    srv.model_size = model_size;
    pthread_t th;
    if (pthread_create(&th, NULL, ds4_efetch_accept_thread, &srv) != 0) {
        close(lfd);
        close(dup_fd);
        srv.listen_fd = 0;
        return 0;
    }
    pthread_detach(th);
    fprintf(stderr,
            "ds4: expert-fetch server listening on :%d (serving %.2f GiB model file reads)\n",
            port, (double)model_size / 1073741824.0);
    return 1;
}

/* ---- client: one socket per fetcher thread, no shared locks ---- */

static int g_efetch_fds[DS4_EFETCH_MAX_CONNS];
static int g_efetch_nconns;

int ds4_dist_expert_fetch_client_init(const char *host, int port, int n_conns, uint64_t model_size) {
    if (!host || !host[0] || n_conns <= 0) return 0;
    if (n_conns > DS4_EFETCH_MAX_CONNS) n_conns = DS4_EFETCH_MAX_CONNS;
    int connected = 0;
    for (int i = 0; i < n_conns; i++) {
        char err[256] = {0};
        int fd = dist_connect_endpoint_once(host, port, NULL, err, sizeof(err));
        if (fd < 0) {
            /* Wave 28: the metal layer retries this init every 2s for minutes
             * (transient EHOSTUNREACH under prefill link saturation); rate-
             * limit the per-attempt noise. */
            static unsigned fail_logged;
            if (fail_logged < 3 || (fail_logged & 15u) == 0) {
                fprintf(stderr, "ds4: expert-fetch connect %s:%d failed (%s)\n",
                        host, port, err);
            }
            fail_logged++;
            break;
        }
        dist_set_socket_low_latency(fd);
        ds4_efetch_set_buffers(fd);
        uint64_t magic = DS4_EFETCH_MAGIC;
        uint64_t hello[2] = {0, 0};
        if (dist_write_full(fd, &magic, sizeof(magic)) != 0 ||
            dist_read_full(fd, hello, sizeof(hello)) != 1 ||
            hello[0] != DS4_EFETCH_MAGIC ||
            (model_size != 0 && hello[1] != model_size)) {
            fprintf(stderr,
                    "ds4: expert-fetch handshake with %s:%d failed (remote size %llu vs local %llu)\n",
                    host, port,
                    (unsigned long long)hello[1], (unsigned long long)model_size);
            close(fd);
            break;
        }
        g_efetch_fds[connected++] = fd;
    }
    g_efetch_nconns = connected;
    if (connected) {
        fprintf(stderr, "ds4: expert-fetch client: %d connection(s) to %s:%d\n",
                connected, host, port);
    }
    return connected;
}

/* ---- Wave 30: reverse-established transport ----
 * The worker's outbound TCP dials to the coordinator fail with instant
 * EHOSTUNREACH for the entire run (including the post-accept quiet window),
 * while the coordinator's in-process dials to the worker succeed every run
 * (control + 6 staging efetch conns).  Rather than keep debugging the
 * asymmetric bridge, reverse only the ESTABLISHMENT direction and keep the
 * wire identical: the worker (logical fetch client) LISTENS and after accept
 * sends the client-side magic; the coordinator (logical pread server) DIALS
 * the worker and then runs the standard conn-serving loop on the dialed
 * socket.  ds4_efetch_conn_thread already starts by reading the magic, so the
 * server side reuses it unmodified. */

int ds4_dist_expert_fetch_accept_init(int port, int n_conns, uint64_t model_size) {
    if (port <= 0 || n_conns <= 0) return 0;
    if (n_conns > DS4_EFETCH_MAX_CONNS) n_conns = DS4_EFETCH_MAX_CONNS;
    char err[256] = {0};
    int lfd = dist_open_listener(NULL, port, err, sizeof(err));
    if (lfd < 0) {
        fprintf(stderr, "ds4: expert-fetch accept-listener :%d failed (%s)\n", port, err);
        return 0;
    }
    fprintf(stderr,
            "ds4: expert-fetch accept mode: waiting for %d peer-dialed conn(s) on :%d\n",
            n_conns, port);
    int connected = 0;
    int waited_ms = 0;
    while (connected < n_conns && waited_ms < 120000) {
        struct pollfd p = { .fd = lfd, .events = POLLIN, .revents = 0 };
        int pr = poll(&p, 1, 1000);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0) { waited_ms += 1000; continue; }
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) continue;
        dist_set_socket_low_latency(fd);
        ds4_efetch_set_buffers(fd);
        uint64_t magic = DS4_EFETCH_MAGIC;
        uint64_t hello[2] = {0, 0};
        if (dist_write_full(fd, &magic, sizeof(magic)) != 0 ||
            dist_read_full(fd, hello, sizeof(hello)) != 1 ||
            hello[0] != DS4_EFETCH_MAGIC ||
            (model_size != 0 && hello[1] != model_size)) {
            fprintf(stderr,
                    "ds4: expert-fetch accept handshake failed (remote size %llu vs local %llu)\n",
                    (unsigned long long)hello[1], (unsigned long long)model_size);
            close(fd);
            continue;
        }
        g_efetch_fds[connected++] = fd;
    }
    close(lfd);
    g_efetch_nconns = connected;
    if (connected) {
        fprintf(stderr, "ds4: expert-fetch client (accept mode): %d connection(s) on :%d\n",
                connected, port);
    } else {
        fprintf(stderr, "ds4: expert-fetch accept mode: no peer dial within 120s; disabled\n");
    }
    return connected;
}

static void *ds4_efetch_serve_dial_main(void *arg) {
    ds4_efetch_server *srv = (ds4_efetch_server *)arg;
    const char *host = g_efetch_dial_host;
    if (!host || !host[0]) return NULL;
    const int port = g_efetch_dial_port;
    /* 4 条连接: 单条 TCP 塞不满桥接带宽, 4 条已到收益平台。 */
    int conns = 4;
    if (conns > DS4_EFETCH_MAX_CONNS) conns = DS4_EFETCH_MAX_CONNS;
    /* The peer's accept-listener comes up a few seconds in (right after the
     * worker accepts the control connection); dial with a 2s backoff.
     * 150 × 2s = 5 分钟上限: 覆盖对端慢启动, 之后放弃并明说, 不无限重试。 */
    enum { EFETCH_DIAL_ATTEMPTS = 150, EFETCH_DIAL_BACKOFF_SEC = 2 };
    int dialed = 0;
    for (int attempt = 0; dialed < conns && attempt < EFETCH_DIAL_ATTEMPTS; attempt++) {
        char err[256] = {0};
        int fd = dist_connect_endpoint_once(host, port, NULL, err, sizeof(err));
        if (fd < 0) {
            if (attempt < 2 || (attempt & 15) == 0) {
                fprintf(stderr, "ds4: expert-fetch serve-dial %s:%d failed (%s)\n",
                        host, port, err);
            }
            struct timespec ts = { .tv_sec = EFETCH_DIAL_BACKOFF_SEC, .tv_nsec = 0 };
            nanosleep(&ts, NULL);
            continue;
        }
        dist_set_socket_low_latency(fd);
        ds4_efetch_set_buffers(fd);
        void **pack = malloc(2 * sizeof(void *));
        if (!pack) { close(fd); break; }
        pack[0] = (void *)(intptr_t)fd;
        pack[1] = srv;
        pthread_t th;
        if (pthread_create(&th, NULL, ds4_efetch_conn_thread, pack) == 0) {
            pthread_detach(th);
            dialed++;
        } else {
            free(pack);
            close(fd);
        }
    }
    if (dialed) {
        fprintf(stderr, "ds4: expert-fetch serve-dial: %d connection(s) to %s:%d\n",
                dialed, host, port);
    } else {
        fprintf(stderr, "ds4: expert-fetch serve-dial to %s:%d gave up\n", host, port);
    }
    return NULL;
}

int ds4_dist_expert_fetch_serve_dial(int model_fd, uint64_t model_size) {
    const char *host = g_efetch_dial_host;
    if (!host || !host[0] || model_fd < 0 || model_size == 0) return 0;
    static ds4_efetch_server srv;   /* one serve-dial set per process */
    if (srv.model_fd > 0) return 1;
    int dup_fd = dup(model_fd);
    if (dup_fd < 0) return 0;
    srv.listen_fd = -1;
    srv.model_fd = dup_fd;
    srv.model_size = model_size;
    pthread_t th;
    if (pthread_create(&th, NULL, ds4_efetch_serve_dial_main, &srv) != 0) {
        close(dup_fd);
        srv.model_fd = 0;
        return 0;
    }
    pthread_detach(th);
    return 1;
}

int ds4_dist_expert_fetch(int slot, uint64_t off, void *dst, uint32_t len) {
    if (!ds4_dist_expert_fetch_send(slot, off, len)) return 0;
    return ds4_dist_expert_fetch_recv(slot, dst, len);
}

/* Split request/response phases so a fetcher thread can keep two requests in
 * flight per connection: the server's pread of request N+1 then overlaps the
 * network transfer of response N, and the request RTT disappears from the
 * per-unit critical path.  Responses arrive strictly in request order (the
 * server is sequential per connection), so recv calls must match send order. */
int ds4_dist_expert_fetch_send(int slot, uint64_t off, uint32_t len) {
    if (slot < 0 || slot >= g_efetch_nconns || len == 0 || len > DS4_EFETCH_MAX_LEN) return 0;
    const int fd = g_efetch_fds[slot];
    if (fd <= 0) return 0;
    ds4_efetch_req req = { off, len };
    if (dist_write_full(fd, &req, sizeof(req)) != 0) {
        close(fd);
        g_efetch_fds[slot] = -1;
        return 0;
    }
    return 1;
}

int ds4_dist_expert_fetch_recv(int slot, void *dst, uint32_t len) {
    if (slot < 0 || slot >= g_efetch_nconns || !dst || len == 0 || len > DS4_EFETCH_MAX_LEN) return 0;
    const int fd = g_efetch_fds[slot];
    if (fd <= 0) return 0;
    uint32_t status = 1;
    if (dist_read_full(fd, &status, sizeof(status)) != 1 ||
        status != 0 ||
        dist_read_full(fd, dst, len) != 1) {
        close(fd);
        g_efetch_fds[slot] = -1;
        return 0;
    }
    return 1;
}
