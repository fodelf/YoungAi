/* tp_bw_probe.c — dual-host Thunderbolt IP THROUGHPUT probe.
 *
 * The code-edit verify-batch wall is cold routed-expert gather: ds4-io shows
 * ~960 MiB/layer pulled at a measured AGGREGATE ~3.78 GB/s, split local-SSD
 * (cold_mib) + remote-over-Thunderbolt (rfetch_mib). Open question (also flagged
 * by the 2026-06 lit review): is that 3.78 GB/s the combined HARDWARE ceiling, or
 * is the remote-fetch (Thunderbolt IP) underutilised and leaving gather headroom?
 * project.md measured single-SSD bandwidth and RTT but NEVER the real TB IP GB/s.
 *
 * This probe measures sustained one-way TCP throughput (server streams bytes,
 * client pulls) for 1/2/4/6 parallel connections — 6 matches the production
 * DS4_METAL_EXPERT_REMOTE_FETCH_CONNS. If aggregate GB/s here >> the remote slice
 * of the gather, the gather is leaving TB headroom (optimise the fetch). If it
 * plateaus at/below the gather rate, the gather is at the hardware wall.
 *
 * Standalone: sockets + pthreads only, no model / Metal / ds4 deps -> memory-safe,
 * never touches the 81GiB base. Pure C99 (project no-C++ rule).
 *
 * Build:
 *   cc -O3 -std=c99 tools/tp_bw_probe.c -o tools/tp_bw_probe -pthread
 *
 * Run (dual-host; per [[tp_reverse_connect_workaround]] the M4 coordinator must be
 * the side that connects out — so put the server on the M1 worker):
 *   M1 (192.168.1.2):  ./tools/tp_bw_probe server 51998
 *   M4 (coordinator):  ./tools/tp_bw_probe client 192.168.1.2 51998 [mib_per_conn]
 *
 * mib_per_conn defaults to 512 (each connection pulls 512 MiB per data point).
 * The client sweeps conns = 1,2,4,6 and prints aggregate + per-conn GB/s.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define CHUNK (4u << 20)        /* 4 MiB read/write granularity */
#define SOCKBUF (4 << 20)       /* 4 MiB SO_RCVBUF/SNDBUF (macOS maxsockbuf-safe) */
#define MAX_CONNS 16

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int read_full(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        sent += (size_t)w;
    }
    return 0;
}

static void tune_sock(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int sb = SOCKBUF;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sb, sizeof(sb));
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sb, sizeof(sb));
}

/* ---------------- server: stream `nbytes` (8-byte request) per connection ------ */
static void *server_conn(void *arg) {
    int fd = (int)(intptr_t)arg;
    tune_sock(fd);
    uint64_t nbytes = 0;
    if (read_full(fd, &nbytes, sizeof(nbytes)) != 0) { close(fd); return NULL; }
    uint8_t *buf = malloc(CHUNK);
    if (!buf) { close(fd); return NULL; }
    memset(buf, 0xA5, CHUNK);
    uint64_t rem = nbytes;
    while (rem > 0) {
        size_t w = rem < CHUNK ? (size_t)rem : CHUNK;
        if (write_full(fd, buf, w) != 0) break;
        rem -= w;
    }
    free(buf);
    close(fd);
    return NULL;
}

static int run_server(int port) {
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(ls, (struct sockaddr *)&a, sizeof(a)) != 0) { perror("bind"); return 1; }
    if (listen(ls, MAX_CONNS) != 0) { perror("listen"); return 1; }
    fprintf(stderr, "tp_bw_probe: server listening on :%d (stream-on-request)\n", port);
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        pthread_t th;
        if (pthread_create(&th, NULL, server_conn, (void *)(intptr_t)fd) != 0) {
            close(fd); continue;
        }
        pthread_detach(th);
    }
    close(ls);
    return 0;
}

/* ---------------- client: K parallel pulls, measure aggregate GB/s ------------- */
typedef struct {
    const char *host;
    int port;
    uint64_t nbytes;
    double secs;     /* out: this connection's wall time */
    int ok;          /* out */
} pull_arg_t;

static void *pull_thread(void *argp) {
    pull_arg_t *pa = (pull_arg_t *)argp;
    pa->ok = 0;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return NULL;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)pa->port);
    if (inet_pton(AF_INET, pa->host, &a.sin_addr) != 1) { close(fd); return NULL; }
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) {
        fprintf(stderr, "tp_bw_probe: connect %s:%d failed: %s\n",
                pa->host, pa->port, strerror(errno));
        close(fd); return NULL;
    }
    tune_sock(fd);
    uint8_t *buf = malloc(CHUNK);
    if (!buf) { close(fd); return NULL; }
    if (write_full(fd, &pa->nbytes, sizeof(pa->nbytes)) != 0) { free(buf); close(fd); return NULL; }
    double t0 = now_ns();
    uint64_t rem = pa->nbytes;
    while (rem > 0) {
        size_t r = rem < CHUNK ? (size_t)rem : CHUNK;
        if (read_full(fd, buf, r) != 0) { fprintf(stderr, "tp_bw_probe: pull broke\n"); free(buf); close(fd); return NULL; }
        rem -= r;
    }
    pa->secs = (now_ns() - t0) / 1e9;
    pa->ok = 1;
    free(buf);
    close(fd);
    return NULL;
}

static int run_sweep(const char *host, int port, uint64_t mib_per_conn) {
    const int conn_set[] = {1, 2, 4, 6};
    const uint64_t nbytes = mib_per_conn * (1u << 20);
    printf("tp_bw_probe: host=%s port=%d  %llu MiB/conn  (4MiB chunks, 4MiB sockbuf)\n",
           host, port, (unsigned long long)mib_per_conn);
    printf("  conns | total GiB | wall s | AGG GB/s | per-conn GB/s (min..max)\n");
    for (size_t ci = 0; ci < sizeof(conn_set) / sizeof(conn_set[0]); ci++) {
        int conns = conn_set[ci];
        pthread_t th[MAX_CONNS];
        pull_arg_t pa[MAX_CONNS];
        for (int i = 0; i < conns; i++) {
            pa[i].host = host; pa[i].port = port; pa[i].nbytes = nbytes; pa[i].ok = 0; pa[i].secs = 0;
        }
        double t0 = now_ns();
        int spawned = 0;
        for (int i = 0; i < conns; i++) {
            if (pthread_create(&th[i], NULL, pull_thread, &pa[i]) != 0) break;
            spawned++;
        }
        for (int i = 0; i < spawned; i++) pthread_join(th[i], NULL);
        double wall = (now_ns() - t0) / 1e9;

        uint64_t total = 0;
        double pcmin = 1e9, pcmax = 0.0;
        int okc = 0;
        for (int i = 0; i < spawned; i++) {
            if (!pa[i].ok) continue;
            okc++;
            total += nbytes;
            double gbps = (double)nbytes / 1e9 / pa[i].secs;
            if (gbps < pcmin) pcmin = gbps;
            if (gbps > pcmax) pcmax = gbps;
        }
        if (okc != conns) {
            printf("  %5d | (only %d/%d conns ok — link/server issue)\n", conns, okc, conns);
            continue;
        }
        double agg = (double)total / 1e9 / wall;
        printf("  %5d | %9.2f | %6.3f | %8.2f | %.2f..%.2f\n",
               conns, (double)total / (double)(1u << 30), wall, agg, pcmin, pcmax);
    }
    printf("  (compare AGG vs the gather's ~3.78 GB/s and the remote/rfetch slice:\n"
           "   AGG >> remote slice => TB headroom, optimise fetch; AGG ~= => hardware wall.)\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  %s server <port>\n"
            "  %s client <host> <port> [mib_per_conn]\n",
            argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "server") == 0) {
        return run_server(atoi(argv[2]));
    }
    if (strcmp(argv[1], "client") == 0) {
        if (argc < 4) { fprintf(stderr, "client needs <host> <port>\n"); return 2; }
        const char *host = argv[2];
        int port = atoi(argv[3]);
        uint64_t mib = (argc >= 5) ? strtoull(argv[4], NULL, 10) : 512u;
        if (mib == 0) mib = 512u;
        return run_sweep(host, port, mib);
    }
    fprintf(stderr, "unknown role %s\n", argv[1]);
    return 2;
}
