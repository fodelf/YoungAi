/* tp_rtt_probe.c — dual-host TP gate (E0): measure single round-trip latency.
 *
 * The two-host TP path syncs an all-reduce per layer (latency-bound for a
 * single-token decode: tiny payloads, n_layers x RTT per token). Whether TP
 * beats the current layer-pipeline forward (~118ms/token serial, wave-68)
 * hinges on the link+stack RTT floor. Gate: p50 RTT <= 50us.
 *
 * Standalone: sockets only, no model / Metal / ds4 deps -> memory-safe, never
 * touches the 81GiB base. Pure C99 (project no-C++ rule).
 *
 * Build:
 *   cc -O3 -std=c99 tools/tp_rtt_probe.c -o tools/tp_rtt_probe
 *
 * Run (dual-host; per [[tp_reverse_connect_workaround]] the M4 coordinator must
 * be the one that connects out — so put the server on the M1 worker):
 *   M1 (192.168.1.2):  ./tools/tp_rtt_probe server 51999 [payload_bytes]
 *   M4 (coordinator):  ./tools/tp_rtt_probe client 192.168.1.2 51999 [iters] [payload_bytes]
 *
 * payload_bytes defaults to 8 (pure latency floor). Pass the real TP all-reduce
 * size (hidden_dim*2 for fp16) as a second run to see the realistic per-sync cost.
 * Reports min / p50 / p90 / p99 / max round-trip in microseconds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define MAX_PAYLOAD (1u << 20)   /* 1 MiB cap; TP single-token payloads are tiny */

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Read exactly n bytes (TCP may fragment a single write into several reads). */
static int read_full(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;   /* peer closed */
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

static void set_nodelay(int fd) {
    int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0)
        fprintf(stderr, "tp_rtt_probe: warning: TCP_NODELAY failed: %s\n", strerror(errno));
}

static int run_server(int port, size_t payload) {
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
    if (listen(ls, 1) != 0) { perror("listen"); return 1; }
    fprintf(stderr, "tp_rtt_probe: server listening on :%d, payload=%zu B (echo loop)\n",
            port, payload);

    uint8_t *buf = malloc(payload);
    if (!buf) { fprintf(stderr, "OOM\n"); return 1; }
    for (;;) {
        struct sockaddr_in c;
        socklen_t cl = sizeof(c);
        int fd = accept(ls, (struct sockaddr *)&c, &cl);
        if (fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        set_nodelay(fd);
        fprintf(stderr, "tp_rtt_probe: client %s connected; echoing\n",
                inet_ntoa(c.sin_addr));
        /* Echo each fixed-size frame straight back until the client closes. */
        while (read_full(fd, buf, payload) == 0) {
            if (write_full(fd, buf, payload) != 0) break;
        }
        close(fd);
        fprintf(stderr, "tp_rtt_probe: client disconnected; waiting for next\n");
    }
    free(buf);
    close(ls);
    return 0;
}

static int cmp_dbl(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

static int run_client(const char *host, int port, int iters, size_t payload) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        fprintf(stderr, "tp_rtt_probe: bad host %s\n", host); return 1;
    }
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) {
        fprintf(stderr, "tp_rtt_probe: connect %s:%d failed: %s\n",
                host, port, strerror(errno));
        return 1;   /* EHOSTUNREACH here => wrong direction; see header note */
    }
    set_nodelay(fd);

    uint8_t *buf = malloc(payload);
    double  *lat = malloc((size_t)iters * sizeof(double));
    if (!buf || !lat) { fprintf(stderr, "OOM\n"); return 1; }
    memset(buf, 0xA5, payload);

    /* Warm up the path (first packets pay route/cache/window costs we don't want
     * in the percentiles). */
    const int warm = iters < 1000 ? iters : 1000;
    for (int i = 0; i < warm; i++) {
        if (write_full(fd, buf, payload) || read_full(fd, buf, payload)) {
            fprintf(stderr, "tp_rtt_probe: link broke during warmup\n"); return 1;
        }
    }

    for (int i = 0; i < iters; i++) {
        double t0 = now_ns();
        if (write_full(fd, buf, payload) || read_full(fd, buf, payload)) {
            fprintf(stderr, "tp_rtt_probe: link broke at iter %d\n", i); return 1;
        }
        lat[i] = (now_ns() - t0) / 1000.0;   /* us */
    }
    close(fd);

    qsort(lat, (size_t)iters, sizeof(double), cmp_dbl);
    double sum = 0.0;
    for (int i = 0; i < iters; i++) sum += lat[i];
    double p50 = lat[(int)(iters * 0.50)];
    double p90 = lat[(int)(iters * 0.90)];
    double p99 = lat[(int)(iters * 0.99)];

    printf("tp_rtt_probe: payload=%zu B  iters=%d\n", payload, iters);
    printf("  round-trip us:  min=%.1f  p50=%.1f  p90=%.1f  p99=%.1f  max=%.1f  mean=%.1f\n",
           lat[0], p50, p90, p99, lat[iters - 1], sum / iters);
    printf("  E0 gate (p50 <= 50us): %s\n", p50 <= 50.0 ? "PASS" : "FAIL");
    free(buf);
    free(lat);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  %s server <port> [payload_bytes]\n"
            "  %s client <host> <port> [iters] [payload_bytes]\n",
            argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "server") == 0) {
        int port = atoi(argv[2]);
        size_t payload = (argc >= 4) ? (size_t)strtoul(argv[3], NULL, 10) : 8u;
        if (payload == 0 || payload > MAX_PAYLOAD) { fprintf(stderr, "bad payload\n"); return 2; }
        return run_server(port, payload);
    }
    if (strcmp(argv[1], "client") == 0) {
        if (argc < 4) { fprintf(stderr, "client needs <host> <port>\n"); return 2; }
        const char *host = argv[2];
        int port = atoi(argv[3]);
        int iters = (argc >= 5) ? atoi(argv[4]) : 20000;
        size_t payload = (argc >= 6) ? (size_t)strtoul(argv[5], NULL, 10) : 8u;
        if (iters < 100) iters = 100;
        if (payload == 0 || payload > MAX_PAYLOAD) { fprintf(stderr, "bad payload\n"); return 2; }
        return run_client(host, port, iters, payload);
    }
    fprintf(stderr, "unknown role %s\n", argv[1]);
    return 2;
}
