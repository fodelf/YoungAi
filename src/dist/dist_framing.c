/* dist_framing.c — 机械拆自 ds4_distributed.c: TCP 帧与连接(TCP Framing And Connections)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * TCP Framing And Connections
 * ========================================================================= */

int dist_set_socket_low_latency(int fd) {
    int one = 1;
    int rc = 0;
    int buffer_bytes = dist_socket_buffer_bytes();
    int timeout_sec = 60;
    const char *timeout_env = getenv("DS4_DIST_SOCKET_TIMEOUT_SEC");
    if (timeout_env && timeout_env[0]) {
        char *end = NULL;
        long v = strtol(timeout_env, &end, 10);
        if (end != timeout_env && *end == '\0' && v > 0 && v <= 3600)
            timeout_sec = (int)v;
    }
    struct timeval tv = {
        .tv_sec = timeout_sec,
        .tv_usec = 0,
    };
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) rc = -1;
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
#ifdef SO_NOSIGPIPE
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) rc = -1;
#endif
    return rc;
}


int dist_write_full(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            /* The TP control socket is non-blocking (shared with the full-duplex
             * all-reduce pump); a not-yet-writable socket must poll-wait, not fail.
             * Blocking sockets (layer-pipeline) never hit EAGAIN, so this is inert
             * for them. wave-72: this was the TP decode send_step/recv_step bug. */
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = { .fd = fd, .events = POLLOUT, .revents = 0 };
                if (poll(&pfd, 1, -1) < 0) return -1;   /* block until peer ready/closed (no timeout band-aid) */
                continue;
            }
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

int dist_read_full(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            /* Non-blocking TP control socket: poll-wait for data instead of
             * failing on EAGAIN. Inert for blocking layer-pipeline sockets.
             * wave-72: recv_step failed here (worker reached recv before the
             * leader's step frame arrived after the ~18s prefill). */
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
                if (poll(&pfd, 1, -1) < 0) return -1;   /* block until peer ready/closed (no timeout band-aid) */
                continue;
            }
            return -1;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

int dist_write_frame_header(int fd, uint32_t type, uint32_t bytes) {
    ds4_dist_frame_header h = {
        htonl(DS4_DIST_MAGIC),
        htonl(type),
        htonl(bytes)
    };
    return dist_write_full(fd, &h, sizeof(h));
}

int dist_read_frame_header(int fd, uint32_t *type, uint32_t *bytes, char *err, size_t errlen) {
    ds4_dist_frame_header h;
    int rc = dist_read_full(fd, &h, sizeof(h));
    if (rc < 0 && errlen) snprintf(err, errlen, "failed to read frame header: %s", strerror(errno));
    if (rc <= 0) return rc;

    uint32_t magic = ntohl(h.magic);
    if (magic != DS4_DIST_MAGIC) {
        if (errlen) snprintf(err, errlen, "bad frame magic 0x%08x", magic);
        return -1;
    }

    *type = ntohl(h.type);
    *bytes = ntohl(h.bytes);
    return 1;
}

int dist_discard_bytes(int fd, uint32_t bytes) {
    unsigned char buf[4096];
    while (bytes > 0) {
        size_t n = bytes < sizeof(buf) ? bytes : sizeof(buf);
        int rc = dist_read_full(fd, buf, n);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        bytes -= (uint32_t)n;
    }
    return 1;
}

int dist_send_error(int fd, const char *msg) {
    if (!msg) msg = "distributed protocol error";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_ERROR, (uint32_t)len) != 0) return -1;
    return dist_write_full(fd, msg, len);
}

void dist_peer_name(int fd, char *host, size_t hostlen, char *port, size_t portlen) {
    if (hostlen) host[0] = '\0';
    if (portlen) port[0] = '\0';

    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getpeername(fd, (struct sockaddr *)&ss, &slen) == 0) {
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        host, (socklen_t)hostlen,
                        port, (socklen_t)portlen,
                        NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
            return;
        }
    }
    if (hostlen) snprintf(host, hostlen, "unknown");
    if (portlen) snprintf(port, portlen, "0");
}

int dist_open_listener(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    const char *host_display = host ? host : "*";

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host_display, portbuf, gai_strerror(gai));
        return -1;
    }

    int listen_fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        dist_set_socket_low_latency(fd);

        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(fd, 64) == 0) {
            listen_fd = fd;
            break;
        }

        close(fd);
    }

    freeaddrinfo(res);
    if (listen_fd < 0 && errlen) {
        snprintf(err, errlen, "unable to listen on %s:%d: %s", host_display, port, strerror(errno));
    }
    return listen_fd;
}

int dist_listener_port(int fd) {
    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getsockname(fd, (struct sockaddr *)&ss, &slen) != 0) return 0;
    char service[NI_MAXSERV];
    if (getnameinfo((struct sockaddr *)&ss, slen,
                    NULL, 0,
                    service, sizeof(service),
                    NI_NUMERICSERV) != 0) {
        return 0;
    }
    char *end = NULL;
    unsigned long v = strtoul(service, &end, 10);
    if (end == service || *end != '\0' || v > 65535ul) return 0;
    return (int)v;
}

static bool dist_connect_errno_retryable(int e) {
    return e == ECONNREFUSED ||
           e == EHOSTUNREACH ||
           e == ENETUNREACH ||
           e == ETIMEDOUT ||
           e == EADDRNOTAVAIL;
}

int dist_connect_endpoint_once(const char *host, int port, int *last_errno, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    if (last_errno) *last_errno = 0;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host, portbuf, gai_strerror(gai));
        if (last_errno) *last_errno = EINVAL;
        return -1;
    }

    int fd = -1;
    int saved_errno = 0;
    /* Diagnostic: DS4_TP_CONNECT_DEBUG=1 logs the resolved target, the chosen
     * source address, and connect errno. Capped so a 200x retry storm cannot
     * flood. */
    static int dbg = -1, dbg_n = 0;
    if (dbg < 0) { const char *e = getenv("DS4_TP_CONNECT_DEBUG"); dbg = (e && *e) ? 1 : 0; }
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        if (dbg && dbg_n < 12) {
            char ip[64] = "?";
            void *ad = ai->ai_family == AF_INET
                ? (void *)&((struct sockaddr_in *)ai->ai_addr)->sin_addr
                : (void *)&((struct sockaddr_in6 *)ai->ai_addr)->sin6_addr;
            inet_ntop(ai->ai_family, ad, ip, sizeof(ip));
            fprintf(stderr, "ds4: [tp-connect-dbg] try family=%s dst=%s:%s\n",
                    ai->ai_family == AF_INET ? "IPv4" : "IPv6", ip, portbuf);
            dbg_n++;
        }
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            saved_errno = errno;
            if (dbg && dbg_n < 12) { fprintf(stderr, "ds4: [tp-connect-dbg] socket errno=%s\n", strerror(saved_errno)); dbg_n++; }
            continue;
        }
        dist_set_socket_low_latency(fd);
        /* Optional: force the connect source address (出接口) to a specific local
         * IP. Diagnoses/works around the kernel picking an unreachable source. */
        const char *src_ip = getenv("DS4_TP_SRC_IP");
        if (src_ip && *src_ip && ai->ai_family == AF_INET) {
            struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));
            sa.sin_family = AF_INET;
#ifdef __APPLE__
            sa.sin_len = sizeof(sa);   /* BSD 专有字段; Linux 的 sockaddr_in 没有 */
#endif
            if (inet_pton(AF_INET, src_ip, &sa.sin_addr) == 1) {
                if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
                    if (dbg && dbg_n < 12) { fprintf(stderr, "ds4: [tp-connect-dbg] bind src %s errno=%s\n", src_ip, strerror(errno)); dbg_n++; }
                } else if (dbg && dbg_n < 12) { fprintf(stderr, "ds4: [tp-connect-dbg] bound src=%s\n", src_ip); dbg_n++; }
            }
        }
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            if (dbg) {
                struct sockaddr_storage ss; socklen_t sl = sizeof(ss); char sip[64] = "?";
                if (getsockname(fd, (struct sockaddr *)&ss, &sl) == 0) {
                    void *sa = ss.ss_family == AF_INET
                        ? (void *)&((struct sockaddr_in *)&ss)->sin_addr
                        : (void *)&((struct sockaddr_in6 *)&ss)->sin6_addr;
                    inet_ntop(ss.ss_family, sa, sip, sizeof(sip));
                }
                fprintf(stderr, "ds4: [tp-connect-dbg] CONNECT OK src=%s\n", sip);
            }
            break;
        }
        saved_errno = errno;
        if (dbg && dbg_n < 12) { fprintf(stderr, "ds4: [tp-connect-dbg] connect errno=%s\n", strerror(saved_errno)); dbg_n++; }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    if (fd < 0 && errlen) {
        if (saved_errno == 0) saved_errno = EIO;
        snprintf(err, errlen, "unable to connect to %s:%d: %s", host, port, strerror(saved_errno));
    }
    if (fd < 0 && last_errno) *last_errno = saved_errno ? saved_errno : EIO;
    return fd;
}

int dist_connect_endpoint(const char *host, int port, char *err, size_t errlen) {
    int last_errno = 0;
    for (int attempt = 0; attempt < 200; attempt++) {
        int fd = dist_connect_endpoint_once(host, port, &last_errno, err, errlen);
        if (fd >= 0) return fd;
        if (!dist_connect_errno_retryable(last_errno)) break;
        struct timespec ts = {0, 25 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    return -1;
}

