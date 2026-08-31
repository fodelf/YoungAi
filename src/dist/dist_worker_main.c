/* dist_worker_main.c — 机械拆自 ds4_distributed.c: worker 入口(Worker Entrypoint)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Worker Entrypoint
 * ========================================================================= */

int dist_run_worker(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size) {
    char layer_end[32];
    if (opt->layers.has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", opt->layers.end);

    char err[256];
    const char *listen_host = opt->listen_host;
    const bool reverse = dist_reverse_connect_enabled();
    /* Data (activation) listener: the coordinator dials this regardless of mode.
     * In reverse-connect mode the configured --listen port is the CONTROL port the
     * coordinator dials, so the data listener takes an ephemeral port (advertised
     * to the coordinator in HELLO). */
    int requested_port = reverse ? 0 : (opt->listen_port > 0 ? opt->listen_port : 0);
    int listen_fd = dist_open_listener(listen_host, requested_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed worker: %s\n", err);
        return 1;
    }
    int listen_port_i = dist_listener_port(listen_fd);
    if (listen_port_i <= 0) {
        fprintf(stderr, "ds4: distributed worker: could not determine data listener port\n");
        close(listen_fd);
        return 1;
    }
    const uint32_t listen_port = (uint32_t)listen_port_i;

    ds4_dist_worker_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.layer_start = opt->layers.start;
    state.layer_end = dist_resolved_layer_end(opt, (uint32_t)ds4_engine_layer_count(engine));
    state.has_output = opt->layers.has_output;
    state.ctx_size = ctx_size;
    state.listen_fd = listen_fd;
    pthread_mutex_init(&state.mu, NULL);

    pthread_t data_tid;
    if (pthread_create(&data_tid, NULL, dist_worker_data_listener_main, &state) != 0) {
        fprintf(stderr, "ds4: distributed worker: pthread_create failed for data listener\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(data_tid);

    if (reverse) {
        /* Reverse-connect: the coordinator dials us. Listen on the configured
         * --listen port for the control channel and serve each coordinator session
         * exactly like the forward read loop; HELLO still advertises our data port.
         * The worker never makes an outbound connection in this mode. */
        int ctrl_fd = dist_open_listener(listen_host, opt->listen_port, err, sizeof(err));
        if (ctrl_fd < 0) {
            fprintf(stderr, "ds4: distributed worker: control listener: %s\n", err);
            close(listen_fd);
            return 1;
        }
        fprintf(stderr,
                "ds4: distributed worker: layers %u:%s model_id=%d data_listen=%s:%u reverse control listen %s:%d, waiting for coordinator\n",
                opt->layers.start,
                layer_end,
                ds4_engine_model_id(engine),
                listen_host ? listen_host : "*",
                listen_port,
                listen_host ? listen_host : "*",
                opt->listen_port);
        for (;;) {
            struct sockaddr_storage ss;
            socklen_t slen = sizeof(ss);
            int fd = accept(ctrl_fd, (struct sockaddr *)&ss, &slen);
            if (fd < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "ds4: distributed worker: control accept failed: %s; retrying\n", strerror(errno));
                dist_sleep_reconnect();
                continue;
            }
            dist_set_socket_low_latency(fd);
            char peer_host[NI_MAXHOST], peer_port[NI_MAXSERV];
            dist_peer_name(fd, peer_host, sizeof(peer_host), peer_port, sizeof(peer_port));
            fprintf(stderr, "ds4: distributed worker: coordinator connected from %s:%s\n", peer_host, peer_port);
            /* Wave 29: dial the reverse-efetch connections NOW -- this is the
             * only window where the Thunderbolt bridge is reliably quiet
             * (coordinator staging saturates it once prefill starts and the
             * worker's ARP probes starve for the entire run). */
            ds4_gpu_expert_remote_fetch_kick();
            if (dist_send_hello(engine, opt, ctx_size, listen_port, fd) != 0) {
                fprintf(stderr, "ds4: distributed worker: failed to send HELLO: %s\n", strerror(errno));
                close(fd);
                continue;
            }
            int rc = dist_worker_read_loop_prefetch(&state, fd);
            close(fd);
            uint32_t dropped_sessions = dist_worker_clear_sessions(&state);
            if (dropped_sessions) {
                fprintf(stderr,
                        "ds4: distributed worker: cleared %u sessions after coordinator disconnect\n",
                        dropped_sessions);
            }
            fprintf(stderr, "ds4: distributed worker: coordinator disconnected%s; waiting\n",
                    rc ? " after error" : "");
        }
    }

    fprintf(stderr,
            "ds4: distributed worker: layers %u:%s model_id=%d data_listen=%s:%u connecting to coordinator %s:%d\n",
            opt->layers.start,
            layer_end,
            ds4_engine_model_id(engine),
            listen_host ? listen_host : "*",
            listen_port,
            opt->coordinator_host,
            opt->coordinator_port);

    for (;;) {
        int fd = dist_connect_endpoint(opt->coordinator_host, opt->coordinator_port, err, sizeof(err));
        if (fd < 0) {
            fprintf(stderr, "ds4: distributed worker: %s; retrying\n", err);
            dist_sleep_reconnect();
            continue;
        }

        char peer_host[NI_MAXHOST], peer_port[NI_MAXSERV];
        dist_peer_name(fd, peer_host, sizeof(peer_host), peer_port, sizeof(peer_port));
        fprintf(stderr, "ds4: distributed worker: connected to coordinator %s:%s\n", peer_host, peer_port);

        if (dist_send_hello(engine, opt, ctx_size, listen_port, fd) != 0) {
            fprintf(stderr, "ds4: distributed worker: failed to send HELLO: %s\n", strerror(errno));
            close(fd);
            dist_sleep_reconnect();
            continue;
        }

        int rc = dist_worker_read_loop_prefetch(&state, fd);
        close(fd);
        uint32_t dropped_sessions = dist_worker_clear_sessions(&state);
        if (dropped_sessions) {
            fprintf(stderr,
                    "ds4: distributed worker: cleared %u sessions after coordinator disconnect\n",
                    dropped_sessions);
        }
        fprintf(stderr, "ds4: distributed worker: coordinator disconnected%s; reconnecting\n",
                rc ? " after error" : "");
        dist_sleep_reconnect();
    }
}

