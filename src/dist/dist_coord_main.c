/* dist_coord_main.c — 机械拆自 ds4_distributed.c: 独立 coordinator 入口(Standalone Coordinator Entrypoint)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Standalone Coordinator Entrypoint
 * ========================================================================= */

/* Layer-pipeline reverse-connect: by default the worker dials the coordinator's
 * control port. DS4_DIST_REVERSE_CONNECT=1 (also honors the legacy
 * DS4_TP_REVERSE_CONNECT) flips it so the COORDINATOR dials a listening worker
 * and the worker only ever accept()s. Works around a host where the worker's
 * outbound connect to the local link is denied (observed: macOS Local Network
 * privacy refusing ds4 over the thunderbolt bridge -> connect EHOSTUNREACH while
 * nc succeeds). The activation/data channel is already coordinator->worker, so
 * only the control direction changes; HELLO still flows worker->coordinator over
 * the socket no matter who dialed. Single remote worker only. */
bool dist_reverse_connect_enabled(void) {
    const char *e = getenv("DS4_DIST_REVERSE_CONNECT");
    if (!e || !*e) e = getenv("DS4_TP_REVERSE_CONNECT");
    return e && *e && e[0] != '0';
}

void *dist_coordinator_reverse_connect_main(void *arg) {
    ds4_dist_reverse_ctx *rc = arg;
    ds4_dist_coordinator_state *state = rc->state;
    for (;;) {
        if (state->shutting_down) break;
        char err[256];
        int fd = dist_connect_endpoint(rc->host, rc->port, err, sizeof(err));
        if (fd < 0) {
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: reverse-connect to %s:%d failed: %s; retrying\n",
                             rc->host ? rc->host : "?", rc->port, err);
            dist_sleep_reconnect();
            continue;
        }
        dist_set_socket_low_latency(fd);
        ds4_dist_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            close(fd);
            dist_sleep_reconnect();
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        /* The worker is the peer of this connection; the activation channel later
         * dials (peer_host, advertised data port), so capture the worker address
         * from the connected socket exactly as the accept path captures it. */
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        if (getpeername(fd, (struct sockaddr *)&ss, &slen) != 0 ||
            getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "%s", rc->host ? rc->host : "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "%d", rc->port);
        }
        DIST_COORD_DEBUG(state,
                         "ds4: distributed coordinator: reverse-connected to worker %s:%s\n",
                         ctx->peer_host, ctx->peer_port);
        /* Runs the full worker control session (reads HELLO, registers, serves);
         * frees ctx and closes fd when the worker disconnects, then we re-dial. */
        dist_coordinator_client_main(ctx);
        dist_sleep_reconnect();
    }
    return NULL;
}

int dist_run_coordinator(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    char err[256];
    const bool reverse = dist_reverse_connect_enabled();
    int listen_fd = -1;
    if (!reverse) {
        listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, sizeof(err));
        if (listen_fd < 0) {
            fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
            return 1;
        }
    }

    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    state.local_start = opt->layers.start;
    state.local_end = dist_resolved_layer_end(opt, state.n_layers);
    state.ctx_size = gen && gen->ctx_size > 0 ? (uint32_t)gen->ctx_size : 0u;
    state.local_has_output = opt->layers.has_output;
    state.local_can_output_head = true;
    state.replay_check = opt->replay_check;
    state.debug = opt->debug;
    state.use_control_for_work = gen && gen->prompt;
    state.prefill_chunk = opt->prefill_chunk;
    state.prefill_window = opt->prefill_window;
    state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    pthread_mutex_init(&state.mu, NULL);

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&state,
                     "ds4: distributed coordinator: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     state.model_id,
                     state.n_layers,
                     opt->layers.start,
                     local_end,
                     state.activation_bits);

    if (reverse) {
        /* Coordinator dials the worker's control listener (--coordinator HOST PORT
         * points at the worker). No local listen socket; the worker only accepts. */
        ds4_dist_reverse_ctx rctx = {
            .state = &state,
            .host = opt->coordinator_host,
            .port = opt->coordinator_port,
        };
        DIST_COORD_DEBUG(&state,
                         "ds4: distributed coordinator: reverse-connect mode, dialing worker %s:%d\n",
                         opt->coordinator_host ? opt->coordinator_host : "?",
                         opt->coordinator_port);
        if (!gen || !gen->prompt) {
            dist_coordinator_reverse_connect_main(&rctx);
            return 0;
        }
        pthread_t reverse_tid;
        if (pthread_create(&reverse_tid, NULL, dist_coordinator_reverse_connect_main, &rctx) != 0) {
            fprintf(stderr, "ds4: distributed coordinator: pthread_create failed for reverse-connect loop\n");
            return 1;
        }
        pthread_detach(reverse_tid);
        return dist_run_coordinator_generation(&state, gen);
    }

    ds4_dist_accept_ctx accept_ctx = {
        .state = &state,
        .listen_fd = listen_fd,
    };
    if (!gen || !gen->prompt) {
        dist_coordinator_accept_main(&accept_ctx);
        return 0;
    }

    pthread_t accept_tid;
    if (pthread_create(&accept_tid, NULL, dist_coordinator_accept_main, &accept_ctx) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: pthread_create failed for accept loop\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(accept_tid);

    return dist_run_coordinator_generation(&state, gen);
}

