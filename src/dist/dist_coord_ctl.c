/* dist_coord_ctl.c — 机械拆自 ds4_distributed.c: coordinator 控制面(Coordinator Control Plane)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Coordinator Control Plane
 * ========================================================================= */

static void dist_coordinator_remove_worker(ds4_dist_coordinator_state *state, int fd) {
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *entry = *link;
        if (entry->fd == fd) {
            *link = entry->next;
            state->generation++;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: removed worker %s:%s layers=%u:%u%s\n",
                             entry->peer_host,
                             entry->peer_port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            pthread_mutex_unlock(&state->mu);
            free(entry);
            if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
            return;
        }
        link = &entry->next;
    }
    pthread_mutex_unlock(&state->mu);
}

static void dist_coordinator_monitor_worker_fd(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port) {
    for (;;) {
        struct pollfd pfd = {
            .fd = fd,
            .events = 0,
            .revents = 0,
        };
        int rc = poll(&pfd, 1, 1000);
        if (rc < 0) {
            if (errno == EINTR) continue;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: worker %s:%s poll failed: %s\n",
                             peer_host,
                             peer_port,
                             strerror(errno));
            break;
        }
        if (rc == 0) continue;
        if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) break;
    }
}

void *dist_coordinator_client_main(void *arg) {
    ds4_dist_client_ctx *ctx = arg;
    int fd = ctx->fd;
    ds4_dist_coordinator_state *state = ctx->state;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    ds4_dist_hello_fixed hello;
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    char err[256];
    int rc = dist_recv_hello(fd, &hello, model_name, sizeof(model_name), err, sizeof(err));
    if (rc <= 0) {
        if (rc < 0) DIST_COORD_DEBUG(state, "ds4: distributed coordinator: bad HELLO from %s:%s: %s\n", peer_host, peer_port, err);
        close(fd);
        return NULL;
    }

    if (hello.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: worker=%u coordinator=%u", hello.model_id, state->model_id);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    const char *expected_model_name = ds4_engine_model_name(state->engine);
    if (!expected_model_name) expected_model_name = "unknown";
    if (strcmp(model_name, expected_model_name) != 0) {
        snprintf(err,
                 sizeof(err),
                 "model family mismatch: worker=%s coordinator=%s",
                 model_name,
                 expected_model_name);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.n_layers != state->n_layers) {
        snprintf(err, sizeof(err), "layer count mismatch: worker=%u coordinator=%u", hello.n_layers, state->n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.quant_bits != 2u && hello.quant_bits != 4u) {
        snprintf(err, sizeof(err), "unsupported worker quant profile Q%u", hello.quant_bits);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output > 1u) {
        snprintf(err, sizeof(err), "invalid worker output-head flag %u", hello.has_output);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_hidden > 1u) {
        snprintf(err, sizeof(err), "invalid worker hidden-state flag %u", hello.has_hidden);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.layer_start >= hello.n_layers || hello.layer_end >= hello.n_layers || hello.layer_end < hello.layer_start) {
        snprintf(err, sizeof(err), "invalid worker layer range %u:%u for %u layers", hello.layer_start, hello.layer_end, hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output && hello.layer_end + 1u != hello.n_layers) {
        snprintf(err,
                 sizeof(err),
                 "worker output head requires final layer: range=%u:%u layers=%u",
                 hello.layer_start,
                 hello.layer_end,
                 hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (state->ctx_size != 0 && hello.ctx_size < state->ctx_size) {
        snprintf(err,
                 sizeof(err),
                 "worker context too small: worker=%u coordinator=%u",
                 hello.ctx_size,
                 state->ctx_size);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.listen_port == 0 || hello.listen_port > 65535u) {
        snprintf(err, sizeof(err), "invalid worker data listen port %u", hello.listen_port);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }

    dist_coordinator_add_worker(state, fd, peer_host, peer_port, &hello, model_name);

    if (state->use_control_for_work) {
        dist_coordinator_monitor_worker_fd(state, fd, peer_host, peer_port);
    } else {
        for (;;) {
            uint32_t type = 0, bytes = 0;
            rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
            if (rc == 0) break;
            if (rc < 0) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s protocol error: %s\n", peer_host, peer_port, err);
                break;
            }
            if (type == DS4_DIST_MSG_HELLO) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s sent duplicate HELLO\n", peer_host, peer_port);
                dist_discard_bytes(fd, bytes);
                break;
            }
            rc = dist_discard_bytes(fd, bytes);
            if (rc <= 0) break;
        }
    }

    dist_coordinator_remove_worker(state, fd);
    close(fd);
    return NULL;
}

void *dist_coordinator_accept_main(void *arg) {
    ds4_dist_accept_ctx *accept_ctx = arg;
    int listen_fd = accept_ctx->listen_fd;
    ds4_dist_coordinator_state *state = accept_ctx->state;

    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (errno == EBADF || errno == EINVAL) break;
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory accepting worker\n");
            close(fd);
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "0");
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, dist_coordinator_client_main, ctx) != 0) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: pthread_create failed\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

uint64_t dist_make_session_id(const void *ptr) {
    uint64_t id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    id ^= ((uint64_t)(uintptr_t)ptr << 17) ^ (uint64_t)(uintptr_t)ptr;
    id ^= (uint64_t)clock();
    return id ? id : 1u;
}

int dist_session_ensure_route(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return 1;
    }
    uint64_t generation = dist_coordinator_generation(&d->state);
    if (d->plan_ready && d->plan_generation == generation) return 0;
    dist_route_plan_free(&d->plan);
    if (!dist_coordinator_ensure_route(&d->state, &d->plan, &generation, err, errlen)) {
        d->plan_ready = false;
        d->plan_generation = 0;
        return 1;
    }
    d->plan_ready = true;
    d->plan_generation = generation;
    return 0;
}

