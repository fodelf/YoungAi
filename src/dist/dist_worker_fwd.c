/* dist_worker_fwd.c — 机械拆自 ds4_distributed.c: worker 转发器(Worker Route Forwarding)。行为零变化。 */
#include "dist_internal.h"

void dist_worker_upstream_init(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_worker_state *state,
        int fd) {
    memset(upstream, 0, sizeof(*upstream));
    upstream->state = state;
    upstream->fd = fd;
    pthread_mutex_init(&upstream->write_mu, NULL);
    pthread_mutex_init(&upstream->forward_mu, NULL);
}

static bool dist_worker_forwarder_enqueue_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id,
        const ds4_dist_telemetry_fixed *telemetry,
        double downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    while (!forwarder->closing &&
           forwarder->pending_depth != 0 &&
           forwarder->pending_count >= forwarder->pending_depth) {
        pthread_cond_wait(&forwarder->queue_not_full, &forwarder->queue_mu);
    }
    if (forwarder->closing) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    ds4_dist_pending_request *node = calloc(1, sizeof(*node));
    if (!node) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    node->request_id = request_id;
    node->downstream_t0 = downstream_t0;
    if (telemetry) node->telemetry = *telemetry;
    if (forwarder->pending_tail) forwarder->pending_tail->next = node;
    else forwarder->pending_head = node;
    forwarder->pending_tail = node;
    forwarder->pending_count++;
    pthread_mutex_unlock(&forwarder->queue_mu);
    return true;
}

static bool dist_worker_forwarder_pop_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t *request_id,
        ds4_dist_telemetry_fixed *telemetry,
        double *downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request *node = forwarder->pending_head;
    if (!node) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    forwarder->pending_head = node->next;
    if (!forwarder->pending_head) forwarder->pending_tail = NULL;
    if (forwarder->pending_count != 0) forwarder->pending_count--;
    pthread_cond_signal(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);

    if (request_id) *request_id = node->request_id;
    if (telemetry) *telemetry = node->telemetry;
    if (downstream_t0) *downstream_t0 = node->downstream_t0;
    free(node);
    return true;
}

static bool dist_worker_forwarder_remove_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request **link = &forwarder->pending_head;
    ds4_dist_pending_request *prev = NULL;
    while (*link) {
        ds4_dist_pending_request *node = *link;
        if (node->request_id == request_id) {
            *link = node->next;
            if (forwarder->pending_tail == node) forwarder->pending_tail = prev;
            if (forwarder->pending_count != 0) forwarder->pending_count--;
            pthread_cond_signal(&forwarder->queue_not_full);
            pthread_mutex_unlock(&forwarder->queue_mu);
            free(node);
            return true;
        }
        prev = node;
        link = &node->next;
    }
    pthread_mutex_unlock(&forwarder->queue_mu);
    return false;
}

static void dist_worker_forwarder_note_send_done(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id,
        uint32_t forward_send_usec,
        double downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    for (ds4_dist_pending_request *node = forwarder->pending_head; node; node = node->next) {
        if (node->request_id == request_id) {
            node->telemetry.forward_send_usec = forward_send_usec;
            node->downstream_t0 = downstream_t0;
            break;
        }
    }
    pthread_mutex_unlock(&forwarder->queue_mu);
}

static void dist_worker_forwarder_clear_requests(ds4_dist_worker_forwarder *forwarder) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request *it = forwarder->pending_head;
    forwarder->pending_head = NULL;
    forwarder->pending_tail = NULL;
    forwarder->pending_count = 0;
    pthread_cond_broadcast(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);

    while (it) {
        ds4_dist_pending_request *next = it->next;
        free(it);
        it = next;
    }
}

static void dist_worker_forwarder_close_queue(ds4_dist_worker_forwarder *forwarder) {
    pthread_mutex_lock(&forwarder->queue_mu);
    forwarder->closing = true;
    pthread_cond_broadcast(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);
}

static void *dist_worker_forwarder_relay_main(void *arg) {
    ds4_dist_worker_forwarder *forwarder = arg;
    ds4_dist_worker_upstream *upstream = forwarder->upstream;
    int fd = forwarder->fd;
    uint8_t *buf = malloc(1024 * 1024);
    if (!buf) {
        shutdown(upstream->fd, SHUT_RDWR);
        return NULL;
    }
    DIST_DEBUG("relay start downstream=%s:%u fd=%d upstream_fd=%d",
               forwarder->host,
               forwarder->port,
               fd,
               upstream->fd);

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc <= 0) {
            uint64_t pending_request = 0;
            if (dist_worker_forwarder_pop_request(forwarder, &pending_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     pending_request,
                                                     "next worker closed connection");
            }
            DIST_DEBUG("relay read header end downstream=%s:%u rc=%d err=%s",
                       forwarder->host,
                       forwarder->port,
                       rc,
                       err);
            break;
        }
        DIST_DEBUG("relay got frame downstream=%s:%u type=%u bytes=%u",
                   forwarder->host,
                   forwarder->port,
                   type,
                   bytes);
        uint64_t expected_request = 0;
        if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
            dist_discard_bytes(fd, bytes);
            if (dist_worker_forwarder_pop_request(forwarder, &expected_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     expected_request,
                                                     "next worker did not return valid RESULT");
            }
            DIST_DEBUG("relay invalid frame downstream=%s:%u", forwarder->host, forwarder->port);
            break;
        }

        ds4_dist_result_fixed wire_result;
        rc = dist_read_full(fd, &wire_result, sizeof(wire_result));
        if (rc <= 0) {
            if (dist_worker_forwarder_pop_request(forwarder, &expected_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     expected_request,
                                                     "next worker closed while returning RESULT");
            }
            DIST_DEBUG("relay read result fixed failed downstream=%s:%u rc=%d",
                       forwarder->host,
                       forwarder->port,
                       rc);
            break;
        }

        ds4_dist_result_fixed result = wire_result;
        dist_result_from_wire(&result);
        const uint64_t got_request = dist_u64_from_halves(result.request_hi, result.request_lo);
        const uint32_t body_bytes = bytes - (uint32_t)sizeof(wire_result);
        if (result.telemetry_bytes % (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
            result.telemetry_count != result.telemetry_bytes / (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
            result.telemetry_bytes > body_bytes ||
            result.payload_bytes != body_bytes - result.telemetry_bytes) {
            dist_discard_bytes(fd, body_bytes);
            if (dist_worker_forwarder_pop_request(forwarder, &expected_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     expected_request,
                                                     "next worker RESULT metadata mismatch");
            }
            DIST_DEBUG("relay result metadata mismatch downstream=%s:%u telemetry=%u payload=%u frame=%u",
                       forwarder->host,
                       forwarder->port,
                       result.telemetry_bytes,
                       result.payload_bytes,
                       body_bytes);
            break;
        }
        ds4_dist_telemetry_fixed local_telemetry;
        double downstream_t0 = 0.0;
        if (!dist_worker_forwarder_pop_request(forwarder, &expected_request, &local_telemetry, &downstream_t0)) {
            dist_discard_bytes(fd, body_bytes);
            DIST_DEBUG("relay got unexpected result request=%llu with no pending request",
                       (unsigned long long)got_request);
            break;
        }
        if (got_request != expected_request) {
            dist_discard_bytes(fd, body_bytes);
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "next worker RESULT metadata mismatch");
            DIST_DEBUG("relay request mismatch expected=%llu got=%llu",
                       (unsigned long long)expected_request,
                       (unsigned long long)got_request);
            break;
        }
        local_telemetry.downstream_wait_usec = dist_usec_since(downstream_t0, dist_now_sec());
        const uint64_t out_telemetry_bytes64 =
            (uint64_t)result.telemetry_bytes + sizeof(ds4_dist_telemetry_fixed);
        const uint32_t out_telemetry_count = result.telemetry_count + 1u;
        if (out_telemetry_bytes64 > UINT32_MAX || out_telemetry_count == 0) {
            dist_discard_bytes(fd, body_bytes);
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "distributed telemetry chain is too large");
            break;
        }
        const uint32_t out_telemetry_bytes = (uint32_t)out_telemetry_bytes64;
        const uint64_t out_frame_bytes64 = sizeof(ds4_dist_result_fixed) +
                                           out_telemetry_bytes64 +
                                           (uint64_t)result.payload_bytes;
        if (out_frame_bytes64 > UINT32_MAX) {
            dist_discard_bytes(fd, body_bytes);
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "distributed RESULT frame is too large");
            break;
        }
        DIST_DEBUG("relay result request=%llu status=%u kind=%u telemetry=%u payload=%u",
                   (unsigned long long)got_request,
                   result.status,
                   result.result_kind,
                   out_telemetry_count,
                   result.payload_bytes);

        pthread_mutex_lock(&upstream->write_mu);
        result.telemetry_count = out_telemetry_count;
        result.telemetry_bytes = out_telemetry_bytes;
        ds4_dist_result_fixed out_wire_result = result;
        dist_result_to_wire(&out_wire_result);
        int write_rc = dist_write_frame_header(upstream->fd,
                                               DS4_DIST_MSG_RESULT,
                                               (uint32_t)out_frame_bytes64);
        if (write_rc == 0) write_rc = dist_write_full(upstream->fd, &out_wire_result, sizeof(out_wire_result));

        uint32_t remaining = result.telemetry_bytes - (uint32_t)sizeof(ds4_dist_telemetry_fixed);
        while (write_rc == 0 && remaining > 0) {
            uint32_t n = remaining < 1024u * 1024u ? remaining : 1024u * 1024u;
            rc = dist_read_full(fd, buf, n);
            if (rc <= 0) {
                write_rc = -1;
                break;
            }
            if (dist_write_full(upstream->fd, buf, n) != 0) {
                write_rc = -1;
                break;
            }
            remaining -= n;
        }
        if (write_rc == 0) {
            ds4_dist_telemetry_fixed local_wire = local_telemetry;
            dist_telemetry_to_wire(&local_wire);
            if (dist_write_full(upstream->fd, &local_wire, sizeof(local_wire)) != 0) {
                write_rc = -1;
            }
        }

        remaining = result.payload_bytes;
        while (write_rc == 0 && remaining > 0) {
            uint32_t n = remaining < 1024u * 1024u ? remaining : 1024u * 1024u;
            rc = dist_read_full(fd, buf, n);
            if (rc <= 0) {
                write_rc = -1;
                break;
            }
            if (dist_write_full(upstream->fd, buf, n) != 0) {
                write_rc = -1;
                break;
            }
            remaining -= n;
        }
        pthread_mutex_unlock(&upstream->write_mu);

        DIST_DEBUG("relay wrote result request=%llu write_rc=%d remaining=%u",
                   (unsigned long long)got_request,
                   write_rc,
                   remaining);
        if (write_rc != 0) break;
        if (remaining != 0) break;
    }

    DIST_DEBUG("relay closing upstream_fd=%d", upstream->fd);
    dist_worker_forwarder_close_queue(forwarder);
    shutdown(upstream->fd, SHUT_RDWR);
    free(buf);
    return NULL;
}

static ds4_dist_worker_forwarder *dist_worker_get_forwarder(
        ds4_dist_worker_upstream *upstream,
        const char *host,
        uint32_t port,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&upstream->forward_mu);
    for (ds4_dist_worker_forwarder *it = upstream->forwarders; it; it = it->next) {
        if (it->port == port && !strcmp(it->host, host)) {
            pthread_mutex_unlock(&upstream->forward_mu);
            return it;
        }
    }

    int fd = dist_connect_endpoint(host, (int)port, err, errlen);
    if (fd < 0) {
        pthread_mutex_unlock(&upstream->forward_mu);
        return NULL;
    }

    ds4_dist_worker_forwarder *forwarder = calloc(1, sizeof(*forwarder));
    if (!forwarder) {
        close(fd);
        pthread_mutex_unlock(&upstream->forward_mu);
        if (errlen) snprintf(err, errlen, "out of memory creating worker-to-worker forwarder");
        return NULL;
    }
    forwarder->upstream = upstream;
    snprintf(forwarder->host, sizeof(forwarder->host), "%s", host);
    forwarder->port = port;
    forwarder->fd = fd;
    forwarder->pending_depth = dist_worker_forward_window();
    pthread_mutex_init(&forwarder->send_mu, NULL);
    pthread_mutex_init(&forwarder->queue_mu, NULL);
    pthread_cond_init(&forwarder->queue_not_full, NULL);
    if (pthread_create(&forwarder->tid, NULL, dist_worker_forwarder_relay_main, forwarder) != 0) {
        pthread_cond_destroy(&forwarder->queue_not_full);
        pthread_mutex_destroy(&forwarder->queue_mu);
        pthread_mutex_destroy(&forwarder->send_mu);
        close(fd);
        free(forwarder);
        pthread_mutex_unlock(&upstream->forward_mu);
        if (errlen) snprintf(err, errlen, "failed to start worker-to-worker relay thread");
        return NULL;
    }
    forwarder->thread_started = true;
    forwarder->next = upstream->forwarders;
    upstream->forwarders = forwarder;
    pthread_mutex_unlock(&upstream->forward_mu);

    fprintf(stderr,
            "ds4: distributed worker: opened pipelined worker-to-worker connection to %s:%u (window %u)\n",
            host,
            port,
            forwarder->pending_depth);
    return forwarder;
}

void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream) {
    pthread_mutex_lock(&upstream->forward_mu);
    ds4_dist_worker_forwarder *forwarders = upstream->forwarders;
    upstream->forwarders = NULL;
    pthread_mutex_unlock(&upstream->forward_mu);

    for (ds4_dist_worker_forwarder *it = forwarders; it; it = it->next) {
        dist_worker_forwarder_close_queue(it);
        if (it->fd >= 0) shutdown(it->fd, SHUT_RDWR);
    }
    while (forwarders) {
        ds4_dist_worker_forwarder *next = forwarders->next;
        if (forwarders->thread_started) pthread_join(forwarders->tid, NULL);
        if (forwarders->fd >= 0) close(forwarders->fd);
        dist_worker_forwarder_clear_requests(forwarders);
        pthread_cond_destroy(&forwarders->queue_not_full);
        pthread_mutex_destroy(&forwarders->queue_mu);
        pthread_mutex_destroy(&forwarders->send_mu);
        free(forwarders);
        forwarders = next;
    }

    pthread_mutex_destroy(&upstream->forward_mu);
    pthread_mutex_destroy(&upstream->write_mu);
}

int dist_forward_work_to_next(
        ds4_dist_worker_upstream *upstream,
        const ds4_dist_route_entry *next,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        const ds4_dist_telemetry_fixed *telemetry,
        const void *route_blob) {
    char err[256];
    ds4_dist_worker_forwarder *forwarder =
        dist_worker_get_forwarder(upstream, next->host, next->port, err, sizeof(err));
    const uint64_t request_id = dist_u64_from_halves(work->request_hi, work->request_lo);
    if (!forwarder) {
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }

    ds4_dist_work_fixed forwarded = *work;
    forwarded.layer_start = next->layer_start;
    forwarded.layer_end = next->layer_end;
    forwarded.route_index = work->route_index + 1u;
    forwarded.flags |= DS4_DIST_WORK_F_INPUT_HC;
    forwarded.input_hc_bits = dist_activation_bits_or_default(work->input_hc_bits);
    if (!dist_activation_wire_bytes_from_f32_bytes(forwarded.input_hc_bits,
                                                   hidden_hc_bytes,
                                                   &forwarded.input_hc_bytes)) {
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    "invalid forwarded hidden-state size");
    }
    if ((next->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        forwarded.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    } else {
        forwarded.flags &= ~DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }

    pthread_mutex_lock(&forwarder->send_mu);
    const double send_t0 = dist_now_sec();
    if (!dist_worker_forwarder_enqueue_request(forwarder, request_id, telemetry, send_t0)) {
        pthread_mutex_unlock(&forwarder->send_mu);
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    "out of memory tracking forwarded request");
    }
    DIST_DEBUG("forward send request=%llu to %s:%u route_index=%u tokens=%u pos=%u bytes=%u",
               (unsigned long long)request_id,
               next->host,
               next->port,
               forwarded.route_index,
               forwarded.n_tokens,
               forwarded.pos0,
               forwarded.input_hc_bytes);
    int rc = dist_send_work_frame(forwarder->fd, &forwarded, tokens, hidden_hc, route_blob);
    const double send_t1 = dist_now_sec();
    dist_worker_forwarder_note_send_done(forwarder,
                                         request_id,
                                         dist_usec_since(send_t0, send_t1),
                                         send_t1);
    pthread_mutex_unlock(&forwarder->send_mu);
    if (rc != 0) {
        DIST_DEBUG("forward send failed request=%llu to %s:%u",
                   (unsigned long long)request_id,
                   next->host,
                   next->port);
        dist_worker_forwarder_remove_request(forwarder, request_id);
        shutdown(forwarder->fd, SHUT_RDWR);
        int err_rc = dist_worker_upstream_send_work_error(upstream,
                                                          request_id,
                                                          "failed to forward distributed work");
        shutdown(upstream->fd, SHUT_RDWR);
        return err_rc;
    }
    DIST_DEBUG("forward send ok request=%llu", (unsigned long long)request_id);
    return 1;
}

