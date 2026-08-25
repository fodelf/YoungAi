/* dist_worker_prefetch.c — 机械拆自 ds4_distributed.c: worker 预取队列(Worker Prefetch Queue)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Worker Prefetch Queue
 * ========================================================================= */

static void dist_worker_job_free(ds4_dist_worker_job *job) {
    if (!job) return;
    free(job->payload);
    free(job);
}

static void dist_worker_job_queue_init(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream) {
    memset(q, 0, sizeof(*q));
    q->state = state;
    q->upstream = upstream;
    q->depth = dist_worker_prefetch_depth();
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void dist_worker_job_queue_clear_locked(ds4_dist_worker_job_queue *q) {
    ds4_dist_worker_job *it = q->head;
    q->head = NULL;
    q->tail = NULL;
    q->queued = 0;
    while (it) {
        ds4_dist_worker_job *next = it->next;
        dist_worker_job_free(it);
        it = next;
    }
}

static void dist_worker_job_queue_destroy(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    dist_worker_job_queue_clear_locked(q);
    pthread_mutex_unlock(&q->mu);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
    pthread_mutex_destroy(&q->mu);
}

static void dist_worker_job_queue_finish(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static void dist_worker_job_queue_cancel(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    q->canceled = true;
    dist_worker_job_queue_clear_locked(q);
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static bool dist_worker_job_queue_enqueue(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_job *job) {
    pthread_mutex_lock(&q->mu);
    while (!q->closed && !q->canceled && q->queued >= q->depth) {
        pthread_cond_wait(&q->not_full, &q->mu);
    }
    if (q->closed || q->canceled) {
        pthread_mutex_unlock(&q->mu);
        return false;
    }
    if (q->tail) q->tail->next = job;
    else q->head = job;
    q->tail = job;
    q->queued++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return true;
}

static ds4_dist_worker_job *dist_worker_job_queue_pop(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    while (!q->head && !q->closed && !q->canceled) {
        pthread_cond_wait(&q->not_empty, &q->mu);
    }
    if (q->canceled || !q->head) {
        pthread_mutex_unlock(&q->mu);
        return NULL;
    }
    ds4_dist_worker_job *job = q->head;
    q->head = job->next;
    if (!q->head) q->tail = NULL;
    q->queued--;
    job->next = NULL;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return job;
}

static void *dist_worker_prefetch_eval_main(void *arg) {
    ds4_dist_worker_job_queue *q = arg;
    for (;;) {
        ds4_dist_worker_job *job = dist_worker_job_queue_pop(q);
        if (!job) break;
        int rc = dist_worker_process_work_payload(q->state,
                                                  q->upstream,
                                                  job->payload,
                                                  job->bytes);
        dist_worker_job_free(job);
        if (rc <= 0) {
            pthread_mutex_lock(&q->mu);
            q->rc = rc == 0 ? 0 : 1;
            pthread_mutex_unlock(&q->mu);
            dist_worker_job_queue_cancel(q);
            shutdown(q->upstream->fd, SHUT_RDWR);
            break;
        }
    }
    return NULL;
}

int dist_worker_read_loop_prefetch(ds4_dist_worker_state *state, int fd) {
    ds4_dist_worker_upstream upstream;
    dist_worker_upstream_init(&upstream, state, fd);

    ds4_dist_worker_job_queue queue;
    dist_worker_job_queue_init(&queue, state, &upstream);

    pthread_t eval_tid;
    if (pthread_create(&eval_tid, NULL, dist_worker_prefetch_eval_main, &queue) != 0) {
        dist_worker_job_queue_destroy(&queue);
        dist_worker_upstream_destroy(&upstream);
        return 1;
    }

    int loop_rc = 0;
    fprintf(stderr,
            "ds4: distributed worker: receive prefetch depth %u enabled\n",
            queue.depth);

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc == 0) break;
        if (rc < 0) {
            fprintf(stderr, "ds4: distributed worker: protocol error: %s\n", err);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_ERROR) {
            char msg[512];
            uint32_t n = bytes < sizeof(msg) - 1u ? bytes : (uint32_t)sizeof(msg) - 1u;
            rc = dist_read_full(fd, msg, n);
            if (rc <= 0) {
                loop_rc = 1;
                break;
            }
            msg[n] = '\0';
            if (bytes > n) dist_discard_bytes(fd, bytes - n);
            fprintf(stderr, "ds4: distributed worker: coordinator error: %s\n", msg);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_WORK) {
            ds4_dist_worker_job *job = calloc(1, sizeof(*job));
            if (!job) {
                dist_discard_bytes(fd, bytes);
                dist_worker_upstream_send_work_error(&upstream, 0, "out of memory queueing distributed WORK");
                loop_rc = 1;
                break;
            }
            job->payload = malloc(bytes);
            job->bytes = bytes;
            if (!job->payload) {
                dist_worker_job_free(job);
                dist_discard_bytes(fd, bytes);
                dist_worker_upstream_send_work_error(&upstream, 0, "out of memory reading distributed WORK frame");
                loop_rc = 1;
                break;
            }
            rc = dist_read_full(fd, job->payload, bytes);
            if (rc <= 0) {
                dist_worker_job_free(job);
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            if (!dist_worker_job_queue_enqueue(&queue, job)) {
                dist_worker_job_free(job);
                loop_rc = 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_SAVE_REQ) {
            rc = dist_worker_handle_snapshot_save(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN) {
            rc = dist_worker_handle_snapshot_load(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        rc = dist_discard_bytes(fd, bytes);
        if (rc <= 0) {
            loop_rc = rc == 0 ? 0 : 1;
            break;
        }
        pthread_mutex_lock(&upstream.write_mu);
        dist_send_error(fd, "unsupported distributed worker frame");
        pthread_mutex_unlock(&upstream.write_mu);
        fprintf(stderr, "ds4: distributed worker: rejected unsupported frame type %u\n", type);
        loop_rc = 1;
        break;
    }

    if (loop_rc == 0) dist_worker_job_queue_finish(&queue);
    else dist_worker_job_queue_cancel(&queue);
    pthread_join(eval_tid, NULL);
    if (loop_rc == 0 && queue.rc != 0) loop_rc = 1;
    dist_worker_job_queue_destroy(&queue);
    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static void *dist_worker_data_client_main(void *arg) {
    ds4_dist_data_client_ctx *ctx = arg;
    ds4_dist_worker_state *state = ctx->state;
    int fd = ctx->fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")
        ? dist_worker_read_loop(state, fd)
        : dist_worker_read_loop_prefetch(state, fd);
    if (rc != 0) {
        fprintf(stderr,
                "ds4: distributed worker: data connection %s:%s closed after error\n",
                peer_host,
                peer_port);
    }

    close(fd);
    return NULL;
}

void *dist_worker_data_listener_main(void *arg) {
    ds4_dist_worker_state *state = arg;
    int listen_fd = state->listen_fd;
    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "ds4: distributed worker: data accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_data_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            fprintf(stderr, "ds4: distributed worker: out of memory accepting data connection\n");
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
        if (pthread_create(&tid, NULL, dist_worker_data_client_main, ctx) != 0) {
            fprintf(stderr, "ds4: distributed worker: pthread_create failed for data connection\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

