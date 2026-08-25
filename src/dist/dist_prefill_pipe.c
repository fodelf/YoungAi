/* dist_prefill_pipe.c — 机械拆自 ds4_distributed.c: 流水线 prefill 发送/读取线程(Pipelined Prefill: sender/reader)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Pipelined Prefill
 * =========================================================================
 *
 * Long prompt ingestion is chunked so the coordinator can compute its local
 * slice for chunk N+1 while downstream workers process chunk N. Intermediate
 * chunks are ACK-only; only the final chunk needs to return hidden state or
 * logits to the coordinator.
 */

int dist_prefill_sender_init(
        ds4_dist_prefill_sender *sender,
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        int fd,
        uint32_t chunk_count,
        uint32_t max_hidden_bytes,
        char *err,
        size_t errlen) {
    memset(sender, 0, sizeof(*sender));
    sender->state = state;
    sender->plan = plan;
    sender->prompt = prompt;
    sender->session_id = session_id;
    sender->fd = fd;
    sender->slot_count = dist_prefill_send_depth(chunk_count);
    pthread_mutex_init(&sender->mu, NULL);
    pthread_cond_init(&sender->can_enqueue, NULL);
    pthread_cond_init(&sender->can_dequeue, NULL);

    sender->slots = calloc(sender->slot_count, sizeof(sender->slots[0]));
    if (!sender->slots) {
        if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender slots");
        return 1;
    }
    for (uint32_t i = 0; i < sender->slot_count; i++) {
        sender->slots[i].hidden = malloc(max_hidden_bytes);
        if (!sender->slots[i].hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender hidden-state buffers");
            return 1;
        }
    }
    return 0;
}

void dist_prefill_sender_destroy(ds4_dist_prefill_sender *sender) {
    if (!sender) return;
    if (sender->slots) {
        for (uint32_t i = 0; i < sender->slot_count; i++) {
            free(sender->slots[i].hidden);
        }
        free(sender->slots);
    }
    pthread_cond_destroy(&sender->can_dequeue);
    pthread_cond_destroy(&sender->can_enqueue);
    pthread_mutex_destroy(&sender->mu);
}

ds4_dist_prefill_send_slot *dist_prefill_sender_acquire_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    while (!sender->stop && sender->queued == sender->slot_count) {
        pthread_cond_wait(&sender->can_enqueue, &sender->mu);
    }
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return NULL;
    }
    ds4_dist_prefill_send_slot *slot = &sender->slots[sender->tail];
    pthread_mutex_unlock(&sender->mu);
    return slot;
}

int dist_prefill_sender_enqueue_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return 1;
    }
    sender->tail = (sender->tail + 1u) % sender->slot_count;
    sender->queued++;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    return 0;
}

void dist_prefill_sender_finish(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
}

void dist_prefill_sender_cancel(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    sender->stop = true;
    pthread_cond_broadcast(&sender->can_enqueue);
    pthread_cond_broadcast(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    shutdown(sender->fd, SHUT_RDWR);
}

void *dist_prefill_sender_main(void *arg) {
    ds4_dist_prefill_sender *sender = arg;
    for (;;) {
        pthread_mutex_lock(&sender->mu);
        while (!sender->stop && sender->queued == 0 && !sender->producer_done) {
            pthread_cond_wait(&sender->can_dequeue, &sender->mu);
        }
        if (sender->stop || (sender->queued == 0 && sender->producer_done)) {
            pthread_mutex_unlock(&sender->mu);
            break;
        }
        ds4_dist_prefill_send_slot *slot = &sender->slots[sender->head];
        pthread_mutex_unlock(&sender->mu);

        char send_err[256];
        const double send_t0 = dist_now_sec();
        int rc = dist_coordinator_send_remote_work_on_fd(sender->state,
                                                         sender->plan,
                                                         sender->fd,
                                                         sender->prompt->v + slot->pos,
                                                         slot->n_tokens,
                                                         slot->pos,
                                                         sender->session_id,
                                                         slot->request_id,
                                                         slot->prefix_hash,
                                                         slot->result_hash,
                                                         slot->reset_session,
                                                         slot->ack_only,
                                                         slot->hidden,
                                                         slot->hidden_bytes,
                                                         0, 0, 0,
                                                         send_err,
                                                         sizeof(send_err));
        const double send_t1 = dist_now_sec();

        pthread_mutex_lock(&sender->mu);
        uint32_t slot_hidden_wire_bytes = slot->hidden_bytes;
        (void)dist_activation_wire_bytes_from_f32_bytes(sender->state->activation_bits,
                                                        slot->hidden_bytes,
                                                        &slot_hidden_wire_bytes);
        sender->send_sec += send_t1 - send_t0;
        sender->send_bytes += (uint64_t)sizeof(ds4_dist_work_fixed) +
                              (uint64_t)slot->n_tokens * sizeof(uint32_t) +
                              (uint64_t)slot_hidden_wire_bytes +
                              sender->plan->blob_bytes;
        if (rc != 0) {
            sender->rc = 1;
            snprintf(sender->err, sizeof(sender->err), "%s", send_err);
            sender->stop = true;
            pthread_cond_broadcast(&sender->can_enqueue);
            pthread_cond_broadcast(&sender->can_dequeue);
            pthread_mutex_unlock(&sender->mu);
            shutdown(sender->fd, SHUT_RDWR);
            break;
        }
        sender->head = (sender->head + 1u) % sender->slot_count;
        sender->queued--;
        pthread_cond_signal(&sender->can_enqueue);
        pthread_mutex_unlock(&sender->mu);
    }
    return NULL;
}

static void dist_prefill_reader_signal_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t completed,
        bool done) {
    pthread_mutex_lock(&reader->progress_mu);
    if (completed > reader->progress_completed) reader->progress_completed = completed;
    if (done) reader->progress_done = true;
    pthread_cond_broadcast(&reader->progress_cv);
    pthread_mutex_unlock(&reader->progress_mu);
}

void dist_prefill_reader_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported || !reader->progress_session) return;

    pthread_mutex_lock(&reader->progress_mu);
    uint32_t completed = reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);

    while (*reported < completed) {
        (*reported)++;
        uint32_t rel = (*reported) * reader->chunk_cap;
        if (rel > reader->total_tokens) rel = reader->total_tokens;
        uint32_t current = reader->progress_base + rel;
        if (current > reader->progress_total) current = reader->progress_total;
        ds4_session_report_progress(reader->progress_session,
                                    "prefill_chunk",
                                    (int)current,
                                    (int)reader->progress_total);
    }
}

bool dist_prefill_reader_wait_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported) return true;

    pthread_mutex_lock(&reader->progress_mu);
    while (reader->progress_completed <= *reported && !reader->progress_done) {
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
    }
    const bool done = reader->progress_done;
    pthread_mutex_unlock(&reader->progress_mu);

    dist_prefill_reader_emit_progress(reader, reported);
    pthread_mutex_lock(&reader->progress_mu);
    const bool finished = done && *reported >= reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);
    return finished;
}

bool dist_prefill_reader_wait_flow_window(
        ds4_dist_prefill_result_reader *reader,
        uint32_t submitted,
        uint32_t window,
        uint32_t *reported) {
    if (!reader || window == 0) return true;

    for (;;) {
        pthread_mutex_lock(&reader->progress_mu);
        const uint32_t completed = reader->progress_completed;
        const bool done = reader->progress_done;
        const bool has_room = submitted < completed + window;
        if (done || has_room) {
            pthread_mutex_unlock(&reader->progress_mu);
            dist_prefill_reader_emit_progress(reader, reported);
            return !done && has_room;
        }
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
        pthread_mutex_unlock(&reader->progress_mu);
        dist_prefill_reader_emit_progress(reader, reported);
    }
}

void *dist_prefill_result_reader_main(void *arg) {
    ds4_dist_prefill_result_reader *reader = arg;
    reader->rc = 0;
    reader->err[0] = '\0';
    reader->final_kind = 0;
    reader->final_payload = NULL;
    reader->final_payload_bytes = 0;

    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(reader->state->engine) * sizeof(float));
    for (uint32_t i = 0; i < reader->count; i++) {
        const uint64_t request_id = reader->first_request_id + (uint64_t)i;
        uint32_t kind = 0;
        uint32_t payload_bytes = 0;
        uint64_t result_hash = 0;
        void *payload = NULL;
        int recv_rc = dist_recv_result_alloc(reader->fd,
                                             reader->state,
                                             request_id,
                                             &kind,
                                             &result_hash,
                                             &payload,
                                             &payload_bytes,
                                             NULL,
                                             NULL,
                                             reader->err,
                                             sizeof(reader->err));
        if (recv_rc != 0) {
            reader->rc = recv_rc;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (reader->expected_hashes && result_hash != reader->expected_hashes[i]) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill prefix hash mismatch");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        const uint32_t pos0 = i * reader->chunk_cap;
        const uint32_t remaining = reader->total_tokens - pos0;
        const uint32_t chunk = remaining < reader->chunk_cap ? remaining : reader->chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * reader->hc_values * sizeof(float);
        const bool final_chunk = i + 1u == reader->count;
        const bool valid_ack = !final_chunk &&
                               kind == DS4_DIST_RESULT_ACK &&
                               payload_bytes == 0;
        const bool valid_logits = kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes;
        const bool valid_hidden = reader->allow_hidden &&
                                  hidden_bytes64 <= UINT32_MAX &&
                                  kind == DS4_DIST_RESULT_HIDDEN_STATE &&
                                  payload_bytes == (uint32_t)hidden_bytes64;
        if (!valid_ack && !valid_logits && !valid_hidden) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill returned invalid result");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (final_chunk) {
            reader->final_kind = kind;
            reader->final_payload = payload;
            reader->final_payload_bytes = payload_bytes;
            payload = NULL;
        }
        free(payload);
        dist_prefill_reader_signal_progress(reader, i + 1u, final_chunk);
    }
    dist_prefill_reader_signal_progress(reader, reader->count, true);
    return NULL;
}

