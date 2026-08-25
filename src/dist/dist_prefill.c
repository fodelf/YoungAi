/* dist_prefill.c — 机械拆自 ds4_distributed.c: 流水线 prefill 主流程(Pipelined Prefill: prompt paths)。行为零变化。 */
#include "dist_internal.h"

bool dist_coordinator_can_pipeline_prefill(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        ds4_session *session,
        uint32_t n_tokens,
        uint32_t chunk_cap) {
    if (getenv("DS4_DIST_DISABLE_PREFILL_PIPELINE")) return false;
    if (!state || !plan) return false;
    (void)session;
    if (chunk_cap == 0 || n_tokens <= chunk_cap) return false;
    if (plan->count == 0) return false;
    if (plan->entry[0].fd < 0) return false;
    const ds4_dist_route_entry *final = &plan->entry[plan->count - 1u];
    if ((final->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0) {
        return final->layer_end + 1u == state->n_layers &&
               state->local_can_output_head;
    }
    return true;
}

int dist_coordinator_prefill_chunk_cap(
        const ds4_dist_coordinator_state *state,
        ds4_session *session,
        uint32_t *chunk_cap,
        char *err,
        size_t errlen) {
    if (!chunk_cap) return 1;
    const int prefill_cap_i = ds4_session_prefill_cap(session);
    if (prefill_cap_i <= 0) {
        if (errlen) snprintf(err, errlen, "distributed coordinator has no prefill capacity");
        return 1;
    }
    const uint32_t prefill_cap = (uint32_t)prefill_cap_i;
    uint32_t requested = state ? state->prefill_chunk : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_CHUNK");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_CHUNK", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested == 0) requested = prefill_cap;
    if (requested > prefill_cap) {
        if (errlen) {
            snprintf(err,
                     errlen,
                     "distributed prefill chunk %u exceeds session prefill cap %u",
                     requested,
                     prefill_cap);
        }
        return 1;
    }
    *chunk_cap = requested;
    return 0;
}

static int dist_coordinator_prefill_window(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        uint32_t chunk_count,
        uint32_t *window,
        char *err,
        size_t errlen) {
    if (!window) return 1;
    uint32_t requested = state ? state->prefill_window : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_WINDOW");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_WINDOW", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested > 64u) {
        if (errlen) snprintf(err, errlen, "distributed prefill window %u exceeds limit 64", requested);
        return 1;
    }
    if (requested == 0) {
        const uint32_t remote_stages = plan ? plan->count : 0u;
        requested = remote_stages + 2u;
        if (requested < 2u) requested = 2u;
        if (requested > 8u) requested = 8u;
    }
    if (chunk_count != 0 && requested > chunk_count) requested = chunk_count;
    *window = requested ? requested : 1u;
    return 0;
}

void dist_report_prefill_progress(ds4_session *session, uint32_t current, uint32_t total) {
    if (!session) return;
    if (current > (uint32_t)INT_MAX) current = (uint32_t)INT_MAX;
    if (total > (uint32_t)INT_MAX) total = (uint32_t)INT_MAX;
    ds4_session_report_progress(session, "prefill_chunk", (int)current, (int)total);
}

int dist_coordinator_prefill_prompt_pipelined(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint32_t span_start,
        uint32_t n_tokens,
        bool reset_first_chunk,
        uint32_t chunk_cap,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    const uint32_t total = n_tokens;
    if (!prompt ||
        span_start > (uint32_t)prompt->len ||
        n_tokens == 0 ||
        n_tokens > (uint32_t)prompt->len - span_start) {
        if (errlen) snprintf(err, errlen, "invalid distributed pipelined prefill span");
        return 1;
    }
    const uint32_t span_end = span_start + n_tokens;
    const uint32_t chunk_count = (total + chunk_cap - 1u) / chunk_cap;
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t max_hidden_bytes64 = (uint64_t)chunk_cap * hc_values * sizeof(float);
    if (max_hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    const uint32_t max_hidden_bytes = (uint32_t)max_hidden_bytes64;
    uint32_t flow_window = 0;
    if (dist_coordinator_prefill_window(state,
                                        plan,
                                        chunk_count,
                                        &flow_window,
                                        err,
                                        errlen) != 0) {
        return 1;
    }

    ds4_dist_prefill_sender sender;
    if (dist_prefill_sender_init(&sender,
                                 state,
                                 plan,
                                 prompt,
                                 session_id,
                                 plan->entry[0].fd,
                                 chunk_count,
                                 max_hidden_bytes,
                                 err,
                                 errlen) != 0) {
        dist_prefill_sender_destroy(&sender);
        return 1;
    }

    ds4_dist_prefill_result_reader reader;
    memset(&reader, 0, sizeof(reader));
    reader.state = state;
    reader.fd = plan->entry[0].fd;
    reader.progress_session = session;
    reader.first_request_id = *request_id;
    reader.count = chunk_count;
    reader.total_tokens = total;
    reader.chunk_cap = chunk_cap;
    reader.progress_base = span_start;
    reader.progress_total = (uint32_t)prompt->len;
    reader.hc_values = hc_values;
    reader.allow_hidden =
        (plan->entry[plan->count - 1u].flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0;
    pthread_mutex_init(&reader.progress_mu, NULL);
    pthread_cond_init(&reader.progress_cv, NULL);
    reader.expected_hashes = calloc(chunk_count, sizeof(reader.expected_hashes[0]));
    if (!reader.expected_hashes) {
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "out of memory allocating distributed prefill hashes");
        return 1;
    }
    uint64_t chunk_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    for (uint32_t i = 0, hash_pos = span_start; i < chunk_count; i++) {
        const uint32_t remaining = span_end - hash_pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        chunk_prefix_hash = dist_token_hash_update_span(chunk_prefix_hash,
                                                        prompt->v + hash_pos,
                                                        chunk);
        reader.expected_hashes[i] = chunk_prefix_hash;
        hash_pos += chunk;
    }

    pthread_t reader_tid;
    if (pthread_create(&reader_tid, NULL, dist_prefill_result_reader_main, &reader) != 0) {
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill result reader");
        return 1;
    }
    pthread_t sender_tid;
    if (pthread_create(&sender_tid, NULL, dist_prefill_sender_main, &sender) != 0) {
        dist_prefill_sender_cancel(&sender);
        pthread_join(reader_tid, NULL);
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill sender");
        return 1;
    }

    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: pipelined prefill %u chunks of up to %u tokens through %u worker%s, first hop %s:%u, send depth %u, flow window %u\n",
                     chunk_count,
                     chunk_cap,
                     plan->count,
                     plan->count == 1u ? "" : "s",
                     plan->entry[0].host,
                     plan->entry[0].port,
                     sender.slot_count,
                     flow_window);

    int rc = 0;
    double local_eval_sec = 0.0;
    const double pipeline_t0 = dist_now_sec();
    uint32_t pos = span_start;
    uint64_t next_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    uint32_t reported_chunks = 0;
    uint32_t submitted_chunks = 0;
    while (pos < span_end) {
        if (!dist_prefill_reader_wait_flow_window(&reader,
                                                  submitted_chunks,
                                                  flow_window,
                                                  &reported_chunks)) {
            if (errlen) snprintf(err, errlen, "distributed prefill result reader stopped");
            rc = 1;
            break;
        }
        const uint32_t remaining = span_end - pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * hc_values * sizeof(float);
        const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
        ds4_dist_prefill_send_slot *slot =
            dist_prefill_sender_acquire_slot(&sender, err, errlen);
        if (!slot) {
            rc = 1;
            break;
        }
        if (pos == span_start &&
            reset_first_chunk &&
            ds4_session_layer_slice_reset(session, err, errlen) != 0) {
            rc = 1;
            break;
        }
        const double local_t0 = dist_now_sec();
        rc = ds4_session_eval_layer_slice(session,
                                          prompt->v + pos,
                                          chunk,
                                          pos,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          slot->hidden,
                                          false,
                                          NULL,
                                          err,
                                          errlen);
        const double local_t1 = dist_now_sec();
        local_eval_sec += local_t1 - local_t0;
        if (rc != 0) break;

        slot->pos = pos;
        slot->n_tokens = chunk;
        slot->hidden_bytes = hidden_bytes;
        slot->request_id = *request_id;
        slot->prefix_hash = next_prefix_hash;
        slot->result_hash = reader.expected_hashes[submitted_chunks];
        slot->reset_session = reset_first_chunk && pos == span_start;
        slot->ack_only = !getenv("DS4_DIST_DISABLE_PREFILL_ACK_ONLY") &&
                         pos + chunk < span_end;
        rc = dist_prefill_sender_enqueue_slot(&sender, err, errlen);
        if (rc != 0) break;

        dist_prefill_reader_emit_progress(&reader, &reported_chunks);
        (*request_id)++;
        submitted_chunks++;
        next_prefix_hash = slot->result_hash;
        pos += chunk;
    }

    if (rc == 0) dist_prefill_sender_finish(&sender);
    else dist_prefill_sender_cancel(&sender);
    pthread_join(sender_tid, NULL);
    if (rc == 0 && sender.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender.err[0] ? sender.err : "distributed prefill sender failed");
        rc = 1;
    }
    if (rc != 0) {
        shutdown(plan->entry[0].fd, SHUT_RDWR);
    }
    if (rc == 0) {
        while (!dist_prefill_reader_wait_emit_progress(&reader, &reported_chunks)) {
            ;
        }
    }
    pthread_join(reader_tid, NULL);
    const double pipeline_t1 = dist_now_sec();
    if (rc == 0 && reader.rc == 0) {
        const double total_sec = pipeline_t1 - pipeline_t0;
        DIST_COORD_DEBUG(state,
                         "ds4: distributed coordinator: pipelined prefill done tokens=%u chunks=%u total=%.3fs %.2f t/s local=%.3fs send=%.3fs %.2f MiB/s\n",
                         total,
                         chunk_count,
                         total_sec,
                         total_sec > 0.0 ? (double)total / total_sec : 0.0,
                         local_eval_sec,
                         sender.send_sec,
                         sender.send_sec > 0.0
                             ? ((double)sender.send_bytes / (1024.0 * 1024.0)) / sender.send_sec
                             : 0.0);
    }
    if (reader.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s", reader.err[0] ? reader.err : "distributed pipelined prefill failed");
        int reader_rc = reader.rc;
        free(reader.final_payload);
        free(reader.expected_hashes);
        dist_prefill_sender_destroy(&sender);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        return reader_rc;
    }
    dist_prefill_sender_destroy(&sender);
    free(reader.expected_hashes);
    pthread_cond_destroy(&reader.progress_cv);
    pthread_mutex_destroy(&reader.progress_mu);
    if (rc != 0) {
        free(reader.final_payload);
        return 1;
    }
    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (reader.final_kind == DS4_DIST_RESULT_LOGITS &&
        reader.final_payload_bytes == logits_bytes) {
        memcpy(logits, reader.final_payload, logits_bytes);
        free(reader.final_payload);
        return 0;
    }
    if (reader.final_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        reader.final_payload) {
        const uint32_t last_pos = (chunk_count - 1u) * chunk_cap;
        const uint32_t last_tokens = total - last_pos;
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           reader.final_payload,
                                                           last_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(reader.final_payload);
        return head_rc;
    }
    free(reader.final_payload);
    if (errlen) snprintf(err, errlen, "distributed pipelined prefill did not return a final result");
    return 1;
}

int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    uint32_t chunk_cap = 0;
    if (dist_coordinator_prefill_chunk_cap(state, session, &chunk_cap, err, errlen) != 0) {
        return 1;
    }
    const uint32_t prompt_len = (uint32_t)prompt->len;
    if (dist_coordinator_can_pipeline_prefill(state, plan, session, prompt_len, chunk_cap)) {
        return dist_coordinator_prefill_prompt_pipelined(state,
                                                        session,
                                                        plan,
                                                        prompt,
                                                        0,
                                                        prompt_len,
                                                        true,
                                                        chunk_cap,
                                                        session_id,
                                                        request_id,
                                                        logits,
                                                        err,
                                                        errlen);
    }

    uint32_t pos = 0;
    while (pos < prompt_len) {
        uint32_t remaining = prompt_len - pos;
        uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        int eval_rc = dist_coordinator_eval_span(state, session, plan,
                                                 prompt->v + pos, chunk, pos,
                                                 session_id, (*request_id)++,
                                                 pos == 0, logits, NULL, err, errlen);
        if (eval_rc != 0) {
            return eval_rc;
        }
        pos += chunk;
        dist_report_prefill_progress(session, pos, prompt_len);
    }
    return 0;
}

