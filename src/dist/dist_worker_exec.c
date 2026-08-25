/* dist_worker_exec.c — 机械拆自 ds4_distributed.c: worker 层执行(Worker Layer Execution)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Worker Layer Execution
 * ========================================================================= */

int dist_worker_process_work_payload(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        const void *payload,
        uint32_t bytes) {
    uint64_t request_id = 0;
    char err[256];
    if (bytes < sizeof(ds4_dist_work_fixed)) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "truncated distributed WORK frame");
    }

    ds4_dist_mem_reader reader = {
        .p = payload,
        .remaining = bytes,
    };
    ds4_dist_work_fixed work;
    int rc = dist_mem_read(&reader, &work, (uint32_t)sizeof(work));
    if (rc <= 0) return -1;
    dist_work_from_wire(&work);
    const uint64_t session_id = dist_u64_from_halves(work.session_hi, work.session_lo);
    request_id = dist_u64_from_halves(work.request_hi, work.request_lo);
    const uint64_t work_prefix_hash = dist_u64_from_halves(work.prefix_hash_hi,
                                                           work.prefix_hash_lo);
    const uint64_t work_result_hash = dist_u64_from_halves(work.result_hash_hi,
                                                           work.result_hash_lo);
    DIST_DEBUG("worker work request=%llu layers=%u:%u tokens=%u pos=%u flags=0x%x token_bytes=%u input_hc=%u/%ub route_count=%u route_index=%u route_bytes=%u",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               work.flags,
               work.token_bytes,
               work.input_hc_bytes,
               work.input_hc_bits,
               work.route_count,
               work.route_index,
               work.route_bytes);

    const uint32_t remaining = bytes - (uint32_t)sizeof(work);
    const uint64_t token_bytes_expected = (uint64_t)work.n_tokens * sizeof(uint32_t);
    const uint64_t payload_bytes_expected =
        (uint64_t)work.token_bytes + work.input_hc_bytes + work.route_bytes;
    if ((uint64_t)work.token_bytes != token_bytes_expected ||
        payload_bytes_expected != remaining) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK payload sizes");
    }
    if (work.route_count == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame is missing distributed route");
    }
    if (work.route_index >= work.route_count) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK route metadata");
    }
    if (work.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: work=%u worker=%u", work.model_id, state->model_id);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if (work.layer_start != state->layer_start || work.layer_end != state->layer_end) {
        snprintf(err, sizeof(err), "worker is assigned layers %u:%u but request asked for %u:%u",
                 state->layer_start, state->layer_end, work.layer_start, work.layer_end);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & ~DS4_DIST_WORK_F_VALID_MASK) != 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK flags");
    }
    if (work.n_tokens == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame has no tokens");
    }
    if (work.pos0 > (uint32_t)state->ctx_size ||
        work.n_tokens > (uint32_t)state->ctx_size - work.pos0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token span exceeds worker context");
    }

    const bool output_logits = (work.flags & DS4_DIST_WORK_F_OUTPUT_LOGITS) != 0;
    const bool input_hc_present = (work.flags & DS4_DIST_WORK_F_INPUT_HC) != 0;
    const bool ack_only = (work.flags & DS4_DIST_WORK_F_ACK_ONLY) != 0;
    if (input_hc_present && work.layer_start == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "layer 0 WORK must not provide input hidden-state");
    }
    if (!input_hc_present && work.layer_start != 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "nonzero layer WORK requires input hidden-state");
    }
    if (output_logits && !state->has_output) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "worker was not assigned the output head");
    }
    const uint32_t n_layers = (uint32_t)ds4_engine_layer_count(state->engine);
    if (output_logits && work.layer_end + 1u != n_layers) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK logits require final transformer layer");
    }

    int *tokens = malloc((size_t)work.n_tokens * sizeof(tokens[0]));
    if (!tokens) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory reading WORK tokens");
    }
    for (uint32_t i = 0; i < work.n_tokens; i++) {
        uint32_t wire_token = 0;
        rc = dist_mem_read(&reader, &wire_token, (uint32_t)sizeof(wire_token));
        if (rc <= 0) {
            free(tokens);
            return -1;
        }
        uint32_t token = ntohl(wire_token);
        if (token > (uint32_t)INT_MAX || token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token id is outside the model vocabulary");
        }
        tokens[i] = (int)token;
    }
    if (dist_token_hash_update_span(work_prefix_hash, tokens, work.n_tokens) !=
        work_result_hash) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token prefix hash metadata mismatch");
    }

    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t expected_hc_values = (uint64_t)work.n_tokens * hc_values;
    const uint64_t expected_hc_bytes64 = expected_hc_values * sizeof(float);
    if (expected_hc_bytes64 > UINT32_MAX) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "distributed hidden-state payload is too large");
    }
    const uint32_t expected_hc_bytes = (uint32_t)expected_hc_bytes64;
    const uint32_t input_hc_bits = dist_activation_bits_or_default(work.input_hc_bits);

    float *input_hc = NULL;
    const void *input_hc_wire = NULL;
    if (input_hc_present) {
        uint32_t expected_hc_wire_bytes = 0;
        if (!dist_activation_bits_valid(input_hc_bits) ||
            !dist_activation_wire_bytes(input_hc_bits,
                                        expected_hc_values,
                                        &expected_hc_wire_bytes)) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed activation width");
        }
        if (work.input_hc_bytes != expected_hc_wire_bytes) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "input hidden-state size does not match token span");
        }
        if (work.input_hc_bytes > reader.remaining) {
            DIST_DEBUG("worker input hidden read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       -1,
                       work.input_hc_bytes);
            free(tokens);
            return -1;
        }
        input_hc_wire = reader.p;
        reader.p += work.input_hc_bytes;
        reader.remaining -= work.input_hc_bytes;
        DIST_DEBUG("worker input hidden read ok request=%llu bytes=%u bits=%u",
                   (unsigned long long)request_id,
                   work.input_hc_bytes,
                   input_hc_bits);
    } else if (work.input_hc_bytes != 0) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame has hidden bytes without input flag");
    }
    void *route_blob = NULL;
    if (work.route_bytes != 0) {
        route_blob = malloc(work.route_bytes);
        if (!route_blob) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory reading distributed route");
        }
        rc = dist_mem_read(&reader, route_blob, work.route_bytes);
        if (rc <= 0) {
            DIST_DEBUG("worker route read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       rc,
                       work.route_bytes);
            free(route_blob);
            free(tokens);
            return -1;
        }
        DIST_DEBUG("worker route read ok request=%llu bytes=%u",
                   (unsigned long long)request_id,
                   work.route_bytes);
    }

    ds4_dist_route_entry current_route;
    ds4_dist_route_entry next_route;
    const bool has_route = work.route_count != 0;
    const bool has_next = has_route && work.route_index + 1u < work.route_count;
    if (has_route) {
        if (!dist_route_validate_blob(route_blob, work.route_bytes, work.route_count,
                                      n_layers,
                                      err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (!dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index, &current_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (current_route.layer_start != work.layer_start ||
            current_route.layer_end != work.layer_end) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK layer range does not match route entry");
        }
        const bool route_output_logits = (current_route.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
        if (route_output_logits != output_logits) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK logits flag does not match route entry");
        }
        if (has_next &&
            !dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index + 1u, &next_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
    }
    if (has_next && output_logits) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "non-final route entry requested logits");
    }
    if (has_route && !has_next) {
        ds4_dist_route_return ret;
        if (!dist_route_get_return_target(route_blob, work.route_bytes, work.route_count,
                                          &ret, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "unsupported final result destination");
        }
    }

    const bool final_ack_only = ack_only && !has_next;
    const bool local_output_logits = output_logits && !has_next && !final_ack_only;
    const bool produce_hidden = !local_output_logits && !final_ack_only;
    /* docs/archive/mtp.md Phase 1: a VERIFY frame runs the per-row output head over the whole
     * K-token candidate batch and returns K logit rows (one per position). */
    const bool is_verify = local_output_logits &&
                           (work.flags & DS4_DIST_WORK_F_VERIFY) != 0;
    const uint32_t vocab_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    const uint32_t result_kind = final_ack_only
        ? DS4_DIST_RESULT_ACK
        : (local_output_logits ? DS4_DIST_RESULT_LOGITS : DS4_DIST_RESULT_HIDDEN_STATE);
    const uint32_t result_bytes = final_ack_only
        ? 0u
        : (local_output_logits
            ? (is_verify ? (uint32_t)((uint64_t)work.n_tokens * vocab_bytes) : vocab_bytes)
            : expected_hc_bytes);
    float *result = result_bytes ? malloc(result_bytes) : NULL;
    if (result_bytes && !result) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory allocating distributed result");
    }

        uint32_t draft_wire[16];
    uint32_t draft_n = 0;

    bool input_hc_uses_wire = false;
    uint32_t input_hc_decoded_bytes = 0;
    if (input_hc_present &&
        dist_decode_activation_payload(input_hc_wire,
                                       input_hc_bits,
                                       work.input_hc_bytes,
                                       &input_hc,
                                       &input_hc_decoded_bytes,
                                       &input_hc_uses_wire,
                                       err,
                                       sizeof(err)) != 0) {
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if (input_hc_present && input_hc_decoded_bytes != expected_hc_bytes) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "decoded input hidden-state size does not match token span");
    }

    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_session *session = dist_worker_get_session_locked(state, session_id, err, sizeof(err));
    if (!session) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    /* docs/archive/mtp.md Phase 1: if the previous frame was a speculative VERIFY batch, roll
     * this worker's layer-slice KV back to base + accept_len before validating
     * the new frame. Output correctness never depends on this: a wrong rollback
     * only trips the prefix-hash check below and forces a transcript rebuild. */
    if (session->spec_pending &&
        (work.flags & DS4_DIST_WORK_F_RESET_SESSION) == 0 &&
        (work.flags & DS4_DIST_WORK_F_VERIFY_CONT) == 0) {
        /* wave 69: a VERIFY_CONT chunk is mid-batch -- rolling back here would undo
         * the prior chunk's KV. Only the first (non-CONT) chunk of a batch rolls
         * back the previous batch's rejected tail. */
        uint32_t keep = session->spec_base_len + work.accept_len;
        if (ds4_session_layer_slice_rollback(session->session, keep, err, sizeof(err)) == 0) {
            const ds4_tokens *tl = ds4_session_tokens(session->session);
            if (tl && tl->len >= 0) {
                session->token_hash = dist_token_hash_prefix(tl->v, (uint32_t)tl->len);
                session->token_hash_valid = true;
            }
        }
        session->spec_pending = false;
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0 &&
        ds4_session_layer_slice_reset(session->session, err, sizeof(err)) != 0) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0) {
        session->spec_pending = false;
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0) {
        session->token_hash = DS4_DIST_TOKEN_HASH_INIT;
        session->token_hash_valid = true;
    } else if (!session->token_hash_valid) {
        const ds4_tokens *timeline = ds4_session_tokens(session->session);
        if (!timeline || timeline->len < 0) {
            pthread_mutex_unlock(&state->mu);
            if (!input_hc_uses_wire) free(input_hc);
            free(result);
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "worker session has no token timeline");
        }
        session->token_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
        session->token_hash_valid = true;
    }
    if (session->token_hash != work_prefix_hash) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "worker KV prefix hash mismatch");
    }
    const double eval_t0 = dist_now_sec();
    int eval_rc;
    if (is_verify) {
        /* docs/archive/mtp.md Phase 1: per-row batch verification. Runs this worker's final
         * layer slice + output head over all K candidates and fills K logit rows.
         * Records spec base so the next frame's accept_len can roll back. */
        eval_rc = ds4_session_verify_batch_argmax(session->session,
                                                  tokens,
                                                  work.n_tokens,
                                                  work.pos0,
                                                  work.layer_start,
                                                  work.layer_end,
                                                  input_hc,
                                                  result,
                                                  err,
                                                  sizeof(err));
    } else {
        eval_rc = ds4_session_eval_layer_slice(session->session,
                                               tokens,
                                               work.n_tokens,
                                               work.pos0,
                                               work.layer_start,
                                               work.layer_end,
                                               input_hc,
                                               produce_hidden ? result : NULL,
                                               local_output_logits,
                                               local_output_logits ? result : NULL,
                                               err,
                                               sizeof(err));
    }
    const double eval_t1 = dist_now_sec();
    if (eval_rc == 0) {
        session->token_hash = work_result_hash;
        session->token_hash_valid = true;
        if (is_verify) {
            session->spec_pending = true;
            /* wave 69: keep the batch-start base recorded by the first chunk; a
             * CONT chunk's pos0 is mid-batch and must not overwrite it. */
            if ((work.flags & DS4_DIST_WORK_F_VERIFY_CONT) == 0) {
                session->spec_base_len = work.pos0;
            }
        }
        /* worker MTP draft 已整族删除(2026-08-05)。 */
    } else {
        session->token_hash_valid = false;
    }
    pthread_mutex_unlock(&state->mu);
    DIST_DEBUG("worker eval request=%llu layers=%u:%u tokens=%u pos=%u has_next=%d output=%d rc=%d",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               has_next ? 1 : 0,
               local_output_logits ? 1 : 0,
               eval_rc);

    if (eval_rc != 0) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }

    uint32_t result_wire_bytes = result_bytes;
    if (result_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        !dist_activation_wire_bytes_from_f32_bytes(input_hc_bits,
                                                   result_bytes,
                                                   &result_wire_bytes)) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid output hidden-state size");
    }

    ds4_dist_telemetry_fixed telemetry = {
        .layer_start = work.layer_start,
        .layer_end = work.layer_end,
        .route_index = work.route_index,
        .pos0 = work.pos0,
        .n_tokens = work.n_tokens,
        .eval_usec = dist_usec_since(eval_t0, eval_t1),
        .downstream_wait_usec = 0,
        .forward_send_usec = 0,
        .input_bytes = work.token_bytes + work.input_hc_bytes,
        .output_bytes = result_wire_bytes,
    };

    int send_rc;
    if (has_next) {
        send_rc = dist_forward_work_to_next(upstream,
                                            &next_route,
                                            &work,
                                            tokens,
                                            result,
                                            result_bytes,
                                            &telemetry,
                                            route_blob);
    } else {
        send_rc = dist_worker_upstream_send_work_result(upstream,
                                                        request_id,
                                                        work_result_hash,
                                                        0,
                                                        result_kind,
                                                        result_kind == DS4_DIST_RESULT_HIDDEN_STATE ? input_hc_bits : 32u,
                                                        &telemetry,
                                                        1,
                                                        result,
                                                        result_bytes,
                                                        draft_n ? draft_wire : NULL,
                                                        draft_n);
    }
    DIST_DEBUG("worker send complete request=%llu has_next=%d send_rc=%d",
               (unsigned long long)request_id,
               has_next ? 1 : 0,
               send_rc);
    if (!input_hc_uses_wire) free(input_hc);
    free(result);
    free(route_blob);
    free(tokens);
    return send_rc;
}
