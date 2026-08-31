/* dist_coord_eval.c — 机械拆自 ds4_distributed.c: coordinator 工作分发(Coordinator Work Dispatch: eval_remote/eval_span)。行为零变化。 */
#include "dist_internal.h"


static int dist_coordinator_eval_remote_on_fd(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        int fd,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t expected_result_hash,
        bool reset_session,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        float *logits,
        ds4_dist_spec_io *spec,
        char *err,
        size_t errlen) {
    const bool verify = spec && (spec->extra_flags & DS4_DIST_WORK_F_VERIFY) != 0;
    int rc = dist_coordinator_send_remote_work_on_fd(state,
                                                     plan,
                                                     fd,
                                                     tokens,
                                                     n_tokens,
                                                     pos0,
                                                     session_id,
                                                     request_id,
                                                     prefix_hash,
                                                     expected_result_hash,
                                                     reset_session,
                                                     false,
                                                     hidden_hc,
                                                     hidden_hc_bytes,
                                                     spec ? spec->draft_cap : 0,
                                                     spec ? spec->accept_len : 0,
                                                     spec ? spec->extra_flags : 0,
                                                     err,
                                                     errlen);
    /* Wave 68+ spec-pipe: the WORK is now in flight; run the speculative
     * next-cycle local compute here, overlapping the worker's compute of THIS
     * batch (the measured ~50% coordinator-idle t_remote_blocked window).
     * Default NULL => no-op (byte-identical).  The cb's own errors are swallowed:
     * a failed precompute just means the next cycle recomputes (no correctness
     * impact -- the verify batch below still gates every token). */
    if (rc == 0 && state->spec_overlap_cb) {
        (void)state->spec_overlap_cb(state->spec_overlap_ctx);
    }
    uint32_t kind = 0, payload_bytes = 0;
    uint64_t result_hash = 0;
    void *payload = NULL;
    if (rc == 0) {
        rc = dist_recv_result_alloc(fd,
                                    state,
                                    request_id,
                                    &kind,
                                    &result_hash,
                                    &payload,
                                    &payload_bytes,
                                    spec ? spec->drafts : NULL,
                                    spec ? &spec->draft_n : NULL,
                                    err,
                                    errlen);
    }
    if (rc != 0) return rc;
    if (result_hash != expected_result_hash) {
        free(payload);
        if (errlen) snprintf(err, errlen, "distributed result prefix hash mismatch");
        return 1;
    }

    const uint32_t logits_bytes = (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (verify) {
        const uint32_t want = (uint32_t)((uint64_t)n_tokens * logits_bytes);
        if (!spec->verify_logits) {
            free(payload);
            if (errlen) snprintf(err, errlen, "distributed verify has no logit sink");
            return 1;
        }
        if (kind == DS4_DIST_RESULT_LOGITS && payload_bytes == want) {
            memcpy(spec->verify_logits, payload, want);
            free(payload);
            return 0;
        }
        /* 本机 MTP: the worker holds no output head and returns one hidden-state
         * row per verify position; run the local output head on each row to fill
         * the N verify-logit rows the coordinator argmaxes for the accept prefix. */
        const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
        const uint32_t hidden_want =
            (uint32_t)((uint64_t)n_tokens * hc_values * sizeof(float));
        if (kind == DS4_DIST_RESULT_HIDDEN_STATE && payload_bytes == hidden_want) {
            const float *rows = (const float *)payload;
            const uint32_t vocab = (uint32_t)ds4_engine_vocab_size(state->engine);
            for (uint32_t i = 0; i < n_tokens; i++) {
                int hrc = ds4_session_eval_output_head_from_hc(
                    session, rows + (uint64_t)i * hc_values, 1,
                    spec->verify_logits + (uint64_t)i * vocab, err, errlen);
                if (hrc != 0) { free(payload); return hrc; }
            }
            /* carry-over draft: keep the raw per-row hidden so the caller can
             * re-seed cur_hc from the boundary row and draft the next cycle. */
            if (spec->hidden_rows) {
                memcpy(spec->hidden_rows, rows, hidden_want);
            }
            free(payload);
            return 0;
        }
        free(payload);
        if (errlen) snprintf(err, errlen,
                             "distributed verify returned %u bytes, want %u logits or %u hidden",
                             payload_bytes, want, hidden_want);
        return 1;
    }
    if (kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes) {
        memcpy(logits, payload, logits_bytes);
        free(payload);
        return 0;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE && payload_bytes == hidden_hc_bytes) {
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           payload,
                                                           n_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(payload);
        return head_rc;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        free(payload);
        if (errlen) snprintf(err, errlen, "distributed route returned invalid hidden-state size");
        return 1;
    }
    free(payload);
    if (errlen) snprintf(err, errlen, "distributed route did not return logits or hidden-state");
    return 1;
}

int dist_coordinator_eval_span(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        bool reset_session,
        float *logits,
        ds4_dist_spec_io *spec,
        char *err,
        size_t errlen) {
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t hidden_bytes64 = (uint64_t)n_tokens * hc_values * sizeof(float);
    if (hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    uint64_t prefix_hash = DS4_DIST_TOKEN_HASH_INIT;
    if (reset_session) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "distributed reset span must start at position 0");
            return 1;
        }
    } else if (dist_session_token_hash_prefix(session,
                                              pos0,
                                              &prefix_hash,
                                              err,
                                              errlen) != 0) {
        return 1;
    }
    const uint64_t result_hash = dist_token_hash_update_span(prefix_hash, tokens, n_tokens);
    const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
    float *hidden = NULL;
    if (plan->count != 0) {
        hidden = malloc(hidden_bytes);
        if (!hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating coordinator hidden-state");
            return 1;
        }
    }
    if (reset_session &&
        ds4_session_layer_slice_reset(session, err, errlen) != 0) {
        free(hidden);
        return 1;
    }

    const bool local_logits = plan->count == 0;
    int remote_fd = -1;
    if (plan->count != 0) {
        const ds4_dist_route_entry *first = &plan->entry[0];
        remote_fd = first->fd;
        if (remote_fd < 0) {
            if (errlen) snprintf(err, errlen, "distributed route has no live first-hop connection");
            free(hidden);
            return 1;
        }
    }

    int rc;
    if (state->spec_precomputed_hidden && !local_logits && plan->count != 0) {
        /* Wave 68+ spec-pipe: reuse the hidden computed during the PREVIOUS
         * cycle's overlap window (its layer_slice already advanced this slice's
         * KV by n_tokens), so skip the compute here -- this is the critical-path
         * saving that collapses coord(t_local)+worker(t_remote) toward max(). */
        memcpy(hidden, state->spec_precomputed_hidden, hidden_bytes);
        rc = 0;
    } else {
        rc = ds4_session_eval_layer_slice(session,
                                          tokens,
                                          n_tokens,
                                          pos0,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          local_logits ? NULL : hidden,
                                          local_logits,
                                          local_logits ? logits : NULL,
                                          err,
                                          errlen);
    }
    if (rc == 0 && plan->count != 0) {
        rc = dist_coordinator_eval_remote_on_fd(state,
                                                session,
                                                plan,
                                                remote_fd,
                                                tokens,
                                                n_tokens,
                                                pos0,
                                                session_id,
                                                request_id,
                                                prefix_hash,
                                                result_hash,
                                                reset_session,
                                                hidden,
                                                hidden_bytes,
                                                logits,
                                                spec,
                                                err,
                                                errlen);
    }
    free(hidden);
    return rc;
}

