/* dist_coord_dispatch.c — 机械拆自 ds4_distributed.c: coordinator 工作分发(Coordinator Work Dispatch: recv/send)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Coordinator Work Dispatch
 * ========================================================================= */

int dist_recv_result_alloc(
        int fd,
        const ds4_dist_coordinator_state *state,
        uint64_t request_id,
        uint32_t *kind,
        uint64_t *result_hash,
        void **payload,
        uint32_t *payload_bytes,
        uint32_t *draft_out,      /* optional caller buffer of >=16 ids */
        uint32_t *draft_n_out,    /* optional out: number of draft ids read */
        char *err,
        size_t errlen) {
    *payload = NULL;
    *payload_bytes = 0;
    *kind = 0;
    if (result_hash) *result_hash = 0;
    if (draft_n_out) *draft_n_out = 0;

    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid frame");
        return 1;
    }

    ds4_dist_result_fixed result;
    rc = dist_read_full(fd, &result, sizeof(result));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed result");
        return 1;
    }
    dist_result_from_wire(&result);
    const uint64_t got_request = dist_u64_from_halves(result.request_hi, result.request_lo);
    const uint64_t got_hash = dist_u64_from_halves(result.result_hash_hi,
                                                  result.result_hash_lo);
    const uint32_t body_bytes = bytes - (uint32_t)sizeof(result);
    /* docs/archive/mtp.md Phase 1: draft token ids ride after telemetry + payload. */
    if (result.draft_count > 16u) {
        dist_discard_bytes(fd, body_bytes);
        if (errlen) snprintf(err, errlen, "distributed result draft count out of range");
        return 1;
    }
    const uint32_t draft_bytes = result.draft_count * (uint32_t)sizeof(uint32_t);
    if (result.telemetry_bytes % (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
        result.telemetry_count != result.telemetry_bytes / (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
        result.telemetry_bytes > body_bytes ||
        draft_bytes > body_bytes - result.telemetry_bytes ||
        result.payload_bytes != body_bytes - result.telemetry_bytes - draft_bytes) {
        dist_discard_bytes(fd, body_bytes);
        if (errlen) snprintf(err, errlen, "distributed result telemetry metadata mismatch");
        return 1;
    }
    if (got_request != request_id) {
        dist_discard_bytes(fd, bytes - (uint32_t)sizeof(result));
        if (errlen) snprintf(err, errlen, "distributed result metadata mismatch");
        return 1;
    }

    if (result.telemetry_bytes != 0) {
        if (dist_coordinator_debug_enabled(state)) {
            ds4_dist_telemetry_fixed *telemetry = malloc(result.telemetry_bytes);
            if (!telemetry) {
                dist_discard_bytes(fd, result.telemetry_bytes);
                if (errlen) snprintf(err, errlen, "out of memory reading distributed telemetry");
                return 1;
            }
            rc = dist_read_full(fd, telemetry, result.telemetry_bytes);
            if (rc <= 0) {
                free(telemetry);
                if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
                return 1;
            }
            for (uint32_t i = 0; i < result.telemetry_count; i++) {
                dist_telemetry_from_wire(&telemetry[i]);
                DIST_COORD_DEBUG(state,
                                 "ds4: distributed telemetry: request=%llu hop=%u layers=%u:%u route=%u pos=%u tokens=%u eval=%.3fms downstream_wait=%.3fms forward_send=%.3fms input=%.2fMiB output=%.2fMiB\n",
                                 (unsigned long long)got_request,
                                 i,
                                 telemetry[i].layer_start,
                                 telemetry[i].layer_end,
                                 telemetry[i].route_index,
                                 telemetry[i].pos0,
                                 telemetry[i].n_tokens,
                                 (double)telemetry[i].eval_usec / 1000.0,
                                 (double)telemetry[i].downstream_wait_usec / 1000.0,
                                 (double)telemetry[i].forward_send_usec / 1000.0,
                                 (double)telemetry[i].input_bytes / (1024.0 * 1024.0),
                                 (double)telemetry[i].output_bytes / (1024.0 * 1024.0));
            }
            free(telemetry);
        } else if (dist_discard_bytes(fd, result.telemetry_bytes) <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
            return 1;
        }
    }

    void *buf = NULL;
    if (result.payload_bytes != 0) {
        buf = malloc(result.payload_bytes);
        if (!buf) {
            dist_discard_bytes(fd, result.payload_bytes);
            if (errlen) snprintf(err, errlen, "out of memory reading distributed result");
            return 1;
        }
        rc = dist_read_full(fd, buf, result.payload_bytes);
        if (rc <= 0) {
            free(buf);
            if (errlen) snprintf(err, errlen, "failed to read distributed result payload");
            return 1;
        }
    }

    if (result.status != 0) {
        if (errlen) {
            if (buf && result.payload_bytes) {
                size_t n = result.payload_bytes < errlen - 1 ? result.payload_bytes : errlen - 1;
                memcpy(err, buf, n);
                err[n] = '\0';
            } else {
                snprintf(err, errlen, "distributed worker returned an error");
            }
        }
        free(buf);
        return DS4_DIST_RECV_REMOTE_ERROR;
    }

    if (result.result_kind == DS4_DIST_RESULT_HIDDEN_STATE && result.payload_bytes != 0) {
        float *decoded = NULL;
        uint32_t decoded_bytes = 0;
        bool uses_wire = false;
        if (dist_decode_activation_payload(buf,
                                           result.payload_bits,
                                           result.payload_bytes,
                                           &decoded,
                                           &decoded_bytes,
                                           &uses_wire,
                                           err,
                                           errlen) != 0) {
            free(buf);
            return 1;
        }
        if (!uses_wire) {
            free(buf);
            buf = decoded;
        }
        result.payload_bytes = decoded_bytes;
    }

    /* docs/archive/mtp.md Phase 1: read the trailing MTP draft token ids (status==0 only;
     * the sender forces draft_count=0 on error frames). */
    for (uint32_t i = 0; i < result.draft_count; i++) {
        uint32_t t = 0;
        if (dist_read_full(fd, &t, sizeof(t)) <= 0) {
            free(buf);
            if (errlen) snprintf(err, errlen, "failed to read distributed draft tokens");
            return 1;
        }
        if (draft_out && i < 16u) draft_out[i] = ntohl(t);
    }
    if (draft_n_out) *draft_n_out = result.draft_count;

    *kind = result.result_kind;
    if (result_hash) *result_hash = got_hash;
    *payload = buf;
    *payload_bytes = result.payload_bytes;
    return 0;
}

int dist_coordinator_send_remote_work_on_fd(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        int fd,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t result_hash,
        bool reset_session,
        bool ack_only,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        uint32_t draft_cap,       /* docs/archive/mtp.md Phase 1: ask last-layer worker to draft */
        uint32_t accept_len,      /* docs/archive/mtp.md Phase 1: roll back prev spec batch first */
        uint32_t extra_flags,     /* extra DS4_DIST_WORK_F_* bits (e.g. DRAFT/VERIFY) */
        char *err,
        size_t errlen) {
    if (plan->count == 0) {
        if (errlen) snprintf(err, errlen, "distributed route has no remote worker");
        return 1;
    }
    const ds4_dist_route_entry *first = &plan->entry[0];

    ds4_dist_work_fixed work;
    memset(&work, 0, sizeof(work));
    work.model_id = state->model_id;
    dist_u64_to_halves(session_id, &work.session_hi, &work.session_lo);
    dist_u64_to_halves(request_id, &work.request_hi, &work.request_lo);
    dist_u64_to_halves(prefix_hash, &work.prefix_hash_hi, &work.prefix_hash_lo);
    dist_u64_to_halves(result_hash, &work.result_hash_hi, &work.result_hash_lo);
    work.pos0 = pos0;
    work.n_tokens = n_tokens;
    work.layer_start = first->layer_start;
    work.layer_end = first->layer_end;
    work.flags = DS4_DIST_WORK_F_INPUT_HC | (extra_flags & DS4_DIST_WORK_F_VALID_MASK);
    if (reset_session) work.flags |= DS4_DIST_WORK_F_RESET_SESSION;
    if (ack_only) work.flags |= DS4_DIST_WORK_F_ACK_ONLY;
    if ((first->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        work.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }
    work.draft_cap = draft_cap;
    work.accept_len = accept_len;
    uint32_t wire_hidden_hc_bytes = 0;
    if (!dist_activation_wire_bytes_from_f32_bytes(state->activation_bits,
                                                   hidden_hc_bytes,
                                                   &wire_hidden_hc_bytes)) {
        if (errlen) snprintf(err, errlen, "invalid distributed hidden-state size");
        return 1;
    }
    work.token_bytes = n_tokens * sizeof(uint32_t);
    work.input_hc_bytes = wire_hidden_hc_bytes;
    work.input_hc_bits = state->activation_bits;
    work.route_count = plan->count;
    work.route_index = 0;
    work.route_bytes = plan->blob_bytes;

    if (dist_send_work_frame(fd, &work, tokens, hidden_hc, plan->blob) != 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed work");
        return 1;
    }
    return 0;
}

