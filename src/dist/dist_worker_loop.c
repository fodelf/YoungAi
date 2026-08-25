/* dist_worker_loop.c — 机械拆自 ds4_distributed.c: worker 控制循环与结果帧(Worker Control Loop And Result Frames)。行为零变化。 */
#include "dist_internal.h"

/* 文件内前置声明(原文件同名声明原样搬入): 定义在调用者之后。 */
static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);

/* =========================================================================
 * Worker Control Loop And Result Frames
 * ========================================================================= */

int dist_worker_read_loop(ds4_dist_worker_state *state, int fd) {
    ds4_dist_worker_upstream upstream;
    dist_worker_upstream_init(&upstream, state, fd);
    int loop_rc = 0;

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
            rc = dist_worker_handle_work(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
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

    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

int dist_send_work_result(
        int fd,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes,
        const uint32_t *draft_tokens,
        uint32_t draft_count) {
    if (payload_bytes != 0 && !payload) return -1;
    if (telemetry_count != 0 && !telemetry) return -1;
    if (draft_count != 0 && !draft_tokens) return -1;
    uint32_t wire_payload_bytes = payload_bytes;
    uint64_t hidden_values = 0;
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        payload_bits = dist_activation_bits_or_default(payload_bits);
        if (!dist_activation_bits_valid(payload_bits) ||
            (payload_bytes % (uint32_t)sizeof(float)) != 0)
            return -1;
        hidden_values = payload_bytes / (uint32_t)sizeof(float);
        if (!dist_activation_wire_bytes(payload_bits, hidden_values, &wire_payload_bytes))
            return -1;
    } else if (status == 0 && result_kind == DS4_DIST_RESULT_LOGITS) {
        payload_bits = 32u;
    } else {
        payload_bits = 0;
    }
    /* docs/archive/mtp.md Phase 1: draft tokens (uint32 each) ride after the logits payload,
     * present only when the WORK frame requested a draft and status==0. */
    if (status != 0) draft_count = 0;
    const uint64_t draft_bytes64 = (uint64_t)draft_count * sizeof(uint32_t);
    const uint64_t telemetry_bytes64 =
        (uint64_t)telemetry_count * sizeof(ds4_dist_telemetry_fixed);
    if (telemetry_bytes64 > UINT32_MAX) return -1;
    const uint32_t telemetry_bytes = (uint32_t)telemetry_bytes64;
    const uint64_t frame_bytes = sizeof(ds4_dist_result_fixed) +
                                 telemetry_bytes64 +
                                 (uint64_t)wire_payload_bytes +
                                 draft_bytes64;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_result_fixed r;
    dist_u64_to_halves(request_id, &r.request_hi, &r.request_lo);
    dist_u64_to_halves(status == 0 ? result_hash : 0,
                       &r.result_hash_hi,
                       &r.result_hash_lo);
    r.status = status;
    r.result_kind = result_kind;
    r.telemetry_count = telemetry_count;
    r.telemetry_bytes = telemetry_bytes;
    r.payload_bytes = wire_payload_bytes;
    r.payload_bits = payload_bits;
    r.draft_count = draft_count;

    ds4_dist_result_fixed wire = r;
    dist_result_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_RESULT, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < telemetry_count; i++) {
        ds4_dist_telemetry_fixed tw = telemetry[i];
        dist_telemetry_to_wire(&tw);
        if (dist_write_full(fd, &tw, sizeof(tw)) != 0) return -1;
    }
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE && wire_payload_bytes != 0) {
        if (dist_write_activation_payload(fd, payload, hidden_values, payload_bits) != 0) return -1;
    } else if (payload_bytes && payload && dist_write_full(fd, payload, payload_bytes) != 0) {
        return -1;
    }
    for (uint32_t i = 0; i < draft_count; i++) {
        uint32_t t = htonl(draft_tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return 1;
}

int dist_send_work_error(int fd, uint64_t request_id, const char *msg) {
    if (!msg) msg = "distributed work failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    return dist_send_work_result(fd, request_id, 0, 1, 0, 0, NULL, 0, msg, (uint32_t)len, NULL, 0);
}

int dist_send_snapshot_begin(
        int fd,
        const ds4_dist_snapshot_begin_fixed *begin,
        const int *tokens,
        const char *msg) {
    uint64_t token_bytes64 = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (token_bytes64 > UINT32_MAX || begin->token_bytes != (uint32_t)token_bytes64) return -1;
    uint32_t msg_len = msg ? (uint32_t)strlen(msg) : 0u;
    if (msg_len != begin->message_bytes) return -1;
    uint64_t frame_bytes64 = sizeof(*begin) + token_bytes64 + msg_len;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_begin_fixed wire = *begin;
    dist_snapshot_begin_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_BEGIN, (uint32_t)frame_bytes64) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < begin->token_count; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    if (msg_len && dist_write_full(fd, msg, msg_len) != 0) return -1;
    return 1;
}

int dist_send_snapshot_error(
        int fd,
        uint64_t request_id,
        uint64_t session_id,
        uint32_t model_id,
        uint32_t layer_start,
        uint32_t layer_end,
        const char *msg) {
    if (!msg) msg = "distributed snapshot failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    ds4_dist_snapshot_begin_fixed begin;
    memset(&begin, 0, sizeof(begin));
    begin.model_id = model_id;
    dist_u64_to_halves(session_id, &begin.session_hi, &begin.session_lo);
    dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
    begin.layer_start = layer_start;
    begin.layer_end = layer_end;
    begin.status = 1;
    begin.message_bytes = (uint32_t)len;
    return dist_send_snapshot_begin(fd, &begin, NULL, msg);
}

int dist_send_snapshot_done(int fd, uint64_t request_id, uint32_t status, const char *msg) {
    if (!msg) msg = "";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    ds4_dist_snapshot_done_fixed done;
    memset(&done, 0, sizeof(done));
    dist_u64_to_halves(request_id, &done.request_hi, &done.request_lo);
    done.status = status;
    done.message_bytes = (uint32_t)len;
    uint64_t frame_bytes64 = sizeof(done) + len;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_done_fixed wire = done;
    dist_snapshot_done_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_DONE, (uint32_t)frame_bytes64) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (len && dist_write_full(fd, msg, len) != 0) return -1;
    return 1;
}

int dist_send_snapshot_file_chunks(int fd, uint64_t request_id, FILE *fp, uint64_t bytes) {
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) return -1;
    int rc = 1;
    while (bytes != 0) {
        const uint32_t n = bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ?
            DS4_DIST_SNAPSHOT_CHUNK_BYTES : (uint32_t)bytes;
        if (fread(buf, 1, n, fp) != n) {
            rc = -1;
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        dist_u64_to_halves(request_id, &chunk.request_hi, &chunk.request_lo);
        chunk.chunk_bytes = n;
        ds4_dist_snapshot_chunk_fixed wire = chunk;
        dist_snapshot_chunk_to_wire(&wire);
        const uint32_t frame_bytes = (uint32_t)sizeof(wire) + n;
        if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_CHUNK, frame_bytes) != 0 ||
            dist_write_full(fd, &wire, sizeof(wire)) != 0 ||
            dist_write_full(fd, buf, n) != 0) {
            rc = -1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}


static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    void *payload = malloc(bytes);
    if (!payload) {
        dist_discard_bytes(upstream->fd, bytes);
        return dist_worker_upstream_send_work_error(upstream, 0, "out of memory reading distributed WORK frame");
    }
    int rc = dist_read_full(upstream->fd, payload, bytes);
    if (rc <= 0) {
        free(payload);
        return rc == 0 ? 0 : -1;
    }
    rc = dist_worker_process_work_payload(state, upstream, payload, bytes);
    free(payload);
    return rc;
}

