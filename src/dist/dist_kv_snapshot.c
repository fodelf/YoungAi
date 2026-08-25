/* dist_kv_snapshot.c — 机械拆自 ds4_distributed.c: 分布式 KV 快照传输(Distributed KV Snapshot Transport)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Distributed KV Snapshot Transport
 * ========================================================================= */

int dist_write_snapshot_load_begin(
        int fd,
        const ds4_dist_snapshot_begin_fixed *begin,
        const int *tokens) {
    uint64_t token_bytes64 = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (token_bytes64 > UINT32_MAX ||
        begin->token_bytes != (uint32_t)token_bytes64 ||
        begin->message_bytes != 0)
        return -1;
    uint64_t frame_bytes64 = sizeof(*begin) + token_bytes64;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_begin_fixed wire = *begin;
    dist_snapshot_begin_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN,
                                (uint32_t)frame_bytes64) != 0)
        return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < begin->token_count; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return 1;
}

int dist_read_snapshot_begin_frame(
        int fd,
        ds4_dist_snapshot_begin_fixed *begin,
        char *msg,
        size_t msg_cap,
        char *err,
        size_t errlen) {
    if (msg_cap) msg[0] = '\0';
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed snapshot connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_SNAPSHOT_BEGIN ||
        bytes < sizeof(ds4_dist_snapshot_begin_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid snapshot frame");
        return 1;
    }
    rc = dist_read_full(fd, begin, sizeof(*begin));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed snapshot header");
        return 1;
    }
    dist_snapshot_begin_from_wire(begin);
    uint32_t body = bytes - (uint32_t)sizeof(*begin);
    uint64_t expected_token_bytes = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (expected_token_bytes > UINT32_MAX ||
        begin->token_bytes != (uint32_t)expected_token_bytes ||
        begin->token_bytes > body ||
        begin->message_bytes > body - begin->token_bytes) {
        dist_discard_bytes(fd, body);
        if (errlen) snprintf(err, errlen, "invalid distributed snapshot response header");
        return 1;
    }
    if (begin->token_bytes != 0) {
        rc = dist_discard_bytes(fd, begin->token_bytes);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot response tokens");
            return 1;
        }
        body -= begin->token_bytes;
    }
    if (begin->message_bytes != 0) {
        uint32_t n = begin->message_bytes;
        uint32_t copy = msg_cap && n < msg_cap ? n : (msg_cap ? (uint32_t)msg_cap - 1u : 0u);
        if (copy != 0) {
            rc = dist_read_full(fd, msg, copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to read distributed snapshot response message");
                return 1;
            }
            msg[copy] = '\0';
        }
        if (n > copy) {
            rc = dist_discard_bytes(fd, n - copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot response message");
                return 1;
            }
        }
        body -= n;
    }
    if (body != 0) {
        rc = dist_discard_bytes(fd, body);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard trailing distributed snapshot response bytes");
            return 1;
        }
    }
    return 0;
}

int dist_read_snapshot_done_frame(
        int fd,
        uint64_t request_id,
        char *err,
        size_t errlen) {
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed before snapshot completion");
        return 1;
    }
    if (type != DS4_DIST_MSG_SNAPSHOT_DONE ||
        bytes < sizeof(ds4_dist_snapshot_done_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid snapshot completion frame");
        return 1;
    }
    ds4_dist_snapshot_done_fixed done;
    rc = dist_read_full(fd, &done, sizeof(done));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed snapshot completion");
        return 1;
    }
    dist_snapshot_done_from_wire(&done);
    uint32_t body = bytes - (uint32_t)sizeof(done);
    char msg[256];
    msg[0] = '\0';
    if (done.message_bytes > body) {
        dist_discard_bytes(fd, body);
        if (errlen) snprintf(err, errlen, "invalid distributed snapshot completion message");
        return 1;
    }
    if (done.message_bytes != 0) {
        uint32_t copy = done.message_bytes < sizeof(msg) ?
            done.message_bytes : (uint32_t)sizeof(msg) - 1u;
        rc = dist_read_full(fd, msg, copy);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed snapshot completion message");
            return 1;
        }
        msg[copy] = '\0';
        if (done.message_bytes > copy) {
            rc = dist_discard_bytes(fd, done.message_bytes - copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot completion message");
                return 1;
            }
        }
        body -= done.message_bytes;
    }
    if (body != 0) {
        rc = dist_discard_bytes(fd, body);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard trailing distributed snapshot completion bytes");
            return 1;
        }
    }
    uint64_t got_request = dist_u64_from_halves(done.request_hi, done.request_lo);
    if (got_request != request_id) {
        if (errlen) snprintf(err, errlen, "distributed snapshot completion request mismatch");
        return 1;
    }
    if (done.status != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             msg[0] ? msg : "distributed worker failed snapshot request");
        return 1;
    }
    return 0;
}

int dist_receive_snapshot_chunks_to_file(
        int fd,
        uint64_t request_id,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) {
        if (errlen) snprintf(err, errlen, "out of memory receiving distributed KV shard");
        return 1;
    }
    uint64_t received = 0;
    int fail = 0;
    while (!fail && received < payload_bytes) {
        uint32_t type = 0, bytes = 0;
        int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
        if (rc <= 0) {
            if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed while sending KV shard");
            fail = 1;
            break;
        }
        if (type != DS4_DIST_MSG_SNAPSHOT_CHUNK ||
            bytes < sizeof(ds4_dist_snapshot_chunk_fixed)) {
            dist_discard_bytes(fd, bytes);
            if (errlen) snprintf(err, errlen, "expected distributed KV shard chunk");
            fail = 1;
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        rc = dist_read_full(fd, &chunk, sizeof(chunk));
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed KV shard chunk header");
            fail = 1;
            break;
        }
        dist_snapshot_chunk_from_wire(&chunk);
        uint64_t got_request = dist_u64_from_halves(chunk.request_hi,
                                                    chunk.request_lo);
        uint32_t chunk_bytes = bytes - (uint32_t)sizeof(chunk);
        if (got_request != request_id ||
            chunk.chunk_bytes != chunk_bytes ||
            chunk_bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ||
            chunk_bytes > payload_bytes - received) {
            dist_discard_bytes(fd, chunk_bytes);
            if (errlen) snprintf(err, errlen, "invalid distributed KV shard chunk");
            fail = 1;
            break;
        }
        rc = dist_read_full(fd, buf, chunk_bytes);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed KV shard chunk");
            fail = 1;
            break;
        }
        if (fwrite(buf, 1, chunk_bytes, fp) != chunk_bytes) {
            if (errlen) snprintf(err, errlen, "failed to write distributed KV shard");
            fail = 1;
            break;
        }
        received += chunk_bytes;
    }
    free(buf);
    return fail;
}

