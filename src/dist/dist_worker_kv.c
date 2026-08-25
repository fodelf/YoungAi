/* dist_worker_kv.c — 机械拆自 ds4_distributed.c: worker KV 会话与快照处理(Worker KV Sessions And Snapshot Handlers)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Worker KV Sessions And Snapshot Handlers
 * ========================================================================= */

ds4_dist_worker_session *dist_worker_get_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id,
        char *err,
        size_t errlen) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }

    ds4_dist_worker_session *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return NULL;
    }
    if (ds4_session_create(&entry->session, state->engine, state->ctx_size) != 0) {
        free(entry);
        if (errlen) snprintf(err, errlen, "failed to create distributed worker session");
        return NULL;
    }
    entry->session_id = session_id;
    entry->next = state->sessions;
    state->sessions = entry;
    return entry;
}

static ds4_dist_worker_session *dist_worker_find_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }
    return NULL;
}

uint32_t dist_worker_clear_sessions(ds4_dist_worker_state *state) {
    uint32_t n = 0;
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_session *it = state->sessions;
    state->sessions = NULL;
    pthread_mutex_unlock(&state->mu);

    while (it) {
        ds4_dist_worker_session *next = it->next;
        ds4_session_free(it->session);
        free(it);
        it = next;
        n++;
    }
    return n;
}

int dist_mem_read(ds4_dist_mem_reader *r, void *dst, uint32_t len) {
    if (!r || len > r->remaining) return -1;
    if (len != 0) memcpy(dst, r->p, len);
    r->p += len;
    r->remaining -= len;
    return 1;
}

static int dist_temp_file(const char *prefix, char *path, size_t path_len, FILE **fp_out) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s.XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) return -1;
    FILE *fp = fdopen(fd, "w+b");
    if (!fp) {
        close(fd);
        unlink(tmpl);
        return -1;
    }
    snprintf(path, path_len, "%s", tmpl);
    *fp_out = fp;
    return 0;
}

int dist_worker_handle_snapshot_save(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_snapshot_req_fixed req;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    if (bytes != sizeof(req)) {
        dist_discard_bytes(upstream->fd, bytes);
        pthread_mutex_lock(&upstream->write_mu);
        int rc = dist_send_snapshot_error(upstream->fd, 0, 0, state->model_id,
                                          state->layer_start, state->layer_end,
                                          "invalid distributed snapshot save request");
        pthread_mutex_unlock(&upstream->write_mu);
        return rc;
    }
    int rc = dist_read_full(upstream->fd, &req, sizeof(req));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_snapshot_req_from_wire(&req);
    request_id = dist_u64_from_halves(req.request_hi, req.request_lo);
    session_id = dist_u64_from_halves(req.session_hi, req.session_lo);
    const uint64_t token_hash = dist_u64_from_halves(req.token_hash_hi, req.token_hash_lo);

    char err[256] = {0};
    FILE *tmp = NULL;
    char tmp_path[PATH_MAX];
    uint64_t payload_bytes = 0;

    if (req.model_id != state->model_id ||
        req.layer_start != state->layer_start ||
        req.layer_end != state->layer_end ||
        req.token_count > (uint32_t)state->ctx_size) {
        snprintf(err, sizeof(err), "snapshot save request does not match worker state");
    } else {
        pthread_mutex_lock(&state->mu);
        ds4_dist_worker_session *session = dist_worker_find_session_locked(state, session_id);
        if (!session) {
            snprintf(err, sizeof(err), "worker has no distributed session to snapshot");
        } else {
            const ds4_tokens *timeline = ds4_session_tokens(session->session);
            uint64_t live_hash = 0;
            if (!timeline || timeline->len < 0 || (uint32_t)timeline->len != req.token_count) {
                snprintf(err, sizeof(err), "worker snapshot token count mismatch");
            } else {
                live_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
                if (live_hash != token_hash) {
                    snprintf(err, sizeof(err), "worker snapshot token hash mismatch");
                }
            }
            if (!err[0] && dist_temp_file("ds4-dist-save", tmp_path, sizeof(tmp_path), &tmp) != 0) {
                snprintf(err, sizeof(err), "failed to create worker snapshot temp file");
            }
            if (!err[0] &&
                ds4_session_save_layer_payload(session->session,
                                               tmp,
                                               state->layer_start,
                                               state->layer_end,
                                               err,
                                               sizeof(err)) != 0) {
                if (!err[0]) snprintf(err, sizeof(err), "failed to save worker KV shard");
            }
            if (!err[0] && fflush(tmp) != 0) {
                snprintf(err, sizeof(err), "failed to flush worker KV shard");
            }
            if (!err[0]) {
                off_t pos = ftello(tmp);
                if (pos < 0) snprintf(err, sizeof(err), "failed to measure worker KV shard");
                else payload_bytes = (uint64_t)pos;
            }
            if (!err[0] && fseeko(tmp, 0, SEEK_SET) != 0) {
                snprintf(err, sizeof(err), "failed to rewind worker KV shard");
            }
        }
        pthread_mutex_unlock(&state->mu);
    }

    pthread_mutex_lock(&upstream->write_mu);
    if (err[0]) {
        rc = dist_send_snapshot_error(upstream->fd,
                                      request_id,
                                      session_id,
                                      state->model_id,
                                      state->layer_start,
                                      state->layer_end,
                                      err);
    } else {
        ds4_dist_snapshot_begin_fixed begin;
        memset(&begin, 0, sizeof(begin));
        begin.model_id = state->model_id;
        dist_u64_to_halves(session_id, &begin.session_hi, &begin.session_lo);
        dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
        dist_u64_to_halves(token_hash, &begin.token_hash_hi, &begin.token_hash_lo);
        begin.layer_start = state->layer_start;
        begin.layer_end = state->layer_end;
        dist_u64_to_halves(payload_bytes, &begin.payload_hi, &begin.payload_lo);
        rc = dist_send_snapshot_begin(upstream->fd, &begin, NULL, NULL);
        if (rc > 0) rc = dist_send_snapshot_file_chunks(upstream->fd, request_id, tmp, payload_bytes);
        if (rc > 0) rc = dist_send_snapshot_done(upstream->fd, request_id, 0, NULL);
    }
    pthread_mutex_unlock(&upstream->write_mu);

    if (tmp) fclose(tmp);
    if (tmp) unlink(tmp_path);
    return rc;
}

int dist_worker_handle_snapshot_load(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_snapshot_begin_fixed begin;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    char err[256] = {0};
    if (bytes < sizeof(begin)) {
        dist_discard_bytes(upstream->fd, bytes);
        return -1;
    }
    int rc = dist_read_full(upstream->fd, &begin, sizeof(begin));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_snapshot_begin_from_wire(&begin);
    request_id = dist_u64_from_halves(begin.request_hi, begin.request_lo);
    session_id = dist_u64_from_halves(begin.session_hi, begin.session_lo);
    const uint64_t token_hash = dist_u64_from_halves(begin.token_hash_hi,
                                                     begin.token_hash_lo);
    const uint64_t payload_bytes = dist_u64_from_halves(begin.payload_hi,
                                                        begin.payload_lo);
    const uint32_t body_bytes = bytes - (uint32_t)sizeof(begin);
    uint64_t expected_token_bytes = (uint64_t)begin.token_count * sizeof(uint32_t);
    if (expected_token_bytes > UINT32_MAX ||
        begin.token_bytes != (uint32_t)expected_token_bytes ||
        begin.message_bytes != 0 ||
        body_bytes != begin.token_bytes) {
        dist_discard_bytes(upstream->fd, body_bytes);
        snprintf(err, sizeof(err), "invalid distributed snapshot load header");
    }

    int *tokens = NULL;
    if (!err[0]) {
        tokens = malloc((size_t)begin.token_count * sizeof(tokens[0]));
        if (!tokens && begin.token_count != 0) {
            dist_discard_bytes(upstream->fd, begin.token_bytes);
            snprintf(err, sizeof(err), "out of memory reading snapshot tokens");
        }
    }
    for (uint32_t i = 0; !err[0] && i < begin.token_count; i++) {
        uint32_t wire_token = 0;
        rc = dist_read_full(upstream->fd, &wire_token, sizeof(wire_token));
        if (rc <= 0) {
            free(tokens);
            return rc == 0 ? 0 : -1;
        }
        uint32_t token = ntohl(wire_token);
        if (token > (uint32_t)INT_MAX ||
            token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            snprintf(err, sizeof(err), "snapshot token id is outside the model vocabulary");
            tokens[i] = 0;
        } else {
            tokens[i] = (int)token;
        }
    }
    if (!err[0] &&
        dist_token_hash_prefix(tokens, begin.token_count) != token_hash) {
        snprintf(err, sizeof(err), "snapshot load token hash mismatch");
    }
    if (!err[0] &&
        (begin.model_id != state->model_id ||
         begin.layer_start != state->layer_start ||
         begin.layer_end != state->layer_end ||
         begin.token_count > (uint32_t)state->ctx_size)) {
        snprintf(err, sizeof(err), "snapshot load request does not match worker state");
    }

    FILE *tmp = NULL;
    char tmp_path[PATH_MAX];
    if (!err[0] && dist_temp_file("ds4-dist-load", tmp_path, sizeof(tmp_path), &tmp) != 0) {
        snprintf(err, sizeof(err), "failed to create worker snapshot restore temp file");
    }

    uint8_t *buf = NULL;
    if (!err[0]) {
        buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
        if (!buf) snprintf(err, sizeof(err), "out of memory restoring worker KV shard");
    }
    uint64_t received = 0;
    while (!err[0] && received < payload_bytes) {
        uint32_t type = 0, chunk_frame_bytes = 0;
        rc = dist_read_frame_header(upstream->fd, &type, &chunk_frame_bytes, err, sizeof(err));
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        if (type != DS4_DIST_MSG_SNAPSHOT_CHUNK ||
            chunk_frame_bytes < sizeof(ds4_dist_snapshot_chunk_fixed)) {
            dist_discard_bytes(upstream->fd, chunk_frame_bytes);
            snprintf(err, sizeof(err), "expected distributed snapshot chunk");
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        rc = dist_read_full(upstream->fd, &chunk, sizeof(chunk));
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        dist_snapshot_chunk_from_wire(&chunk);
        uint64_t got_request = dist_u64_from_halves(chunk.request_hi, chunk.request_lo);
        uint32_t chunk_bytes = chunk_frame_bytes - (uint32_t)sizeof(chunk);
        if (got_request != request_id ||
            chunk.chunk_bytes != chunk_bytes ||
            chunk_bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ||
            chunk_bytes > payload_bytes - received) {
            dist_discard_bytes(upstream->fd, chunk_bytes);
            snprintf(err, sizeof(err), "invalid distributed snapshot chunk");
            break;
        }
        rc = dist_read_full(upstream->fd, buf, chunk_bytes);
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        if (fwrite(buf, 1, chunk_bytes, tmp) != chunk_bytes) {
            snprintf(err, sizeof(err), "failed to write worker KV shard temp file");
            break;
        }
        received += chunk_bytes;
    }
    free(buf);

    if (!err[0] && fflush(tmp) != 0) {
        snprintf(err, sizeof(err), "failed to flush worker KV shard restore file");
    }
    if (!err[0] && fseeko(tmp, 0, SEEK_SET) != 0) {
        snprintf(err, sizeof(err), "failed to rewind worker KV shard restore file");
    }
    if (!err[0]) {
        pthread_mutex_lock(&state->mu);
        ds4_dist_worker_session *session =
            dist_worker_get_session_locked(state, session_id, err, sizeof(err));
        if (session &&
            ds4_session_load_layer_payload(session->session,
                                           tmp,
                                           payload_bytes,
                                           tokens,
                                           begin.token_count,
                                           state->layer_start,
                                           state->layer_end,
                                           err,
                                           sizeof(err)) == 0) {
            session->token_hash = token_hash;
            session->token_hash_valid = true;
        } else {
            if (!err[0]) snprintf(err, sizeof(err), "failed to restore worker KV shard");
            if (session) session->token_hash_valid = false;
        }
        pthread_mutex_unlock(&state->mu);
    }

    if (tmp) fclose(tmp);
    if (tmp) unlink(tmp_path);
    free(tokens);

    pthread_mutex_lock(&upstream->write_mu);
    rc = dist_send_snapshot_done(upstream->fd, request_id, err[0] ? 1u : 0u,
                                 err[0] ? err : NULL);
    pthread_mutex_unlock(&upstream->write_mu);
    if (err[0] && received < payload_bytes) return -1;
    return rc;
}

