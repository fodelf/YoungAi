/* dist_payload.c — 机械拆自 ds4_distributed.c: DSV4 载荷组装与分片(DSV4 Payload Assembly And Sharding)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * DSV4 Payload Assembly And Sharding
 * =========================================================================
 *
 * Save and load operate on the normal single-node DSV4 payload format. This
 * block maps that payload to/from the layer slices advertised by the current
 * distributed route, keeping session files independent from cluster topology.
 */

static uint64_t dist_kv_state_bytes(uint32_t ratio, uint32_t head_dim, bool *ok) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    uint64_t bytes = 0;
    uint64_t tmp = 0;
    if (!dist_u64_mul(coff, head_dim, &tmp) ||
        !dist_u64_mul(tmp, coff, &tmp) ||
        !dist_u64_mul(tmp, ratio, &tmp) ||
        !dist_u64_mul(tmp, sizeof(float), &bytes)) {
        if (ok) *ok = false;
        return 0;
    }
    return bytes;
}

static bool dist_kv_layer_tensor_bytes(
        ds4_engine *engine,
        const ds4_dist_kv_layout *layout,
        uint32_t layer,
        uint32_t n_comp,
        uint32_t n_index_comp,
        uint64_t *out) {
    if (!layout || !out) return false;
    uint64_t bytes = 0;
    uint64_t tmp = 0;
    if (!dist_u64_mul(layout->raw_live, layout->head_dim, &tmp) ||
        !dist_u64_mul(tmp, sizeof(float), &tmp) ||
        !dist_u64_add(&bytes, tmp))
        return false;

    const uint32_t ratio = ds4_engine_layer_compress_ratio(engine, layer);
    if (ratio != 0) {
        if (!dist_u64_mul(n_comp, layout->head_dim, &tmp) ||
            !dist_u64_mul(tmp, sizeof(float), &tmp) ||
            !dist_u64_add(&bytes, tmp))
            return false;
        bool ok = true;
        const uint64_t attn_state = dist_kv_state_bytes(ratio, layout->head_dim, &ok);
        if (!ok ||
            !dist_u64_add(&bytes, attn_state) ||
            !dist_u64_add(&bytes, attn_state))
            return false;
        if (ratio == 4u) {
            if (!dist_u64_mul(n_index_comp, layout->indexer_head_dim, &tmp) ||
                !dist_u64_mul(tmp, sizeof(float), &tmp) ||
                !dist_u64_add(&bytes, tmp))
                return false;
            const uint64_t index_state =
                dist_kv_state_bytes(ratio, layout->indexer_head_dim, &ok);
            if (!ok ||
                !dist_u64_add(&bytes, index_state) ||
                !dist_u64_add(&bytes, index_state))
                return false;
        }
    }

    *out = bytes;
    return true;
}

static bool dist_kv_layout_matches(
        const ds4_dist_kv_layout *a,
        const ds4_dist_kv_layout *b) {
    return a->ctx == b->ctx &&
           a->prefill_cap == b->prefill_cap &&
           a->raw_cap == b->raw_cap &&
           a->raw_window == b->raw_window &&
           a->comp_cap == b->comp_cap &&
           a->token_count == b->token_count &&
           a->n_layers == b->n_layers &&
           a->head_dim == b->head_dim &&
           a->indexer_head_dim == b->indexer_head_dim &&
           a->raw_live == b->raw_live;
}

bool dist_kv_raw_live_valid(const ds4_dist_kv_layout *layout) {
    if (!layout || layout->raw_window == 0 || layout->raw_cap == 0) return false;
    const uint32_t expected =
        layout->token_count < layout->raw_window ? layout->token_count : layout->raw_window;
    return layout->raw_live == expected &&
           layout->raw_live <= layout->raw_cap;
}

int dist_kv_parse_layer_payload(
        ds4_engine *engine,
        FILE *fp,
        uint64_t bytes,
        uint32_t expected_start,
        uint32_t expected_end,
        ds4_dist_kv_layout *layout,
        bool *layout_set,
        uint32_t *n_comp,
        uint32_t *n_index_comp,
        ds4_dist_kv_shard_file *shard,
        char *err,
        size_t errlen) {
    if (dist_rewind_file(fp, "distributed KV shard", err, errlen) != 0)
        return 1;
    uint64_t remaining = bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0)
            return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        if (errlen) snprintf(err, errlen, "unsupported distributed KV layer payload");
        return 1;
    }
    ds4_dist_kv_layout got = {
        .ctx = h[2],
        .prefill_cap = h[3],
        .raw_cap = h[4],
        .raw_window = h[5],
        .comp_cap = h[6],
        .token_count = h[7],
        .n_layers = h[8],
        .head_dim = h[9],
        .indexer_head_dim = h[10],
        .vocab = layout ? layout->vocab : 0,
        .raw_live = h[13],
    };
    const uint32_t layer_start = h[11];
    const uint32_t layer_end = h[12];
    if (layer_start != expected_start ||
        layer_end != expected_end ||
        layer_start > layer_end ||
        layer_end >= got.n_layers) {
        if (errlen) snprintf(err, errlen, "distributed KV shard range mismatch");
        return 1;
    }
    if (got.n_layers != (uint32_t)ds4_engine_layer_count(engine) ||
        !dist_kv_raw_live_valid(&got)) {
        if (errlen) snprintf(err, errlen, "distributed KV shard layout is invalid");
        return 1;
    }
    if (layout_set && *layout_set) {
        if (!dist_kv_layout_matches(layout, &got)) {
            if (errlen) snprintf(err, errlen, "distributed KV shards use different layouts");
            return 1;
        }
    } else if (layout && layout_set) {
        const uint32_t vocab = layout->vocab;
        *layout = got;
        layout->vocab = vocab;
        *layout_set = true;
    }

    const uint32_t slice_layers = layer_end - layer_start + 1u;
    for (uint32_t i = 0; i < slice_layers; i++) {
        const uint32_t il = layer_start + i;
        if (dist_payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0)
            return 1;
        if (n_comp[il] > got.comp_cap) {
            if (errlen) snprintf(err, errlen, "distributed KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < slice_layers; i++) {
        const uint32_t il = layer_start + i;
        if (dist_payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0)
            return 1;
        if (n_index_comp[il] > got.comp_cap) {
            if (errlen) snprintf(err, errlen, "distributed KV shard has invalid indexer row count");
            return 1;
        }
    }

    off_t tensor_pos = ftello(fp);
    if (tensor_pos < 0) {
        if (errlen) snprintf(err, errlen, "failed to locate distributed KV shard tensors");
        return 1;
    }
    uint64_t expected_tensor_bytes = 0;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        uint64_t layer_bytes = 0;
        if (!dist_kv_layer_tensor_bytes(engine, &got, il,
                                        n_comp[il], n_index_comp[il],
                                        &layer_bytes) ||
            !dist_u64_add(&expected_tensor_bytes, layer_bytes)) {
            if (errlen) snprintf(err, errlen, "distributed KV shard tensor size overflow");
            return 1;
        }
    }
    if (remaining != expected_tensor_bytes) {
        if (errlen) snprintf(err, errlen, "distributed KV shard tensor byte count mismatch");
        return 1;
    }
    if (shard) {
        shard->layer_start = layer_start;
        shard->layer_end = layer_end;
        shard->tensor_offset = (uint64_t)tensor_pos;
        shard->tensor_bytes = remaining;
    }
    return 0;
}

int dist_kv_write_session_header(
        FILE *fp,
        const ds4_dist_kv_layout *layout,
        char *err,
        size_t errlen) {
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        layout->ctx,
        layout->prefill_cap,
        layout->raw_cap,
        layout->raw_window,
        layout->comp_cap,
        layout->token_count,
        layout->n_layers,
        layout->head_dim,
        layout->indexer_head_dim,
        layout->vocab,
        layout->raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    return 0;
}

static int dist_kv_write_layer_header(
        FILE *fp,
        const ds4_dist_kv_layout *layout,
        uint32_t layer_start,
        uint32_t layer_end,
        const uint32_t *n_comp,
        const uint32_t *n_index_comp,
        char *err,
        size_t errlen) {
    const uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        layout->ctx,
        layout->prefill_cap,
        layout->raw_cap,
        layout->raw_window,
        layout->comp_cap,
        layout->token_count,
        layout->n_layers,
        layout->head_dim,
        layout->indexer_head_dim,
        layer_start,
        layer_end,
        layout->raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (dist_payload_write_u32(fp, n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (dist_payload_write_u32(fp, n_index_comp[il], err, errlen) != 0) return 1;
    }
    return 0;
}

uint32_t dist_kv_route_shard_count(const ds4_dist_session *d) {
    return d ? 1u + d->plan.count : 0u;
}

void dist_kv_route_shard(
        const ds4_dist_session *d,
        uint32_t shard,
        uint32_t *layer_start,
        uint32_t *layer_end,
        const ds4_dist_route_entry **entry) {
    if (entry) *entry = NULL;
    if (shard == 0) {
        if (layer_start) *layer_start = d->state.local_start;
        if (layer_end) *layer_end = d->state.local_end;
        return;
    }
    const ds4_dist_route_entry *e = &d->plan.entry[shard - 1u];
    if (layer_start) *layer_start = e->layer_start;
    if (layer_end) *layer_end = e->layer_end;
    if (entry) *entry = e;
}

int dist_kv_route_validate(
        const ds4_dist_session *d,
        char *err,
        size_t errlen) {
    if (!d || d->state.n_layers == 0 ||
        d->state.local_start != 0 ||
        d->state.local_end >= d->state.n_layers) {
        if (errlen) snprintf(err, errlen, "distributed KV route does not start at layer 0");
        return 1;
    }
    uint32_t prev = d->state.local_end;
    for (uint32_t i = 0; i < d->plan.count; i++) {
        const ds4_dist_route_entry *e = &d->plan.entry[i];
        if (prev == UINT32_MAX ||
            e->layer_start != prev + 1u ||
            e->layer_end < e->layer_start ||
            e->layer_end >= d->state.n_layers) {
            if (errlen) snprintf(err, errlen, "distributed KV route is not contiguous");
            return 1;
        }
        prev = e->layer_end;
    }
    if (prev + 1u != d->state.n_layers) {
        if (errlen) snprintf(err, errlen, "distributed KV route does not cover all layers");
        return 1;
    }
    return 0;
}

void dist_kv_shards_close(ds4_dist_kv_shard_file *shards, uint32_t count) {
    if (!shards) return;
    for (uint32_t i = 0; i < count; i++) {
        if (shards[i].fp) fclose(shards[i].fp);
    }
}

int dist_save_remote_shard_to_file(
        ds4_dist_session *d,
        const ds4_dist_route_entry *entry,
        const ds4_tokens *tokens,
        uint64_t token_hash,
        FILE *fp,
        uint64_t *payload_bytes_out,
        char *err,
        size_t errlen) {
    if (payload_bytes_out) *payload_bytes_out = 0;
    uint64_t request_id = d->request_id++;
    int fd = dist_connect_endpoint(entry->host, (int)entry->port, err, errlen);
    if (fd < 0) return 1;

    ds4_dist_snapshot_req_fixed req;
    memset(&req, 0, sizeof(req));
    req.model_id = d->state.model_id;
    dist_u64_to_halves(d->session_id, &req.session_hi, &req.session_lo);
    dist_u64_to_halves(request_id, &req.request_hi, &req.request_lo);
    dist_u64_to_halves(token_hash, &req.token_hash_hi, &req.token_hash_lo);
    req.token_count = (uint32_t)tokens->len;
    req.layer_start = entry->layer_start;
    req.layer_end = entry->layer_end;
    ds4_dist_snapshot_req_fixed wire = req;
    dist_snapshot_req_to_wire(&wire);
    int rc = 1;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_SAVE_REQ,
                                (uint32_t)sizeof(wire)) != 0 ||
        dist_write_full(fd, &wire, sizeof(wire)) != 0) {
        if (errlen) snprintf(err, errlen, "failed to request distributed KV shard");
        goto cleanup;
    }

    ds4_dist_snapshot_begin_fixed begin;
    char msg[256];
    if (dist_read_snapshot_begin_frame(fd, &begin, msg, sizeof(msg),
                                       err, errlen) != 0)
        goto cleanup;
    if (begin.status != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             msg[0] ? msg : "distributed worker refused KV snapshot");
        goto cleanup;
    }
    const uint64_t payload_bytes = dist_u64_from_halves(begin.payload_hi,
                                                        begin.payload_lo);
    uint64_t got_session = dist_u64_from_halves(begin.session_hi, begin.session_lo);
    uint64_t got_request = dist_u64_from_halves(begin.request_hi, begin.request_lo);
    uint64_t got_hash = dist_u64_from_halves(begin.token_hash_hi, begin.token_hash_lo);
    if (begin.model_id != d->state.model_id ||
        got_session != d->session_id ||
        got_request != request_id ||
        got_hash != token_hash ||
        (begin.token_count != 0 && begin.token_count != (uint32_t)tokens->len) ||
        begin.layer_start != entry->layer_start ||
        begin.layer_end != entry->layer_end) {
        if (errlen) snprintf(err, errlen, "distributed KV shard metadata mismatch");
        goto cleanup;
    }
    if (dist_receive_snapshot_chunks_to_file(fd, request_id, fp,
                                             payload_bytes, err, errlen) != 0)
        goto cleanup;
    if (dist_read_snapshot_done_frame(fd, request_id, err, errlen) != 0)
        goto cleanup;
    if (payload_bytes_out) *payload_bytes_out = payload_bytes;
    rc = 0;

cleanup:
    close(fd);
    return rc;
}

int dist_prepare_shard_from_session_payload(
        ds4_dist_session *d,
        FILE *src,
        uint64_t *remaining,
        const ds4_dist_kv_layout *layout,
        const uint32_t *n_comp,
        const uint32_t *n_index_comp,
        uint32_t layer_start,
        uint32_t layer_end,
        FILE **tmp_out,
        uint64_t *payload_bytes_out,
        char *err,
        size_t errlen) {
    if (tmp_out) *tmp_out = NULL;
    if (payload_bytes_out) *payload_bytes_out = 0;
    FILE *tmp = dist_tmpfile_or_err("distributed KV shard", err, errlen);
    if (!tmp) return 1;

    int rc = 1;
    if (dist_kv_write_layer_header(tmp, layout, layer_start, layer_end,
                                   n_comp, n_index_comp, err, errlen) != 0)
        goto cleanup;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        uint64_t layer_bytes = 0;
        if (!dist_kv_layer_tensor_bytes(d->state.engine, layout, il,
                                        n_comp[il], n_index_comp[il],
                                        &layer_bytes)) {
            if (errlen) snprintf(err, errlen, "distributed KV layer byte count overflow");
            goto cleanup;
        }
        if (dist_payload_copy_bytes(src, tmp, layer_bytes, remaining, err, errlen) != 0)
            goto cleanup;
    }
    uint64_t bytes = 0;
    if (dist_measure_file(tmp, &bytes, "distributed KV shard", err, errlen) != 0 ||
        dist_rewind_file(tmp, "distributed KV shard", err, errlen) != 0)
        goto cleanup;
    *tmp_out = tmp;
    if (payload_bytes_out) *payload_bytes_out = bytes;
    rc = 0;

cleanup:
    if (rc != 0) fclose(tmp);
    return rc;
}

int dist_load_remote_shard_from_payload(
        ds4_dist_session *d,
        const ds4_dist_route_entry *entry,
        const int *tokens,
        uint32_t token_count,
        uint64_t token_hash,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    uint64_t request_id = d->request_id++;
    int fd = dist_connect_endpoint(entry->host, (int)entry->port, err, errlen);
    if (fd < 0) return 1;

    ds4_dist_snapshot_begin_fixed begin;
    memset(&begin, 0, sizeof(begin));
    begin.model_id = d->state.model_id;
    dist_u64_to_halves(d->session_id, &begin.session_hi, &begin.session_lo);
    dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
    dist_u64_to_halves(token_hash, &begin.token_hash_hi, &begin.token_hash_lo);
    begin.token_count = token_count;
    begin.layer_start = entry->layer_start;
    begin.layer_end = entry->layer_end;
    dist_u64_to_halves(payload_bytes, &begin.payload_hi, &begin.payload_lo);
    begin.token_bytes = token_count * sizeof(uint32_t);

    int rc = 1;
    if (dist_write_snapshot_load_begin(fd, &begin, tokens) <= 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed KV shard restore request");
        goto cleanup;
    }
    if (dist_send_snapshot_file_chunks(fd, request_id, fp, payload_bytes) <= 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed KV shard restore payload");
        goto cleanup;
    }
    if (dist_read_snapshot_done_frame(fd, request_id, err, errlen) != 0)
        goto cleanup;
    rc = 0;

cleanup:
    close(fd);
    return rc;
}

