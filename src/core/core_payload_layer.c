/* core_payload_layer.c — 层切片 payload 存取 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static bool ds4_layer_payload_range_valid(uint32_t layer_start, uint32_t layer_end) {
    return layer_start <= layer_end && layer_end < (uint32_t)DS4_N_LAYER;
}

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end) {
    if (!s || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end))
        return 0;
    if (ds4_session_is_cpu(s)) return 0;
#ifdef DS4_NO_GPU
    (void)layer_start;
    (void)layer_end;
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint64_t bytes = (uint64_t)DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint32_t n_layers = layer_end - layer_start + 1u;
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
#endif
}

int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload save");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before layer snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint32_t header[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        layer_start,
        layer_end,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = layer_start; rc == 0 && il <= layer_end; il++) {
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_write_tensor_span_f16_as_f32(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !tokens ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload load");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)payload_bytes;
    (void)n_tokens;
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session layer payload version");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_layer_start = h[11];
    const uint32_t saved_layer_end = h[12];
    const uint32_t saved_raw_live = h[13];
    (void)saved_prefill_cap;
    if (saved_layer_start != layer_start || saved_layer_end != layer_end) {
        payload_set_err(err, errlen, "KV shard layer range does not match requested worker");
        return 1;
    }
    if (saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens != n_tokens ||
        saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV shard does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM) {
        payload_set_err(err, errlen, "KV shard was written for a different DS4 layout");
        return 1;
    }
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV shard graph chunk layout does not match current runtime");
        return 1;
    }
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap) {
        payload_set_err(err, errlen, "KV shard raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV shard compressed cache is larger than current context");
        return 1;
    }

    const uint32_t n_layers = layer_end - layer_start + 1u;
    uint32_t *n_comp = xcalloc(n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = xcalloc(n_layers, sizeof(n_index_comp[0]));
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_comp[i] > saved_comp_cap || n_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_index_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_index_comp[i] > saved_comp_cap || n_index_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        free(n_comp);
        free(n_index_comp);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV shard restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                     g->layer_attn_comp_cache[il],
                                                     0,
                                                     (uint64_t)n_comp[i] * DS4_N_HEAD_DIM,
                                                     buf,
                                                     DS4_SESSION_IO_CHUNK,
                                                     &remaining,
                                                     err,
                                                     errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[i] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[i] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc == 0 && remaining != 0) {
        payload_set_err(err, errlen, "KV shard has trailing payload bytes");
        rc = 1;
    }
    if (rc == 0 && ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV shard restore");
        rc = 1;
    }
    if (rc == 0) {
        token_vec_free(&s->checkpoint);
        memset(&s->checkpoint, 0, sizeof(s->checkpoint));
        for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
        for (uint32_t i = 0; i < n_layers; i++) {
            const uint32_t il = layer_start + i;
            g->layer_n_comp[il] = n_comp[i];
            g->layer_n_index_comp[il] = n_index_comp[i];
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        g->mtp_n_raw = 0;
    }
    free(n_comp);
    free(n_index_comp);
    return rc;
#endif
}

