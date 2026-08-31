/* dist_coord_kv.c — 机械拆自 ds4_distributed.c: coordinator KV 载荷 API(Coordinator KV Payload API)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Coordinator KV Payload API
 * ========================================================================= */

int ds4_dist_session_save_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen) {
    if (!d || !owner || !fp) {
        if (errlen) snprintf(err, errlen, "invalid distributed payload save");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;
    if (dist_kv_route_validate(d, err, errlen) != 0) return 1;

    const ds4_tokens *tokens = ds4_session_tokens(owner);
    if (!tokens || tokens->len < 0 || (uint64_t)tokens->len > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed session has no valid token timeline");
        return 1;
    }
    const uint32_t token_count = (uint32_t)tokens->len;
    const uint64_t token_hash = dist_token_hash_prefix(tokens->v, token_count);
    const uint32_t vocab = (uint32_t)ds4_engine_vocab_size(d->state.engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        if (errlen) snprintf(err, errlen, "out of memory saving distributed logits");
        return 1;
    }
    if (ds4_session_copy_logits(owner, logits, (int)vocab) != (int)vocab) {
        free(logits);
        if (errlen) snprintf(err, errlen, "failed to copy distributed logits");
        return 1;
    }

    const uint32_t shard_count = dist_kv_route_shard_count(d);
    ds4_dist_kv_shard_file *shards = calloc(shard_count, sizeof(shards[0]));
    uint32_t *n_comp = calloc(d->state.n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = calloc(d->state.n_layers, sizeof(n_index_comp[0]));
    if (!shards || !n_comp || !n_index_comp) {
        free(logits);
        free(shards);
        free(n_comp);
        free(n_index_comp);
        if (errlen) snprintf(err, errlen, "out of memory saving distributed KV payload");
        return 1;
    }

    int rc = 1;
    ds4_dist_kv_layout layout = {.vocab = vocab};
    bool layout_set = false;

    for (uint32_t shard = 0; shard < shard_count; shard++) {
        uint32_t layer_start = 0, layer_end = 0;
        const ds4_dist_route_entry *entry = NULL;
        dist_kv_route_shard(d, shard, &layer_start, &layer_end, &entry);
        shards[shard].fp = dist_tmpfile_or_err("distributed KV shard", err, errlen);
        if (!shards[shard].fp) goto cleanup;
        if (shard == 0) {
            if (ds4_session_save_layer_payload(owner, shards[shard].fp,
                                               layer_start, layer_end,
                                               err, errlen) != 0)
                goto cleanup;
            if (dist_measure_file(shards[shard].fp, &shards[shard].bytes,
                                  "distributed local KV shard", err, errlen) != 0)
                goto cleanup;
        } else {
            if (dist_save_remote_shard_to_file(d, entry, tokens, token_hash,
                                               shards[shard].fp,
                                               &shards[shard].bytes,
                                               err, errlen) != 0)
                goto cleanup;
        }
        if (shards[shard].bytes == 0) {
            if (errlen) snprintf(err, errlen, "distributed KV shard is empty");
            goto cleanup;
        }
        if (dist_kv_parse_layer_payload(d->state.engine,
                                        shards[shard].fp,
                                        shards[shard].bytes,
                                        layer_start,
                                        layer_end,
                                        &layout,
                                        &layout_set,
                                        n_comp,
                                        n_index_comp,
                                        &shards[shard],
                                        err,
                                        errlen) != 0)
            goto cleanup;
    }
    if (!layout_set || layout.token_count != token_count ||
        layout.n_layers != d->state.n_layers ||
        layout.vocab != vocab) {
        if (errlen) snprintf(err, errlen, "distributed KV shard metadata mismatch");
        goto cleanup;
    }

    if (dist_kv_write_session_header(fp, &layout, err, errlen) != 0)
        goto cleanup;
    for (uint32_t i = 0; i < token_count; i++) {
        if (dist_payload_write_u32(fp, (uint32_t)tokens->v[i], err, errlen) != 0)
            goto cleanup;
    }
    if (dist_payload_write_bytes(fp, logits,
                                 (uint64_t)vocab * sizeof(logits[0]),
                                 err, errlen) != 0)
        goto cleanup;
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_write_u32(fp, n_comp[il], err, errlen) != 0)
            goto cleanup;
    }
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_write_u32(fp, n_index_comp[il], err, errlen) != 0)
            goto cleanup;
    }
    for (uint32_t shard = 0; shard < shard_count; shard++) {
        if (dist_copy_file_range(shards[shard].fp,
                                 shards[shard].tensor_offset,
                                 shards[shard].tensor_bytes,
                                 fp,
                                 err,
                                 errlen) != 0)
            goto cleanup;
    }
    rc = 0;

cleanup:
    dist_kv_shards_close(shards, shard_count);
    free(shards);
    free(n_comp);
    free(n_index_comp);
    free(logits);
    return rc;
}

int ds4_dist_session_load_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    if (!d || !owner || !fp) {
        if (errlen) snprintf(err, errlen, "invalid distributed payload load");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;
    if (dist_kv_route_validate(d, err, errlen) != 0) return 1;

    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0)
            return 1;
    }
    if (h[DS4_SPH_MAGIC] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[DS4_SPH_VERSION] != DS4_SESSION_PAYLOAD_VERSION) {
        if (errlen) snprintf(err, errlen, "unsupported DS4 KV payload version");
        return 1;
    }
    ds4_dist_kv_layout layout = {
        .ctx = h[DS4_SPH_CTX],
        .prefill_cap = h[DS4_SPH_PREFILL_CAP],
        .raw_cap = h[DS4_SPH_RAW_CAP],
        .raw_window = h[DS4_SPH_RAW_WINDOW],
        .comp_cap = h[DS4_SPH_COMP_CAP],
        .token_count = h[DS4_SPH_TOKENS],
        .n_layers = h[DS4_SPH_LAYERS],
        .head_dim = h[DS4_SPH_HEAD_DIM],
        .indexer_head_dim = h[DS4_SPH_IDX_HEAD_DIM],
        .vocab = h[DS4_SPH_VOCAB],
        .raw_live = h[DS4_SPH_RAW_LIVE],
    };
    if (layout.n_layers != d->state.n_layers ||
        layout.ctx > (uint32_t)ds4_session_ctx(owner) ||
        layout.token_count >= (uint32_t)ds4_session_ctx(owner) ||
        layout.vocab != (uint32_t)ds4_engine_vocab_size(d->state.engine) ||
        !dist_kv_raw_live_valid(&layout)) {
        if (errlen) snprintf(err, errlen, "DS4 KV payload does not match current distributed runtime");
        return 1;
    }

    int *tokens = layout.token_count ?
        malloc((size_t)layout.token_count * sizeof(tokens[0])) : NULL;
    float *logits = malloc((size_t)layout.vocab * sizeof(logits[0]));
    uint32_t *n_comp = calloc(layout.n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = calloc(layout.n_layers, sizeof(n_index_comp[0]));
    if ((layout.token_count && !tokens) || !logits || !n_comp || !n_index_comp) {
        free(tokens);
        free(logits);
        free(n_comp);
        free(n_index_comp);
        if (errlen) snprintf(err, errlen, "out of memory loading distributed KV payload");
        return 1;
    }
    for (uint32_t i = 0; i < layout.token_count; i++) {
        uint32_t tok = 0;
        if (dist_payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            free(tokens);
            free(logits);
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (tok > (uint32_t)INT_MAX ||
            tok >= (uint32_t)ds4_engine_vocab_size(d->state.engine)) {
            free(tokens);
            free(logits);
            free(n_comp);
            free(n_index_comp);
            if (errlen) snprintf(err, errlen, "distributed KV payload token is outside vocabulary");
            return 1;
        }
        tokens[i] = (int)tok;
    }
    int empty_token_sentinel = 0;
    int *tokens_arg = layout.token_count ? tokens : &empty_token_sentinel;
    const uint64_t token_hash = dist_token_hash_prefix(tokens_arg, layout.token_count);
    if (dist_payload_read_bytes(fp, logits,
                                (uint64_t)layout.vocab * sizeof(logits[0]),
                                &remaining, err, errlen) != 0) {
        free(tokens);
        free(logits);
        free(n_comp);
        free(n_index_comp);
        return 1;
    }
    int rc = 1;
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0)
            goto cleanup;
        if (n_comp[il] > layout.comp_cap) {
            if (errlen) snprintf(err, errlen, "DS4 KV payload has invalid compressed row count");
            goto cleanup;
        }
    }
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0)
            goto cleanup;
        if (n_index_comp[il] > layout.comp_cap) {
            if (errlen) snprintf(err, errlen, "DS4 KV payload has invalid indexer row count");
            goto cleanup;
        }
    }

    const uint32_t shard_count = dist_kv_route_shard_count(d);
    for (uint32_t shard = 0; shard < shard_count; shard++) {
        uint32_t layer_start = 0, layer_end = 0;
        const ds4_dist_route_entry *entry = NULL;
        FILE *tmp = NULL;
        uint64_t shard_bytes = 0;
        dist_kv_route_shard(d, shard, &layer_start, &layer_end, &entry);
        if (dist_prepare_shard_from_session_payload(d,
                                                    fp,
                                                    &remaining,
                                                    &layout,
                                                    n_comp,
                                                    n_index_comp,
                                                    layer_start,
                                                    layer_end,
                                                    &tmp,
                                                    &shard_bytes,
                                                    err,
                                                    errlen) != 0)
            goto cleanup;
        if (shard == 0) {
            if (ds4_session_load_layer_payload(owner, tmp, shard_bytes,
                                               tokens_arg, layout.token_count,
                                               layer_start, layer_end,
                                               err, errlen) != 0) {
                fclose(tmp);
                goto cleanup;
            }
        } else {
            if (dist_load_remote_shard_from_payload(d, entry,
                                                    tokens_arg, layout.token_count,
                                                    token_hash,
                                                    tmp, shard_bytes,
                                                    err, errlen) != 0) {
                fclose(tmp);
                goto cleanup;
            }
        }
        fclose(tmp);
    }
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "DS4 KV payload has trailing bytes");
        goto cleanup;
    }
    if (ds4_session_set_logits(owner, logits, (int)layout.vocab) != 0) {
        if (errlen) snprintf(err, errlen, "failed to restore distributed logits");
        goto cleanup;
    }
    rc = 0;

cleanup:
    free(tokens);
    free(logits);
    free(n_comp);
    free(n_index_comp);
    return rc;
}

