#include "ds4_kvstore.h"
#include "kv_internal.h"

static bool kv_trailer_serialized_size(const ds4_kvstore_trailer_hooks *hooks,
                                       const char *text,
                                       uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!hooks || !hooks->serialized_size) return true;
    return hooks->serialized_size(hooks->ud, text, bytes_out);
}

static bool kv_trailer_write(const ds4_kvstore_trailer_hooks *hooks,
                             FILE *fp, const char *text,
                             uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!hooks || !hooks->write) return true;
    return hooks->write(hooks->ud, fp, text, written_bytes);
}

static void kv_cache_rewrite_trailer(ds4_kvstore *kc, const char *path,
                                     const char *text,
                                     const ds4_kvstore_trailer_hooks *hooks) {
    uint64_t trailer_est = 0;
    if (!hooks || !hooks->write || !hooks->serialized_size ||
        !kv_trailer_serialized_size(hooks, text, &trailer_est) ||
        trailer_est == 0)
    {
        return;
    }
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    uint64_t end = DS4_KVSTORE_FIXED_HEADER + 4ull +
                   (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_trailer_write(hooks, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            uint8_t h[DS4_KVSTORE_FIXED_HEADER];
            uint64_t now = (uint64_t)time(NULL);
            ds4_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason,
                                    (uint8_t)(hdr.ext_flags | hooks->ext_flag),
                                    hdr.tokens, hdr.hits, hdr.ctx_size,
                                    hdr.created_at, now, hdr.payload_bytes);
            ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                 fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)kc;
    (void)ok;
}

bool ds4_kvstore_store_live_prefix_text(ds4_kvstore *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ds4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len) {
    if (!kc->enabled) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    const int original_len = tokens->len;

    ds4_tokens store_tokens = {0};
    ds4_kvstore_tokens_copy_prefix(&store_tokens, tokens, store_len);

    /* 兼容键=routed 类型码(bits 口径会把不同 2-bit 格式塌成同一个 2, 跨模型互认) */
    const int quant_bits = ds4_engine_routed_kv_key(engine);
    if (quant_bits <= 0) {
        ds4_tokens_free(&store_tokens);
        return false;
    }
    const int model_id = ds4_engine_model_id(engine);

    char save_err[160] = {0};
    const ds4_tokens *live_tokens = ds4_session_tokens(session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !ds4_tokens_starts_with(live_tokens, &store_tokens))
    {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                live_tokens ? live_tokens->len : -1);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = NULL;
    const bool text_override = cache_text_override && cache_text_override[0];
    if (text_override) {
        text = kv_xstrdup(cache_text_override);
        text_len = strlen(text);
    } else {
        text = ds4_kvstore_render_tokens_text(engine, &store_tokens, &text_len);
    }
    if (text_len > UINT32_MAX) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d because rendered text is too large",
                kv_log_name(kc), store_tokens.len);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t trailer_est_bytes = 0;
    if (!kv_trailer_serialized_size(hooks, text, &trailer_est_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                kv_log_name(kc), store_tokens.len, reason);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    char *path = ds4_kvstore_path_for_sha(kc, sha);
    const uint8_t reason_code = ds4_kvstore_reason_code(reason);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     model_id,
                                     quant_bits, ds4_session_ctx(session))) {
        kv_cache_rewrite_trailer(kc, path, text, hooks);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return true;
    }

    ds4_session_payload_file staged = {0};
    if (ds4_session_stage_payload(session, &staged,
                                  save_err, sizeof(save_err)) != 0) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because KV payload staging failed: %s",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                save_err[0] ? save_err : "unknown error");
        if (err && err_len) snprintf(err, err_len, "%s",
                                     save_err[0] ? save_err : "unknown error");
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t payload_bytes = staged.bytes;

    uint64_t est_file_bytes = 0, est_required_bytes = 0;
    if (!ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                    trailer_est_bytes,
                                    &est_file_bytes, &est_required_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                (double)est_file_bytes / (1024.0 * 1024.0),
                (double)est_required_bytes / (1024.0 * 1024.0),
                (double)kc->budget_bytes / (1024.0 * 1024.0));
        ds4_session_payload_file_free(&staged);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    ds4_kvstore_eviction_context incoming = {
        .text = text,
        .text_len = text_len,
        .model_id = (uint8_t)model_id,
        .quant_bits = (uint8_t)quant_bits,
        .ctx_size = (uint32_t)ds4_session_ctx(session),
        .reject_different_quant = kc->reject_different_quant,
    };
    ds4_kvstore_evict(kc, live_tokens, est_file_bytes, &incoming);

    kv_buf tmpb = {0};
    kv_buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = kv_buf_take(&tmpb);
    const double save_t0 = kv_now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache failed to create %s: %s save=%.1f ms",
                kv_log_name(kc), tmp, strerror(errno),
                (kv_now_sec() - save_t0) * 1000.0);
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint8_t ext_flags = trailer_est_bytes > 0 && hooks ? hooks->ext_flag : 0;
    if (text_override) ext_flags |= cache_text_ext;
    ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                            reason_code, ext_flags,
                            (uint32_t)store_tokens.len, 0,
                            (uint32_t)ds4_session_ctx(session),
                            now, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);
    uint64_t trailer_bytes = 0;
    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_write_staged_payload(&staged, fp,
                                               save_err, sizeof(save_err)) == 0 &&
              kv_trailer_write(hooks, fp, text, &trailer_bytes) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    uint64_t final_file_bytes = 0, final_required_bytes = 0;
    bool final_size_over_budget = false;
    if (ok && !ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                          trailer_bytes,
                                          &final_file_bytes,
                                          &final_required_bytes))
    {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (kv_now_sec() - save_t0) * 1000.0;
    if (!ok) {
        if (final_size_over_budget) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                    kv_log_name(kc),
                    store_tokens.len,
                    reason,
                    (double)final_file_bytes / (1024.0 * 1024.0),
                    (double)final_required_bytes / (1024.0 * 1024.0),
                    (double)kc->budget_bytes / (1024.0 * 1024.0),
                    save_ms);
        } else {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache store failed (%s): %s save=%.1f ms",
                    kv_log_name(kc),
                    reason,
                    saved_errno ? strerror(saved_errno) :
                    (save_err[0] ? save_err : "unknown error"),
                    save_ms);
        }
        if (err && err_len) {
            snprintf(err, err_len, "%s",
                     saved_errno ? strerror(saved_errno) :
                     (save_err[0] ? save_err : "unknown error"));
        }
        unlink(tmp);
    } else {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms",
                kv_log_name(kc),
                store_tokens.len,
                original_len - store_tokens.len,
                reason,
                text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                (double)(DS4_KVSTORE_FIXED_HEADER + 4ull + text_len + payload_bytes + trailer_bytes) / (1024.0 * 1024.0),
                save_ms);
    }
    ds4_session_payload_file_free(&staged);
    free(tmp);
    free(text);
    free(path);
    ds4_tokens_free(&store_tokens);
    return ok;
}

bool ds4_kvstore_store_live_prefix(ds4_kvstore *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ds4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len) {
    return ds4_kvstore_store_live_prefix_text(kc, engine, session, tokens,
                                              store_len, reason, NULL, 0, NULL,
                                              hooks, err, err_len);
}

bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len) {
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens) return false;
    const int target = ds4_kvstore_continued_store_target(kc, tokens->len);
    if (target == 0) return false;
    if (ds4_kvstore_store_live_prefix(kc, engine, session, tokens, target,
                                      "continued", hooks, err, err_len))
    {
        ds4_kvstore_note_store(kc, target);
        return true;
    }
    return false;
}

int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        ds4_kvstore_entry *e = &kc->entry[i];
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if (e->model_id != (uint8_t)model_id) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            ds4_kvstore_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    return best;
}

int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol) {
    if (result) memset(result, 0, sizeof(*result));
    if (effective_prompt) effective_prompt->len = 0;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = ds4_engine_routed_kv_key(engine);   /* 兼容键=类型码 */
    if (quant_bits <= 0) return 0;
    const int model_id = ds4_engine_model_id(engine);
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = ds4_kvstore_find_text_prefix(kc, prompt_text, model_id, quant_bits,
                                           ds4_session_ctx(session));
    if (idx < 0) return 0;

    ds4_kvstore_entry e = kc->entry[idx];
    char *path = kv_xstrdup(e.path);
    const double load_t0 = kv_now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    ds4_kvstore_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    char *cached_text = NULL;
    if (header_ok) {
        if (hdr.model_id != (uint8_t)model_id) {
            header_ok = false;
            fail_reason = "cached checkpoint was written for a different model";
        } else if ((uint64_t)text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = kv_xmalloc((size_t)text_bytes + 1);
            if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = '\0';
                char text_sha[41];
                ds4_kvstore_sha1_bytes_hex(cached_text, text_bytes, text_sha);
                if (strcmp(text_sha, e.sha)) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!ds4_kvstore_byte_prefix_match(prompt_text, prompt_bytes,
                                                          cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    char err[160] = {0};
    int loaded = 0;
    if (header_ok &&
        ds4_session_load_payload(session, fp, hdr.payload_bytes, err, sizeof(err)) == 0)
    {
        const ds4_tokens *loaded_tokens = ds4_session_tokens(session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt from that exact history and tokenize only the text
                 * suffix after the byte prefix. */
                ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
                    engine, loaded_tokens, prompt_text + text_bytes,
                    effective_prompt);
            }
            if (hooks && hooks->load && (hdr.ext_flags & hooks->ext_flag)) {
                hooks->load(hooks->ud, fp, hooks->load_wanted);
            }
        } else {
            ds4_session_invalidate(session);
            unlink(path);
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache discarded corrupt text-prefix payload%s%s %s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    path);
        }
    } else {
        if (header_ok) ds4_session_invalidate(session);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache load failed%s%s %s: %s load=%.1f ms",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                path,
                header_ok ? err : fail_reason,
                (kv_now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (kv_now_sec() - load_t0) * 1000.0;
        kc->continued_last_store_tokens = loaded;
        const char *key_kind = ds4_kvstore_key_kind(hdr.ext_flags);
        bool consumed = false;
        if (kc->opt.cold_max_tokens > 0 && loaded > kc->opt.cold_max_tokens) {
            unlink(path);
            consumed = true;
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms consumed file=%s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        } else {
            ds4_kvstore_touch_file(path, hdr.hits + 1);
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        }
        if (result) {
            result->tokens = loaded;
            result->text_bytes = text_bytes;
            result->quant_bits = hdr.quant_bits;
            result->ext_flags = hdr.ext_flags;
            result->load_ms = load_ms;
            result->consumed = consumed;
            result->path = kv_xstrdup(path);
        }
    }
    free(cached_text);
    free(path);
    return loaded;
}

void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result) {
    if (!result) return;
    free(result->path);
    memset(result, 0, sizeof(*result));
}
