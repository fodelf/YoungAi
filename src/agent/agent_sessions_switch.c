/* agent_sessions_switch.c — 机械拆分自 ds4_agent.c: Session Listing, History Rendering, And Completion (3/3: 删除/strip/切换)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len) {
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;
    if (unlink(path) != 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }
    if (sha_out) memcpy(sha_out, sha, 41);
    free(path);
    return true;
}

/* Strip the heavy backend payload from a saved session while preserving its
 * rendered transcript. Loading such a file later tokenizes the text and
 * rebuilds the live KV with a full prefill. */
bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint32_t *tokens_out,
                                       char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    char *title = NULL;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    if (ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE))
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    fclose(fp);
    if (!ok) {
        if (!err[0]) snprintf(err, err_len, "failed to read session");
        free(title);
        free(text);
        free(path);
        return false;
    }

    char actual_sha[41];
    agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
    if (strcmp(actual_sha, sha)) {
        snprintf(err, err_len, "cached session identity does not match file name");
        free(title);
        free(text);
        free(path);
        return false;
    }

    ds4_tokens stripped_tokens = {0};
    ds4_tokenize_rendered_chat(w->engine, text, &stripped_tokens);
    uint32_t stripped_token_count = (uint32_t)stripped_tokens.len;
    ds4_tokens_free(&stripped_tokens);

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint64_t now = (uint64_t)time(NULL);
    ds4_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason, hdr.ext_flags,
                            stripped_token_count, hdr.hits, hdr.ctx_size,
                            hdr.created_at, now, 0);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, text_bytes);

    errno = 0;
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
         fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
         fwrite(text, 1, text_bytes, fp) == text_bytes &&
         (!(hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE) ||
          agent_kv_write_title_trailer(fp, title, err, err_len)) &&
         fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    if (!ok) {
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) : "failed to write stripped session");
        unlink(tmp);
    } else {
        if (sha_out) memcpy(sha_out, sha, 41);
        if (tokens_out) *tokens_out = stripped_token_count;
    }

    free(tmp);
    free(title);
    free(text);
    free(path);
    return ok;
}

/* Load a saved session KV into the live transcript and optionally replay recent
 * history for the human. */
bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    bool stripped = false;
    ds4_kvstore_entry entry = {0};
    if (ds4_kvstore_read_entry_file(path, sha, &entry)) {
        stripped = entry.payload_bytes == 0;
        ds4_kvstore_entry_free(&entry);
    }
    if (stripped) {
        printf("rebuilding stripped session %.8s from rendered text...\n", sha);
        fflush(stdout);
    }

    ds4_tokens loaded = {0};
    agent_kv_session_meta meta = {0};
    bool ok = agent_kv_load_path(w, path, sha, NULL, 0, &loaded, &meta,
                                 err, err_len);
    if (ok) {
        ds4_tokens_free(&w->transcript);
        w->transcript = loaded;
        free(w->session_title);
        w->session_title = meta.title ? xstrdup(meta.title) : xstrdup("(no user prompt)");
        w->session_created_at = meta.created_at ? meta.created_at : (uint64_t)time(NULL);
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = meta.legacy_identity ? xstrdup(path) : NULL;
        agent_worker_note_system_prompt_seen(w);
        w->datetime_context_injected = true;
        pthread_mutex_lock(&w->mu);
        w->user_activity = true;
        w->session_dirty = false;
        w->status.state = AGENT_WORKER_IDLE;
        w->status.ctx_used = w->transcript.len;
        w->status.ctx_size = w->cfg->gen.ctx_size;
        w->status.error[0] = '\0';
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        printf("switched to session %.8s (%d tokens%s)\n",
               sha, w->transcript.len, stripped ? ", rebuilt from text" : "");
        if (history_turns > 0)
            (void)agent_worker_show_history(w, history_turns, err, err_len);
    } else {
        ds4_tokens_free(&loaded);
    }
    agent_kv_session_meta_free(&meta);
    free(path);
    return ok;
}
