/* agent_session_persist.c — 机械拆分自 ds4_agent.c: Agent KV Store And Session Persistence (2/2: worker 会话同步/保存)。
 * 逻辑/字符串零改动; 跨文件符号声明见 agent_internal.h。 */
#include "agent_internal.h"

void agent_worker_build_system_tokens(agent_worker *w, ds4_tokens *out) {
    ds4_chat_begin(w->engine, out);
    if (w->cfg->gen.think_mode == DS4_THINK_MAX &&
        effective_think_mode(w->cfg) == DS4_THINK_MAX)
        ds4_chat_append_max_effort_prefix(w->engine, out);
    agent_append_system_prompt(w->engine, out, w->cfg->gen.system);
}

void agent_publish_system_status(agent_worker *w, const char *msg) {
    if (w->cfg->non_interactive) return;
    if (isatty(STDOUT_FILENO)) {
        agent_publish(w, "\x1b[1;33m", strlen("\x1b[1;33m"));
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\x1b[0m\n", strlen("\x1b[0m\n"));
    } else {
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\n", 1);
    }
}

int agent_web_confirm(void *privdata, const char *message,
                             char *err, size_t err_len) {
    agent_worker *w = privdata;
    if (!w || w->cfg->non_interactive) {
        snprintf(err, err_len,
                 "visible Chrome browser startup requires interactive approval");
        return 0;
    }

    pthread_mutex_lock(&w->mu);
    w->web_approval_pending = true;
    w->web_approval_answered = false;
    w->web_approval_result = false;
    w->web_approval_error[0] = '\0';
    snprintf(w->web_approval_message, sizeof(w->web_approval_message),
             "%s", message ? message : "Start visible Chrome browser? (y/n) ");
    agent_wake_locked(w);
    while (!w->stop && !w->web_approval_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    bool ok = w->web_approval_result;
    if (!ok) {
        snprintf(err, err_len, "%s",
                 w->web_approval_error[0] ? w->web_approval_error :
                 "user denied Chrome browser start");
    }
    pthread_mutex_unlock(&w->mu);
    return ok ? 1 : 0;
}

void agent_web_log(void *privdata, const char *message) {
    agent_worker *w = privdata;
    if (!w || !message || !message[0]) return;
    agent_trace(w, "web: %s", message);
}

bool worker_take_web_approval_request(agent_worker *w,
                                             char *message, size_t message_len) {
    pthread_mutex_lock(&w->mu);
    bool pending = w->web_approval_pending;
    if (pending) {
        snprintf(message, message_len, "%s", w->web_approval_message);
        w->web_approval_pending = false;
    }
    pthread_mutex_unlock(&w->mu);
    return pending;
}

void worker_answer_web_approval(agent_worker *w, bool allow,
                                       const char *deny_error) {
    pthread_mutex_lock(&w->mu);
    w->web_approval_result = allow;
    w->web_approval_answered = true;
    if (!allow)
        snprintf(w->web_approval_error, sizeof(w->web_approval_error),
                 "%s", deny_error && deny_error[0] ? deny_error :
                 "user denied Chrome browser start");
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

/* When a model turn finishes with a tool call, queued user messages should not
 * preempt that tool.  The worker asks the UI thread for the queue contents only
 * after the tool result is appended, so the next model input can contain both
 * the tool observation and the user's pending correction. */
char *worker_request_queued_user_drain(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->queued_user_drain_pending = true;
    w->queued_user_drain_answered = false;
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = NULL;
    agent_wake_locked(w);
    pthread_cond_signal(&w->cond);
    while (!w->stop && !w->queued_user_drain_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    char *text = w->queued_user_drain_text;
    w->queued_user_drain_text = NULL;
    w->queued_user_drain_pending = false;
    w->queued_user_drain_answered = false;
    pthread_mutex_unlock(&w->mu);
    return text;
}

bool worker_take_queued_user_drain_request(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool pending = w->queued_user_drain_pending;
    if (pending) w->queued_user_drain_pending = false;
    pthread_mutex_unlock(&w->mu);
    return pending;
}

void worker_answer_queued_user_drain(agent_worker *w, char *text) {
    pthread_mutex_lock(&w->mu);
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = text;
    w->queued_user_drain_answered = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

/* Synchronize the live DS4 session to a transcript.  This is the agent's main
 * cache-saving operation: if the requested transcript extends the live session,
 * only the suffix is prefetched; otherwise the DS4 session rebuilds from the
 * longest common prefix it can retain. */
int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len) {
    int old_pos = ds4_session_pos(w->session);
    int common = ds4_session_common_prefix(w->session, tokens);
    int cached = common == old_pos && tokens->len >= old_pos ? common : 0;
    int suffix = tokens->len - cached;
    if (suffix < 0) suffix = tokens->len;

    if (publish_progress) {
        pthread_mutex_lock(&w->mu);
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    ds4_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    ds4_session_set_display_progress(w->session,
                                     publish_progress ? worker_progress_cb : NULL,
                                     publish_progress ? w : NULL);
    int rc = ds4_session_sync(w->session, tokens, err, err_len);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    return rc;
}

/* Start a new session at the system/tool prompt.  A fixed sysprompt.kv
 * checkpoint avoids paying this prefill cost repeatedly, but only when the
 * rendered prompt text still matches the file.  The same fixed path is shared
 * by Flash and Pro; agent_kv_load_path() checks the model id, so switching
 * model families rebuilds this cache instead of restoring incompatible KV. */
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len) {
    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &sys, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render system prompt");
        ds4_tokens_free(&sys);
        return false;
    }

    bool loaded = false;
    char load_err[160] = {0};
    if (w->sysprompt_path) {
        loaded = agent_kv_load_path(w, w->sysprompt_path, NULL,
                                    text, text_len, &w->transcript,
                                    NULL,
                                    load_err, sizeof(load_err));
        if (loaded) {
            agent_trace(w, "sysprompt kv hit file=%s tokens=%d",
                        w->sysprompt_path, w->transcript.len);
        }
    }

    if (!loaded) {
        if (w->sysprompt_path)
            agent_publish_system_status(w, "Updating system prompt cache...");
        ds4_tokens_free(&w->transcript);
        ds4_tokens_copy(&w->transcript, &sys);
        if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
            free(text);
            ds4_tokens_free(&sys);
            return false;
        }
        if (w->sysprompt_path) {
            char save_err[160] = {0};
            char ignored_sha[41];
            if (!agent_kv_save_path(w, w->sysprompt_path, &w->transcript,
                                    "agent-system", ignored_sha,
                                    NULL, 0,
                                    save_err, sizeof(save_err)))
            {
                if (w->cfg->non_interactive) {
                    fprintf(stderr, "ds4-agent: failed to save system prompt KV: %s\n",
                            save_err);
                } else {
                    agent_buf b = {0};
                    agent_buf_puts(&b, "\nds4-agent: failed to save system prompt KV: ");
                    agent_buf_puts(&b, save_err);
                    agent_buf_puts(&b, "\n");
                    char *msg = agent_buf_take(&b);
                    agent_publish(w, msg, strlen(msg));
                    free(msg);
                }
            } else {
                agent_trace(w, "sysprompt kv stored file=%s tokens=%d",
                            w->sysprompt_path, w->transcript.len);
            }
        }
    }

    agent_worker_note_system_prompt_seen(w);
    pthread_mutex_lock(&w->mu);
    w->user_activity = false;
    w->session_dirty = false;
    w->status.state = AGENT_WORKER_IDLE;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    w->datetime_context_injected = false;
    agent_worker_clear_session_identity(w);
    free(text);
    ds4_tokens_free(&sys);
    return true;
}

static bool agent_worker_should_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool stop = w->stop;
    pthread_mutex_unlock(&w->mu);
    return stop;
}

bool agent_worker_wait_distributed_route(agent_worker *w, char *err, size_t err_len) {
    if (!w || !w->cfg ||
        w->cfg->engine.distributed.role != DS4_DISTRIBUTED_COORDINATOR)
        return true;

    char last[160] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};
    for (;;) {
        int ready = ds4_session_distributed_route_ready(w->session, err, err_len);
        if (ready > 0) {
            if (ticks != 0) {
                if (w->cfg->non_interactive)
                    fprintf(stderr, "ds4-agent: distributed route ready\n");
                else
                    agent_publish_system_status(w, "Distributed route ready.");
            }
            if (err_len) err[0] = '\0';
            return true;
        }
        if (ready < 0) return false;

        const char *why = err && err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            if (w->cfg->non_interactive) {
                fprintf(stderr, "ds4-agent: waiting for distributed route: %s\n", why);
            } else {
                char msg[224];
                snprintf(msg, sizeof(msg), "Waiting for distributed route: %s", why);
                agent_publish_system_status(w, msg);
            }
            snprintf(last, sizeof(last), "%s", why);
        }
        if (agent_worker_should_stop(w)) {
            snprintf(err, err_len, "agent stopped while waiting for distributed route");
            return false;
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

bool agent_worker_has_user_session(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity;
    pthread_mutex_unlock(&w->mu);
    return yes;
}

bool agent_worker_needs_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity && w->session_dirty;
    pthread_mutex_unlock(&w->mu);
    return yes;
}

/* Save the current session under its stable agent identity.  The worker owns
 * the live KV, so busy /save requests are deferred until a stable append-only
 * point and then executed by the worker thread. */
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len) {
    if (!agent_worker_has_user_session(w)) {
        snprintf(err, err_len, "nothing to save");
        return false;
    }

    if (agent_worker_sync_tokens(w, &w->transcript, false, err, err_len) != 0)
        return false;
    if (!agent_mkdir_p(w->cache_dir)) {
        snprintf(err, err_len, "failed to create %s", w->cache_dir);
        return false;
    }

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    if (!w->session_title) {
        w->session_title = agent_session_title_from_text(text, text_len, 0);
    }
    if (w->session_created_at == 0)
        w->session_created_at = (uint64_t)time(NULL);

    char sha[41];
    agent_session_identity_sha(w->session_title, w->session_created_at, sha);
    char *path = agent_kv_path_for_sha(w->cache_dir, sha);

    bool ok = agent_kv_save_path(w, path, &w->transcript,
                                 "agent-session", sha_out,
                                 w->session_title, w->session_created_at,
                                 err, err_len);
    if (ok) {
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        if (w->legacy_session_path_to_delete &&
            strcmp(w->legacy_session_path_to_delete, path) != 0)
        {
            unlink(w->legacy_session_path_to_delete);
        }
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = NULL;
        pthread_mutex_lock(&w->mu);
        w->session_dirty = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        if (tokens_out) *tokens_out = w->transcript.len;
    }
    free(path);
    free(text);
    return ok;
}

bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    int tokens = 0;
    bool ok = agent_worker_save_session_now(w, sha, &tokens, err, err_len);
    if (ok) printf("saved session %.8s (%d tokens)\n", sha, tokens);
    return ok;
}
