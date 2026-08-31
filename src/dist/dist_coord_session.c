/* dist_coord_session.c — 机械拆自 ds4_distributed.c: coordinator 会话 API(Coordinator Session API)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Coordinator Session API
 * =========================================================================
 *
 * These functions are the distributed backend for the normal ds4_session API.
 * Program frontends should keep using ds4_session_sync/eval/save/load; ds4.c
 * selects these calls when the owning session has a coordinator attached.
 */

/* Defined further below (after their dependencies); forward-declared so the
 * session-driven coordinator can opt into reverse-connect. */

int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen) {
    (void)owner;
    if (!out || !engine || !opt) {
        if (errlen) snprintf(err, errlen, "missing distributed session parameters");
        return 1;
    }
    *out = NULL;
    if (opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "distributed session requires coordinator role");
        return 1;
    }
    if (dist_validate_options(opt, err, errlen) != 0) return 1;

    const bool reverse = dist_reverse_connect_enabled();
    int listen_fd = -1;
    if (!reverse) {
        listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, errlen);
        if (listen_fd < 0) return 1;
    }

    ds4_dist_session *d = calloc(1, sizeof(*d));
    if (!d) {
        if (listen_fd >= 0) close(listen_fd);
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return 1;
    }

    /* calloc zeroes listen_fd to 0 (a valid fd); reverse mode never listens, so
     * pin it to -1 so session_free does not shut down stdin. */
    d->listen_fd = reverse ? -1 : listen_fd;
    d->state.engine = engine;
    d->state.model_id = (uint32_t)ds4_engine_model_id(engine);
    d->state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    d->state.local_start = opt->layers.start;
    d->state.local_end = dist_resolved_layer_end(opt, d->state.n_layers);
    d->state.ctx_size = ctx_size > 0 ? (uint32_t)ctx_size : 0u;
    d->state.local_has_output = opt->layers.has_output;
    d->state.local_can_output_head = true;
    d->state.replay_check = opt->replay_check;
    d->state.debug = opt->debug;
    d->state.use_control_for_work = true;
    d->state.mtp_draft = false;
    d->state.mtp_draft_local = false;
    d->state.prefill_chunk = opt->prefill_chunk;
    d->state.prefill_window = opt->prefill_window;
    d->state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    pthread_mutex_init(&d->state.mu, NULL);
    d->session_id = dist_make_session_id(d);
    d->request_id = 1;

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&d->state,
                     "ds4: distributed coordinator API: %s %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     reverse ? "reverse-connect dialing worker" : "listening on",
                     reverse ? opt->coordinator_host : opt->listen_host,
                     reverse ? opt->coordinator_port : opt->listen_port,
                     d->state.model_id,
                     d->state.n_layers,
                     opt->layers.start,
                     local_end,
                     d->state.activation_bits);

    if (reverse) {
        /* Coordinator dials the worker's control listener (--coordinator HOST PORT
         * points at the worker). The worker only accepts, so its host never needs
         * to make a local-network outbound connection. */
        d->reverse_ctx.state = &d->state;
        d->reverse_ctx.host = opt->coordinator_host;
        d->reverse_ctx.port = opt->coordinator_port;
        if (pthread_create(&d->reverse_tid, NULL, dist_coordinator_reverse_connect_main, &d->reverse_ctx) != 0) {
            pthread_mutex_destroy(&d->state.mu);
            free(d);
            if (errlen) snprintf(err, errlen, "failed to start distributed coordinator reverse-connect loop");
            return 1;
        }
        pthread_detach(d->reverse_tid);
        d->reverse_started = true;
        *out = d;
        return 0;
    }

    d->accept_ctx.state = &d->state;
    d->accept_ctx.listen_fd = listen_fd;
    if (pthread_create(&d->accept_tid, NULL, dist_coordinator_accept_main, &d->accept_ctx) != 0) {
        close(listen_fd);
        pthread_mutex_destroy(&d->state.mu);
        free(d);
        if (errlen) snprintf(err, errlen, "failed to start distributed coordinator accept loop");
        return 1;
    }
    pthread_detach(d->accept_tid);
    d->accept_started = true;
    *out = d;
    return 0;
}

void ds4_dist_session_free(ds4_dist_session *d) {
    if (!d) return;
    dist_mtp_print_summary(d, "summary");
    if (d->listen_fd >= 0) {
        shutdown(d->listen_fd, SHUT_RDWR);
        close(d->listen_fd);
        d->listen_fd = -1;
    }
    dist_route_plan_free(&d->plan);
    pthread_mutex_lock(&d->state.mu);
    d->state.shutting_down = true;
    for (ds4_dist_worker_entry *it = d->state.workers; it; it = it->next) {
        if (it->fd >= 0) shutdown(it->fd, SHUT_RDWR);
    }
    pthread_mutex_unlock(&d->state.mu);
    /* Client threads are detached and remove their registry entries after the
     * socket closes. Keep this small coordinator object process-lifetime to
     * avoid racing those threads during application shutdown. */
}

int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return -1;
    }

    ds4_dist_route_plan probe = {0};
    if (!dist_coordinator_build_route_plan(&d->state, &probe, NULL, err, errlen)) {
        return 0;
    }
    dist_route_plan_free(&probe);
    if (errlen) err[0] = '\0';
    return 1;
}

int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !prompt || prompt->len <= 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed sync request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    if (checkpoint &&
        checkpoint->len >= 0 &&
        checkpoint->len <= prompt->len &&
        ds4_tokens_starts_with(prompt, checkpoint))
    {
        if (checkpoint->len == prompt->len) return 0;

        uint32_t chunk_cap = 0;
        if (dist_coordinator_prefill_chunk_cap(&d->state, owner, &chunk_cap, err, errlen) != 0) {
            return 1;
        }
        const uint32_t pos0 = (uint32_t)checkpoint->len;
        const uint32_t suffix = (uint32_t)prompt->len - pos0;
        /* Incremental prefill: pos0 reused from the live checkpoint, only the
         * `suffix` tokens are (re)processed (confirmed correct by the wave-49
         * replay -- turn N prefills only its appended suffix, not the full
         * context).  The per-turn TTFT is then the suffix's cost at the current
         * context depth, which stays W2/attention-bound per token. */
        if (dist_coordinator_can_pipeline_prefill(&d->state, &d->plan, owner, suffix, chunk_cap)) {
            int prefill_rc = dist_coordinator_prefill_prompt_pipelined(&d->state,
                                                                       owner,
                                                                       &d->plan,
                                                                       prompt,
                                                                       pos0,
                                                                       suffix,
                                                                       false,
                                                                       chunk_cap,
                                                                       d->session_id,
                                                                       &d->request_id,
                                                                       logits,
                                                                       err,
                                                                       errlen);
            if (prefill_rc != 0) {
                /* PC.4 diag: the pipelined incremental prefill of just the
                 * suffix failed, so we fall back to re-prefilling the FULL
                 * prompt (pos 0..len) -- this silently defeats incremental
                 * prefill (observed: turn-3 replay re-ran the full 368-token
                 * context).  Log the failure reason + the fallback cost so the
                 * root cause is visible instead of hiding behind a slow turn.
                 * err[] is reused by the rebuild below, so capture it here. */
                fprintf(stderr,
                        "ds4: [PC.4] dist incremental prefill FAILED (rc=%d pos0=%u suffix=%u): %s "
                        "-- falling back to full %d-token rebuild\n",
                        prefill_rc, pos0, suffix, err[0] ? err : "(no detail)",
                        (int)prompt->len);
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
            }
            return 0;
        }

        uint32_t pos = pos0;
        while (pos < (uint32_t)prompt->len) {
            const uint32_t remaining = (uint32_t)prompt->len - pos;
            const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
            int eval_rc = dist_coordinator_eval_span(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt->v + pos,
                                                     chunk,
                                                     pos,
                                                     d->session_id,
                                                     d->request_id++,
                                                     false,
                                                     logits,
                                                     NULL,
                                                     err,
                                                     errlen);
            if (eval_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             eval_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
                return 0;
            }
            pos += chunk;
            dist_report_prefill_progress(owner, pos, (uint32_t)prompt->len);
        }
        return 0;
    }

    /* PC.4 diag: we did NOT take the prefix-reuse branch above, so this is a
     * full cold prefill of the whole prompt (no incremental skip).  In a
     * multi-turn session that should be rare (only turn 1); if it fires on a
     * later turn the dist checkpoint disagreed with the owner's prefix, which
     * silently re-prefills the full context.  Log why the reuse check failed. */
    /* Only the cold first turn legitimately has no checkpoint; if a checkpoint
     * exists but was not a usable prefix we silently re-prefill the full
     * context (defeats PC.4 incremental prefill), so warn with the reason. */
    if (checkpoint != NULL) {
        const int ck_prefix = (checkpoint->len >= 0 && checkpoint->len <= prompt->len)
                                  ? (ds4_tokens_starts_with(prompt, checkpoint) ? 1 : 0)
                                  : -1;
        fprintf(stderr,
                "ds4: [PC.4] dist FULL prefill, no prefix reuse despite checkpoint "
                "(ck_len=%d prompt_len=%d starts_with=%d) -- re-prefilling all %d tokens\n",
                (int)checkpoint->len, (int)prompt->len, ck_prefix, (int)prompt->len);
    }
    int prefill_rc = dist_coordinator_prefill_prompt(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     err,
                                                     errlen);
    if (prefill_rc != 0) {
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            return 1;
        }
        d->plan_ready = true;
    }
    return 0;
}

int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !checkpoint || checkpoint->len < 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed decode request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, checkpoint);
    ds4_tokens_push(&transcript, token);

    int rc = dist_coordinator_eval_span(&d->state,
                                        owner,
                                        &d->plan,
                                        &token,
                                        1,
                                        (uint32_t)checkpoint->len,
                                        d->session_id,
                                        d->request_id++,
                                        false,
                                        logits,
                                        NULL,
                                        err,
                                        errlen);
    if (rc != 0) {
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     &transcript,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            ds4_tokens_free(&transcript);
            return 1;
        }
        d->plan_ready = true;
        rc = 0;
    }
    ds4_tokens_free(&transcript);
    return rc;
}

/* docs/archive/mtp.md Phase 1 (Scheme A) cross-machine speculative decode. Returns the number
 * of tokens committed this call (>=1) into accepted[], or -1 on hard failure.
 *
 * Two cross-machine rounds:
 *   Round 1 (DRAFT): eval first_token through the route; the last-layer worker
 *     also drafts K MTP candidates and returns them with the logits.
 *   Round 2 (VERIFY): if the target's next-token argmax equals drafts[0], push
 *     the K candidates through the route as one batch; the worker returns K logit
 *     rows so the coordinator can find the accepted greedy prefix.
 * Accepted draft KV is kept; the rejected tail is rolled back locally now and on
 * the worker via the next frame's accept_len. Output correctness is gated purely
 * on the target argmax, so a wrong draft only costs speed; KV divergence trips the
 * prefix-hash rebuild fallback, so it is also speed-only. Greedy-only. */
int ds4_dist_session_eval_speculative(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int first_token,
        int eos_token,
        int *accepted,
        int accepted_cap,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !checkpoint || checkpoint->len < 0 || !accepted ||
        accepted_cap <= 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed speculative request");
        return -1;
    }
    /* MTP/投机整族已删除(2026-08-05 用户裁决: Go 定型优化) — 纯单 token 平解码。 */
    (void)eos_token;
    if (ds4_dist_session_eval(d, owner, checkpoint, first_token, logits, err, errlen) != 0) {
        return -1;
    }
    accepted[0] = first_token;
    return 1;
}

