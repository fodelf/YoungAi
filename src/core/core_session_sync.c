/* core_session_sync.c — session sync/重写/惩罚核 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        p->session->repeat_gen_start = -1;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->mtp_draft_valid = false;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}
#endif

/* Bring the live backend state to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (s->distributed) {
        const ds4_tokens *checkpoint = s->checkpoint_valid ? &s->checkpoint : NULL;
        return ds4_dist_session_sync(s->distributed,
                                     s,
                                     checkpoint,
                                     prompt,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint))
        {
            s->mtp_draft_valid = false;
            for (int i = s->checkpoint.len; i < prompt->len; i++) {
                forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                         &e->model,
                                                         &e->weights,
                                                         &s->cpu_cache,
                                                         prompt->v[i],
                                                         (uint32_t)s->checkpoint.len,
                                                         e->directional_steering_dirs,
                                                         e->directional_steering_attn_scale,
                                                         e->directional_steering_ffn_scale,
                                                         &s->cpu_scratch);
                token_vec_push(&s->checkpoint, prompt->v[i]);
                if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i + 1, prompt->len);
            }
            s->checkpoint_valid = true;
            return 0;
        }

        session_cpu_reset_cache(s);
        prefill_layer_major_cpu(s->logits,
                                &e->model,
                                &e->weights,
                                &s->cpu_cache,
                                prompt,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale);
        ds4_tokens_copy(&s->checkpoint, prompt);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", prompt->len, prompt->len);
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)prompt;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    const char *backend_name = ds4_backend_name(e->backend);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint))
    {
        s->mtp_draft_valid = false;
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = metal_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            ds4_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            ds4_session_progress_fn progress_fn =
                s->progress ? ds4_session_note_prefill_progress : NULL;
            bool ok = metal_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        progress_fn,
                                                        progress_fn ? &progress : NULL,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL);
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    if (s->prefill_cap < (uint32_t)prompt->len) {
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ds4_session_progress_fn progress_fn =
            s->progress ? ds4_session_note_prefill_progress : NULL;
        ok = metal_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         progress_fn, progress_fn ? &progress : NULL,
                                         s->display_progress,
                                         s->display_progress_ud);
    } else {
        ok = metal_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud);
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    return ds4_session_sync_internal(s, prompt, err, errlen);
}

/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}


int ds4_session_argmax(ds4_session *s) {
    /* Non-FREE lanes see raw logits by contract (ds4.h): skip the scratch
     * copy entirely -- tool-syntax/copy emission sit in the hot decode loop
     * and repeat_penalize_buf would be a no-op for them anyway. */
    if (s && s->lane != DS4_LANE_FREE)
        return sample_argmax(s->logits, DS4_N_VOCAB);
    if (session_penalties_active(s) && s && s->checkpoint_valid && s->checkpoint.len > 0) {
        /* penalized greedy must match ds4_session_sample(temp 0); work on a
         * scratch copy so diagnostic readers of s->logits stay unpolluted */
        static float *scratch = NULL;
        if (!scratch) scratch = xmalloc((size_t)DS4_MAX_VOCAB * sizeof(scratch[0]));
        memcpy(scratch, s->logits, (size_t)DS4_N_VOCAB * sizeof(scratch[0]));
        repeat_penalize_buf(s, scratch, (uint32_t)s->checkpoint.len);
        return sample_argmax(scratch, DS4_N_VOCAB);
    }
    return sample_argmax(s->logits, DS4_N_VOCAB);
}

/* Raw-logits argmax excluding one id -- deliberately penalty-free, unlike
 * ds4_session_argmax: its only caller today is ds4-bench forced continuation
 * (exclude EOS to keep generating), which wants the model's unmodified
 * second choice for stable speed measurement, not a sampling path. */
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    /* 并列/NaN 规则与 sample_argmax 同: 严格大于才替换(首个最大胜, NaN 永不选); 全 -inf/NaN 时兜底
     * 首个未排除位。CUDA 图末尾的 decode_argmax 核逐字镜像这条规则(预发射对账靠它)。 */
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    if (best < 0) best = (excluded_id == 0) ? 1 : 0;
    return best;
}

int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature, top_k, top_p, min_p, rng);
}

/* Session-aware penalty-activity gate: env-armed penalties (loop break /
 * DS4_REPEAT_FREQ) plus this session's per-request penalties -- a session
 * carrying only req_freq/req_presence must still take the penalized
 * argmax/anticycle-prep paths. Non-FREE lanes report inactive: the lane
 * contract (ds4.h) bypasses every penalty, so callers skip the scratch work
 * outright. */
/* 惩罚只剩客户端显式传的 per-request(OpenAI frequency/presence)。引擎自造的
 * env 惩罚(DS4_REPEAT_FREQ)、断环器、熵门整族已删(2026-08-21 用户铁律:
 * 引擎不得擅自修改模型输出, 默认路径=裸模型真值)。 */
int session_penalties_active(const ds4_session *s) {
    if (s && s->lane != DS4_LANE_FREE) return 0;
    return s && (s->req_freq != 0.0f || s->req_presence != 0.0f);
}

void repeat_penalize_buf(ds4_session *s, float *logits, uint32_t end) {
    if (!s || !logits || end == 0) return;
    /* Lane gate (ds4.h contract): non-FREE lanes get raw logits, no penalty
     * of any kind — a penalty-diverted token corrupts forced tool-call
     * syntax. */
    if (s->lane != DS4_LANE_FREE) return;
    /* Unmarked (-1) → 0 = whole context, NOT the prompt boundary (end).
     * Pinning to `end` excluded the prompt from the repeat window, which left
     * only the few generated tokens penalized early in generation → far too
     * weak for the fragile 2-bit model, so greedy decode drifted/looped. The
     * distributed coordinator's session starts invalidated
     * (repeat_gen_start=-1) and hit this path → dual-host drifted where
     * single-host (which stays at 0) wrote clean code (2026-07-06 root-cause:
     * dual "bug" was this single-vs-dist gen_start mismatch, not cross-GPU
     * fp). 0 keeps both paths identical for unmarked sessions. NEW (lane
     * era): frontends now mark the real boundary via
     * ds4_session_mark_generation_start / _set_generation_start, which scopes
     * the anticycle bans + request penalties below to the generated region so
     * quoting the prompt is never banned; the env freq window keeps ignoring
     * the boundary either way (see repeat_penalize_core). */
    if (s->repeat_gen_start < 0 || s->repeat_gen_start > (int)end)
        s->repeat_gen_start = 0;
    const uint32_t gstart = (uint32_t)s->repeat_gen_start;
    /* Per-request OpenAI penalties, stacked on top of the env freq penalty:
     * logits[t] -= req_freq*count(t) + (count(t)>0 ? req_presence : 0), with
     * count over the generated region only. Negative values are legal and
     * ADD probability (OpenAI [-2,2]). The `seen` scratch marks first
     * occurrences and is wiped by re-walking the same range, so cost stays
     * O(region), not O(vocab); single graph worker => static is safe (same
     * discipline as the ds4_session_argmax scratch). */
    if (s->req_freq != 0.0f || s->req_presence != 0.0f) {
        static uint8_t *seen = NULL;
        if (!seen) seen = xcalloc((size_t)DS4_MAX_VOCAB, sizeof(seen[0]));
        for (uint32_t i = gstart; i < end; i++) {
            const int t = s->checkpoint.v[i];
            if (t < 0 || t >= (int)DS4_N_VOCAB) continue;
            logits[t] -= s->req_freq;
            if (!seen[t]) { seen[t] = 1; logits[t] -= s->req_presence; }
        }
        for (uint32_t i = gstart; i < end; i++) {
            const int t = s->checkpoint.v[i];
            if (t >= 0 && t < (int)DS4_N_VOCAB) seen[t] = 0;
        }
    }
}
