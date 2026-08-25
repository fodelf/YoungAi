/* core_session_sample.c — 采样面 setter/logprob/eval 内核 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

/* ---- Sampling-lane / per-request policy (contract in ds4.h) ----
 * Pure session-state setters/getters; the semantics live in
 * repeat_penalize_buf / session_penalties_active above. All NULL-tolerant:
 * frontends call them unconditionally on paths where the session may not
 * exist yet. */
void ds4_session_set_lane(ds4_session *s, int lane) {
    if (!s) return;
    /* Unknown lane ids fall back to FREE (penalties active): the safe default
     * is current behavior, not an accidental penalty bypass. */
    if (lane != DS4_LANE_TOOL_SYNTAX && lane != DS4_LANE_COPY_EMISSION)
        lane = DS4_LANE_FREE;
    s->lane = lane;
}

int ds4_session_lane(const ds4_session *s) {
    return s ? s->lane : DS4_LANE_FREE;
}

/* Pin the generation boundary at the live checkpoint length: call after the
 * prompt is fully prefilled, before the first sampled token of a response. */
void ds4_session_mark_generation_start(ds4_session *s) {
    if (!s) return;
    s->repeat_gen_start = s->checkpoint.len;
}

/* Explicit boundary for rebuilds whose checkpoint already contains generated
 * tokens (server re-sync of a transcript with a known assistant tail). */
void ds4_session_set_generation_start(ds4_session *s, int pos) {
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->repeat_gen_start = pos;
}

void ds4_session_set_request_penalties(ds4_session *s, float freq, float presence) {
    if (!s) return;
    s->req_freq = freq;
    s->req_presence = presence;
}

void ds4_session_set_spec_greedy(ds4_session *s, int greedy_ok) {
    if (!s) return;
    s->spec_greedy = greedy_ok ? 1 : 0;
}

int ds4_session_spec_greedy_ok(const ds4_session *s) {
    return s ? s->spec_greedy : 1;
}

static void session_apply_repeat_penalty(ds4_session *s) {
    if (!s || !s->logits || !s->checkpoint_valid || s->checkpoint.len <= 0) return;
    repeat_penalize_buf(s, s->logits, (uint32_t)s->checkpoint.len);
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (getenv("DS4_DECODE_DIAG")) {
        /* [decode-diag] inspect the logits the sampler is about to draw from:
         * argmax + its value + how many entries are non-finite (NaN/inf => the
         * distributed worker returned garbage logits rather than a real head). */
        int argmax = 0;
        float amv = s->logits[0];
        uint32_t nonfinite = 0;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
            const float v = s->logits[i];
            if (!isfinite(v)) { nonfinite++; continue; }
            if (v > amv) { amv = v; argmax = (int)i; }
        }
        fprintf(stderr,
                "ds4: [decode-diag] logits argmax=%d val=%.4f nonfinite=%u/%u\n",
                argmax, amv, nonfinite, (unsigned)DS4_N_VOCAB);
    }
    session_apply_repeat_penalty(s);
    return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)DS4_N_VOCAB) return 0;

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : DS4_NEG_INF;
    return 1;
}

int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    if (!s || !out || cap < (int)DS4_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)DS4_N_VOCAB * sizeof(out[0]));
    return (int)DS4_N_VOCAB;
}

int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    if (!s || !logits || n != (int)DS4_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}

int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
    if (!s) return 1;
    if (s->distributed) {
        if (!s->checkpoint_valid) {
            if (errlen) snprintf(err, errlen, "distributed decode requires a valid checkpoint");
            return 1;
        }
        (void)probe_mtp;
        return ds4_dist_session_eval(s->distributed,
                                     s,
                                     &s->checkpoint,
                                     token,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                 &e->model,
                                                 &e->weights,
                                                 &s->cpu_cache,
                                                 token,
                                                 (uint32_t)s->checkpoint.len,
                                                 e->directional_steering_dirs,
                                                 e->directional_steering_attn_scale,
                                                 e->directional_steering_ffn_scale,
                                                 &s->cpu_scratch);
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        (void)probe_mtp;
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)token;
    (void)probe_mtp;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    /* MTP probe 已整族删除(2026-08-05)。 */
    (void)probe_mtp;
    if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    /* MTP draft 已整族删除(2026-08-05)。 */
    return 0;
#endif
}

