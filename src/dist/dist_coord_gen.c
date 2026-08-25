/* dist_coord_gen.c — 机械拆自 ds4_distributed.c: 单机 coordinator 生成工具(One-Shot Coordinator Generation Utilities)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * One-Shot Coordinator Generation Utilities
 * ========================================================================= */

bool dist_prompt_is_rendered_chat(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

static bool dist_json_utf8_valid(const char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        int need = 0;
        if ((c & 0xe0) == 0xc0) need = 2;
        else if ((c & 0xf0) == 0xe0) need = 3;
        else if ((c & 0xf8) == 0xf0) need = 4;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i + 1u];
        if ((c1 & 0xc0) != 0x80) return false;
        if (need == 2 && c < 0xc2) return false;
        if (need == 3 && c == 0xe0 && c1 < 0xa0) return false;
        if (need == 3 && c == 0xed && c1 >= 0xa0) return false;
        if (need == 4 && c == 0xf0 && c1 < 0x90) return false;
        if (need == 4 && c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 2; j < need; j++) {
            if ((((unsigned char)s[i + (size_t)j]) & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void dist_json_write_string(FILE *fp, const char *s, size_t n) {
    const bool valid_utf8 = dist_json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void dist_json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    dist_json_write_string(fp, text ? text : "", text ? n : 0);
    fputs(",\"bytes\":[", fp);
    if (text) {
        for (size_t i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
        }
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

int dist_write_logits_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        const ds4_dist_route_plan *plan,
        const float *logits) {
    FILE *fp = fopen(gen->dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(state->engine);
    const int argmax = dist_logits_argmax(logits, vocab);
    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"vocab\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"argmax_token\":",
            ds4_engine_routed_quant_bits(state->engine),
            prompt->len,
            gen->ctx_size,
            vocab,
            plan->count);
    dist_json_write_token(fp, state->engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }
    return 0;
}

static int dist_logits_top_logprobs(const float *logits, int vocab, ds4_dist_logprob *scores, int k) {
    if (k <= 0) return 0;
    for (int i = 0; i < k; i++) {
        scores[i].id = -1;
        scores[i].logit = -FLT_MAX;
        scores[i].logprob = -FLT_MAX;
    }

    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i]) && logits[i] > max_logit) max_logit = logits[i];
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i])) sum += exp((double)logits[i] - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);

    int n = 0;
    for (int i = 0; i < vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == k && v <= scores[k - 1].logit) continue;
        int pos = n < k ? n++ : k - 1;
        while (pos > 0 && v > scores[pos - 1].logit) {
            scores[pos] = scores[pos - 1];
            pos--;
        }
        scores[pos].id = i;
        scores[pos].logit = v;
        scores[pos].logprob = (float)((double)v - logsum);
    }
    return n;
}

int dist_coordinator_rebuild_from_transcript(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        ds4_dist_route_plan *plan,
        const ds4_tokens *transcript,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        uint64_t *plan_generation,
        bool forget_route,
        char *err,
        size_t errlen) {
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: replaying %d tokens after distributed %s\n",
                     transcript->len,
                     forget_route ? "route failure" : "KV mismatch");
    if (forget_route) {
        fprintf(stderr, "ds4: [diag] distributed root failure (err at forget): %s\n", err);
        dist_coordinator_forget_route_workers(state, plan);
        dist_route_plan_free(plan);
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    } else if (plan->count == 0) {
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    }
    if (dist_coordinator_prefill_prompt(state,
                                        session,
                                        plan,
                                        transcript,
                                        session_id,
                                        request_id,
                                        logits,
                                        err,
                                        errlen) != 0) {
        return 1;
    }
    return 0;
}

int dist_write_logprobs_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        ds4_dist_route_plan *plan,
        ds4_session *session,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits) {
    FILE *fp = fopen(gen->dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        return 1;
    }

    int k = gen->dump_logprobs_top_k > 0 ? gen->dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_dist_logprob *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        return 1;
    }

    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, prompt);

    int max_tokens = gen->n_predict;
    int room = gen->ctx_size - prompt->len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"top_k\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"steps\":[\n",
            prompt->len,
            gen->ctx_size,
            k,
            plan->count);

    char err[256];
    int rc = 0;
    const int eos = ds4_token_eos(state->engine);
    for (int generated = 0; generated < max_tokens; generated++) {
        const int n = dist_logits_top_logprobs(logits, ds4_engine_vocab_size(state->engine), scores, k);
        const int token = dist_logits_argmax(logits, ds4_engine_vocab_size(state->engine));
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        dist_json_write_token(fp, state->engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            dist_json_write_token(fp, state->engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == eos) break;
        const uint32_t token_pos = (uint32_t)prompt->len + (uint32_t)generated;
        ds4_tokens_push(&transcript, token);
        if (dist_coordinator_eval_span(state, session, plan,
                                       &token, 1, token_pos,
                                       session_id, (*request_id)++,
                                       false, logits, NULL, err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed decode failed while dumping logprobs: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         plan,
                                                         &transcript,
                                                         session_id,
                                                         request_id,
                                                         logits,
                                                         NULL,
                                                         true,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed recovery failed while dumping logprobs: %s\n",
                        err);
                rc = 1;
                break;
            }
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        rc = 1;
    }
    ds4_tokens_free(&transcript);
    free(scores);
    return rc;
}

