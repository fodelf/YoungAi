/* dist_coord_recover.c — 机械拆自 ds4_distributed.c: 独立 coordinator 恢复与生成(Standalone Coordinator Recovery And Generation)。行为零变化。 */
#include "dist_internal.h"

/* =========================================================================
 * Standalone Coordinator Recovery And Generation
 * ========================================================================= */

static int dist_replay_check_logits(
        ds4_dist_coordinator_state *state,
        const float *before,
        const float *after) {
    const int vocab = ds4_engine_vocab_size(state->engine);
    float max_abs = 0.0f;
    int max_i = 0;
    uint32_t mismatches = 0;
    for (int i = 0; i < vocab; i++) {
        if (before[i] != after[i]) {
            const float d = fabsf(before[i] - after[i]);
            if (d > max_abs) {
                max_abs = d;
                max_i = i;
            }
            mismatches++;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr,
                "ds4: distributed replay check failed: mismatches=%u max_abs=%g token=%d before=%g after=%g\n",
                mismatches,
                max_abs,
                max_i,
                before[max_i],
                after[max_i]);
        return 1;
    }
    fprintf(stderr, "ds4: distributed replay check passed: logits exact match across reset/replay\n");
    return 0;
}

int dist_run_coordinator_generation(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen) {
    char err[256];
    ds4_dist_route_plan plan;
    uint64_t plan_generation = 0;
    if (!dist_coordinator_ensure_route(state, &plan, &plan_generation, err, sizeof(err))) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, state->engine, gen->ctx_size) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: failed to create local session\n");
        dist_route_plan_free(&plan);
        return 1;
    }

    ds4_tokens prompt = {0};
    if (dist_prompt_is_rendered_chat(gen->prompt)) {
        ds4_tokenize_rendered_chat(state->engine, gen->prompt, &prompt);
    } else {
        ds4_encode_chat_prompt(state->engine, gen->system, gen->prompt, gen->think_mode, &prompt);
    }
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4: distributed coordinator: empty prompt\n");
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    const uint64_t session_id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    uint64_t request_id = 1;
    float *logits = malloc((size_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ds4: distributed coordinator: out of memory allocating logits\n");
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     err,
                                                     sizeof(err));
    if (prefill_rc != 0) {
        fprintf(stderr,
                "ds4: distributed prompt processing failed: %s\n",
                err);
        if (dist_coordinator_rebuild_from_transcript(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     NULL,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed prompt recovery failed: %s\n",
                    err);
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (state->replay_check) {
        const size_t logits_bytes = (size_t)ds4_engine_vocab_size(state->engine) * sizeof(logits[0]);
        float *before = malloc(logits_bytes);
        if (!before) {
            fprintf(stderr, "ds4: distributed replay check: out of memory allocating logits copy\n");
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
        memcpy(before, logits, logits_bytes);
        int replay_prefill_rc = dist_coordinator_prefill_prompt(state,
                                                                session,
                                                                &plan,
                                                                &prompt,
                                                                session_id,
                                                                &request_id,
                                                                logits,
                                                                err,
                                                                sizeof(err));
        if (replay_prefill_rc != 0) {
            fprintf(stderr,
                    "ds4: distributed replay prompt processing failed: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &prompt,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         replay_prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed replay recovery failed: %s\n",
                        err);
                free(before);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
        int replay_rc = dist_replay_check_logits(state, before, logits);
        free(before);
        if (replay_rc != 0) {
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (gen->dump_logits_path) {
        int rc = dist_write_logits_dump(state, gen, &prompt, &plan, logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }
    if (gen->dump_logprobs_path) {
        int rc = dist_write_logprobs_dump(state,
                                          gen,
                                          &prompt,
                                          &plan,
                                          session,
                                          session_id,
                                          &request_id,
                                          logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }

    int generated = 0;
    int max_tokens = gen->n_predict > 0 ? gen->n_predict : 1;
    int room = gen->ctx_size - prompt.len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    uint64_t rng = gen->seed ? gen->seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    const int eos = ds4_token_eos(state->engine);
    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, &prompt);
    /* 生成区边界: prefill 刚落完 prompt, 在本地会话上钉住 —— 会话侧惩罚窗从此
     * 只数生成区(本循环的惩罚由下面的显式调用承担, 不走会话采样器)。 */
    ds4_session_mark_generation_start(session);
    while (generated < max_tokens) {
        /* 裸模型真值: 引擎自造的 repeat/anticycle 惩罚整族已删(2026-08-21 铁律),
         * 这里直接采模型原始 logits。 */
        int token = ds4_sample_logits(logits,
                                      ds4_engine_vocab_size(state->engine),
                                      gen->temperature,
                                      0,
                                      gen->top_p,
                                      gen->min_p,
                                      &rng);
        if (token == eos) break;

        size_t len = 0;
        char *text = ds4_token_text(state->engine, token, &len);
        if (len) fwrite(text, 1, len, stdout);
        fflush(stdout);
        free(text);

        uint32_t token_pos = (uint32_t)prompt.len + (uint32_t)generated;
        generated++;
        ds4_tokens_push(&transcript, token);
        if (generated >= max_tokens) break;

        int decode_rc = dist_coordinator_eval_span(state, session, &plan,
                                                   &token, 1, token_pos,
                                                   session_id, request_id++,
                                                   false, logits, NULL, err, sizeof(err));
        if (decode_rc != 0) {
            fprintf(stderr, "\nds4: distributed decode failed: %s\n", err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &transcript,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         decode_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr, "ds4: distributed decode recovery failed: %s\n", err);
                ds4_tokens_free(&transcript);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
            /* rebuild 重放了含已生成 token 的 transcript, 会话 checkpoint 尾已
             * 越过生成起点 —— 显式恢复边界到 prompt 长度。 */
            ds4_session_set_generation_start(session, prompt.len);
        }
    }
    fputc('\n', stdout);

    ds4_tokens_free(&transcript);
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    dist_route_plan_free(&plan);
    return 0;
}

