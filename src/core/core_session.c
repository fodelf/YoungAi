/* core_session.c — session 创建/释放/尾部访问器 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
}

int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

int ds4_engine_power(ds4_engine *e) {
    return e ? e->power_percent : 100;
}

/* R3-h multi-domain sidecar plugin: swap the corr between generations. Load
 * the NEW sidecar first so a bad path keeps the current domain intact. */
int ds4_engine_corr_switch(ds4_engine *e, const char *path) {
    if (!e) return -1;
    struct ds4_corr *next = NULL;
    if (path && path[0]) {
        next = corr_load(path, e->backend == DS4_BACKEND_METAL);
        if (!next) return -1;
    }
    if (e->model.corr) corr_free(e->model.corr);
    e->model.corr = next;
    return 0;
}

int ds4_engine_set_power(ds4_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}

int ds4_engine_layer_count(ds4_engine *e) {
    (void)e;
    return (int)DS4_N_LAYER;
}

uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer) {
    (void)e;
    if (layer >= DS4_N_LAYER) return 0;
    return ds4_layer_compress_ratio(layer);
}

uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) {
    (void)e;
    return (uint64_t)DS4_N_HC * DS4_N_EMBD;
}

int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return (int)DS4_MODEL_VARIANT;
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    ds4_mm_free(e->mm);
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    model_close(&e->model);
#ifndef DS4_NO_GPU
    ds4_gpu_cleanup();
#endif
    ds4_release_instance_lock();
    ds4_dist_tp_free(e->tp);
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}

/* TP run loops (ds4_distributed.c) reach the engine's peer link through this. */
ds4_dist_tp *ds4_engine_tp(ds4_engine *e) { return e ? e->tp : NULL; }

/* Server /v1/messages image blocks reach the modality registry through this. */
ds4_mm *ds4_engine_mm(ds4_engine *e) { return e ? e->mm : NULL; }

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (e->backend == DS4_BACKEND_CPU) {
        if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
            fprintf(stderr, "ds4: distributed coordinator sessions require the graph backend\n");
            return 1;
        }
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = e;
        session_reset_request_policy(s);
        s->ctx_size = ctx_size;
        s->prefill_cap = ds4_default_prefill_cap_for_prompt(ctx_size);
        kv_cache_init(&s->cpu_cache, (uint32_t)ctx_size, 0);
        cpu_decode_scratch_init(&s->cpu_scratch, (uint32_t)ctx_size);
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        *out = s;
        return 0;
    }
#ifdef DS4_NO_GPU
    return 1;
#else
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    session_reset_request_policy(s);
    s->ctx_size = ctx_size;
    s->prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    /* --dist-prefill-cap: 分布式会话的 prefill 批上限覆盖(压住宽批的专家工作集)。 */
    if (e->distributed.prefill_cap > 0 &&
        e->distributed.role != DS4_DISTRIBUTED_NONE && e->distributed.layers.set &&
        e->distributed.prefill_cap <= (uint32_t)ctx_size) {
        s->prefill_cap = e->distributed.prefill_cap;
    }
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    bool active_slice = false;
    uint32_t active_start = 0;
    uint32_t active_end = (uint32_t)DS4_N_LAYER - 1u;
    if (e->distributed.role != DS4_DISTRIBUTED_NONE && e->distributed.layers.set) {
        active_slice = true;
        active_start = e->distributed.layers.start;
        active_end = e->distributed.layers.has_output ?
                     ((uint32_t)DS4_N_LAYER - 1u) : e->distributed.layers.end;
    }
    s->graph.dspark_capture = (e->dspark.ready && g_ds4_spec_enabled) ? 1 : 0;
    if (!metal_graph_alloc_raw_cap(&s->graph, &e->weights, &e->weights.layer[0],
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap, false,
                                   active_start, active_end, active_slice))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    s->graph.power_percent = (uint32_t)e->power_percent;
    if (e->distributed.tp_enabled) {
        /* Establish the TP peer link once per engine. By default the coordinator
         * listens and the worker connects. --reverse-connect flips the network
         * roles (coordinator connects, worker listens) to work around a host
         * where one direction's connect() fails (observed: an M1 where ds4's
         * outbound connect returns EHOSTUNREACH while nc/plain connect succeed).
         * The TP all-reduce is a symmetric sum, so connect direction does not
         * affect results; tp_owns_low stays tied to role, not to who listens.
         * Blocks until both peers are up; KB-level buffers only. */
        if (!e->tp) {
            char terr[256] = {0};
            bool reverse = e->distributed.reverse_connect;
            bool coordinator = (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR);
            bool i_listen = reverse ? !coordinator : coordinator;
            if (i_listen) {
                e->tp = ds4_dist_tp_listen(e->distributed.listen_host,
                                           e->distributed.listen_port, terr, sizeof(terr));
            } else {
                e->tp = ds4_dist_tp_connect(e->distributed.coordinator_host,
                                            e->distributed.coordinator_port, terr, sizeof(terr));
            }
            e->tp_owns_low = coordinator;
            if (!e->tp) {
                fprintf(stderr, "ds4: TP peer connection failed: %s\n", terr);
                metal_graph_free(&s->graph);
                free(s);
                return 1;
            }
        }
        s->graph.tp = e->tp;
        s->graph.tp_layers = e->distributed.tp_layers ? e->distributed.tp_layers : UINT32_MAX;
        s->graph.tp_owns_low = e->tp_owns_low;
        s->graph.tp_vec = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    }
    if (!metal_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        metal_graph_free(&s->graph);
        free(s);
        return 1;
    }
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR && !e->distributed.tp_enabled) {
        char err[256];
        if (ds4_dist_session_create(&s->distributed,
                                    e,
                                    &e->distributed,
                                    s,
                                    ctx_size,
                                    err,
                                    sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: failed to create distributed coordinator session: %s\n",
                    err[0] ? err : "unknown error");
            metal_graph_free(&s->graph);
            free(s->logits);
            free(s->mtp_logits);
            free(s);
            return 1;
        }
    }
    *out = s;
    return 0;
#endif
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    ds4_dist_session_free(s->distributed);
    if (ds4_session_is_cpu(s)) {
        kv_cache_free(&s->cpu_cache);
        cpu_decode_scratch_free(&s->cpu_scratch);
    }
#ifndef DS4_NO_GPU
    else {
        metal_graph_free(&s->graph);
    }
#endif
    token_vec_free(&s->checkpoint);
    free(s->logits);
    free(s->mtp_logits);
    free(s);
}

int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen) {
    if (!s || !s->distributed) {
        if (errlen) snprintf(err, errlen, "session is not a distributed coordinator");
        return -1;
    }
    return ds4_dist_session_route_ready(s->distributed, err, errlen);
}

int ds4_session_power(ds4_session *s) {
    if (!s || !s->engine) return 100;
    return s->engine->power_percent;
}

bool ds4_session_is_distributed(ds4_session *s) {
    return s && s->distributed != NULL;
}

int ds4_session_set_power(ds4_session *s, int power_percent) {
    if (!s || !s->engine || power_percent < 1 || power_percent > 100) return 1;
    s->engine->power_percent = power_percent;
#ifndef DS4_NO_GPU
    if (!ds4_session_is_cpu(s)) s->graph.power_percent = (uint32_t)power_percent;
#endif
    return 0;
}

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total) {
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}

int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen) {
    if (!s) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    ds4_session_invalidate(s);
    if (ds4_session_is_cpu(s)) {
        session_cpu_reset_cache(s);
        return 0;
    }
#ifdef DS4_NO_GPU
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        if (errlen) snprintf(err, errlen, "%s layer-slice state reset failed",
                             ds4_backend_name(s->engine->backend));
        return 1;
    }
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen) {
    if (!s || !s->engine || !hidden_hc || n_tokens == 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid output-head hidden-state input");
        return 1;
    }

    ds4_engine *e = s->engine;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const float *last_hc = hidden_hc + (uint64_t)(n_tokens - 1u) * hc_dim;

    if (ds4_session_is_cpu(s)) {
        output_logits_one(logits, &e->model, &e->weights, last_hc);
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)e;
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_tensor_write(g->cur_hc,
                                   0,
                                   last_hc,
                                   hc_dim * sizeof(float)) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head(g,
                                                &e->model,
                                                &e->weights,
                                                e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(g->logits,
                                     0,
                                     logits,
                                     (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after output-head hidden-state failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s output-head hidden-state evaluation failed",
                             ds4_backend_name(e->backend));
        return 1;
    }
    return 0;
#endif
}

int ds4_session_slice_check_timeline(
        ds4_session *s,
        const int   *tokens,
        uint32_t     n_tokens,
        uint32_t     pos0,
        char        *err,
        size_t       errlen) {
    if (!s || !tokens || n_tokens == 0) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice token span");
        return 1;
    }
    const uint32_t ctx_size = (uint32_t)s->ctx_size;
    if (pos0 > (uint32_t)INT_MAX || n_tokens > (uint32_t)INT_MAX ||
        pos0 > ctx_size || n_tokens > ctx_size - pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice token span exceeds context");
        return 1;
    }
    if (!s->checkpoint_valid) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "layer-slice session needs reset before pos %u", pos0);
            return 1;
        }
        return 0;
    }
    if ((uint32_t)s->checkpoint.len != pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice KV position mismatch: have %d want %u",
                             s->checkpoint.len, pos0);
        return 1;
    }
    return 0;
}

DS4_MAYBE_UNUSED void ds4_session_slice_commit_timeline(ds4_session *s, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
}

void ds4_session_invalidate(ds4_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    /* Also drops lane/request-penalty/spec-greedy back to defaults (and
     * repeat_gen_start to -1): an invalidated checkpoint means the next
     * request re-renders and re-declares its policy from scratch. */
    session_reset_request_policy(s);
    s->mtp_draft_valid = false;
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
    /* 冷回卷(pos==0)= 从头重放: 必须连 comp/indexer 计数与压缩器累积 state 一起清。
     * 此前只截 token 时间线 ⇒ 连跑多题时 layer_n_comp 只涨不回, 2050 行容量在
     * ~28 题后溢出("compressed KV cache capacity exceeded", 2026-08-18 328 题
     * 基准 500 连锁的第一层根因)。部分回卷(pos>0)的 comp 精确回滚仍是已知债
     * (state 含非边界脏贡献), CC 增量场景语义不变。 */
#ifndef DS4_NO_GPU
    if (pos == 0) (void)metal_graph_reset_prefill_state(&s->graph);
#endif
}

int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}

int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}

int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}

