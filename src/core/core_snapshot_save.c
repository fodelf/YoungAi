/* core_snapshot_save.c — payload 写盘/暂存 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    /* Sharded per-machine slice: layer 0 may belong to the other machine, so
     * probe the first routed-expert layer this slice actually holds. The routed
     * quant is uniform across layers, so any present layer reports the profile. */
    const ds4_tensor *gate = NULL;
    for (uint32_t il = 0; il < DS4_N_LAYER && !gate; il++) {
        gate = e->weights.layer[il].ffn_gate_exps;
    }
    /* 合一 VQ GGUF: gate 死重不入文件, routed 源=内嵌 blob(等效 go1b 档) → 报 2。 */
    if (!gate) return (e->model.residual && e->model.residual->present) ? 2 : 0;
    return gate->type == DS4_TENSOR_Q4_K ? 4 : 2;
}

/* 磁盘 KV 兼容键: routed 专家张量【类型码】, 不是位宽。位宽口径把 IQ2_XXS/Q2_K/
 * GO1B/GO2B 全塌成 2 —— 两个不同 2-bit 档 GGUF(如 allq2 与 vq86h)存的 KV 互相
 * 通过校验被复用, 而 KV 数值由 routed 前向决定 = 静默数值污染。换类型码后旧缓存
 * (存 2/4)自然失配重算, 方向安全。展示/dist 协议仍走 bits 口径(那边是显示值)。 */
int ds4_engine_routed_kv_key(ds4_engine *e) {
    const ds4_tensor *gate = NULL;
    for (uint32_t il = 0; il < DS4_N_LAYER && !gate; il++) {
        gate = e->weights.layer[il].ffn_gate_exps;
    }
    if (!gate) return (e->model.residual && e->model.residual->present) ? 254 : 0;   /* 254=合一 VQ blob 档 */
    return (int)gate->type;
}

/* Mode P / Mode G dynamic routing brain: a cheap, model-free heuristic that
 * classifies a user prompt as a programming task (-> resident programming model,
 * Mode P) vs everyday chat (-> full cached model, Mode G). Pure text signals so
 * the router decides BEFORE any model is loaded. Returns true for programming. */
bool ds4_prompt_is_programming(const char *prompt) {
    if (!prompt) return false;
    const size_t len = strlen(prompt);
    if (len == 0) return false;
    int score = 0;
    if (strstr(prompt, "```")) score += 6;                 /* fenced code block */
    /* Strong code signals (a single one already routes to Mode P). */
    static const char *const kw[] = {
        "def ", "function ", "class ", "import ", "return ", "public ", "private ",
        "void ", "const ", "let ", "var ", "func ", "struct ", "#include", "println",
        "console.log", "printf", "std::", "() {", ");", "=>", "->", "elif ",
        "async ", "await ", "lambda", "useState", "useEffect", "self.", "this.",
        "</", "/>", "@app", "SELECT ", "npm ", "git ", NULL};
    for (int i = 0; kw[i]; i++) if (strstr(prompt, kw[i])) score += 3;
    static const char *const ext[] = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cpp",
        ".sh", ".sql", ".html", ".css", ".json", ".yaml", NULL};
    for (int i = 0; ext[i]; i++) if (strstr(prompt, ext[i])) score += 3;
    /* Programming-intent words (EN + 中文 + frameworks). +2 each. */
    static const char *const verb[] = {
        "implement", "debug", "refactor", "compile", "stack trace", "exception",
        "syntax", "runtime", "API", "React", "Vue", "Python", "JavaScript",
        "TypeScript", "Golang", "Rust", "函数", "代码", "编译", "报错", "算法",
        "重构", "变量", "数组", "循环", "接口", "调试", "返回值", "递归", "指针",
        "编程", "脚本", "排序", "组件", "登录", "数据库", "框架", "前端", "后端",
        "正则", "并发", "异步", "类型", "继承", "封装", "bug", "方法", "对象",
        "装饰器", "闭包", "泛型", "多态", "线程", "进程", "队列", "哈希", "迭代器",
        "生成器", "协程", "序列化", "指令", "编译器", "解释器", "字节码", NULL};
    for (int i = 0; verb[i]; i++) if (strstr(prompt, verb[i])) score += 2;
    /* Symbol density: code is punctuation-heavy relative to prose. */
    size_t sym = 0;
    for (size_t i = 0; i < len; i++) {
        switch (prompt[i]) {
            case '{': case '}': case ';': case '(': case ')': case ':':
            case '[': case ']': case '<': case '>': case '=': sym++; break;
            default: break;
        }
    }
    if (sym * 100u / len >= 8u) score += 3;                 /* >= 8% symbols */
    /* Everyday prompts carry zero of these signals, so a low bar is safe. */
    return score >= 3;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}

#ifndef DS4_NO_GPU
typedef struct {
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    uint32_t mtp_n_raw;
} ds4_spec_frontier;

#endif

uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    if (s->distributed) return 0;
    if (ds4_session_is_cpu(s)) {
        uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += session_cpu_payload_live_tensor_bytes(s);
        return bytes;
    }
#ifdef DS4_NO_GPU
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
#endif
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = ds4_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_save_payload(s->distributed, s, fp, err, errlen);
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t raw_live = session_cpu_raw_live_rows(s);
        const uint32_t raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t comp_cap = session_cpu_comp_cap(s);
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            raw_cap,
            raw_cap,
            comp_cap,
            (uint32_t)s->checkpoint.len,
            DS4_N_LAYER,
            DS4_N_HEAD_DIM,
            DS4_N_INDEXER_HEAD_DIM,
            DS4_N_VOCAB,
            raw_live,
        };
        for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (int i = 0; i < s->checkpoint.len; i++) {
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_index_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (raw_live > layer->n_raw) {
                payload_set_err(err, errlen, "CPU session raw cache has fewer live rows than checkpoint");
                return 1;
            }
            const uint32_t raw_start = layer->n_raw - raw_live;
            if (payload_write_bytes(fp,
                                    layer->raw_kv + (uint64_t)raw_start * DS4_N_HEAD_DIM,
                                    (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            if (payload_write_bytes(fp,
                                    layer->attn_comp_kv,
                                    (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (ratio == 4) {
                if (payload_write_bytes(fp,
                                        layer->index_comp_kv,
                                        (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                        err,
                                        errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
            }
        }
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_write_tensor_span_f16_as_f32(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

