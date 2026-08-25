/* core_engine_open.c — 拆分中间态: 批次尚未切出的切片仍并在本文件(自 ds4.c). */
#include "core_internal.h"


static int g_ds4_lock_fd = -1;


/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

/* 取料入口 setter 家族(ds4.h 同名注释): CLI 参数是对外入口, 进程内的唯一消费点
 * 目前仍是 DS4_CAP_DIR/DS4_EVAL_* 的 getenv 读点(散在 capture/终审仪器几处),
 * 所以 setter 落到 setenv——单一事实源不变, 旗标即时生效, 不造第二条配置路径。
 * (2026-08-22 env→CLI 迁移只 land 了 CLI 半边, setter 无实现曾链接失败;
 * 消费点集中化到进程内全局属 ds4.c 拆分工序, 见重构阶段4。) */
void ds4_tool_set_cap_dir(const char *p)     { if (p) setenv("DS4_CAP_DIR", p, 1); }
const char *ds4_tool_cap_dir(void)           { return getenv("DS4_CAP_DIR"); }
void ds4_tool_set_eval_ids(const char *p)    { if (p) setenv("DS4_EVAL_IDS", p, 1); }
const char *ds4_tool_eval_ids(void)          { return getenv("DS4_EVAL_IDS"); }
void ds4_tool_set_eval_hdump(const char *p)  { if (p) setenv("DS4_EVAL_HDUMP", p, 1); }
const char *ds4_tool_eval_hdump(void)        { return getenv("DS4_EVAL_HDUMP"); }
void ds4_tool_set_eval_logits(const char *p) { if (p) setenv("DS4_EVAL_LOGITS", p, 1); }
const char *ds4_tool_eval_logits(void)       { return getenv("DS4_EVAL_LOGITS"); }
void ds4_tool_set_eval_no_bos(int v)         { if (v) setenv("DS4_EVAL_NO_BOS", "1", 1); else unsetenv("DS4_EVAL_NO_BOS"); }

const char *ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_METAL: return "metal";
    case DS4_BACKEND_CUDA:  return "cuda";
    case DS4_BACKEND_CPU:   return "cpu";
    }
    return "unknown";
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE: return "none";
    case DS4_THINK_HIGH: return "high";
    case DS4_THINK_MAX:  return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return DS4_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}


/* Per-request sampling-policy defaults. Called at session creation AND from
 * ds4_session_invalidate: the server reuses one session across requests, so a
 * stale lane / request penalty / spec-greedy flag from the previous request
 * must never leak into the next (the ds4.h contract makes frontends
 * re-declare all of them per request). repeat_gen_start goes back to -1 =
 * unmarked, same as a checkpoint rebuild. */
void session_reset_request_policy(ds4_session *s) {
    s->lane = DS4_LANE_FREE;
    s->req_freq = 0.0f;
    s->req_presence = 0.0f;
    s->spec_greedy = 1;
    s->repeat_gen_start = -1;
}

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated Metal tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > DS4_SESSION_IO_CHUNK ? DS4_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

#ifndef DS4_NO_GPU
/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical Metal tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (ds4_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read Metal F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (ds4_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}
#endif

bool ds4_session_is_cpu(const ds4_session *s) {
    return s && s->engine && s->engine->backend == DS4_BACKEND_CPU;
}

uint32_t session_cpu_raw_live_rows(const ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    uint32_t rows = ds4_default_raw_cap((uint32_t)s->ctx_size);
    if (rows > (uint32_t)s->checkpoint.len) rows = (uint32_t)s->checkpoint.len;
    return rows;
}

uint32_t session_cpu_comp_cap(const ds4_session *s) {
    if (!s) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio == 4) return layer->comp_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio != 0) return layer->comp_cap;
    }
    return (uint32_t)s->ctx_size;
}

uint64_t session_cpu_payload_live_tensor_bytes(const ds4_session *s) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_cpu_raw_live_rows(s);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;
        bytes += (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

void session_cpu_reset_cache(ds4_session *s) {
    kv_cache_free(&s->cpu_cache);
    kv_cache_init(&s->cpu_cache, (uint32_t)s->ctx_size, 0);
}

static bool ds4_layer_payload_range_valid(uint32_t layer_start, uint32_t layer_end) {
    return layer_start <= layer_end && layer_end < (uint32_t)DS4_N_LAYER;
}

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end) {
    if (!s || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end))
        return 0;
    if (ds4_session_is_cpu(s)) return 0;
#ifdef DS4_NO_GPU
    (void)layer_start;
    (void)layer_end;
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint64_t bytes = (uint64_t)DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint32_t n_layers = layer_end - layer_start + 1u;
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
#endif
}

int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload save");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before layer snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint32_t header[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        layer_start,
        layer_end,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = layer_start; rc == 0 && il <= layer_end; il++) {
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

int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !tokens ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload load");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)payload_bytes;
    (void)n_tokens;
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session layer payload version");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_layer_start = h[11];
    const uint32_t saved_layer_end = h[12];
    const uint32_t saved_raw_live = h[13];
    (void)saved_prefill_cap;
    if (saved_layer_start != layer_start || saved_layer_end != layer_end) {
        payload_set_err(err, errlen, "KV shard layer range does not match requested worker");
        return 1;
    }
    if (saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens != n_tokens ||
        saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV shard does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM) {
        payload_set_err(err, errlen, "KV shard was written for a different DS4 layout");
        return 1;
    }
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV shard graph chunk layout does not match current runtime");
        return 1;
    }
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap) {
        payload_set_err(err, errlen, "KV shard raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV shard compressed cache is larger than current context");
        return 1;
    }

    const uint32_t n_layers = layer_end - layer_start + 1u;
    uint32_t *n_comp = xcalloc(n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = xcalloc(n_layers, sizeof(n_index_comp[0]));
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_comp[i] > saved_comp_cap || n_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_index_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_index_comp[i] > saved_comp_cap || n_index_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        free(n_comp);
        free(n_index_comp);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV shard restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                     g->layer_attn_comp_cache[il],
                                                     0,
                                                     (uint64_t)n_comp[i] * DS4_N_HEAD_DIM,
                                                     buf,
                                                     DS4_SESSION_IO_CHUNK,
                                                     &remaining,
                                                     err,
                                                     errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[i] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[i] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc == 0 && remaining != 0) {
        payload_set_err(err, errlen, "KV shard has trailing payload bytes");
        rc = 1;
    }
    if (rc == 0 && ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV shard restore");
        rc = 1;
    }
    if (rc == 0) {
        token_vec_free(&s->checkpoint);
        memset(&s->checkpoint, 0, sizeof(s->checkpoint));
        for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
        for (uint32_t i = 0; i < n_layers; i++) {
            const uint32_t il = layer_start + i;
            g->layer_n_comp[il] = n_comp[i];
            g->layer_n_index_comp[il] = n_index_comp[i];
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        g->mtp_n_raw = 0;
    }
    free(n_comp);
    free(n_index_comp);
    return rc;
#endif
}

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

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_load_payload(s->distributed, s, fp, payload_bytes, err, errlen);
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_raw_cap = h[4];
        const uint32_t saved_raw_window = h[5];
        const uint32_t saved_comp_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_raw_live = h[12];
        const uint32_t cpu_raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t cpu_comp_cap = session_cpu_comp_cap(s);
        if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint does not fit current context");
            return 1;
        }
        if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
            h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
        {
            payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
            return 1;
        }
        /* prefill_cap is scratch scheduling capacity, not durable KV layout.
         * Old checkpoints remain valid as long as the raw KV window matches. */
        (void)saved_prefill_cap;
        if (saved_raw_window != cpu_raw_cap) {
            payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
            return 1;
        }
        const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
        if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
            saved_raw_live > saved_raw_cap || saved_raw_live > cpu_raw_cap)
        {
            payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
            return 1;
        }
        if (saved_comp_cap > cpu_comp_cap) {
            payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0)
        {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        uint32_t n_comp[DS4_MAX_LAYER];
        uint32_t n_index_comp[DS4_MAX_LAYER];
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_comp[il] > saved_comp_cap || n_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
                return 1;
            }
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
                return 1;
            }
        }

        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        session_cpu_reset_cache(s);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (payload_read_bytes(fp,
                                   layer->raw_kv,
                                   (uint64_t)saved_raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            layer->n_raw = saved_raw_live;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            layer->n_comp = n_comp[il];
            layer->n_index_comp = n_index_comp[il];
            if (payload_read_bytes(fp,
                                   layer->attn_comp_kv,
                                   (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (ratio == 4) {
                if (payload_read_bytes(fp,
                                       layer->index_comp_kv,
                                       (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                       &remaining,
                                       err,
                                       errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0)
                {
                    token_vec_free(&new_checkpoint);
                    return 1;
                }
            }
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
            return 1;
        }
        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                     g->layer_attn_comp_cache[il],
                                                     0,
                                                     (uint64_t)n_comp[il] * DS4_N_HEAD_DIM,
                                                     buf,
                                                     DS4_SESSION_IO_CHUNK,
                                                     &remaining,
                                                     err,
                                                     errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

#ifndef DS4_NO_GPU
static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "ds4: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "ds4: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}

static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}
#endif

int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)dataset_path;
    (void)output_path;
    (void)ctx_size;
    (void)max_prompts;
    (void)max_tokens;
    fprintf(stderr, "ds4: imatrix collection requires a graph backend build\n");
    return 1;
#else
    if (!e || !dataset_path || !output_path) return 1;
    /* 收集器本身是后端无关的: 采集循环全在宿主侧, 回读只用 ds4_gpu_tensor_read
     * (Metal/CUDA 都实现), 三个采样张量 batch_ffn_norm / batch_router_selected /
     * batch_routed_mid 也都由共享宿主图物化。旧的 "requires --metal" 是 Mac 独占
     * 时代的遗留闸, 拆掉 (2026-08-21: 全 q2 基座第一次要在 CUDA 上吃语料量化)。 */
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) {
        fprintf(stderr, "ds4: imatrix collection requires a GPU graph backend (--metal / --cuda)\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false,
                                        0, (uint32_t)DS4_N_LAYER - 1u, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate imatrix graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;
    g.power_percent = (uint32_t)e->power_percent;

    ds4_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "ds4: failed to allocate imatrix collector\n");
        metal_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "ds4: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, DS4_MODEL_SHAPE_NAME, DS4_N_LAYER, DS4_N_EXPERT, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            ds4_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!metal_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "ds4: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = metal_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector);
                } else {
                    ok = metal_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
                if (!ok) {
                    fprintf(stderr, "ds4: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "ds4: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "ds4: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    metal_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (ds4_backend_uses_graph(e->backend)) {
#ifndef DS4_NO_GPU
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s generation requested but the graph backend is unavailable\n",
                    ds4_backend_name(e->backend));
            return 1;
        }
        return generate_metal_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->power_percent,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
#else
        fprintf(stderr, "ds4: %s generation requested but this build has no graph backend support\n",
                ds4_backend_name(e->backend));
        return 1;
#endif
    }

    return generate_raw_swa_cpu(model, vocab, weights, prompt, n_predict,
                                ctx_size,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale,
                                emit, done, emit_ud, progress, progress_ud);
}

int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_decode_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal full graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_first_token_full_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal full graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal prompt graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_prompt_logits_test(&e->model, &e->weights, prompt, ctx_size);
#else
    (void)e;
    (void)prompt;
    (void)ctx_size;
    fprintf(stderr, "ds4: Metal prompt graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: head test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = xmalloc((size_t)prompt->len * DS4_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, DS4_N_EMBD, prompt_embd);

    const uint32_t n_hc = DS4_N_HC;
    float *hc0 = xmalloc((size_t)DS4_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * DS4_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, DS4_N_EMBD);

    float *attn_norm0 = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    float *q0 = xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, DS4_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv0, DS4_N_HEAD_DIM);

    float *attn_heads = xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, DS4_N_EMBD);

    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, DS4_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *after_ffn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}

int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: first-token test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    float *hc = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(hc[0]));
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    forward_first_token_cpu(hc, model, weights, prompt->v[0]);
    print_vec_stats("first-token final_hc", hc, (uint64_t)DS4_N_HC * DS4_N_EMBD);
    output_logits_one(logits, model, weights, hc);
    print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after first-token whole-model CPU pass:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(hc);
    return 0;
}

#ifndef DS4_NO_GPU
/* project.md P2.1: register the locally-loaded routed layers' router metadata
 * (F16 gate_inp + optional exp_probs_b + expert tensor offsets) so the GPU
 * backend can re-evaluate the next layer's router on the CPU during decode and
 * read predicted experts ahead.  Only the local slice is registered: issuing
 * read-ahead for layers another machine executes would waste SSD bandwidth.
 * Purely advisory metadata; never affects inference results. */
void engine_register_layer_routers(ds4_engine *e, uint32_t start, uint32_t end) {
    if (e->model.expert_shrunken) return;   /* compact-slot models: ids differ */
    if (end >= DS4_MAX_LAYER) end = DS4_MAX_LAYER - 1;
    uint32_t registered = 0;
    for (uint32_t il = start; il <= end && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (!l->ffn_gate_exps) continue;   /* no routed MoE on this layer */
        const char *why = NULL;
        uint64_t hash_off = UINT64_MAX;
        uint32_t hash_k = 0, hash_rows = 0;
        if (!l->ffn_up_exps || !l->ffn_down_exps) why = "missing expert tensors";
        else if (!l->ffn_gate_inp) why = "no gate_inp";
        else if (l->ffn_gate_inp->type != DS4_TENSOR_F16 &&
                 l->ffn_gate_inp->type != DS4_TENSOR_F32) why = "gate_inp quantized";
        else if (l->ffn_gate_tid2eid) {
            /* Early layers hash-route by token id (tid2eid I32 [k][n_vocab]):
             * exact prediction, register the table. */
            const ds4_tensor *t = l->ffn_gate_tid2eid;
            if (t->type == DS4_TENSOR_I32 && t->ndim == 2 &&
                t->dim[0] == DS4_N_EXPERT_USED && t->dim[1] > 0) {
                hash_off = t->abs_offset;
                hash_k = (uint32_t)t->dim[0];
                hash_rows = (uint32_t)t->dim[1];
            } else {
                why = "unrecognized tid2eid layout";
            }
        }
        if (!l->ffn_gate_exps || !l->ffn_up_exps) continue;   /* 内嵌 VQ: blob 为源, 不注册 base 专家预取 */
        const uint64_t n_exp = l->ffn_gate_exps->dim[2];
        if (!why && (n_exp == 0 || n_exp != DS4_N_EXPERT)) why = "expert count";
        if (why) {
            fprintf(stderr,
                    "ds4: layer %u router not registered for prefetch (%s; gate_inp type=%u, "
                    "n_exp=%llu)\n",
                    il, why,
                    l->ffn_gate_inp ? l->ffn_gate_inp->type : 9999u,
                    (unsigned long long)n_exp);
            continue;
        }
        if (ds4_gpu_register_layer_router(e->model.map,
                                          il,
                                          l->ffn_gate_inp->abs_offset,
                                          l->ffn_gate_inp->type == DS4_TENSOR_F32,
                                          l->ffn_exp_probs_b ? l->ffn_exp_probs_b->abs_offset
                                                             : UINT64_MAX,
                                          l->ffn_gate_exps->abs_offset,
                                          l->ffn_up_exps->abs_offset,
                                          l->ffn_down_exps->abs_offset,
                                          l->ffn_gate_exps->bytes / n_exp,
                                          l->ffn_down_exps->bytes / n_exp,
                                          (uint32_t)l->ffn_gate_inp->dim[0],
                                          (uint32_t)n_exp,
                                          hash_off,
                                          hash_k,
                                          hash_rows)) {
            registered++;
        }
    }
    if (registered) {
        fprintf(stderr,
                "ds4: registered %u local routed layers for cross-layer expert prefetch\n",
                registered);
    }
}
#endif

/* ---- DS4_EVAL_IDS 终审仪器 -------------------------------------------------
 * 用途: 与量化器锚文件里同序列的 FP logits 直接对账(held 区 Σmin), 判"引擎口径
 * 还原率 vs 量化器口径"。差距大时再用 HDUMP 的逐层 hidden 找第一分歧层。
 *
 *   DS4_EVAL_IDS=<ids文件>     原始 token id 流, 空白分隔的十进制整数(每行一个亦可)
 *   DS4_EVAL_LOGITS=<out.bin>  每位置最终 logits, f32 [S][DS4_N_VOCAB], 顺序流式写盘
 *   DS4_EVAL_HDUMP=<dir>       每层出口 hidden → h_L%02d.bin(见 eval_hdump_batch_layer)
 *   DS4_EVAL_NO_BOS=1          不在流首插 BOS(默认插, 即"裸 BOS 起")
 *
 * 走的是分布式推理那条已验证的裸 token 通道(ds4_session_eval_layer_slice), 天然不过
 * chat 模板/DSML, 与量化器 embed(ids) 直喂同口径。跑完 exit(0), 不采样。
 * 内存: logits 每次只留一个位置(VOCAB f32 ≈ 0.5 MB), 写一行冲一行, 不驻留 S×VOCAB。 */
static int *eval_ids_load(const char *path, uint32_t *out_n, uint32_t vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ds4: [EVAL_IDS] 打不开 %s -- aborting\n", path);
        exit(1);
    }
    uint32_t cap = 4096, n = 0;
    int *ids = xmalloc((size_t)cap * sizeof(int));
    long v;
    while (fscanf(f, "%ld", &v) == 1) {
        if (v < 0 || (uint64_t)v >= (uint64_t)vocab) {
            fprintf(stderr, "ds4: [EVAL_IDS] %s 第 %u 个 id=%ld 越界 [0,%u) -- aborting\n",
                    path, n, v, vocab);
            exit(1);
        }
        if (n == cap) {
            cap *= 2;
            int *nb = realloc(ids, (size_t)cap * sizeof(int));
            if (!nb) { fprintf(stderr, "ds4: [EVAL_IDS] ids 扩容失败\n"); exit(1); }
            ids = nb;
        }
        ids[n++] = (int)v;
    }
    fclose(f);
    if (!n) {
        fprintf(stderr, "ds4: [EVAL_IDS] %s 里没有解析出任何 id -- aborting\n", path);
        exit(1);
    }
    *out_n = n;
    return ids;
}

/* 并发批实测(DS4_MULTI_BENCH=N, 2026-08-21): 建 N 个会话各喂不同 prompt, 先各自
 * 单独解码若干步做基准, 再用 ds4_session_eval_multi 批量推进同样的步数, 对比
 *   ①聚合 token/s(红利有多大)  ②每个会话选出的 token 序列是否与单独解码逐字相同(无损)。
 * 用 DS4_MULTI_BENCH_STEPS 调步数(默认 48)。 */
void ds4_multi_bench_run(ds4_engine *e) {
    const char *env = getenv("DS4_MULTI_BENCH");
    if (!env || !env[0]) return;
    uint32_t n = (uint32_t)atoi(env);
    if (n < 2u && !getenv("DS4_MULTI_FORCE")) n = 2u;
    if (n < 1u) n = 1u;
    if (n > 8u) n = 8u;
    uint32_t steps = 48u;
    { const char *sv = getenv("DS4_MULTI_BENCH_STEPS"); if (sv && atoi(sv) > 0) steps = (uint32_t)atoi(sv); }
    if (steps > 500u) steps = 500u;
    /* 只跑批模式(跳过单路基准): 演示 8 路并发时不必等基准 */
    const int batch_only = getenv("DS4_MULTI_BENCH_BATCH_ONLY") != NULL;

    /* 8 道真代码题(并发批处理演示用): 各自独立、长度相近, 便于横向比较 */
    static const char *prompts[8] = {
        "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Return only the code.",
        "Write a Go function `LRUCache` with Get and Put in O(1). Return only the code.",
        "Write a C function that reverses a singly linked list in place. Return only the code.",
        "Write a SQL query that returns each department's second-highest salary. Return only the query.",
        "Write a Python function `binary_search(arr, target)` returning the index or -1. Return only the code.",
        "Write a bash script that finds the 10 largest files under a directory. Return only the script.",
        "Write a JavaScript function `debounce(fn, wait)` and explain nothing. Return only the code.",
        "Write a Rust function that counts word frequencies in a string. Return only the code.",
    };
    char err[256];
    ds4_session *ss[8] = {0};
    ds4_tokens toks[8];
    memset(toks, 0, sizeof(toks));
    for (uint32_t i = 0; i < n; i++) {
        if (ds4_session_create(&ss[i], e, 4096) != 0 || !ss[i]) {
            fprintf(stderr, "ds4: [multi-bench] 会话 %u 创建失败\n", i); exit(1);
        }
        { uint32_t pi = i;
          const char *sv = getenv("DS4_MULTI_BENCH_START");
          if (sv) pi = ((uint32_t)atoi(sv) + i) & 7u;
          const char *ptext = getenv("DS4_MULTI_BENCH_SAMEPROMPT") ? prompts[0] : prompts[pi];
          /* 与 CLI 同一条渲染路: 走 chat 模板才是正经问答, 裸文本只会续写题面 */
          if (getenv("DS4_MULTI_BENCH_RAW")) ds4_tokenize_text(e, ptext, &toks[i]);
          else ds4_encode_chat_prompt(e, NULL, ptext, DS4_THINK_NONE, &toks[i]); }
        if (ds4_session_sync(ss[i], &toks[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 会话 %u prefill 失败: %s\n", i, err); exit(1);
        }
    }
    /* 基准: 各会话单独逐 token 解码 steps 步, 记下选出的 token */
    int (*ref)[512] = xmalloc((size_t)n * sizeof(*ref));
    float *ref_l1 = xmalloc((size_t)n * DS4_N_VOCAB * sizeof(float));   /* 第 1 步后的 logits */
    int (*bat)[512] = xmalloc((size_t)n * sizeof(*bat));
    double t0 = now_sec();
    for (uint32_t i = 0; !batch_only && i < n; i++) {
        for (uint32_t k = 0; k < steps; k++) {
            int best = 0; float bv = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (ss[i]->logits[v] > bv) { bv = ss[i]->logits[v]; best = (int)v; }
            ref[i][k] = best;
            if (ds4_session_eval(ss[i], best, err, sizeof err) != 0) {
                fprintf(stderr, "ds4: [multi-bench] 单路解码失败: %s\n", err); exit(1);
            }
            if (k == 0) memcpy(ref_l1 + (uint64_t)i * DS4_N_VOCAB, ss[i]->logits,
                               (size_t)DS4_N_VOCAB * sizeof(float));
        }
    }
    const double seq_s = now_sec() - t0;
    if (!batch_only)
        fprintf(stderr, "ds4: [multi-bench] 单路逐个跑: %u 会话 × %u token = %u, 用时 %.2fs ⇒ 聚合 %.2f t/s\n",
                n, steps, n * steps, seq_s, (double)(n * steps) / seq_s);

    /* 批: 重建会话到同一起点, 用 eval_multi 同步推进 */
    for (uint32_t i = 0; i < n; i++) {
        ds4_session_free(ss[i]);
        ss[i] = NULL;
        if (ds4_session_create(&ss[i], e, 4096) != 0 ||
            ds4_session_sync(ss[i], &toks[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 会话重建失败\n"); exit(1);
        }
    }
    uint32_t mismatch = 0, first_bad = 0;
    int cur[8];
    t0 = now_sec();
    for (uint32_t k = 0; k < steps; k++) {
        for (uint32_t i = 0; i < n; i++) {
            int best = 0; float bv = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (ss[i]->logits[v] > bv) { bv = ss[i]->logits[v]; best = (int)v; }
            cur[i] = best;
            bat[i][k] = best;
            if (best != ref[i][k]) { if (!mismatch) first_bad = k; mismatch++; }
        }
        if (ds4_session_eval_multi(ss, cur, n, err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [multi-bench] 批解码失败: %s\n", err); exit(1);
        }
        if (k == 0 && !batch_only) {   /* 第 1 步 logits 与单路对账 */
            for (uint32_t i = 0; i < n; i++) {
                const float *r = ref_l1 + (uint64_t)i * DS4_N_VOCAB;
                double dmax = 0.0; int ar = 0, ab = 0; float br = -1e30f, bb = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++) {
                    const double d = fabs((double)r[v] - (double)ss[i]->logits[v]);
                    if (d > dmax) dmax = d;
                    if (r[v] > br) { br = r[v]; ar = (int)v; }
                    if (ss[i]->logits[v] > bb) { bb = ss[i]->logits[v]; ab = (int)v; }
                }
                fprintf(stderr, "ds4: [multi-bench] 会话%u(位置%d) 第1步: max|Δlogit|=%.4g 单路top1=%d 批top1=%d %s\n",
                        i, ss[i]->checkpoint.len, dmax, ar, ab, ar == ab ? "(一致)" : "(不同)");
            }
        }
    }
    const double bat_s = now_sec() - t0;
    fprintf(stderr, "ds4: [multi-bench] %u 路批处理: %u token, 用时 %.2fs ⇒ 聚合 %.2f t/s%s\n",
            n, n * steps, bat_s, (double)(n * steps) / bat_s,
            batch_only ? "" : "");
    if (!batch_only)
        fprintf(stderr, "ds4: [multi-bench] 相对单路加速 %.2fx\n", seq_s / bat_s);
    /* 质量目视: 两种模式各自的文本(逐位不同是数值等价的正常结果, 关键看是否连贯) */
    if (getenv("DS4_MULTI_BENCH_TEXT")) {
        for (uint32_t i = 0; i < n; i++) {
            static char buf[16384]; size_t off = 0;
            for (uint32_t k = 0; k < steps && off < sizeof(buf) - 64; k++) {
                size_t l = 0;
                const char *p = ds4_token_text(e, ref[i][k], &l);
                if (p && off + l < sizeof(buf) - 1) { memcpy(buf + off, p, l); off += l; }
            }
            buf[off] = 0;
            if (!batch_only) fprintf(stderr, "ds4: [multi-bench] 会话%u 单路: %s\n", i, buf);
            off = 0;
            for (uint32_t k = 0; k < steps && off < sizeof(buf) - 64; k++) {
                size_t l = 0;
                const char *p = ds4_token_text(e, bat[i][k], &l);
                if (p && off + l < sizeof(buf) - 1) { memcpy(buf + off, p, l); off += l; }
            }
            buf[off] = 0;
            fprintf(stderr, "\n===== 会话 %u =====\n%s\n", i, buf);
        }
    }
    if (!batch_only) fprintf(stderr, "ds4: [multi-bench] 与单路逐位对照: %s (不同 %u/%u%s)\n",
            mismatch ? "有差异" : "完全一致", mismatch, n * steps,
            mismatch ? "" : ", 无损");
    if (mismatch && !batch_only) fprintf(stderr, "ds4: [multi-bench] 首个不同在第 %u 步\n", first_bad);
    for (uint32_t i = 0; i < n; i++) ds4_session_free(ss[i]);
    free(ref);
    exit(0);
}

void ds4_eval_ids_run(ds4_engine *e) {
    const char *idp = getenv("DS4_EVAL_IDS");
    if (!idp || !idp[0]) return;
#ifdef DS4_NO_GPU
    (void)e;
    fprintf(stderr, "ds4: [EVAL_IDS] 需要 graph 后端 -- aborting\n");
    exit(1);
#else
    const uint32_t vocab = (uint32_t)DS4_N_VOCAB;
    uint32_t nfile = 0;
    int *file_ids = eval_ids_load(idp, &nfile, vocab);

    /* 默认在流首插 BOS(裸 BOS 起); DS4_EVAL_NO_BOS=1 则原样喂。 */
    const int no_bos = getenv("DS4_EVAL_NO_BOS") != NULL;
    const uint32_t n = no_bos ? nfile : nfile + 1u;
    int *ids = xmalloc((size_t)n * sizeof(int));
    if (no_bos) {
        memcpy(ids, file_ids, (size_t)nfile * sizeof(int));
    } else {
        ids[0] = e->vocab.bos_id;
        memcpy(ids + 1, file_ids, (size_t)nfile * sizeof(int));
    }
    free(file_ids);
    /* 首 8 个生效 id 打出来: 与锚文件对齐与否一眼可验(错位一格 Σmin 就全废)。 */
    fprintf(stderr, "ds4: [EVAL_IDS] S=%u (文件 %u%s) vocab=%u 首8: ",
            n, nfile, no_bos ? ", 无 BOS" : ", 首插 BOS", vocab);
    for (uint32_t i = 0; i < n && i < 8u; i++) fprintf(stderr, "%d ", ids[i]);
    fprintf(stderr, "\n");

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, (int)n + 64) != 0 || !s) {
        fprintf(stderr, "ds4: [EVAL_IDS] 会话创建失败 -- aborting\n");
        exit(1);
    }
    const uint32_t pcap = (uint32_t)ds4_session_prefill_cap(s);
    if (!pcap) { fprintf(stderr, "ds4: [EVAL_IDS] prefill_cap=0 -- aborting\n"); exit(1); }
    /* 不设 DS4_METAL_PREFILL_CHUNK 时 prefill_cap 等于整个 ctx(一次灌完), 那会让 hc
     * 暂存和单批显存都按 S 放大。仪器自己按 512 分块(DS4_EVAL_CHUNK 可调), 与 A3
     * 的 PREFILL_CHUNK=512 口径一致; 分块只影响批大小, 不影响数值。 */
    uint32_t cap = 512u;
    { const char *cv = getenv("DS4_EVAL_CHUNK"); if (cv && atoi(cv) > 0) cap = (uint32_t)atoi(cv); }
    if (cap > pcap) cap = pcap;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    float *hc = xmalloc((size_t)cap * hc_dim * sizeof(float));

    FILE *lf = NULL; float *lg = NULL;
    const char *lp = getenv("DS4_EVAL_LOGITS");
    if (lp && lp[0]) {
        lf = fopen(lp, "wb");
        if (!lf) {
            fprintf(stderr, "ds4: [EVAL_IDS] 打不开 %s -- aborting\n", lp);
            exit(1);
        }
        lg = xmalloc((size_t)vocab * sizeof(float));
    }

    char err[256];
    for (uint32_t p0 = 0; p0 < n; p0 += cap) {
        const uint32_t nt = (n - p0 < cap) ? (n - p0) : cap;
        /* layer_start=0 ⇒ 由 token 直接 embed(input_hc=NULL); 走到最后一层拿整批出口 HC。
         * 逐层 hidden 由 eval_hdump_batch_layer 在同一趟前向里顺带写出, 不重复前向。 */
        if (ds4_session_eval_layer_slice(s, ids + p0, nt, p0, 0, (uint32_t)DS4_N_LAYER - 1u,
                                         NULL, hc, false, NULL, err, sizeof err) != 0) {
            fprintf(stderr, "ds4: [EVAL_IDS] prefill 失败 @pos %u: %s -- aborting\n", p0, err);
            exit(1);
        }
        if (lf) {
            /* 逐位置过输出头: eval_output_head_from_hc 只算传入批的最后一行, 所以按
             * n_tokens=1 逐位置喂该位置的 HC 隐状态。 */
            for (uint32_t t = 0; t < nt; t++) {
                if (ds4_session_eval_output_head_from_hc(s, hc + (uint64_t)t * hc_dim, 1u,
                                                         lg, err, sizeof err) != 0) {
                    fprintf(stderr, "ds4: [EVAL_IDS] 输出头失败 @pos %u: %s -- aborting\n",
                            p0 + t, err);
                    exit(1);
                }
                if (fwrite(lg, sizeof(float), vocab, lf) != vocab) {
                    fprintf(stderr, "ds4: [EVAL_IDS] logits 短写 @pos %u -- aborting\n", p0 + t);
                    exit(1);
                }
            }
            fflush(lf);
        }
        fprintf(stderr, "ds4: [EVAL_IDS] %u/%u\n", p0 + nt, n);
    }
    if (lf && fclose(lf) != 0) {
        fprintf(stderr, "ds4: [EVAL_IDS] 关闭 %s 失败 -- aborting\n", lp);
        exit(1);
    }
    fprintf(stderr, "ds4: [EVAL_IDS] 完成 S=%u vocab=%u%s%s\n", n, vocab,
            lp && lp[0] ? " logits已写" : "",
            getenv("DS4_EVAL_HDUMP") ? " hidden已写" : "");
    ds4_session_free(s);
    free(hc); free(lg); free(ids);
    exit(0);
#endif
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->distributed = opt->distributed;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    if (e->power_percent > 100) e->power_percent = 100;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = ds4_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

    bool load_slice = opt->load_slice;
    uint32_t load_layer_start = opt->load_layer_start;
    uint32_t load_layer_end = opt->load_layer_end;
    bool load_output = opt->load_output;
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE &&
        opt->distributed.layers.set)
    {
        load_slice = true;
        load_layer_start = opt->distributed.layers.start;
        load_layer_end = opt->distributed.layers.has_output ?
                         UINT32_MAX : opt->distributed.layers.end;
        load_output = opt->distributed.layers.has_output;
    }
    /* MTP drafter 拓扑整族已删除(2026-08-05)。 */
    const bool include_output_head = load_output;
    const bool mtp_keep_token_embd = false;
    const bool graph_backend = ds4_backend_uses_graph(opt->backend);
    ds4_profile_load_begin();
    /* BASE model: the only open allowed to arm go1b/go2b env defaults (the
     * MTP draft open below and all sidecar opens leave the flag false). */
    g_model_open_arm_env_defaults = true;
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    g_model_open_arm_env_defaults = false;
    if (opt->warm_weights) model_warm_weights(&e->model);
    if (!opt->inspect_only) vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    weights_bind(&e->weights, &e->model);
    dspark_bind_with_draft(&e->dspark, &e->model, graph_backend);
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }

    /* go1b "hidden variable z^L" four-loss correction sidecar. An explicit --corr
     * PATH is always honoured; otherwise, when this model uses strict-1-bit (go1b)
     * routed experts, auto-detect ds4-go1b-corr.gguf next to the -m model. Absent
     * or unreadable => exactly the pure 1-bit path (fully backward compatible). */
    {
        const char *corr_path = opt->corr_path;
        char corr_auto[1024];
        if (!corr_path || !corr_path[0]) {
            bool is_go1b = false;
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                if (e->weights.layer[il].ffn_gate_exps &&
                    e->weights.layer[il].ffn_gate_exps->type == DS4_TENSOR_GO1B) {
                    is_go1b = true;
                    break;
                }
            }
            if (is_go1b && opt->model_path) {
                const char *slash = strrchr(opt->model_path, '/');
                if (slash) {
                    size_t dlen = (size_t)(slash - opt->model_path) + 1;   /* keep '/' */
                    if (dlen < sizeof(corr_auto) - sizeof("ds4-go1b-corr.gguf")) {
                        memcpy(corr_auto, opt->model_path, dlen);
                        memcpy(corr_auto + dlen, "ds4-go1b-corr.gguf", sizeof("ds4-go1b-corr.gguf"));
                        if (access(corr_auto, R_OK) == 0) corr_path = corr_auto;
                    }
                } else if (access("ds4-go1b-corr.gguf", R_OK) == 0) {
                    snprintf(corr_auto, sizeof(corr_auto), "ds4-go1b-corr.gguf");
                    corr_path = corr_auto;
                }
            }
        }
        if (corr_path && corr_path[0]) {
            e->model.corr = corr_load(corr_path, graph_backend);
        }
        /* 1-bit residual sidecar (--residual): a second go1b layer per hot expert,
         * summed into the base expert output. Absent => single 1-bit (today). */
        {   /* --residual / DS4_RESIDUAL(2026-07-14): server 等无 CLI 旋钮的宿主经 env 挂热残差侧车 */
            const char *res_path = (opt->residual_path && opt->residual_path[0])
                                 ? opt->residual_path : getenv("DS4_RESIDUAL");
            const char *vq_dir = getenv("DS4_VQ_DIR");
            if (vq_dir && !e->model.residual) {
                e->model.residual = vq_dir_load(vq_dir);
                if (e->model.residual) res_path = NULL;
            }
            if (res_path && res_path[0])
                e->model.residual = residual_load(res_path, graph_backend);
            /* 合一 VQ GGUF: env 均未指定时, 文件自带 blob 张量即自动装载(文件即权威)。 */
            if (!e->model.residual)
                e->model.residual = vq_model_load(&e->model);
        }
        /* go-onebit 优化链: 外部 --zchain/DS4_ZCHAIN 文件优先(实验覆盖); 否则合一
         * GGUF 内嵌 blk.L.opt_* 张量(ds4.zchain.present)自动装载。GE 增益乘进
         * 路由权重 + 逐 token routed 缩放 λ(x)。都缺 => 素颜 1bit+signref 基座。 */
        {
            const char *zchain_path = opt->zchain_path && opt->zchain_path[0]
                                    ? opt->zchain_path : getenv("DS4_ZCHAIN");
            if (zchain_path && zchain_path[0]) {
                e->model.zchain = ds4_zchain_load(zchain_path, DS4_N_LAYER,
                                                  DS4_N_EXPERT, DS4_N_EMBD);
                /* 显式请求的侧车打不开/空链 => 硬失败。静默裸跑过一次假对照
                 * (+z==裸, 2026-08-20), 判决容不得兜底。 */
                if (!e->model.zchain) {
                    fprintf(stderr, "ds4: zchain %s requested but unusable -- aborting (no silent bare-model fallback)\n",
                            zchain_path);
                    exit(1);
                }
            } else {
                e->model.zchain = zchain_from_model(&e->model);
            }
        /* 第4文件(2026-08-20 用户四文件设计): drafter 反修放大器侧车 DS4_DRAFT_ZCHAIN。
         * 3 层链(mtp.0/1/2)合并进主链尾部槽 43..45 ⇒ 单 GPU 表一次上传;
         * 合并链同挂主/draft 两个 model, ffn_batch 的 zch=model->zchain 两侧都取到,
         * drafter FFN 按 il=43+b 索引。显式请求打不开 => 硬失败(无静默兜底)。 */
        {
            const char *draft_zc = getenv("DS4_DRAFT_ZCHAIN");
            if (draft_zc && draft_zc[0]) {
                if (!g_draft_model || !e->dspark.ready) {
                    fprintf(stderr, "ds4: DS4_DRAFT_ZCHAIN requires a mounted drafter (DS4_DRAFT_GGUF)\n");
                    exit(1);
                }
                struct ds4_zchain *dz = ds4_zchain_load(draft_zc, 3, DS4_N_EXPERT, DS4_N_EMBD);
                if (!dz) {
                    fprintf(stderr, "ds4: draft zchain %s unusable -- aborting\n", draft_zc);
                    exit(1);
                }
                struct ds4_zchain *base = e->model.zchain;
                struct ds4_zchain *mg = xmalloc(sizeof(*mg));
                memset(mg, 0, sizeof(*mg));
                mg->n_layer = (uint32_t)DS4_N_LAYER + 3u;
                mg->n_expert = DS4_N_EXPERT;
                mg->d_model = DS4_N_EMBD;
                mg->layer = xcalloc(mg->n_layer, sizeof(mg->layer[0]));
                if (base) {
                    memcpy(mg->layer, base->layer, (size_t)DS4_N_LAYER * sizeof(mg->layer[0]));
                    mg->n_ops_total = base->n_ops_total;
                    mg->n_ge_layers = base->n_ge_layers;
                    mg->map = base->map; mg->map_size = base->map_size;
                }
                memcpy(mg->layer + DS4_N_LAYER, dz->layer, 3u * sizeof(mg->layer[0]));
                mg->n_ops_total += dz->n_ops_total;
                mg->n_ge_layers += dz->n_ge_layers;
                /* base/dz 壳被 merged alias(单例, 进程生命周期), 不 free */
                e->model.zchain = mg;
                g_draft_model->zchain = mg;
                fprintf(stderr, "ds4: draft zchain merged: %s (3 layers @ slots 43..45)\n", draft_zc);
            } else if (g_draft_model && e->model.zchain) {
                /* 无 draft 侧车但有主侧车: drafter FFN 的 zch 取 dmodel->zchain,
                 * 保持 NULL 即旁路(主链槽 0..42 与 drafter il 43+b 互不相扰) */
                g_draft_model->zchain = NULL;
            }
        }
#ifndef DS4_NO_GPU
            if (e->model.zchain && graph_backend &&
                !zchain_gpu_upload(e->model.zchain)) {
                fprintf(stderr, "ds4: zchain GPU upload failed -- aborting (no silent quality downgrade)\n");
                exit(1);
            }
#endif
        }
        /* go-trie/ref-corpus drafter 设施已随 copy-spec/MTP 整族删除(2026-08-05 用户裁决)。 */
        /* 多模态 registry, same tokenizer bridge. The image family binds an
         * external encoder command when one is present -- resolution:
         * DS4_MM_IMAGE_CMD env, else ./mm-ui (the frontend-domain UI-sketch
         * tool, `make mm-ui`; cwd-relative like the metal shader dir, so
         * --chdir applies). Absent => image content is honestly rejected
         * upstream (server 400s image blocks instead of dropping them). */
        e->mm = ds4_mm_create(engine_tokenize_cb, e);
        if (e->mm) {
            const char *mm_cmd = getenv("DS4_MM_IMAGE_CMD");
            if ((!mm_cmd || !mm_cmd[0]) && access("mm-ui", X_OK) == 0)
                mm_cmd = "./mm-ui";
            if (mm_cmd && mm_cmd[0] &&
                ds4_mm_register_command(e->mm, "image", mm_cmd) == 0)
                fprintf(stderr, "ds4: multimodal image encoder: %s\n", mm_cmd);
            /* 前端域两插件, 顺序=节顺序: 物理方位先(关系), CSS 后(换算)。
             * 与编码器解耦: 换编码器(DS4_MM_IMAGE_CMD)后输出若非草图格式,
             * 两节自然缺席, 不伪造。 */
            ds4_mm_register_enricher(e->mm, "image", engine_mm_spatial_enrich, NULL);
            ds4_mm_register_enricher(e->mm, "image", engine_mm_css_enrich, NULL);
        }
    }
    if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    /* MTP 支持模型加载已整族删除(2026-08-05 用户裁决: Go 定型优化)。 */
#ifndef DS4_NO_GPU
    if (e->backend == DS4_BACKEND_CUDA) {
#ifdef __APPLE__
        fprintf(stderr, "ds4: CUDA backend requested but this build is linked with Metal, not CUDA\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (e->backend == DS4_BACKEND_METAL) {
#ifndef __APPLE__
        fprintf(stderr, "ds4: Metal backend requested but this build is linked with CUDA, not Metal\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (graph_backend) {
        e->metal_ready = ds4_gpu_init() != 0;
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_quality(e->quality);
        (void)ds4_gpu_set_model_fd(e->model.fd);
        /* project.md P2.2 low-cost variant: when DS4_DIST_EXPERT_FETCH_SERVE=1
         * (worker side), serve raw model-file range reads so the peer's expert
         * gather can draw from this machine's faster idle SSD over Thunderbolt. */
        (void)ds4_dist_expert_fetch_maybe_serve(e->model.fd, e->model.size);
        /* Wave 30 reverse-established variant: when the worker cannot dial out
         * (asymmetric bridge), the coordinator dials the worker's accept-mode
         * listener instead and serves preads on the dialed sockets. */
        (void)ds4_dist_expert_fetch_serve_dial(e->model.fd, e->model.size);
        int model_map_ok = 0;
        uint64_t base_l1_resident_bytes = 0;
        /* Under DS4_MTP_NO_RESIDENCY the draft model is left evictable (not
         * wired), so it must not count against the L1 resident budget gate. */
        const uint64_t mtp_l1_resident_bytes =
            (e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") == NULL) ?
            e->mtp_model.size - e->mtp_model.tensor_data_pos : 0;
        /* Dynamic resident/offload route (replaces the old hardcoded env flag):
         * keep routed experts resident -- direct GPU read, no per-layer CPU gather,
         * full decode speed -- whenever the fully-resident model fits the memory
         * budget; only stream when it would bust it. DS4_METAL_EXPERT_OFFLOAD still
         * works as an explicit override (1 = force stream, 0 = force resident). */
        const char *expert_offload_env = getenv("DS4_METAL_EXPERT_OFFLOAD");
        bool expert_offload_requested;
        if (expert_offload_env && expert_offload_env[0]) {
            expert_offload_requested =
                !(expert_offload_env[0] == '0' && expert_offload_env[1] == '\0');
        } else {
            uint64_t full_resident_bytes = 0;
            if (load_slice) {
                ds4_model_map_span_vec bb_probe, exp_probe;
                if (weights_model_map_spans_split_slice(&e->weights, load_layer_start,
                        load_layer_end, include_output_head, mtp_keep_token_embd,
                        &bb_probe, &exp_probe)) {
                    for (uint32_t i = 0; i < bb_probe.len; i++)
                        full_resident_bytes += bb_probe.v[i].end - bb_probe.v[i].off;
                    for (uint32_t i = 0; i < exp_probe.len; i++)
                        full_resident_bytes += exp_probe.v[i].end - exp_probe.v[i].off;
                    free(bb_probe.v);
                    free(exp_probe.v);
                }
            } else {
                full_resident_bytes = e->model.size - e->model.tensor_data_pos;
            }
            uint64_t auto_budget = ds4_runtime_mem_budget_bytes();
            if (auto_budget == 0) auto_budget = ds4_gpu_recommended_max_working_set_bytes();
            const uint64_t planned = full_resident_bytes + mtp_l1_resident_bytes;
            expert_offload_requested = (auto_budget > 0) && (full_resident_bytes > 0) &&
                (planned > (uint64_t)((double)auto_budget * 0.85));
            fprintf(stderr,
                    "ds4: expert-offload AUTO: full-resident %.2f GiB vs %.2f GiB budget -> %s\n",
                    (double)planned / DS4_GIB, (double)auto_budget / DS4_GIB,
                    expert_offload_requested ? "stream (offload)" : "resident (fast)");
        }
        /* VQ blob 专家不进 Metal span(ds4.c:2074): 前向唯一路径=CPU gather→f16 scratch
         * (ds4.c:2277 从 mmap vq_raw 读)。resident(offload=0)会让 MoE kernel 去 residency
         * set 直读不存在的专家 span → 崩。双机切层后 planned<budget 时 AUTO 会误选 resident,
         * 故 VQ blob 恒强制 offload。 */
        if (g_vq_experts_blob && !expert_offload_requested) {
            fprintf(stderr, "ds4: VQ blob 专家: 强制 offload(CPU gather 是唯一前向路径, 覆盖 AUTO resident)\n");
            expert_offload_requested = true;
        }
        ds4_gpu_set_expert_offload(expert_offload_requested ? 1 : 0);
        if (load_slice) {
            char load_end[32];
            if (load_output && load_layer_end == UINT32_MAX) {
                snprintf(load_end, sizeof(load_end), "output");
            } else if (load_output) {
                snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
            } else {
                snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
            }

            if (expert_offload_requested) {
                ds4_model_map_span_vec bb, exp;
                if (!weights_model_map_spans_split_slice(&e->weights,
                                                         load_layer_start,
                                                         load_layer_end,
                                                         include_output_head,
                                                         mtp_keep_token_embd,
                                                         &bb,
                                                         &exp))
                {
                    fprintf(stderr, "ds4: invalid expert-offload model load layer slice %u:%s\n",
                            load_layer_start,
                            load_end);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
                const uint32_t total = bb.len + exp.len;
                uint64_t *offsets = xmalloc((size_t)total * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)total * sizeof(sizes[0]));
                bool *resident = xmalloc((size_t)total * sizeof(resident[0]));
                uint64_t resident_bytes = 0, reclaimable_bytes = 0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < bb.len; i++) {
                    offsets[n] = bb.v[i].off;
                    sizes[n] = bb.v[i].end - bb.v[i].off;
                    resident[n] = true;
                    resident_bytes += sizes[n];
                    n++;
                }
                for (uint32_t i = 0; i < exp.len; i++) {
                    offsets[n] = exp.v[i].off;
                    sizes[n] = exp.v[i].end - exp.v[i].off;
                    resident[n] = false;
                    reclaimable_bytes += sizes[n];
                    n++;
                }
                uint64_t split_max_tensor = bb.max_tensor_bytes;
                if (exp.max_tensor_bytes > split_max_tensor) split_max_tensor = exp.max_tensor_bytes;
                base_l1_resident_bytes = resident_bytes;
                fprintf(stderr,
                        "ds4: restricting %s model map to layers %u:%s with expert offload "
                        "(%.2f GiB backbone resident, %.2f GiB routed experts reclaimable; "
                        "%u backbone + %u expert spans)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        (double)resident_bytes / 1073741824.0,
                        (double)reclaimable_bytes / 1073741824.0,
                        bb.len,
                        exp.len);
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans_split(e->model.map,
                                                                 e->model.size,
                                                                 offsets,
                                                                 sizes,
                                                                 resident,
                                                                 total,
                                                                 split_max_tensor);
                if (model_map_ok) {
                    engine_register_layer_routers(e, load_layer_start, load_layer_end);
                }
                free(offsets);
                free(sizes);
                free(resident);
                free(bb.v);
                free(exp.v);
            } else {
                ds4_model_map_span_vec spans;
                if (!weights_model_map_spans(&e->weights,
                                             load_layer_start,
                                             load_layer_end,
                                             include_output_head,
                                             mtp_keep_token_embd,
                                             &spans))
                {
                    fprintf(stderr, "ds4: invalid model load layer slice %u:%s\n",
                            load_layer_start,
                            load_end);
                    ds4_engine_close(e);
                    *out = NULL;
                    return 1;
                }
                uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
                uint64_t span_bytes = 0;
                for (uint32_t i = 0; i < spans.len; i++) {
                    offsets[i] = spans.v[i].off;
                    sizes[i] = spans.v[i].end - spans.v[i].off;
                    span_bytes += sizes[i];
                }
                base_l1_resident_bytes = span_bytes;
                fprintf(stderr,
                        "ds4: restricting %s model map to layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        spans.len,
                        (double)span_bytes / 1073741824.0);
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                            e->model.size,
                                                            offsets,
                                                            sizes,
                                                            spans.len,
                                                            spans.max_tensor_bytes);
                free(offsets);
                free(sizes);
                free(spans.v);
            }
        } else if (expert_offload_requested) {
            /* Reduced-memory load: wire only the backbone (attn / shared FFN /
             * embedding / output) into the GPU residency set and keep the routed
             * experts reclaimable. The hot path still resolves every tensor's
             * buffer; cold experts just are not pinned resident. */
            ds4_model_map_span_vec bb, exp;
            if (!weights_model_map_spans_split(&e->weights, &bb, &exp)) {
                fprintf(stderr,
                        "ds4: DS4_METAL_EXPERT_OFFLOAD requested but span split failed; "
                        "falling back to the full-residency loader\n");
                base_l1_resident_bytes = e->model.size - e->model.tensor_data_pos;
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                           e->model.size,
                                                           e->model.tensor_data_pos,
                                                           e->model.size - e->model.tensor_data_pos,
                                                           e->model.max_tensor_bytes);
            } else {
                const uint32_t total = bb.len + exp.len;
                uint64_t *offsets = xmalloc((size_t)total * sizeof(offsets[0]));
                uint64_t *sizes = xmalloc((size_t)total * sizeof(sizes[0]));
                bool *resident = xmalloc((size_t)total * sizeof(resident[0]));
                uint64_t resident_bytes = 0, reclaimable_bytes = 0;
                uint32_t n = 0;
                for (uint32_t i = 0; i < bb.len; i++) {
                    offsets[n] = bb.v[i].off;
                    sizes[n] = bb.v[i].end - bb.v[i].off;
                    resident[n] = true;
                    resident_bytes += sizes[n];
                    n++;
                }
                for (uint32_t i = 0; i < exp.len; i++) {
                    offsets[n] = exp.v[i].off;
                    sizes[n] = exp.v[i].end - exp.v[i].off;
                    resident[n] = false;
                    reclaimable_bytes += sizes[n];
                    n++;
                }
                uint64_t split_max_tensor = bb.max_tensor_bytes;
                if (exp.max_tensor_bytes > split_max_tensor) split_max_tensor = exp.max_tensor_bytes;
                fprintf(stderr,
                        "ds4: expert-offload model map: %.2f GiB backbone resident, "
                        "%.2f GiB routed experts reclaimable (%u backbone + %u expert spans)\n",
                        (double)resident_bytes / 1073741824.0,
                        (double)reclaimable_bytes / 1073741824.0,
                        bb.len, exp.len);
                base_l1_resident_bytes = resident_bytes;
                ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
                if (getenv("DS4_METAL_EXPERT_OFFLOAD_DEBUG") != NULL) {
                    for (uint32_t i = 0; i < total; i++) {
                        fprintf(stderr, "ds4:   span[%u] %s %.4f..%.4f GiB (%.1f MiB)\n",
                                i, resident[i] ? "RES" : "exp",
                                (double)offsets[i] / 1073741824.0,
                                (double)(offsets[i] + sizes[i]) / 1073741824.0,
                                (double)sizes[i] / 1048576.0);
                    }
                }
                model_map_ok = ds4_gpu_set_model_map_spans_split(e->model.map,
                                                                 e->model.size,
                                                                 offsets,
                                                                 sizes,
                                                                 resident,
                                                                 total,
                                                                 split_max_tensor);
                if (model_map_ok) {
                    engine_register_layer_routers(e, 0, DS4_MAX_LAYER - 1);
                }
                free(offsets);
                free(sizes);
                free(resident);
                free(bb.v);
                free(exp.v);
            }
        } else {
            base_l1_resident_bytes = e->model.size - e->model.tensor_data_pos;
            ds4_l1_budget_gate(base_l1_resident_bytes + mtp_l1_resident_bytes, 0);
            model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                       e->model.size,
                                                       e->model.tensor_data_pos,
                                                       e->model.size - e->model.tensor_data_pos,
                                                       e->model.max_tensor_bytes);
        }
        if (!model_map_ok) {
            fprintf(stderr,
                    "ds4: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        /* DS4_MTP_NO_RESIDENCY: wrap the draft model's views evictable (not
         * wired) so they do not pin the worker's GPU residency budget. */
        const bool mtp_nonresident = e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") != NULL;
        if (mtp_nonresident) ds4_gpu_set_model_map_nonresident_hint(1);
        if (e->mtp_ready &&
            !ds4_gpu_set_model_map_range(e->mtp_model.map,
                                           e->mtp_model.size,
                                           e->mtp_model.tensor_data_pos,
                                           e->mtp_model.size - e->mtp_model.tensor_data_pos,
                                           e->mtp_model.max_tensor_bytes))
        {
            if (mtp_nonresident) ds4_gpu_set_model_map_nonresident_hint(0);
            fprintf(stderr,
                    "ds4: %s failed to map MTP model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (mtp_nonresident) ds4_gpu_set_model_map_nonresident_hint(0);
        /* 副 drafter map 注册必须在主模型 map 之后: ds4_gpu_set_model_map 换 base 时
         * release_all 会连带释放先注册的 range(2026-08-21 实测顺序坑)。 */
        /* 副 map 注册失败 => drafter 读到的是不可用的裸指针(实测 logits NaN, 草稿全废、
         * acc 塌到 1.00 而速度看似"正常")。明确停用 drafter, 不接受静默劣化。 */
        /* ★默认关(2026-08-21 实测): 副 map 整体 cudaHostRegister + HostGetDevicePointer
         * 在 5.59GB 文件映射上不可靠 —— drafter 同输入两跑 MoE 输出不同(Fin/路由/权重
         * 逐字节相同), 关掉即完全确定; 注册前把页全部触实也无效。改走 cuda_model_range_ptr
         * 的按 range 懒注册(页对齐分段, 已验证确定)。DS4_AUX_REG=1 可强开做对照。 */
        if (g_draft_model && getenv("DS4_NO_AUX_REG") == NULL &&
            !ds4_gpu_register_aux_model_map(g_draft_model->map, g_draft_model->size)) {
            fprintf(stderr, "ds4: draft gguf aux map registration failed -- disabling drafter "
                            "(speculation off; plain decode continues)\n");
            e->dspark.ready = false;
            g_dspark_ready_global = 0;
            g_dspark_bound_for_prefill = NULL;
        }
        if (!accelerator_cache_model_tensors(e->backend, &e->model)) {
            fprintf(stderr, "ds4: %s failed to prepare startup model cache\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        /* Also populate the HBM cache for the MTP support model when loaded.
         * Without this, MTP-block tensor reads at decode time hit the UVA-
         * mapped pointer (slow) instead of cudaMalloc'd HBM copies (fast).
         * The MoE expert filter in accelerator_cache_model_tensor_spans
         * skips `mtp.0.ffn_*_exps.weight` automatically.
         *
         * DS4_MTP_NO_RESIDENCY=1: skip wiring the MTP draft model into the GPU
         * residency set.  On Metal the draft tensors stay no-copy mmap shared
         * buffers (read directly, not slow -- same as the main model), just
         * evictable.  This frees the ~3.8 GiB the draft would otherwise pin,
         * letting the memory-tight worker (M1, ~10.67 GiB GPU budget) hold its
         * layer-slice backbone + the draft on the default fast split instead of
         * OOMing ("residency wired 51 views" / kIOGPUCommandBufferCallbackError
         * OutOfMemory).  The draft forward pages the (hot, every-step) tensors
         * back via the page cache. */
        if (e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") == NULL &&
            !accelerator_cache_model_tensors(e->backend, &e->mtp_model)) {
            fprintf(stderr, "ds4: %s failed to prepare MTP startup model cache\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->mtp_ready && getenv("DS4_MTP_NO_RESIDENCY") != NULL) {
            fprintf(stderr,
                    "ds4: MTP draft model left non-resident (DS4_MTP_NO_RESIDENCY=1); "
                    "draft tensors stay evictable mmap views to save GPU residency\n");
        }
        fprintf(stderr, "ds4: %s backend initialized for graph diagnostics\n",
                ds4_backend_name(e->backend));
    }
#else
    if (graph_backend) {
        fprintf(stderr, "ds4: %s backend requested but this build has no graph backend support; aborting startup\n",
                ds4_backend_name(e->backend));
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
#endif

    ds4_profile_load_end();
    ds4_multi_bench_run(e);   /* DS4_MULTI_BENCH=N: 并发批实测(跑完退出) */
    ds4_eval_ids_run(e);   /* DS4_EVAL_IDS 在场则跑完仪器直接退出, 不返回 */
    *out = e;
    return 0;
}

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
    if (e->mtp_ready) model_close(&e->mtp_model);
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
    const char *dist_prefill_env = getenv("DS4_DIST_PREFILL_CAP");
    if (dist_prefill_env && dist_prefill_env[0] &&
        e->distributed.role != DS4_DISTRIBUTED_NONE && e->distributed.layers.set) {
        char *endp = NULL;
        unsigned long v = strtoul(dist_prefill_env, &endp, 10);
        if (endp != dist_prefill_env && v > 0 && v <= (unsigned long)ctx_size) {
            s->prefill_cap = (uint32_t)v;
        }
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
    s->graph.dspark_capture = e->dspark.ready ? 1 : 0;
    if (!metal_graph_alloc_raw_cap(&s->graph, &e->weights, &e->weights.layer[0],
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap, e->mtp_ready,
                                   active_start, active_end, active_slice))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    s->graph.power_percent = (uint32_t)e->power_percent;
    if (e->distributed.tp_enabled) {
        /* Establish the TP peer link once per engine. By default the coordinator
         * listens and the worker connects. DS4_TP_REVERSE_CONNECT=1 flips the
         * network roles (coordinator connects, worker listens) to work around a
         * host where one direction's connect() fails (observed: an M1 where ds4's
         * outbound connect returns EHOSTUNREACH while nc/plain connect succeed).
         * The TP all-reduce is a symmetric sum, so connect direction does not
         * affect results; tp_owns_low stays tied to role, not to who listens.
         * Blocks until both peers are up; KB-level buffers only. */
        if (!e->tp) {
            char terr[256] = {0};
            const char *rev = getenv("DS4_TP_REVERSE_CONNECT");
            bool reverse = (rev && *rev && rev[0] != '0');
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
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
        s->mtp_draft_token = -1;
    }
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

int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen) {
    if (!s || !s->engine) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    if (layer_start > layer_end || layer_end >= (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice layer range %u:%u",
                             layer_start, layer_end);
        return 1;
    }
    if (layer_start != 0 && !input_hc) {
        if (errlen) snprintf(err, errlen, "layer-slice layer %u requires input hidden-state",
                             layer_start);
        return 1;
    }
    if (output_logits && layer_end + 1u != (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "layer-slice logits require final transformer layer");
        return 1;
    }
    if (output_logits && !logits) {
        if (errlen) snprintf(err, errlen, "layer-slice logits output is missing");
        return 1;
    }
    /* A distributed prefill pipeline may need only the KV side effect for
     * non-final chunks. In that case both output_hc and logits are NULL. */
    if (ds4_session_slice_check_timeline(s, tokens, n_tokens, pos0, err, errlen) != 0) {
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        if (errlen) snprintf(err, errlen, "layer slices require the graph backend");
        s->checkpoint_valid = false;
        return 1;
    }
#ifdef DS4_NO_GPU
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    s->checkpoint_valid = false;
    return 1;
#else
    if (n_tokens > s->prefill_cap) {
        if (errlen) snprintf(err, errlen, "layer-slice chunk %u exceeds prefill cap %u",
                             n_tokens, s->prefill_cap);
        return 1;
    }

    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    if (!input_hc && !output_hc && output_logits &&
        layer_start == 0 && layer_end + 1u == (uint32_t)DS4_N_LAYER) {
        bool ok = false;
        ds4_tokens span = {0};
        if (pos0 == 0) {
            span.v = (int *)tokens;
            span.len = (int)n_tokens;
            span.cap = (int)n_tokens;
            ok = metal_graph_prefill_layer_major(g,
                                                 &e->model,
                                                 &e->weights,
                                                 &span,
                                                 0,
                                                 n_tokens,
                                                 logits,
                                                 false,
                                                 NULL,
                                                 NULL,
                                                 NULL);
        } else if (n_tokens == 1) {
            ok = metal_graph_eval_token_raw_swa(g,
                                                &e->model,
                                                &e->weights,
                                                tokens[0],
                                                pos0,
                                                logits);
        } else {
            if (pos0 > (uint32_t)INT_MAX - n_tokens) {
                if (errlen) snprintf(err, errlen, "layer-slice full span is too large");
                s->checkpoint_valid = false;
                return 1;
            }
            span.len = (int)(pos0 + n_tokens);
            span.cap = span.len;
            span.v = calloc((size_t)span.len, sizeof(span.v[0]));
            if (span.v) {
                for (uint32_t i = 0; i < n_tokens; i++) span.v[pos0 + i] = tokens[i];
                ok = metal_graph_prefill_layer_major(g,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     pos0,
                                                     n_tokens,
                                                     logits,
                                                     false,
                                                     NULL,
                                                     NULL,
                                                     NULL);
            }
            free(span.v);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice full failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice full evaluation failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }
        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = (uint64_t)n_tokens * hc_dim * sizeof(float);
    if (n_tokens == 1 && pos0 > 0) {
        if (g->raw_cap == 0) {
            if (errlen) snprintf(err, errlen, "%s layer-slice decode has no raw KV cache",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        bool ok = true;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(g->cur_hc, 0, input_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok && !input_hc) {
            ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                               e->model.map,
                                               e->model.size,
                                               e->weights.token_embd->abs_offset,
                                               (uint32_t)e->weights.token_embd->dim[1],
                                               (uint32_t)tokens[0],
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;
        }
        const uint32_t raw_row = pos0 % g->raw_cap;
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, 1);
        const uint32_t split_after_layers = metal_graph_token_split_after_layers();
        uint32_t encoded_layers = 0;
        for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
            ok = metal_graph_encode_decode_layer(g,
                                                 &e->model,
                                                 &e->weights.layer[il],
                                                 il,
                                                 pos0,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 tokens[0]);
            ds4_gpu_tensor *tmp = g->cur_hc;
            g->cur_hc = g->after_ffn_hc;
            g->after_ffn_hc = tmp;
            encoded_layers++;
            if (ok &&
                split_after_layers != 0 &&
                (encoded_layers % split_after_layers) == 0 &&
                il < layer_end)
            {
                if (metal_graph_direct_expert_read_enabled()) {
                    ok = ds4_gpu_end_commands() != 0 && ds4_gpu_begin_commands() != 0;
                } else {
                    ok = ds4_gpu_flush_commands() != 0;
                }
            }
        }
        if (ok && output_logits) {
            ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
        if (ok && output_hc) {
            ok = ds4_gpu_tensor_read(g->cur_hc, 0, output_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok && output_logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice decode failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice decode failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    ds4_tokens span = {
        .v = (int *)tokens,
        .len = (int)n_tokens,
        .cap = (int)n_tokens,
    };

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, &span, 0, n_tokens);
    if (ok && input_hc) {
        ok = ds4_gpu_tensor_write(g->batch_cur_hc, 0, input_hc, hc_bytes) != 0;
    } else if (ok) {
        ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     0,
                                                     n_tokens);
    }

    ds4_gpu_tensor *last_hc = NULL;
    ds4_gpu_tensor *saved_cur = NULL;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
        ok = metal_graph_encode_layer_batch(g,
                                            &e->model,
                                            &e->weights.layer[il],
                                            il,
                                            pos0,
                                            n_tokens);
    }
    if (ok && output_logits) {
        saved_cur = g->cur_hc;
        last_hc = metal_graph_tensor_row_view(g->batch_cur_hc, n_tokens - 1u, hc_dim);
        ok = last_hc != NULL;
        if (ok) {
            g->cur_hc = last_hc;
            ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
            g->cur_hc = saved_cur;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (saved_cur) g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);

    if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(g->batch_cur_hc, 0, output_hc, hc_bytes) != 0;
    }
    if (ok && output_logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after layer-slice failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s layer-slice failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }

    ds4_session_slice_commit_timeline(s, tokens, n_tokens);
    return 0;
#endif
}

/* docs/archive/mtp.md Phase 1 (Scheme A) cross-machine verifier: run a K-token candidate
 * batch through this worker's layer slice (layer_start..layer_end, which must be
 * the final transformer layer) and emit the per-row logits into
 * row_logits[i*vocab .. ]. This is the batch verification pass (docs/archive/mtp.md §3.2.2,
 * "末端出 K 组 logits"): row i predicts batch position i+1, so the coordinator
 * argmaxes each row to find the accepted speculative prefix and reuses the
 * boundary row to seed the next sampling step. The batch writes layer KV for
 * positions pos0..pos0+n_tokens-1 and commits all n_tokens to the timeline; the
 * rejected tail is rolled back afterward by truncating the timeline
 * (ds4_session_layer_slice_rollback) so the stale ring rows are overwritten on
 * the next eval. */
int ds4_session_verify_batch_argmax(ds4_session *s,
                                    const int *tokens,
                                    uint32_t n_tokens,
                                    uint32_t pos0,
                                    uint32_t layer_start,
                                    uint32_t layer_end,
                                    const float *input_hc,
                                    float *row_logits,
                                    char *err,
                                    size_t errlen) {
    if (!s || !s->engine || !tokens || !row_logits || n_tokens == 0) {
        if (errlen) snprintf(err, errlen, "invalid verify batch request");
        return 1;
    }
    if (layer_end + 1u != (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "verify batch requires the final transformer layer");
        return 1;
    }
    if (layer_start != 0 && !input_hc) {
        if (errlen) snprintf(err, errlen, "verify batch on a nonzero layer needs input hidden-state");
        return 1;
    }
    if (ds4_session_slice_check_timeline(s, tokens, n_tokens, pos0, err, errlen) != 0) {
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)pos0; (void)layer_start;
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    if (!g->spec_logits) {
        if (errlen) snprintf(err, errlen, "verify batch needs the MTP spec-logits buffer");
        return 1;
    }
    if (n_tokens > s->prefill_cap) {
        if (errlen) snprintf(err, errlen, "verify batch %u exceeds prefill cap %u",
                             n_tokens, s->prefill_cap);
        return 1;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = (uint64_t)n_tokens * hc_dim * sizeof(float);
    ds4_tokens span = { .v = (int *)tokens, .len = (int)n_tokens, .cap = (int)n_tokens };

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, &span, 0, n_tokens);
    if (ok && input_hc) {
        ok = ds4_gpu_tensor_write(g->batch_cur_hc, 0, input_hc, hc_bytes) != 0;
    } else if (ok) {
        ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     0,
                                                     n_tokens);
    }
    const bool vprof = getenv("DS4_SPEC_PROF") != NULL;
    const double vt0 = vprof ? now_sec() : 0.0;
    if (vprof) ds4_gpu_span_begin();
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    /* 批 CUDA 图(2026-08-21): 43 层 ~3.5k kernel 的相邻间隙吃掉 31% GPU 时间。
     * 捕获成图后单次发射, 间隙归零。编码会推进 host 侧压缩器计数 ⇒ 捕获失败必须先
     * 还原再重编码, 否则 KV 记账错位。 */
    uint32_t saved_n_comp[DS4_N_LAYER], saved_n_index[DS4_N_LAYER];
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        saved_n_comp[il] = g->layer_n_comp[il];
        saved_n_index[il] = g->layer_n_index_comp[il];
    }
    const int bgraph = (ok && layer_start == 0) ? ds4_gpu_batch_graph_begin((int)(pos0 & 3u)) : 0;
    for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
        ok = metal_graph_encode_layer_batch(g, &e->model, &e->weights.layer[il],
                                            il, pos0, n_tokens);
    }
    if (bgraph) {
        const int r = ds4_gpu_batch_graph_end_launch(ok ? 1 : 0);
        if (r < 0) {   /* 捕获失败: 图内 kernel 未执行 ⇒ 还原计数后重编码直发 */
            for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
                g->layer_n_comp[il] = saved_n_comp[il];
                g->layer_n_index_comp[il] = saved_n_index[il];
            }
            ok = true;
            for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
                ok = metal_graph_encode_layer_batch(g, &e->model, &e->weights.layer[il],
                                                    il, pos0, n_tokens);
            }
        }
    }
    const double vt1 = vprof ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (vprof) {
        static double enc_acc, wait_acc, gpu_acc; static uint32_t vn;
        const float gpu_ms = ds4_gpu_span_end();
        enc_acc += vt1 - vt0; wait_acc += now_sec() - vt1; vn++;
        if (gpu_ms > 0.0f) gpu_acc += gpu_ms;
        if ((vn & 15u) == 0)
            fprintf(stderr, "ds4: [vfy-prof] n=%u avg_ms: encode_cpu=%.1f end_wait=%.1f gpu_span=%.1f\n",
                    vn, enc_acc * 1e3 / vn, wait_acc * 1e3 / vn, gpu_acc / vn);
    }
    else (void)ds4_gpu_synchronize();
    if (!ok) {
        if (errlen) snprintf(err, errlen, "%s verify batch layers failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }

    /* Output head on all n_tokens rows -> spec_logits, read back K logit rows. */
    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head_batch(g, &e->model, &e->weights,
                                                      n_tokens, e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) {
        ok = ds4_gpu_tensor_read(g->spec_logits, 0, row_logits,
                                 (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    if (!ok) {
        if (errlen) snprintf(err, errlen, "%s verify batch output head failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    /* Commit all K candidate tokens to the timeline so checkpoint.len advances by
     * n_tokens (KV rows pos0..pos0+n_tokens-1 are live). The rejected tail is
     * trimmed afterward via ds4_session_layer_slice_rollback. */
    ds4_session_slice_commit_timeline(s, tokens, n_tokens);
    return 0;
#endif
}

/* docs/archive/mtp.md Phase 1: truncate the layer-slice timeline back to new_len positions
 * after a speculative batch so the rejected tail is dropped. The position-indexed
 * KV ring rows are not cleared; the next eval at new_len overwrites them, exactly
 * like the single-machine MTP rollback. */
int ds4_session_layer_slice_rollback(ds4_session *s, uint32_t new_len,
                                     char *err, size_t errlen) {
    if (!s) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    if (!s->checkpoint_valid || (uint32_t)s->checkpoint.len < new_len) {
        if (errlen) snprintf(err, errlen, "layer-slice rollback target %u exceeds timeline %d",
                             new_len, s->checkpoint.len);
        return 1;
    }
    s->checkpoint.len = (int)new_len;
    s->mtp_draft_valid = false;
    return 0;
}

uint32_t ds4_session_layer_slice_len(const ds4_session *s) {
    if (!s || !s->checkpoint_valid || s->checkpoint.len < 0) return 0;
    return (uint32_t)s->checkpoint.len;
}

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
    if (!g_prof.enabled) return ds4_session_sync_internal(s, prompt, err, errlen);
    /* Tokens actually prefilled = prompt length minus the matching live prefix.
     * Captured before the call because the internal sync mutates the checkpoint. */
    const int start_len = (s && s->checkpoint_valid &&
                           prompt && prompt->len >= s->checkpoint.len &&
                           ds4_tokens_starts_with(prompt, &s->checkpoint))
                          ? s->checkpoint.len : 0;
    const double t0 = now_sec();
    int rc = ds4_session_sync_internal(s, prompt, err, errlen);
    if (rc == 0 && prompt && prompt->len > start_len) {
        ds4_profile_add_prefill((uint64_t)(prompt->len - start_len), now_sec() - t0);
    }
    return rc;
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
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
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

/* ===== 请求批处理: 多会话共享一次前向 (2026-08-21) =====================
 * 为什么值得做: 本机是带宽墙 —— 每 token 要把 4.64GB 权重从 LPDDR5x 搬进计算单元
 * (骨干 2.19GB + 该 token 路由到的 6 个专家/层 2.10GB + 输出头 0.35GB), 29ms/token
 * 就是这么来的。多路请求各跑各的, 这 4.64GB 要各读一遍; 拼进同一次前向, 行无关的
 * 部分只读一遍。实测批路径每行成本: 1 行 35.9ms / 2 行 26.4 / 4 行 18.9 / 8 行 14.9
 * ⇒ 8 路聚合约 1.96×。
 *
 * 实现上不动 KV 结构(那是 6 个数组 180 处引用): 层内本来就分两半 ——
 *   ①注意力半层(KV/压缩器/索引器) 各会话回自己的图算, n_tokens=1;
 *   ②FFN/MoE 半层完全行无关, 借 ss[0] 的图当执行器一次算 N 行。
 * 两半的接口是 batch_after_attn_hc(注意力写) → batch_next_hc(FFN 写), 中间按行拷贝
 * (每层每行 64KB, N=4 时 22MB/步 ≈ 0.1ms, 可忽略)。
 * 注意力半层用 n_tokens=1 与单 token 解码路已验证逐位一致(尾批对账 43 层全同)。 */
int ds4_session_eval_multi(ds4_session **ss, const int *tokens, uint32_t n,
                           char *err, size_t errlen) {
#ifdef DS4_NO_GPU
    (void)ss; (void)tokens; (void)n;
    if (errlen) snprintf(err, errlen, "multi-session batch needs a GPU backend");
    return 1;
#else
    if (!ss || !tokens || n == 0) {
        if (errlen) snprintf(err, errlen, "invalid multi-session batch request");
        return 1;
    }
    if (n == 1 && !getenv("DS4_MULTI_FORCE")) return ds4_session_eval(ss[0], tokens[0], err, errlen);
    ds4_engine *e = ss[0]->engine;
    ds4_gpu_graph *gb = &ss[0]->graph;          /* 共享执行器(它的 batch_* 缓冲够 N 行) */
    if (n > gb->prefill_cap) {
        if (errlen) snprintf(err, errlen, "multi-session batch %u exceeds prefill cap %u",
                             n, gb->prefill_cap);
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (!ss[i] || ss[i]->engine != e || ss[i]->distributed || ds4_session_is_cpu(ss[i])) {
            if (errlen) snprintf(err, errlen, "multi-session batch needs same-engine GPU sessions");
            return 1;
        }
        if (ss[i]->checkpoint.len + 1 >= ss[i]->ctx_size) {
            if (errlen) snprintf(err, errlen, "multi-session batch: session %u out of context", i);
            return 1;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    const uint64_t q_bytes  = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    const uint64_t nm_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);

    bool ok = ds4_gpu_begin_commands() != 0;
    /* 前 3 层是哈希路由层: 专家由 token id 查表决定 ⇒ 共享图的 token 缓冲必须是本批的
     * N 个 id, 否则读到上一次 prefill 的残留 id, 每行选错专家(实测: 同 prompt 的 4 路
     * 输出互不相同, ffn_moe_probs 全同但 ffn_moe_topk 6/6 不同)。 */
    {
        ds4_tokens span = { .v = (int *)tokens, .len = (int)n, .cap = (int)n };
        if (ok) ok = metal_graph_upload_prompt_tokens(gb->prefill_tokens, &span, 0, n);
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_tokens one = { .v = (int *)&tokens[i], .len = 1, .cap = 1 };
            ok = metal_graph_upload_prompt_tokens(ss[i]->graph.prefill_tokens, &one, 0, 1u);
        }
    }
    /* N 个 token 的嵌入一次落共享图的 N 行。必须用批版(token 从张量读): 单 token 版
     * ds4_gpu_embed_token_hc_tensor 把 id 放在一个**全局单槽**里异步拷到设备, 连续调 N 次
     * 时 GPU 真正执行前那个槽已被最后一个 token 覆盖 —— 实测 4 行全拿到同一个嵌入。 */
    if (ok) ok = ds4_gpu_embed_tokens_hc_tensor(gb->batch_cur_hc,
                                                gb->prefill_tokens,
                                                e->model.map, e->model.size,
                                                e->weights.token_embd->abs_offset,
                                                (uint32_t)e->weights.token_embd->dim[1],
                                                n, DS4_N_EMBD, DS4_N_HC) != 0;
    /* 对照开关(DS4_MULTI_NO_SHARE=1): 每个会话整层都在自己的图上跑, 只共享输出头。
     * 用来把"驱动接线"和"分段共享"分开定位 —— 这条路应当与单路解码同残差档。 */
    static int no_share = -1;
    if (no_share < 0) no_share = getenv("DS4_MULTI_NO_SHARE") ? 1 : 0;
    if (ok && no_share) {
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_gpu_graph *gi = &ss[i]->graph;
            /* 直接嵌入各自图: 不能从 gb 取行 —— 会话 0 跑完 43 层后 gb->batch_cur_hc
             * 指针已被交换 43 次, 那里放的是它自己的层输出, 不是嵌入。 */
            if (i != 0) ok = ds4_gpu_embed_tokens_hc_tensor(gi->batch_cur_hc, gi->prefill_tokens,
                                                            e->model.map, e->model.size,
                                                            e->weights.token_embd->abs_offset,
                                                            (uint32_t)e->weights.token_embd->dim[1],
                                                            1u, DS4_N_EMBD, DS4_N_HC) != 0;
            for (uint32_t il = 0; ok && il < (uint32_t)DS4_N_LAYER; il++) {
                ok = metal_graph_encode_layer_attention_batch(gi, &e->model, &e->weights.layer[il],
                                                              il, (uint32_t)ss[i]->checkpoint.len, 1u) &&
                     metal_graph_encode_layer_ffn_batch(gi, &e->model, &e->weights.layer[il],
                                                        il, (uint32_t)ss[i]->checkpoint.len, 1u);
                if (ok) { ds4_gpu_tensor *t = gi->batch_cur_hc; gi->batch_cur_hc = gi->batch_next_hc; gi->batch_next_hc = t; }
            }
            if (ok && i != 0) ok = ds4_gpu_tensor_copy(gb->batch_cur_hc, (uint64_t)i * hc_bytes,
                                                       gi->batch_cur_hc, 0, hc_bytes) != 0;
        }
        goto multi_head;
    }
    for (uint32_t il = 0; ok && il < (uint32_t)DS4_N_LAYER; il++) {
        const ds4_layer_weights *lw = &e->weights.layer[il];
        const uint64_t qr_bytes = lw->attn_q_a->dim[1] * sizeof(float);
        /* ① 投影段: 共享图一次算 N 行(q_a/q_b/kv 权重只读一遍)。跳过 rope —— N 行来自
         *    不同会话, 位置各不相同, 下面按行用真实位置补。 */
        ok = metal_graph_encode_layer_attention_batch_stages(gb, &e->model, lw, il, 0u, n,
                                                             DS4_ATTN_STAGE_PRE | DS4_ATTN_STAGE_NOROPE);
        /* ①b 逐行 rope + kv fp8 量化(各用自己的位置) */
        {
            const uint32_t ratio_l = ds4_layer_compress_ratio(il);
            const bool comp_l = ratio_l != 0;
            const float fb = layer_rope_freq_base(il), fs = layer_rope_freq_scale(il);
            const float ef = (comp_l && DS4_ROPE_SCALE_FACTOR > 1.0f) ? 1.0f : 0.0f;
            float af = 1.0f;
            if (ef != 0.0f && fs > 0.0f) af /= 1.0f + 0.1f * logf(1.0f / fs);
            for (uint32_t i = 0; ok && i < n; i++) {
                const uint32_t p = (uint32_t)ss[i]->checkpoint.len;
                ds4_gpu_tensor *qv = ds4_gpu_tensor_view(gb->batch_q, (uint64_t)i * q_bytes, q_bytes);
                ds4_gpu_tensor *kvv = ds4_gpu_tensor_view(gb->batch_kv, (uint64_t)i * kv_bytes, kv_bytes);
                ok = qv && kvv &&
                     ds4_gpu_rope_tail_tensor(qv, 1u, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, p,
                                              comp_l ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                                              fb, fs, ef, af,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0 &&
                     ds4_gpu_rope_tail_tensor(kvv, 1u, 1u, DS4_N_HEAD_DIM, DS4_N_ROT, p,
                                              comp_l ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                                              fb, fs, ef, af,
                                              DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0 &&
                     ds4_gpu_dsv4_fp8_kv_quantize_tensor(kvv, 1u, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                ds4_gpu_tensor_free(kvv);
                ds4_gpu_tensor_free(qv);
            }
        }
        /* ② KV 段: 每个会话回自己的图算(压缩器/索引器/注意力状态都在那儿) */
        static int pre_per_sess = -1;
        if (pre_per_sess < 0) pre_per_sess = getenv("DS4_MULTI_PRE_PER_SESSION") ? 1 : 0;
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_gpu_graph *gi = &ss[i]->graph;
            if (pre_per_sess && i != 0) {
                /* 二分用: 会话自己跑投影段(要先把自己的 hc 行搬回来), 不走拷贝集 */
                ok = ds4_gpu_tensor_copy(gi->batch_cur_hc, 0, gb->batch_cur_hc,
                                         (uint64_t)i * hc_bytes, hc_bytes) != 0 &&
                     metal_graph_encode_layer_attention_batch_stages(gi, &e->model, lw, il,
                                                                    (uint32_t)ss[i]->checkpoint.len,
                                                                    1u, DS4_ATTN_STAGE_PRE);
            } else if (i != 0) {
                ok = ds4_gpu_tensor_copy(gi->batch_q, 0, gb->batch_q, (uint64_t)i * q_bytes, q_bytes) != 0 &&
                     ds4_gpu_tensor_copy(gi->batch_kv, 0, gb->batch_kv, (uint64_t)i * kv_bytes, kv_bytes) != 0 &&
                     ds4_gpu_tensor_copy(gi->batch_attn_norm, 0, gb->batch_attn_norm,
                                         (uint64_t)i * nm_bytes, nm_bytes) != 0 &&
                     ds4_gpu_tensor_copy(gi->batch_qr_norm, 0, gb->batch_qr_norm,
                                         (uint64_t)i * qr_bytes, qr_bytes) != 0;
            }
            if (ok) ok = metal_graph_encode_layer_attention_batch_stages(
                    gi, &e->model, lw, il, (uint32_t)ss[i]->checkpoint.len, 1u,
                    DS4_ATTN_STAGE_KV);
            if (ok && i != 0)
                ok = ds4_gpu_tensor_copy(gb->batch_heads, (uint64_t)i * q_bytes,
                                         gi->batch_heads, 0, q_bytes) != 0;
        }
        /* ③ 出口段 + FFN/MoE: 共享图一次算 N 行。出口段开头还有一次"逆 rope"(把注意力
         *    输出转回), 同样依赖每行的真实位置 ⇒ 先逐行补, 再让出口段跳过。 */
        {
            const uint32_t ratio_l = ds4_layer_compress_ratio(il);
            const bool comp_l = ratio_l != 0;
            const float fb = layer_rope_freq_base(il), fs = layer_rope_freq_scale(il);
            const float ef = (comp_l && DS4_ROPE_SCALE_FACTOR > 1.0f) ? 1.0f : 0.0f;
            float af = 1.0f;
            if (ef != 0.0f && fs > 0.0f) af /= 1.0f + 0.1f * logf(1.0f / fs);
            for (uint32_t i = 0; ok && i < n; i++) {
                ds4_gpu_tensor *hv = ds4_gpu_tensor_view(gb->batch_heads, (uint64_t)i * q_bytes, q_bytes);
                ok = hv && ds4_gpu_rope_tail_tensor(hv, 1u, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
                                                    (uint32_t)ss[i]->checkpoint.len,
                                                    comp_l ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
                                                    fb, fs, ef, af,
                                                    DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
                ds4_gpu_tensor_free(hv);
            }
        }
        static int post_per_sess = -1;
        if (post_per_sess < 0) post_per_sess = getenv("DS4_MULTI_POST_PER_SESSION") ? 1 : 0;
        if (ok && post_per_sess) {
            /* 二分用: 出口段也各自跑(需要把该行的 hc 与 hc_split 搬回去) */
            const uint64_t mix_bytes = (2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC) * sizeof(float);
            for (uint32_t i = 0; ok && i < n; i++) {
                ds4_gpu_graph *gi = &ss[i]->graph;
                if (i != 0) {
                    ok = ds4_gpu_tensor_copy(gi->batch_cur_hc, 0, gb->batch_cur_hc,
                                             (uint64_t)i * hc_bytes, hc_bytes) != 0 &&
                         ds4_gpu_tensor_copy(gi->batch_hc_split, 0, gb->batch_hc_split,
                                             (uint64_t)i * mix_bytes, mix_bytes) != 0;
                }
                if (ok) ok = metal_graph_encode_layer_attention_batch_stages(
                        gi, &e->model, lw, il, (uint32_t)ss[i]->checkpoint.len, 1u,
                        DS4_ATTN_STAGE_POST | DS4_ATTN_STAGE_NOROPE);
                if (ok && i != 0)
                    ok = ds4_gpu_tensor_copy(gb->batch_after_attn_hc, (uint64_t)i * hc_bytes,
                                             gi->batch_after_attn_hc, 0, hc_bytes) != 0;
            }
        } else if (ok) {
            ok = metal_graph_encode_layer_attention_batch_stages(gb, &e->model, lw, il, 0u, n,
                                                                 DS4_ATTN_STAGE_POST | DS4_ATTN_STAGE_NOROPE);
        }
        if (ok) ok = metal_graph_encode_layer_ffn_batch(gb, &e->model, lw, il, 0u, n);
        if (ok) {   /* 与 encode_layer_batch 同款 cur/next 交换 */
            ds4_gpu_tensor *tmp = gb->batch_cur_hc;
            gb->batch_cur_hc = gb->batch_next_hc;
            gb->batch_next_hc = tmp;
        }
    }
multi_head:
    if (ok) ok = metal_graph_encode_output_head_batch(gb, &e->model, &e->weights,
                                                      n, e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && !gb->spec_logits) {
        if (errlen) snprintf(err, errlen, "multi-session batch needs the batch-logits buffer");
        ok = false;
    }
    static float *mb_logits = NULL; static uint32_t mb_cap = 0;
    if (ok && mb_cap < n) {
        free(mb_logits);
        mb_logits = xmalloc((size_t)n * DS4_N_VOCAB * sizeof(float));
        mb_cap = n;
    }
    if (ok) ok = ds4_gpu_tensor_read(gb->spec_logits, 0, mb_logits,
                                     (uint64_t)n * DS4_N_VOCAB * sizeof(float)) != 0;
    if (!ok) {
        if (errlen) snprintf(err, errlen, "%s multi-session batch failed",
                             ds4_backend_name(e->backend));
        for (uint32_t i = 0; i < n; i++) ss[i]->checkpoint_valid = false;
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        memcpy(ss[i]->logits, mb_logits + (uint64_t)i * DS4_N_VOCAB,
               (size_t)DS4_N_VOCAB * sizeof(float));
        token_vec_push(&ss[i]->checkpoint, tokens[i]);
        ss[i]->checkpoint_valid = true;
    }
    return 0;
#endif
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    if (!g_prof.enabled) return ds4_session_eval_internal(s, token, true, err, errlen);
    const double t0 = now_sec();
    int rc = ds4_session_eval_internal(s, token, true, err, errlen);
    if (rc == 0) ds4_profile_add_decode(1, now_sec() - t0);
    return rc;
}

/* Append N KNOWN tokens to the live KV in ONE layer-major batch (2026-07-14).
 *
 * Motivation: the guided tool-call primer injects dozens of tokens whose values
 * the SERVER already knows (the DSML structure) -- no sampling involved. Feeding
 * them one at a time pays the full per-token decode-bandwidth wall each
 * (measured dual-host: inject=28.9 s for 39 structure tokens vs a 352-token
 * batched prefill at 9.9 t/s -- the same expert bytes, 7x the throughput,
 * because a K-token batch reads the backbone ONCE; see the copy-spec note above
 * for the same physics).
 *
 * This is the batch path WITHOUT the speculative accept/rollback machinery: the
 * tokens are given, not drafted, so every one commits. Distributed sessions send
 * a single multi-token WORK span; Metal/CPU sessions use the same batched entry
 * the resume-prefill already uses. Falls back to a per-token loop on any error
 * so the caller never needs a second code path. Returns 0 on success. */
int ds4_session_eval_span(ds4_session *s, const int *tokens, int n,
                          char *err, size_t errlen) {
    if (!s || !tokens || n <= 0) {
        if (errlen) snprintf(err, errlen, "invalid span eval request");
        return 1;
    }
    if (ds4_session_pos(s) + n >= s->ctx_size) {
        if (errlen) snprintf(err, errlen, "span exceeds context");
        return 1;
    }
    if (n == 1) return ds4_session_eval(s, tokens[0], err, errlen);

    /* Batched path: extend the checkpoint timeline with the known span. The
     * session's own sync entry drives the backend-specific batch (dist: one
     * WORK span; metal: resume-prefill), and it is exactly the prompt-prefill
     * path, so KV rows finalize in the same order as a cold prompt. */
    ds4_tokens want = {0};
    if (s->checkpoint_valid)
        for (int i = 0; i < s->checkpoint.len; i++) ds4_tokens_push(&want, s->checkpoint.v[i]);
    for (int i = 0; i < n; i++) ds4_tokens_push(&want, tokens[i]);
    const double t0 = now_sec();
    int rc = ds4_session_sync_internal(s, &want, err, errlen);
    ds4_tokens_free(&want);
    if (rc == 0) {
        if (g_prof.enabled) ds4_profile_add_decode((uint64_t)n, now_sec() - t0);
        return 0;
    }
    /* Any batch failure: fall back to the exact per-token semantics. */
    for (int i = 0; i < n; i++)
        if (ds4_session_eval(s, tokens[i], err, errlen) != 0) return 1;
    return 0;
}

/* copy-spec 单机整族已删除(2026-08-05 用户裁决)。 */

/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let MTP recursively draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    if (s->distributed) {
        if (!accepted) return 0;
        if (!s->checkpoint_valid) {
            if (errlen) snprintf(err, errlen, "distributed decode requires a valid checkpoint");
            return -1;
        }
        /* docs/archive/mtp.md Phase 1: cross-machine MTP speculation. The driver commits
         * first_token + verified drafts into the session checkpoint and returns
         * the committed count; s->logits is left predicting the next token. */
        int cap = accepted_cap < max_tokens ? accepted_cap : max_tokens;
        int n = ds4_dist_session_eval_speculative(s->distributed, s, &s->checkpoint,
                                                  first_token, eos_token,
                                                  accepted, cap, s->logits,
                                                  err, errlen);
        if (n < 0) { s->checkpoint_valid = false; return -1; }
        return n;
    }
    if (ds4_session_is_cpu(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "GPU support is not compiled in");
    return -1;
#else
    ds4_engine *e = s->engine;

    /* copy-spec 整族已删除(2026-08-05 用户裁决: 环境变量硬编码型行为机制, 与
     * auto-arm 同判)。无显式 MTP draft 模型 => 纯单 token 平解码; 投机只剩
     * 显式配置的 MTP drafter 一条路。 */
    /* DSpark 投机主循环(DS4_DSPARK_SPEC=1, 2026-08-18): draft 块(5)+bonus 走 6 位
     * verify 批; 接受链 argmax 对照; 部分接受= state 恢复+接受位重放(官方
     * checkpoint-restore 口径)。verify 批复用 batch 层包装 ⇒ mh 抓取/drafter 建窗自动。 */
    if (getenv("DS4_DSPARK_PROBE")) {
        static int gate_diag = 0;
        if (!gate_diag) { gate_diag = 1;
            fprintf(stderr, "ds4: [dspark-gate] ready=%d capture=%d greedy=%d env=%d\n",
                    (int)e->dspark.ready, s->graph.dspark_capture, s->spec_greedy,
                    getenv("DS4_DSPARK_SPEC") != NULL);
        }
    }
    if (e->dspark.ready && s->graph.dspark_capture && s->spec_greedy &&
        getenv("DS4_DSPARK_SPEC") != NULL) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        int n_acc = 0;
        accepted[n_acc++] = first_token;
        static float *row_logits = NULL;
        if (!row_logits) row_logits = xmalloc(6ull * DS4_N_VOCAB * sizeof(float));
        /* 候选数可调(2026-08-21 调度杠杆): verify 是字节受限, 而专家并集随候选数增长
         * (实测 6 候选=22.1/36 唯一专家)。每字节产出 acc/bytes 在 k=3-4 处更优。 */
        static uint32_t spec_k = 0u;
        if (spec_k == 0u) {
            const char *e = getenv("DS4_SPEC_CAND");
            /* 默认 3(2026-08-21 扫参): 每 token 毫秒 k=6 48 / k=4 44 / k=3 38.7 / k=2 39.8。
             * verify 是字节受限且专家并集随候选数增长(实测 6 候选=22.1 唯一专家), 多草稿
             * 的边际接受收益跑不过边际字节成本。verify 语义与 k 无关 ⇒ 质量不受影响。 */
            spec_k = e ? (uint32_t)atoi(e) : 3u;
            if (spec_k < 2u) spec_k = 2u;
            if (spec_k > 6u) spec_k = 6u;
        }
        const bool spec_prof = getenv("DS4_SPEC_PROF") != NULL;
        double pf_draft = 0, pf_snap = 0, pf_verify = 0, pf_replay = 0, pf_misc = 0, pf_round_wall = 0;
        uint32_t pf_rounds = 0; double pf_t0 = 0;
        while (n_acc < max_tokens && n_acc + (int)DS4_DSPARK_BLK + 1 <= accepted_cap) {
            if (spec_prof) pf_t0 = now_sec();
            int next = 0; float best = -1e30f;
            for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                if (s->logits[v] > best) { best = s->logits[v]; next = (int)v; }
            if (next == eos_token) break;
            int cand[6];
            cand[0] = next;
            int ids[DS4_DSPARK_BLK] = {0};
            const uint32_t pos_now = (uint32_t)s->checkpoint.len;
            if (spec_prof) { pf_misc += now_sec() - pf_t0; pf_t0 = now_sec(); }
            /* 置信调度(DS4_SPEC_SCHED=1, 论文 2607.05147 Alg.1 单请求版):
             * 草稿一次出满块并拿到逐位置信 c_i; 前缀存活率 a_j = ∏_{i<=j} c_i 就是"验第 j 个
             * 候选能多拿到的期望 token 数"。多验一个候选的边际成本是 m 毫秒(在线最小二乘
             * 从 (k-1, verify_ms) 拟合), 当前吞吐 T = 已接受 token / 已用毫秒。只有
             * a_j > T*m 时这个候选才划算 —— 这正是把"能不能白送"这个物理事实写进调度。 */
            static int sched = -1;
            if (sched < 0) sched = getenv("DS4_SPEC_SCHED") ? 1 : 0;
            static double sch_tok = 0.0, sch_ms = 0.0;          /* 在线吞吐 */
            static double rg_n = 0, rg_x = 0, rg_y = 0, rg_xx = 0, rg_xy = 0;  /* 边际回归 */
            static double sch_m = 12.0;                          /* 每候选边际 ms */
            /* 在线校准(论文的 STS 在线版): 原始置信头没针对"q2 drafter + 贪心验证"标定,
             * 用自己的接受结果做逐位置乘性校正 g_j = 实测接受 / 预测和。低估就放大, 高估
             * 就收缩 —— 调度阈值才对得上真实的边际收益。 */
            static double cal_sum[DS4_DSPARK_BLK] = {0};
            static double cal_hit[DS4_DSPARK_BLK] = {0};
            static uint32_t cal_n[DS4_DSPARK_BLK] = {0};
            /* 投机开关闸(2026-08-21): drafter 不是白送的 —— 一轮草稿 ~9ms, 每个候选边际
             * ~12ms。文本难预测时(bash/概念解释)接受率掉到 1.8, 投机反而比纯解码慢 20%。
             * 在线对比两种模式的实测 token/ms, 谁快用谁, 并按固定比例回探另一种以便文本
             * 变好预测时能切回来。纯解码轮直接走单 token 解码路 = 无损且零草稿开销。 */
            static double md_tok[2] = {0, 0}, md_ms[2] = {0, 0};   /* 0=纯解码 1=投机 */
            static uint32_t md_n[2] = {0, 0}, md_round = 0;
            int mode = 1;
            if (sched) {
                /* 起步先给投机 16 轮把 T/m/校准跑热, 之后按实测吞吐择优, 每 32 轮回探 1 轮。 */
                const int explore = (md_round % 32u) == 31u;
                if (md_round >= 16u && md_n[0] >= 3u && md_n[1] >= 8u) {
                    const double t0 = md_tok[0] / md_ms[0], t1 = md_tok[1] / md_ms[1];
                    mode = (t1 >= t0 * 0.98) ? 1 : 0;   /* 平手偏投机(它还带 acc 上升空间) */
                } else if (md_round >= 16u && md_n[0] < 3u) {
                    mode = 0;                            /* 先取几个纯解码样本 */
                }
                if (explore) mode = 1 - mode;
                md_round++;
            }
            const double mode_t0 = now_sec();
            if (sched && mode == 0) {
                /* 纯解码轮: 提交 next, 不草稿不批验证 */
                if (ds4_session_eval(s, next, err, errlen) != 0) { s->checkpoint_valid = false; return -1; }
                accepted[n_acc++] = next;
                const double dt = (now_sec() - mode_t0) * 1e3;
                md_tok[0] += 1.0; md_ms[0] += dt; md_n[0]++;
                /* 不喂 sch_tok/sch_ms: k 调度器的 T 是"投机模式下的吞吐", 掺进纯解码轮会
                 * 抬高阈值 → k 变小 → 投机更差 → 更偏纯解码, 形成死亡螺旋。 */
                if (getenv("DS4_SPEC_SCHED_LOG") && (md_round % 32u) == 0)
                    fprintf(stderr, "ds4: [sched-mode] 纯解码 %.4f tok/ms(n=%u) vs 投机 %.4f(n=%u)\n",
                            md_n[0] ? md_tok[0] / md_ms[0] : 0.0, md_n[0],
                            md_n[1] ? md_tok[1] / md_ms[1] : 0.0, md_n[1]);
                continue;
            }

            float conf[DS4_DSPARK_BLK] = {0};
            uint32_t draft_n = sched ? (uint32_t)DS4_DSPARK_BLK : spec_k - 1u;
            if (!metal_graph_dspark_step_n(&s->graph, &e->model, &e->weights, &e->dspark,
                                           next, pos_now - 1u, ids, draft_n,
                                           sched ? conf : NULL)) {
                if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-gate] step FAILED\n");
                break;
            }
            uint32_t round_k = spec_k;
            if (sched) {
                const double T = (sch_ms > 1.0) ? (sch_tok / sch_ms) : 0.030;   /* token/ms */
                const double thr = T * sch_m;
                double a = 1.0;
                uint32_t adm = 0;
                for (uint32_t j = 0; j < draft_n; j++) {
                    double c = (double)conf[j];
                    if (cal_n[j] >= 8u && cal_sum[j] > 1e-6) {
                        double g = (cal_hit[j] + 1.0) / (cal_sum[j] + 1.0);
                        c *= g;
                        if (c > 0.999) c = 0.999;
                        if (c < 0.001) c = 0.001;
                    }
                    a *= c;
                    if (a <= thr) break;
                    adm++;
                }
                round_k = adm + 1u;
                if (round_k < 2u) round_k = 2u;
                if (round_k > (uint32_t)DS4_DSPARK_BLK + 1u) round_k = (uint32_t)DS4_DSPARK_BLK + 1u;
                if (getenv("DS4_SPEC_SCHED_LOG"))
                    fprintf(stderr, "ds4: [sched] c=[%.2f %.2f %.2f %.2f %.2f] T=%.4f m=%.1f 阈=%.3f k=%u\n",
                            conf[0], conf[1], conf[2], conf[3], conf[4], T, sch_m, thr, round_k);
            }
            for (uint32_t i = 0; i + 1u < round_k; i++) cand[1 + i] = ids[i];
            if (spec_prof) { pf_draft += now_sec() - pf_t0; pf_t0 = now_sec(); }
            if (!metal_graph_dspark_state_snapshot(&s->graph)) {
                if (getenv("DS4_DSPARK_PROBE")) fprintf(stderr, "ds4: [dspark-gate] snapshot FAILED\n");
                break;
            }
            if (spec_prof) { pf_snap += now_sec() - pf_t0; pf_t0 = now_sec(); }
            /* 诊断路径(DS4_SPEC_SEQ_VERIFY=1): verify 不走批, 而是逐候选走单 token
             * 解码路 —— 与纯解码逐字节同一条 kernel 链。没有批的加速, 只用来量
             * drafter 对"真解码"的接受率上限, 以及判定批 verify 是否引入了偏差。
             * 状态天然随接受推进, 不需要 snapshot/restore/快进。 */
            const bool xcheck = getenv("DS4_SPEC_XCHECK") != NULL;
            const bool seq_verify = xcheck || getenv("DS4_SPEC_SEQ_VERIFY") != NULL;
            int seq_acc = 1;
            static float *xc_batch = NULL;
            if (xcheck) {
                /* 同一位置先走批 verify(标签 b), 再走单 token 解码路(标签 d), 逐行比 logits。
                 * 批的结果只用于对账, 提交仍走解码路 ⇒ 轨迹与纯解码相同。 */
                if (!xc_batch) xc_batch = xmalloc(6ull * DS4_N_VOCAB * sizeof(float));
                g_dump_tag = "b";
                s->graph.spec_comp_capture = 1;
                const int rc = ds4_session_verify_batch_argmax(s, cand, round_k, pos_now,
                                                              0u, (uint32_t)DS4_N_LAYER - 1u,
                                                              NULL, row_logits, err, errlen);
                s->graph.spec_comp_capture = 0;
                g_dump_tag = "";
                if (rc != 0) { s->checkpoint_valid = false; return -1; }
                memcpy(xc_batch, row_logits, (size_t)round_k * DS4_N_VOCAB * sizeof(float));
                if (!metal_graph_dspark_state_restore(&s->graph)) break;
                s->checkpoint.len = (int)pos_now;
            }
            if (seq_verify) {
                if (xcheck) g_dump_tag = "d";
                for (uint32_t i = 0; i < round_k; i++) {
                    if (ds4_session_eval(s, cand[i], err, errlen) != 0) {
                        s->checkpoint_valid = false;
                        return -1;
                    }
                    memcpy(row_logits + (uint64_t)i * DS4_N_VOCAB, s->logits,
                           (size_t)DS4_N_VOCAB * sizeof(float));
                    if (i + 1u >= round_k) break;
                    int am = 0; float bb = -1e30f;
                    for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                        if (s->logits[v] > bb) { bb = s->logits[v]; am = (int)v; }
                    if (am != cand[i + 1]) break;
                    seq_acc++;
                }
                if (xcheck) {
                    g_dump_tag = "";
                    static uint32_t xc_n = 0, xc_same = 0; static double xc_dmax = 0.0;
                    for (uint32_t i = 0; i < round_k; i++) {
                        const float *rb = xc_batch + (uint64_t)i * DS4_N_VOCAB;
                        const float *rd = row_logits + (uint64_t)i * DS4_N_VOCAB;
                        int ab = 0, ad = 0; float bb = -1e30f, bd = -1e30f; double dmax = 0.0;
                        for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++) {
                            if (rb[v] > bb) { bb = rb[v]; ab = (int)v; }
                            if (rd[v] > bd) { bd = rd[v]; ad = (int)v; }
                            const double d = fabs((double)rb[v] - (double)rd[v]);
                            if (d > dmax) dmax = d;
                        }
                        xc_n++; if (ab == ad) xc_same++;
                        if (dmax > xc_dmax) xc_dmax = dmax;
                        if (i == 0 && ab != ad)
                            fprintf(stderr, "ds4: [xcheck] pos=%u row0 批argmax=%d 解码argmax=%d max|Δlogit|=%.4g\n",
                                    pos_now, ab, ad, dmax);
                    }
                    if ((xc_n % 16u) == 0)
                        fprintf(stderr, "ds4: [xcheck] 行数=%u argmax一致=%.1f%% 全程max|Δlogit|=%.4g\n",
                                xc_n, 100.0 * xc_same / xc_n, xc_dmax);
                }
            } else {
            s->graph.spec_comp_capture = 1;   /* verify 批捕获压缩器输入行(快进用) */
            if (ds4_session_verify_batch_argmax(s, cand, round_k, pos_now,
                                                0u, (uint32_t)DS4_N_LAYER - 1u,
                                                NULL, row_logits, err, errlen) != 0) {
                s->graph.spec_comp_capture = 0;
                s->checkpoint_valid = false;
                return -1;
            }
            s->graph.spec_comp_capture = 0;
            }
            if (spec_prof) { pf_verify += now_sec() - pf_t0; pf_t0 = now_sec(); }
            if (getenv("DS4_DSPARK_DIAG")) {
                /* drafter 主干 top1(markov 前口径不可得, 打 markov 后 d0) vs 主模型 row0 top1 */
                int am0 = 0; float b0 = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (row_logits[v] > b0) { b0 = row_logits[v]; am0 = (int)v; }
                static int dn = 0;
                if (dn < 8) {
                    float mh2[2] = {0}, mx2[2] = {0}, lg2[2] = {0};
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_hidden, 0, mh2, sizeof(mh2));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_x, 0, mx2, sizeof(mx2));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_logits, 0, lg2, sizeof(lg2));
                    fprintf(stderr, "ds4: [dspark-diag] verify_row0_top1=%d draft0=%d draft=[%d %d %d %d %d] mh=%.3g %.3g mx=%.3g %.3g lg=%.3g %.3g\n",
                            am0, cand[1], cand[1], cand[2], cand[3], cand[4], cand[5],
                            mh2[0], mh2[1], mx2[0], mx2[1], lg2[0], lg2[1]);
                    dn++;
                }
            }
            int acc = 1;
            for (int i = 0; !seq_verify && i + 1 < (int)round_k; i++) {
                int am = 0; float bb = -1e30f;
                const float *row = row_logits + (uint64_t)i * DS4_N_VOCAB;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (row[v] > bb) { bb = row[v]; am = (int)v; }
                if (am != cand[i + 1]) break;
                acc++;
            }
            if (seq_verify) acc = seq_acc;
            static int raw_restore_on = -1;
            if (raw_restore_on < 0) raw_restore_on = getenv("DS4_SPEC_NO_RAW_RESTORE") ? 0 : 1;
            if (raw_restore_on && !seq_verify && acc < (int)round_k &&
                !metal_graph_spec_raw_restore(&s->graph, pos_now, (uint32_t)acc, round_k)) {
                if (errlen) snprintf(err, errlen, "spec raw KV restore failed");
                s->checkpoint_valid = false;
                return -1;
            }
            if (!seq_verify && !metal_graph_dspark_win_commit(&s->graph, pos_now, (uint32_t)acc)) {
                if (errlen) snprintf(err, errlen, "dspark window commit failed");
                s->checkpoint_valid = false;
                return -1;
            }
            /* timeline 已 commit 6 位 → 截到接受数 */
            if (!seq_verify) s->checkpoint.len = (int)(pos_now + (uint32_t)acc);
            if (!seq_verify && acc < (int)round_k) {
                if (!metal_graph_dspark_state_restore(&s->graph)) break;
                if (getenv("DS4_SPEC_REPLAY_OLD")) {
                    /* A/B 对照: 旧 replay 全前向路径 */
                    s->checkpoint.len = (int)pos_now;
                    if (ds4_session_verify_batch_argmax(s, cand, (uint32_t)acc, pos_now,
                                                        0u, (uint32_t)DS4_N_LAYER - 1u,
                                                        NULL, row_logits, err, errlen) != 0) {
                        s->checkpoint_valid = false;
                        return -1;
                    }
                } else if (!metal_graph_spec_comp_fastforward(&s->graph, &e->model, &e->weights,
                                                              pos_now, (uint32_t)acc)) {
                    /* replay 消除(2026-08-20): KV raw 行 verify 已写好且 restore 不动;
                     * 压缩器/indexer 态用 verify 捕获的输入行快进 acc 位。 */
                    if (errlen) snprintf(err, errlen, "spec compressor fast-forward failed");
                    s->checkpoint_valid = false;
                    return -1;
                }
                s->checkpoint.len = (int)(pos_now + (uint32_t)acc);
            }
            memcpy(s->logits, row_logits + (uint64_t)(acc - 1) * DS4_N_VOCAB,
                   (size_t)DS4_N_VOCAB * sizeof(float));
            if (spec_prof) {
                pf_replay += now_sec() - pf_t0; pf_t0 = now_sec();
                pf_rounds++;
                { static double last_top = 0.0; const double nowv = now_sec();
                  if (last_top > 0.0) pf_round_wall += nowv - last_top;
                  last_top = nowv; }
                if ((pf_rounds & 7u) == 0) {
                    fprintf(stderr, "ds4: [spec-prof] rounds=%u win8_ms: draft=%.1f snap=%.1f verify=%.1f replay=%.1f misc=%.1f | round_wall=%.1f\n",
                            pf_rounds, pf_draft * 1e3 / 8.0, pf_snap * 1e3 / 8.0,
                            pf_verify * 1e3 / 8.0, pf_replay * 1e3 / 8.0, pf_misc * 1e3 / 8.0,
                            pf_round_wall * 1e3 / 8.0);
                    pf_draft = pf_snap = pf_verify = pf_replay = pf_misc = pf_round_wall = 0;   /* 滑窗 */
                }
            }
            if (sched) {
                const double dt_mode = (now_sec() - mode_t0) * 1e3;
                md_tok[1] += (double)acc; md_ms[1] += dt_mode; md_n[1]++;
                /* 校准喂数: 草稿位 j 被真正验证过(前缀全接受)才计数; j = acc-1 是被拒的那位。 */
                for (uint32_t j = 0; j + 1u < round_k && j < (uint32_t)acc; j++) {
                    cal_n[j]++; cal_sum[j] += (double)conf[j];
                    if ((int)j < acc - 1) cal_hit[j] += 1.0;
                }
                if (getenv("DS4_SPEC_SCHED_LOG")) {
                    static uint32_t cl = 0;
                    if (((++cl) % 32u) == 0) {
                        fprintf(stderr, "ds4: [sched-cal]");
                        for (uint32_t j = 0; j < (uint32_t)DS4_DSPARK_BLK; j++)
                            fprintf(stderr, " p%u: 预测%.2f 实测%.2f(n=%u)", j + 1,
                                    cal_n[j] ? cal_sum[j] / cal_n[j] : 0.0,
                                    cal_n[j] ? cal_hit[j] / cal_n[j] : 0.0, cal_n[j]);
                        fprintf(stderr, "\n");
                    }
                }
                /* 在线标定: 本轮墙钟与接受数喂吞吐; (k-1, verify_ms) 喂边际最小二乘。
                 * x 有方差后才用拟合值, 否则保持上一次的 m(初值 12ms)。 */
                static double last_top2 = 0.0;
                const double nowv2 = now_sec();
                if (last_top2 > 0.0) {
                    const double dt_ms = (nowv2 - last_top2) * 1e3;
                    sch_tok += (double)acc; sch_ms += dt_ms;
                    const double x = (double)(round_k - 1u);
                    rg_n += 1; rg_x += x; rg_y += dt_ms; rg_xx += x * x; rg_xy += x * dt_ms;
                    const double den = rg_n * rg_xx - rg_x * rg_x;
                    if (rg_n >= 8 && den > 1e-6) {
                        const double slope = (rg_n * rg_xy - rg_x * rg_y) / den;
                        if (slope > 1.0 && slope < 60.0) sch_m = slope;
                    }
                }
                last_top2 = nowv2;
            }
            if (getenv("DS4_DSPARK_STAT")) {
                static uint32_t st_rounds = 0, st_acc = 0;
                st_rounds++; st_acc += (uint32_t)acc;
                if ((st_rounds & 7u) == 0)
                    fprintf(stderr, "ds4: [dspark-stat] rounds=%u avg_acc=%.2f\n",
                            st_rounds, (double)st_acc / st_rounds);
            }
            /* mh: verify/重放批的末接受位 → dspark_main_hidden(3 slot 连续拷贝)。
             * seq 诊断路径下单 token 解码自己写 mh, 不需要也不能从批里拷。 */
            for (uint32_t sl = 0; !seq_verify && sl < 3u; sl++)
                (void)ds4_gpu_tensor_copy(s->graph.dspark_main_hidden,
                                          (uint64_t)sl * DS4_N_EMBD * sizeof(float),
                                          s->graph.dspark_pf_hidden,
                                          ((uint64_t)(acc - 1) * 3u + sl) * DS4_N_EMBD * sizeof(float),
                                          (uint64_t)DS4_N_EMBD * sizeof(float));
            for (int i = 0; i < acc && n_acc < accepted_cap; i++) accepted[n_acc++] = cand[i];
            if (acc >= 1 && cand[acc - 1] == eos_token) break;
        }
        return n_acc;
    }
    if (!e->mtp_ready) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        /* DSpark 探针(DS4_DSPARK_PROBE=1): eval 后 main_hidden 已被 graph 抓取,
         * 跑一次 drafter 块打印草稿(活性/质量人工判读, 主循环接线前的最小验证)。 */
        if (getenv("DS4_DSPARK_PROBE")) {
            static int diag1 = 0;
            if (!diag1) { diag1 = 1;
                fprintf(stderr, "ds4: [dspark-diag] ready=%d g=%p capture=%d buf=%p\n",
                        (int)e->dspark.ready, (void *)&s->graph, s->graph.dspark_capture, (void *)s->graph.dspark_main_hidden);
            }
        }
        if (e->dspark.ready && s->graph.dspark_capture && getenv("DS4_DSPARK_PROBE")) {
            /* 零轨迹噪声命中统计(2026-08-21): PROBE 下草稿不被接受 ⇒ 生成序列与纯解码
             * 完全一致, 三种 drafter 配置走同一条轨迹 ⇒ 命中率可直接比, 不含 acc 那种
             * ±0.09 的轨迹漂移噪声。上一轮的草稿首位 vs 本轮真实 token。 */
            /* 逐位命中(2026-08-21): 首位 91% 但 SPEC acc 仅 2.0/3 ⇒ 推出第二位接受率
             * ~9%, 与 probe 原始输出"常连中 5 位"矛盾 ⇒ 需要逐位真值来定位是能力还是
             * SPEC 路径的问题。ring 存最近一次草稿的 5 位, 与后续 5 个真实 token 比。 */
            static int pend = -1;
            static uint32_t hit = 0, tot = 0;
            static int dr[DS4_DSPARK_BLK] = {-1,-1,-1,-1,-1};
            static int dr_age = 99;
            static uint32_t phit[DS4_DSPARK_BLK] = {0}, ptot[DS4_DSPARK_BLK] = {0};
            static int probe_n = 0;
            if (probe_n < 100000) {
                int next = 0; float best = -1e30f;
                for (uint32_t v = 0; v < (uint32_t)DS4_N_VOCAB; v++)
                    if (s->logits[v] > best) { best = s->logits[v]; next = (int)v; }
                /* 上一轮草稿首位预测的正是本轮的 next(不是本轮的输入 token) */
                if (pend >= 0) { tot++; if (pend == next) hit++; }
                if (dr_age < (int)DS4_DSPARK_BLK) {
                    ptot[dr_age]++;
                    if (dr[dr_age] == next) phit[dr_age]++;
                    else dr_age = 99;   /* 链式: 一旦某位错, 后续位不再计(与 SPEC 接受语义一致) */
                    if (dr_age != 99) dr_age++;
                }
                if (getenv("DS4_DSPARK_PROBE_STAT") && tot && (tot % 64u) == 0) {
                    fprintf(stderr, "ds4: [probe-hit] %u/%u = %.1f%% | 逐位链式:", hit, tot, 100.0 * hit / tot);
                    for (uint32_t q = 0; q < DS4_DSPARK_BLK; q++)
                        fprintf(stderr, " p%u=%.0f%%(%u/%u)", q + 1,
                                ptot[q] ? 100.0 * phit[q] / ptot[q] : 0.0, phit[q], ptot[q]);
                    fprintf(stderr, "\n");
                }
                int ids[DS4_DSPARK_BLK] = {0};
                if (metal_graph_dspark_step(&s->graph, &e->model, &e->weights, &e->dspark,
                                            next, (uint32_t)(s->checkpoint.len - 1), ids)) {
                    pend = ids[0];
                    if (dr_age >= (int)DS4_DSPARK_BLK) {   /* 上一条草稿已用完/断链, 换新的 */
                        for (uint32_t q = 0; q < DS4_DSPARK_BLK; q++) dr[q] = ids[q];
                        dr_age = 0;
                    }
                    float mh[4] = {0}, lg[4] = {0}, mx[4] = {0};
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_hidden, 0, mh, sizeof(mh));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_main_x, 0, mx, sizeof(mx));
                    (void)ds4_gpu_tensor_read(s->graph.dspark_logits, 0, lg, sizeof(lg));
                    if (probe_n < 8)
                        fprintf(stderr, "ds4: [dspark] pos=%d next=%d draft= %d %d %d %d %d | mh=%.3g %.3g mx=%.3g %.3g lg=%.3g %.3g\n",
                                s->checkpoint.len - 1, next, ids[0], ids[1], ids[2], ids[3], ids[4],
                                mh[0], mh[1], mx[0], mx[1], lg[0], lg[1]);
                } else if (probe_n < 8) {
                    fprintf(stderr, "ds4: [dspark] step failed at pos=%d\n", s->checkpoint.len - 1);
                }
                probe_n++;
            }
        }
        return 1;
    }
    /* MTP 投机整族已删除(2026-08-05 用户裁决: Go 定型优化)。mtp_ready 恒 false,
     * 上面的 plain 路径即全部行为; 此处永不可达。 */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

