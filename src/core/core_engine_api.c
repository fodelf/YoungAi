/* core_engine_api.c — engine API 门面/实例锁 (机械拆分自 ds4.c, 重构阶段4)。 */
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

/* 取料入口 setter 家族(ds4.h 同名注释): CLI 参数是唯一对外入口, 存进程内全局,
 * capture/终审仪器直接读 getter——env 传输层已删(2026-08-31 env 大扫除收口)。 */
static const char *g_tool_cap_dir;
static const char *g_tool_cap_layers;
static const char *g_tool_eval_ids;
static const char *g_tool_eval_hdump;
static const char *g_tool_eval_logits;
static const char *g_tool_eval_nll;
static const char *g_tool_eval_topk_out;
static int g_tool_eval_topk;
static const char *g_tool_amp_anchor;
static int g_tool_amp_anchor_route;
static int g_tool_eval_no_bos;
static int g_tool_multi_bench;
static int g_tool_prefill_chunk = -1;   /* <0 = 自动(按后端/URL prompt 长度) */
void ds4_tool_set_cap_dir(const char *p)     { g_tool_cap_dir = p; }
const char *ds4_tool_cap_dir(void)           { return g_tool_cap_dir; }
void ds4_tool_set_cap_layers(const char *p)  { g_tool_cap_layers = p; }
const char *ds4_tool_cap_layers(void)        { return g_tool_cap_layers; }
void ds4_tool_set_eval_ids(const char *p)    { g_tool_eval_ids = p; }
const char *ds4_tool_eval_ids(void)          { return g_tool_eval_ids; }
void ds4_tool_set_eval_hdump(const char *p)  { g_tool_eval_hdump = p; }
const char *ds4_tool_eval_hdump(void)        { return g_tool_eval_hdump; }
void ds4_tool_set_eval_logits(const char *p) { g_tool_eval_logits = p; }
const char *ds4_tool_eval_logits(void)       { return g_tool_eval_logits; }
void ds4_tool_set_eval_nll(const char *p)    { g_tool_eval_nll = p; }
const char *ds4_tool_eval_nll(void)          { return g_tool_eval_nll; }
void ds4_tool_set_eval_topk(int k, const char *p) { g_tool_eval_topk = k; g_tool_eval_topk_out = p; }
int  ds4_tool_eval_topk(void)                { return g_tool_eval_topk; }
const char *ds4_tool_eval_topk_out(void)     { return g_tool_eval_topk_out; }
void ds4_tool_set_eval_no_bos(int v)         { g_tool_eval_no_bos = v; }
int  ds4_tool_eval_no_bos(void)              { return g_tool_eval_no_bos; }
void ds4_tool_set_amp_anchor(const char *p, int route_on) { g_tool_amp_anchor = p; g_tool_amp_anchor_route = route_on; }
const char *ds4_tool_amp_anchor(void)        { return g_tool_amp_anchor; }
int  ds4_tool_amp_anchor_route(void)         { return g_tool_amp_anchor_route; }
void ds4_tool_set_multi_bench(int n)         { g_tool_multi_bench = n; }
int  ds4_tool_multi_bench(void)              { return g_tool_multi_bench; }
void ds4_tool_set_prefill_chunk(int chunk)   { g_tool_prefill_chunk = chunk; }
int  ds4_tool_prefill_chunk(void)            { return g_tool_prefill_chunk; }

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
    const char *path = "/tmp/ds4.lock";

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

