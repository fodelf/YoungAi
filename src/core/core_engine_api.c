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

