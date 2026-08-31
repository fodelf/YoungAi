/* core_util.c — die/分配守卫/xmalloc族/内存看门狗/日志 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

/* 内存护栏两道线(同值不同义, 别合并):
 *  - 看门狗越线 = 预算 90%: 运行时 phys_footprint 实测, 早于 100% abort, 抢在
 *    wired 暴涨/页抖动把机器拖死之前(12G 红线是系统级最高约束)。
 *  - L1 静态闸 = 预算 85%: 启动时的静态估计(模型驻留+KV/scratch 估计), 留 15%
 *    余量吸收估计误差与运行期增长; 运行时仍由看门狗兜底。 */
#define DS4_WATCHDOG_TRIP_FRAC 0.90
#define DS4_L1_GATE_FRAC       0.85

/* =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */

void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}


void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool g_alloc_guard_enabled;
static const char *g_alloc_guard_phase;

void ds4_alloc_guard_begin(const char *phase) {
    g_alloc_guard_phase = phase;
    g_alloc_guard_enabled = true;
}

void ds4_alloc_guard_end(void) {
    g_alloc_guard_enabled = false;
    g_alloc_guard_phase = NULL;
}

static void ds4_alloc_guard_check(const char *op, size_t size) {
    if (!g_alloc_guard_enabled) return;
    fprintf(stderr,
            "ds4: internal allocation during %s: %s(%zu). "
            "CPU decode is expected to reuse preallocated scratch buffers.\n",
            g_alloc_guard_phase ? g_alloc_guard_phase : "guarded phase",
            op,
            size);
    exit(1);
}

void *xcalloc(size_t n, size_t size) {
    ds4_alloc_guard_check("calloc", n * size);
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

void *xmalloc(size_t size) {
    ds4_alloc_guard_check("malloc", size);
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

char *ds4_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

void *xrealloc(void *ptr, size_t size) {
    ds4_alloc_guard_check("realloc", size);
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

void *xmalloc_zeroed(size_t n, size_t size) {
    if (size != 0 && n > SIZE_MAX / size) ds4_die("allocation size overflow");
    const size_t total = n * size;
    void *p = xmalloc(total ? total : 1);
    /*
     * This is intentionally not calloc(). Large untouched calloc ranges may be
     * represented by the VM through shared zero-page bookkeeping. The CPU decode
     * KV cache grows one token at a time, so using calloc here can move thousands
     * of first-touch faults into generation. On Darwin we have observed this end
     * in a kernel cpt_mapcnt_inc overflow panic instead of a user-space error.
     *
     * Explicitly writing the zeroes while the cache is allocated keeps those VM
     * faults out of the token loop and gives the cache private resident pages.
     */
    memset(p, 0, total);
    return p;
}

/* ---- Stage 0: phys_footprint watchdog (see task/02) ----
 * Watchdog samples the Mach phys_footprint every 200ms (NOT ps/rss, which is
 * blind to Metal no-copy mmap residency: 9.3GiB resident reads as ~38MiB)
 * and _exit()s before crossing 90% of the --mem-budget-mb budget so the
 * machine never page-thrashes. A constructor self-starts it so the whole
 * process -- including the big model mmap + residency wiring during load -- is
 * covered, without touching any hot path. The always-on system-pressure guard
 * keeps a lightweight poll thread running for every process (one sysctl / 200ms)
 * so the machine can never be driven into a kernel-watchdog panic; safety
 * guards take no opt-out. --mem-budget-mb adds the phys_footprint ceiling. */
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
static uint64_t ds4_phys_footprint_bytes(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return (uint64_t)info.phys_footprint;
}
/* System-wide memory-pressure level (kern.memorystatus_vm_pressure_level):
 * 1=normal, 2=warn, 4=critical. Distinct from phys_footprint above, which is
 * BLIND to clean file-backed page-cache pages -- the expert-offload path's
 * MADV_WILLNEED prefetch fills those, and they don't count toward the process
 * footprint yet DO drive system pressure. Sustained critical pressure starves
 * watchdogd -> kernel watchdog panic + reboot (observed 2026-07-06: single-host
 * offload generation on the full mono panicked M4 while footprint read a healthy
 * 8.2 GiB). Returns 1 (normal) on any read failure so the guard fails safe. */
static int ds4_system_mem_pressure_level(void) {
    int level = 1; size_t len = sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, NULL, 0) != 0) return 1;
    return level;
}
#else
static uint64_t ds4_phys_footprint_bytes(void) { return 0; }
static int ds4_system_mem_pressure_level(void) { return 1; }
#endif
static pthread_t         g_mem_watch_thread;
static volatile int      g_mem_watch_run = 0;
static int               g_mem_watch_started = 0;
static uint64_t          g_mem_budget_bytes = 0;

static void *ds4_mem_watchdog_main(void *arg) {
    (void)arg;
    const unsigned interval_ms = 200;
    unsigned crit_ms = 0;   /* consecutive system-CRITICAL accumulator for the pressure guard */
    while (g_mem_watch_run) {
        uint64_t fp = ds4_phys_footprint_bytes();
        if (g_mem_budget_bytes != 0 && fp > (uint64_t)((double)g_mem_budget_bytes * DS4_WATCHDOG_TRIP_FRAC)) {
            fprintf(stderr,
                    "\n[ds4-watchdog] phys_footprint %.2f GiB crossed %.0f%% of the "
                    "%.2f GiB budget -- aborting before page thrash.\n",
                    (double)fp / DS4_GIB, DS4_WATCHDOG_TRIP_FRAC * 100.0,
                    (double)g_mem_budget_bytes / DS4_GIB);
            fflush(stderr);
            _exit(137);
        }
        /* Always-on system-pressure guard: abort after ~4s sustained CRITICAL,
         * well ahead of the ~94s watchdogd-starvation kernel panic. Catches the
         * offload page-cache blowup that phys_footprint (above) cannot see.
         * Safety guards take no opt-out. */
        if (ds4_system_mem_pressure_level() >= 4 /* critical */) {
            crit_ms += interval_ms;
            if (crit_ms >= 4000) {
                fprintf(stderr,
                        "\n[ds4-watchdog] system memory pressure CRITICAL for %ums "
                        "(phys_footprint %.2f GiB) -- aborting to keep the OS watchdog "
                        "from panicking the machine. Split across hosts or lower --ctx.\n",
                        crit_ms, (double)fp / DS4_GIB);
                fflush(stderr);
                _exit(137);
            }
        } else {
            crit_ms = 0;
        }
        usleep(interval_ms * 1000);
    }
    return NULL;
}

/* --mem-budget-mb: 进程内存红线(字节)。看门狗线程由构造器先起(预算可为 0=只有系统
 * 压力护栏), CLI 解析后调本函数补上预算, 下一个 200ms 采样周期即生效; L1 闸在
 * engine open 时读同一全局。 */
void ds4_set_mem_budget_mb(int mb) {
    if (mb > 0) g_mem_budget_bytes = (uint64_t)mb * 1024ull * 1024ull;
}

static void ds4_mem_watchdog_start(void) {
    if (g_mem_watch_started) return;
    /* The system-pressure guard is always on, so the watchdog runs for every
     * process -- this is the default-safe behavior that prevents a kernel panic. */
    g_mem_watch_run = 1;
    if (pthread_create(&g_mem_watch_thread, NULL, ds4_mem_watchdog_main, NULL) != 0) {
        g_mem_watch_run = 0;
        return;
    }
    g_mem_watch_started = 1;
}

/* Exposed to the GPU backend (ds4_metal.m) so memory-hungry caches can size
 * themselves dynamically against the live Mach phys_footprint and the
 * configured budget instead of a fixed MB cap (2026-06-15 policy: use the
 * headroom up to the 12G total, not a fixed slice). */
uint64_t ds4_runtime_phys_footprint_bytes(void) {
    return ds4_phys_footprint_bytes();
}

uint64_t ds4_runtime_mem_budget_bytes(void) {
    return g_mem_budget_bytes;
}

__attribute__((constructor)) static void ds4_stage0_autostart(void) {
    ds4_mem_watchdog_start();
}

/* L1 pre-flight static budget gate. Called once the resident model byte total is
 * known (model map span sums), before any GPU buffer is bound. Aborts a planned
 * OOM at load time -- well ahead of the runtime watchdog -- when the closed-form
 * resident estimate crosses 85% of --mem-budget-mb. kv_and_scratch_bytes is the
 * caller's estimate of KV + prefill scratch + fixed overhead (0 if unknown; the
 * runtime watchdog still backstops). No budget set => no-op (zero behavior change). */
DS4_MAYBE_UNUSED void ds4_l1_budget_gate(uint64_t resident_model_bytes,
                                                uint64_t kv_and_scratch_bytes) {
    if (g_mem_budget_bytes == 0) return;
    const uint64_t planned = resident_model_bytes + kv_and_scratch_bytes;
    const uint64_t limit = (uint64_t)((double)g_mem_budget_bytes * DS4_L1_GATE_FRAC);
    if (planned > limit) {
        fprintf(stderr,
                "\n[ds4-l1-gate] planned resident %.2f GiB (model %.2f + kv/scratch %.2f) "
                "exceeds %.0f%% of the %.2f GiB budget -- refusing to load.\n",
                (double)planned / DS4_GIB,
                DS4_L1_GATE_FRAC * 100.0,
                (double)resident_model_bytes / DS4_GIB,
                (double)kv_and_scratch_bytes / DS4_GIB,
                (double)g_mem_budget_bytes / DS4_GIB);
        fflush(stderr);
        _exit(137);
    }
    fprintf(stderr,
            "ds4: L1 budget gate: planned resident %.2f GiB within %.2f GiB budget\n",
            (double)planned / DS4_GIB, (double)g_mem_budget_bytes / DS4_GIB);
}

void sleep_sec(double sec) {
    if (sec <= 0.0 || !isfinite(sec)) return;
    struct timespec req;
    req.tv_sec = (time_t)sec;
    req.tv_nsec = (long)((sec - (double)req.tv_sec) * 1000000000.0);
    if (req.tv_nsec < 0) req.tv_nsec = 0;
    if (req.tv_nsec >= 1000000000L) {
        req.tv_sec++;
        req.tv_nsec -= 1000000000L;
    }
    /* Do not resume after EINTR: Ctrl+C should cut through throttling sleeps. */
    (void)nanosleep(&req, &req);
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case DS4_LOG_PREFILL:
    case DS4_LOG_TIMING:
        return "\x1b[36m";
    case DS4_LOG_GENERATION:
    case DS4_LOG_OK:
        return "\x1b[32m";
    case DS4_LOG_KVCACHE:
        return "\x1b[33m";
    case DS4_LOG_TOOL:
        return "\x1b[90m";
    case DS4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case DS4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != DS4_LOG_DEFAULT && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

