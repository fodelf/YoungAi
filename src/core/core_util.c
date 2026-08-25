/* core_util.c — die/分配守卫/xmalloc族/内存看门狗/profile/日志 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

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

/* ---- Stage 0: phys_footprint watchdog + DS4_PROFILE timing (see task/02) ----
 * Watchdog samples the Mach phys_footprint every 200ms (NOT ps/rss, which is
 * blind to Metal no-copy mmap residency: 9.3GiB resident reads as ~38MiB),
 * tracks the peak, and _exit()s before crossing 90% of DS4_MEM_BUDGET_MB so the
 * machine never page-thrashes. A constructor self-starts it so the whole
 * process -- including the big model mmap + residency wiring during load -- is
 * covered, without touching any hot path. A default-on system-pressure guard
 * keeps a lightweight poll thread running for every process (one sysctl / 200ms)
 * so the machine can never be driven into a kernel-watchdog panic; opt out with
 * DS4_NO_MEM_PRESSURE_GUARD=1. DS4_MEM_BUDGET_MB adds the phys_footprint ceiling.
 * DS4_PROFILE accumulates load/prefill/decode wall time and, at exit, writes one
 * CSV-ish line to DS4_PROFILE_FILE (or stderr). */
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
static volatile uint64_t g_mem_peak_footprint = 0;

ds4_profile_state g_prof;
static DS4_MAYBE_UNUSED double   g_prof_load_begin_sec;
static DS4_MAYBE_UNUSED uint64_t g_prof_load_footprint_begin;

static void ds4_profile_init(void) {
    if (g_prof.inited) return;
    g_prof.inited = 1;
    g_prof.enabled = getenv("DS4_PROFILE") != NULL;
    g_prof.csv_path = getenv("DS4_PROFILE_FILE");
}

static void ds4_profile_flush(void) {
    if (!g_prof.enabled && !g_mem_watch_started) return;
    uint64_t peak = g_mem_peak_footprint;
    uint64_t now = ds4_phys_footprint_bytes();
    if (now > peak) peak = now;
    double ptps = g_prof.prefill_sec > 0.0 ? (double)g_prof.prefill_tokens / g_prof.prefill_sec : 0.0;
    double dtps = g_prof.decode_sec  > 0.0 ? (double)g_prof.decode_tokens  / g_prof.decode_sec  : 0.0;
    FILE *f = stderr;
    int close_f = 0;
    if (g_prof.csv_path && g_prof.csv_path[0]) {
        FILE *cf = fopen(g_prof.csv_path, "a");
        if (cf) { f = cf; close_f = 1; }
    }
    fprintf(f,
            "ds4_profile load_sec=%.3f load_footprint_gib=%.3f "
            "prefill_tokens=%" PRIu64 " prefill_chunks=%u prefill_sec=%.3f prefill_tps=%.2f "
            "decode_tokens=%" PRIu64 " decode_sec=%.3f decode_tps=%.2f "
            "peak_footprint_gib=%.3f budget_gib=%.3f\n",
            g_prof.load_sec, (double)g_prof.load_footprint_delta / DS4_GIB,
            g_prof.prefill_tokens, g_prof.prefill_chunks, g_prof.prefill_sec, ptps,
            g_prof.decode_tokens, g_prof.decode_sec, dtps,
            (double)peak / DS4_GIB, (double)g_mem_budget_bytes / DS4_GIB);
    if (close_f) fclose(f);
}

DS4_MAYBE_UNUSED void ds4_profile_load_begin(void) {
    ds4_profile_init();
    if (!g_prof.enabled) return;
    g_prof_load_begin_sec = now_sec();
    g_prof_load_footprint_begin = ds4_phys_footprint_bytes();
}

DS4_MAYBE_UNUSED void ds4_profile_load_end(void) {
    if (!g_prof.enabled) return;
    g_prof.load_sec += now_sec() - g_prof_load_begin_sec;
    uint64_t fp = ds4_phys_footprint_bytes();
    if (fp > g_prof_load_footprint_begin) g_prof.load_footprint_delta += fp - g_prof_load_footprint_begin;
}

DS4_MAYBE_UNUSED void ds4_profile_add_prefill(uint64_t tokens, double sec) {
    if (!g_prof.enabled) return;
    g_prof.prefill_tokens += tokens;
    g_prof.prefill_sec += sec;
    g_prof.prefill_chunks++;
}

DS4_MAYBE_UNUSED void ds4_profile_add_decode(uint64_t tokens, double sec) {
    if (!g_prof.enabled) return;
    g_prof.decode_tokens += tokens;
    g_prof.decode_sec += sec;
}

static void *ds4_mem_watchdog_main(void *arg) {
    (void)arg;
    const int fp_log = getenv("DS4_FOOTPRINT_LOG") != NULL;
    const int pressure_guard = getenv("DS4_NO_MEM_PRESSURE_GUARD") == NULL;
    const unsigned interval_ms = fp_log ? 50 : 200;
    uint64_t fp_min = ~0ull, fp_max = 0;
    int fp_n = 0;
    unsigned crit_ms = 0;   /* consecutive system-CRITICAL accumulator for the pressure guard */
    while (g_mem_watch_run) {
        uint64_t fp = ds4_phys_footprint_bytes();
        if (fp > g_mem_peak_footprint) g_mem_peak_footprint = fp;
        if (fp_log) {
            if (fp < fp_min) fp_min = fp;
            if (fp > fp_max) fp_max = fp;
            if (++fp_n % 4 == 0) {
#ifdef DS4_NO_GPU
                fprintf(stderr, "[fp] phys=%.3f swing=%.0fMiB\n",
                        (double)fp / DS4_GIB,
                        (double)(fp_max - fp_min) / (1024.0 * 1024.0));
#else
                uint64_t ga = ds4_gpu_current_allocated_bytes();
                uint64_t gm = ds4_gpu_recommended_max_working_set_bytes();
                fprintf(stderr,
                        "[fp] phys=%.3f swing=%.0fMiB | GPU alloc=%.3f / max=%.3f GiB %s\n",
                        (double)fp / DS4_GIB,
                        (double)(fp_max - fp_min) / (1024.0 * 1024.0),
                        (double)ga / DS4_GIB, (double)gm / DS4_GIB,
                        (gm && ga > gm) ? "<<OVER-MAX (paging)" : "ok");
#endif
                fflush(stderr);
            }
        }
        if (g_mem_budget_bytes != 0 && fp > (uint64_t)((double)g_mem_budget_bytes * 0.9)) {
            fprintf(stderr,
                    "\n[ds4-watchdog] phys_footprint %.2f GiB crossed 90%% of the "
                    "%.2f GiB budget -- aborting before page thrash.\n",
                    (double)fp / DS4_GIB, (double)g_mem_budget_bytes / DS4_GIB);
            fflush(stderr);
            _exit(137);
        }
        /* Default-on system-pressure guard: abort after ~4s sustained CRITICAL,
         * well ahead of the ~94s watchdogd-starvation kernel panic. Catches the
         * offload page-cache blowup that phys_footprint (above) cannot see. Opt
         * out with DS4_NO_MEM_PRESSURE_GUARD=1. */
        if (pressure_guard && ds4_system_mem_pressure_level() >= 4 /* critical */) {
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

static void ds4_mem_watchdog_start(void) {
    if (g_mem_watch_started) return;
    ds4_profile_init();
    const char *bud = getenv("DS4_MEM_BUDGET_MB");
    if (bud && bud[0]) {
        long mb = strtol(bud, NULL, 10);
        if (mb > 0) g_mem_budget_bytes = (uint64_t)mb * 1024ull * 1024ull;
    }
    const int pressure_guard = getenv("DS4_NO_MEM_PRESSURE_GUARD") == NULL;
    /* Start the thread whenever anything needs it. The system-pressure guard is
     * on by default, so the watchdog now runs for every process unless explicitly
     * disabled -- this is the default-safe behavior that prevents a kernel panic. */
    if (!g_prof.enabled && g_mem_budget_bytes == 0 && !pressure_guard) return;
    g_mem_watch_run = 1;
    if (pthread_create(&g_mem_watch_thread, NULL, ds4_mem_watchdog_main, NULL) != 0) {
        g_mem_watch_run = 0;
        return;
    }
    g_mem_watch_started = 1;
    /* Only the profiler emits an exit line; the pressure-guard-only case stays
     * silent so normal runs see no new stderr output. */
    if (g_prof.enabled || g_mem_budget_bytes != 0) atexit(ds4_profile_flush);
}

/* Exposed to the GPU backend (ds4_metal.m) so memory-hungry caches (e.g. the
 * routed-expert source cache) can size themselves dynamically against the live
 * Mach phys_footprint and the configured budget instead of a fixed MB cap
 * (2026-06-15 policy: use the headroom up to the 12G total, not a fixed slice).
 * Budget falls back to the env so it works even before the watchdog thread
 * has populated g_mem_budget_bytes. */
uint64_t ds4_runtime_phys_footprint_bytes(void) {
    return ds4_phys_footprint_bytes();
}

uint64_t ds4_runtime_mem_budget_bytes(void) {
    if (g_mem_budget_bytes != 0) return g_mem_budget_bytes;
    const char *bud = getenv("DS4_MEM_BUDGET_MB");
    if (bud && bud[0]) {
        long mb = strtol(bud, NULL, 10);
        if (mb > 0) return (uint64_t)mb * 1024ull * 1024ull;
    }
    return 0;
}

__attribute__((constructor)) static void ds4_stage0_autostart(void) {
    ds4_mem_watchdog_start();
}

/* L1 pre-flight static budget gate. Called once the resident model byte total is
 * known (model map span sums), before any GPU buffer is bound. Aborts a planned
 * OOM at load time -- well ahead of the runtime watchdog -- when the closed-form
 * resident estimate crosses 85% of DS4_MEM_BUDGET_MB. kv_and_scratch_bytes is the
 * caller's estimate of KV + prefill scratch + fixed overhead (0 if unknown; the
 * runtime watchdog still backstops). No budget set => no-op (zero behavior change). */
DS4_MAYBE_UNUSED void ds4_l1_budget_gate(uint64_t resident_model_bytes,
                                                uint64_t kv_and_scratch_bytes) {
    ds4_profile_init();
    if (g_mem_budget_bytes == 0) {
        const char *bud = getenv("DS4_MEM_BUDGET_MB");
        if (bud && bud[0]) {
            long mb = strtol(bud, NULL, 10);
            if (mb > 0) g_mem_budget_bytes = (uint64_t)mb * 1024ull * 1024ull;
        }
    }
    if (g_mem_budget_bytes == 0) return;
    const uint64_t planned = resident_model_bytes + kv_and_scratch_bytes;
    const uint64_t limit = (uint64_t)((double)g_mem_budget_bytes * 0.85);
    if (planned > limit) {
        fprintf(stderr,
                "\n[ds4-l1-gate] planned resident %.2f GiB (model %.2f + kv/scratch %.2f) "
                "exceeds 85%% of the %.2f GiB budget -- refusing to load.\n",
                (double)planned / DS4_GIB,
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

