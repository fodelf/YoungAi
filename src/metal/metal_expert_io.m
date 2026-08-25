/* metal_expert_io.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_expert_pread_fd_nocache(void) {
    static int cached_fd = -2;
    if (cached_fd == -2) {
        cached_fd = -1;
        if (g_model_fd >= 0) {
            char path[MAXPATHLEN];
            memset(path, 0, sizeof(path));
            if (fcntl(g_model_fd, F_GETPATH, path) == 0) {
                int nfd = open(path, O_RDONLY);
                if (nfd >= 0) {
                    if (fcntl(nfd, F_NOCACHE, 1) != -1) {
                        cached_fd = nfd;
                        fprintf(stderr,
                                "ds4: batch expert gather cold reads use a separate "
                                "F_NOCACHE descriptor (DS4_METAL_EXPERT_BATCH_NOCACHE=0 disables)\n");
                    } else {
                        close(nfd);
                    }
                }
            }
        }
        if (cached_fd < 0) {
            fprintf(stderr,
                    "ds4: batch F_NOCACHE descriptor unavailable; "
                    "batch gathers fall back to the shared cached fd\n");
        }
    }
    return cached_fd;
}

int ds4_gpu_expert_pread_fd(void) {
    static int cached_fd = -2;
    if (cached_fd == -2) {
        cached_fd = g_model_fd;
        if (ds4_gpu_env_bool("DS4_METAL_EXPERT_PREAD_NOCACHE") > 0 && g_model_fd >= 0) {
            char path[MAXPATHLEN];
            memset(path, 0, sizeof(path));
            if (fcntl(g_model_fd, F_GETPATH, path) == 0) {
                int nfd = open(path, O_RDONLY);
                if (nfd >= 0) {
                    if (fcntl(nfd, F_NOCACHE, 1) != -1) {
                        cached_fd = nfd;
                        fprintf(stderr,
                                "ds4: expert pread cold reads use a separate F_NOCACHE descriptor\n");
                    } else {
                        close(nfd);
                    }
                }
            }
            if (cached_fd == g_model_fd) {
                fprintf(stderr,
                        "ds4: DS4_METAL_EXPERT_PREAD_NOCACHE=1 requested but F_NOCACHE descriptor "
                        "unavailable; using the shared model fd\n");
            }
        }
    }
    return cached_fd;
}

int ds4_gpu_expert_io_profile_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_IO_PROFILE") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4-io: expert IO profile on "
                    "(fault/memcpy/pread are thread-summed CPU ms; wall/drain are wall-clock ms)\n");
        }
    }
    return cached;
}

/* Thread-summed accumulators for the current gather call (reset per layer call,
 * written by gather workers via atomic adds, read after the join). */
uint64_t g_io_prof_fault_ns;

uint64_t g_io_prof_copy_ns;

uint64_t g_io_prof_pread_ns;

uint64_t g_io_prof_remote_ns;

uint64_t g_io_prof_cold_bytes;

uint64_t g_io_prof_hit_bytes;

uint64_t g_io_prof_remote_bytes;

uint64_t g_io_prof_pread_fallbacks;

/* cumulative, never reset */
volatile uint64_t g_io_prof_touch_sink;

void ds4_gpu_expert_io_prof_reset(void) {
    g_io_prof_fault_ns = 0;
    g_io_prof_copy_ns = 0;
    g_io_prof_pread_ns = 0;
    g_io_prof_remote_ns = 0;
    g_io_prof_cold_bytes = 0;
    g_io_prof_hit_bytes = 0;
    g_io_prof_remote_bytes = 0;
}

/* P2.1 prediction-accuracy counters (defined here so the ds4-io line can carry
 * them; maintained by the prefetch section below). */
uint64_t g_pf_pred_hits, g_pf_pred_total;

void ds4_gpu_expert_io_prof_report(const char *site,
                                          const char *mode,
                                          uint32_t layer,
                                          uint32_t n_active,
                                          uint32_t n_tokens,
                                          double wall_ms,
                                          double drain_ms) {
    const double mib = 1024.0 * 1024.0;
    const uint64_t moved = g_io_prof_cold_bytes + g_io_prof_hit_bytes + g_io_prof_remote_bytes;
    const double bw_gbps = wall_ms > 0.0 ? ((double)moved / (wall_ms * 1e-3)) / 1e9 : 0.0;
    fprintf(stderr,
            "ds4-io: site=%s mode=%s layer=%u n_active=%u n_tokens=%u "
            "cold_mib=%.1f hit_mib=%.1f rfetch_mib=%.1f wall_ms=%.2f fault_ms=%.2f memcpy_ms=%.2f "
            "pread_ms=%.2f rfetch_ms=%.2f drain_ms=%.2f bw_gbps=%.2f pread_fallbacks=%llu pf=%llu/%llu\n",
            site, mode, layer, n_active, n_tokens,
            (double)g_io_prof_cold_bytes / mib,
            (double)g_io_prof_hit_bytes / mib,
            (double)g_io_prof_remote_bytes / mib,
            wall_ms,
            (double)g_io_prof_fault_ns / 1e6,
            (double)g_io_prof_copy_ns / 1e6,
            (double)g_io_prof_pread_ns / 1e6,
            (double)g_io_prof_remote_ns / 1e6,
            drain_ms,
            bw_gbps,
            (unsigned long long)g_io_prof_pread_fallbacks,
            (unsigned long long)g_pf_pred_hits,
            (unsigned long long)g_pf_pred_total);
}

/* Touch one byte per page so the mmap fault cost can be timed separately from
 * the memcpy when the IO profile is on (mmap path only). */
uint64_t ds4_gpu_expert_touch_pages(const uint8_t *p, size_t len) {
    uint64_t sink = 0;
    const size_t page = 16384;
    for (size_t i = 0; i < len; i += page) sink += p[i];
    if (len) sink += p[len - 1];
    return sink;
}

#define DS4_METAL_PF_QUEUE 4u

ds4_metal_layer_router g_layer_router[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

int ds4_gpu_register_layer_router(
        const void *model_map,
        uint32_t layer,
        uint64_t gate_inp_offset,
        int gate_inp_is_f32,
        uint64_t probs_bias_offset,
        uint64_t gate_exps_offset,
        uint64_t up_exps_offset,
        uint64_t down_exps_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t n_embd,
        uint32_t n_expert,
        uint64_t hash_table_offset,
        uint32_t hash_k,
        uint32_t hash_rows) {
    if (!model_map || layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS ||
        n_embd == 0 || n_embd > DS4_METAL_PF_MAX_EMBD ||
        n_expert == 0 || n_expert > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }
    if (hash_table_offset != UINT64_MAX && (hash_k == 0 || hash_k > 16u || hash_rows == 0)) {
        return 0;
    }
    ds4_metal_layer_router *r = &g_layer_router[layer];
    r->model_map = model_map;
    r->gate_inp_off = gate_inp_offset;
    r->gate_inp_is_f32 = gate_inp_is_f32;
    r->hash_off = hash_table_offset;
    r->hash_k = hash_k;
    r->hash_rows = hash_rows;
    r->probs_bias_off = probs_bias_offset;
    r->gate_exps_off = gate_exps_offset;
    r->up_exps_off = up_exps_offset;
    r->down_exps_off = down_exps_offset;
    r->gate_expert_bytes = gate_expert_bytes;
    r->down_expert_bytes = down_expert_bytes;
    r->n_embd = n_embd;
    r->n_expert = n_expert;
    r->valid = 1;
    return 1;
}
