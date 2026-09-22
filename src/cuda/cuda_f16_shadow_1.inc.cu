/* cuda_f16_shadow_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * host 探针辅助 + 全q2 f16 影子(f16 专线家族权重装载时一次性 dequant→f16 device buffer)。
 */
/* host 侧 f16→f32(探针用) */
static float dev_host_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) { if (!m) f = s; else { e = 112; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | (e << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static int g_q2k_probe_out;
/* host 侧 q2_K 单块 dequant(canonical GGML 布局), 探针用 */
static void host_deq_q2k_block(const uint8_t *blk, float *out256) {
    const uint8_t *sc = blk, *qs = blk + 16;
    uint16_t hd, hm; memcpy(&hd, blk + 80, 2); memcpy(&hm, blk + 82, 2);
    const float d = dev_host_f16(hd), dm = dev_host_f16(hm);
    for (int j = 0; j < 16; j++) {
        const float dj = d * (sc[j] & 0xF), mj = dm * (sc[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            const int idx = j * 16 + ii;
            const int qpos = (idx / 128) * 32 + (idx % 32);
            const int q = (qs[qpos] >> ((idx % 128) / 32 * 2)) & 3;
            out256[idx] = dj * q - mj;
        }
    }
}


/* ==== 全q2 f16 影子(2026-08-19 用户令"零例外全q2, 引擎不适配就改"): f16 专线家族
 * (embd/compressor/indexer/hc_fn/router) 的 q2_K 权重装载时一次性 dequant→f16 device
 * buffer, cuda_model_range_ptr 按 offset 命中重定向 — q8 repack 同类机制:
 * 文件全 q2, 运行时每 token 零额外成本。 ==== */
#define Q2K_SHADOW_MAX 1024
/* base 必须入表(2026-08-21): 只按 offset 命中会让副模型(DS4_DRAFT_GGUF)的张量撞上主模型
 * 的影子 —— 读到完全不相干的权重。drafter 同输入两跑草稿不同的真因之一。 */
static struct { const void *base; uint64_t off; const char *ptr; } g_q2k_shadow[Q2K_SHADOW_MAX];
static int g_q2k_shadow_n = 0;

static int cuda_ok(cudaError_t err, const char *what);

int ds4_gpu_register_q2k_f16_shadow(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t rows, uint64_t cols) {
    /* 装载期先于 ds4_gpu_init 到达也安全: cudaMalloc 隐式建 device-0 primary context。 */
    if (g_q2k_shadow_n >= Q2K_SHADOW_MAX || cols == 0 || cols % 256u != 0) return 0;
    const uint64_t blocks = cols / 256u;
    const uint64_t src_bytes = rows * blocks * sizeof(cuda_block_q2_K);
    if (offset > model_size || src_bytes > model_size - offset) return 0;
    const uint8_t *src = (const uint8_t *)model_map + offset;
    __half *hbuf = (__half *)malloc(rows * cols * sizeof(__half));
    if (!hbuf) return 0;
    float tmp[256];
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t b = 0; b < blocks; b++) {
            host_deq_q2k_block(src + (r * blocks + b) * sizeof(cuda_block_q2_K), tmp);
            for (int i = 0; i < 256; i++)
                hbuf[r * cols + b * 256u + (uint64_t)i] = __float2half(tmp[i]);
        }
    if (((const char *)0) /* DS4_Q2K_SHADOW_CHECK: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
        uint64_t bad = 0;
        for (uint64_t i = 0; i < rows * cols; i++) {
            const float v = __half2float(hbuf[i]);
            if (isnan(v) || isinf(v)) bad++;
        }
        fprintf(stderr, "ds4: [q2k-shadow] off=%llu rows=%llu cols=%llu bad=%llu\n",
                (unsigned long long)offset, (unsigned long long)rows,
                (unsigned long long)cols, (unsigned long long)bad);
    }
    void *dbuf = NULL;
    if (!cuda_ok(cudaMalloc(&dbuf, rows * cols * sizeof(__half)), "q2k shadow")) { free(hbuf); return 0; }
    if (!cuda_ok(cudaMemcpy(dbuf, hbuf, rows * cols * sizeof(__half), cudaMemcpyHostToDevice),
                 "q2k shadow up")) { free(hbuf); return 0; }
    free(hbuf);
    g_q2k_shadow[g_q2k_shadow_n].base = model_map;
    g_q2k_shadow[g_q2k_shadow_n].off = offset;
    g_q2k_shadow[g_q2k_shadow_n].ptr = (const char *)dbuf;
    g_q2k_shadow_n++;
    return 1;
}

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static uint64_t g_model_cache_limit_override;   /* ds4_gpu_set_model_cache_limit_mb: 0 = 平台默认 */
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
static int g_quality_mode;

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;

static int cuda_ok(cudaError_t err, const char *what);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void *g_q2k_part = NULL;
static uint64_t g_q2k_part_bytes = 0;

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

#ifndef DS4_Q2K_BLOCKS_PER_SM
#define DS4_Q2K_BLOCKS_PER_SM 4   /* dense q2 每 SM 常驻块数(占用率闸, 2026-08-21 A/B) */
#endif
static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

/* Allocate a device-resident copy of [offset, offset+bytes) from model_map and
 * push it into g_model_ranges so future cuda_model_range_ptr lookups hit it.
 * Returns the device pointer on success, NULL on cudaMalloc/cudaMemcpy failure.
 * Caller is responsible for any policy gating (budget cap, env opt-out, etc.) */
static const char *cuda_model_range_populate_device_copy(const void *model_map,
                                                          uint64_t offset,
                                                          uint64_t bytes,
                                                          const char *what) {
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

/* 只查设备副本, 不注册不回落(2026-09-20 缓存封顶模式用): 命中返回设备指针, 否则 NULL 让调用方自己选路。
 * 与下面 cuda_model_range_ptr 的前两段命中逻辑同, 只是把 host_registered(UVA 懒注册段)排除在外。 */
static const char *cuda_model_range_cached_ptr(const void *model_map, uint64_t offset, uint64_t bytes) {
    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && !r.host_registered && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges)
        if (r.host_base == model_map && !r.host_registered && offset >= r.offset && end >= offset && end <= r.offset + r.bytes)
            return r.device_ptr + (offset - r.offset);
    return NULL;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    /* 全q2 f16 影子命中(小表线性扫, 每 launch 一次, 开销可忽略) */
    for (int i = 0; i < g_q2k_shadow_n; i++)
        if (g_q2k_shadow[i].off == offset && g_q2k_shadow[i].base == model_map)
            return g_q2k_shadow[i].ptr;
    if (bytes == 0) return cuda_model_ptr(model_map, offset);

    /* Device-resident HBM cache hits win over UVA-mapped registered pointers:
     * direct HBM reads are ~10% faster than mapped reads through host page
     * tables (measured on plain decode at GB10).  Cache lookup runs first; the
     * registered-mapped shortcut below is the cold fallback when an allocation
     * hasn't been pre-populated. */
    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    /* 这三条"整模型已就绪"捷径只对主 map 成立(2026-08-21): 副 map(DS4_DRAFT_GGUF)命中
     * 时会拿到未注册裸指针 —— 实测 drafter logits 直接 NaN、acc 塌到 1.00 而速度看似正常。
     * 非主 base 一律落到下面的按 range 懒注册路, 保证任何情况下指针可用。 */
    const int is_primary_map = (model_map == g_model_host_base);
    if (is_primary_map && (g_model_device_owned || g_model_registered))
        return cuda_model_ptr(model_map, offset);
    if (is_primary_map && g_model_hmm_direct &&
        1) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = ((const char *)0) /* DS4_CUDA_DIRECT_MODEL: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (is_primary_map && direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

    if (1) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                    fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                            what ? what : "weights",
                            (double)bytes / 1048576.0);
                }
                return dev_ptr;
            }
            fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    /* 非主 map(drafter/zchain/corr 侧车)注册不成时回落裸指针 —— 拷贝路对这些小而
     * 不规则的 range 会 invalid argument(2026-08-21 实测), 且裸指针对侧车本就够用。 */
    if (!is_primary_map) return cuda_model_ptr(model_map, offset);
    return cuda_model_range_populate_device_copy(model_map, offset, bytes, what);
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

/* No byte cap on the expanded cache itself: the reserve below is the guard. */
static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed && ((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */ == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    /* Unified-memory boards (GB10 etc.): cudaMemGetInfo free is distorted by
     * reclaimable page cache holding the mmapped model, so a free-based veto
     * flips kernel choice mid-run and mutates the token-graph topology.  Trust
     * cudaMalloc as the real verdict; its failure path already disables the
     * cache permanently. */
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) return 1;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

