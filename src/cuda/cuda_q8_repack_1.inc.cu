/* cuda_q8_repack_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * q8 repack 缓存(decode gemv 专用)。
 */
/* ==== q8 repack 缓存(decode gemv 专用) ====
 * 原 q8_0 行布局是 34B 交错(2B scale + 32B qs), qs 永远非 4B 对齐 —— dp4a 内核
 * 每块 8 次非对齐 4B 读, LSU 事务 4x 放大(ncu: q8 gemv 仅吃出 16-32% 带宽)。
 * 启动/预填充期一次 repack 成两平面: [f16 scale × blocks×rows][int8 qs 16B 对齐],
 * decode 内核换 int4(128bit) 对齐读。数值 bit 级不变(同 scale 同 qs)。 */
typedef struct { const void *host_base; uint64_t offset, bytes; __half *scales; int8_t *qs; uint64_t rows, blocks; } cuda_q8r_entry;
static std::vector<cuda_q8r_entry> g_q8r_entries;
static std::unordered_map<uint64_t, size_t> g_q8r_by_offset;
static std::unordered_set<uint64_t> g_q8r_denied;

__global__ static void q8_repack_kernel(__half *scales, int8_t *qs,
                                        const unsigned char *w, uint64_t rows, uint64_t blocks) {
    const uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const unsigned char *wr = w + r * blocks * 34u;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        __half sh; memcpy(&sh, wr + b * 34u, 2);
        scales[r * blocks + b] = sh;
        const unsigned char *src = wr + b * 34u + 2u;
        int8_t *dst = qs + (r * blocks + b) * 32u;
        for (int i = 0; i < 32; i++) dst[i] = (int8_t)src[i];
    }
}

static const cuda_q8r_entry *cuda_q8r_get(const void *model_map, uint64_t offset,
                                          uint64_t rows, uint64_t blocks) {
    auto it = g_q8r_by_offset.find(offset);
    if (it != g_q8r_by_offset.end()) {
        const cuda_q8r_entry &e = g_q8r_entries[it->second];
        if (e.host_base == model_map && e.rows == rows && e.blocks == blocks) return &g_q8r_entries[it->second];
        return NULL;
    }
    if (g_q8r_denied.count(offset)) return NULL;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(0, &cs);
    if (cs != cudaStreamCaptureStatusNone) return NULL;   /* capture 期不建, 回落原路 */
    const uint64_t weight_bytes = rows * blocks * 34u;
    const char *w = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8r_src");
    if (!w) { g_q8r_denied.insert(offset); return NULL; }
    __half *sc = NULL; int8_t *qs = NULL;
    if (cudaMalloc(&sc, (size_t)rows * blocks * sizeof(__half)) != cudaSuccess) {
        (void)cudaGetLastError(); g_q8r_denied.insert(offset); return NULL;
    }
    if (cudaMalloc(&qs, (size_t)rows * blocks * 32u) != cudaSuccess) {
        (void)cudaGetLastError(); (void)cudaFree(sc); g_q8r_denied.insert(offset); return NULL;
    }
    q8_repack_kernel<<<(unsigned)rows, 256>>>(sc, qs, (const unsigned char *)w, rows, blocks);
    if (cudaDeviceSynchronize() != cudaSuccess) {   /* 建完才发布, 同 fp16 cache */
        (void)cudaGetLastError(); (void)cudaFree(sc); (void)cudaFree(qs);
        g_q8r_denied.insert(offset); return NULL;
    }
    g_q8r_entries.push_back({model_map, offset, weight_bytes, sc, qs, rows, blocks});
    g_q8r_by_offset[offset] = g_q8r_entries.size() - 1u;
    return &g_q8r_entries.back();
}

int ds4_gpu_q8r_preload(const void *model_map, uint64_t model_size,
                                   uint64_t offset, uint64_t in_dim, uint64_t out_dim) {
    if (!model_map || in_dim == 0 || out_dim == 0 || (in_dim & 31u)) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (offset > model_size || out_dim * blocks * 34u > model_size - offset) return 0;
    return cuda_q8r_get(model_map, offset, out_dim, blocks) != NULL;
}

static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f32_by_offset.find(offset);
    if (exact != g_q8_f32_by_offset.end()) {
        const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    g_q8_f32_bytes += out_bytes;
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) { return 1; }

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\rds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (0 ||
        0 ||
        0 ||
        0) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    /* ★默认同步★(2026-08-17 GB10 实锤): 本机 UsesHostPageTables=1(ATS), 异步页迁移
     * 与在飞 kernel 并发时, 迁移中的页被读出垃圾 —— 表现为批量 prefill 输出塌 BOS、
     * NaN 首层随运行漂移(迁移进度决定谁踩雷)、compute-sanitizer(强制串行)下反而正确、
     * DS4_CUDA_NO_MODEL_PREFETCH=1 立即痊愈。迁移是启动期一次性成本(秒级), 等它完成
     * 换全程正确性。DS4_CUDA_MODEL_PREFETCH_ASYNC=1 可回旧行为(仅限已证安全的平台)。 */
    if (1) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

/* 整模型拷贝的 staging 块 64 MB(原 DS4_CUDA_MODEL_COPY_CHUNK_MB 旋钮, 定死) */
static uint64_t cuda_model_copy_chunk_bytes(void) { return 64ull * 1048576ull; }

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (0 || !model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || 0 || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

/* 设备权重缓存的平台默认预算。2026-10-07 前由编译宏 DS4_CUDA_SPARK_HBM_CACHE 二选一(make cuda-spark 才走统一内存公式,
 * cuda-generic 编出来的二进制放到 GB10 上只会缓 24 GiB), 现在按设备属性运行时判(cuda_unified_memory_host)。 */
static uint64_t cuda_model_cache_limit_default_bytes(void) {
    if (cuda_unified_memory_host()) {
        /* 统一内存(GB10/Grace): 实测 registered-host 页表读被钉在 ~185 GB/s, 设备拷贝 255
         * (2026-08-17)。预算=整机内存-余量, 整模型尽量收编; 拷贝后源 mmap 页
         * madvise(DONTNEED), 净占用不翻倍。
         * ★2026-09-15 single.md S1: 余量 24 → 8 GiB★
         * 24 GiB 是 V4 时代按"给 page cache 留活路"定的, 而拷进设备的段立刻 MADV_DONTNEED,
         * page cache 根本不承重。代价是实打实的: 117.9 GB 的模型有约 12 GiB(最后 4~5 层的专家 blob)
         * 装不进预算, 只能走 cudaHostRegister 的主机映射 —— 那些页在 130 GB 机器上被 kswapd 一直回收,
         * 每步都有几层缺页从 NVMe 重读。**实测这几层的专家核 235 µs → 1100~4500 µs**(逐核时间线,
         * gateup 第 10 步 L35/L37/L38/L39), 一步 66 ms 里 5~25 ms 是这个长尾, 首步更是 95 ms。
         * 8 GiB 盖得住实际的非模型占用: KV@32k 2.4 + 中间张量暂存 ≈2 + CUDA 上下文/cuBLASLt ≈1.5, 留 2 兜底。
         * ★别 OOM 是最高约束★: 起跑后由 core_model_map.c 的对账打印 + available 检查兜底(见那里的注释)。 */
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        const uint64_t total = (uint64_t)sysconf(_SC_PHYS_PAGES) * page;
        const uint64_t headroom = 8ull * 1073741824ull;
        if (total > headroom * 2) return total - headroom;
    }
    /* 独显: 显存是独立池。V4 时代定的 24 GiB(够装 attn 投影 + 嵌入 + 输出头 + 共享 FFN; 路由专家走映射指针)
     * 对 80 GB 卡合适, 对 24 GB 卡就是把显存吃光 —— 再按本卡显存封顶: 留 max(4 GiB, 20%) 给 KV/scratch/cuBLAS。
     * ★这条分支 2026-10-07 没有独显机器验证★(cuda-generic 以前根本不走缓存), 只保证不比"固定 24 GiB"更激进。 */
    uint64_t cap = 24ull * 1073741824ull;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { (void)cudaGetLastError(); total_b = 0; }
    if (total_b) {
        uint64_t reserve = (uint64_t)total_b / 5u;
        if (reserve < 4ull * 1073741824ull) reserve = 4ull * 1073741824ull;
        const uint64_t avail = (uint64_t)total_b > reserve ? (uint64_t)total_b - reserve : 0;
        if (avail < cap) cap = avail;
    }
    return cap;
}
/* 显式封顶(ds4_gpu_set_model_cache_limit_mb)只能往下压, 不许越过平台默认 —— 默认那条线是"别 OOM"的最高约束。 */
static uint64_t cuda_model_cache_limit_bytes(void) {
    const uint64_t def = cuda_model_cache_limit_default_bytes();
    return (g_model_cache_limit_override && g_model_cache_limit_override < def) ? g_model_cache_limit_override : def;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    /* arena 块 1792 MB(原 DS4_CUDA_WEIGHT_ARENA_CHUNK_MB 旋钮, 定死): 单块够摊薄 cudaMalloc 次数, 尾块浪费又不至于太大 */
    uint64_t bytes = 1792ull * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

