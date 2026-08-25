/* cuda_q8_repack_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * q8 repack 缓存(decode gemv 专用)。
 */
static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

/* A raw host pointer is safe for kernels only after CUDA owns, registered, or
 * HMM-prefetched the mapping.  Otherwise let the caller try per-range mapping
 * or a device copy instead of surfacing an async illegal access later. */
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset) {
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct ||
        0) {
        return cuda_model_ptr(model_map, offset);
    }
    return NULL;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (0) return NULL;
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (0 ||
        0 ||
        0 ||
        0) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "ds4: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (((const char *)0) /* DS4_CUDA_MODEL_COPY_VERBOSE: 诊断开关已删(2026-08-22) */ != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

