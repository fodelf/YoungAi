/* cuda_modelmap.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * 模型 map 注册/range/keep_lut/layer_router/cache_q8_f16。
 */
/* 副模型 map 注册(2026-08-21 draft 5x 慢根因): 主模型整文件 cudaHostRegister 后,
 * cuda_model_ptr 对"非主 base"退回裸 host 指针 —— 未注册路径在 GB10 上按缺页走,
 * drafter FFN 实测 12ms/层 vs verify 2.3ms/层。这里把副 map(DS4_DRAFT_GGUF)整体注册
 * 并挂进 g_model_ranges, 让 range 查找先命中 ⇒ 与主模型同速。 */
int ds4_gpu_register_aux_model_map(const void *map, uint64_t size) {
    if (!map || size == 0) return 0;
    for (const cuda_model_range &r : g_model_ranges)
        if (r.host_base == map && r.offset == 0 && r.bytes >= size) return 1;
    /* ★设备驻留拷贝(2026-08-21 定版): 早先用 cudaHostRegister+HostGetDevicePointer 整体
     * 映射副 gguf, 在 5.59GB 文件映射上**不可靠** —— drafter 同输入两跑 MoE 输出不同
     * (Fin/路由/权重逐字节相同), 注册前触实所有页也无效。改为 cudaMalloc + H2D 拷贝:
     * GB10 统一内存上 5.6GB 完全负担得起, 且拿到的是正规设备页表(既确定又快)。
     * DS4_AUX_HOSTREG=1 可退回旧映射法做对照; 拷贝失败则回落按 range 懒注册(正确但慢)。 */
    if (1) {
        void *dev = NULL;
        cudaError_t e2 = cudaMalloc(&dev, (size_t)size);
        if (e2 == cudaSuccess) {
            e2 = cudaMemcpy(dev, map, (size_t)size, cudaMemcpyHostToDevice);
            if (e2 == cudaSuccess) {
                g_model_ranges.push_back({map, 0, size, (char *)dev, NULL, NULL, 0, 0, 0});
                fprintf(stderr, "ds4: CUDA aux map device-resident copy %.2f GiB\n",
                        (double)size / 1073741824.0);
                return 1;
            }
            (void)cudaFree(dev);
        }
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA aux map device copy failed (%.2f GiB); 回落懒注册\n",
                (double)size / 1073741824.0);
        return 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaHostRegister((void *)map, (size_t)size, cudaHostRegisterMapped);
    if (err != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (cudaHostGetDevicePointer(&dev, (void *)map, 0) != cudaSuccess || !dev) {
        (void)cudaGetLastError(); (void)cudaHostUnregister((void *)map); return 0;
    }
    g_model_ranges.push_back({map, 0, size, (char *)dev, (void *)map, (char *)dev, size, 1, 0});
    fprintf(stderr, "ds4: CUDA registered aux map %.2f GiB (host-mapped, 旧法)\n",
            (double)size / 1073741824.0);
    return 1;
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
        g_model_device_owned = 0;
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
        g_model_registered = 0;
    }
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    const char *copy_env = ((const char *)0) /* DS4_CUDA_COPY_MODEL: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    /* GB10 / driver 580.142 reports cudaDevAttrHostRegisterReadOnlySupported = 0,
     * so requesting cudaHostRegisterReadOnly here fails with cudaErrorNotSupported
     * and the entire model-resident fast path falls back to per-deref H2D streaming.
     * Plain `Mapped` works on Spark. */
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       cudaHostRegisterMapped);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

/* Metal-only residency hint (see ds4_gpu.h); CUDA uses an HBM-cache model and
 * ignores it. No-op so the shared core links against either backend. */
void ds4_gpu_set_model_map_nonresident_hint(int on) { (void)on; }

/* Dynamic resident/offload route (ds4_gpu.h) is a Metal unified-memory concept;
 * CUDA manages residency via its HBM weight cache. Accept the host verdict as a
 * no-op and report no working-set ceiling so the AUTO path falls back to the
 * explicit DS4_MEM_BUDGET_MB (or resident) without a spurious offload. */
void ds4_gpu_set_expert_offload(int enabled) { (void)enabled; }
uint64_t ds4_gpu_recommended_max_working_set_bytes(void) { return 0; }

/* ---- 共享核心无条件调用、但只有 Metal 实现过的接口 ----
 * ds4.o / ds4_distributed.o 对两个后端只编译一份, 所以 CUDA 必须给出符号
 * (ds4_gpu_expert_remote_fetch_kick 的 CPU 版 no-op 见 ds4.c 顶部, 同一套路)。
 * 下面没有一个是占位兜底: 要么是真实现, 要么是"CUDA 确实没有这个能力"的语义
 * 正确回答, 且调用点都有对应的处理分支 —— 详见每个函数的注释。 */

/* 内存看门狗的 GPU 用量读数。CUDA 没有 Metal 那种 per-device allocated 计数,
 * 改用 driver 的 total-free: 它含其他进程和 context 开销, 对看门狗真正关心的
 * "这张卡现在还剩多少" 反而比纯自家分配量更准。 */
uint64_t ds4_gpu_current_allocated_bytes(void) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return (uint64_t)(total_b - free_b);
}

/* go1b 残差 MoE 会把 router_selected 原地重映射成紧凑 slot, Metal 因此要留一份
 * 重映射前的快照给 corr 侧车按原始 expert id 取 C[e]/beta[e]。CUDA 没实现 go1b
 * 残差(见 routed_moe 包装里的 (void)residual), 从不重映射, 所以没有快照可留 ——
 * 返回 NULL, 调用点(ds4.c 的 `if (!corr_sel) corr_sel = g->router_selected`)
 * 退回活的 selected, 而那正是 CUDA 路径下它此刻的正确值。 */
ds4_gpu_tensor *ds4_gpu_corr_saved_selected(void) { return NULL; }

/* TP(tensor-parallel) 的 shared-FFN down 行切片: 为"两台 16G Mac 拆一个单机装不
 * 下的模型"设计, 算完半个 out_dim 再跨机 all-reduce。CUDA 侧走单机整模型, 没有
 * 对端可 all-reduce, 也不需要拆。返回 0 => 调用点 ok=false, 不进 TP 分支。 */
int ds4_gpu_matmul_q8_0_rowslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim_full, uint64_t in_dim_slice,
        uint64_t out_dim, const ds4_gpu_tensor *x) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim_full; (void)in_dim_slice; (void)out_dim; (void)x;
    return 0;
}

/* 裁剪专家模型(expert_shrunken)的 full-256 -> 紧凑 slot 路由翻译。CUDA 的 routed
 * MoE kernel 没有 keep-map 支持。返回 0 让 model_open 在上传 LUT 时 ds4_die 退出:
 * 这是要的行为 —— 与其拿未翻译的 id 去索引裁剪后的专家张量、静默算出一堆数字汤,
 * 不如在加载期就拒绝。全量模型(无 keep-map)两个函数都不会被调到。 */
int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer) {
    (void)lut; (void)n_layer;
    return 0;
}
int ds4_gpu_translate_expert_ids(
        ds4_gpu_tensor *selected, uint32_t layer, uint32_t n_expert_used,
        uint32_t n_tokens, uint32_t n_total_expert) {
    (void)selected; (void)layer; (void)n_expert_used;
    (void)n_tokens; (void)n_total_expert;
    return 0;
}

/* 双机 worker 在 coordinator 刚连上、链路还安静时反向拨通 efetch 连接。单机
 * CUDA 没有远端专家字节源, no-op。 */
extern "C" void ds4_gpu_expert_remote_fetch_kick(void) {}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (0 &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;

    if (0) {
        if (count > 1) {
            for (uint32_t i = 0; i < count; i++) {
                (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
            }
            return 1;
        }
        for (uint32_t i = 0; i < count; i++) {
            if (!cuda_model_copy_chunked(model_map, model_size, offsets[i], sizes[i])) {
                (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
            }
        }
    }
    return 1;
}

int ds4_gpu_set_model_map_spans_split(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        const bool *resident_flags,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    /* CUDA does not use Metal's residency-set hint: cold routed-expert reads
     * already fall back to the UVA-mapped pointer (see accelerator_cache_model
     * _tensor_spans in ds4.c, which skips "_exps." at the HBM cache stage). The
     * resident flags are therefore advisory only here; map every span as usual. */
    (void)resident_flags;
    return ds4_gpu_set_model_map_spans(model_map, model_size, offsets, sizes,
                                       count, max_tensor_bytes);
}

/* Cross-layer router prediction prefetch is a Metal-side optimization (NVMe
 * read-ahead for SSD-streamed experts); the CUDA backend accepts and ignores
 * the registration so shared engine code links unchanged. */
int ds4_gpu_register_layer_router(const void *model_map, uint32_t layer,
                                             uint64_t gate_inp_offset, int gate_inp_is_f32,
                                             uint64_t probs_bias_offset,
                                             uint64_t gate_exps_offset, uint64_t up_exps_offset,
                                             uint64_t down_exps_offset, uint64_t gate_expert_bytes,
                                             uint64_t down_expert_bytes, uint32_t n_embd,
                                             uint32_t n_expert, uint64_t hash_table_offset,
                                             uint32_t hash_k, uint32_t hash_rows) {
    (void)model_map; (void)layer; (void)gate_inp_offset; (void)gate_inp_is_f32;
    (void)probs_bias_offset;
    (void)gate_exps_offset; (void)up_exps_offset; (void)down_exps_offset;
    (void)gate_expert_bytes; (void)down_expert_bytes; (void)n_embd; (void)n_expert;
    (void)hash_table_offset; (void)hash_k; (void)hash_rows;
    return 1;
}

int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (1) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
#ifndef DS4_CUDA_SPARK_HBM_CACHE
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
    (void)label;
    return 1;
#else
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    /* Startup walk: force-populate the device-resident HBM cache so hot
     * tensors hit cudaMalloc copies rather than the UVA-mapped fallback.
     * Skip silently if over budget or opted out — the mapped pointer still
     * works for any tensor we don't pre-cache. */
    if (0) return 1;
    if (g_model_device_owned) return 1;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes >= limit || bytes > limit - g_model_range_bytes) return 1;
    const char *what = label ? label : "model_tensor";
    /* Skip if this span is already populated. */
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && bytes <= r.bytes && !r.host_registered) return 1;
    }
    if (cuda_model_range_populate_device_copy(model_map, offset, bytes, what) == NULL) return 0;
    /* ★拷进设备后立刻丢掉这段的 page cache★
     * 统一内存机器上 mmap 的 page cache 与设备副本是同一块物理内存的两份占用:
     * 89.77GiB 模型 + 89.77GiB 副本 = 179GiB 远超 121GiB, 实测 free 掉到 7GiB 后
     * 系统忙于回收, 加载阶段直接卡死。这段字节此后由设备副本供给(offset→device ptr
     * 已登记), mmap 侧不再需要驻留 —— MADV_DONTNEED 对文件映射只丢干净页, 不损数据。 */
    if (1) {
        const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
        uintptr_t a = (uintptr_t)model_map + offset;
        uintptr_t b = a + bytes;
        a = (a + pg - 1) & ~(pg - 1);      /* 只丢整页, 不碰边界的半页 */
        b = b & ~(pg - 1);
        if (b > a) (void)madvise((void *)a, (size_t)(b - a), MADV_DONTNEED);
    }
    return 1;
#endif
}

int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (0 &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
        optional_q8_preload_disabled = 1;
        return 1;
    }
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
    optional_q8_preload_disabled = 1;
    return 1;
}

void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    if (g_cublas_ready) {
        const cublasMath_t math_mode =
            (g_quality_mode || 0)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}

/* DSpark 抓取(2026-08-18): target 层输出 HC 4 流均值 → main_hidden 的 slot 段。
 * 官方语义: h.mean(dim=2)(inference/model.py)。 */
__global__ static void hc_mean_slot_kernel(
        float *dst, const float *hc, uint32_t n_embd, uint32_t n_hc,
        uint32_t slot, uint32_t n_tokens) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= n_embd || t >= n_tokens) return;
    const float *h = hc + ((uint64_t)t * n_hc) * n_embd;
    float s = 0.0f;
    for (uint32_t k = 0; k < n_hc; k++) s += h[(uint64_t)k * n_embd + i];
    dst[((uint64_t)t * 3u + slot) * n_embd + i] = s / (float)n_hc;
}

/* token id 从 device 读的变体: 图重放时取参数槽最新值(流水线前提) */
__global__ static void embed_token_hc_dev_kernel(float *out, const unsigned short *w, const int32_t *tok_dev, uint32_t n_vocab, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    int32_t ti = *tok_dev;
    uint32_t token = ti < 0 ? 0u : (uint32_t)ti;
    if (n_vocab && token >= n_vocab) token = 0;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

