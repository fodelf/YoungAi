/* cuda_lifecycle.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * init/cleanup, tensor 生命周期, side stream。
 */
int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode =
            (g_quality_mode || 0)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
    return 1;
}

void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
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
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
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
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    /* DtoD 走异步(ptds 同流有序): 同步版每次吸收整条 GPU 队列等待, spec 捕获
     * 172 次/轮时是 verify 空隙主因(2026-08-21 nsys cudaMemcpy 54%)。 */
    return cuda_ok(cudaMemcpyAsync((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

/* ---- decode 单 token CUDA graph ----
 * 动机: decode 每 token ~900 个小 kernel launch, 提交间隙+延迟尾实测吃 ~10ms/token。
 * 方案(llama.cpp 同款): 每 token stream-capture(比真实提交便宜) → 首次 Instantiate,
 * 之后 cudaGraphExecUpdate 增量打参数补丁 → GraphLaunch 一次成图重放。
 * 前置: 本文件以 -default-stream per-thread 编译(legacy 流不可捕获, PTDS 让全部
 * 无流 launch 落在可捕获的 per-thread 流上; 其余 stream 的同步均已显式化)。
 * 任何一步失败 ⇒ 永久回退直跑(capture 态下 kernel 未执行, 调用方需重编码一遍)。 */
/* 乒乓双 exec 流水线(2026-08-17 第六夜): GPU 跑 token N 的图时, CPU 捕获/更新
 * token N+1 的图到另一个 exec 槽 —— encode(实测 1.1-4.6ms/token)整体移出临界路径。
 * token id 经"pinned 参数槽"间接: 图内录一条 4B H2D memcpy(捕获记录的是地址),
 * 发射前才写 *g_tok_id_host, 重放时读到最新值。竞态规避: capture 期间绝不写参数槽
 * (上一个图可能还没执行到它的 memcpy), 只暂存到 g_tok_id_want。 */
static cudaGraphExec_t g_tok_execs[4] = {NULL, NULL, NULL, NULL};   /* 08-20 四相: ratio-4 压缩器周期拓扑 */
static int g_tok_cur = 0;                 /* 最近发射的槽 */
static int g_tok_pending = -1;            /* 预编码完成待发射的槽 */
static uint32_t g_tok_pending_pos = 0;
static int g_tok_pending_logits = 0;
static int32_t *g_tok_id_dev = NULL;      /* device 侧 token id */
/* 双流并发(2026-08-17 第八夜): routed MoE 与 shared FFN 同读 ffn_norm、输出末端相加,
 * 数据流独立却串行。mark(主流, MoE 发射前) → begin(侧流 wait+切流) → join(主流 wait)。
 * capture 中 event record/wait 构建图依赖边; 资源须 capture 外创建。 */
static cudaStream_t g_side_stream = NULL;
static cudaEvent_t g_side_fork_ev = NULL, g_side_join_ev = NULL;
static cudaStream_t g_cur_stream = 0;     /* 0 == PTDS(-default-stream per-thread) */
static int g_side_active = 0;
static int32_t *g_tok_id_host = NULL;     /* pinned host 参数槽 */
static int32_t g_tok_id_want = 0;         /* capture 期暂存的 token */
static int g_tok_graph_on = -1;
static uint32_t g_tok_launch_pos = 0;   /* 本 token pos(end_launch 相位选槽用) */

static void side_stream_ensure(void) {
    if (g_side_stream) return;
    if (cudaStreamCreateWithFlags(&g_side_stream, cudaStreamNonBlocking) != cudaSuccess) {
        g_side_stream = NULL; (void)cudaGetLastError(); return;
    }
    if (cudaEventCreateWithFlags(&g_side_fork_ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g_side_join_ev, cudaEventDisableTiming) != cudaSuccess) {
        g_side_fork_ev = NULL; g_side_join_ev = NULL;
        (void)cudaStreamDestroy(g_side_stream); g_side_stream = NULL;
        (void)cudaGetLastError();
    }
}

int ds4_gpu_side_mark(void) {
    if (((const char *)0) /* DS4_NO_SIDE_STREAM: 路径开关已删(2026-08-22 隐形炸弹清理) */) return 0;   /* 二分诊断: 关闭双流并发 */
    if (!g_side_stream || !g_side_fork_ev) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(cudaStreamPerThread, &cs);
        if (cs != cudaStreamCaptureStatusNone) return 0;
        side_stream_ensure();
        if (!g_side_stream) return 0;
    }
    if (cudaEventRecord(g_side_fork_ev, cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    return 1;
}

int ds4_gpu_side_begin(void) {
    if (!g_side_stream || !g_side_fork_ev) return 0;
    if (cudaStreamWaitEvent(g_side_stream, g_side_fork_ev, 0) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    g_cur_stream = g_side_stream;
    g_side_active = 1;
    return 1;
}

/* 切回主流但保持 fork(侧流尾未 join): comp 链发完后主流继续 indexer/attention */
int ds4_gpu_side_main(void) {
    g_cur_stream = 0;
    return 1;
}

int ds4_gpu_side_join(void) {
    g_cur_stream = 0;
    if (!g_side_active) return 1;
    g_side_active = 0;
    if (cudaEventRecord(g_side_join_ev, g_side_stream) != cudaSuccess ||
        cudaStreamWaitEvent(cudaStreamPerThread, g_side_join_ev, 0) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    return 1;
}

