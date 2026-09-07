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
            g_quality_mode
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        /* 固定工作区(09-07): 流捕获态里 cuBLAS 不能自己 cudaMalloc 工作区 —— verify 批 CUDA 图里 drafter 建窗的
         * q8 f16 影子 GEMM 报 status 14 让整张图捕获失败。给定 32 MB 用户工作区后 cuBLAS 不再内部分配, 图内可用。 */
        static void *cublas_ws = NULL;
        const size_t ws_bytes = 32u << 20;
        if (!cublas_ws && cudaMalloc(&cublas_ws, ws_bytes) != cudaSuccess) { cublas_ws = NULL; (void)cudaGetLastError(); }
        if (cublas_ws) (void)cublasSetWorkspace(g_cublas, cublas_ws, ws_bytes);
        g_cublas_ready = 1;
        /* 预热(09-07 1M 投机剖面): 进程里第一次 cublasSgemm / GemmEx(f16) 各要 ~0.5 s 装核与选算法, 落在投机首轮里
         * (drafter 建窗的 f16 影子 GEMM 与输出头前的 f32 GEMM), 64 token 的尺被拖 10%; 这里用 16×16 空算把它挪到初始化。 */
        {
            void *wa = NULL, *xa = NULL, *ya = NULL;
            if (cudaMalloc(&wa, 16u * 16u * 4u) == cudaSuccess && cudaMalloc(&xa, 16u * 16u * 4u) == cudaSuccess &&
                cudaMalloc(&ya, 16u * 16u * 4u) == cudaSuccess) {
                const float alpha = 1.0f, beta = 0.0f;
                (void)cudaMemset(wa, 0, 16u * 16u * 4u); (void)cudaMemset(xa, 0, 16u * 16u * 4u);
                (void)cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, 16, 16, 16, &alpha, (const float *)wa, 16,
                                  (const float *)xa, 16, &beta, (float *)ya, 16);
                (void)cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, 16, 16, 16, &alpha, wa, CUDA_R_16F, 16,
                                   xa, CUDA_R_16F, 16, &beta, ya, CUDA_R_32F, 16, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                (void)cudaDeviceSynchronize(); (void)cudaGetLastError();
            }
            if (wa) (void)cudaFree(wa);
            if (xa) (void)cudaFree(xa);
            if (ya) (void)cudaFree(ya);
        }
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

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;
    /* 2026-09-06: 原规则"KV ≥ 8 GiB 一律托管 / 上下文 ≥ 8 GiB 且余量 < total/4 也托管"让 1M 上下文(KV 13.6 GiB)
     * 全程走 cudaMallocManaged: 合成上下文四点实测 1M prefill 76.9 / decode 14.7, 比按核账算的低一大截, 512k 同样
     * 中招。121 GB 统一内存放得下就用设备内存; 只在真装不下时才托管 —— 装不下的判据是留 6 GiB 给系统, 与
     * speed_champ_spark.sh 外部看门狗(available < 6000 MB 杀)同一条线。 */
    /* 余量口径 = /proc/meminfo MemAvailable(与外部看门狗同一口径)。cudaMemGetInfo 在 GB10 统一内存上报的 free
     * 是 MemFree: 模型载入后 page cache 里躺着几十 GB 干净页, 它报 0.79 GiB —— 1M 上下文按它判永远是"装不下"
     * (09-06 实撞), 而 cudaMalloc 本身会回收这些干净页。没有 /proc/meminfo 的平台退回 cudaMemGetInfo。 */
    uint64_t avail_b = 0;
    {
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char line[256];
            while (fgets(line, sizeof line, mf)) {
                unsigned long long kb = 0;
                if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { avail_b = (uint64_t)kb * 1024ull; break; }
            }
            fclose(mf);
        }
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { (void)cudaGetLastError(); free_b = 0; total_b = 0; }
    if (avail_b == 0) avail_b = (uint64_t)free_b;
    const uint64_t floor_bytes = 6ull * 1073741824ull;
    const int managed = avail_b < context_bytes + floor_bytes;
    fprintf(stderr, "ds4: KV policy: MemAvailable %.2f GiB (cuda free %.2f / total %.2f), context %.2f GiB, kv %.2f GiB -> %s\n",
            (double)avail_b / 1073741824.0, (double)free_b / 1073741824.0, (double)total_b / 1073741824.0,
            (double)context_bytes / 1073741824.0, (double)kv_cache_bytes / 1073741824.0,
            managed ? "managed" : "device");
    return managed;
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


/* f32 暂存 → f16 缓存(2026-09-06): 此前只有 Metal 实现, CUDA 的压缩/indexer 缓存一直是 f32 所以从没被调过;
 * 1M 战役把 CUDA 缓存改 f16 后由 core 的 commit 路(metal_graph_commit_*_stage)调用。偏移按字节, count 按元素。 */
/* f32 [rows][512] → 压缩缓存行格式(ds4_gpu_core.h DS4_GPU_COMP_ROW_*): 前 448 维转 f16, 后 64 维原样 f32 */
__global__ static void comp_rows_commit_kernel(uint8_t *dst, const float *src, uint64_t n) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t r = i / 512u, d = i - r * 512u;
    uint8_t *row = dst + r * DS4_GPU_COMP_ROW_BYTES;
    const float v = src[i];
    if (d < DS4_GPU_COMP_ROW_NOPE) ((__half *)row)[d] = __float2half(v);
    else *(float *)(row + DS4_GPU_COMP_ROW_NOPE * 2u + (d - DS4_GPU_COMP_ROW_NOPE) * 4u) = v;
}
int ds4_gpu_comp_rows_commit(ds4_gpu_tensor *dst, uint64_t first_row, const ds4_gpu_tensor *src, uint64_t rows) {
    if (!dst || !src) return 0;
    if (rows == 0) return 1;
    const uint64_t dst_off = first_row * DS4_GPU_COMP_ROW_BYTES, dst_bytes = rows * DS4_GPU_COMP_ROW_BYTES;
    const uint64_t src_bytes = rows * 512u * sizeof(float);
    if (src_bytes > src->bytes || dst_off > dst->bytes || dst_bytes > dst->bytes - dst_off) return 0;
    const uint64_t n = rows * 512u;
    ds4_launch_pdl(comp_rows_commit_kernel, (unsigned)((n + 255u) / 256u), 256, 0, 0, 
        (uint8_t *)dst->ptr + dst_off, (const float *)src->ptr, n);   /* 无流参数 = PTDS 默认流, 捕获态照样进图 */
    return cuda_ok(cudaGetLastError(), "comp rows commit launch");
}

int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (count == 0) return 1;
    const uint64_t src_bytes = count * sizeof(float), dst_bytes = count * sizeof(uint16_t);
    if (src_offset > src->bytes || src_bytes > src->bytes - src_offset ||
        dst_offset > dst->bytes || dst_bytes > dst->bytes - dst_offset) return 0;
    /* 无流参数 = PTDS 默认流(与其余无流 launch 同队, 捕获态里照样进图); g_cur_stream 定义在本文件更后面 */
    ds4_launch_pdl(f32_to_f16_kernel, (unsigned)((count + 255u) / 256u), 256, 0, 0, 
        (__half *)((char *)dst->ptr + dst_offset), (const float *)((const char *)src->ptr + src_offset), count);
    return cuda_ok(cudaGetLastError(), "tensor copy f32->f16 launch");
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
static int32_t *g_tok_id_host = NULL;     /* pinned host 参数槽(直发路写 token 用) */
static int32_t g_tok_id_want = 0;         /* capture 期暂存的 token */
static int g_tok_graph_on = -1;
static uint32_t g_tok_launch_pos = 0;   /* 本 token pos(end_launch 相位选槽用) */
/* 预发射(09-07): 图开头 tok_id_load 核从"本相位的 pinned 槽"取 {mode, id}: mode=0 用 id(主机给的
 * token), mode=1 用 g_tok_next_dev(上一图末尾 decode_argmax 写的 argmax)。四相各一槽, 槽只在该相位
 * 图发射前由主机写, 该图跑完(readback 落地)之前不会再被写 ⇒ 无竞态。旧的 4B H2D memcpy 节点删除:
 * 它既是每 token 一个非核节点(切断 PDL 链), 又要求发射前主机已知 token。 */
static int32_t *g_tok_slots_host = NULL;  /* pinned [4][2] = {mode, id} */
static int32_t *g_tok_slots_dev = NULL;   /* 同一块内存的设备侧地址(零拷贝映射) */
static int32_t *g_tok_next_dev = NULL;    /* 设备 argmax 槽 */
static float *g_argmax_pm = NULL;         /* 两级 argmax 的分段候选 [DS4_ARGMAX_BLOCKS] */
static int32_t *g_argmax_pi = NULL;
static int64_t g_tok_prelaunched_pos = -1;   /* 已预发射的 pos(-1 无) */
static cudaEvent_t g_readback_ev = NULL;     /* logits/next-tok 异步回传完成事件 */

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

