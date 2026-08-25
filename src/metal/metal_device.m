/* metal_device.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

void ds4_gpu_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
    if (getenv("DS4_DRAIN_GPU_TIME")) {
        /* GPU-busy vs host-wait splitter: kernelEnd-kernelStart = scheduling+
         * encode validation; GPUEnd-GPUStart = shader execution. */
        fprintf(stderr, "ds4: cb-time %s gpu=%.2fms sched=%.2fms\n",
                label ? label : "?",
                (cb.GPUEndTime - cb.GPUStartTime) * 1e3,
                (cb.kernelEndTime - cb.kernelStartTime) * 1e3);
    }
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal %s failed: %s\n",
                label, [[cb.error localizedDescription] UTF8String]);
        /* [diag] at the moment of failure, is the GPU working set over the
         * device ceiling? currentAllocated > recommendedMax => wired-set OOM. */
        fprintf(stderr,
                "ds4: [diag] CB '%s' failure: device currentAllocated %.2f GiB, "
                "recommendedMax %.2f GiB, residency wired %llu views\n",
                label,
                (double)[g_device currentAllocatedSize] / (1024.0 * 1024.0 * 1024.0),
                (double)[g_device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)g_model_residency_count);
        return 0;
    }
    return 1;
}

int ds4_gpu_wait_pending_command_buffers(const char *label) {
    int ok = 1;
    for (id<MTLCommandBuffer> pending in g_pending_cbs) {
        if (!ds4_gpu_wait_command_buffer(pending, label)) ok = 0;
    }
    [g_pending_cbs removeAllObjects];
    return ok;
}

int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    if (!owned) return 1;

    [cb commit];
    int ok = ds4_gpu_wait_pending_command_buffers(label);
    if (!ds4_gpu_wait_command_buffer(cb, label)) ok = 0;
    if (!ok && cb.error && getenv("DS4_RESIDUAL_DEBUG"))
        fprintf(stderr, "ds4: [cb-error] %s: %s\n", label ? label : "?",
                cb.error.localizedDescription.UTF8String);
    [g_transient_buffers removeAllObjects];
    return ok;
}

/* Fire-and-forget submit for owned CBs whose results are only consumed by
 * LATER GPU work or behind an existing pending-drain: commit in queue order
 * (Metal executes same-queue CBs in commit order) and park on g_pending_cbs
 * so every subsequent drain covers it. No host wait — this is what makes the
 * per-layer corr dispatches ~free instead of a commit+waitUntilCompleted pair
 * per routed layer. Only valid for encoders that touch NO transient buffers. */
int ds4_gpu_submit_command_buffer_async(id<MTLCommandBuffer> cb, int owned) {
    if (!owned) return 1;
    [cb commit];
    [g_pending_cbs addObject:cb];
    return 1;
}

static int ds4_gpu_use_m5_private_scratch(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = ds4_gpu_device_name_contains("M5");
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_scratch_needs_cpu_access(const char *label) {
    if (!label) return 0;
    return strstr(label, "mask") != NULL ||
           strcmp(label, "ds4_attention_output_group_ids") == 0;
}

int ds4_gpu_ensure_scratch_buffer(
        id<MTLBuffer> __strong *buffer,
        NSUInteger    *capacity,
        NSUInteger     bytes,
        const char    *label) {
    if (*buffer && *capacity >= bytes) return 1;
    if (bytes == 0) bytes = 1;
    if (bytes > NSUIntegerMax) return 0;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (ds4_gpu_use_m5_private_scratch() &&
        !ds4_gpu_scratch_needs_cpu_access(label)) {
        /*
         * M5 scratch buffers that only flow between Metal kernels do not need
         * CPU-visible shared storage. This reduces shared-memory traffic and
         * residency pressure for the long prefill scratch pools without
         * changing the public buffer lifetime model. Keep default hazard
         * tracking because the graph reuses these buffers across dependent
         * compute encoders.
         */
        options = MTLResourceStorageModePrivate;
    }

    *buffer = [g_device newBufferWithLength:bytes options:options];
    if (!*buffer && options != MTLResourceStorageModeShared) {
        *buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    if (!*buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal scratch buffer %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        *capacity = 0;
        return 0;
    }
    (*buffer).label = [NSString stringWithUTF8String:label];
    *capacity = bytes;
    return 1;
}

uint64_t ds4_gpu_effective_model_max_tensor_bytes(uint64_t map_size, uint64_t max_tensor_bytes) {
    if (max_tensor_bytes != 0) return max_tensor_bytes;
    return map_size < DS4_METAL_FALLBACK_MAX_TENSOR_BYTES ?
           map_size : DS4_METAL_FALLBACK_MAX_TENSOR_BYTES;
}

double ds4_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int ds4_gpu_progress_enabled(void) {
    return ds4_log_is_tty(stderr);
}

void ds4_gpu_progress_begin(const char *what) {
    if (!ds4_gpu_progress_enabled()) return;
    fprintf(stderr, "ds4: %s...", what);
    fflush(stderr);
}

void ds4_gpu_progress_done(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

void ds4_gpu_progress_failed(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}
